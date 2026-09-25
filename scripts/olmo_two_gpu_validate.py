#!/usr/bin/env python3
"""Two-rank NCCL correctness against canonical single-GPU accumulation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from cdrm.pretrained.artifacts import write_json, sha256_file
from cdrm.pretrained.ddp_training import EagerDDPTrainer
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_step
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo_artifacts import MANIFEST_FILENAME
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase, build_model, boundary_digests
from scripts.olmo_rt_large_batch import (batch_for, compiler_configuration, set_arm,
    optimizer_for, configure_determinism, require_container_gpu, backend_context,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer, OnlineTracker)

TERMS = ('ce', 'latent', 'kl')
PROTOCOL = ROOT/'docs/reports/olmo-two-gpu/protocol.md'


@contextmanager
def preserve_local_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state())
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def tensor_check(actual, expected, *, atol=None, rtol=None):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    delta = a-b
    err2, ref2 = float(delta.square().sum()), float(b.square().sum())
    maximum, refmax = float(delta.abs().max()), float(b.abs().max())
    l2 = math.sqrt(err2/max(ref2, 1e-60))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    passed = (bool(torch.allclose(a, b, atol=atol, rtol=rtol)) if atol is not None
              else (l2 <= 1e-4 or maximum <= 1e-7) and maximum <= max(1e-7, 5e-4*refmax))
    return dict(passed=finite and passed, finite=finite, relative_l2=l2,
        max_absolute=maximum, max_relative=maximum/max(refmax, 1e-60),
        delta_sq=err2, reference_sq=ref2, exact=torch.equal(a,b))


def check_gradients(model, expected):
    actual = {n: p.grad for n,p in model.named_parameters() if p.grad is not None}
    rows = {n: tensor_check(p, expected[n]) for n,p in actual.items() if n in expected}
    global_l2 = math.sqrt(sum(r['delta_sq'] for r in rows.values()) /
                          max(sum(r['reference_sq'] for r in rows.values()), 1e-60))
    return dict(passed=actual.keys()==expected.keys() and global_l2<=2e-5 and
        all(r['passed'] for r in rows.values()), global_relative_l2=global_l2,
        ownership_exact=actual.keys()==expected.keys(), tensors=rows)


def update_snapshot(model, optimizer, scheduler, counters):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                scheduler=cpu_copy(scheduler.state_dict()), counters=asdict(counters))


def check_update(model, optimizer, scheduler, counters, expected):
    model_rows = {n: tensor_check(p, expected['model'][n], atol=2e-7, rtol=2e-6)
                  for n,p in model.state_dict().items()}
    current = optimizer.state_dict()
    states_equal = current['state'].keys()==expected['optimizer']['state'].keys()
    moment_rows = {}
    for index, state in current['state'].items():
        target = expected['optimizer']['state'].get(index, {})
        states_equal &= state.keys()==target.keys()
        for name, value in state.items():
            if name in target:
                moment_rows[f'{index}/{name}'] = tensor_check(value, target[name], atol=1e-8, rtol=5e-4)
    metadata = (current['param_groups']==expected['optimizer']['param_groups'] and
                scheduler.state_dict()==expected['scheduler'] and asdict(counters)==expected['counters'])
    return dict(passed=states_equal and metadata and all(r['passed'] for r in
        [*model_rows.values(), *moment_rows.values()]), metadata_exact=metadata,
        optimizer_ownership_exact=states_equal, parameters=model_rows, moments=moment_rows)


def anchor_to_reference(model, optimizer, scheduler, counters, reference):
    """Reset a completed eager update to canonical state, without replacing weights.

    Call only after candidate metrics/comparisons have been recorded and the DDP
    step has cleared gradients. This diagnostic deliberately prevents independent
    trajectory drift. It is not a passing multi-step trajectory comparison.
    """
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Anchor only at a completed, cleared-gradient update boundary')
    parameters=tuple((n,id(p),p.data_ptr()) for n,p in model.named_parameters())
    model.load_state_dict(reference['model'],strict=True,assign=False)
    # Avoid aliasing CPU test/reference tensors into a mutable optimizer. GPU
    # loading also follows the optimizer's supported fused-state device rules.
    optimizer.load_state_dict(cpu_copy(reference['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(reference['scheduler']))
    for name,value in reference['counters'].items(): setattr(counters,name,value)
    storage=parameters==tuple((n,id(p),p.data_ptr()) for n,p in model.named_parameters())
    cleared=all(p.grad is None for p in model.parameters())
    actual=boundary_digests(model,optimizer,scheduler,counters)
    exact=actual==tree_digests(reference)
    return dict(passed=storage and cleared and exact,canonical_state_exact=exact,
                parameter_storage_preserved=storage,gradients_cleared=cleared,state_digest=actual)


def persist_checkpoint_reference(artifacts,output_dir):
    """Copy verified native provenance, without duplicating the weight checkpoint."""
    artifacts,output_dir=Path(artifacts),Path(output_dir)
    manifest=validate_prepared_manifest(artifacts)
    source=artifacts/MANIFEST_FILENAME
    target=output_dir/'checkpoint-reference'/MANIFEST_FILENAME
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    digest=sha256_file(source)
    if sha256_file(target)!=digest:
        raise IOError('Copied native artifact manifest differs from verified source')
    return dict(checkpoint=manifest['checkpoint'],manifest_path=str(target.relative_to(output_dir)),
                manifest_sha256=digest,manifest_size_bytes=target.stat().st_size,
                source_manifest_path=str(source))


def finalize_report(output_dir,report,tracker,*,original_error=None):
    """Persist primary outcome before SDK finalization; never replace its error."""
    path=Path(output_dir)/'report.json'
    write_json(path,report)
    try:
        if tracker is not None: tracker.finish(succeeded=report['passed'])
    except BaseException as error:
        report['tracking_finish_error']=dict(type=type(error).__name__,message=str(error))
        report['passed']=False
        if original_error is not None:
            original_error.add_note(f'Tracking finalization also failed: {error}')
        else:
            report.setdefault('error',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            raise
    finally:
        write_json(path,report)


def gather(value):
    values = [None]*dist.get_world_size()
    dist.all_gather_object(values,value)
    return values


def assert_all(passed, message):
    flags = gather(bool(passed))
    if not all(flags): raise AssertionError(f'{message}; rank flags={flags}')


def move_batch(batch, device):
    return NextLatBatch(**{n: None if t is None else t.to(device) for n,t in vars(batch).items()})


def make_batch(case, tokenizer, update, rank, micro, *, tiny):
    serial = update*4+rank*2+micro
    if tiny:
        generator = torch.Generator().manual_seed(920+serial)
        ids = torch.randint(2,59,(case.batch_size,case.length),generator=generator)
        valid = torch.ones_like(ids,dtype=torch.bool)
        docs = torch.arange(case.batch_size)[:,None].expand_as(ids).clone()
        batch = NextLatBatch(ids,valid,docs,valid.clone(),valid.clone(),valid.clone())
    else:
        batch = batch_for(tokenizer,case,serial)
    ce, latent, kl = [batch.valid_mask.clone() for _ in range(3)]
    ce[:,(rank+micro)%3::3] = False
    latent[:,1::3] = False
    kl[:,2::4] = False
    # Predictor contributes only in rank0's earlier no_sync microbatch. On
    # update3 it is globally absent, exercising return to grad=None.
    if rank or micro or update>=2:
        latent.zero_(); kl.zero_()
    return replace(batch,ce_mask=ce,latent_mask=latent,kl_mask=kl)


def construct(case, args, device):
    if args.tiny:
        config = OLMoConfig.tiny()
        base = OLMoTiledRTForCausalLM(config,attention_backend='math',attention_precision='fp32')
        state = base.state_dict()
        model = build_model(state,case,device=device,model_config=config,chunk_size=128,backend='math')
        del base,state
        model.backbone.backbone.attention_precision = 'fp32'
        model.backbone.backbone.ordinary_activation_checkpointing = True
        model.backbone.backbone.reuse_rope = True
        model.backbone.backbone.kv_only_writes = True
    else:
        validate_prepared_manifest(args.artifacts)
        state = load_native_state_dict(args.artifacts)
        model = build_model(state,case,device=device)
        del state
        set_arm(model,'compiled-native')
        base = model.backbone.backbone
        base.ordinary_activation_checkpointing=True
        base.cast_weights_once=True
        base.tile_backend='triton'
        base.backward_tile_backend='triton'
        base.backward_memory='recompute'
    return model


def canonical_references(model, batches, mode, updates, config):
    optimizer,scheduler = optimizer_for(model,'compiled-native')
    counters=TrainingCounters()
    references=[]
    original_clip=torch.nn.utils.clip_grad_norm_
    for update in range(updates):
        gradients={}
        def capture_then_clip(*args,**kwargs):
            gradients.update({n:cpu_copy(p.grad) for n,p in model.named_parameters() if p.grad is not None})
            return original_clip(*args,**kwargs)
        with patch('torch.nn.utils.clip_grad_norm_',capture_then_clip):
            metrics=optimizer_step(model,optimizer,batches[update],config=config,
                backbone_kwargs={'mode':mode,'full_valid_causal':True},scheduler=scheduler,counters=counters)
        references.append(dict(gradients=gradients,metrics=metrics,
                               state=update_snapshot(model,optimizer,scheduler,counters)))
        print(f'canonical update {update+1} complete',flush=True)
    del optimizer,scheduler
    return references


def run_case(args, name, tracker):
    rank=dist.get_rank(); device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    is_rt='rt' in name; is_fbt='fbt' in name or name=='combined'
    is_nextlat='nextlat' in name or name=='combined'
    is_rt |= name=='combined'
    case=IntegrationCase(name,fbt=is_fbt,nextlat=is_nextlat,
        rt_layers=(0,1 if args.tiny else 15) if is_rt else (),
        batch_size=args.batch_size,length=args.length,updates=args.updates)
    torch.random.default_generator.manual_seed(20260925)
    torch.cuda.manual_seed(20260925)
    model=construct(case,args,device)
    config=LMTrainingConfig(precision='fp32' if args.tiny else 'bf16_mixed')
    tokenizer=None if args.tiny else load_native_tokenizer(args.artifacts)
    anchor_updates=args.anchor_updates
    owns_references=rank==0 or anchor_updates
    initial=cpu_copy(model.state_dict()) if owns_references else None
    references=[]
    if owns_references:
        batches=[[move_batch(make_batch(case,tokenizer,u,r,m,tiny=args.tiny),device)
                  for r in range(2) for m in range(2)] for u in range(args.updates)]
        references=canonical_references(model,batches,case.mode(),args.updates,config)
        model.load_state_dict(initial,assign=False)
        model.zero_grad(set_to_none=True)
        del initial,batches
        gc.collect();torch.cuda.empty_cache()
    dist.barrier()
    trainer=EagerDDPTrainer(model)
    optimizer,scheduler=optimizer_for(model,'compiled-native')
    counters=TrainingCounters()
    report=dict(name=name,case=asdict(case),precision=config.precision,checks=[],passed=False,
        anchor_updates=anchor_updates,comparison_scope=('each candidate step starts from canonical weights/moments; '
        'not an independent multi-step trajectory' if anchor_updates else 'independent multi-step trajectories'),
        physical_candidate_updates=0,physical_canonical_updates=args.updates*(2 if anchor_updates else 1))
    progress_path=args.output_dir/(f'{name}-progress.json' if rank==0 else f'{name}-rank-{rank}-progress.json')
    write_json(progress_path,report)
    for update in range(args.updates):
        batches=[move_batch(make_batch(case,tokenizer,update,rank,m,tiny=args.tiny),device) for m in range(2)]
        result=trainer.backward(batches,config=config,
                                backbone_kwargs={'mode':case.mode(),'full_valid_causal':True})
        grad_check=check_gradients(model,references[update]['gradients']) if owns_references else {'passed':True}
        # Record candidate raw gradients before any assertion or anchor reset.
        raw_digest=tree_digests({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
        replicas=gather(raw_digest)
        exact_gradients=replicas[0]==replicas[1]
        row=dict(update=update+1,phase='raw_gradients',gradients=grad_check,
                 replica_gradients_exact=exact_gradients,anchor_applied=False)
        report['checks'].append(row)
        write_json(progress_path,report)
        assert_all(grad_check['passed'] and exact_gradients,'raw reduced gradient comparison')
        metrics=trainer.step(result,optimizer,scheduler=scheduler,counters=counters)
        report['physical_candidate_updates']+=1
        row.update(phase='candidate_stepped',metrics=metrics)
        write_json(progress_path,report)
        update_check=check_update(model,optimizer,scheduler,counters,references[update]['state']) if owns_references else {'passed':True}
        replicas=gather(boundary_digests(model,optimizer,scheduler,counters))
        exact_state=replicas[0]==replicas[1]
        loss_check=True
        if owns_references:
            expected=references[update]['metrics']
            loss_check=(metrics['counts']==expected['counts'] and
                all(math.isclose(metrics['loss_sums'][t],expected['loss_sums'][t],rel_tol=2e-6,abs_tol=2e-6) for t in TERMS))
        row.update(phase='candidate_complete_update',state=update_check,
                   replica_state_exact=exact_state,loss_counts_match=loss_check,metrics=metrics)
        write_json(progress_path,report)
        if rank==0:
            tracker.log({'update':update+1,f'{name}/gradient_l2':grad_check['global_relative_l2'],
                f'{name}/objective':metrics['objective'],f'{name}/replicas_exact':exact_state})
            print(f'{name} DDP update {update+1}: gradient L2={grad_check["global_relative_l2"]:.3g}; state={update_check["passed"]}',flush=True)
        assert_all(update_check['passed'] and exact_state and loss_check,'complete update comparison')
        if anchor_updates:
            anchored=anchor_to_reference(model,optimizer,scheduler,counters,references[update]['state'])
            anchor_replicas=gather(anchored['state_digest'])
            anchored['replicas_exact']=anchor_replicas[0]==anchor_replicas[1]
            anchored['passed'] &= anchored['replicas_exact']
            del anchored['state_digest']
            row['anchor']=anchored
            row['anchor_applied']=True
            write_json(progress_path,report)
            assert_all(anchored['passed'],'canonical anchor restore and replica agreement')
        if owns_references: references[update]=None
        del result,batches
    report['passed']=True
    report['final_state_is_canonical_anchor']=anchor_updates
    if rank==0: write_json(args.output_dir/f'{name}-report.json',report)
    del trainer,optimizer,scheduler,model,references
    gc.collect();torch.cuda.empty_cache();dist.barrier()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--case',choices=('all','ordinary','rt','combined'),default='all')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--length',type=int,default=512)
    parser.add_argument('--updates',type=int,default=2)
    parser.add_argument('--anchor-updates',action='store_true',
        help='Diagnostic: restore each canonical state before the next candidate update; not independent trajectories')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args=parser.parse_args()
    configure_determinism(True)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    runtime=require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    compiler_configuration()
    dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])),timeout=timedelta(minutes=30))
    if dist.get_world_size()!=2: raise RuntimeError('Requires two ranks')
    rank=dist.get_rank();tracker=None;original_error=None
    report=dict(schema='olmo-two-gpu-eager-v1',runtime=runtime,cases=[],passed=False,
                anchor_updates=args.anchor_updates,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if rank==0:
            args.output_dir.mkdir(parents=True,exist_ok=False)
            sources=[*sorted((ROOT/'cdrm/pretrained').glob('*.py')),*sorted((ROOT/'scripts').glob('olmo*.py')),
                     ROOT/'scripts/experiment_tracking.py',PROTOCOL]
            report['sources']={str(p.relative_to(ROOT)):sha256_file(p) for p in sources}
            report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            for p in sources:
                target=args.output_dir/'source-snapshot'/p.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            if not args.tiny:
                report['checkpoint_reference']=persist_checkpoint_reference(args.artifacts,args.output_dir)
            write_json(args.output_dir/'report.json',report)
            tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',group='olmo-two-gpu',
                name=args.output_dir.name,output_dir=args.output_dir,preserve_state=preserve_local_rng)
            tracker.start({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
            report['wandb']=tracker.record
            print(tracker.record['run_url'],flush=True)
        dist.barrier()
        if args.case!='all': names=[args.case]
        elif args.tiny:
            names=['-'.join(n for n,on in [('rt',r),('fbt',f),('nextlat',n)] if on) or 'ordinary'
                   for f in (False,True) for r in (False,True) for n in (False,True)]
        else: names=['ordinary','rt','combined']
        with disable_autocast_weight_cache(),backend_context('math' if args.tiny else 'flash'):
            for name in names:
                report['cases'].append(run_case(args,name,tracker))
                if rank==0: write_json(args.output_dir/'progress.json',report)
        report['passed']=True
    except BaseException as error:
        original_error=error
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if rank==0 and args.output_dir.exists():
            finalize_report(args.output_dir,report,tracker,original_error=original_error)
        # A failed peer may be inside a collective. The launcher owns whole-job
        # termination; never introduce an additional teardown collective on error.
        if original_error is None: dist.destroy_process_group()


if __name__=='__main__': main()
