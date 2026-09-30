#!/usr/bin/env python3
"""Bounded production-state migration and fresh-process graph continuation.

Unlike the allocation benchmark, this restores the actual scheduler, counters,
logical data position and mapped rank RNG. Scientific native scope is127->128.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule, build_campaign_model, build_campaign_adamw, feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, load_distributed_checkpoint, save_distributed_checkpoint, _local_rng
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.packed_campaign_data import PackedCursor
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_allocation_benchmark import recipe_from_record, release_completed_graph_runner, source_hashes as allocation_sources
from scripts.olmo_allocation_acceptance import logical_fixture, fixture_for_rank, literal_counts
from scripts.olmo_campaign_execution import validate_clocks
from scripts.olmo_campaign_loop import Coordinator
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import checkpoint_disk_preflight
from scripts.olmo_campaign_ssd_storage import SSDCheckpointStorage
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import boundary_digests
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_pilot_async_storage import AsyncCheckpointRetention
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_two_gpu_recovery import seed_local
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_topology_contract import make_topology_contract, destination_configuration, destination_fingerprint, cursor_record, expected_counters_since_origin
from scripts.olmo_topology_checkpoint import load_topology_checkpoint
from scripts.olmo_topology_storage import retain_in_child, source_hashes as storage_source_hashes

SCHEMA = 'olmo-topology-execution-v1'
FIXTURE_SHA = hashlib.sha256(b'olmo-topology-packed-fixture-v1').hexdigest()


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest_json(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    result = allocation_sources()
    result.update(storage_source_hashes())
    names = ['olmo_topology_execute.py', 'olmo_topology_contract.py', 'olmo_topology_checkpoint.py',
        'olmo_allocation_acceptance.py', 'olmo_campaign_execution.py', 'olmo_campaign_restart.py',
        'olmo_campaign_ssd_storage.py', 'olmo_pilot_async_storage.py', 'olmo_campaign_execution_restore.py',
        'olmo_f1_common.py', 'olmo_lm_common.py', 'olmo_two_gpu_recovery.py']
    result.update({'scripts/' + n: sha256_file(ROOT / 'scripts' / n) for n in names})
    return dict(sorted(result.items()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('seed', 'migrate', 'resume'), required=True)
    p.add_argument('--scale', choices=('tiny', 'native'), required=True)
    p.add_argument('--arm', choices=('B', 'NFR'), default='NFR')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--stop-after', type=int, required=True)
    p.add_argument('--lineage', required=True)
    p.add_argument('--storage-prefix', help='Native requires existing asynchronous GCS publication')
    p.add_argument('--source-rank', type=int, help='Explicit single-rank source RNG selection')
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.stop_after < 0:
        p.error('Positive physical batch and nonnegative terminal boundary required')
    if a.phase == 'seed':
        if a.scale != 'tiny' or a.checkpoint or a.checkpoint_sha256:
            p.error('Only tiny seed has no checkpoint')
    elif not a.checkpoint or not a.checkpoint_sha256:
        p.error('Migrate/resume requires checkpoint and independent manifest SHA')
    if a.checkpoint_sha256 and (len(a.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in a.checkpoint_sha256)):
        p.error('Invalid manifest SHA256')
    if a.scale == 'native' and (a.arm != 'NFR' or a.batch_size != 12 or a.stop_after not in (127, 128) or not a.storage_prefix):
        p.error('Native scope is NFR B12 and boundaries127/128 with retained checkpoints')
    if a.scale == 'tiny' and (a.batch_size != 2 or a.stop_after > 3):
        p.error('Tiny fixture uses physicalB2 and at most3updates')
    a.output_dir = a.output_dir.absolute(); a.checkpoint_root = a.checkpoint_root.absolute()
    if not a.output_dir.is_relative_to(ROOT) or a.output_dir.exists() or a.checkpoint_root.exists():
        p.error('Require new output/checkpoint roots and persistent project evidence')
    if not a.checkpoint_root.is_relative_to(Path('/mnt/localssd/cdrm-checkpoints/topology-migration')):
        p.error('All large artifacts must use the explicit volatile SSD namespace')
    return a


def execution_config(config):
    return config.get('execution') or config['execution_identity']['payload']['execution']


def data_config(config):
    return config.get('data') or config['execution_identity']['payload']['data']


def construct(config, payload, device):
    recipe = recipe_from_record(config['recipe'])
    e = execution_config(config)
    kwargs = dict(attention_backend='math' if config['training']['precision']=='fp32' else 'sdpa',
        attention_precision=e['rt_attention_precision'], ordinary_activation_checkpointing=e['ordinary_activation_checkpointing'],
        cast_weights_once=e['cast_weights_once'], tile_backend=e['rt_forward_tiles'], backward_tile_backend=e['rt_backward_tiles'],
        backward_memory=e['backward_memory'], reuse_rope=e['reuse_rope'], kv_only_writes=e['kv_only_writes'],
        ordinary_pointwise_backend=e['ordinary_pointwise_backend'], ordinary_rope_backend=e['ordinary_rope_backend'])
    if payload is None:
        torch.manual_seed(20261001)
        base = OLMoTiledRTForCausalLM(OLMoConfig.from_dict(config['backbone']), **kwargs)
    else:
        base = OLMoTiledRTForCausalLM(OLMoConfig.from_dict(config['backbone']), device='meta', dtype=torch.float32, **kwargs)
        prefix = 'backbone.backbone.'
        base.load_state_dict({n[len(prefix):]:v for n,v in payload['model'].items() if n.startswith(prefix)}, strict=True, assign=True)
    model = build_campaign_model(base, recipe)
    if config.get('model'):
        model.config = NextLatConfig(**config['model'])
        if model.predictor is not None:
            model.predictor.config = replace(model.predictor.config,lambda_kl=model.config.lambda_kl,lambda_latent=model.config.lambda_latent)
    model.to(device).train()
    return model, build_campaign_adamw(model, recipe, fused=True), recipe


def tiny_seed_config(arm, world_size, batch_size):
    recipe = CampaignRecipe(arm, sequence_length=8, rt_layers=(0,1), warmup_tokens=50,
                            document_policy='continuous-stream-v1')
    return plain({'schema':'olmo-topology-tiny-origin-v1','backbone':OLMoConfig.tiny().to_dict(),
        'recipe':recipe.to_dict(), 'training':{'precision':'fp32','max_grad_norm':recipe.max_grad_norm},
        'world_size':world_size,'physical_batch_per_rank':batch_size,
        'data':{'kind':'tiny-packed-allocation-v1','manifest_sha256':FIXTURE_SHA,'split':'train'},
        'execution':{'rt_attention_precision':'fp32','ordinary_activation_checkpointing':True,'cast_weights_once':True,
            'rt_forward_tiles':'eager','rt_backward_tiles':'eager','backward_memory':'recompute','reuse_rope':True,
            'kv_only_writes':True,'ordinary_pointwise_backend':'eager','ordinary_rope_backend':'native'},
        'ddp':{'static_graph':True,'find_unused_parameters':False,'broadcast_buffers':False,
               'gradient_as_bucket_view':False,'bucket_cap_mb':25}})


def tiny_plans(recipe):
    result=[];chunk=0
    for i in range(3):
        fixture=logical_fixture(i)
        batches=fixture_for_rank(recipe,i,rank=0,world_size=1)[0]
        counts=literal_counts(batches,nextlat=recipe.nextlat)
        start=PackedCursor(FIXTURE_SHA,'train',chunk,i);chunk+=len(fixture.rows)
        result.append(SimpleNamespace(rows=fixture.rows,start_cursor=start,next_cursor=PackedCursor(FIXTURE_SHA,'train',chunk,i+1),
            counts=SimpleNamespace(valid_tokens=fixture.counts.presented_tokens,packed_rows=len(fixture.rows),
                ce_targets=counts['ce'],latent_pairs=counts['latent'],kl_triples=counts['kl'])))
    return result


def input_rows(batches,noises,keys):
    result=[]
    for b,n,ks in zip(batches,noises,keys):
        for row,key in enumerate(ks):
            masks={k:None if getattr(b,k) is None else getattr(b,k)[row] for k in
                   ('valid_mask','document_ids','ce_mask','latent_mask','kl_mask')}
            result.append({'key':key,'input_sha256':digest_json(tree_digests(b.input_ids[row])),
                'masks_sha256':digest_json(tree_digests(masks)),
                'jitter_sha256':digest_json(tree_digests(None if n is None else [x[row] for x in n]))})
    return result


def artifact(path, key=None):
    return {'path':str(path),'sha256':sha256_file(path),'key':key}


def run(a,c,device,report,tracker,persist):
    rank,world=c.rank,c.world_size
    if a.phase=='seed':
        source_manifest=None;config=tiny_seed_config(a.arm,world,a.batch_size);payload=None
    else:
        source_manifest=c.call('authenticate source',lambda:inspect_distributed_checkpoint(a.checkpoint,
            expected_manifest_sha256=a.checkpoint_sha256,verify_state=True))
        config=source_manifest['metadata']['configuration']
        payload=torch.load(a.checkpoint/'state.pt',map_location='cpu',weights_only=True,mmap=True)
        if a.scale=='native' and (source_manifest['counters']['optimizer_updates'] not in (127,128)
                or config['backbone']!=OLMoConfig.native_1b().to_dict()):
            raise ValueError('Native source outside declared endpoint fixture')
    if config['training']['precision'] != report['precision']:
        raise ValueError('Execution precision differs from checkpoint contract')
    if a.phase == 'resume' and config.get('topology_migration', {}).get('sources') != report['sources']:
        raise ValueError('Strict resume runtime source pins differ')
    model,optimizer,recipe=c.call('construct model',lambda:construct(config,payload,device))
    if recipe.arm!=a.arm:raise ValueError('Arm differs from source')
    token_counts=None if payload is None else [y-x for x,y in zip(payload['scheduler']['token_prefix'],payload['scheduler']['token_prefix'][1:])]
    del payload;gc.collect()
    data=None
    if a.scale=='tiny':
        plans=tiny_plans(recipe)
        if token_counts is None:token_counts=[p.counts.valid_tokens for p in plans]
    else:
        dc=data_config(config)
        data=c.call('authenticate ordered data',lambda:OrderedCampaignData(Path(dc['corpus']),Path(dc['index'])))
        if data.manifest_sha256!=dc['index_manifest_sha256']:raise ValueError('Data authority differs')
        plans=[];cursor=data.cursor()
        for tokens in token_counts:
            p=data.peek_update(cursor,tokens)
            if p is None or p.counts.valid_tokens!=tokens:raise ValueError('Immutable finite data plan differs')
            plans.append(p);cursor=p.next_cursor
    if token_counts != [p.counts.valid_tokens for p in plans]:raise ValueError('Fixture and saved token plan differ')
    scheduler=CampaignTokenSchedule(optimizer,token_counts,warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    generators={'data':torch.Generator().manual_seed(recipe.jitter_seed+rank),
                'local':torch.Generator(device=device).manual_seed(recipe.jitter_seed+world+rank)}
    seed_local(recipe.jitter_seed+rank,device)
    counters=TrainingCounters();migration=None
    if a.phase=='seed':
        config.update(model=model.config.to_dict(),schedule=scheduler.checkpoint_contract())
        config=plain(config);fingerprint={'schema':'olmo-topology-origin-sources-v1','sources':report['sources'],'sha256':digest_json({'configuration':config,'sources':report['sources']})}
        cursor=plans[0].start_cursor
    elif a.phase=='migrate':
        kwargs={} if a.source_rank is None else {'rng_rank_map':[a.source_rank]}
        migration=make_topology_contract(source_manifest,world_size=world,physical_batch_per_rank=a.batch_size,
             lineage=a.lineage,sources=report['sources'],**kwargs)
        restored=load_topology_checkpoint(a.checkpoint,model,optimizer,scheduler=scheduler,migration=migration,
             expected_manifest_sha256=a.checkpoint_sha256,generators=generators,device=device)
        counters=restored['counters'];config=restored['configuration'];fingerprint=restored['source_fingerprint']
        cursor=PackedCursor(**restored['data_cursor']['cursor'])
        report['migration_receipt']=restored['migration_receipt']
    else:
        if config.get('world_size')!=world or config.get('physical_batch_per_rank')!=a.batch_size:
            raise ValueError('Strict child resume allocation differs')
        fingerprint=source_manifest['metadata']['source_fingerprint']
        restored=load_distributed_checkpoint(a.checkpoint,model,optimizer,scheduler=scheduler,configuration=config,
             source_fingerprint=fingerprint,expected_manifest_sha256=a.checkpoint_sha256,generators=generators,device=device)
        counters=restored['counters'];cursor=PackedCursor(**restored['data_cursor']['cursor'])
        if restored['data_cursor']!=cursor_record(asdict(cursor),rank=rank,world_size=world,batch_size=a.batch_size):
            raise ValueError('Strict child cursor envelope differs')
        migration=config.get('topology_migration')
    if counters.optimizer_updates>a.stop_after or a.stop_after>len(plans):raise ValueError('Segment outside saved finite horizon')
    expected_cursor=plans[0].start_cursor if counters.optimizer_updates==0 else plans[counters.optimizer_updates-1].next_cursor
    if cursor!=expected_cursor:raise ValueError('Imported logical cursor differs from immutable plan')
    if data is not None:data.restore_cursor(asdict(cursor))
    logical_expected=asdict(expected_counters_since_origin(TrainingCounters(),plans,counters.optimizer_updates,recipe,world_size=world,batch_size=a.batch_size))
    if any(asdict(counters)[k]!=v for k,v in logical_expected.items() if k!='microbatches'):
        raise ValueError('Imported logical counters differ from immutable plan')
    if a.phase=='resume' and migration is not None:
        saved_expected=expected_counters_since_origin(migration['origin_counters'],plans,counters.optimizer_updates,recipe,world_size=world,batch_size=a.batch_size)
        if asdict(saved_expected)!=asdict(counters):raise ValueError('Strict child historical counters differ')
    origin_counters=copy.deepcopy(counters)
    def envelope():return cursor_record(asdict(cursor),rank=rank,world_size=world,batch_size=a.batch_size)
    def current_boundary():
        state=boundary_digests(model,optimizer,scheduler,counters)
        c.same('replicated state',state)
        return {'state':state,'canonical_cursor':asdict(cursor),
            'rank_rng':{str(i):x for i,x in enumerate(c.gather(tree_digests(_local_rng(device,generators))))}}
    c.call('restored clocks',lambda:validate_clocks(optimizer,scheduler,counters))
    active=[n for n,p in model.named_parameters() if p.requires_grad]
    report.update(configuration=config,source_fingerprint=fingerprint,active_parameter_names=active,
       common_origin_manifest_sha256=a.checkpoint_sha256,recipe=copy.deepcopy(config['recipe']),objective_weights=model.objective_weights(),
       imported_counters=asdict(counters),boundaries={'imported':current_boundary()},update_evidence={},updates=[])
    persist('imported')
    runner=None;manager=None;storage=None
    def drain():
        if manager is not None:
            record=manager.drain()
            if record is not None:report.setdefault('publications',[]).append(record)
    if a.storage_prefix:
        def create_storage():
            nonlocal storage,manager
            storage=SSDCheckpointStorage.create(a.checkpoint_root,a.output_dir,
                execution_identity_sha256=config['execution_identity']['sha256'],storage_prefix=a.storage_prefix,
                keep_local_completed=4,resume_source=a.checkpoint)
            manager=AsyncCheckpointRetention(storage,evidence_dir=a.output_dir,storage_prefix=a.storage_prefix,
                source_pins=report['sources'],retain_hook=retain_in_child)
        c.call('storage ownership',create_storage,rank_zero=True)
    else:c.call('new checkpoint root',lambda:a.checkpoint_root.mkdir(parents=True,exist_ok=False),rank_zero=True)
    def save(reason):
        number=counters.optimizer_updates;destination=a.checkpoint_root/f'update-{number:06d}'
        c.call('drain earlier publication',drain,rank_zero=True)
        if a.storage_prefix:c.call('checkpoint destination',lambda:storage.destination(number),rank_zero=True)
        c.call('checkpoint disk preflight',lambda:checkpoint_disk_preflight(destination,model))
        before=current_boundary()
        with runner.checkpoint_boundary() if runner is not None else nullcontext():
            receipt=save_distributed_checkpoint(destination,model,optimizer,scheduler=scheduler,counters=counters,
                 data_cursor=envelope(),configuration=config,source_fingerprint=fingerprint,generators=generators,device=device)
        if current_boundary()!=before:raise ValueError('Save changed live boundary')
        report.setdefault('checkpoints',[]).append({'reason':reason,'receipt':receipt,'boundary':before})
        if a.storage_prefix:
            def submit():report.setdefault('submissions',[]).append(manager.submit(receipt))
            c.call('submit asynchronous retention',submit,rank_zero=True)
        persist('saved/'+str(number))
        return destination,receipt
    try:
        previous_path,previous_receipt=save('imported-boundary')
        comparison_origin_pin=(migration['comparison_origin_manifest_sha256'] if migration is not None and counters.optimizer_updates==migration['origin_counters']['optimizer_updates'] else (a.checkpoint_sha256 or previous_receipt['manifest_sha256']))
        report['common_origin_manifest_sha256']=comparison_origin_pin
        if counters.optimizer_updates<a.stop_after:
            def materialize(index):
                if data is None:return fixture_for_rank(recipe,index,rank=rank,world_size=world)
                plan=plans[index]
                packed=data.rank_batches(plan,rank=rank,world_size=world,physical_batch_size=a.batch_size)
                noises=tuple(feedback_noise_for_rows(recipe,ks,logical_update=index,sequence_length=recipe.sequence_length,
                    width=model.config.model_dim,physical_batch_size=a.batch_size) for ks in packed.keys)
                return packed.batches,noises,packed.keys
            batches,noises,keys=c.call('graph preparation input',lambda:materialize(counters.optimizer_updates))
            counts=sum_objective_counts(c.gather(sum_objective_counts([model.counts(b) for b in batches])))
            adapter=CampaignObjective(model,batches[0],mode=recipe.mode(),global_counts=counts,world_size=world,
                feedback_noise=noises[0],config=LMTrainingConfig(precision=report['precision'],max_grad_norm=recipe.max_grad_norm))
            runner=CampaignDDPGraphTraining(adapter)
            start=time.monotonic();before=current_boundary()
            def phase(name,action):c.call('graph observer',lambda:persist('graph/'+name+'/'+action),rank_zero=True)
            runner.prepare(warmup=11,phase_observer=phase)
            runner.capture(warmup=11,release_transient_cache=True,phase_observer=phase)
            report['preparation_seconds_by_rank']=c.gather(time.monotonic()-start)
            report['boundaries']['prepared']=current_boundary()
            if before!=report['boundaries']['prepared']:raise ValueError('Graph preparation changed restored boundary')
            del batches,noises,keys
            persist('prepared')
            while counters.optimizer_updates<a.stop_after:
                index=counters.optimizer_updates
                start_boundary=current_boundary()
                batches,noises,keys=c.call('update input',lambda:materialize(index))
                counts=sum_objective_counts(c.gather(sum_objective_counts([model.counts(b) for b in batches])))
                rows=sum(c.gather(input_rows(batches,noises,keys)),[])
                row_map={r['key']:r for r in rows}
                canonical_keys=[r.key for r in plans[index].rows]
                if len(row_map)!=len(rows) or set(row_map)!=set(canonical_keys):raise ValueError('Logical input membership differs')
                inputs={'logical_update':index,'counts':counts,'lr_used':[g['lr'] for g in optimizer.param_groups],'input_tokens':plans[index].counts.valid_tokens,
                        'rows':[row_map[k] for k in canonical_keys]}
                scheduler.validate_next_update(plans[index].counts.valid_tokens)
                started=time.time();result=runner.backward(batches,feedback_noises=noises,replay=True)
                grad_path=a.checkpoint_root/f'raw-gradients-{index+1:06d}.pt'
                def export_gradients():
                    grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
                    if any(not bool(torch.isfinite(t).all()) for t in grads.values()):raise ValueError('Nonfinite raw gradients')
                    torch.save(grads,grad_path)
                    return artifact(grad_path)
                grad_record=c.call('raw gradient artifact',export_gradients,rank_zero=True)
                metrics=runner.step(result,optimizer,scheduler=scheduler,counters=counters)
                if data is not None:data.commit(plans[index].start_cursor,plans[index])
                cursor=plans[index].next_cursor
                expected=expected_counters_since_origin(origin_counters,plans,counters.optimizer_updates,recipe,world_size=world,batch_size=a.batch_size)
                if asdict(counters)!=asdict(expected):raise ValueError('Counters differ after topology change')
                c.call('completed clocks',lambda:validate_clocks(optimizer,scheduler,counters))
                final=current_boundary();duration=time.time()-started
                report['updates'].append({'update':counters.optimizer_updates,'metrics':metrics,'wall_seconds':duration,
                    'memory_by_rank':c.gather(memory()),'input':inputs})
                path,receipt=save('completed-update')
                evidence={'boundaries':{'imported':start_boundary,'prepared':start_boundary,'final':final},
                    'next_update_input':inputs,'numeric_artifacts':{
                        'initial':{'path':str(previous_path/'state.pt'),'sha256':previous_receipt['state']['sha256'],'key':'model'},
                        'final':{'path':str(path/'state.pt'),'sha256':receipt['state']['sha256'],'key':'model'},'gradients':grad_record},
                    'common_origin_manifest_sha256':comparison_origin_pin,
                    'active_parameter_names':active,'world_size':world}
                report['update_evidence'][str(counters.optimizer_updates)]=evidence
                report['next_update_input']=inputs;report['numeric_artifacts']=evidence['numeric_artifacts']
                report['boundaries']['final']=final
                def log():tracker.log({'update':counters.optimizer_updates,'train/objective':metrics['objective'],
                    'train/gradient_norm':metrics['gradient_norm_before_clip'],'train/lr':inputs['lr_used'][0]})
                c.call('tracking update',log,rank_zero=True)
                previous_path,previous_receipt=path,receipt
                comparison_origin_pin=receipt['manifest_sha256']
                del batches,noises,keys
                persist('update/'+str(counters.optimizer_updates))
        else:
            report['boundaries']['prepared']=report['boundaries']['imported']
            report['boundaries']['final']=report['boundaries']['imported']
        if counters.optimizer_updates==len(plans):
            try:scheduler.validate_next_update(1)
            except ValueError:report['exhausted_schedule_rejected']=True
            else:raise ValueError('Exhausted schedule accepted another update')
        c.call('terminal cloud drain',drain,rank_zero=True)
        if a.storage_prefix:c.call('final storage authority',lambda:storage.validate(),rank_zero=True)
        report['final_counters']=asdict(counters);persist('updates_complete')
    except BaseException:
        if manager is not None:
            try:manager.abort()
            except BaseException:pass
        raise
    else:
        if manager is not None:manager.close()
    finally:
        if data is not None:data.close()
    if runner is not None:
        release_completed_graph_runner(runner);del runner,adapter
    del model,optimizer,scheduler;gc.collect();torch.cuda.synchronize(device)


def main(argv=None):
    a=parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Use required project GPU container')
    if any(os.environ.get(k)!='0' for k in ('NCCL_ASYNC_ERROR_HANDLING','TORCH_NCCL_ASYNC_ERROR_HANDLING')):
        raise RuntimeError('Captured NCCL needs both async-error flags0 and external timeout')
    device,runtime,determinism=configure_cuda_runtime(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',timeout=timedelta(seconds=1800),device_id=device)
    c=Coordinator();report={'schema':SCHEMA,'status':'running','phase':a.phase,'scale':a.scale,'arm':a.arm,
        'world_size':c.world_size,'physical_batch_per_rank':a.batch_size,'precision':'fp32' if a.scale=='tiny' else 'bf16_mixed',
        'sources':source_hashes(),'runtime':runtime,'determinism':determinism,'scope':'Bounded topology/restart functionality; native finite127-to128 horizon unchanged'}
    tracker=None;error=None;start=time.monotonic()
    def persist(stage=None):
        if stage is not None:report['stage']=stage
        report['elapsed_seconds']=time.monotonic()-start
        if c.rank==0:
            report['wandb']=None if tracker is None else tracker.record
            write_json(a.output_dir/'report.json',report)
    try:
        def setup():
            nonlocal tracker
            a.output_dir.mkdir(parents=True,exist_ok=False)
            for name,pin in report['sources'].items():
                dest=a.output_dir/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
                if sha256_file(dest)!=pin:raise ValueError('Source changed during snapshot')
            tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',output_dir=a.output_dir,
                name=a.output_dir.name,group='olmo-topology-readiness',preserve_state=preserve_local_rng)
            tracker.start({k:report[k] for k in ('phase','scale','arm','world_size','physical_batch_per_rank','scope')})
            print({'wandb':tracker.record['run_url']},flush=True)
        c.call('output and tracking',setup,rank_zero=True);persist('setup')
        with disable_autocast_weight_cache(),sdpa_kernel(SDPBackend.MATH if a.scale=='tiny' else SDPBackend.FLASH_ATTENTION):
            run(a,c,device,report,tracker,persist)
        if source_hashes()!=report['sources']:raise ValueError('Source changed during execution')
        dist.destroy_process_group();report.update(status='completed',teardown='completed')
    except BaseException as exc:
        error=exc;report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if c.rank==0 and a.output_dir.is_dir():
            persist()
            try:
                if tracker is not None:tracker.finish(succeeded=error is None)
            finally:persist()

if __name__=='__main__':main()
