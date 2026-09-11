#!/usr/bin/env python3
"""Held-out token/document CE for a saved RT precision-pilot checkpoint."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_precision_data import file_sha256
from rt_precision_eval_contract import validate_evaluation_contract
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit


def aggregate_documents(losses, boundaries):
    """Map shifted targets to their source document without changing supervision.

    Input losses are [rows,T-1]. Position zero in each packed row has no target;
    transitions between documents inside a row remain ordinary supervised tokens.
    """
    if losses.ndim != 2 or not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError('Require finite nonnegative [rows,T-1] CE')
    full = np.zeros((len(losses), losses.shape[1] + 1), dtype=np.float64)
    full[:, 1:] = losses
    counts = np.ones_like(full, dtype=np.int64)
    counts[:, 0] = 0
    sums = np.concatenate(([0.], full.reshape(-1).cumsum()))
    nums = np.concatenate(([0], counts.reshape(-1).cumsum()))
    rows, end = [], 0
    for document in boundaries:
        begin, stop = document['token_begin'], document['token_end']
        if begin != end or stop <= begin:
            raise ValueError('Document boundaries must cover the token stream exactly once')
        end = stop
        if begin >= full.size:
            break
        stop = min(stop, full.size)
        count = int(nums[stop] - nums[begin])
        rows.append({'text_sha256': document['text_sha256'], 'targets': count,
                     'ce_sum': float(sums[stop] - sums[begin])})
    if end < full.size or sum(row['targets'] for row in rows) != losses.size:
        raise ValueError('Document boundaries do not cover all supervised targets')
    return rows


def paired_document_interval(reference, candidate, *, resamples=10000, seed=20260912):
    """Paired document bootstrap of a token-weighted mean CE difference."""
    if len(reference) != len(candidate) or len(reference) < 2 or resamples < 2:
        raise ValueError('Require matching document packets and at least two resamples')
    for a, b in zip(reference, candidate):
        if a['text_sha256'] != b['text_sha256'] or a['targets'] != b['targets']:
            raise ValueError('Paired evaluation documents/target counts differ')
    count = np.array([a['targets'] for a in reference], dtype=np.float64)
    delta = np.array([b['ce_sum']-a['ce_sum'] for a,b in zip(reference,candidate)])
    valid = count > 0
    count, delta = count[valid], delta[valid]
    if len(count) < 2:
        raise ValueError('Insufficient documents with supervised targets')
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, resamples, 100):
        indices = rng.integers(len(count), size=(min(100,resamples-start), len(count)))
        values.extend((delta[indices].sum(1)/count[indices].sum(1)).tolist())
    return {'candidate_minus_reference_nats_per_target':float(delta.sum()/count.sum()),
            'interval_95_percent':np.quantile(values,[.025,.975]).tolist(),
            'one_sided_95_upper':float(np.quantile(values,.95)),
            'resamples':resamples,'seed':seed,'documents':len(count),
            'qualification':'Document bootstrap on this fixed packed slice; no long-run or peak-LR claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--role', choices=('diagnostic','dev','confirmation'), required=True)
    parser.add_argument('--precision', choices=('fp32','native'), required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args=parser.parse_args()
    if args.batch < 1 or not args.wandb_project:
        parser.error('Positive evaluation batch and online W&B required')
    if args.output_dir.exists(): raise FileExistsError('Use a fresh evaluation directory')
    args.output_dir.mkdir(parents=True)
    report={'schema':'rt-precision-evaluation-v1','status':'running','role':args.role,
            'precision':args.precision,'batch':args.batch,'cuda_graphs':False,
            'scope':'Observer-free native forward with compiled helpers, no optimizer updates',
            'normalization':'Per supervised token (511 per 512-token row)'}
    tracker=None; started=time.monotonic()
    try:
        report['hardware']=require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors=False
        torch._dynamo.utils.counters.clear()
        report['runtime']={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                           'cache':os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        sources=[Path(__file__),Path('scripts/rt_precision_data.py'),Path('scripts/rt_precision_eval_contract.py'),Path('scripts/stage_a_common.py'),
                 Path('scripts/stage_b_train.py'),Path('scripts/experiment_tracking.py')]
        sources+=sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256']={}
        for source in sources:
            relative=source.relative_to(Path.cwd()) if source.is_absolute() else source
            data=source.read_bytes();report['source_sha256'][str(relative)]=hashlib.sha256(data).hexdigest()
            target=args.output_dir/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest=json.loads(args.data_manifest.read_text())
        if manifest['schema']!='rt-precision-c4-data-v1' or manifest['mode']!='heldout':
            raise ValueError('Require pinned heldout data manifest')
        role=manifest['roles'][args.role]
        data_path=args.data_manifest.parent/role['ids_path']
        boundary_path=args.data_manifest.parent/role['boundaries_path']
        if file_sha256(data_path)!=role['ids_sha256'] or file_sha256(boundary_path)!=role['boundaries_sha256']:
            raise ValueError('Heldout data or boundaries changed')
        ids=np.load(data_path,mmap_mode='r',allow_pickle=False)
        if list(ids.shape)!=role['shape'] or ids.dtype!=np.uint16 or ids.shape[1]!=512:
            raise ValueError('Unexpected heldout token matrix')
        report['data']={'manifest_sha256':file_sha256(args.data_manifest),**role}
        report['checkpoint']={'path':str(args.checkpoint),'sha256':file_sha256(args.checkpoint)}
        training_report=json.loads(args.training_report.read_text())
        report['training_report']={'path':str(args.training_report),'sha256':file_sha256(args.training_report)}
        report['protocol_sha256']=file_sha256(args.protocol)
        # Only locally generated, trusted project checkpoints are accepted here.
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if state.get('schema')!='rt-precision-state-v1':
            raise ValueError('Require rt-precision-state-v1 checkpoint')
        raw=dict(state['model_config']);raw.update(init_device='cuda',precision=None)
        from olmo.config import ModelConfig
        from olmo.model import OLMo, OLMoRecurrentBlockTiled
        config=ModelConfig(**raw)
        report['model_config']=dataclasses.asdict(config)
        report['checkpoint_update']=state.get('completed_updates',state.get('update'))
        tracker=OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
                              name=args.wandb_run_name,output_dir=args.output_dir)
        report['wandb']=tracker.record
        tracker.start({k:v for k,v in report.items() if k not in ('source_sha256','wandb')})
        seed_all(20260912,deterministic=True)
        model=OLMo(config).to('cuda',dtype=torch.float32).eval()
        report['checkpoint_identity']=validate_evaluation_contract(
            training_report,state,checkpoint_sha256=report['checkpoint']['sha256'],
            heldout_manifest_sha256=report['data']['manifest_sha256'],
            current_source_sha256=report['source_sha256'],protocol_sha256=report['protocol_sha256'],
            expected_parameters=dict(model.named_parameters()))
        model.load_state_dict(state['model'],strict=True);del state
        if len(model.transformer.blocks)!=12 or not all(isinstance(b,OLMoRecurrentBlockTiled) for b in model.transformer.blocks):
            raise ValueError('Evaluation requires the 12-layer all-tiled RT')
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 master weights required')
        losses=np.empty((len(ids),511),dtype=np.float32)
        native=args.precision=='native'
        with torch.no_grad():
            for begin in range(0,len(ids),args.batch):
                end=min(begin+args.batch,len(ids))
                tokens=torch.as_tensor(np.array(ids[begin:end],dtype=np.int64),device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=native):
                    hidden=model(tokens,return_pre_logits=True,return_logits=False).pre_logits
                    for part in range(0,len(tokens),2):
                        logits=model.transformer.ff_out(hidden[part:part+2])
                        if config.scale_logits:logits=logits*(config.d_model**-.5)
                        ce=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]).float(),
                                           tokens[part:part+2,1:].reshape(-1),reduction='none')
                        losses[begin+part:min(begin+part+2,end)]=ce.reshape(-1,511).cpu().numpy()
                        del logits,ce
                del hidden,tokens
        if not np.isfinite(losses).all():raise FloatingPointError('Nonfinite heldout loss')
        np.save(args.output_dir/'token-ce.npy',losses,allow_pickle=False)
        boundaries=[json.loads(line) for line in boundary_path.read_text().splitlines()]
        if len(boundaries)!=role['documents'] or not boundaries or boundaries[-1]['token_end']!=ids.size:
            raise ValueError('Production boundaries must end exactly at the complete evaluated stream')
        documents=aggregate_documents(losses,boundaries)
        atomic_json(args.output_dir/'documents.json',documents)
        report['ce_nats_per_supervised_token']=float(losses.mean(dtype=np.float64))
        report['supervised_tokens']=int(losses.size)
        report['documents']=len(documents)
        report['token_ce_sha256']=file_sha256(args.output_dir/'token-ce.npy')
        report['documents_sha256']=file_sha256(args.output_dir/'documents.json')
        report['compiler']=compiler_audit(True)
        for path,digest in report['source_sha256'].items():
            if file_sha256(Path(path))!=digest:raise RuntimeError('Evaluation source changed during run')
        tracker.log({'evaluation/ce_nats_per_supervised_token':report['ce_nats_per_supervised_token']})
        tracker.summary({'ce_nats_per_supervised_token':report['ce_nats_per_supervised_token'],
                         'supervised_tokens':report['supervised_tokens']})
        report['status']='complete'
    except BaseException as error:
        report.update(status='execution_failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        try:
            if tracker:tracker.finish(succeeded=report['status']=='complete')
        except BaseException as error:
            report.update(status='execution_failed',sync_error_type=type(error).__name__);raise
        finally:atomic_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'ce':report['ce_nats_per_supervised_token']}),flush=True)


if __name__=='__main__':main()
