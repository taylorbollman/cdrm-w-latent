#!/usr/bin/env python3
"""Bounded full-stack RT training capacity probe; random updates are discarded."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.functional as F

from experiment_tracking import OnlineTracker, add_wandb_arguments
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit


def head_chunked_backward(model, ids, *, chunk_size=2, bf16=True):
    """Released trainer's head-only microbatching and B*T loss denominator.

    All recurrent layers see the complete physical batch. Only the output head
    is split. Its accumulated hidden cotangent is propagated through the full
    backbone exactly once; no recurrent gradient accumulation is introduced.
    """
    if chunk_size < 1 or ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError('Require positive head chunks and [B,T>=2] token IDs')
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=bf16):
        hidden = model(ids, return_pre_logits=True, return_logits=False).pre_logits
    detached = hidden.detach().requires_grad_(True)
    labels = ids[:, 1:].contiguous()
    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    denominator = ids.numel()
    for begin in range(0, len(ids), chunk_size):
        end = min(begin + chunk_size, len(ids))
        with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model.transformer.ff_out(detached[begin:end])
            if model.config.scale_logits:
                logits = logits * (model.config.d_model ** -0.5)
            shifted = logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1])
            loss_sum = F.cross_entropy(shifted.float(), labels[begin:end].reshape(-1), reduction='sum')
        total += loss_sum.detach()
        (loss_sum / denominator).backward()
        del logits, shifted, loss_sum
    if detached.grad is None:
        raise AssertionError('Missing accumulated output-head input gradient')
    hidden.backward(detached.grad)
    return total / denominator


def memory():
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_free_bytes': free, 'device_total_bytes': total}


def precision_check(model, optimizer):
    parameters = list(model.parameters())
    gradients = [p.grad for p in parameters]
    moments = [state[key] for state in optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq')]
    if any(g is None for g in gradients) or len(moments) != 2 * len(parameters):
        raise AssertionError('Every parameter needs a gradient and allocated Adam moments')
    for label, values in (('parameters', parameters), ('gradients', gradients), ('moments', moments)):
        if any(value.dtype != torch.float32 for value in values):
            raise AssertionError(f'{label} must remain FP32')
        if not torch.stack([torch.isfinite(value).all() for value in values]).all().item():
            raise FloatingPointError(f'Nonfinite {label}')
    return {'all_finite': True, 'parameters_gradients_moments': 'torch.float32',
            'parameter_tensors': len(parameters), 'moment_tensors': len(moments),
            'adam_state_bytes': sum(v.numel()*v.element_size() for v in moments)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--policy', choices=('bf16_fp32_state', 'legacy'), default='bf16_fp32_state')
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--cuda-graph', action='store_true',
                        help='Opt in to validated fixed-shape full forward/backward CUDA capture')
    parser.add_argument('--bwd-mlp-chunks', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='recurrent-transformer-capacity')
    args = parser.parse_args()
    if not 1 <= args.batch <= 1024 or not 2 <= args.warmup < args.updates <= 20 or args.bwd_mlp_chunks < 1:
        parser.error('Require batch1..1024 and 2<=warmup<updates<=20')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh process and output directory for each candidate')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-batch-profile-v2', 'status': 'running', 'batch': args.batch,
              'length': 512, 'warmup_updates': args.warmup, 'updates': [], 'policy': args.policy,
              'scope': 'Physical full-backbone training batch; random data; no task-performance or new numerical-clearance claim',
              'head_microbatch': 2, 'backbone_accumulation_steps': 1,
              'loss': 'Sum CE over511 shifted targets per sequence divided by B*512, matching released trainer',
              'cuda_graphs': args.cuda_graph, 'whole_model_compile': False,
              'outer_activation_checkpointing': False, 'internal_tiled_recomputation': True,
              'optimizer': {'name': 'AdamW', 'lr': 1e-3, 'betas': [0.9,0.95], 'eps': 1e-8,
                            'weight_decay': 0., 'foreach': False, 'fused': False, 'clip_norm': 1.},
              'random_seed': 20260910, 'checkpoints_retained': False}
    tracker = None
    started = time.monotonic()
    exit_code = 0
    try:
        report['hardware'] = require_cuda_container()
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                             'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        paths = [Path(__file__), Path('configs/stage_a/full.json'),
                 Path('scripts/rt_cuda_graph.py'),
                 Path('scripts/stage_a_common.py'), Path('scripts/stage_b_train.py'),
                 Path('scripts/experiment_tracking.py')]
        paths += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256'] = {}
        for path in paths:
            relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
            data = path.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
            target = args.output_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        raw = json.loads(Path('configs/stage_a/full.json').read_text())
        raw.update(block_type='recurrent', recurrent_layers=None, recurrent_backend='tiled',
                   recurrent_write_rho=1., recurrent_precision_policy=args.policy,
                   ordinary_attention_precision_policy='legacy', reference_eager=False,
                   cdrm_enabled=False, init_device='cuda', precision=None,
                   bwd_mlp_chunks=args.bwd_mlp_chunks)
        from olmo.config import ModelConfig
        from olmo.model import OLMo, OLMoRecurrentBlockTiled
        config = ModelConfig(**raw)
        report['model_config'] = dataclasses.asdict(config)
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({k:v for k,v in report.items() if k not in ('source_sha256','wandb','updates')})
        seed_all(20260910, deterministic=True)
        model = OLMo(config).to(device='cuda', dtype=torch.float32).train()
        assert len(model.transformer.blocks) == 12
        assert all(isinstance(block, OLMoRecurrentBlockTiled) for block in model.transformer.blocks)
        assert model.activation_checkpointing_strategy is None and not model.config.weight_tying
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['backbone_parameters'] = report['parameter_count'] - sum(
            p.numel() for module in (model.transformer.wte, model.transformer.ff_out) for p in module.parameters())
        assert report['parameter_count'] == 216843264 and report['backbone_parameters'] == 151045120
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.95), eps=1e-8,
                                      weight_decay=0., foreach=False, fused=False)
        generator = torch.Generator(device='cuda').manual_seed(20260910)
        ids = torch.randint(0, config.vocab_size, (args.batch,512), device='cuda', generator=generator)
        report['ids_sha256'] = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report['baseline_memory'] = memory()
        atomic_json(args.output_dir/'progress.json', report, replace=True)
        captured = None
        if args.cuda_graph:
            from rt_cuda_graph import CapturedRTBackward
            # An uncaptured check at the actual physical shape and initial weights.
            # CPU copies are temporary RAM only; no large random checkpoints retained.
            print('Checking uncaptured gradients before capture', flush=True)
            reference_loss = head_chunked_backward(model, ids).item()
            reference_grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            print('Warming up and capturing complete forward/backward', flush=True)
            captured = CapturedRTBackward(model, ids, precision_policy=args.policy)
            actual_loss = captured.replay(ids).item()
            gradient_rows = []
            for name, parameter in model.named_parameters():
                reference = reference_grads.pop(name)
                actual = parameter.grad.detach().cpu()
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7, msg=name)
                delta = actual-reference
                gradient_rows.append({'name': name, 'max_abs_error': delta.abs().max().item(),
                                      'relative_l2': (delta.double().norm()/reference.double().norm().clamp_min(1e-30)).item(),
                                      'bitwise_equal': torch.equal(actual, reference)})
            del reference_grads, reference, actual, delta
            if abs(actual_loss-reference_loss) > 1e-6 + 1e-6*abs(reference_loss):
                raise AssertionError('Captured loss differs from uncaptured reference')
            report['capture_validation'] = {'status': 'passed', 'reference_loss': reference_loss,
                                            'captured_loss': actual_loss, 'gradients': gradient_rows,
                                            'gradient_budget': {'rtol': 1e-5, 'atol': 1e-7}}
            report['capture_setup_memory'] = memory()
            report['capture'] = captured.metadata()
            torch.cuda.reset_peak_memory_stats()
            atomic_json(args.output_dir/'progress.json',report,replace=True)
            print(json.dumps({'capture_validation': 'passed', 'capture': report['capture'],
                              'capture_setup_memory': report['capture_setup_memory']}),flush=True)
        for index in range(args.updates):
            if index == args.warmup:
                report['cold_peak_memory'] = memory()
                report['compiler_after_warmup'] = compiler_audit(True)
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            if captured is None:
                optimizer.zero_grad(set_to_none=True)
                loss = head_chunked_backward(model, ids)
            else:
                loss = captured.replay(ids)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite loss')
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-update_start
            row = {'update': index+1, 'warmup': index<args.warmup, 'seconds': elapsed,
                   'loss': loss.item(), 'gradient_norm': gradient_norm.item(), 'memory': memory()}
            del loss, gradient_norm
            report['updates'].append(row)
            tracker.log({'update': index+1, 'benchmark/update_seconds': elapsed,
                         'benchmark/tokens_per_second': args.batch*512/elapsed,
                         'benchmark/peak_allocated_GiB': row['memory']['peak_allocated_bytes']/2**30,
                         'benchmark/device_free_GiB': row['memory']['device_free_bytes']/2**30,
                         'benchmark/random_data_loss': row['loss']})
            atomic_json(args.output_dir/'progress.json',report,replace=True)
            print(json.dumps(row),flush=True)
        report['steady_peak_memory'] = memory()
        if captured is not None:
            report['capture'] = captured.metadata()
        report['final_precision_check'] = precision_check(model,optimizer)
        report['compiler'] = compiler_audit(True)
        old_graphs = report['compiler_after_warmup']['counters'].get('stats',{}).get('unique_graphs',0)
        new_graphs = report['compiler']['counters'].get('stats',{}).get('unique_graphs',0)
        report['steady_compilation_free'] = old_graphs==new_graphs
        if not report['steady_compilation_free']:
            raise RuntimeError('New helper compilations during measured updates')
        times = [row['seconds'] for row in report['updates'] if not row['warmup']]
        report['timing'] = {'median_update_seconds': statistics.median(times),
                            'mean_update_seconds': statistics.mean(times),
                            'tokens_per_second': args.batch*512/statistics.mean(times),
                            'measured_updates': len(times)}
        report['status'] = 'complete'
        tracker.summary({**report['timing'], 'batch': args.batch,
                         'peak_allocated_GiB': report['steady_peak_memory']['peak_allocated_bytes']/2**30,
                         'peak_reserved_GiB': report['steady_peak_memory']['peak_reserved_bytes']/2**30})
        for path,digest in report['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Profile sources changed during measurement')
    except torch.OutOfMemoryError as error:
        report.update(status='out_of_memory', error_type=type(error).__name__, error=str(error), oom_memory=memory())
        exit_code = 42
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            if tracker:
                tracker.finish(succeeded=report['status']=='complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            exit_code = 1
        atomic_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(args.output_dir/'report.json')}),flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
