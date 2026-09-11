#!/usr/bin/env python3
"""Observe compiled tiled RT tensor boundaries without changing its arithmetic.

Run each arm in a fresh CUDA-container process. This small diagnostic does not
capture CUDA graphs or perform optimizer updates; separate validation clears
the observer-free captured execution path at its actual training dimensions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import head_chunked_backward
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit

ARMS = {
    'A': {'policy': 'bf16_fp32_state', 'bf16': True},
    'B': {'policy': 'legacy', 'bf16': True},
    'C': {'policy': 'bf16_fp32_state', 'bf16': False},
}
SEED = 20260910
HELPER_ARGUMENTS = {
    'block_attention_add': ('weighted_values', 'max_logit', 'sum_scores', 'k', 'v', 'q', 'attention_bias'),
    'recompute_alphas': ('final_k', 'k_init', 'q', 'attention_bias'),
    'recompute_atts': ('final_v', 'v_init', 'alphas'),
    'mlp_batched_body': ('attention', 'x'),
}
HELPER_RESULTS = {
    'block_attention_add': ('weighted_values', 'max_logit', 'sum_scores'),
    'recompute_alphas': ('alphas',),
    'recompute_atts': ('attention',),
    'mlp_batched_body': ('output',),
}


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def copied_gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32 or parameter.grad is None:
            raise AssertionError(f'Missing gradient or non-FP32 master parameter: {name}')
        value = parameter.grad.detach().cpu().clone()
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise AssertionError(f'Gradient must be finite FP32: {name}')
        result[name] = value
    return result


def exact_comparison(reference, actual):
    """Require equality of every raw gradient, including small coordinates."""
    if set(reference) != set(actual):
        raise AssertionError('Parameter gradient names changed under observation')
    rows = {}
    for name in sorted(reference):
        left, right = reference[name], actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'Gradient shape or dtype changed: {name}')
        rows[name] = {
            'dtype': str(left.dtype), 'numel': left.numel(),
            'exact': torch.equal(left, right),
            'max_abs_error': (left.double() - right.double()).abs().max().item(),
        }
    return {'exact': all(row['exact'] for row in rows.values()),
            'tensor_count': len(rows), 'numel': sum(row['numel'] for row in rows.values()),
            'max_abs_error': max((row['max_abs_error'] for row in rows.values()), default=0.),
            'reference_sha256': tensor_digest(reference), 'actual_sha256': tensor_digest(actual),
            'tensors': rows}


class DtypeObservations:
    """Collect metadata only: no tensor reads, copies, math, or retained tensors.

    Some observed gradient buffers are intentionally uninitialized at allocation,
    so even finite-value probes would be inappropriate at these boundaries.
    """
    def __init__(self):
        self._rows = {}

    def record(self, layer, phase, tensors, token_index=None):
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            key = (str(layer), phase, name, str(tensor.dtype), tuple(tensor.shape),
                   tensor.requires_grad, torch.is_autocast_enabled('cuda'))
            row = self._rows.setdefault(key, {
                'layer': layer, 'phase': phase, 'tensor': name,
                'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                'requires_grad': tensor.requires_grad,
                'ambient_cuda_autocast_enabled': torch.is_autocast_enabled('cuda'),
                'observations': 0, 'token_indices': [],
            })
            row['observations'] += 1
            if token_index is not None and token_index not in row['token_indices']:
                row['token_indices'].append(token_index)

    def rows(self):
        return [self._rows[key] for key in sorted(self._rows)]

    @contextmanager
    def installed(self, model):
        import olmo.model as model_module

        blocks = tuple(model.transformer.blocks)
        if any(getattr(block, '_recurrent_precision_observer', None) is not None for block in blocks):
            raise AssertionError('This diagnostic requires a model without existing observers')
        layers = {id(block): index for index, block in enumerate(blocks)}
        old_attributes = [(block, hasattr(block, '_recurrent_precision_observer'),
                           getattr(block, '_recurrent_precision_observer', None)) for block in blocks]
        original_helper = model_module.recurrent_helper

        def observed_helper(block, helper, *args):
            name = getattr(helper, '_torchdynamo_orig_callable', helper).__name__
            if name not in HELPER_ARGUMENTS:
                raise AssertionError(f'Unexpected recurrent helper: {name}')
            self.record(layers[id(block)], f'helper.{name}.input',
                        dict(zip(HELPER_ARGUMENTS[name], args)))
            result = original_helper(block, helper, *args)
            outputs = result if isinstance(result, tuple) else (result,)
            self.record(layers[id(block)], f'helper.{name}.output',
                        dict(zip(HELPER_RESULTS[name], outputs)))
            return result

        def head_observer(module, args, output):
            self.record('head', 'forward.head', {'hidden': args[0], 'logits': output})

        handle = None
        try:
            for index, block in enumerate(blocks):
                def callback(phase, tensors, token_index=None, layer=index):
                    self.record(layer, phase, tensors, token_index)
                object.__setattr__(block, '_recurrent_precision_observer', callback)
            # This wrapper runs outside compiled helper bodies. It invokes the
            # same decorated functions with the same arguments and return values.
            model_module.recurrent_helper = observed_helper
            handle = model.transformer.ff_out.register_forward_hook(head_observer)
            yield
        finally:
            model_module.recurrent_helper = original_helper
            if handle is not None:
                handle.remove()
            for block, existed, value in old_attributes:
                if existed:
                    object.__setattr__(block, '_recurrent_precision_observer', value)
                elif hasattr(block, '_recurrent_precision_observer'):
                    object.__delattr__(block, '_recurrent_precision_observer')


def tiny_config(arm):
    from olmo.config import ModelConfig
    raw = json.loads(Path('configs/stage_a/full.json').read_text())
    raw.update(d_model=64, n_heads=4, n_kv_heads=4, mlp_hidden_size=256, n_layers=2,
               max_sequence_length=16, vocab_size=128, embedding_size=128,
               block_type='recurrent', recurrent_layers=None, recurrent_backend='tiled',
               recurrent_write_rho=1., recurrent_precision_policy=ARMS[arm]['policy'],
               ordinary_attention_precision_policy='legacy', reference_eager=False,
               cdrm_enabled=False, init_device='cuda', precision=None, bwd_mlp_chunks=4)
    return ModelConfig(**raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(ARMS), required=True,
                        help='Use a fresh process/output directory for each arm')
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh dtype-observation output directory')
    if not args.wandb_project:
        parser.error('Online W&B is required')
    args.output_dir.mkdir(parents=True)
    report = {
        'schema': 'rt-precision-dtype-v1', 'status': 'running', 'arm': args.arm,
        **ARMS[args.arm], 'seed': SEED, 'shape': [3, 16], 'head_chunk_size': 2,
        'loss': 'Native shifted CE sum over 15 targets per sequence, divided by B*16',
        'warmup_backwards': 2, 'optimizer_updates': 0, 'cuda_graphs': False,
        'whole_model_compile': False, 'compiled_recurrent_helpers': True,
        'scope': 'Tensor boundary dtypes on a tiny all-recurrent fixture; no full-size numerical clearance',
        'observation_limit': 'Tensor dtypes do not establish GEMM accumulator precision or internal fused-operation/exponential dtypes',
        'comparisons': [],
    }
    tracker = None
    started = time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        sources = [Path(__file__), Path('scripts/rt_batch_profile.py'), Path('scripts/stage_a_common.py'),
                   Path('scripts/stage_b_train.py'), Path('scripts/experiment_tracking.py'),
                   Path('configs/stage_a/full.json')]
        sources += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256'] = {}
        for source in sources:
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            data = source.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
            target = args.output_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        config = tiny_config(args.arm)
        report['model_config'] = dataclasses.asdict(config)
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb')})
        seed_all(SEED, deterministic=True)
        report['runtime'] = {
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        from olmo.model import OLMo, OLMoRecurrentBlockTiled
        model = OLMo(config).to(device='cuda', dtype=torch.float32).train()
        if len(model.transformer.blocks) != 2 or not all(
                isinstance(block, OLMoRecurrentBlockTiled) for block in model.transformer.blocks):
            raise AssertionError('Require exactly two tiled recurrent blocks in this fixture')
        if model.activation_checkpointing_strategy is not None or model.config.weight_tying:
            raise AssertionError('Require no outer checkpointing and an untied head')
        generator = torch.Generator(device='cuda').manual_seed(SEED + 1)
        ids = torch.randint(0, config.vocab_size, report['shape'], device='cuda', generator=generator)
        report['ids_sha256'] = tensor_digest({'ids': ids})
        report['initial_state_sha256'] = tensor_digest(model.state_dict())
        report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        bf16 = ARMS[args.arm]['bf16']

        def backward():
            model.zero_grad(set_to_none=True)
            loss = head_chunked_backward(model, ids, chunk_size=2, bf16=bf16)
            torch.cuda.synchronize()
            if loss.dtype != torch.float32 or not torch.isfinite(loss).item():
                raise AssertionError('Native CE loss must be finite FP32')
            return loss.detach().cpu().clone()

        for _ in range(report['warmup_backwards']):
            backward()
        reference_loss = backward()
        reference_grads = copied_gradients(model)
        report['reference_loss'] = reference_loss.item()
        report['reference_gradients_sha256'] = tensor_digest(reference_grads)
        report['compiler_before_observation'] = compiler_audit(True)
        observations = DtypeObservations()
        with observations.installed(model):
            observed_loss = backward()
        report['observations'] = observations.rows()
        report['compiler_after_observation'] = compiler_audit(True)

        def check(label, loss):
            row = {'execution': label, 'loss': loss.item(),
                   'loss_exact': torch.equal(reference_loss, loss),
                   'raw_gradients': exact_comparison(reference_grads, copied_gradients(model))}
            report['comparisons'].append(row)
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
            if not row['loss_exact'] or not row['raw_gradients']['exact']:
                raise AssertionError(f'Observation neutrality failed for {label}')

        check('observed', observed_loss)
        check('unobserved_after_removal', backward())
        report['state_dict_unchanged'] = tensor_digest(model.state_dict()) == report['initial_state_sha256']
        report['observers_removed'] = not any(
            getattr(module, '_recurrent_precision_observer', None) or module._forward_hooks
            or module._forward_pre_hooks or module._backward_hooks for module in model.modules())
        if not report['state_dict_unchanged'] or not report['observers_removed']:
            raise AssertionError('Observer cleanup or model-state preservation failed')
        for layer in range(2):
            phases = {row['phase'] for row in report['observations'] if row['layer'] == layer}
            required = {'forward.projected', 'forward.initial_state', 'forward.running_state',
                        'backward.recomputed_attention', 'backward.buffers', 'backward.attention_adjoint',
                        'helper.block_attention_add.input', 'helper.block_attention_add.output',
                        'helper.recompute_alphas.output', 'helper.recompute_atts.output',
                        'helper.mlp_batched_body.output'}
            if not required <= phases:
                raise AssertionError(f'Missing dtype boundaries in layer {layer}: {sorted(required - phases)}')
        if not any(row['phase'] == 'forward.head' for row in report['observations']):
            raise AssertionError('Output-head dtype was not observed')
        report['compiler'] = compiler_audit(True)
        for path, digest in report['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise AssertionError(f'Diagnostic source changed during execution: {path}')
        report['status'] = 'complete'
        tracker.log({'dtype/neutrality_exact': True, 'dtype/reference_loss': report['reference_loss'],
                     'dtype/gradient_tensors': len(reference_grads),
                     'dtype/boundary_variants': len(report['observations'])})
        tracker.summary({'dtype/passed': True, 'dtype/arm': args.arm,
                         'dtype/state_dict_unchanged': True, 'dtype/observers_removed': True})
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker is not None:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'arm': args.arm,
                      'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
