#!/usr/bin/env python3
"""Conditional attention VJP diagnosis on a retained legacy RT B2 fixture.

The incoming attention cotangent and rounded Q/K/V operands are frozen from
the actual custom backward. FP32/FP64 references differentiate independent
causal attention, not the whole recurrent model or its parameter gradients.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import head_chunked_backward
from rt_cuda_graph_validate import digest_tensors, validation_config
from rt_precision_dtype import copied_gradients, exact_comparison
from rt_precision_train import cpu_tree, save_checkpoint
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit, file_digest

OPERANDS = ('q', 'k_init', 'v_init', 'final_k', 'final_v')
ADJOINTS = ('q_grad', 'k_init_grad', 'v_init_grad', 'permanent_k_grad', 'permanent_v_grad')


def causal_attention(q, k_init, v_init, final_k, final_v, bias=None):
    """Independent [L,B,H,D] attention: provisional self, permanent earlier KV.

    q is already scaled. Masking out diagonal permanent values directly avoids
    reconstructing self values by subtracting/adding permanent values.
    """
    if not all(value.shape == q.shape for value in (k_init, v_init, final_k, final_v)) or q.ndim != 4:
        raise ValueError('Require equal [L,B,H,D] operands')
    length = q.shape[0]
    q_math, k_math = q.permute(1, 2, 0, 3), final_k.permute(1, 2, 0, 3)
    scores = q_math @ k_math.transpose(-2, -1)
    diagonal = torch.eye(length, device=q.device, dtype=torch.bool)
    self_scores = (q * k_init).sum(-1).permute(1, 2, 0)
    scores = torch.where(diagonal, self_scores.unsqueeze(-1), scores)
    if bias is not None:
        scores = scores + bias
    future = torch.ones((length, length), device=q.device, dtype=torch.bool).triu(1)
    probabilities = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
    earlier = probabilities.masked_fill(diagonal, 0.)
    attention = earlier @ final_v.permute(1, 2, 0, 3)
    attention = attention + probabilities.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * v_init.permute(1, 2, 0, 3)
    return attention.permute(2, 0, 1, 3), probabilities.transpose(-2, -1)


def conditional_vjp(packet, dtype, device):
    with torch.enable_grad(), torch.autocast(torch.device(device).type, enabled=False):
        values = [packet['operands'][name].to(device=device, dtype=dtype).detach().requires_grad_(True)
                  for name in OPERANDS]
        bias = packet['operands']['attention_bias']
        bias = None if bias is None else bias.to(device=device, dtype=dtype)
        attention, alphas = causal_attention(*values, bias)
        g = packet['g'].to(device=device, dtype=dtype)
        gradients = torch.autograd.grad(attention, values, grad_outputs=g)
    return {'attention': attention.detach().cpu(), 'alphas': alphas.detach().cpu(),
            'adjoints': {name: value.detach().cpu() for name, value in zip(ADJOINTS, gradients)}}


def error_metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError('Compared tensor shapes differ')
    x, y = reference.detach().cpu().reshape(-1).double(), actual.detach().cpu().reshape(-1).double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError('Nonfinite attention reference or observed tensor')
    delta = y - x
    nx, ny, nd = x.norm().item(), y.norm().item(), delta.norm().item()
    return {'reference_dtype': str(reference.dtype), 'actual_dtype': str(actual.dtype),
            'numel': x.numel(), 'reference_l2': nx, 'actual_l2': ny,
            'error_l2': nd, 'relative_l2': nd / nx if nx else (0. if not nd else None),
            'reference_max_abs': x.abs().max().item(), 'max_abs_error': delta.abs().max().item(),
            'cosine': torch.dot(x, y).item() / (nx * ny) if nx and ny else (1. if not nd else None),
            'nonzero_actual_at_zero_reference': int(((x == 0) & (y != 0)).sum()),
            'reduction': 'CPU FP64, every coordinate; exact zero references remain explicit'}


class CapturedAttention:
    def __init__(self, layer):
        self.layer = layer
        self.operands = {}
        self.per_token = {name: {} for name in ('g', 'permanent_k_grad', 'permanent_v_grad', 'forward_attention')}
        self.actual = {}
        self.calls = {}

    def observe(self, phase, tensors, token_index=None):
        if phase == 'backward.attention_adjoint':
            for destination, name in (('g', 'g'), ('permanent_k_grad', 'k_grad'), ('permanent_v_grad', 'v_grad')):
                if token_index in self.per_token[destination]:
                    raise AssertionError('Repeated token in attention-adjoint observer')
                self.per_token[destination][token_index] = tensors[name].detach().cpu().clone()
        elif phase == 'backward.pre_attention_adjoint':
            if self.actual:
                raise AssertionError('Repeated pre-attention adjoint observation')
            self.actual = {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
        elif phase == 'forward.mlp_input':
            # Convert only the layout, preserving its exact BF16 operand values.
            value = tensors['attention']
            batch = value.shape[0]
            self.per_token['forward_attention'][token_index] = value.detach().reshape(
                batch, 1, self.heads, self.head_dim).permute(1, 0, 2, 3).cpu().clone()

    @contextmanager
    def installed(self, model):
        import olmo.model as module
        block = model.transformer.blocks[self.layer]
        if getattr(block, '_recurrent_precision_observer', None) is not None:
            raise AssertionError('Existing precision observer would mix diagnostic state')
        existed = hasattr(block, '_recurrent_precision_observer')
        previous_observer = getattr(block, '_recurrent_precision_observer', None)
        original_helper = module.recurrent_helper
        self.heads, self.head_dim = block.config.n_heads, block.config.d_model // block.config.n_heads

        def helper(owner, function, *args):
            name = getattr(function, '_torchdynamo_orig_callable', function).__name__
            selected = owner is block and name in ('recompute_alphas', 'recompute_atts')
            if selected:
                self.calls[name] = self.calls.get(name, 0) + 1
                names = ('final_k', 'k_init', 'q', 'attention_bias') if name == 'recompute_alphas' else ('final_v', 'v_init')
                for key, value in zip(names, args):
                    self.operands[key] = None if value is None else value.detach().cpu().clone()
            result = original_helper(owner, function, *args)
            if selected:
                # The custom backward mutates alphas later. Take its own copy now.
                key = 'alphas' if name == 'recompute_alphas' else 'attention'
                self.operands['observed_' + key] = result.detach().cpu().clone()
            return result

        try:
            object.__setattr__(block, '_recurrent_precision_observer', self.observe)
            module.recurrent_helper = helper
            yield
        finally:
            module.recurrent_helper = original_helper
            if existed:
                object.__setattr__(block, '_recurrent_precision_observer', previous_observer)
            else:
                object.__delattr__(block, '_recurrent_precision_observer')

    def packet(self):
        if self.calls != {'recompute_alphas': 1, 'recompute_atts': 1}:
            raise AssertionError('Require one complete attention reconstruction for the selected block')
        length = self.operands['q'].shape[0]
        rows = {}
        for name, values in self.per_token.items():
            if set(values) != set(range(length)):
                raise AssertionError(f'Incomplete token coverage: {name}')
            rows[name] = torch.cat([values[index] for index in range(length)], dim=0)
        if set(self.actual) != {'q_grad', 'k_init_grad', 'v_init_grad'}:
            raise AssertionError('Incomplete pre-attention adjoints')
        observed = {name: self.operands.pop('observed_' + name) for name in ('attention', 'alphas')}
        result = {'layer': self.layer, 'operands': self.operands, 'g': rows['g'],
                  'forward_attention': rows['forward_attention'], 'observed_reconstruction': observed,
                  'actual_adjoints': dict(self.actual, **{name: rows[name] for name in ('permanent_k_grad', 'permanent_v_grad')})}
        for name in OPERANDS:
            if result['operands'][name].shape != result['g'].shape:
                raise AssertionError(f'Captured operand layout mismatch: {name}')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=9, help='Zero-based recurrent block index')
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh attention-probe directory')
    if not 0 <= args.layer < 12 or not args.wandb_project:
        parser.error('Require a block index in 0..11 and online W&B')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-attention-probe-v1', 'status': 'running', 'layer': args.layer,
              'policy': 'legacy', 'shape': [2, 512], 'head_chunk_size': 2, 'optimizer_updates': 0,
              'cuda_graphs': False, 'comparisons': {},
              'scope': 'Conditional attention VJP on identical rounded operands and frozen actual g; not a whole-model FP64 reference',
              'qualification': 'g includes the actual legacy recurrence credit. The probe does not differentiate through g or reconstruct an alternate-precision recurrent trajectory.'}
    tracker = None
    started = time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.utils.counters.clear()
        case_path = args.case_dir / 'report.json'
        case = json.loads(case_path.read_text())
        if (case['status'] != 'complete' or case['tier'] != 'full' or case['batch'] != 2
                or case['starting_optimizer']['step'] != 0 or case['reference_microbatch'] is not None):
            raise ValueError('Require the completed full physical B2 initialization comparison')
        report['case_report'] = {'path': str(case_path), 'sha256': file_digest(case_path)}
        for name, expected in case['source_sha256'].items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f'Case source differs from current source: {name}')
        source_paths = [Path(__file__), Path('scripts/rt_precision_dtype.py'), Path('scripts/rt_precision_train.py')]
        source_paths += [Path(name) for name in case['source_sha256']]
        report['source_sha256'] = {}
        for source in source_paths:
            relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
            data = source.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
            target = args.output_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        report['seed'] = case['seed']
        config = validation_config('full', 'legacy')
        for name, value in dataclasses.asdict(config).items():
            if name != 'recurrent_precision_policy' and case['model_config'].get(name) != value:
                raise ValueError(f'Case model configuration differs: {name}')
        report['model_config'] = dataclasses.asdict(config)
        report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                             'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        if report['runtime'] != case['runtime']:
            raise ValueError('Probe must retain the case runtime and compiler cache')
        for key in ('starting_state', 'ids'):
            path = Path(case[key]['path'])
            if file_digest(path) != case[key]['sha256']:
                raise ValueError(f'Retained {key} hash differs')
            report[key] = case[key]
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb')})
        seed_all(case['seed'], deterministic=True)
        starting = torch.load(Path(case['starting_state']['path']), map_location='cpu', weights_only=False)
        from olmo.model import OLMo
        model = OLMo(config).to(device='cuda', dtype=torch.float32).train()
        model.load_state_dict(starting['model'], strict=True)
        if digest_tensors(model.state_dict()) != case['starting_model_digest']:
            raise AssertionError('Probe initialization differs from the retained case')
        report['starting_model_digest'] = case['starting_model_digest']
        del starting
        array = np.load(Path(case['ids']['path']), allow_pickle=False)
        if array.shape != (2, 512):
            raise ValueError('Probe IDs must have shape B2/T512')
        ids = torch.from_numpy(array).to(device='cuda', dtype=torch.long)
        observer = CapturedAttention(args.layer)
        with observer.installed(model):
            observed_loss = head_chunked_backward(model, ids, bf16=True).item()
        observed_grads = copied_gradients(model)
        packet = observer.packet()
        model.zero_grad(set_to_none=True)
        unobserved_loss = head_chunked_backward(model, ids, bf16=True).item()
        actual_grads = copied_gradients(model)
        report['neutrality'] = {'observed_loss': observed_loss, 'unobserved_loss': unobserved_loss,
                                'loss_exact': observed_loss == unobserved_loss,
                                'raw_gradients': exact_comparison(observed_grads, actual_grads)}
        if not report['neutrality']['loss_exact'] or not report['neutrality']['raw_gradients']['exact']:
            raise AssertionError('Attention observation changed legacy loss or parameter gradients')
        del observed_grads
        retained_path = Path(case['arms']['B']['packet']['path'])
        if file_digest(retained_path) != case['arms']['B']['packet']['sha256']:
            raise ValueError('Retained B packet hash differs')
        retained = torch.load(retained_path, map_location='cpu', weights_only=False)
        report['retained_B'] = {'packet': case['arms']['B']['packet'],
                                'loss_exact': retained['loss'].item() == unobserved_loss,
                                'raw_gradients': exact_comparison(retained['raw_gradients'], actual_grads)}
        if not report['retained_B']['loss_exact'] or not report['retained_B']['raw_gradients']['exact']:
            raise AssertionError('Unobserved probe does not reproduce the retained legacy B packet')
        del retained, actual_grads
        report['model_state_unchanged'] = digest_tensors(model.state_dict()) == case['starting_model_digest']
        report['observers_removed'] = not any(getattr(module, '_recurrent_precision_observer', None)
                                              for module in model.modules())
        if not report['model_state_unchanged'] or not report['observers_removed']:
            raise AssertionError('Probe changed weights or left observers installed')
        report['compiler'] = compiler_audit(True)
        del model
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        references = {name: conditional_vjp(packet, dtype, 'cuda')
                      for name, dtype in [('fp64', torch.float64), ('fp32', torch.float32)]}
        for name in ADJOINTS:
            reference = references['fp64']['adjoints'][name]
            report['comparisons'][name] = {
                'legacy_vs_fp64': error_metrics(reference, packet['actual_adjoints'][name]),
                'fp32_math_vs_fp64': error_metrics(reference, references['fp32']['adjoints'][name]),
                'first_query_or_record': error_metrics(reference[0], packet['actual_adjoints'][name][0]),
                'last_query_or_record': error_metrics(reference[-1], packet['actual_adjoints'][name][-1]),
            }
        report['reconstruction'] = {
            name: {'legacy_vs_fp64': error_metrics(references['fp64'][name], packet['observed_reconstruction'][name]),
                   'fp32_math_vs_fp64': error_metrics(references['fp64'][name], references['fp32'][name])}
            for name in ('attention', 'alphas')}
        report['reconstruction']['forward_vs_backward_attention'] = error_metrics(
            packet['forward_attention'], packet['observed_reconstruction']['attention'])
        report['reconstruction']['first_forward_vs_backward_attention'] = error_metrics(
            packet['forward_attention'][0], packet['observed_reconstruction']['attention'][0])
        report['first_query_reference'] = {
            'attention_equals_provisional_v': torch.equal(references['fp64']['attention'][0], packet['operands']['v_init'][0].double()),
            'q_gradient_exactly_zero': bool((references['fp64']['adjoints']['q_grad'][0] == 0).all()),
            'k_init_gradient_exactly_zero': bool((references['fp64']['adjoints']['k_init_grad'][0] == 0).all()),
            'v_init_gradient_equals_g': torch.equal(references['fp64']['adjoints']['v_init_grad'][0], packet['g'][0].double()),
            'last_permanent_k_gradient_exactly_zero': bool((references['fp64']['adjoints']['permanent_k_grad'][-1] == 0).all()),
            'last_permanent_v_gradient_exactly_zero': bool((references['fp64']['adjoints']['permanent_v_grad'][-1] == 0).all()),
        }
        if not all(report['first_query_reference'].values()):
            raise AssertionError('Conditional reference violates causal/self-only identities')
        packet.update(schema='rt-precision-attention-packet-v1', references=references,
                      completed_updates=0, next_data_row=0, source_sha256=report['source_sha256'],
                      case_report=report['case_report'], seed=case['seed'])
        report['packet'] = save_checkpoint(args.output_dir / 'attention-packet.pt', cpu_tree(packet))
        for name, expected in report['source_sha256'].items():
            if file_digest(Path(name)) != expected:
                raise AssertionError(f'Probe source changed: {name}')
        report['status'] = 'complete'
        tracker.log({f'attention/{name}/legacy_relative_l2': row['legacy_vs_fp64']['relative_l2']
                     for name, row in report['comparisons'].items()})
        tracker.summary({'attention/probe_complete': True, 'attention/observer_neutrality_exact': True,
                         'attention/retained_B_reproduced_exactly': True})
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
    print(json.dumps({'status': report['status'], 'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
