#!/usr/bin/env python3
"""Same-state RT precision measurements; numerical review flags are not failures.

Runs C (strict FP32), A (BF16 with FP32 recurrent state), then B (legacy BF16)
with one GPU model at a time. CUDA-graph equivalence is validated separately.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_batch_profile import head_chunked_backward, memory, precision_check
from rt_cuda_graph_validate import digest_tensors, validation_config
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit, file_digest

EPS = 2.0 ** -7
GRADIENT_LIMITS = {'global_relative_l2': 2 * EPS, 'tensor_relative_l2': 4 * EPS,
                   'tensor_max_relative_to_reference_max': 8 * EPS}
OPTIMIZER_OPTIONS = dict(lr=1e-3, betas=(.9, .95), eps=1e-8, weight_decay=0.,
                         foreach=False, fused=False)
ARMS = {'C': ('legacy', False), 'A': ('bf16_fp32_state', True), 'B': ('legacy', True)}
CHUNK = 1_048_576


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_packet(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as handle:
        torch.save(value, handle)
    temporary.rename(path)
    return {'path': str(path), 'sha256': file_digest(path), 'bytes': path.stat().st_size}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else (0. if numerator == 0 else None)


def compare_tensors(reference, actual, *, limits=None):
    """Exact name/shape coverage and FP64 reductions without an absolute floor.

Undefined relative errors for nonzero error at zero reference are represented
as null plus a review flag, never omitted or converted into an infinite JSON.
"""
    if not reference or set(reference) != set(actual):
        raise ValueError('Tensor names must be nonempty and match exactly')
    rows = {}
    sums = dict(reference_squared=0., actual_squared=0., error_squared=0., dot=0.)
    for name in sorted(reference):
        left, right = reference[name], actual[name]
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or left.shape != right.shape or left.dtype != right.dtype):
            raise ValueError(f'Missing tensor or shape/dtype mismatch: {name}')
        xall, yall = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
        row = dict(numel=xall.numel(), dtype=str(left.dtype), exact=torch.equal(xall, yall),
                   reference_squared=0., actual_squared=0., error_squared=0., dot=0.,
                   reference_max_abs=0., error_max_abs=0., sign_flips=0,
                   reference_zero_coordinates=0, top_error_coordinates=[])
        for start in range(0, xall.numel(), CHUNK):
            x, y = xall[start:start+CHUNK].double(), yall[start:start+CHUNK].double()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise FloatingPointError(f'Nonfinite tensor: {name}')
            delta = y - x
            row['reference_squared'] += float(torch.dot(x, x))
            row['actual_squared'] += float(torch.dot(y, y))
            row['error_squared'] += float(torch.dot(delta, delta))
            row['dot'] += float(torch.dot(x, y))
            row['reference_max_abs'] = max(row['reference_max_abs'], float(x.abs().max()))
            row['error_max_abs'] = max(row['error_max_abs'], float(delta.abs().max()))
            row['sign_flips'] += int(((x < 0) & (y > 0) | (x > 0) & (y < 0)).sum())
            row['reference_zero_coordinates'] += int((x == 0).sum())
            errors, indices = delta.abs().topk(min(4, len(delta)))
            row['top_error_coordinates'].extend(
                {'flat_index': start + int(i), 'reference': float(x[i]),
                 'actual': float(y[i]), 'error': float(delta[i])}
                for e, i in zip(errors, indices) if float(e) != 0)
            row['top_error_coordinates'] = sorted(row['top_error_coordinates'],
                                                  key=lambda item: abs(item['error']), reverse=True)[:4]
        row['reference_l2'] = math.sqrt(row['reference_squared'])
        row['actual_l2'] = math.sqrt(row['actual_squared'])
        row['error_l2'] = math.sqrt(row['error_squared'])
        row['reference_rms'] = row['reference_l2'] / math.sqrt(max(1, row['numel']))
        row['reference_rms_below_2e-6'] = row['reference_rms'] < 2e-6
        row['relative_l2'] = ratio(row['error_l2'], row['reference_l2'])
        row['error_max_over_reference_max'] = ratio(row['error_max_abs'], row['reference_max_abs'])
        row['error_max_over_reference_rms'] = ratio(row['error_max_abs'], row['reference_rms'])
        denominator = row['reference_l2'] * row['actual_l2']
        row['cosine'] = min(1., max(-1., row['dot'] / denominator)) if denominator else (
            1. if row['error_l2'] == 0 else None)
        row['review_flags'] = []
        if limits:
            if row['error_l2'] > limits['tensor_relative_l2'] * row['reference_l2']:
                row['review_flags'].append('tensor_relative_l2')
            if row['error_max_abs'] > limits['tensor_max_relative_to_reference_max'] * row['reference_max_abs']:
                row['review_flags'].append('tensor_max_relative_to_reference_max')
        for key in sums:
            sums[key] += row[key]
        rows[name] = row
    refnorm, actnorm, errnorm = (math.sqrt(sums[key]) for key in
                                ('reference_squared', 'actual_squared', 'error_squared'))
    cosine = min(1., max(-1., sums['dot'] / (refnorm * actnorm))) if refnorm * actnorm else (
        1. if errnorm == 0 else None)
    flags = []
    if limits and errnorm > limits['global_relative_l2'] * refnorm:
        flags.append('global_relative_l2')
    return {'reduction': 'CPU FP64; every coordinate; no absolute acceptance floor',
            'limits': limits, 'absolute_acceptance_floor': 0.,
            'tensor_count': len(rows), 'numel': sum(row['numel'] for row in rows.values()),
            'reference_l2': refnorm, 'actual_l2': actnorm, 'error_l2': errnorm,
            'relative_l2': ratio(errnorm, refnorm), 'cosine': cosine,
            'exact': all(row['exact'] for row in rows.values()),
            'global_review_flags': flags,
            'tensor_review_flags': {name: row['review_flags'] for name, row in rows.items() if row['review_flags']},
            'requires_review': bool(flags or any(row['review_flags'] for row in rows.values())),
            'tensors': rows}


def adam_deltas(starting_model, updated_model):
    if set(starting_model) != set(updated_model):
        raise ValueError('Parameter names differ in Adam delta')
    return {name: updated_model[name].double() - value.double() for name, value in starting_model.items()}


def near_zero_update_analysis(reference_grads, actual_grads, reference_delta, actual_delta):
    """Historical near-zero bucket is descriptive, not a gradient error floor."""
    names = set(reference_grads)
    if any(set(tree) != names for tree in (actual_grads, reference_delta, actual_delta)):
        raise ValueError('Near-zero analysis requires matching parameter coverage')
    totals = {'inside': {'coordinates': 0, 'sign_flips': 0, 'update_error_squared': 0.},
              'outside': {'coordinates': 0, 'sign_flips': 0, 'update_error_squared': 0.}}
    per_tensor = {}
    for name in sorted(names):
        g, h = reference_grads[name].reshape(-1), actual_grads[name].reshape(-1)
        d, e = reference_delta[name].reshape(-1), actual_delta[name].reshape(-1)
        if not (g.shape == h.shape == d.shape == e.shape):
            raise ValueError(f'Near-zero shape mismatch: {name}')
        squared = sum(float(torch.dot(part.double(), part.double())) for part in g.split(CHUNK))
        threshold = 2 * EPS * math.sqrt(squared / max(g.numel(), 1)) + 2e-6
        row = {'threshold': threshold, 'inside': {'coordinates': 0, 'sign_flips': 0, 'update_error_squared': 0.},
               'outside': {'coordinates': 0, 'sign_flips': 0, 'update_error_squared': 0.}}
        for start in range(0, len(g), CHUNK):
            x, y = g[start:start+CHUNK].double(), h[start:start+CHUNK].double()
            delta = e[start:start+CHUNK].double() - d[start:start+CHUNK].double()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(delta).all()):
                raise FloatingPointError(f'Nonfinite near-zero input: {name}')
            inside = x.abs() <= threshold
            flips = (x < 0) & (y > 0) | (x > 0) & (y < 0)
            for label, mask in (('inside', inside), ('outside', ~inside)):
                values = delta[mask]
                row[label]['coordinates'] += int(mask.sum())
                row[label]['sign_flips'] += int((flips & mask).sum())
                row[label]['update_error_squared'] += float(torch.dot(values, values))
        per_tensor[name] = row
        for label in ('inside', 'outside'):
            for key in totals[label]:
                totals[label][key] += row[label][key]
    energy = sum(row['update_error_squared'] for row in totals.values())
    return {'mask': 'abs(g_reference) <= 2*BF16_epsilon*RMS(g_reference_tensor) + 2e-6',
            'purpose': 'Descriptive first-step sign sensitivity; not an acceptance floor',
            **totals, 'inside_error_energy_fraction': ratio(totals['inside']['update_error_squared'], energy),
            'per_tensor': per_tensor}


def load_ids(path, *, batch, length, vocab, seed):
    if path is None:
        array = np.random.default_rng(seed + 1).integers(0, vocab, size=(batch, length), dtype=np.int64)
    else:
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind not in 'iu':
            raise ValueError('Token array must contain integers')
    if array.ndim != 2 or array.shape[1] != length or array.shape[0] < batch:
        raise ValueError('Token array must contain at least batch rows of the requested length')
    array = array[:batch]
    if array.shape != (batch, length) or array.size == 0 or array.min() < 0 or array.max() >= vocab:
        raise ValueError('Token array must match physical [batch,length] and valid vocabulary IDs')
    return np.ascontiguousarray(array, dtype=np.int64)


def reference_microbatch_backward(model, ids, *, microbatch, head_chunk_size=2):
    """Strict FP32 reference with one global B*T denominator and no early clip.

Each sequence still executes the complete recurrent stack. Partitioning batch
changes FP32 GEMM/reduction order, so this is a separately identified reference.
It does not measure physical-batch throughput or clear graph execution.
"""
    if (ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 2
            or not 1 <= microbatch <= len(ids) or head_chunk_size < 1):
        raise ValueError('Require valid token batch and positive bounded reference/head microbatches')
    if torch.is_autocast_enabled(ids.device.type):
        raise ValueError('FP32 reference must run outside autocast')
    denominator = ids.numel()
    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    for begin in range(0, len(ids), microbatch):
        chunk = ids[begin:begin+microbatch]
        hidden = model(chunk, return_pre_logits=True, return_logits=False).pre_logits
        detached = hidden.detach().requires_grad_(True)
        for head_start in range(0, len(chunk), head_chunk_size):
            head_end = min(head_start + head_chunk_size, len(chunk))
            logits = model.transformer.ff_out(detached[head_start:head_end])
            if model.config.scale_logits:
                logits = logits * (model.config.d_model ** -.5)
            shifted = logits[:, :-1].contiguous().reshape(-1, logits.shape[-1])
            labels = chunk[head_start:head_end, 1:].reshape(-1)
            loss_sum = F.cross_entropy(shifted.float(), labels, reduction='sum')
            total += loss_sum.detach()
            (loss_sum / denominator).backward()
            del loss_sum, logits, shifted
        if detached.grad is None:
            raise AssertionError('Missing reference head-input cotangent')
        hidden.backward(detached.grad)
        del hidden, detached
    return total / denominator


def validate_starting_state(packet, expected_config, parameter_names):
    required = {'model_config', 'model', 'optimizer', 'optimizer_parameter_names'}
    if not required <= set(packet):
        raise ValueError('Checkpoint requires model_config, model, optimizer and optimizer_parameter_names')
    if packet.get('schema', packet.get('format')) not in ('rt-precision-state-v1', 'rt-precision-arm-v1'):
        raise ValueError('Unsupported precision checkpoint schema')
    allowed = {'init_device', 'precision', 'recurrent_precision_policy', 'reference_eager'}
    expected = dataclasses.asdict(expected_config)
    actual = packet['model_config']
    if set(actual) != set(expected) or any(actual[key] != value for key, value in expected.items() if key not in allowed):
        raise ValueError('Checkpoint architecture/semantics differ from the requested tier')
    if packet['optimizer_parameter_names'] != list(parameter_names):
        raise ValueError('Checkpoint canonical optimizer parameter order differs')
    if set(packet['model']) != set(parameter_names):
        raise ValueError('Checkpoint model parameter coverage differs')
    for name, value in packet['model'].items():
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Checkpoint parameters must be finite FP32: {name}')
    groups = packet['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(parameter_names):
        raise ValueError('Only one complete canonical AdamW group is supported')
    group = groups[0]
    if len(set(group['params'])) != len(parameter_names):
        raise ValueError('Duplicate optimizer parameter IDs')
    if (group.get('amsgrad', False) or group.get('maximize', False) or group.get('foreach') is not False
            or group.get('fused') is not False or group.get('capturable', False)
            or group.get('differentiable', False) or group['weight_decay'] != 0):
        raise ValueError('Unsupported AdamW execution settings')
    if not (0 < group['lr'] < math.inf and 0 < group['eps'] < math.inf
            and len(group['betas']) == 2 and all(0 <= b < 1 for b in group['betas'])):
        raise ValueError('Invalid AdamW hyperparameters')
    states = packet['optimizer']['state']
    if states and set(states) != set(group['params']):
        raise ValueError('Nonempty Adam state must cover every parameter')
    steps = []
    for name, index in zip(parameter_names, group['params']):
        if not states:
            break
        state = states[index]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Unsupported or missing Adam state fields')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not math.isfinite(float(step)):
            raise ValueError('Invalid Adam step')
        steps.append(float(step))
        if float(step) < 0 or float(step) != int(step):
            raise ValueError('Invalid Adam step')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            if value.shape != packet['model'][name].shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid Adam moment: {name}/{key}')
        if bool((state['exp_avg_sq'] < 0).any()):
            raise ValueError('Adam second moments must be nonnegative')
    if steps and len(set(steps)) != 1:
        raise ValueError('Adam steps must agree across parameters')
    trained = bool(steps and steps[0] > 0)
    if not trained and states and any(bool((state[key] != 0).any())
                                     for state in states.values() for key in ('exp_avg', 'exp_avg_sq')):
        raise ValueError('Zero-step Adam state must have zero moments')
    return {'trained_moments': trained, 'step': steps[0] if steps else 0., 'group': cpu_tree(group)}


def optimizer_named_tensors(packet):
    group = packet['optimizer']['param_groups'][0]
    return {f'{name}/{key}': packet['optimizer']['state'][index][key]
            for name, index in zip(packet['optimizer_parameter_names'], group['params'])
            for key in ('step', 'exp_avg', 'exp_avg_sq')}


def run_arm(arm, config, starting, ids, output_dir, *, observe=False, reference_microbatch=None):
    from olmo.model import OLMo
    policy, bf16 = ARMS[arm]
    # In-memory specialization caches are independent; the explicit disk cache is
    # retained. Audit before clearing so an earlier failure cannot be erased.
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    raw = dataclasses.asdict(config)
    raw.update(recurrent_precision_policy=policy, init_device='cuda', precision=None, reference_eager=False)
    model = OLMo(type(config)(**raw)).to(dtype=torch.float32).train()
    model.load_state_dict(starting['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
    optimizer.load_state_dict(copy.deepcopy(starting['optimizer']))
    token_ids = torch.from_numpy(ids).to('cuda')
    optimizer.zero_grad(set_to_none=True)
    samples, observed_grads, observed_loss = {}, None, None
    positions = sorted({0, len(ids[0]) // 4, len(ids[0]) // 2, 3 * len(ids[0]) // 4, len(ids[0]) - 1})
    handles = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        def backward():
            if arm == 'C' and reference_microbatch is not None:
                return reference_microbatch_backward(model, token_ids, microbatch=reference_microbatch)
            return head_chunked_backward(model, token_ids, bf16=bf16)
        if observe:
            indices = torch.tensor(positions, device='cuda')
            def observe_output(layer):
                def hook(module, inputs, output):
                    name = f'layer_{layer}'
                    previous = samples.get(name)
                    remaining = 2 - (len(previous) if previous is not None else 0)
                    if remaining > 0:
                        value = output[0].detach()[:remaining].index_select(1, indices).cpu().clone()
                        samples[name] = value if previous is None else torch.cat((previous, value))
                return hook
            handles = [block.register_forward_hook(observe_output(index))
                       for index, block in enumerate(model.transformer.blocks)]
            try:
                observed_loss = float(backward())
                observed_grads = cpu_tree({name: p.grad for name, p in model.named_parameters()})
            finally:
                for handle in handles:
                    handle.remove()
                handles = []
            optimizer.zero_grad(set_to_none=True)
        loss = backward()
        raw_gradients = cpu_tree({name: p.grad for name, p in model.named_parameters()})
        if any(value is None for value in raw_gradients.values()):
            raise AssertionError('Missing canonical parameter gradient')
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in raw_gradients.values()):
            raise FloatingPointError('Raw gradients must be finite FP32')
        observation = {'enabled': observe, 'positions': positions, 'examples': min(2, len(ids))}
        if observe:
            observation['unobserved_loss_bitwise'] = observed_loss == float(loss)
            observation['unobserved_gradients_bitwise'] = all(torch.equal(observed_grads[name], value)
                                                              for name, value in raw_gradients.items())
            if not all(observation[key] for key in ('unobserved_loss_bitwise', 'unobserved_gradients_bitwise')):
                save_packet(output_dir / f'{arm}-observation-mismatch.pt',
                            {'observed_loss': observed_loss, 'loss': loss.detach().cpu(),
                             'observed_gradients': observed_grads, 'unobserved_gradients': raw_gradients,
                             'samples': samples})
                raise AssertionError('Layer observations changed accepted gradients or loss')
            del observed_grads
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
        coefficient = (torch.ones_like(norm) / (norm + 1e-6)).clamp(max=1.)
        clipped = cpu_tree({name: p.grad for name, p in model.named_parameters()})
        optimizer.step()
        torch.cuda.synchronize()
        precision = precision_check(model, optimizer)
        initial_states = starting['optimizer']['state']
        expected_step = 1 + (float(next(iter(initial_states.values()))['step']) if initial_states else 0)
        if any(float(state['step']) != expected_step for state in optimizer.state.values()):
            raise AssertionError('Adam step did not advance exactly once for every parameter')
        packet = {'schema': 'rt-precision-arm-v1', 'arm': arm, 'bf16_autocast': bf16,
                  'model_config': dataclasses.asdict(model.config),
                  'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                  'optimizer_parameter_names': list(dict(model.named_parameters())),
                  'loss': loss.detach().cpu(), 'raw_gradients': raw_gradients, 'clipped_gradients': clipped,
                  'gradient_norm': norm.detach().cpu(), 'clip_coefficient': coefficient.detach().cpu(),
                  'layer_output_samples': samples, 'observation': observation,
                  'starting_state_sha256': file_digest(output_dir / 'starting-state.pt')}
        record = {'arm': arm, 'policy': policy, 'bf16_autocast': bf16, 'cuda_graphs': False,
                  'physical_backbone_batch': reference_microbatch if arm == 'C' and reference_microbatch else len(ids),
                  'global_batch': len(ids),
                  'optimizer_step_advanced_exactly': True,
                  'reference_microbatch_reduction_order_changed': arm == 'C' and reference_microbatch is not None and reference_microbatch < len(ids),
                  'loss': float(loss), 'gradient_norm': float(norm), 'clip_coefficient': float(coefficient),
                  'precision': precision, 'observation': observation, 'memory': memory(),
                  'seconds_including_cpu_copies': time.monotonic() - started, 'compiler': compiler_audit(True)}
        if observe:
            record['layer_output_summary'] = {
                name: {'dtype': str(value.dtype), 'shape': list(value.shape), 'positions': {
                    str(position): {'rms': float(value[:, index].double().square().mean().sqrt()),
                                    'max_abs': float(value[:, index].abs().max())}
                    for index, position in enumerate(positions)}} for name, value in samples.items()}
        record['packet'] = save_packet(output_dir / f'{arm}-packet.pt', packet)
        return record
    finally:
        for handle in handles:
            handle.remove()
        del model, optimizer, token_ids
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def compare_arm_packets(starting, reference, actual, *, trained):
    reference_delta = adam_deltas(starting['model'], reference['model'])
    actual_delta = adam_deltas(starting['model'], actual['model'])
    gradients = compare_tensors(reference['raw_gradients'], actual['raw_gradients'], limits=GRADIENT_LIMITS)
    updates = compare_tensors(reference_delta, actual_delta)
    updates['applicable_review_screen'] = ('relative_l2 <= 0.015625' if trained else 'cosine >= 0.99')
    updates['requires_review'] = ((updates['relative_l2'] is None or updates['relative_l2'] > 2 * EPS)
                                 if trained else (updates['cosine'] is None or updates['cosine'] < .99))
    gap = float(actual['loss']) - float(reference['loss'])
    reference_adam, actual_adam = optimizer_named_tensors(reference), optimizer_named_tensors(actual)
    steps = compare_tensors({key: value for key, value in reference_adam.items() if key.endswith('/step')},
                            {key: value for key, value in actual_adam.items() if key.endswith('/step')})
    if not steps['exact']:
        raise AssertionError('Compared Adam states have different step counts')
    updates['max_abs_error_over_learning_rate'] = max(row['error_max_abs'] for row in updates['tensors'].values()) / starting['optimizer']['param_groups'][0]['lr']
    result = {'reference': reference['arm'], 'candidate': actual['arm'],
              'comparison_role': ('Accuracy against strict FP32' if reference['arm'] == 'C'
                                  else 'Pairwise engineering comparison; mixed-policy reference is not an FP32 oracle'),
              'raw_gradients': gradients,
              'clipped_gradients': compare_tensors(reference['clipped_gradients'], actual['clipped_gradients']),
              'adam_delta': updates,
              'adam_moments': compare_tensors({key: value for key, value in reference_adam.items() if not key.endswith('/step')},
                                              {key: value for key, value in actual_adam.items() if not key.endswith('/step')}),
              'adam_steps': steps,
              'loss_gap_nats': gap, 'loss_review_limit_nats': .01, 'loss_requires_review': abs(gap) > .01,
              'clip_coefficient_gap': float(actual['clip_coefficient']) - float(reference['clip_coefficient']),
              'near_zero': near_zero_update_analysis(reference['raw_gradients'], actual['raw_gradients'],
                                                      reference_delta, actual_delta)}
    if reference['layer_output_samples'] or actual['layer_output_samples']:
        result['layer_output_samples'] = compare_tensors(reference['layer_output_samples'], actual['layer_output_samples'])
        def position_samples(packet):
            return {f'{name}/position_{position}': value[:, index] for name, value in packet['layer_output_samples'].items()
                    for index, position in enumerate(packet['observation']['positions'])}
        result['layer_and_position_samples'] = compare_tensors(position_samples(reference), position_samples(actual))
    result['requires_review'] = gradients['requires_review'] or updates['requires_review'] or result['loss_requires_review']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('tiny', 'full'), required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ids-path', type=Path)
    parser.add_argument('--reference-microbatch', type=int,
                        help='Explicitly partition only C; preserve global B*T normalization and clip/Adam once')
    parser.add_argument('--observe-layer-outputs', action='store_true')
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if not 1 <= args.batch <= 1024 or args.seed < 0:
        parser.error('Require physical batch 1..1024 and nonnegative seed')
    if args.reference_microbatch is not None and not 1 <= args.reference_microbatch <= args.batch:
        parser.error('Reference microbatch must be between1 and the global batch')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh comparison directory')
    args.output_dir.mkdir(parents=True)
    tracker = None
    report = {'schema': 'rt-precision-comparison-v1', 'status': 'running', 'seed': args.seed,
              'tier': args.tier, 'batch': args.batch, 'arms': {}, 'comparisons': {},
              'scope': 'Compiled, uncaptured same-state actual next-token CE, gradients and one Adam update',
              'loss': '511 shifted targets at T512; summed CE divided by B*T; head chunks2',
              'gradient_review_limits': GRADIENT_LIMITS, 'absolute_acceptance_floor': 0.,
              'review_is_not_execution_failure': True, 'cuda_graphs': False}
    report['reference_microbatch'] = args.reference_microbatch
    report['reference_scope'] = ('Explicit FP32 batch partition; one global loss denominator; FP32 reduction order can change'
                                 if args.reference_microbatch else 'Full physical-batch FP32 reference')
    started = time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        seed_all(args.seed, deterministic=True)
        config = validation_config(args.tier)
        report['model_config'] = dataclasses.asdict(config)
        report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                             'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        paths = [Path(__file__), Path('scripts/rt_cuda_graph_validate.py'), Path('scripts/rt_cuda_graph.py'),
                 Path('scripts/rt_batch_profile.py'), Path('scripts/stage_a_common.py'),
                 Path('scripts/stage_b_train.py'), Path('scripts/experiment_tracking.py'),
                 Path('configs/stage_a/full.json')]
        paths += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
        report['source_sha256'] = {}
        for path in paths:
            relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
            content = path.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(content).hexdigest()
            destination = args.output_dir / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        criteria = {'schema': 'rt-precision-review-criteria-v1', 'gradient_limits': GRADIENT_LIMITS,
                    'absolute_acceptance_floor': 0., 'mean_ce_absolute_gap_nats': .01,
                    'initial_adam_cosine_minimum': .99, 'trained_adam_delta_relative_l2_maximum': 2 * EPS,
                    'near_zero_bucket': 'abs(g_FP32) <= 2*BF16_epsilon*RMS(g_FP32_tensor)+2e-6; descriptive only',
                    'disposition': 'Prospective engineering review flags, not paper criteria or hard numerical failures',
                    'hard_failures': ['nonfinite or missing gradients/state', 'wrong parameter ownership',
                                      'observation changes execution', 'incorrect Adam step advancement',
                                      'compiler fallback', 'source mutation'],
                    'comparisons': ['A_vs_C', 'B_vs_C', 'B_vs_A'],
                    'pairwise_comparison': 'B_vs_A uses the same review screens with protected BF16 as reference; not an FP32 accuracy oracle',
                    'source_sha256': report['source_sha256']}
        atomic_json(args.output_dir / 'criteria.json', criteria)
        report['criteria'] = {'path': str(args.output_dir / 'criteria.json'),
                              'sha256': file_digest(args.output_dir / 'criteria.json')}
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb')})
        seed_all(args.seed, deterministic=True)
        from olmo.model import OLMo
        model = OLMo(config).to(dtype=torch.float32).train()
        names = list(dict(model.named_parameters()))
        if args.checkpoint:
            starting = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            report['checkpoint'] = {'path': str(args.checkpoint), 'sha256': file_digest(args.checkpoint)}
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
            starting = {'schema': 'rt-precision-state-v1', 'model_config': dataclasses.asdict(config),
                        'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                        'optimizer_parameter_names': names, 'seed': args.seed}
            del optimizer
        report['starting_optimizer'] = validate_starting_state(starting, config, names)
        # Strict load validates actual tensor shapes as well as canonical names.
        model.load_state_dict(starting['model'], strict=True)
        report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        report['starting_model_digest'] = digest_tensors(starting['model'])
        report['starting_state'] = save_packet(args.output_dir / 'starting-state.pt', starting)
        del model
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        ids = load_ids(args.ids_path, batch=args.batch, length=config.max_sequence_length,
                       vocab=config.vocab_size, seed=args.seed)
        with (args.output_dir / 'ids.npy').open('xb') as handle:
            np.save(handle, ids, allow_pickle=False)
        report['ids'] = {'path': str(args.output_dir / 'ids.npy'), 'sha256': file_digest(args.output_dir / 'ids.npy'),
                         'shape': list(ids.shape), 'source': str(args.ids_path) if args.ids_path else 'random generated tokens',
                         'selection': 'First batch rows of supplied array' if args.ids_path else 'Fresh seeded array',
                         'source_sha256': file_digest(args.ids_path) if args.ids_path else None}
        for arm in ('C', 'A', 'B'):
            seed_all(args.seed + 2, deterministic=True)
            print(json.dumps({'arm': arm, 'status': 'starting'}), flush=True)
            report['arms'][arm] = run_arm(arm, config, starting, ids, args.output_dir,
                                         observe=args.observe_layer_outputs, reference_microbatch=args.reference_microbatch)
            tracker.log({'update': len(report['arms']), f'precision/{arm}/loss': report['arms'][arm]['loss']})
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
        reference = torch.load(args.output_dir / 'C-packet.pt', map_location='cpu', weights_only=False)
        for arm in ('A', 'B'):
            actual = torch.load(args.output_dir / f'{arm}-packet.pt', map_location='cpu', weights_only=False)
            result = compare_arm_packets(starting, reference, actual,
                                         trained=report['starting_optimizer']['trained_moments'])
            report['comparisons'][f'{arm}_vs_C'] = result
            tracker.summary({f'precision/{arm}/gradient_relative_l2': result['raw_gradients']['relative_l2'],
                             f'precision/{arm}/adam_delta_relative_l2': result['adam_delta']['relative_l2'],
                             f'precision/{arm}/adam_delta_cosine': result['adam_delta']['cosine'],
                             f'precision/{arm}/loss_gap_nats': result['loss_gap_nats'],
                             f'precision/{arm}/requires_review': result['requires_review']})
            del actual
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
        del reference
        protected = torch.load(args.output_dir / 'A-packet.pt', map_location='cpu', weights_only=False)
        legacy = torch.load(args.output_dir / 'B-packet.pt', map_location='cpu', weights_only=False)
        pairwise = compare_arm_packets(starting, protected, legacy,
                                       trained=report['starting_optimizer']['trained_moments'])
        report['comparisons']['B_vs_A'] = pairwise
        tracker.summary({'precision/B_vs_A/gradient_relative_l2': pairwise['raw_gradients']['relative_l2'],
                         'precision/B_vs_A/adam_delta_relative_l2': pairwise['adam_delta']['relative_l2'],
                         'precision/B_vs_A/adam_delta_cosine': pairwise['adam_delta']['cosine'],
                         'precision/B_vs_A/loss_gap_nats': pairwise['loss_gap_nats'],
                         'precision/B_vs_A/requires_review': pairwise['requires_review']})
        del protected, legacy
        atomic_json(args.output_dir / 'progress.json', report, replace=True)
        for path, digest in report['source_sha256'].items():
            if file_digest(Path(path)) != digest:
                raise RuntimeError(f'Comparison source changed: {path}')
        report['status'] = 'complete'
        report['requires_review'] = any(value['requires_review'] for value in report['comparisons'].values())
        report['disposition'] = 'review_required' if report['requires_review'] else 'within_prospective_review_thresholds'
    except torch.OutOfMemoryError as error:
        report.update(status='out_of_memory', error_type=type(error).__name__, error=str(error), oom_memory=memory())
        raise
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'requires_review': report['requires_review'],
                      'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
