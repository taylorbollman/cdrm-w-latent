#!/usr/bin/env python3
"""Bounded terminal-output credit and frozen-forward scale sanity for tiled RT.

This is a synthetic vector-Jacobian probe, not a task loss or optimizer test.
The untied vocabulary head is deliberately unused by the pre-logit objective.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_cuda_graph_validate import digest_tensors, validation_config
from rt_precision_compare import ARMS, GRADIENT_LIMITS, compare_tensors, cpu_tree, load_ids, save_packet
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, compiler_audit, file_digest

SCALES = (1., 1. / 32., 32.)
SCALE_LIMITS = {'global_relative_l2': 1e-5, 'tensor_relative_l2': 3e-5,
                'tensor_max_relative_to_reference_max': 1e-4}
UNUSED_PARAMETERS = {'transformer.ff_out.weight'}


def snapshot_sources(output_dir, extra=()):
    paths = [Path(__file__), Path('scripts/rt_precision_compare.py'),
             Path('scripts/rt_cuda_graph_validate.py'), Path('scripts/rt_cuda_graph.py'),
             Path('scripts/rt_batch_profile.py'), Path('scripts/stage_a_common.py'),
             Path('scripts/stage_b_train.py'), Path('scripts/experiment_tracking.py'),
             Path('configs/stage_a/full.json'), *map(Path, extra)]
    paths += sorted(Path('recurrent-transformer/olmo').rglob('*.py'))
    result = {}
    for path in paths:
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        value = path.read_bytes()
        result[str(relative)] = hashlib.sha256(value).hexdigest()
        destination = output_dir / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
    return result


def verify_sources(sources):
    for path, expected in sources.items():
        if file_digest(Path(path)) != expected:
            raise RuntimeError(f'Probe source changed: {path}')


def terminal_cotangent(shape, *, seed):
    if len(shape) != 3 or any(size < 1 for size in shape) or shape[1] < 2:
        raise ValueError('Require nonempty [batch, sequence>=2, width] pre-logits')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    result = torch.zeros(shape, dtype=torch.float32)
    result[:, -1] = torch.randn((shape[0], shape[2]), generator=generator)
    return result


def backbone_gradients(model, *, unused=UNUSED_PARAMETERS):
    parameters = dict(model.named_parameters())
    if not unused <= parameters.keys():
        raise AssertionError('Declared unused head does not match parameter ownership')
    result = {}
    for name, parameter in parameters.items():
        if parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f'Parameter must remain finite FP32: {name}')
        if name in unused:
            if parameter.grad is not None:
                raise AssertionError(f'Pre-logit objective unexpectedly used head: {name}')
            continue
        if parameter.grad is None:
            raise AssertionError(f'Missing backbone gradient: {name}')
        if parameter.grad.dtype != torch.float32 or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f'Backbone gradient must be finite FP32: {name}')
        result[name] = parameter.grad.detach().cpu().clone()
    return result


def write_observer(layer, records):
    def observe(phase, tensors, token_index=None):
        # The generic buffers phase contains uninitialized gs/g_dot_atts: never
        # inspect it. Here k/v buffers contain all future-consumer contributions.
        if phase != 'backward.attention_adjoint':
            return
        if not isinstance(token_index, int) or token_index < 0:
            raise AssertionError('Missing backward token index')
        for kind in ('k', 'v'):
            name = f'layer_{layer}/token_{token_index}/{kind}'
            if name in records:
                raise AssertionError(f'Duplicate write adjoint: {name}')
            value = tensors[f'{kind}_grad']
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Write adjoint must be finite FP32: {name}')
            records[name] = value.detach().cpu().clone()
    return observe


def credit_summary(records, *, layers, length):
    expected = {f'layer_{layer}/token_{token}/{kind}' for layer in range(layers)
                for token in range(length) for kind in ('k', 'v')}
    if set(records) != expected:
        raise AssertionError('Write-adjoint coverage differs from every layer/token/K/V')
    rows, missing_credit, terminal_credit = {}, [], []
    for layer in range(layers):
        for token in range(length):
            name = f'layer_{layer}/token_{token}'
            norms = {}
            for kind in ('k', 'v'):
                value = records[f'{name}/{kind}']
                if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'Nonfinite/non-FP32 write credit: {name}/{kind}')
                norms[f'{kind}_l2'] = float(value.double().norm())
            norms['combined_l2'] = (norms['k_l2'] ** 2 + norms['v_l2'] ** 2) ** .5
            rows[name] = norms
            if token < length - 1 and norms['combined_l2'] == 0:
                missing_credit.append(name)
            if token == length - 1 and norms['combined_l2'] != 0:
                terminal_credit.append(name)
    return {'pass': not missing_credit and not terminal_credit,
            'earlier_write_nonzero_failures': missing_credit,
            'terminal_write_zero_failures': terminal_credit, 'writes': rows,
            'scope': 'Nonzero combined K/V credit for every earlier write; terminal write has no later same-layer consumer'}


def normalized_tensors(values, scale):
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite cotangent scale required')
    return {name: value / scale for name, value in values.items()}


def run_credit_arm(arm, config, initial_state, ids, cotangent, output_dir):
    from olmo.model import OLMo
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    policy, bf16 = ARMS[arm]
    raw = dataclasses.asdict(config)
    raw['recurrent_precision_policy'] = policy
    model = OLMo(type(config)(**raw)).to(dtype=torch.float32).train()
    model.load_state_dict(initial_state, strict=True)
    tokens = torch.from_numpy(ids).to('cuda')
    incoming = cotangent.to('cuda')
    expected_input = cpu_tree(cotangent)
    records, gradients, credit = {}, {}, {}
    blocks = list(model.transformer.blocks)
    started = time.monotonic()
    try:
        def forward():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                return model(tokens, return_pre_logits=True, return_logits=False).pre_logits
        # One graph and identical saved forward operands for all three VJPs.
        output = forward()
        frozen_output = output.detach().cpu().clone()
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Pre-logits must be finite FP32')
        for scale in SCALES:
            model.zero_grad(set_to_none=True)
            observed = {}
            for layer, block in enumerate(blocks):
                block._recurrent_precision_observer = write_observer(layer, observed)
            output.backward(incoming * scale, retain_graph=True)
            if not torch.equal(output.detach().cpu(), frozen_output):
                raise AssertionError('A backward pass mutated the frozen forward output')
            key = str(scale)
            gradients[key] = backbone_gradients(model)
            credit[key] = observed
            records[key] = {'forward_bitwise_unchanged': True,
                            'credit': credit_summary(observed, layers=config.n_layers,
                                                     length=config.max_sequence_length)}
            if not records[key]['credit']['pass']:
                save_packet(output_dir / f'{arm}-credit-failure.pt',
                            {'scale': scale, 'write_adjoints': observed, 'gradients': gradients[key]})
                raise AssertionError('Earlier-write or terminal-write credit contract failed')
        for block in blocks:
            del block._recurrent_precision_observer
        del output
        model.zero_grad(set_to_none=True)
        unobserved = forward()
        if not torch.equal(unobserved.detach().cpu(), frozen_output):
            raise AssertionError('Observer-free replay changed pre-logits')
        unobserved.backward(incoming)
        baseline = backbone_gradients(model)
        confirmation = compare_tensors(gradients['1.0'], baseline)
        if not confirmation['exact']:
            save_packet(output_dir / f'{arm}-observer-failure.pt',
                        {'observed': gradients['1.0'], 'unobserved': baseline})
            raise AssertionError('Observed/repeated-backward gradients differ from fresh unobserved replay')
        if not torch.equal(incoming.cpu(), expected_input):
            raise AssertionError('Backward mutated the common cotangent')
        comparisons = {}
        for scale in SCALES[1:]:
            comparisons[str(scale)] = {
                'gradients': compare_tensors(gradients['1.0'], normalized_tensors(gradients[str(scale)], scale), limits=SCALE_LIMITS),
                'write_adjoints': compare_tensors(credit['1.0'], normalized_tensors(credit[str(scale)], scale), limits=SCALE_LIMITS)}
        packet = {'schema': 'rt-precision-credit-arm-v1', 'arm': arm,
                  'model_config': dataclasses.asdict(model.config), 'frozen_pre_logits': frozen_output,
                  'gradients_by_scale': gradients, 'write_adjoints_by_scale': credit,
                  'unused_parameters': sorted(UNUSED_PARAMETERS)}
        record = {'policy': policy, 'bf16_autocast': bf16, 'cuda_graphs': False,
                  'scales': records, 'scale_comparisons': comparisons,
                  'observer_free_replay_forward_bitwise': True,
                  'observer_free_replay_gradients': confirmation,
                  'requires_review': any(item['requires_review'] for row in comparisons.values() for item in row.values()),
                  'compiler': compiler_audit(True), 'seconds': time.monotonic() - started,
                  'packet': save_packet(output_dir / f'{arm}-packet.pt', packet)}
        return record
    finally:
        for block in blocks:
            if hasattr(block, '_recurrent_precision_observer'):
                del block._recurrent_precision_observer
        del model, tokens, incoming
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260911)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh credit-probe directory')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-credit-v1', 'status': 'running', 'seed': args.seed,
              'scope': 'Tiny B3/D64/H4/F256/L2/T16 terminal pre-logit VJP; no task-loss or optimizer claim',
              'scales': SCALES, 'scale_limits': SCALE_LIMITS, 'unused_parameters': sorted(UNUSED_PARAMETERS),
              'arms': {}, 'cross_precision': {}}
    tracker, started = None, time.monotonic()
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        seed_all(args.seed, deterministic=True)
        report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                             'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        report['source_sha256'] = snapshot_sources(args.output_dir)
        criteria = {'scale_limits': SCALE_LIMITS, 'absolute_acceptance_floor': 0.,
                    'review_not_failure': 'Scale-rounding and cross-precision thresholds are review flags',
                    'hard_failures': ['nonfinite or missing used gradient', 'wrong unused-head ownership',
                                      'missing earlier write credit', 'nonzero terminal write credit',
                                      'observer-free replay mismatch', 'mutated forward/cotangent',
                                      'compiler fallback', 'source mutation'],
                    'source_sha256': report['source_sha256']}
        atomic_json(args.output_dir / 'criteria.json', criteria)
        report['criteria_sha256'] = file_digest(args.output_dir / 'criteria.json')
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb')})
        seed_all(args.seed, deterministic=True)
        config = validation_config('tiny')
        report['model_config'] = dataclasses.asdict(config)
        from olmo.model import OLMo
        initial = OLMo(config).to(dtype=torch.float32).train()
        initial_state = cpu_tree(initial.state_dict())
        report['starting_model_digest'] = digest_tensors(initial_state)
        del initial
        gc.collect(); torch.cuda.empty_cache()
        ids = load_ids(None, batch=3, length=16, vocab=config.vocab_size, seed=args.seed)
        cotangent = terminal_cotangent((3, 16, 64), seed=args.seed + 2)
        report['fixture'] = save_packet(args.output_dir / 'fixture.pt',
                                         {'model_config': dataclasses.asdict(config), 'model': initial_state,
                                          'ids': torch.from_numpy(ids), 'cotangent': cotangent})
        for index, arm in enumerate(('C', 'A', 'B')):
            seed_all(args.seed + 3, deterministic=True)
            print(json.dumps({'arm': arm, 'status': 'starting'}), flush=True)
            report['arms'][arm] = run_credit_arm(arm, config, initial_state, ids, cotangent, args.output_dir)
            tracker.log({'update': index + 1, f'credit/{arm}/scale_requires_review': report['arms'][arm]['requires_review']})
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
        packets = {arm: torch.load(args.output_dir / f'{arm}-packet.pt', weights_only=False, map_location='cpu') for arm in ARMS}
        for actual, reference in (('A', 'C'), ('B', 'C'), ('B', 'A')):
            report['cross_precision'][f'{actual}_vs_{reference}'] = {
                'gradients': compare_tensors(packets[reference]['gradients_by_scale']['1.0'], packets[actual]['gradients_by_scale']['1.0'], limits=GRADIENT_LIMITS),
                'write_adjoints': compare_tensors(packets[reference]['write_adjoints_by_scale']['1.0'], packets[actual]['write_adjoints_by_scale']['1.0'], limits=GRADIENT_LIMITS),
                'qualification': 'Different precision forwards; not an independent fixed-operand backward oracle'}
            comparison = report['cross_precision'][f'{actual}_vs_{reference}']
            tracker.summary({f'credit/{actual}_vs_{reference}/gradient_relative_l2': comparison['gradients']['relative_l2'],
                             f'credit/{actual}_vs_{reference}/write_adjoint_relative_l2': comparison['write_adjoints']['relative_l2']})
        verify_sources(report['source_sha256'])
        report['requires_scale_review'] = any(row['requires_review'] for row in report['arms'].values())
        report['status'] = 'complete'
        tracker.summary({'credit/requires_scale_review': report['requires_scale_review'], 'credit/structural_checks_passed': True})
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
    print(json.dumps({'status': report['status'], 'requires_scale_review': report['requires_scale_review']}), flush=True)


if __name__ == '__main__':
    main()
