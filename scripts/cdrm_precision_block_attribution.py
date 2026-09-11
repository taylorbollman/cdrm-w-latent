#!/usr/bin/env python3
"""Replay ordinary block zero with fixed primal inputs and incoming cotangents.

Block zero owns no fabric parameters, so its isolated gradients can be checked
bitwise against the corresponding complete-model packet. Counterfactual saved
cotangents separate local precision effects from changes arriving downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import cdrm_tiled_common as common
from cdrm_precision_probe import ARMS, intervention, exact_tree
from cdrm_tiled_validate import cpu, digest, metric, preserve_rng, save_json
from experiment_tracking import OnlineTracker


def vector_components(reference, actual, fixed_cotangent):
    if list(reference) != list(actual) or list(reference) != list(fixed_cotangent):
        raise ValueError('Local gradient coverage differs')
    rows = {}
    for name, value in reference.items():
        r, a, c = (x.double() for x in (value, actual[name], fixed_cotangent[name]))
        local, incoming, total = c-r, a-c, a-r
        rows[name] = {'reference_energy': float(r.square().sum()),
                      'local_precision_energy': float(local.square().sum()),
                      'incoming_cotangent_energy': float(incoming.square().sum()),
                      'twice_cross_inner_product': float(2*(local*incoming).sum()),
                      'total_error_energy': float(total.square().sum())}
    total = {key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}
    scale = max(total['reference_energy'], 1e-30)**.5
    for key in ('local_precision', 'incoming_cotangent', 'total_error'):
        total[key+'_relative_l2'] = total[key+'_energy']**.5/scale
    total['energy_identity_residual'] = (total['total_error_energy']-total['local_precision_energy']
                                       -total['incoming_cotangent_energy']-total['twice_cross_inner_product'])
    if abs(total['energy_identity_residual']) > 1e-12*max(total['reference_energy'], total['total_error_energy'], 1e-30):
        raise AssertionError('Attribution energy identity does not close')
    return {'global': total, 'rows': rows,
            'interpretation': 'Exact counterfactual vector decomposition, not additive error percentages or a derivative through quantization. Local term includes forward rounding and backward casts within block zero at a fixed incoming cotangent.'}


def replay(checkpoint, arm_name, x, dy, bias):
    arm = ARMS[arm_name]
    net, _ = common.build_model('tiled', 'bf16_fp32_state' if arm.bf16 else 'fp32', checkpoint=checkpoint)
    block = net.transformer.blocks[0]
    names = dict(block.named_parameters())
    x = x.cuda().detach().requires_grad_()
    with intervention(net, arm), torch.autocast('cuda', dtype=torch.bfloat16, enabled=arm.bf16):
        y, _ = block(x, attention_bias=bias.cuda())
    result = torch.autograd.grad(y, (*names.values(), x), dy.cuda())
    return {'output': cpu(y), 'gradients': dict(zip((*names, 'input'), map(cpu, result)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, tracker, retained = time.monotonic(), None, {}
    report = {'schema': 'cdrm-block-zero-cotangent-attribution-v1', 'status': 'running',
              'numerical_clearance': False, 'scope': 'Captured physical B64/T256 block-zero numerical counterfactuals only; no training or changed acceptance criteria.',
              'probe': {'path': str(args.probe_dir), 'report_sha256': digest(args.probe_dir/'report.json'),
                        'tensors_sha256': digest(args.probe_dir/'tensors.pt')},
              'checkpoint': {'path': str(args.checkpoint), 'sha256': digest(args.checkpoint)},
              'replay_parity': {}, 'attribution': {}}
    try:
        report['runtime'] = common.setup(0)
        tracked = common.sources((Path(__file__).resolve().relative_to(Path.cwd()),
                                  Path('scripts/cdrm_precision_probe.py'), Path('scripts/cdrm_tiled_validate.py'),
                                  Path('tests/test_cdrm_precision_block_attribution.py')))
        report['source_sha256'] = tracked
        common.snapshot_sources(args.output_dir, tracked)
        pr = json.loads((args.probe_dir/'report.json').read_text())
        if (pr['status'] != 'diagnostics_complete' or not pr['compiler']['all_arms_pass']
                or pr['tensor_artifact']['sha256'] != report['probe']['tensors_sha256']
                or pr['checkpoint']['sha256'] != report['checkpoint']['sha256']):
            raise ValueError('Require an audited matching complete-model probe')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        data = torch.load(args.probe_dir/'tensors.pt', map_location='cpu', weights_only=False)
        if checkpoint['model_config']['cdrm_early_layer'] != 1:
            raise ValueError('Block zero must not share parameters with the fabric')
        arms = ('current_fp32', 'current_bf16', 'bf16_attention_fp32')
        captures = {name: data['captures'][name]['blocks']['0'] for name in arms}
        x = captures['current_fp32']['input']
        dy = captures['current_fp32']['output_gradient']
        if tuple(x.shape) != (64, 256, 128) or not all(torch.equal(x, c['input']) for c in captures.values()):
            raise ValueError('Require identical original physical B64/T256/D128 block-zero inputs')
        bias = captures['current_fp32']['sdpa']['bias_before_cast']
        if not all(torch.equal(bias, c['sdpa']['bias_before_cast']) for c in captures.values()):
            raise ValueError('Original FP32 attention biases differ')
        tracker = OnlineTracker(project='cdrm-numerical-resolution', entity='taylorbollman',
                                group='20260907T232931Z', name='block-zero-fixed-cotangent',
                                output_dir=args.output_dir, preserve_state=preserve_rng)
        report['wandb'] = tracker.record
        tracker.start({k: report[k] for k in ('scope', 'probe', 'checkpoint', 'source_sha256')})
        for name in arms:
            native = replay(checkpoint, name, x, captures[name]['output_gradient'], bias)
            full_parameters = data['packets'][name]['parameters']
            expected = {key: full_parameters['transformer.blocks.0.'+key]
                        for key in native['gradients'] if key != 'input'}
            expected['input'] = captures[name]['input_gradient']
            differences = exact_tree({'output': captures[name]['output'], 'gradients': expected}, native)
            retained[name] = {'native_replay': native}
            report['replay_parity'][name] = {'bitwise_equal': not differences, 'differences': differences}
            if differences:
                raise AssertionError(f'Block-zero isolated replay differs: {name}')
            if name == 'current_fp32':
                continue
            counter = replay(checkpoint, name, x, dy, bias)
            retained[name]['fixed_fp32_cotangent'] = counter
            reference = retained['current_fp32']['native_replay']
            names = [key for key in reference['gradients'] if key != 'input']
            obs = {'parameter_gradients': vector_components(
                        {k: reference['gradients'][k] for k in names},
                        {k: native['gradients'][k] for k in names},
                        {k: counter['gradients'][k] for k in names}),
                   'input_gradient': vector_components(
                        {'input': reference['gradients']['input']},
                        {'input': native['gradients']['input']}, {'input': counter['gradients']['input']}),
                   'forward_output': metric(reference['output'], native['output']),
                   'incoming_cotangent': metric(dy, captures[name]['output_gradient'])}
            report['attribution'][name] = obs
            tracker.log({f'{name}/{key}': value for key, value in obs['parameter_gradients']['global'].items()})
            print(json.dumps({'arm': name, 'parameter_gradients': obs['parameter_gradients']['global'],
                              'input_gradient': obs['input_gradient']['global'],
                              'forward_relative_l2': obs['forward_output']['rel_l2'],
                              'incoming_relative_l2': obs['incoming_cotangent']['rel_l2']}), flush=True)
        common.verify_sources(tracked)
        if (digest(args.checkpoint) != report['checkpoint']['sha256']
                or digest(args.probe_dir/'report.json') != report['probe']['report_sha256']
                or digest(args.probe_dir/'tensors.pt') != report['probe']['tensors_sha256']):
            raise RuntimeError('Replay inputs changed')
        report['status'] = 'diagnostics_complete'
        tracker.summary({'all_block_replays_bitwise_equal': True, 'numerical_clearance': False})
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            if retained:
                torch.save(retained, args.output_dir/'references.pt')
                report['tensor_artifact'] = {'path': str(args.output_dir/'references.pt'), 'sha256': digest(args.output_dir/'references.pt')}
            if tracker:
                tracker.finish(succeeded=report['status'] == 'diagnostics_complete')
        except Exception as error:
            report.update(status='failed', finalization_error_type=type(error).__name__)
            raise
        finally:
            save_json(args.output_dir/'report.json', report)


if __name__ == '__main__':
    main()
