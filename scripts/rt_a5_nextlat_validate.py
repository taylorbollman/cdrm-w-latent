#!/usr/bin/env python3
"""Bounded GPU checks of A5 NextLat training; backbone-only evaluation.

Invoke as a script through the project Docker launcher. All updates here are
discarded fixtures, not part of either research training trajectory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import numpy as np
import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import build_model, canonical_parameter_sha256, configure_fp32_runtime, fp32_context, make_optimizer, task_loss
from scripts.rt_a5_data import generate_unique_words, prefix_labels
from scripts.rt_a5_nextlat import build_nextlat_model, nextlat_objective
from scripts.rt_a5_nextlat_train import source_manifest, train_step
from scripts.rt_a5_train import atomic_json, preserve_rng
from scripts.stage_a_common import require_cuda_container
# This CLI is invoked by filename, so the historical validation helpers'
# script-local imports resolve without modifying those frozen source files.
from rt_a5_validate import compare_tensors, finite_state

ROOT = Path(__file__).resolve().parents[1]


def fixture(batch, length, seed, output, name):
    words = generate_unique_words(batch, length, seed=seed)
    labels = prefix_labels(words)
    path = output / (name + '.npz')
    np.savez(path, inputs=words, labels=labels)
    return (torch.tensor(words, dtype=torch.long, device='cuda'),
            torch.tensor(labels, dtype=torch.long, device='cuda'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def copied_gradients(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def run_checks(args, report, tracker):
    x, y = fixture(2, 12, args.seed + 20, args.output_dir, 'checks-b2-t12')
    report['zero_auxiliary_limit'] = {}
    for arm in ('rt', 'seq'):
        plain = build_model(arm, width=128, seed=args.seed, device='cuda')
        model = build_nextlat_model(arm, width=128, seed=args.seed, predictor_seed=args.predictor_seed, device='cuda')
        require(canonical_parameter_sha256(plain) == canonical_parameter_sha256(model.backbone), 'Backbone initialization changed')
        aopt, bopt = make_optimizer(plain), make_optimizer(model)
        with fp32_context('cuda'):
            a = plain(x).logits
            aloss = task_loss(a, y)
            aloss.backward()
            packet = nextlat_objective(model, x, y, latent_weight=0.0)
            packet['loss'].backward()
        ag, bg = copied_gradients(plain), copied_gradients(model.backbone)
        require(ag.keys() == bg.keys(), 'Disabled auxiliary changed gradient coverage')
        checks = {'logits': torch.equal(a, packet['logits']), 'loss': torch.equal(aloss, packet['loss']),
                  'gradients': all(torch.equal(ag[n], bg[n]) for n in ag),
                  'predictor_has_no_gradients': all(p.grad is None for p in model.predictor.parameters())}
        for m, opt in ((plain, aopt), (model, bopt)):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        checks['one_adam_update'] = all(torch.equal(p, dict(model.backbone.named_parameters())[n]) for n, p in plain.named_parameters())
        checks['passed'] = all(checks.values())
        report['zero_auxiliary_limit'][arm] = checks
        require(checks['passed'], f'{arm} disabled-auxiliary limit differs')
        # Forward evaluation must not touch the training-only predictor.
        def forbidden(*unused):
            raise AssertionError('Backbone inference called the predictor')
        handle = model.predictor.register_forward_pre_hook(forbidden)
        try:
            with torch.no_grad(), fp32_context('cuda'):
                model(x)
        finally:
            handle.remove()
        report['zero_auxiliary_limit'][arm]['backbone_only_inference'] = True
        del plain, model, aopt, bopt, a, aloss, packet, ag, bg
    gc.collect()
    torch.cuda.empty_cache()

    packets = {}
    for backend in ('naive', 'tiled'):
        model = build_nextlat_model('rt', width=128, seed=args.seed, predictor_seed=args.predictor_seed, device='cuda', backend=backend)
        with fp32_context('cuda'):
            p = nextlat_objective(model, x, y)
            p['loss'].backward()
        check = finite_state(model, require_gradients=True)
        require(check['passed'], 'Combined backward produced missing/nonfinite/non-FP32 gradients')
        packets[backend] = {k: p[k].detach().cpu() for k in ('logits', 'loss', 'state_loss', 'latent_loss')}
        packets[backend]['gradients'] = copied_gradients(model)
        packets[backend]['finite_state'] = check
        del model, p
    a, b = packets['naive'], packets['tiled']
    require(a['gradients'].keys() == b['gradients'].keys(), 'Backend gradient coverage differs')
    comparison = {k: compare_tensors(a[k], b[k]) for k in ('logits', 'loss', 'state_loss', 'latent_loss')}
    comparison['gradients'] = {n: compare_tensors(a['gradients'][n], b['gradients'][n]) for n in a['gradients']}
    comparison['passed'] = all(comparison[k]['passed'] for k in ('logits', 'loss', 'state_loss', 'latent_loss')) and all(v['passed'] for v in comparison['gradients'].values())
    comparison['scope'] = 'Single B2/T12/D128 FP32 combined-loss oracle; inherited atol2e-6, rtol2e-5'
    report['combined_backward'] = comparison
    torch.save(packets, args.output_dir / 'combined-backward.pt')
    atomic_json(args.output_dir / 'progress.json', report)
    require(comparison['passed'], 'Combined FP32 tiled/naive comparison failed; inspect retained packet')
    tracker.log({'check/combined_backward_passed': True,
                 'check/max_gradient_absolute_error': max(v['max_absolute_error'] for v in comparison['gradients'].values())})


def run_profile(args, report, tracker):
    x, y = fixture(1024, 12, args.seed + 30, args.output_dir, 'profile-b1024-t12')
    report['arms'] = {}
    for arm in ('rt', 'seq'):
        model = build_nextlat_model(arm, width=512, seed=args.seed, predictor_seed=args.predictor_seed, device='cuda')
        optimizer = make_optimizer(model)
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, 76):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            values = train_step(model, optimizer, x, y, diagnostics=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            if step > 25:
                times.append(seconds)
            if step % 25 == 0:
                tracker.log({'update': step, f'benchmark/{arm}/seconds': seconds, f'benchmark/{arm}/loss': values['loss']})
        check = finite_state(model, optimizer, require_gradients=True)
        require(check['passed'], f'{arm} B1024 state invalid')
        case = {'warmup_updates': 25, 'timed_updates': 50, 'batch': 1024, 'length': 12, 'width': 512,
                'mean_seconds': statistics.mean(times), 'median_seconds': statistics.median(times),
                'first_half_median_seconds': statistics.median(times[:25]), 'last_half_median_seconds': statistics.median(times[25:]),
                'training_only_minutes_10000': statistics.mean(times) * 10000 / 60,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'last_metrics': values, 'finite_state': check,
                'scope': 'Discarded fixed-fixture updates, timed synchronized train_step incl scalar metrics; excludes data transfer/evaluation/checkpoint/W&B'}
        report['arms'][arm] = case
        tracker.log({f'benchmark/{arm}/{k}': v for k, v in case.items() if isinstance(v, (int, float))})
        atomic_json(args.output_dir / 'progress.json', report)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run_overfit(args, report, tracker):
    x, y = fixture(32, 4, args.seed + 40, args.output_dir, 'overfit-b32-t4')
    report['scope'] = 'Discarded 32-word short-set plumbing smoke, not length generalization; LR1e-3'
    report['arms'] = {}
    for arm in ('rt', 'seq'):
        model = build_nextlat_model(arm, width=128, seed=args.seed, predictor_seed=args.predictor_seed, device='cuda')
        optimizer = make_optimizer(model, lr=1e-3)
        rows, reached = [], False
        for step in range(1, 301):
            values = train_step(model, optimizer, x, y, diagnostics=False)
            if step == 1 or step % 10 == 0:
                with torch.no_grad(), fp32_context('cuda'):
                    correct = model(x).logits.argmax(-1).eq(y)
                row = {'update': step, **values, 'post_update_exact': correct.all(-1).float().mean().item()}
                rows.append(row)
                tracker.log({'update': step, **{f'train/{arm}/{k}': v for k, v in row.items() if k != 'update'}})
                if row['post_update_exact'] == 1.0:
                    reached = True
                    break
        check = finite_state(model, optimizer, require_gradients=True)
        report['arms'][arm] = {'passed': reached and check['passed'], 'rows': rows, 'finite_state': check}
        atomic_json(args.output_dir / 'progress.json', report)
        require(report['arms'][arm]['passed'], f'{arm} short-set overfit smoke did not pass')
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('checks', 'profile', 'overfit'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--predictor-seed', default=1235, type=int)
    parser.add_argument('--wandb-group', required=True)
    args = parser.parse_args()
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh validation output directory')
    args.output_dir.mkdir(parents=True)
    sources = source_manifest()
    for extra in (__file__, ROOT / 'scripts/rt_a5_validate.py'):
        path = Path(extra).resolve()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in sources:
        target = args.output_dir / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    report = {'schema': 'rt-a5-nextlat-validation-v1', 'status': 'running', 'mode': args.mode,
              'hardware': hardware, 'runtime': runtime, 'source_files': sources, 'confirmation_evaluated': False,
              'evaluation_route': 'backbone_only', 'latent_recurrence_evaluated': False}
    tracker = OnlineTracker(project='rt-a5-state-tracking', output_dir=args.output_dir,
                            group=args.wandb_group, name='nextlat-' + args.mode, preserve_state=preserve_rng)
    started = time.perf_counter()
    try:
        tracker.start({'mode': args.mode, 'seed': args.seed, 'predictor_seed': args.predictor_seed, **runtime,
                       'evaluation_route': 'backbone_only', 'scope': 'Discarded bounded correctness/timing fixtures'})
        {'checks': run_checks, 'profile': run_profile, 'overfit': run_overfit}[args.mode](args, report, tracker)
        report['status'] = 'passed'
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started, wandb=tracker.record)
        atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'mode': args.mode, 'wandb': report['wandb']['run_url']}), flush=True)


if __name__ == '__main__':
    main()
