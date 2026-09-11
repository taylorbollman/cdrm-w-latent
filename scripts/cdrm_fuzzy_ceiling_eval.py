#!/usr/bin/env python3
"""Post-selection, same-checkpoint fuzzy recall coverage and precision diagnostics.

The official epoch-10 BF16 metrics must replay exactly. Oracle-availability
subsets and a strict FP32 evaluation describe the selected checkpoint; they do
not replace native labels, select an LR, or train a model.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import cdrm_fuzzy_common as common
from experiment_tracking import OnlineTracker, add_wandb_arguments, scalar_metrics
from stage_a_common import configure_compiled_helpers
from stage_b_train import compiler_audit


def masks_for(labels, oracle_available):
    labels, available = np.asarray(labels), np.asarray(oracle_available)
    if labels.ndim != 2 or available.shape != labels.shape or available.dtype != np.bool_:
        raise ValueError('Require aligned [N,T] labels and an explicit boolean availability mask')
    native = labels != -100
    if np.any(available & ~native):
        raise ValueError('Oracle availability includes a non-native target')
    return {'native': native, 'oracle_available': available.copy(),
            'oracle_unavailable': native & ~available}


def subset_metrics(predictions, labels, mask, token_ce=None):
    predictions, labels, mask = map(np.asarray, (predictions, labels, mask))
    if labels.ndim != 2 or predictions.shape != labels.shape or mask.shape != labels.shape or mask.dtype != np.bool_:
        raise ValueError('Require aligned predictions, labels and a boolean [N,T] mask')
    if np.any(mask & (labels == -100)):
        raise ValueError('A subset cannot add ignored targets')
    correct = (predictions == labels) & mask
    eligible = mask.any(axis=1)
    exact = ((correct | ~mask).all(axis=1) & eligible)
    targets, examples = int(mask.sum()), int(eligible.sum())
    result = {'targets': targets, 'correct': int(correct.sum()), 'errors': targets - int(correct.sum()),
              'examples_with_targets': examples, 'examples_without_targets': int((~eligible).sum()),
              'exact': int(exact.sum()), 'examples_with_errors': examples - int(exact.sum()),
              'token_accuracy': int(correct.sum()) / targets if targets else None,
              'sequence_exact_match': int(exact.sum()) / examples if examples else None}
    if token_ce is not None:
        token_ce = np.asarray(token_ce)
        if token_ce.shape != labels.shape or not np.isfinite(token_ce[mask]).all():
            raise ValueError('Per-position CE must align and be finite on scored targets')
        result['ce_sum'] = float(token_ce[mask].sum(dtype=np.float64))
        result['ce'] = result['ce_sum'] / targets if targets else None
    return result


def prediction_changes(bf16, fp32, labels, mask):
    left, right = subset_metrics(bf16, labels, mask), subset_metrics(fp32, labels, mask)
    changed = (bf16 != fp32) & mask
    left_correct, right_correct = bf16 == labels, fp32 == labels
    eligible = mask.any(axis=1)
    left_exact = (left_correct | ~mask).all(axis=1) & eligible
    right_exact = (right_correct | ~mask).all(axis=1) & eligible
    return {'targets': left['targets'], 'prediction_flips': int(changed.sum()),
            'bf16_wrong_fp32_right': int((mask & ~left_correct & right_correct).sum()),
            'bf16_right_fp32_wrong': int((mask & left_correct & ~right_correct).sum()),
            'both_wrong_prediction_changed': int((changed & ~left_correct & ~right_correct).sum()),
            'examples_with_targets': left['examples_with_targets'],
            'exact_examples_gained_in_fp32': int((~left_exact & right_exact).sum()),
            'exact_examples_lost_in_fp32': int((left_exact & ~right_exact).sum()),
            'token_accuracy_fp32_minus_bf16': right['token_accuracy'] - left['token_accuracy'] if left['targets'] else None,
            'sequence_exact_match_fp32_minus_bf16': right['sequence_exact_match'] - left['sequence_exact_match']
                if left['examples_with_targets'] else None}


def validate_authority(checkpoint, training_report, checkpoint_ref, dataset):
    if (checkpoint.get('format') != common.FORMAT or training_report.get('schema') != 'cdrm-fuzzy-calibration-v1'
            or training_report.get('status') != 'complete'):
        raise ValueError('Require a completed authoritative fuzzy calibration report and training checkpoint')
    arguments = training_report.get('arguments', {})
    declared_output = Path(arguments.get('output_dir', ''))
    if (training_report.get('recovery_comparison') is not None or arguments.get('reference_final') is not None
            or declared_output.parent.name != 'calibration' or not declared_output.name.endswith('-e10')):
        raise ValueError('Require the authoritative calibration *-e10 endpoint report, not a recovery control')
    identity = checkpoint['identity']
    if (identity != training_report['identity'] or common.json_digest(identity) != checkpoint['identity_sha256']
            or checkpoint['identity_sha256'] != training_report['identity_sha256']):
        raise ValueError('Checkpoint and training-report identities differ')
    if (checkpoint['arm'] != identity['arm'] or training_report['arm'] != identity['arm']
            or identity['arm'] not in common.ARMS or checkpoint['precision'] != 'bf16'
            or training_report['precision'] != 'bf16' or identity['precision'] != 'bf16'
            or identity['physical_batch'] != 128 or identity['length'] != 256
            or identity['updates_per_epoch'] != 100 or identity['maximum_epochs'] != 10):
        raise ValueError('Require the declared BF16 B128/T256 ten-epoch calibration identity')
    for record in (checkpoint, training_report):
        if (record['completed_updates'], record['completed_epochs'], record['batch_in_epoch']) != (1000, 10, 0):
            raise ValueError('This diagnostic requires the authoritative epoch-10 endpoint')
        if record['model_config'] != identity['model_config']:
            raise ValueError('Model configuration differs from the frozen calibration identity')
    if identity['model_config']['ordinary_attention_precision_policy'] != 'fp32':
        raise ValueError('Require the validated ordinary-attention FP32 policy')
    if checkpoint_ref != training_report['checkpoints'].get('1000'):
        raise ValueError('Checkpoint is not the authoritative epoch-10 checkpoint in the supplied report')
    common.verify_sources(identity['source_sha256'])
    common.validate_data(dataset, 'dev', length=256)
    if len(dataset) != 1280 or common.dataset_identity(dataset) != identity['data']['dev']:
        raise ValueError('Require the same full retained calibration development corpus')
    if common.reference(identity['data_manifest']['path']) != identity['data_manifest']:
        raise ValueError('Frozen calibration data manifest changed')
    for name in ('native', 'answer'):
        if checkpoint['development']['1000'][name] != training_report['development']['1000'][name]:
            raise ValueError('Checkpoint and report disagree on the official native evaluation')
    return training_report['development']['1000']


def require_native_replay(actual, expected):
    equal = all(actual[name] == expected[name] for name in ('native', 'answer'))
    if not equal:
        raise AssertionError('BF16 native development metrics differ from the authoritative epoch-10 report')
    return True


@torch.no_grad()
def capture_evaluation(model, dataset, precision):
    if next(model.parameters()).device.type != 'cuda':
        raise RuntimeError('Evaluation requires the authorized CUDA container; no CPU fallback')
    modes = {module: module.training for module in model.modules()}
    totals = {name: {'ce_sum': 0., 'targets': 0, 'correct': 0, 'exact': 0, 'examples': 0} for name in ('native', 'answer')}
    logits_rows, predictions, token_ce = [], [], []
    started = time.perf_counter()
    try:
        with common.preserve_tracking_rng():
            model.eval()
            for begin in range(0, len(dataset), 128):
                ids = torch.as_tensor(dataset.input_ids[begin:begin + 128], device='cuda')
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    logits = model(ids).logits
                if logits.dtype != (torch.bfloat16 if precision == 'bf16' else torch.float32):
                    raise AssertionError('Observed logits precision differs from this evaluation arm')
                # Preserve the native evaluator's batch order, reductions and metric arithmetic.
                for name, array in (('native', dataset.labels), ('answer', dataset.answer_labels)):
                    labels = torch.as_tensor(array[begin:begin + 128], device='cuda')
                    for key, value in common.metric_counts(logits, labels).items():
                        totals[name][key] += value
                logits_rows.append(logits.cpu())
                predictions.append(logits.argmax(-1).cpu())
                losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                         ignore_index=-100, reduction='none').reshape(labels.shape)
                token_ce.append(losses.cpu())
            torch.cuda.synchronize()
    finally:
        for module, training in modes.items():
            module.training = training
    metrics = {name: common.finish_metrics(values) for name, values in totals.items()}
    return metrics, {'logits': torch.cat(logits_rows), 'predictions': torch.cat(predictions),
                     'token_ce': torch.cat(token_ce)}, time.perf_counter() - started


def save_tensor_packet(path, packet):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(packet, temporary)
    temporary.replace(path)
    return common.reference(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='cdrm-150m-fuzzy-recall', wandb_group='20260908T031418Z')
    args = parser.parse_args(argv)
    if not args.wandb_project:
        parser.error('Online W&B is required for the evaluation diagnostic')
    return args


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new evaluation diagnostic output directory')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'cdrm-fuzzy-ceiling-evaluation-v1', 'status': 'running',
              'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'scope': 'Post-selection development diagnostic; official BF16 native metrics and LR selection remain unchanged.',
              'subset_exact_denominator': 'Only examples with at least one target in that subset.',
              'subset_ce_arithmetic': 'FP32 per-position CE accumulated in FP64; official native replay retains the frozen batch-reduction arithmetic.',
              'interpretation': 'Oracle availability describes causal lookup coverage, not a universal statistical ceiling or proof that unavailable targets cannot be predicted.',
              'evaluations': {}, 'numerical_clearance': False}
    tracker = None
    started = time.monotonic()
    try:
        report['runtime'] = common.setup(0)
        checkpoint_ref, training_ref = common.reference(args.checkpoint), common.reference(args.training_report)
        report.update(checkpoint=checkpoint_ref, training_report=training_ref)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        training_report = json.loads(args.training_report.read_text())
        dataset = common.load_dataset(args.data_root, common.TASK, 'dev', verify=True)
        expected = validate_authority(checkpoint, training_report, checkpoint_ref, dataset)
        if report['runtime']['execution_contract'] != checkpoint['identity']['execution_contract']:
            raise ValueError('Evaluation runtime/cache differs from the recorded training policy')
        source_map = {**common.sources((str(Path(__file__).relative_to(Path.cwd())),)),
                      **checkpoint['identity']['source_sha256']}
        common.verify_sources(source_map)
        common.snapshot_sources(args.output_dir, source_map)
        masks = masks_for(dataset.labels, dataset.oracle_available_mask)
        report.update(arm=checkpoint['arm'], checkpoint=checkpoint_ref, training_report=training_ref,
                      identity_sha256=checkpoint['identity_sha256'], source_sha256=source_map,
                      dataset=common.dataset_identity(dataset), dataset_manifest=dataset.manifest,
                      official_bf16_native_metrics={name: expected[name] for name in ('native', 'answer')},
                      mask_sha256={name: common.state_digest(value) for name, value in masks.items()})
        mask_path = args.output_dir / 'native-labels-and-masks.npz'
        np.savez_compressed(mask_path, labels=dataset.labels, input_ids=dataset.input_ids, **masks)
        report['mask_packet'] = common.reference(mask_path)
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir, preserve_state=common.preserve_tracking_rng)
        report['wandb'] = tracker.record
        tracker.start({'evidence': 'post_selection_development_evaluation', 'arm': checkpoint['arm'],
                       'checkpoint_sha256': checkpoint_ref['sha256'], 'training_report_sha256': training_ref['sha256'],
                       'dataset': report['dataset'], 'completed_updates': 1000, 'native_labels_unchanged': True})
        predictions = {}
        for precision in ('bf16', 'fp32'):
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            configure_compiled_helpers(True)
            model, construction = common.build_model(checkpoint['arm'], precision, checkpoint=checkpoint, length=256)
            before_rng = common.state_digest(common.rng_state())
            metrics, packet, seconds = capture_evaluation(model, dataset, precision)
            unchanged = (common.state_digest(model.state_dict()) == construction['starting_weights_sha256']
                         and before_rng == common.state_digest(common.rng_state())
                         and all(parameter.grad is None for parameter in model.parameters()))
            if not unchanged:
                raise AssertionError('Evaluation changed model weights, RNG or gradients')
            preds, token_ce = packet['predictions'].numpy(), packet['token_ce'].numpy()
            subsets = {name: subset_metrics(preds, dataset.labels, mask, token_ce) for name, mask in masks.items()}
            for key in ('targets', 'correct', 'exact'):
                if subsets['native'][key] != metrics['native'][key]:
                    raise AssertionError('Captured predictions disagree with native metric counts')
            errors = np.argwhere(masks['native'] & (preds != dataset.labels))
            packet['native_error_coordinates'] = torch.from_numpy(errors)
            record = {'precision': precision, 'native_metrics': metrics, 'subsets': subsets,
                      'construction': construction, 'state_unchanged': unchanged, 'seconds': seconds,
                      'compiler': compiler_audit(True, require_graphs=checkpoint['arm'] == 'cdrm'),
                      'packet': save_tensor_packet(args.output_dir / f'{precision}-predictions.pt', packet),
                      'error_sample': [{'example': int(row), 'position': int(col), 'target': int(dataset.labels[row, col]),
                                        'prediction': int(preds[row, col]), 'oracle_available': bool(masks['oracle_available'][row, col])}
                                       for row, col in errors[:100]],
                      'all_error_coordinates_retained_in_packet': True}
            report['evaluations'][precision] = record
            if precision == 'bf16':
                report['bf16_native_replay_exact'] = require_native_replay(metrics, expected)
            predictions[precision] = preds.copy()
            tracker.log(scalar_metrics(subsets, f'evaluation/{precision}'))
            common.atomic_json(args.output_dir / 'progress.json', report, replace=True)
            del model, packet
            gc.collect()
            torch.cuda.empty_cache()
        report['prediction_changes'] = {name: prediction_changes(predictions['bf16'], predictions['fp32'], dataset.labels, mask)
                                        for name, mask in masks.items()}
        tracker.summary({'bf16_native_replay_exact': report['bf16_native_replay_exact'],
                         **scalar_metrics(report['prediction_changes'], 'fp32_minus_bf16')})
        if common.reference(args.checkpoint) != checkpoint_ref or common.reference(args.training_report) != training_ref:
            raise AssertionError('Scientific checkpoint or training report changed during evaluation')
        if common.dataset_identity(common.load_dataset(args.data_root, common.TASK, 'dev', verify=True)) != report['dataset']:
            raise AssertionError('Retained development data changed during evaluation')
        common.verify_sources(source_map)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if tracker:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', final_sync_error_type=type(error).__name__)
            raise
        finally:
            report['elapsed_seconds'] = time.monotonic() - started
            common.atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'bf16_native_replay_exact': report['bf16_native_replay_exact'],
                      'report': str(args.output_dir / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
