#!/usr/bin/env python3
"""CPU-only paired document analysis of completed RT precision evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiment_tracking import OnlineTracker, add_wandb_arguments

ROOT = Path(__file__).resolve().parents[1]
RESAMPLES, BOOTSTRAP_SEED, MARGIN = 10000, 20260912, .005
POLICIES = {'A': 'bf16_fp32_state', 'B': 'legacy'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def validate_protocol(protocol):
    if protocol.get('schema') != 'rt-precision-alignment-protocol-v1':
        raise ValueError('Unexpected frozen protocol schema')
    evaluation = protocol['evaluation']
    if (evaluation.get('ce_margin_nats_per_supervised_token') != MARGIN
            or evaluation.get('common_evaluation') != 'FP32 native forward, per supervised token CE'):
        raise ValueError('Protocol margin or common-evaluation definition differs')
    # The frozen protocol stores the bootstrap settings in a textual field.
    uncertainty = evaluation.get('uncertainty', '')
    if not all(value in uncertainty for value in ('paired bootstrap', 'document', '95% upper',
                                                  'token-weighted', '10000', 'seed20260912')):
        raise ValueError('Protocol bootstrap settings differ from this bounded implementation')
    return evaluation


def validate_documents(losses, documents, boundaries, *, length):
    """Reconcile every document against independently indexed token losses."""
    if (losses.ndim != 2 or losses.shape[1] != length - 1 or losses.dtype != np.float32
            or not np.isfinite(losses).all() or np.any(losses < 0)):
        raise ValueError('Require finite nonnegative FP32 shifted-token CE')
    if len(documents) != len(boundaries) or len(documents) < 2:
        raise ValueError('Document packet and boundary coverage differ')
    flat = losses.reshape(-1)
    end = total_targets = 0
    seen = set()
    checked = []
    for document, boundary in zip(documents, boundaries):
        begin, stop = boundary['token_begin'], boundary['token_end']
        if (type(begin) is not int or type(stop) is not int or begin != end or stop <= begin
                or stop > len(losses) * length):
            raise ValueError('Boundaries must cover the evaluated packed stream exactly once')
        end = stop
        name = boundary['text_sha256']
        if name in seen or document['text_sha256'] != name:
            raise ValueError('Document identity/order is duplicated or mismatched')
        seen.add(name)
        # Number of supervised tokens strictly before packed position p is
        # p-ceil(p/T): each row's position zero is an input without a target.
        left = begin - (begin + length - 1) // length
        right = stop - (stop + length - 1) // length
        targets = right - left
        expected_sum = float(flat[left:right].sum(dtype=np.float64))
        actual_sum = document['ce_sum']
        if (type(document['targets']) is not int or document['targets'] != targets
                or not isinstance(actual_sum, (int, float)) or not math.isfinite(actual_sum)
                or actual_sum < 0 or not math.isclose(actual_sum, expected_sum, rel_tol=1e-12, abs_tol=1e-8)):
            raise ValueError('Document counts or CE sums disagree with token CE')
        total_targets += targets
        checked.append({'text_sha256': name, 'targets': targets, 'ce_sum': float(actual_sum)})
    if end != len(losses) * length or total_targets != losses.size:
        raise ValueError('Document boundaries do not cover all evaluated targets')
    return checked


def paired_bootstrap(reference, candidate, *, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    if len(reference) != len(candidate) or len(reference) < 2 or resamples < 2:
        raise ValueError('Require matching paired documents and at least two resamples')
    for a, b in zip(reference, candidate):
        if a['text_sha256'] != b['text_sha256'] or a['targets'] != b['targets']:
            raise ValueError('Paired document identities/order or target counts differ')
        if (type(a['targets']) is not int or a['targets'] < 0
                or any(not math.isfinite(row['ce_sum']) or row['ce_sum'] < 0 for row in (a, b))
                or (a['targets'] == 0 and (a['ce_sum'] != 0 or b['ce_sum'] != 0))):
            raise ValueError('Invalid paired document counts or losses')
    count = np.array([row['targets'] for row in reference], dtype=np.float64)
    difference = np.array([b['ce_sum'] - a['ce_sum'] for a, b in zip(reference, candidate)], dtype=np.float64)
    keep = count > 0
    count, difference = count[keep], difference[keep]
    if len(count) < 2:
        raise ValueError('Require at least two documents with supervised targets')
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for begin in range(0, resamples, 100):
        end = min(begin + 100, resamples)
        indices = rng.integers(0, len(count), size=(end - begin, len(count)))
        draws[begin:end] = difference[indices].sum(1) / count[indices].sum(1)
    return {'candidate_minus_reference_nats_per_target': float(difference.sum() / count.sum()),
            'one_sided_95_upper': float(np.quantile(draws, .95, method='linear')),
            'two_sided_95_interval': np.quantile(draws, [.025, .975], method='linear').tolist(),
            'resamples': resamples, 'seed': seed, 'resampled_documents': len(count),
            'excluded_zero_target_documents': int((~keep).sum()), 'supervised_tokens': int(count.sum()),
            'method': 'Uniform paired document resampling with replacement; each replicate is sum(B-A CE)/sum(targets)',
            'quantile_method': 'linear', 'draws_sha256': hashlib.sha256(draws.tobytes()).hexdigest()}


def validate_pair_identity(a, b, train_a, train_b):
    for label, report, training in [('A', a, train_a), ('B', b, train_b)]:
        identity = report.get('checkpoint_identity', {})
        if (report.get('schema') != 'rt-precision-evaluation-v1' or report.get('status') != 'complete'
                or identity.get('status') != 'verified'
                or training.get('schema') != 'rt-precision-training-v1' or training.get('status') != 'complete'):
            raise ValueError('Require completed evaluation/training reports with verified checkpoint identity')
        policy = POLICIES[label]
        if (identity.get('training_policy') != policy or training.get('policy') != policy
                or report['model_config']['recurrent_precision_policy'] != policy):
            raise ValueError('Explicit A/B inputs must be protected/legacy respectively')
        update = identity.get('completed_updates')
        if (update not in (100, 500) or report.get('checkpoint_update') != update
                or training.get('completed_updates') != update or training.get('endpoint') != update
                or training.get('seed') != identity.get('seed')):
            raise ValueError('Evaluation seed/update does not match its completed training endpoint')
        matches = [row for row in training.get('checkpoints', []) if row.get('sha256') == report['checkpoint']['sha256']]
        if (len(matches) != 1 or matches[0].get('completed_updates') != update
                or identity.get('checkpoint_sha256') != report['checkpoint']['sha256']):
            raise ValueError('Evaluator checkpoint is not its recorded training endpoint')
        if not identity.get('saved_parameters_cpu_fp32_finite') or not identity.get('canonical_parameter_coverage'):
            raise ValueError('Evaluator did not verify finite canonical FP32 parameters')
        if (identity.get('protocol_sha256') != report['protocol_sha256']
                or training.get('protocol_sha256') != report['protocol_sha256']
                or identity.get('heldout_manifest_sha256') != report['data']['manifest_sha256']
                or training.get('heldout_manifest_sha256') != report['data']['manifest_sha256']
                or identity.get('training_manifest_sha256') != training['data']['manifest_sha256']):
            raise ValueError('Protocol or held-out/training data identity differs')
        if training.get('precision') != 'bf16' or report.get('cuda_graphs') is not False:
            raise ValueError('Unexpected training precision or evaluation path')
        checked_sources = identity.get('checked_source_sha256', {})
        model_sources = {name for name in training['source_sha256'] if name.startswith('recurrent-transformer/olmo/')}
        if not model_sources or not model_sources <= set(checked_sources):
            raise ValueError('Evaluator lacks complete checked model-source identity')
        for name, expected in checked_sources.items():
            if report['source_sha256'].get(name) != expected or training['source_sha256'].get(name) != expected:
                raise ValueError('Checked evaluator source differs from training/source snapshots')
    for key in ('role', 'precision', 'batch', 'normalization', 'data', 'protocol_sha256', 'source_sha256'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Paired evaluator setting differs: {key}')
    if a['precision'] not in ('fp32', 'native') or a['role'] not in ('diagnostic', 'dev', 'confirmation'):
        raise ValueError('Unsupported evaluation precision or role')
    for key in ('seed', 'completed_updates', 'parameter_count', 'parameter_tensors'):
        if a['checkpoint_identity'].get(key) != b['checkpoint_identity'].get(key):
            raise ValueError(f'Paired checkpoint identity differs: {key}')
    for key in ('initial_state_sha256', 'seed', 'data', 'schedule', 'optimizer', 'source_sha256'):
        if not train_a.get(key) or train_a.get(key) != train_b.get(key):
            raise ValueError(f'Paired training conditions differ: {key}')
    ca, cb = dict(a['model_config']), dict(b['model_config'])
    ca.pop('recurrent_precision_policy'); cb.pop('recurrent_precision_policy')
    if ca != cb:
        raise ValueError('Paired model architecture or other precision settings differ')
    return {'role': a['role'], 'precision': a['precision'], 'evaluation_batch': a['batch'],
            'seed': a['checkpoint_identity']['seed'], 'completed_updates': a['checkpoint_update'],
            'initial_state_sha256': train_a['initial_state_sha256'],
            'common_fp32_primary': a['precision'] == 'fp32',
            'interpretation': 'Common FP32 evaluation of differently trained weights' if a['precision'] == 'fp32'
                              else 'Secondary native comparison: combines differently trained weights and their execution precision'}


def analyze(a_dir, b_dir, manifest_path, protocol_path):
    inputs = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input artifact hash differs: {path}')
        inputs[str(path)] = digest
        return path

    manifest = read_json(record(manifest_path))
    protocol = read_json(record(protocol_path))
    validate_protocol(protocol)
    if (manifest.get('schema') != 'rt-precision-c4-data-v1' or manifest.get('status') != 'complete'
            or manifest.get('mode') != 'heldout'):
        raise ValueError('Require the completed frozen held-out manifest')
    reports, training, documents, means = {}, {}, {}, {}
    for label, directory in [('A', Path(a_dir)), ('B', Path(b_dir))]:
        report = read_json(record(directory / 'report.json'))
        if report.get('status') != 'complete':
            raise ValueError('Evaluation has not completed')
        record(protocol_path, report['protocol_sha256'])
        record(manifest_path, report['data']['manifest_sha256'])
        training[label] = read_json(record(project_path(report['training_report']['path']), report['training_report']['sha256']))
        for name, expected in report['source_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe evaluation source snapshot path')
            record(directory / 'source' / name, expected)
        role = manifest['roles'][report['role']]
        if report['data'] != {'manifest_sha256': sha256(manifest_path), **role}:
            raise ValueError('Evaluation role/data metadata differs from frozen manifest')
        base = Path(manifest_path).resolve().parent
        for key in ('ids_path', 'boundaries_path'):
            if (base / role[key]).resolve().parent != base:
                raise ValueError('Role artifact path escapes the frozen data directory')
        data_path = record(base / role['ids_path'], role['ids_sha256'])
        ids = np.load(data_path, mmap_mode='r', allow_pickle=False)
        if list(ids.shape) != role['shape'] or ids.dtype != np.uint16 or ids.shape[1] != 512:
            raise ValueError('Frozen role IDs have unexpected shape or dtype')
        boundary_path = record(base / role['boundaries_path'], role['boundaries_sha256'])
        boundaries = [json.loads(line) for line in boundary_path.read_text().splitlines()]
        token_path = record(directory / 'token-ce.npy', report['token_ce_sha256'])
        losses = np.load(token_path, allow_pickle=False)
        if losses.shape != (ids.shape[0], 511):
            raise ValueError('Token CE shape does not match evaluated IDs')
        document_path = record(directory / 'documents.json', report['documents_sha256'])
        documents[label] = validate_documents(losses, read_json(document_path), boundaries, length=512)
        if (report['supervised_tokens'] != losses.size or report['documents'] != len(documents[label])
                or role['documents'] != len(documents[label])):
            raise ValueError('Evaluation coverage counts disagree')
        means[label] = float(losses.mean(dtype=np.float64))
        if not math.isclose(means[label], report['ce_nats_per_supervised_token'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Reported mean CE differs from token artifact')
        reports[label] = report
    identity = validate_pair_identity(reports['A'], reports['B'], training['A'], training['B'])
    # The manifest digest binds tokenizer identity for both arms. Check retained
    # tokenizer files as well; no unselected role's token/document data is opened.
    tokenizer = manifest['tokenizer']
    for name, value in tokenizer['files'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected tokenizer snapshot filename')
        record(Path(manifest_path).parent / 'tokenizer' / name, value['sha256'])
    bootstrap = paired_bootstrap(documents['A'], documents['B'])
    if not math.isclose(bootstrap['candidate_minus_reference_nats_per_target'], means['B'] - means['A'],
                        rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError('Paired document point estimate differs from token-weighted mean gap')
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError('Analysis input changed while computing the paired interval')
    upper = bootstrap['one_sided_95_upper']
    return {'schema': 'rt-precision-paired-evaluation-v1', 'status': 'complete',
            'identity': identity, 'tokenizer': tokenizer, 'input_sha256': inputs,
            'evaluation_source_sha256': reports['A']['source_sha256'],
            'ce_nats_per_supervised_token': means, 'bootstrap': bootstrap,
            'margin_nats_per_supervised_token': MARGIN, 'upper_strictly_below_margin': upper < MARGIN,
            'primary_margin_assessment': ('within_margin_on_this_pilot_slice' if upper < MARGIN else 'margin_not_cleared')
                                         if identity['common_fp32_primary'] else 'native_comparison_is_secondary',
            'scope': 'One paired initialization/update/role on a fixed packed corpus slice; document resampling conditional on these trained weights',
            'qualification': 'No uncertainty over training seeds, no long-run or peak-LR clearance; document contexts share packed sequences. Native comparison includes inference-rounding effects.'}


def render_markdown(report):
    identity, interval = report['identity'], report['bootstrap']
    a, b = report['ce_nats_per_supervised_token']['A'], report['ce_nats_per_supervised_token']['B']
    return (f"# Paired RT evaluation: {identity['role']}, seed {identity['seed']}, update {identity['completed_updates']}\n\n"
            f"{identity['interpretation']}. A is protected BF16 training; B is legacy BF16 training.\n\n"
            f"A CE: **{a:.8f}**; B CE: **{b:.8f}** nats per supervised token. "
            f"B−A: **{interval['candidate_minus_reference_nats_per_target']:+.8f}**.\n\n"
            f"Paired token-weighted document bootstrap: one-sided 95% upper bound "
            f"**{interval['one_sided_95_upper']:+.8f}**, versus the frozen **{MARGIN:.3f}**-nat margin. "
            f"Assessment: `{report['primary_margin_assessment']}`.\n\n"
            f"{interval['resampled_documents']} documents, {interval['supervised_tokens']:,} targets, "
            f"{interval['resamples']:,} resamples, seed {interval['seed']}. "
            "Token/document hashes, source snapshots and pairing identities were verified.\n\n"
            f"{report['scope']}. {report['qualification']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-dir', type=Path, required=True)
    parser.add_argument('--b-dir', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh paired-analysis output directory')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-paired-evaluation-v1', 'status': 'running'}
    tracker = None
    started = time.monotonic()
    try:
        source_hashes = {}
        for source in (Path(__file__), ROOT / 'scripts/experiment_tracking.py'):
            name = str(source.relative_to(ROOT))
            value = source.read_bytes()
            source_hashes[name] = hashlib.sha256(value).hexdigest()
            target = args.output_dir / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
        report = analyze(args.a_dir, args.b_dir, args.data_manifest, args.protocol)
        report['runtime'] = {'numpy': np.__version__, 'execution': 'CPU-only; no torch/CUDA imports or model execution'}
        report['source_sha256'] = source_hashes
        for name, expected in source_hashes.items():
            if sha256(ROOT / name) != expected:
                raise ValueError('Analysis source changed during execution')
        if args.wandb_project:
            tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                    group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
            report['wandb'] = tracker.record
            tracker.start({'identity': report['identity'], 'margin': MARGIN,
                           'bootstrap_resamples': RESAMPLES, 'bootstrap_seed': BOOTSTRAP_SEED})
            tracker.log({'paired/A_ce': report['ce_nats_per_supervised_token']['A'],
                         'paired/B_ce': report['ce_nats_per_supervised_token']['B'],
                         'paired/B_minus_A': report['bootstrap']['candidate_minus_reference_nats_per_target'],
                         'paired/one_sided_95_upper': report['bootstrap']['one_sided_95_upper'],
                         'paired/margin': MARGIN})
            tracker.summary({'paired/primary_assessment': report['primary_margin_assessment'],
                             'paired/upper_strictly_below_margin': report['upper_strictly_below_margin']})
        (args.output_dir / 'results.md').write_text(render_markdown(report))
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker is not None:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__); raise
        finally:
            write_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'assessment': report['primary_margin_assessment'],
                      'B_minus_A': report['bootstrap']['candidate_minus_reference_nats_per_target'],
                      'one_sided_95_upper': report['bootstrap']['one_sided_95_upper']}), flush=True)


if __name__ == '__main__':
    main()
