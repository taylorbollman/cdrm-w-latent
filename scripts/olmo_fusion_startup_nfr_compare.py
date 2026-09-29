#!/usr/bin/env python3
"""CPU-only endpoint geometry for the fixed update4 -> update20 NFR pair."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA
from scripts.olmo_campaign_precision_components import gradient_geometry
from scripts.olmo_campaign_probe import component
from scripts.olmo_fusion_startup_nfr_updates import BF16, DeltaView, SCHEMA as ORIGIN_SCHEMA, expected_counters
from scripts.olmo_fusion_startup_nfr_continue import PLAN, SCHEMA as CONTINUATION_SCHEMA
from scripts.olmo_lm_common import tree_digests

SCHEMA = 'olmo-fusion-startup-nfr-endpoint-comparison-v1'
TERMS = ('ce', 'latent', 'kl')
COMPONENTS = ('all', 'backbone', 'fusion', 'predictor')
DEPENDENCIES = (
    'scripts/olmo_campaign_precision_components.py', 'scripts/olmo_campaign_probe.py',
    'scripts/olmo_fusion_startup_nfr_updates.py', 'scripts/olmo_fusion_startup_nfr_continue.py',
    'scripts/olmo_lm_common.py', 'cdrm/pretrained/lm_training.py')
OWN = ('scripts/olmo_fusion_startup_nfr_compare.py', 'tests/test_fusion_startup_nfr_compare.py',
       'docs/reports/olmo-fusion-startup/nfr-comparison-protocol.md')


def source_hashes():
    return {name: sha256_file(ROOT/name) for name in (*DEPENDENCIES, *OWN)}


def signature(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require an ordinary immutable local file')
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def read_report(path, digest):
    before = signature(path)
    if before[2] > 128*1024*1024 or sha256_file(path) != digest:
        raise ValueError('Report differs from bounded independent SHA256 pin')
    report = json.loads(Path(path).read_text())
    if before != signature(path):
        raise ValueError('Report changed while reading')
    return report


def retained_record(report, step, *, trajectory=None):
    rows = [r for r in report['checkpoints'] if r['optimizer_updates'] == step
            and (trajectory is None or r.get('trajectory') == trajectory)]
    if len(rows) != 1:
        raise ValueError('Missing or ambiguous required retained checkpoint')
    row, gcs = rows[0], rows[0].get('gcs', {})
    if (not gcs.get('generation') or row['sha256'] != gcs.get('sha256')
            or row['size_bytes'] != gcs.get('size_bytes')
            or any(gcs.get('verification', {}).get(k) is not True
                   for k in ('download_sha256', 'server_md5', 'server_size', 'sha256_metadata'))):
        raise ValueError('Checkpoint lacks matching full-byte cloud retention')
    return row


def require(condition, name, checks):
    checks[name] = bool(condition)
    if not condition:
        raise ValueError('Comparison control failed: '+name)


def report_controls(origin, fp, bf, *, current_sources=None, origin_sha256=None):
    checks = {}
    require(origin.get('schema') == ORIGIN_SCHEMA and origin.get('status') == 'passed_bounded_functionality'
            and origin.get('passed') is True and origin.get('integrity') and all(origin['integrity'].values())
            and origin.get('optimizer_calls') == 8, 'completed_original_four_update_pair', checks)
    origin_receipt = retained_record(origin, 4, trajectory=BF16)
    for name, report in (('fp32', fp), ('bf16_mixed', bf)):
        require(report.get('schema') == CONTINUATION_SCHEMA and report.get('status') == 'completed_segment'
                and report.get('passed') is True and report.get('integrity') and all(report['integrity'].values())
                and report.get('plan') == PLAN and report.get('precision') == name
                and report.get('starting_update') == 4 and report.get('physical_optimizer_updates') == 16
                and report['final_counters']['optimizer_updates'] == 20,
                name+'/completed_fixed_continuation', checks)
        require(report['configuration']['origin_checkpoint_sha256'] == origin_receipt['sha256']
                and report['configuration']['origin_trajectory'] == BF16
                and report['origin_boundary'] == origin['final_boundary_pins'][BF16]
                and report['configuration']['origin_boundary'] == report['origin_boundary'],
                name+'/same_original_bf16_boundary', checks)
        require(all(report['sources'].get(k) == v for k, v in origin['sources'].items()),
                name+'/original_sources_preserved', checks)
        require(report['configuration']['sources'] == report['sources']
                and report['source_fingerprint']['sources'] == report['sources']
                and report['source_fingerprint']['origin_checkpoint_sha256'] == origin_receipt['sha256'],
                name+'/source_configuration_identity', checks)
        require(report['configuration']['recipe'] == origin['recipe']
                and report['configuration']['data_manifest_sha256'] == origin['data_manifest_sha256']
                and report['configuration']['fixture_sha256'] == origin['fixture_sha256']
                and report['runtime'] == report['configuration']['runtime'] == origin['runtime']
                and report['determinism'] == report['configuration']['determinism'] == origin['determinism']
                and report['source_fingerprint']['data_manifest_sha256'] == origin['data_manifest_sha256']
                and report['source_fingerprint']['fixture_sha256'] == origin['fixture_sha256']
                and (origin_sha256 is None or report['source_fingerprint']['nfr_report_sha256'] == origin_sha256),
                name+'/same_recipe_data_fixture_and_report_authority', checks)
        require(report['actual_execution_mode'] == {'beta': 1., 'document_policy': 'isolated-v1', 'enabled': True,
                'feedback_jitter': .02, 'first_pass_policy': 'configured-rt-v1', 'num_passes': 4,
                'rt_mode': {'alpha': 1., 'selected_layers': [0, 15]}}, name+'/actual_full_nfr_mode', checks)
        require(all(report['starting_boundary'][k] == report['origin_boundary'][k]
                    for k in report['origin_boundary'] if k != 'scheduler')
                and report['starting_boundary']['scheduler'] == report['configuration']['schedule_fork']['new_scheduler']
                and all(report['configuration']['schedule_fork']['checks'].values()),
                name+'/only_declared_schedule_fork', checks)
        require([r['update'] for r in report['updates']] == list(range(5, 21))
                and [r['source_training_index'] for r in report['updates']] == list(range(148, 164)),
                name+'/sixteen_exact_data_selections', checks)
        metadata = report['configuration']['training_metadata']
        selections = report['configuration']['training_data_selections']
        require(len(metadata) == len(selections) == 20
                and metadata[:4] == origin['training_metadata']
                and selections[:4] == origin['training_data_selections']
                and [r['source_training_index'] for r in selections] == list(range(144, 164))
                and report['final_counters'] == asdict(expected_counters(metadata, 20))
                and report['final_cursor'] == selections[-1]['next_cursor'],
                name+'/actual_token_and_cursor_clocks', checks)
        require([r['update'] for r in report['evaluations']] == [4, 12, 20], name+'/dev_clocks', checks)
        for e in report['evaluations']:
            require(all(e['checks'].values()) and e['precision'] == 'fp32_math_eager'
                    and set(e['per_pass']) == {'pass_'+str(i) for i in range(4)}
                    and e['per_pass_observer']['counts_exact'] and e['per_pass_observer']['method_restored'],
                    name+'/dev'+str(e['update'])+'/read_only_fp32', checks)
            for term in TERMS:
                weights = [.5, 1/6, 1/6, 1/6] if term == 'ce' else [.25]*4
                value = sum(e['per_pass']['pass_'+str(i)]['loss_means'][term]*weights[i] for i in range(4))
                require(math.isclose(value, e['loss_means'][term], rel_tol=1e-6, abs_tol=1e-6)
                        and all(e['per_pass']['pass_'+str(i)]['counts'] == e['counts'] for i in range(4)),
                        name+'/dev'+str(e['update'])+'/'+term+'/weighted_pass_accounting', checks)
        require(all(report['evaluations'][0][k] == origin['evaluations']['4'][BF16][k]
                    for k in ('loss_sums', 'loss_means', 'counts', 'combined_objective')),
                name+'/common_origin_dev_exact', checks)
        retained_record(report, 20)
    require(fp['sources'] == bf['sources'], 'same_continuation_sources', checks)
    require(fp['source_fingerprint'] == bf['source_fingerprint'], 'same_source_fingerprint', checks)
    require({k: v for k, v in fp['configuration'].items() if k != 'precision'} ==
            {k: v for k, v in bf['configuration'].items() if k != 'precision'}, 'only_precision_contract_differs', checks)
    require(fp['starting_boundary'] == bf['starting_boundary'], 'same_complete_start_boundary', checks)
    if current_sources is not None:
        require(all(fp['sources'].get(k) == current_sources[k] for k in DEPENDENCIES),
                'imported_analysis_dependencies_match_frozen_execution', checks)
    for a, b in zip(fp['updates'], bf['updates']):
        require(all(a[k] == b[k] for k in ('update', 'source_training_index', 'input_pins', 'counters', 'lr_used', 'lr_next'))
                and all(a['metrics'][k] == b['metrics'][k] for k in ('counts', 'microbatches', 'documents', 'input_tokens')),
                'update'+str(a['update'])+'/same_data_and_schedule', checks)
        for r in (a, b):
            meta = fp['configuration']['training_metadata'][r['update']-1]
            require(all(r['metrics'][k] == meta[k] for k in ('counts', 'microbatches', 'documents', 'input_tokens'))
                    and all(math.isfinite(v) for v in (r['metrics']['objective'], *r['metrics']['loss_sums'].values(),
                        r['gradient_norm_before_clip'], r['clip_scale'], *r['raw_gradient_norms'].values(),
                        *r['master_parameter_norms'].values())),
                    r['path']+'/update'+str(r['update'])+'/finite_exact_counts', checks)
    return checks


def payload_boundary(payload):
    return tree_digests({'schema': ORIGIN_SCHEMA, 'model': payload['model'], 'optimizer': payload['optimizer'],
        'scheduler': payload['scheduler'], 'rng': payload['rng'], 'counters': payload['counters'],
        'ownership': payload['optimizer_ownership'], 'layout': payload['parameter_layout'], 'modes': payload['module_training']})


def parameter_and_moment_views(payload):
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('model_type') != 'cdrm.pretrained.fbt_training.FBTNextLatLM':
        raise ValueError('Unexpected model checkpoint schema/type')
    parameters, aliases = {}, set()
    for row in payload['parameter_layout']:
        name = row['name']
        if name in parameters or row['requires_grad'] is not True or row['dtype'] != 'torch.float32' or not row['aliases'] or row['aliases'][0] != name:
            raise ValueError('Unexpected trainability/canonical parameter ownership')
        value = payload['model'][name]
        if value.device.type != 'cpu' or str(value.dtype) != row['dtype'] or list(value.shape) != row['shape'] or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite or incompatible CPU master parameter')
        for alias in row['aliases']:
            if alias in aliases or not torch.equal(value, payload['model'][alias]):
                raise ValueError('Duplicate or mismatched tied alias')
            aliases.add(alias)
        parameters[name] = value
    moments = {key: {} for key in ('exp_avg', 'exp_avg_sq')}
    groups, ownership = payload['optimizer']['param_groups'], payload['optimizer_ownership']
    if len(groups) != len(ownership):
        raise ValueError('Optimizer group inventory differs')
    seen = set()
    for group, names in zip(groups, ownership):
        if group.get('param_names') != names or len(group['params']) != len(names):
            raise ValueError('Optimizer names/order differ')
        for identifier, name in zip(group['params'], names):
            if identifier in seen or name not in parameters or name in moments['exp_avg']:
                raise ValueError('Missing, duplicate or foreign optimizer owner')
            seen.add(identifier)
            state = payload['optimizer']['state'][identifier]
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != payload['counters']['optimizer_updates']:
                raise ValueError('Adam step/state inventory differs')
            for key in moments:
                value = state[key]
                if (value.device.type != 'cpu' or value.dtype != torch.float32 or value.shape != parameters[name].shape
                        or not bool(torch.isfinite(value).all()) or (key == 'exp_avg_sq' and bool((value < 0).any()))):
                    raise ValueError('Invalid Adam moment')
                moments[key][name] = value
    if seen != set(payload['optimizer']['state']) or parameters.keys() != moments['exp_avg'].keys():
        raise ValueError('Incomplete Adam ownership')
    return {'model': parameters, **moments}


def load_checkpoint(path, record, *, configuration, fingerprint, expected_boundary):
    path = Path(path)
    before = signature(path)
    if before[2] != record['size_bytes'] or sha256_file(path) != record['sha256'] or before != signature(path):
        raise ValueError('Checkpoint bytes differ from the independently pinned retained receipt')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload['configuration'] != configuration or payload['source_fingerprint'] != fingerprint
            or payload['counters']['optimizer_updates'] != record['optimizer_updates']
            or payload_boundary(payload) != expected_boundary or before != signature(path)):
        raise ValueError('Checkpoint contract or complete serialized boundary differs')
    return payload, parameter_and_moment_views(payload), before


def origin_norms(views):
    result = {}
    for label, values in views.items():
        squares = dict.fromkeys(COMPONENTS, 0.)
        for name, tensor in values.items():
            value = float(tensor.double().square().sum())
            squares['all'] += value
            squares[component(name)] += value
        result[label] = {name: math.sqrt(value) for name, value in squares.items()}
    return result


def endpoint_geometry(origin, fp, bf, *, observe=lambda label, value: None):
    result = {}
    for label in ('model', 'exp_avg', 'exp_avg_sq'):
        result[label] = gradient_geometry(bf[label], fp[label])
        observe(label, result[label])
        result[label+'_change_from_common_origin'] = gradient_geometry(DeltaView(bf[label], origin[label]), DeltaView(fp[label], origin[label]))
        observe(label+'_change_from_common_origin', result[label+'_change_from_common_origin'])
    return result


def scalar_rows(fp, bf):
    training = []
    for report in (fp, bf):
        for r in report['updates']:
            training.append({'precision': report['precision'], 'update': r['update'],
                'means': {term: r['metrics']['loss_sums'][term]/r['metrics']['counts'][term] for term in TERMS},
                'combined': r['metrics']['objective'], 'raw_gradient_norms': r['raw_gradient_norms'], 'clip_scale': r['clip_scale']})
    return {'training': training, 'common_fp32_dev': {r['precision']: r['evaluations'] for r in (fp, bf)}}


def plot_rows(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex='row')
    for col, term in enumerate(TERMS):
        for precision, color in (('fp32', 'tab:blue'), ('bf16_mixed', 'tab:orange')):
            train = [r for r in rows['training'] if r['precision'] == precision]
            dev = rows['common_fp32_dev'][precision]
            axes[0, col].plot([r['update'] for r in train], [r['means'][term] for r in train], color=color, label=precision)
            axes[1, col].plot([r['update'] for r in dev], [r['loss_means'][term] for r in dev], 'o-', color=color, label=precision)
        axes[0, col].set_title(term.upper()); axes[1, col].set_xlabel('Completed optimizer updates')
        for row in range(2): axes[row, col].grid(alpha=.25)
    axes[0, 0].set_ylabel('Training mean (different batches)'); axes[1, 0].set_ylabel('Fixed dev mean, evaluated in FP32')
    axes[0, 0].legend(); fig.suptitle('Full NFR: both paths start from the same BF16 update4 state')
    fig.tight_layout(); files = []
    for suffix in ('png', 'pdf'):
        name = output/('loss-trajectories.'+suffix); fig.savefig(name, dpi=160); files.append(name)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for col, term in enumerate(TERMS):
        for precision, style in (('fp32', '-'), ('bf16_mixed', '--')):
            dev = rows['common_fp32_dev'][precision]
            for index in range(4):
                axes[col].plot([r['update'] for r in dev], [r['per_pass']['pass_'+str(index)]['loss_means'][term] for r in dev],
                    linestyle=style, marker='o', color='C'+str(index), label=f'pass {index+1}, {precision}')
        axes[col].set_title(term.upper()); axes[col].set_xlabel('Completed optimizer updates'); axes[col].grid(alpha=.25)
    axes[0].set_ylabel('Unweighted per-pass dev mean, FP32 evaluation'); axes[2].legend(fontsize=7)
    fig.suptitle('Same held-out documents: pass color, training-precision line style'); fig.tight_layout()
    for suffix in ('png', 'pdf'):
        name = output/('dev-per-pass.'+suffix); fig.savefig(name, dpi=160); files.append(name)
    plt.close(fig)
    return {path.name: sha256_file(path) for path in files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('origin-report', 'fp32-report', 'bf16-report'):
        parser.add_argument('--'+name, type=Path, required=True); parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent') or torch.cuda.is_available():
        raise RuntimeError('Use the project container with CDRM_DOCKER_GPUS=none; this is CPU analysis only')
    for name in ('origin_report_sha256', 'fp32_report_sha256', 'bf16_report_sha256'):
        value = getattr(args, name)
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value): parser.error('Require independent lowercase SHA256 pins')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT): parser.error('Keep evidence under the persistent project checkout')
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4); started = time.monotonic()
    sources = source_hashes()
    for name in sources:
        target = output/'source-snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    result = {'schema': SCHEMA, 'status': 'running', 'sources': sources,
        'qualification': 'CPU descriptive endpoint comparison after16 conditional updates from common BF16 update4. No new model execution, quality claim, BF16 tolerance or production clearance.'}
    def persist(stage):
        result.update(stage=stage, elapsed_seconds=time.monotonic()-started); write_json(output/'report.json', result)
    try:
        reports = [read_report(getattr(args, name+'_report'), getattr(args, name+'_report_sha256')) for name in ('origin', 'fp32', 'bf16')]
        origin, fp, bf = reports
        result['checks'] = report_controls(origin, fp, bf, current_sources=sources, origin_sha256=args.origin_report_sha256)
        result['report_pins'] = {name: getattr(args, name+'_report_sha256') for name in ('origin', 'fp32', 'bf16')}
        records = [retained_record(origin, 4, trajectory=BF16), retained_record(fp, 20), retained_record(bf, 20)]
        common = origin['checkpoint_configuration']
        configs = [{**common, 'trajectory': BF16}, fp['configuration'], bf['configuration']]
        fingerprints = [{'checkpoint_sha256': origin['origin_checkpoint_sha256'], 'base': common['source_checkpoint'], 'sources': origin['sources']},
                        fp['source_fingerprint'], bf['source_fingerprint']]
        boundaries = [origin['final_boundary_pins'][BF16], fp['final_boundary'], bf['final_boundary']]
        payloads, views, files = [], [], []
        for name, record, configuration, fingerprint, boundary in zip(('origin', 'fp32', 'bf16'), records, configs, fingerprints, boundaries):
            persist('load/'+name)
            path = getattr(args, name+'_report').parent/Path(record['path']).name
            payload, view, sig = load_checkpoint(path, record, configuration=configuration, fingerprint=fingerprint, expected_boundary=boundary)
            payloads.append(payload); views.append(view); files.append((path, sig))
        require(all(payloads[i]['parameter_layout'] == payloads[0]['parameter_layout'] and
                    payloads[i]['optimizer_ownership'] == payloads[0]['optimizer_ownership'] and
                    payloads[i]['optimizer_descriptor'] == payloads[0]['optimizer_descriptor'] and
                    payloads[i]['scheduler_descriptor'] == payloads[0]['scheduler_descriptor'] for i in (1, 2)),
                'endpoint_ownership_and_descriptors', result['checks'])
        require(payloads[1]['scheduler'] == payloads[2]['scheduler'] and payloads[1]['data_cursor'] == payloads[2]['data_cursor']
                and tree_digests(payloads[1]['rng']) == tree_digests(payloads[2]['rng']), 'matching_final_clocks_cursor_rng', result['checks'])
        result['checkpoint_receipts'] = dict(zip(('origin', 'fp32', 'bf16'), records))
        persist('origin_scalar_norms'); result['origin_norms'] = origin_norms(views[0])
        result['geometry'] = {}
        def observed(label, value):
            result['geometry'][label] = value; persist('geometry/'+label)
        persist('geometry/start'); endpoint_geometry(*views, observe=observed)
        result['scalar_rows'] = scalar_rows(fp, bf); persist('plots')
        result['plot_sha256'] = plot_rows(result['scalar_rows'], output)
        require(all(signature(path) == sig for path, sig in files), 'checkpoint_files_unchanged', result['checks'])
        require(all(sha256_file(getattr(args, name+'_report')) == getattr(args, name+'_report_sha256') for name in ('origin', 'fp32', 'bf16')),
                'report_files_unchanged', result['checks'])
        require(source_hashes() == sources, 'analysis_sources_unchanged', result['checks'])
        result.update(status='passed_descriptive_comparison', passed=True); persist('complete')
    except BaseException as error:
        result.update(status='failed', passed=False, error={'type': type(error).__name__, 'message': str(error)}); persist('failed'); raise
    print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'report_sha256': sha256_file(output/'report.json')}))


if __name__ == '__main__': main()
