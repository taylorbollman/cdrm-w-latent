"""Explicit audited NFR64 endpoints: unchanged K1..32 curves and finite residuals.

This is a weights-only observation, never a training resume. Historical source
inventories and the original component helper's NFR32 allowlist are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, _local_rng
from scripts import olmo_fbt_component_curves as previous
from scripts import olmo_nfr_kl_contract as training_contract
from scripts import olmo_nfr_kl_summary as pair_summary

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'olmo-nfr-endpoint-curves-v1'
SCOPE_SCHEMA = 'olmo-nfr-endpoint-scope-v1'
RESIDUAL_SCHEMA = 'olmo-nfr-finite-reference-residual-v1'
PROTOCOL = 'docs/reports/olmo-nfr-stability-128/protocol.md'
POLICY = {'arm': 'NFR', 'after_update': 64, 'num_passes': 32,
          'precision': 'common_fp32_no_jitter_v1', 'feedback_jitter': 0.,
          'panel_rows': 8, 'sequence_length': 1024, 'rt_layers': [0, 15],
          'comparison_passes': [4, 8], 'reference_pass': 32,
          'panel_membership_sha256': '407798544e9372034f76d12e60cfe601ab667b0306309ea4623f0de47c1914df'}
REFS = ('parent', 'control', 'reduced', 'audit', 'training_scope', 'f_resolved')
EXTRA_SOURCES = ('scripts/olmo_nfr_endpoint_curves.py', 'tests/test_nfr_endpoint_curves.py',
                 'scripts/olmo_fbt_component_curves.py', 'scripts/olmo_nfr_kl_summary.py',
                 'scripts/olmo_kl_continuation_summary.py', PROTOCOL)
legacy = previous.historical.contract.legacy
require = pair_summary.require


def validate_scope(scope):
    legacy.exact_fields(scope, ('schema', *POLICY, 'references', 'checkpoints',
                               'implementation_sources'), 'NFR endpoint scope')
    require(scope['schema'] == SCOPE_SCHEMA and all(scope[k] == v for k, v in POLICY.items()),
            'Only the explicit NFR64 K32 common-panel diagnostic is supported')
    legacy.exact_fields(scope['references'], REFS, 'endpoint references')
    legacy.exact_fields(scope['checkpoints'], ('control', 'reduced'), 'endpoint checkpoints')
    for ref in [*scope['references'].values(), *scope['checkpoints'].values()]:
        legacy.exact_fields(ref, ('path', 'sha256'), 'pinned endpoint reference')
        legacy.local_path(ref['path']); legacy.pin(ref['sha256'])
    require(bool(scope['implementation_sources']), 'Empty endpoint implementation authority')
    for name, digest in scope['implementation_sources'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source path')
        legacy.pin(digest)
    return scope


def merged_sources(f_sources, training_sources):
    sources = dict(f_sources)
    for name, pin in training_sources.items():
        require(name not in sources or sources[name] == pin, 'Historical source conflict: '+name)
        sources[name] = pin
    sources.update({name: sha256_file(ROOT/name) for name in EXTRA_SOURCES})
    return dict(sorted(sources.items()))


def authenticate_endpoint(reference, manifest, spec, *, case, manifest_sha256, sources):
    """Bind the audited branch to its unique published checkpoint and ordered plan."""
    weight = {'control': 1., 'reduced': .1}[case]
    require(manifest['world_size'] == 2 and manifest['counters']['optimizer_updates'] == 64
            and manifest['counters']['input_tokens'] == 64*524288,
            'Require saved two-rank NFR update64 exposure')
    require(reference['status'] == 'stopped_at_boundary' and reference['arm'] == 'NFR'
            and reference['schema'] == previous.kl_run.SCHEMA
            and reference['declaration_sha256'] == previous.historical.DECL
            and reference['resolved_sha256'] == previous.historical.RESOLVED
            and reference['configuration'] == manifest['metadata']['configuration']
            and reference['source_fingerprint'] == manifest['metadata']['source_fingerprint']
            and reference['sources'] == sources, 'Endpoint checkpoint/report lineage differs')
    publications = [p for p in reference['published_checkpoints'] if p['manifest_sha256'] == manifest_sha256]
    require(len(publications) == 1 and publications[0]['state'] == manifest['state'],
            'Endpoint checkpoint lacks its unique verified publication')
    config = reference['configuration']; identity = config['execution_identity']; payload = identity['payload']
    require(identity['sha256'] == legacy.digest(payload)
            and payload['resolved_contract_sha256'] == spec['resolved']['contract_sha256']
            and payload['arm'] == 'NFR' and payload['sources'] == sources
            and payload['plan'] == spec['plan'], 'Endpoint execution identity differs')
    expected_cursor = spec['plan']['updates'][63]['next_cursor']
    require(len(manifest['rank_cursors']) == 2 and all(
        cursor.get('rank') == rank and cursor.get('world_size') == 2
        and cursor.get('physical_batch_per_rank') == 12 and cursor.get('cursor') == expected_cursor
        for rank, cursor in enumerate(manifest['rank_cursors'])), 'Endpoint ordered cursor differs')
    branch = reference['branch']
    require(config.get('objective_branch') == branch and branch['parent_update'] == 32
            and branch['review_stop'] == 64 and branch['kl_weight'] == weight
            and reference['parent_manifest_sha256'] == branch['parent_manifest_sha256'],
            'Endpoint KL branch differs')
    recipe = previous.declared_recipe(spec['recipe'], kl_weight=weight,
        parent_manifest_sha256=branch['parent_manifest_sha256'])
    require(config['recipe'] == payload['recipe'] == branch['recipe_as_declared'] == recipe,
            'Endpoint operative recipe differs')
    return weight


def load_context(scope, *, case, verify_state):
    validate_scope(scope)
    refs = scope['references']
    loaded = {name: legacy.read_json(legacy.local_path(ref['path']), ref['sha256'], limit=128*1024**2)
              for name, ref in refs.items()}
    pair_summary.check_authorities(loaded['control'], loaded['reduced'], loaded['parent'], loaded['audit'], refs)
    require(refs['training_scope']['sha256'] == loaded['audit']['inputs']['scope']['sha256']
            and loaded['control']['nfr_scope']['declaration'] == loaded['training_scope'],
            'Pair audit and original NFR training scope differ')
    training_contract.validate_scope(loaded['training_scope'])
    saved_sources = training_contract.source_hashes(legacy.local_path(refs['training_scope']['path']))
    require(loaded['control']['sources'] == saved_sources, 'Historical 215-source authority changed')
    spec = previous.historical.diagnostic_spec('NFR')
    plan, f_sources = previous.shared_panel(spec, legacy.local_path(refs['f_resolved']['path']), refs['f_resolved']['sha256'])
    require(plan['panel']['fixed_plan']['updates'][0]['membership_sha256'] == scope['panel_membership_sha256'],
            'Original eight-row panel membership differs')
    require(scope['implementation_sources'] == merged_sources(f_sources, saved_sources),
            'Endpoint diagnostic source authority differs')
    checkpoint = scope['checkpoints'][case]
    manifest = inspect_distributed_checkpoint(legacy.local_path(checkpoint['path']),
        expected_manifest_sha256=checkpoint['sha256'], verify_state=verify_state)
    weight = authenticate_endpoint(loaded[case], manifest, spec, case=case,
        manifest_sha256=checkpoint['sha256'], sources=saved_sources)
    return spec, plan, manifest, weight


def hidden_residual_sums(states, valid, *, comparisons=(4, 8), reference_pass=32):
    """Direct finite-state residuals. K32 is a finite reference, not exact online."""
    require(len(states) == reference_pass and all(1 <= k <= reference_pass for k in comparisons),
            'Finite reference/pass count differs')
    reference = states[reference_pass-1].float()
    require(reference.ndim == 3 and valid.shape == reference.shape[:2] and valid.dtype == torch.bool,
            'Finite reference tensor/mask differs')
    position = torch.arange(valid.shape[1], device=valid.device)[None, :]
    reference_square = reference.square().mean(-1)
    records = []
    for k in comparisons:
        require(states[k-1].shape == reference.shape, 'Compared hidden shape differs')
        difference = (states[k-1].float()-reference).square().mean(-1)
        regions = {}
        masks = {'all': valid, 'tail_128': valid & (position >= max(0, valid.shape[1]-128)),
                 'beyond_guaranteed_prefix': valid & (position >= k)}
        for name, active in masks.items():
            regions[name] = {'positions': int(active.sum()),
                'difference_square_sum': float(difference[active].double().sum()),
                'reference_square_sum': float(reference_square[active].double().sum())}
        records.append({'pass': k, 'reference_pass': reference_pass, 'regions': regions})
    return {'schema': RESIDUAL_SCHEMA, 'comparisons': records}


def summarize_residuals(rows):
    require(bool(rows), 'No finite residual rows')
    keys = [(r['pass'], r['reference_pass']) for r in rows[0]['comparisons']]
    require(all(r['schema'] == RESIDUAL_SCHEMA and
        [(c['pass'], c['reference_pass']) for c in r['comparisons']] == keys for r in rows),
        'Residual references differ across rows')
    comparisons = []
    for index, (k, ref) in enumerate(keys):
        regions = {}
        for name in ('all', 'tail_128', 'beyond_guaranteed_prefix'):
            values = [row['comparisons'][index]['regions'][name] for row in rows]
            require(all(type(v['positions']) is int and v['positions'] >= 0 and
                all(math.isfinite(v[q]) and v[q] >= 0 for q in ('difference_square_sum', 'reference_square_sum'))
                for v in values), 'Malformed residual sums')
            n = sum(v['positions'] for v in values)
            delta, reference = (math.fsum(v[q] for v in values)
                                for q in ('difference_square_sum', 'reference_square_sum'))
            regions[name] = {'positions': n, 'difference_square_sum': delta, 'reference_square_sum': reference,
                'delta_rms': math.sqrt(delta/n) if n else None,
                'reference_rms': math.sqrt(reference/n) if n else None,
                'relative_l2': math.sqrt(delta/max(reference, n*1e-24)) if n else None}
        comparisons.append({'pass': k, 'reference_pass': ref, 'regions': regions})
    return {'schema': RESIDUAL_SCHEMA, 'comparisons': comparisons,
        'scope': 'Hidden states against finite K32, not exact online; pooled coordinate-mean squared sums; '
                 'beyond_guaranteed_prefix excludes first K positions, not K-1 consecutive-pass prefix'}


def endpoint_probe_batch(model, batch, recipe):
    captured = []
    def observe(_module, _args, output):
        captured.append(hidden_residual_sums(output.pass_hidden_states, batch.valid_mask))
    handle = model.backbone.register_forward_hook(observe)
    try:
        result = previous.component_probe_batch(model, batch, recipe, passes=32)
    finally:
        handle.remove()
    require(len(captured) == 1, 'Endpoint observation requires one canonical K32 forward')
    return result, captured[0]


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); tracker = None
    report = {'schema': SCHEMA, 'status': 'preflight', 'arm': 'NFR', 'after_update': 64,
        'case': args.case, 'num_passes': 32, 'optimizer_updates_performed': 0, 'rows': [],
        'predictor_execution': 'not_called; saved tensors retained',
        'scope': 'Audited NFR64 endpoint; common packed eight-row FP32/no-jitter curves; '
                 'finite K32 reference, not exact online or BF16/quality acceptance'}
    def publish(event=None):
        report['elapsed_seconds'] = time.monotonic()-started
        if event is not None:
            report.setdefault('progress', []).append(event); print(json.dumps(event), flush=True)
        write_json(args.output_dir/'report.json', report)
    try:
        scope = legacy.read_json(args.scope, args.scope_sha256)
        spec, plan, manifest, weight = load_context(scope, case=args.case, verify_state=not args.preflight_only)
        scope_path = args.scope.resolve()
        require(scope_path.is_relative_to(ROOT), 'Keep explicit endpoint scope inside project')
        sources = dict(scope['implementation_sources']) | {str(scope_path.relative_to(ROOT)): args.scope_sha256}
        report.update(kl_weight=weight, endpoint_scope={'sha256': args.scope_sha256, 'declaration': scope},
            input_authorities=scope['references'] | {'checkpoint': scope['checkpoints'][args.case], 'state': manifest['state']},
            sources=sources, membership_sha256=scope['panel_membership_sha256'],
            index_manifest_sha256=plan['panel']['index_manifest_sha256'], probe_policy=plan)
        for name, pin in sources.items():
            require(sha256_file(ROOT/name) == pin, 'Diagnostic source changed: '+name)
            target = args.output_dir/'source-snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
        batches = previous.materialize_panel(spec, plan)
        report['batch_tensor_sha256'] = [previous.tree_digests(vars(batch)) for batch in batches]
        if args.preflight_only:
            report['status'] = 'preflight_validated'
            publish({'phase': 'preflight_complete', 'gpu_execution': False, 'state_bytes_verified': False}); return
        previous.configure_determinism(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.set_num_threads(8)
        report['runtime'] = previous.require_container_gpu()
        require(torch.cuda.device_count() == 1 and not torch.distributed.is_initialized(),
                'Expose exactly one assigned GPU to this standalone diagnostic')
        tracker = previous.OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
            output_dir=args.output_dir, group='nfr-endpoint-curves', name=args.output_dir.name,
            preserve_state=previous.preserve_local_rng)
        tracker.start({'arm': 'NFR', 'checkpoint_update': 64, 'kl_weight': weight, 'max_passes': 32,
                       'scope_sha256': args.scope_sha256, 'optimizer_updates_performed': 0})
        for prefix in ('curves/*', 'finite_reference/*'):
            tracker._call('pass axis', lambda prefix=prefix: tracker._run.define_metric(prefix, step_metric='total_pass'))
        report['tracking'] = tracker.record; publish({'phase': 'constructing_model'})
        model, _, _ = previous.historical.construct(spec, torch.device('cuda:0'))
        previous.import_saved_weights(model, spec, manifest,
            legacy.local_path(scope['checkpoints'][args.case]['path']), kl_weight=weight)
        before = previous.state_pins(model); rng = previous.tree_digests(_local_rng(torch.device('cuda:0'), None))
        with previous.evaluation_runtime(model) as preservation:
            for row_index, batch in enumerate(batches):
                row_started = time.perf_counter()
                result, residuals = endpoint_probe_batch(model, batch.to('cuda:0'), spec['recipe'])
                record = {'row_index': row_index, 'seconds': time.perf_counter()-row_started,
                          'result': result, 'finite_residuals': residuals}
                report['rows'].append(record); write_json(args.output_dir/f'row-{row_index:02d}.json', record)
                publish({'phase': 'row_completed', 'row_index': row_index, 'seconds': record['seconds']})
        report['preservation'] = preservation
        counts = plan['panel']['fixed_plan']['updates'][0]['counts']
        report['result'] = previous.probe.summarize([r['result'] for r in report['rows']],
            expected_tokens=counts['valid_tokens'], expected_ce_targets=counts['ce_targets'])
        report['finite_residuals'] = summarize_residuals([r['finite_residuals'] for r in report['rows']])
        residual_by_pass = {r['pass']: r['regions'] for r in report['finite_residuals']['comparisons']}
        for p in report['result']['passes']:
            metrics = previous.scalar_metrics({n: r['metrics'] for n, r in p['regions'].items()}, 'curves')
            if p['pass'] in residual_by_pass:
                metrics.update(previous.scalar_metrics(residual_by_pass[p['pass']], 'finite_reference'))
            tracker.log({'total_pass': p['pass'], **metrics}, step=p['pass'])
        report['weights_unchanged'] = before == previous.state_pins(model)
        report['rng_unchanged'] = rng == previous.tree_digests(_local_rng(torch.device('cuda:0'), None))
        report['gradient_buffers_absent'] = all(p.grad is None for p in model.parameters())
        require(all(report[k] for k in ('weights_unchanged', 'rng_unchanged', 'gradient_buffers_absent')),
                'Endpoint probe changed observed model state')
        require(all(sha256_file(ROOT/n) == pin for n, pin in sources.items()), 'Endpoint source changed during observation')
        report['status'] = 'completed'; tracker.summary({'completed': True, 'optimizer_updates_performed': 0})
        tracker.finish(succeeded=True); publish({'phase': 'completed'})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if tracker is not None: tracker.finish(succeeded=False)
        publish(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--scope-sha256', required=True)
    parser.add_argument('--case', choices=('control', 'reduced'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv); legacy.pin(args.scope_sha256)
    run(args)


if __name__ == '__main__':
    main()
