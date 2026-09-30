"""Saved reduced-KL NFR128 observer, reusing the frozen NFR64 curve math.

This explicit terminal-checkpoint scope adds no training or new measurement
definitions. K32 remains a finite reference rather than exact online inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, _local_rng
from scripts import olmo_nfr_endpoint_curves as endpoint
from scripts import olmo_nfr_128_contract as continuation

ROOT = endpoint.ROOT
legacy, previous, require = endpoint.legacy, endpoint.previous, endpoint.require
SCHEMA = 'olmo-nfr-final-curves-v1'
SCOPE_SCHEMA = 'olmo-nfr-final-curve-scope-v1'
PROTOCOL = 'docs/reports/olmo-nfr-stability-128/final-probe-protocol.md'
POLICY = endpoint.POLICY | {'after_update': 128, 'kl_weight': .1}
REFS = ('endpoint64_report', 'continuation_scope', 'continuation_resolved',
        'terminal_report', 'terminal_publication')
EXTRA_SOURCES = ('scripts/olmo_nfr_final_curves.py', 'tests/test_nfr_final_curves.py', PROTOCOL)


def validate_scope(scope):
    legacy.exact_fields(scope, ('schema', *POLICY, 'references', 'checkpoint',
                               'implementation_sources'), 'NFR128 curve scope')
    require(scope['schema'] == SCOPE_SCHEMA and all(scope[k] == v for k, v in POLICY.items()),
            'Only reduced-KL NFR128 on the unchanged K32 panel is supported')
    legacy.exact_fields(scope['references'], REFS, 'NFR128 curve references')
    for ref in [*scope['references'].values(), scope['checkpoint']]:
        legacy.exact_fields(ref, ('path', 'sha256'), 'NFR128 curve authority')
        legacy.local_path(ref['path']); legacy.pin(ref['sha256'])
    require(bool(scope['implementation_sources']), 'Missing final diagnostic source authority')
    for name, pin in scope['implementation_sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source path')
        legacy.pin(pin)
    return scope


def final_sources(endpoint_sources, training_sources):
    sources = dict(endpoint_sources)
    for name, pin in training_sources.items():
        require(name not in sources or sources[name] == pin, 'Historical source conflict: '+name)
        sources[name] = pin
    sources.update({name: sha256_file(ROOT/name) for name in EXTRA_SOURCES})
    return dict(sorted(sources.items()))


def authenticate_final(reference, manifest, publication, resolution, spec, *, manifest_sha256):
    """Require the completed, synced, published128 continuation under exact metadata."""
    require(reference['schema'] == previous.kl_run.SCHEMA and reference['arm'] == 'NFR'
            and reference['scale'] == 'native' and reference['status'] == 'completed_plan'
            and reference['segment_completed'] is True and reference['plan_completed'] is True
            and reference['segment_stop_after'] == 128 and reference['last_verified_cloud_update'] == 128
            and reference['wandb']['status'] == 'synced', 'Require completed/cloud128/synced continuation')
    require(manifest['world_size'] == 2 and manifest['counters']['optimizer_updates'] == 128
            and manifest['counters']['input_tokens'] == 128*524288
            and reference['final_counters'] == manifest['counters'], 'Final checkpoint exposure differs')
    require(manifest['manifest_sha256'] == manifest_sha256
            and reference['configuration'] == manifest['metadata']['configuration'] == resolution['configuration']
            and reference['sources'] == resolution['sources']
            and reference['source_fingerprint'] == manifest['metadata']['source_fingerprint']
            and reference['declaration_sha256'] == previous.historical.DECL
            and reference['resolved_sha256'] == previous.historical.RESOLVED,
            'Final checkpoint/configuration/source lineage differs')
    continuation.verify_publication(publication, manifest)
    require([p for p in reference['published_checkpoints'] if p['manifest_sha256'] == manifest_sha256]
            == [publication], 'Final checkpoint lacks its unique verified publication')
    config = reference['configuration']; identity = config['execution_identity']; payload = identity['payload']
    require(identity['sha256'] == legacy.digest(payload) and payload['plan'] == spec['plan']
            and payload['sources'] == resolution['sources']
            and config['continuation'] == resolution['continuation']
            and reference['continuation'] == resolution['continuation']
            and reference['nfr128_scope'] == {'schema': 'olmo-nfr-128-execution-scope-v1',
                'declaration': resolution['scope'], 'sha256': resolution['scope_sha256']},
            'Final continuation identity differs')
    cursor = spec['plan']['updates'][127]['next_cursor']
    require(len(manifest['rank_cursors']) == 2 and all(c.get('rank') == rank
            and c.get('world_size') == 2 and c.get('physical_batch_per_rank') == 12
            and c.get('cursor') == cursor for rank, c in enumerate(manifest['rank_cursors'])),
            'Final ordered reader cursor differs')


def load_context(scope, *, verify_state):
    validate_scope(scope)
    refs = scope['references']
    objects = {name: legacy.read_json(legacy.local_path(ref['path']), ref['sha256'], limit=128*1024**2)
               for name, ref in refs.items()}
    old = objects['endpoint64_report']; preserved = old.get('preservation', {})
    require(old['schema'] == endpoint.SCHEMA and old['status'] == 'completed'
            and old['arm'] == 'NFR' and old['after_update'] == 64 and old['kl_weight'] == .1
            and old['case'] == 'reduced' and old['optimizer_updates_performed'] == 0
            and all(old.get(k) is True for k in ('weights_unchanged', 'rng_unchanged', 'gradient_buffers_absent'))
            and preserved.get('integrity_passed') is True and preserved.get('restored') is True
            and preserved.get('checks') and all(v is True for v in preserved['checks'].values()),
            'Require preserved reduced64 comparison probe')
    resolution = continuation.resolve(objects['continuation_scope'], legacy.local_path(refs['continuation_scope']['path']))
    require(resolution == objects['continuation_resolved'], 'Frozen training resolution differs')
    require(old['input_authorities']['checkpoint'] == resolution['scope']['parent64_checkpoint']
            and old['membership_sha256'] == scope['panel_membership_sha256'],
            'Comparison64 is not the exact continued checkpoint/panel')
    spec = previous.historical.diagnostic_spec('NFR')
    f_ref = old['input_authorities']['f_resolved']
    plan, _ = previous.shared_panel(spec, legacy.local_path(f_ref['path']), f_ref['sha256'])
    require(plan == old['probe_policy'], 'Shared eight-row panel policy differs')
    require(scope['implementation_sources'] == final_sources(old['sources'], resolution['sources']),
            'Final curve source authority differs')
    ref = scope['checkpoint']
    manifest = inspect_distributed_checkpoint(legacy.local_path(ref['path']),
        expected_manifest_sha256=ref['sha256'], verify_state=verify_state)
    authenticate_final(objects['terminal_report'], manifest, objects['terminal_publication'],
                       resolution, spec, manifest_sha256=ref['sha256'])
    return spec, plan, manifest, old


def make_scope(references, checkpoint):
    """Late bind only after the final report and publication have immutable pins."""
    read = lambda name: legacy.read_json(legacy.local_path(references[name]['path']),
                                         references[name]['sha256'], limit=128*1024**2)
    scope = {'schema': SCOPE_SCHEMA, **POLICY, 'references': references, 'checkpoint': checkpoint,
        'implementation_sources': final_sources(read('endpoint64_report')['sources'],
                                                 read('continuation_resolved')['sources'])}
    load_context(scope, verify_state=False)
    return scope


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); tracker = None
    report = {'schema': SCHEMA, 'status': 'preflight', 'arm': 'NFR', 'after_update': 128,
        'kl_weight': .1, 'num_passes': 32, 'optimizer_updates_performed': 0, 'rows': [],
        'predictor_execution': 'not_called; saved tensors retained',
        'scope': 'Reduced-KL NFR128; unchanged packed eight-row FP32/no-jitter panel and frozen64 math; '
                 'finite K32 reference, not exact online or BF16/quality acceptance'}
    def publish(event=None):
        report['elapsed_seconds'] = time.monotonic()-started
        if event is not None:
            report.setdefault('progress', []).append(event); print(json.dumps(event), flush=True)
        write_json(args.output_dir/'report.json', report)
    try:
        scope = legacy.read_json(args.scope, args.scope_sha256)
        spec, plan, manifest, old = load_context(scope, verify_state=not args.preflight_only)
        path = args.scope.resolve(); require(path.is_relative_to(ROOT), 'Keep scope inside project')
        sources = dict(scope['implementation_sources']) | {str(path.relative_to(ROOT)): args.scope_sha256}
        report.update(final_scope={'sha256': args.scope_sha256, 'declaration': scope},
            input_authorities=scope['references'] | {'checkpoint': scope['checkpoint'], 'state': manifest['state']},
            sources=sources, membership_sha256=scope['panel_membership_sha256'],
            index_manifest_sha256=plan['panel']['index_manifest_sha256'], probe_policy=plan)
        for name, pin in sources.items():
            require(sha256_file(ROOT/name) == pin, 'Diagnostic source changed: '+name)
            target = args.output_dir/'source-snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
        batches = previous.materialize_panel(spec, plan)
        report['batch_tensor_sha256'] = [previous.tree_digests(vars(batch)) for batch in batches]
        require(report['batch_tensor_sha256'] == old['batch_tensor_sha256'], 'Final panel tensors differ from64')
        if args.preflight_only:
            report['status'] = 'preflight_validated'
            publish({'phase': 'preflight_complete', 'gpu_execution': False, 'state_bytes_verified': False}); return
        previous.configure_determinism(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.set_num_threads(8)
        report['runtime'] = previous.require_container_gpu()
        require(torch.cuda.device_count() == 1 and not torch.distributed.is_initialized(),
                'Expose exactly one assigned GPU to the final diagnostic')
        tracker = previous.OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
            output_dir=args.output_dir, group='nfr-final-curves', name=args.output_dir.name,
            preserve_state=previous.preserve_local_rng)
        tracker.start({'arm': 'NFR', 'checkpoint_update': 128, 'kl_weight': .1, 'max_passes': 32,
                       'scope_sha256': args.scope_sha256, 'optimizer_updates_performed': 0})
        for prefix in ('curves/*', 'finite_reference/*'):
            tracker._call('pass axis', lambda prefix=prefix: tracker._run.define_metric(prefix, step_metric='total_pass'))
        report['tracking'] = tracker.record; report['status'] = 'observing'; publish({'phase': 'constructing_model'})
        model, _, _ = previous.historical.construct(spec, torch.device('cuda:0'))
        previous.import_saved_weights(model, spec, manifest, legacy.local_path(scope['checkpoint']['path']), kl_weight=.1)
        before = previous.state_pins(model); rng = previous.tree_digests(_local_rng(torch.device('cuda:0'), None))
        with previous.evaluation_runtime(model) as preservation:
            for row_index, batch in enumerate(batches):
                row_started = time.perf_counter()
                result, residuals = endpoint.endpoint_probe_batch(model, batch.to('cuda:0'), spec['recipe'])
                record = {'row_index': row_index, 'seconds': time.perf_counter()-row_started,
                          'result': result, 'finite_residuals': residuals}
                report['rows'].append(record); write_json(args.output_dir/f'row-{row_index:02d}.json', record)
                publish({'phase': 'row_completed', 'row_index': row_index, 'seconds': record['seconds']})
        report['preservation'] = preservation
        counts = plan['panel']['fixed_plan']['updates'][0]['counts']
        report['result'] = previous.probe.summarize([r['result'] for r in report['rows']],
            expected_tokens=counts['valid_tokens'], expected_ce_targets=counts['ce_targets'])
        report['finite_residuals'] = endpoint.summarize_residuals([r['finite_residuals'] for r in report['rows']])
        residuals = {r['pass']: r['regions'] for r in report['finite_residuals']['comparisons']}
        for p in report['result']['passes']:
            metrics = previous.scalar_metrics({n: r['metrics'] for n, r in p['regions'].items()}, 'curves')
            if p['pass'] in residuals: metrics.update(previous.scalar_metrics(residuals[p['pass']], 'finite_reference'))
            tracker.log({'total_pass': p['pass'], **metrics}, step=p['pass'])
        report['weights_unchanged'] = before == previous.state_pins(model)
        report['rng_unchanged'] = rng == previous.tree_digests(_local_rng(torch.device('cuda:0'), None))
        report['gradient_buffers_absent'] = all(p.grad is None for p in model.parameters())
        require(all(report[k] for k in ('weights_unchanged', 'rng_unchanged', 'gradient_buffers_absent')),
                'Final observation changed model state')
        require(all(sha256_file(ROOT/n) == pin for n, pin in sources.items()), 'Source changed during final observation')
        tracker.summary({'completed': True, 'optimizer_updates_performed': 0})
        tracker.finish(succeeded=True); report['status'] = 'completed'; publish({'phase': 'completed'})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if tracker is not None: tracker.finish(succeeded=False)
        publish(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True); parser.add_argument('--scope-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv); legacy.pin(args.scope_sha256); run(args)


if __name__ == '__main__': main()
