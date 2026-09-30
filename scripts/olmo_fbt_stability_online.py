"""Saved F-only weights: isolated cropped finite passes versus exact online FBT.

This is an observational weights-only import, not a training restart. The two
spans are selected by document metadata before token materialization. They do
not have the original packed T1024 context: each crop resets its position and
cache. Numerical agreement is not a language-quality or BF16 acceptance claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, _local_rng
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo_fbt import FBTOnlineMode
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_fbt_stability_contract as contract
from scripts import olmo_fbt_stability_execute as execution
from scripts import olmo_campaign_manifest as legacy
from scripts.olmo_campaign_evaluation import evaluation_runtime, _ACTIVE
from scripts.olmo_campaign_recurrence_precision import state_pins
from scripts.olmo_feedback_diagnostic import validate_weights_payload
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu
from scripts.experiment_tracking import OnlineTracker, scalar_metrics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'olmo-fbt-stability-online-v1'
UPDATES = (0, 32, 64, 96, 100, 128, 192)
PASSES = (2, 4, 8, 32)


def select_spans(rows, *, crop_length, required=2):
    """First qualifying document segments in row/offset order, no token access."""
    if type(crop_length) is not int or crop_length < 2 or type(required) is not int or required < 1:
        raise ValueError('Positive bounded span selection is required')
    selected = []
    for row_ordinal, row in enumerate(rows):
        segments = row['segments']
        offsets = [s['chunk_offset'] for s in segments]
        if offsets != sorted(set(offsets)):
            raise ValueError('Document segments must be ordered and nonoverlapping')
        for segment in segments:
            if segment['length'] >= crop_length:
                selected.append({'panel_row': row_ordinal, 'descriptor': row['descriptor'],
                    'segment': segment, 'row_start_offset': segment['chunk_offset'],
                    'document_start_offset': segment['document_token_offset'],
                    'crop_length': crop_length})
                if len(selected) == required:
                    return selected
    raise ValueError('First eight dev-main rows contain too few qualifying single-document spans; no content-based fallback')


def load_panel(spec, *, crop_length=128):
    """Authenticate original eight-row panel, select on metadata, then read tokens."""
    if crop_length not in (128, 256):
        raise ValueError('Native online panel allows explicit crop lengths 128 or 256 only')
    panel = spec['probe_plan']['panel']; planned = panel['fixed_plan']['updates'][0]
    data_spec = spec['manifest']['data']
    with OrderedCampaignData(legacy.local_path(data_spec['corpus']), legacy.local_path(panel['index'])) as data:
        if (data.manifest_sha256 != panel['index_manifest_sha256']
                or data.split != 'dev' or data.manifest['panel'] != 'dev-main'):
            raise ValueError('Online diagnostic panel authority changed')
        origin = data.cursor(); logical = data.peek_update(origin, panel['target_valid_tokens'])
        if logical is None or len(logical.rows) != 8:
            raise ValueError('Require the same first eight packed dev-main rows')
        actual = {'start_cursor': asdict(origin), 'next_cursor': asdict(logical.next_cursor),
            'counts': asdict(logical.counts), 'target_valid_tokens': panel['target_valid_tokens'],
            'overshoot_tokens': logical.overshoot_tokens,
            'unique_documents_in_this_update': logical.unique_document_count,
            'membership_sha256': legacy.digest([asdict(row) for row in logical.rows])}
        if any(planned[key] != value for key, value in actual.items()):
            raise ValueError('Original packed panel membership/counts differ')
        descriptors = [{'descriptor': asdict(row), 'segments': [asdict(s) for s in data._segments(row)]}
                       for row in logical.rows]
        selected = select_spans(descriptors, crop_length=crop_length)
        if data._token_fds:
            raise ValueError('Selection unexpectedly accessed token payloads')
        batches, records = [], []
        for choice in selected:
            descriptor = logical.rows[choice['panel_row']]
            materialized = data.batch([descriptor], physical_batch_size=1)
            offset = choice['row_start_offset']; end = offset+crop_length
            ids = materialized.input_ids[:, offset:end].clone()
            valid = materialized.valid_mask[:, offset:end].clone()
            docs = materialized.document_ids[:, offset:end].clone()
            if (ids.shape != (1, crop_length) or not bool(valid.all())
                    or docs.unique().numel() != 1 or int(docs[0, 0]) != choice['segment']['document_index']):
                raise ValueError('Materialized crop differs from selected single-document span')
            batch = NextLatBatch(ids, valid, docs)
            content = {'input_ids': ids[0].tolist(), 'document_ids': docs[0].tolist()}
            records.append({**choice, **content, 'token_identity_sha256': legacy.digest(content)})
            batches.append(batch)
        if data.cursor() != origin:
            raise ValueError('Diagnostic advanced the ordered reader')
        data.validate_integrity()
    provenance = {'schema': SCHEMA, 'panel': 'dev-main', 'crop_length': crop_length,
        'index_manifest_sha256': panel['index_manifest_sha256'],
        'packed_membership_sha256': planned['membership_sha256'], 'packed_rows_considered': 8,
        'selection': 'First two metadata-ordered document segments with length >= crop_length; no token/content or result selection',
        'position_policy': 'Each crop resets RoPE positions to 0..length-1',
        'context_scope': 'Independent single-document crops; omitted preceding packed tokens and preceding document context',
        'cache_policy': 'Fresh cache per crop and model checkpoint; no carry between rows',
        'records': records}
    return batches, {**provenance, 'identity_sha256': legacy.digest(provenance)}


def validate_saved_authority(manifest, spec, model):
    """Reject a valid checkpoint from another arm, recipe or training lineage."""
    update = manifest['counters']['optimizer_updates']
    if type(update) is not int or update not in UPDATES:
        raise ValueError('Saved update is outside the declared F milestone scope')
    if manifest['world_size'] != 2 or manifest['counters']['input_tokens'] != update*524288:
        raise ValueError('Saved world size or input exposure differs from F study')
    expected_cursor = spec['plan']['first_cursor'] if update == 0 else spec['plan']['updates'][update-1]['next_cursor']
    for rank, cursor in enumerate(manifest['rank_cursors']):
        if (cursor.get('rank') != rank or cursor.get('world_size') != 2
                or cursor.get('physical_batch_per_rank') != 12 or cursor.get('cursor') != expected_cursor):
            raise ValueError('Saved rank cursor differs from exact F ordered plan boundary')
    if len(manifest['rank_cursors']) != 2:
        raise ValueError('Saved rank count differs')
    config = manifest['metadata']['configuration']; identity = config['execution_identity']
    if (identity.get('sha256') != legacy.digest(identity['payload'])
            or identity['payload'].get('resolved_contract_sha256') != spec['resolved']['contract_sha256']
            or identity['payload'].get('scope') != 'clean-F-only-stability'
            or identity['payload'].get('arm') != 'F'
            or identity['payload'].get('declaration') != spec['declaration']
            or identity['payload'].get('sources') != spec['resolved']['sources']):
        raise ValueError('Saved F execution identity or implementation lineage differs')
    if (config['recipe'] != spec['recipe'].to_dict()
            or config['backbone'] != model.backbone.backbone.config.to_dict()
            or config['model'] != model.config.to_dict()):
        raise ValueError('Saved F architecture/recipe differs')


def import_saved_weights(model, spec, directory, manifest_sha256):
    manifest = inspect_distributed_checkpoint(directory,
        expected_manifest_sha256=manifest_sha256, verify_state=True)
    validate_saved_authority(manifest, spec, model)
    payload = torch.load(Path(directory)/manifest['state']['filename'], map_location='cpu',
                         weights_only=True, mmap=True)
    validate_weights_payload(payload, manifest, model)
    identities = {name: id(p) for name, p in model.named_parameters()}
    model.load_state_dict(payload['model'], strict=True, assign=False)
    if identities != {name: id(p) for name, p in model.named_parameters()}:
        raise ValueError('Weights-only import replaced parameter ownership')
    del payload; gc.collect()
    return {'manifest_sha256': manifest_sha256, 'state': manifest['state'],
        'counters': manifest['counters'], 'saved_world_size': manifest['world_size'],
        'mode': 'strict weights-only diagnostic import; no optimizer/scheduler/RNG restoration',
        'optimizer_updates_performed': 0}


def _position_errors(actual, reference):
    a, b = actual.float(), reference.float()
    delta = (a-b).double().square().sum(-1)
    scale = b.double().square().sum(-1)
    values = {'difference_square': delta, 'reference_square': scale,
        'relative_l2': torch.sqrt(delta/scale.clamp_min(1e-60)),
        'max_abs': (a-b).abs().amax(-1).double()}
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError('Nonfinite online/finite comparison')
    return {key: value.detach().cpu().tolist() for key, value in values.items()}


def compare_states(core, actual, reference, input_ids, *, chunk=32):
    """Per-position hidden/logit errors and CE, retaining no full-row logits."""
    if (actual.shape != reference.shape or actual.ndim != 3 or actual.shape[0] != 1
            or input_ids.shape != actual.shape[:2]):
        raise ValueError('Online comparison requires matching single-row states and token prefix')
    if type(chunk) is not int or not 1 <= chunk <= 256:
        raise ValueError('Bounded vocabulary position chunk required')
    length = actual.shape[1]
    hidden = _position_errors(actual[0], reference[0])
    logit = {key: [] for key in hidden}; actual_ce = []; reference_ce = []
    for start in range(0, length, chunk):
        end = min(length, start+chunk)
        scores = core.project_logits(actual[0, start:end]).float()
        expected = core.project_logits(reference[0, start:end]).float()
        errors = _position_errors(scores, expected)
        for key in logit:
            logit[key].extend(errors[key])
        stop = min(end, length-1)
        if stop > start:
            labels = input_ids[0, start+1:stop+1]
            actual_ce.extend(F.cross_entropy(scores[:stop-start], labels, reduction='none').double().cpu().tolist())
            reference_ce.extend(F.cross_entropy(expected[:stop-start], labels, reduction='none').double().cpu().tolist())
    positions = []
    for p in range(length):
        ce = None if p == length-1 else actual_ce[p]
        ref_ce = None if p == length-1 else reference_ce[p]
        positions.append({'position': p, 'target_position': p+1 if p < length-1 else None,
            'hidden': {key: value[p] for key, value in hidden.items()},
            'logits': {key: value[p] for key, value in logit.items()},
            'finite_ce': ce, 'online_ce': ref_ce,
            'ce_gap': None if ce is None else ce-ref_ce})
    return {'length': length, 'positions': positions,
        'regions': {name: summarize_positions([r for r in positions if left <= r['position'] < length])
                    for name, left in (('all', 0), ('tail_32', max(0, length-32)), ('late_half', length//2))}}


def summarize_positions(positions):
    if not positions:
        raise ValueError('Cannot summarize an empty isolated comparison')
    summary = {'positions': len(positions), 'ce_targets': sum(p['finite_ce'] is not None for p in positions)}
    for kind in ('hidden', 'logits'):
        error = math.fsum(p[kind]['difference_square'] for p in positions)
        reference = math.fsum(p[kind]['reference_square'] for p in positions)
        summary[kind] = {'relative_l2': math.sqrt(error/max(reference, 1e-60)),
            'difference_square': error, 'reference_square': reference,
            'max_abs': max(p[kind]['max_abs'] for p in positions)}
    for name in ('finite_ce', 'online_ce', 'ce_gap'):
        values = [p[name] for p in positions if p[name] is not None]
        summary[name] = math.fsum(values)/len(values) if values else None
    if any(not math.isfinite(value) for value in scalar_metrics(summary).values()):
        raise ValueError('Nonfinite summarized online comparison')
    return summary


def compare_crop(model, recipe, batch, *, passes=PASSES):
    """Two distinct execution APIs on exactly the same isolated teacher-forced crop."""
    if id(model) not in _ACTIVE.get() or torch.is_grad_enabled():
        raise ValueError('Use the preserved no-grad common FP32 evaluation runtime')
    if (recipe.arm != 'F' or model.enabled or model.predictor is not None
            or recipe.mode().rt_mode.selected_layers or batch.input_ids.shape[0] != 1
            or not bool(batch.valid_mask.all()) or batch.document_ids.unique().numel() != 1):
        raise ValueError('Require F-only, no RT, one fully valid single-document row')
    if (not passes or tuple(sorted(set(passes))) != tuple(passes)
            or any(type(k) is not int or not 1 <= k <= 32 for k in passes)):
        raise ValueError('Finite comparison pass count must be ordered, unique and bounded to 32')
    core = model.backbone
    kwargs = {'attention_mask': batch.valid_mask, 'document_ids': batch.document_ids,
              'return_logits': False}
    started = time.perf_counter()
    online = core.forward_online(batch.input_ids, mode=FBTOnlineMode(beta=1., rt_mode=RTMode(())),
                                 use_cache=False, **kwargs)
    if online.past_key_values is not None:
        raise ValueError('Standalone online crop retained a cross-row cache')
    if batch.input_ids.device.type == 'cuda':
        torch.cuda.synchronize(batch.input_ids.device)
    online_seconds = time.perf_counter()-started
    started = time.perf_counter()
    mode = replace(recipe.mode(), num_passes=max(passes), beta=1., feedback_jitter=0.,
                   document_policy='isolated-v1')
    finite = core(batch.input_ids, mode=mode, right_padded_causal=True, **kwargs)
    if batch.input_ids.device.type == 'cuda':
        torch.cuda.synchronize(batch.input_ids.device)
    finite_seconds = time.perf_counter()-started
    results = [{'total_pass': k, **compare_states(core, finite.pass_hidden_states[k-1],
                         online.last_hidden_state, batch.input_ids)} for k in passes]
    return {'schema': SCHEMA, 'length': batch.input_ids.shape[1], 'passes': results,
        'finite_mode': asdict(mode), 'online_mode': asdict(FBTOnlineMode()),
        'one_document_per_row': True, 'fresh_cache': True, 'position_origin': 0,
        'online_seconds': online_seconds, 'finite_all_passes_seconds': finite_seconds,
        'timing_scope': 'Synchronized forward wall time only; diagnostic FP32 math, not production throughput'}


def run(args):
    started = time.monotonic(); tracker = None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': SCHEMA, 'status': 'preflight', 'optimizer_updates_performed': 0,
        'precision': 'common_fp32_no_jitter_v1', 'cases': [],
        'scope': 'Isolated cropped teacher-forced Jacobi-versus-exact-online states; distinct from packed T1024 dev CE; no quality/BF16 clearance'}
    def publish(event=None):
        report['elapsed_seconds'] = time.monotonic()-started
        if event is not None:
            report.setdefault('progress', []).append(event)
            print(json.dumps(event, allow_nan=False), flush=True)
        write_json(args.output_dir/'report.json', report)
    try:
        spec = execution.load_spec(SimpleNamespace(arm='F', checkpoint_mode='async',
            declaration=args.declaration, declaration_sha256=args.declaration_sha256,
            resolved=args.resolved, resolved_sha256=args.resolved_sha256, stop_after=None))
        if spec['kind'] != 'native':
            raise ValueError('Standalone diagnostic requires the pinned native F declaration')
        original_sources = spec['resolved']['sources']
        extra = ('scripts/olmo_fbt_stability_online.py', 'tests/test_fbt_stability_online.py')
        report['sources'] = original_sources | {name: sha256_file(ROOT/name) for name in extra}
        for name, expected in report['sources'].items():
            if sha256_file(ROOT/name) != expected:
                raise ValueError('Source changed before diagnostic: '+name)
            destination = args.output_dir/'source-snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        batches, fixture = load_panel(spec, crop_length=args.crop_length)
        report['fixture'] = fixture; write_json(args.output_dir/'fixture.json', fixture)
        report['authorities'] = {'declaration_sha256': args.declaration_sha256,
            'resolved_sha256': args.resolved_sha256, 'manifest_sha256': args.manifest_sha256}
        configure_determinism(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.set_num_threads(8)
        report['runtime'] = require_container_gpu()
        if torch.distributed.is_initialized():
            raise ValueError('Standalone online diagnostics do not run inside DDP')
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=args.output_dir,
            group='fbt-only-stability-online', name=args.output_dir.name, preserve_state=preserve_local_rng)
        tracker.start({'schema': SCHEMA, 'fixture_sha256': fixture['identity_sha256'],
            'manifest_sha256': args.manifest_sha256, 'crop_length': args.crop_length,
            'precision': report['precision'], 'optimizer_updates_performed': 0})
        tracker._call('online pass axis', lambda: tracker._run.define_metric('online/*', step_metric='total_pass'))
        report['tracking'] = tracker.record
        publish({'phase': 'constructing_model', 'wandb': tracker.record['run_url']})
        model, _, startup = execution.construct(spec, torch.device('cuda:0'))
        report['startup_import'] = startup
        report['checkpoint'] = import_saved_weights(model, spec, args.checkpoint, args.manifest_sha256)
        before = state_pins(model); rng = tree_digests(_local_rng(torch.device('cuda:0'), None))
        publish({'phase': 'weights_loaded', 'checkpoint_update': report['checkpoint']['counters']['optimizer_updates']})
        with evaluation_runtime(model) as preservation:
            for index, batch in enumerate(batches):
                case = compare_crop(model, spec['recipe'], batch.to('cuda:0'))
                case['crop_index'] = index; report['cases'].append(case)
                write_json(args.output_dir/f'crop-{index:02d}.json', case)
                publish({'phase': 'crop_completed', 'crop_index': index,
                         'pass32': case['passes'][-1]['regions']['all']})
        report['preservation'] = preservation
        report['summary'] = [{'total_pass': k,
            'regions': {region: summarize_positions([position for case in report['cases']
                for position in next(p for p in case['passes'] if p['total_pass'] == k)['positions']
                if position['position'] >= left]) for region, left in
                    (('all', 0), ('tail_32', args.crop_length-32), ('late_half', args.crop_length//2))}}
            for k in PASSES]
        for row in report['summary']:
            tracker.log({'total_pass': row['total_pass'], **scalar_metrics(row['regions'], 'online')},
                        step=row['total_pass'])
        report['weights_unchanged'] = before == state_pins(model)
        report['rng_unchanged'] = rng == tree_digests(_local_rng(torch.device('cuda:0'), None))
        report['gradient_buffers_absent'] = all(p.grad is None for p in model.parameters())
        if not all(report[key] for key in ('weights_unchanged', 'rng_unchanged', 'gradient_buffers_absent')):
            raise ValueError('Online diagnostic changed live model state')
        if any(sha256_file(ROOT/name) != expected for name, expected in report['sources'].items()):
            raise ValueError('Diagnostic sources changed during execution')
        report['status'] = 'complete'
        tracker.summary({'completed': True, 'optimizer_updates_performed': 0,
            'checkpoint_update': report['checkpoint']['counters']['optimizer_updates']})
        tracker.finish(succeeded=True); publish({'phase': 'complete'})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if tracker is not None:
            tracker.finish(succeeded=False)
        publish(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('declaration', 'resolved', 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('declaration-sha256', 'resolved-sha256', 'manifest-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--crop-length', type=int, choices=(128, 256), default=128)
    args = parser.parse_args(argv)
    for pin in (args.declaration_sha256, args.resolved_sha256, args.manifest_sha256):
        legacy.pin(pin)
    run(args)


if __name__ == '__main__':
    main()
