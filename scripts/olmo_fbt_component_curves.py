"""Saved NF/NFR inference-only pass curves on the identical F-only dev prefix.

No training source is edited and no auxiliary predictor is executed. Canonical
OLMoFBT.forward supplies finite-pass states; the existing F-probe reducer owns
all measurement definitions. This is a weights-only diagnostic, never a resume.
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

import torch
from torch.nn import functional as F

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint, _local_rng
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts import olmo_fbt_stability_probe as probe
from scripts import olmo_feedback_diagnostic as historical
from scripts import olmo_kl_continuation as kl_run
from scripts.olmo_kl_branch import set_kl_weight, declared_recipe
from scripts.olmo_campaign_evaluation import evaluation_runtime, _ACTIVE, FP32_FLAGS, tensor_metadata
from scripts.olmo_campaign_recurrence_precision import state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.experiment_tracking import OnlineTracker, scalar_metrics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'olmo-fbt-component-curves-v1'
PROTOCOL = 'docs/reports/olmo-fbt-stability/component-curves-protocol.md'
ALLOWED = {'NF': (0, 32, 64), 'NFR': (32,)}


def _expected_report_sources(reference):
    if reference['schema'] == 'olmo-pilot-async-execute-report-v1':
        return json.loads((ROOT/'.runtime/olmo-pilot-async/runtime-sources.json').read_text())
    if reference['schema'] == kl_run.SCHEMA:
        return kl_run.source_hashes()
    raise ValueError('Require accepted old200 or KL210 saved training report')


def authenticate_report(reference, manifest, spec, *, expected_manifest_sha256):
    """Bind checkpoint bytes to a successful retained report and its exact plan."""
    arm = spec['recipe'].arm; update = manifest['counters']['optimizer_updates']
    if (type(update) is not int or update not in ALLOWED.get(arm, ())
            or manifest['world_size'] != 2 or manifest['counters']['input_tokens'] != update*524288):
        raise ValueError('Saved condition/update/exposure is outside NF0/32/64 and NFR32 scope')
    if (reference.get('status') not in ('stopped_at_boundary', 'completed_plan')
            or reference.get('arm') != arm or reference.get('declaration_sha256') != historical.DECL
            or reference.get('resolved_sha256') != historical.RESOLVED
            or reference['configuration'] != manifest['metadata']['configuration']
            or reference['source_fingerprint'] != manifest['metadata']['source_fingerprint']):
        raise ValueError('Saved checkpoint and successful report lineage differ')
    publications = [p for p in reference['published_checkpoints']
                    if p['manifest_sha256'] == expected_manifest_sha256]
    if len(publications) != 1 or publications[0]['state'] != manifest['state']:
        raise ValueError('Checkpoint is not uniquely published by the pinned report')
    sources = _expected_report_sources(reference)
    if reference['sources'] != sources or any(sha256_file(ROOT/n) != pin for n, pin in sources.items()):
        raise ValueError('Saved condition implementation authority changed')
    config = reference['configuration']; identity = config['execution_identity']
    if (identity['sha256'] != historical.contract.legacy.digest(identity['payload'])
            or identity['payload']['resolved_contract_sha256'] != spec['resolved']['contract_sha256']
            or identity['payload']['arm'] != arm or identity['payload']['sources'] != sources
            or identity['payload']['plan'] != spec['plan']):
        raise ValueError('Saved execution identity differs from the authenticated condition')
    expected_cursor = spec['plan']['first_cursor'] if update == 0 else spec['plan']['updates'][update-1]['next_cursor']
    if len(manifest['rank_cursors']) != 2 or any(
            cursor.get('rank') != rank or cursor.get('world_size') != 2
            or cursor.get('physical_batch_per_rank') != 12 or cursor.get('cursor') != expected_cursor
            for rank, cursor in enumerate(manifest['rank_cursors'])):
        raise ValueError('Saved row cursor differs from original ordered membership')
    branch = config.get('objective_branch')
    if branch is None:
        if reference['schema'] != 'olmo-pilot-async-execute-report-v1' or update == 64:
            raise ValueError('Expected original accepted condition or explicitly branched NF64')
        kl_weight = 1.; recipe = spec['recipe'].to_dict()
    else:
        if (reference['schema'] != kl_run.SCHEMA or arm != 'NF' or update != 64
                or reference.get('branch') != branch or branch.get('parent_update') != 32
                or branch.get('review_stop') != 64 or branch.get('kl_weight') not in (1., .1)
                or reference.get('parent_manifest_sha256') != branch.get('parent_manifest_sha256')):
            raise ValueError('Require explicit NF32-to64 KL1 or KL0.1 branch')
        kl_weight = branch['kl_weight']
        recipe = declared_recipe(spec['recipe'], kl_weight=kl_weight,
                                  parent_manifest_sha256=branch['parent_manifest_sha256'])
        if branch.get('recipe_as_declared') != recipe:
            raise ValueError('KL branch declaration differs')
    if config['recipe'] != recipe or identity['payload']['recipe'] != recipe:
        raise ValueError('Saved operative recipe differs')
    return {'arm': arm, 'after_update': update, 'kl_weight': kl_weight,
            'saved_sources': sources, 'weights_only': True, 'optimizer_updates_performed': 0}


def shared_panel(spec, f_resolved, f_resolved_sha256):
    """Read the F-study pinned resolution without adding new training sources."""
    resolved = historical.contract.legacy.read_json(f_resolved, f_resolved_sha256, limit=128*1024**2)
    if (resolved['schema'] != 'olmo-fbt-stability-resolved-v1'
            or resolved['contract_sha256'] != historical.contract.legacy.digest(
                {k: v for k, v in resolved.items() if k != 'contract_sha256'})
            or resolved['manifest']['data'] != spec['manifest']['data']):
        raise ValueError('F-study resolution/data does not identify the same underlying ordered corpus')
    for name, pin in resolved['sources'].items():
        if sha256_file(ROOT/name) != pin:
            raise ValueError('F-study source authority changed: '+name)
    plan = resolved['probe_plan']
    if (plan['schema'] != probe.SCHEMA or plan['panel_rows'] != 8
            or plan['length'] != 1024 or plan['physical_batch_per_rank'] != 1):
        raise ValueError('Require the exact eight-row T1024 F-study panel')
    return plan, resolved['sources']


def import_saved_weights(model, spec, manifest, directory, *, kl_weight):
    """Preserve all predictor tensors/ownership; match objective metadata only."""
    set_kl_weight(model, kl_weight)
    config = manifest['metadata']['configuration']
    if (config['backbone'] != model.backbone.backbone.config.to_dict()
            or config['model'] != model.config.to_dict()):
        raise ValueError('Saved condition model configuration differs')
    payload = torch.load(Path(directory)/manifest['state']['filename'], map_location='cpu',
                         weights_only=True, mmap=True)
    historical.validate_weights_payload(payload, manifest, model)
    identities = {name: id(p) for name, p in model.named_parameters()}
    model.load_state_dict(payload['model'], strict=True, assign=False)
    if identities != {name: id(p) for name, p in model.named_parameters()}:
        raise ValueError('Saved-weight import replaced parameter ownership')
    del payload; gc.collect()


def component_probe_batch(model, batch, recipe, *, passes=32, vocab_position_chunk=256):
    """Same additive F statistics, with canonical NF/NFR core states and no predictor."""
    if id(model) not in _ACTIVE.get() or torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise ValueError('Use the preserved common FP32 no-grad evaluation runtime')
    if (recipe.arm not in ('F', 'NF', 'NFR') or model.enabled != recipe.nextlat
            or model.config.document_policy != recipe.document_policy
            or model.pass_loss_policy != 'campaign_v1' or model.gamma != 1.):
        raise ValueError('Declared finite-pass model contract differs')
    base = model.backbone.backbone
    if any(getattr(base, name) != value for name, value in FP32_FLAGS.items()):
        raise ValueError('Component curves require unchanged common-FP32 runtime')
    if (type(passes) is not int or not 1 <= passes <= 32
            or type(vocab_position_chunk) is not int or not 1 <= vocab_position_chunk <= 2048
            or batch.input_ids.shape != (1, recipe.sequence_length)
            or batch.input_ids.device != next(model.parameters()).device):
        raise ValueError('Bounded B1 matching-context component probe required')
    before = {name: None if value is None else tensor_metadata(value) for name, value in vars(batch).items()}
    pre_norm, inputs = [], []
    def observe_norm(_module, args):
        pre_norm.append(args[0].float().square().mean(-1))
    def observe_inputs(_module, args, kwargs):
        inputs.append(kwargs['inputs_embeds'].float().square().mean(-1))
    def predictor_forbidden(*_args):
        raise ValueError('Inference-only pass diagnostic unexpectedly executed NextLat predictor')
    handles = [base.norm.register_forward_pre_hook(observe_norm),
               base.register_forward_pre_hook(observe_inputs, with_kwargs=True)]
    if model.predictor is not None:
        handles.append(model.predictor.register_forward_pre_hook(predictor_forbidden))
    try:
        mode = replace(recipe.mode(), num_passes=passes, feedback_jitter=0.)
        output = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
            document_ids=batch.document_ids, mode=mode, right_padded_causal=True, return_logits=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(output.pass_hidden_states) != passes or len(pre_norm) != passes or len(inputs) != passes:
        raise ValueError('Canonical stack/norm/pass invocation count differs')
    ce_mask = torch.zeros_like(batch.valid_mask)
    ce_mask[:, :-1] = build_nextlat_masks(batch, document_policy=recipe.document_policy)['ce']
    records = []; previous = None
    for index, hidden in enumerate(output.pass_hidden_states):
        hidden_square = hidden.float().square().mean(-1)
        ce, entropy = probe._vocabulary_scalars(model.backbone, hidden, batch, ce_mask, vocab_position_chunk)
        if previous is not None:
            delta_square = (hidden.float()-previous.float()).square().mean(-1)
            previous_square = previous.float().square().mean(-1)
            cosine = F.cosine_similarity(hidden.float(), previous.float(), dim=-1, eps=1e-12)
        regions = {}
        for name, active in probe.region_masks(batch.valid_mask, index+1).items():
            targets = active & ce_mask
            values = {'positions': int(active.sum()), 'ce_targets': int(targets.sum()),
                'difference_positions': 0 if previous is None else int(active.sum()),
                'hidden_square_sum': float(hidden_square[active].double().sum()),
                'pre_norm_square_sum': float(pre_norm[index][active].double().sum()),
                'input_square_sum': float(inputs[index][active].double().sum()),
                'delta_square_sum': 0. if previous is None else float(delta_square[active].double().sum()),
                'previous_square_sum': 0. if previous is None else float(previous_square[active].double().sum()),
                'cosine_sum': 0. if previous is None else float(cosine[active].double().sum()),
                'ce_sum': float(ce[targets].double().sum()), 'entropy_sum': float(entropy[targets].double().sum())}
            if not all(math.isfinite(values[key]) for key in probe.SUM_FIELDS):
                raise ValueError('Nonfinite component curve measurement')
            regions[name] = values
        records.append({'pass': index+1, 'regions': regions}); previous = hidden
    if before != {name: None if value is None else tensor_metadata(value) for name, value in vars(batch).items()}:
        raise ValueError('Component curves modified input tensors')
    return {'schema': probe.SCHEMA, 'policy': 'common_fp32_no_jitter_v1', 'passes': records,
        'beta': 1., 'input_tokens': int(batch.valid_mask.sum()), 'ce_targets': int(ce_mask.sum())}


def materialize_panel(spec, plan):
    panel = plan['panel']; planned = panel['fixed_plan']['updates'][0]
    with OrderedCampaignData(historical.contract.legacy.local_path(spec['manifest']['data']['corpus']),
            historical.contract.legacy.local_path(panel['index'])) as reader:
        if (reader.manifest_sha256 != panel['index_manifest_sha256']
                or reader.split != 'dev' or reader.manifest['panel'] != 'dev-main'):
            raise ValueError('Component diagnostic panel identity differs')
        cursor = reader.cursor(); logical = reader.peek_update(cursor, panel['target_valid_tokens'])
        if logical is None or len(logical.rows) != 8:
            raise ValueError('Require all eight original packed rows')
        actual = {'start_cursor': asdict(cursor), 'next_cursor': asdict(logical.next_cursor),
            'counts': asdict(logical.counts), 'target_valid_tokens': panel['target_valid_tokens'],
            'overshoot_tokens': logical.overshoot_tokens,
            'unique_documents_in_this_update': logical.unique_document_count,
            'membership_sha256': historical.contract.legacy.digest([asdict(row) for row in logical.rows])}
        if any(planned[key] != value for key, value in actual.items()):
            raise ValueError('Component diagnostic changed F-study packed membership/counts')
        batches = [reader.batch([row], physical_batch_size=1) for row in logical.rows]
        if reader.cursor() != cursor:
            raise ValueError('Observation advanced ordered reader')
        reader.validate_integrity()
    return batches


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); tracker = None
    report = {'schema': SCHEMA, 'status': 'preflight', 'arm': args.arm,
        'num_passes': args.max_passes, 'rows': [], 'optimizer_updates_performed': 0,
        'predictor_execution': 'not_called; resident tensors and training-mode ownership preserved',
        'scope': 'Common-FP32 no-jitter teacher-forced finite passes on the identical packed F-study eight-row panel; no BF16 or quality acceptance'}
    def publish(event=None):
        report['elapsed_seconds'] = time.monotonic()-started
        if event is not None:
            report.setdefault('progress', []).append(event)
            print(json.dumps(event, allow_nan=False), flush=True)
        write_json(args.output_dir/'report.json', report)
    try:
        spec = historical.diagnostic_spec(args.arm)
        reference = historical.contract.legacy.read_json(args.reference_report,
            args.reference_report_sha256, limit=128*1024**2)
        manifest = inspect_distributed_checkpoint(args.checkpoint,
            expected_manifest_sha256=args.manifest_sha256, verify_state=True)
        authority = authenticate_report(reference, manifest, spec, expected_manifest_sha256=args.manifest_sha256)
        plan, f_sources = shared_panel(spec, args.f_resolved, args.f_resolved_sha256)
        extra = ('scripts/olmo_fbt_component_curves.py', 'tests/test_fbt_component_curves.py', PROTOCOL)
        sources = dict(f_sources)
        for name, pin in authority['saved_sources'].items():
            if name in sources and sources[name] != pin:
                raise ValueError('Saved F/component source pin conflict')
            sources[name] = pin
        sources.update({name: sha256_file(ROOT/name) for name in extra})
        report.update(after_update=authority['after_update'], kl_weight=authority['kl_weight'],
            membership_sha256=plan['panel']['fixed_plan']['updates'][0]['membership_sha256'],
            index_manifest_sha256=plan['panel']['index_manifest_sha256'], sources=sources,
            input_authorities={'reference_report': {'path': str(args.reference_report), 'sha256': args.reference_report_sha256},
                'f_resolved': {'path': str(args.f_resolved), 'sha256': args.f_resolved_sha256},
                'manifest_sha256': args.manifest_sha256, 'state': manifest['state']},
            probe_policy=plan)
        for name, pin in sources.items():
            if sha256_file(ROOT/name) != pin:
                raise ValueError('Component diagnostic source changed: '+name)
            target = args.output_dir/'source-snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
        batches = materialize_panel(spec, plan)
        report['batch_tensor_sha256'] = [tree_digests(vars(batch)) for batch in batches]
        if args.preflight_only:
            report['status'] = 'preflight_validated'; publish({'phase': 'preflight_complete', 'gpu_execution': False}); return
        configure_determinism(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.set_num_threads(8)
        report['runtime'] = require_container_gpu()
        if torch.distributed.is_initialized():
            raise ValueError('Standalone diagnostic must not join a training process group')
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=args.output_dir,
            group='fbt-component-curves', name=args.output_dir.name, preserve_state=preserve_local_rng)
        tracker.start({'arm': args.arm, 'checkpoint_update': authority['after_update'],
            'kl_weight': authority['kl_weight'], 'max_passes': args.max_passes,
            'manifest_sha256': args.manifest_sha256, 'optimizer_updates_performed': 0})
        tracker._call('component pass axis', lambda: tracker._run.define_metric('curves/*', step_metric='total_pass'))
        report['tracking'] = tracker.record; publish({'phase': 'constructing_model'})
        model, _, _ = historical.construct(spec, torch.device('cuda:0'))
        import_saved_weights(model, spec, manifest, args.checkpoint, kl_weight=authority['kl_weight'])
        before = state_pins(model); rng = tree_digests(_local_rng(torch.device('cuda:0'), None))
        with evaluation_runtime(model) as preservation:
            for row_index, batch in enumerate(batches):
                row_started = time.perf_counter()
                result = component_probe_batch(model, batch.to('cuda:0'), spec['recipe'], passes=args.max_passes)
                record = {'row_index': row_index, 'seconds': time.perf_counter()-row_started, 'result': result}
                report['rows'].append(record)
                write_json(args.output_dir/f'row-{row_index:02d}.json', record)
                publish({'phase': 'row_completed', 'row_index': row_index, 'seconds': record['seconds']})
        report['preservation'] = preservation
        counts = plan['panel']['fixed_plan']['updates'][0]['counts']
        report['result'] = probe.summarize([row['result'] for row in report['rows']],
            expected_tokens=counts['valid_tokens'], expected_ce_targets=counts['ce_targets'])
        for p in report['result']['passes']:
            tracker.log({'total_pass': p['pass'], **scalar_metrics(
                {name: values['metrics'] for name, values in p['regions'].items()}, 'curves')}, step=p['pass'])
        report['weights_unchanged'] = before == state_pins(model)
        report['rng_unchanged'] = rng == tree_digests(_local_rng(torch.device('cuda:0'), None))
        report['gradient_buffers_absent'] = all(p.grad is None for p in model.parameters())
        if not all(report[key] for key in ('weights_unchanged', 'rng_unchanged', 'gradient_buffers_absent')):
            raise ValueError('Component probe changed model state')
        if any(sha256_file(ROOT/name) != pin for name, pin in sources.items()):
            raise ValueError('Component diagnostic sources changed during execution')
        report['status'] = 'completed'
        tracker.summary({'completed': True, 'optimizer_updates_performed': 0})
        tracker.finish(succeeded=True); publish({'phase': 'completed'})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if tracker is not None:
            tracker.finish(succeeded=False)
        publish(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=tuple(ALLOWED))
    for name in ('reference-report', 'f-resolved', 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('reference-report-sha256', 'f-resolved-sha256', 'manifest-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--max-passes', type=int, choices=(4, 8, 32), default=32)
    parser.add_argument('--preflight-only', action='store_true', help='CPU authority/data check; do not construct or execute a model')
    args = parser.parse_args(argv)
    for pin in (args.reference_report_sha256, args.f_resolved_sha256, args.manifest_sha256):
        historical.contract.legacy.pin(pin)
    run(args)


if __name__ == '__main__':
    main()
