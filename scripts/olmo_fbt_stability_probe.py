"""Bounded, no-update FBT pass curves over a fixed held-out ordered prefix.

This module deliberately does not change the accepted model/evaluator. Passes
stream through its existing input preparation, stack and fusion operations.
Only the preceding hidden state and position-chunked vocabulary scores remain
live. The ordinary evaluator owns live-state/RNG/runtime preservation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time

import torch
from torch.nn import functional as F

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.document_policy import feedback_eligibility
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_pilot_ordered_data as ordered
from scripts.olmo_campaign_eval_control import graph_boundary
from scripts.olmo_campaign_evaluation import (evaluation_runtime, FP32_FLAGS,
    _ACTIVE, tensor_metadata)
from scripts.olmo_pilot_eval_control import resolve_evaluation

SCHEMA = 'olmo-fbt-stability-probe-v1'
REGIONS = ('all', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
           'tail_128', 'unsettled_suffix')
SUM_FIELDS = ('hidden_square_sum', 'pre_norm_square_sum', 'input_square_sum',
              'delta_square_sum', 'previous_square_sum', 'cosine_sum',
              'ce_sum', 'entropy_sum')


def resolve_probe_plan(data_spec, *, length, updates, world_size,
                       physical_batch=1, panel_rows=8):
    """Pin actual dev-main row membership using the accepted metadata planner."""
    for name, value in [('length', length), ('updates', updates), ('world_size', world_size),
                        ('physical_batch', physical_batch), ('panel_rows', panel_rows)]:
        legacy.integer(value, name)
    if panel_rows > 8 or physical_batch > 2:
        raise ValueError('Stability probes are bounded to eight rows and physical batch <=2')
    policy = {'kind': 'ordered_named_dev_panels_v1',
        'panels': [{'name': 'dev-main', 'target_valid_tokens': panel_rows*length}],
        'physical_batch_by_arm': {'F': physical_batch}, 'every_updates': 8,
        'precision': 'fp32', 'feedback_jitter': 0.,
        'report_passes': 'all_trained_passes', 'generation': 'not_implemented'}
    resolved = resolve_evaluation(data_spec, policy, length=length, updates=updates,
                                  partitions={'F': (world_size, physical_batch)})
    deep = [x for x in (0, 32, 64, 96, 100, 128, 192) if x <= updates]
    return {'schema': SCHEMA, 'panel': resolved['panels']['dev-main'],
        'length': length, 'panel_rows': panel_rows, 'world_size': world_size,
        'physical_batch_per_rank': physical_batch,
        'scheduled_updates': sorted(set([0, *range(8, updates+1, 8), *deep])),
        'deep_updates': deep, 'shallow_passes': 8, 'deep_passes': 32,
        'vocab_position_chunk': 256, 'precision': 'common_fp32_no_jitter_v1',
        'hidden': 'Native final-normalized backbone output; pre_norm is input to native final norm',
        'difference': 'Consecutive total passes; RMS(delta)/RMS(previous), denominator floor 1e-12',
        'entropy': 'Full-vocabulary entropy at exactly the eligible CE prediction positions',
        'unsettled_suffix': 'At total pass p, retain hidden positions >= p-1; omit known settled causal prefix',
        'regions': list(REGIONS), 'scope': 'Teacher-forced finite Jacobi passes; no exact-online claim'}


def _guard(model, batch, recipe):
    if id(model) not in _ACTIVE.get() or torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise ValueError('Use evaluation_runtime with no gradients and all modules in eval mode')
    if (recipe.arm != 'F' or model.enabled or recipe.mode().rt_mode.selected_layers
            or model.config.document_policy != recipe.document_policy):
        raise ValueError('This initial stability probe requires the declared F-only condition')
    if any(getattr(model.backbone.backbone, key) != value for key, value in FP32_FLAGS.items()):
        raise ValueError('Stability probe requires the preserved common FP32 runtime')
    device = next(model.parameters()).device
    if batch.input_ids.device != device or batch.input_ids.shape[1] != recipe.sequence_length:
        raise ValueError('Probe batch must be on the local model device and declared sequence length')
    if torch.is_autocast_enabled(device.type):
        raise ValueError('Probe autocast must be disabled')


def stream_pass_states(model, batch, recipe, *, passes, beta=1.):
    """Yield (one-based pass, hidden, input/pre-norm per-token squared RMS).

    The algebra matches OLMoFBT.forward, including continuous document policy,
    first-pass policy, padding, positions and shifted eligibility. A temporary
    hook measures native pre-output scale and is removed before every yield.
    It never changes the forward output or retains a vocabulary tensor.
    """
    _guard(model, batch, recipe)
    if type(passes) is not int or not 1 <= passes <= 32:
        raise ValueError('Bounded probe requires 1..32 total passes')
    mode = replace(recipe.mode(), num_passes=passes, feedback_jitter=0., beta=beta)
    core = model.backbone; base = core.backbone
    embeds, valid, positions, _, _, _ = base._prepare_right_padded_inputs(
        batch.input_ids, None, batch.valid_mask, None)
    documents = core._documents(valid, batch.document_ids, mode.document_policy)
    eligible = feedback_eligibility(valid, documents, mode.document_policy)
    current = embeds
    for index in range(passes):
        scales = {}
        def pre_norm(_module, args):
            scales['pre_norm_square'] = args[0].float().square().mean(-1)
        handle = base.norm.register_forward_pre_hook(pre_norm)
        try:
            hidden = core._stack(current, mode.initial_rt_mode if index == 0 else mode.rt_mode,
                attention_mask=valid, position_ids=positions,
                right_padded_causal=True).last_hidden_state
        finally:
            handle.remove()
        if 'pre_norm_square' not in scales:
            raise ValueError('Native final norm hook did not run')
        scales['input_square'] = current.float().square().mean(-1)
        yield index+1, hidden, scales
        if index+1 < passes:
            suffix = core._blend(hidden[:, :-1], embeds[:, 1:], mode.beta, eligible)
            current = torch.cat((embeds[:, :1], suffix), dim=1)


def region_masks(valid, total_pass):
    """Position regions refer to fixed row coordinates, never document offsets."""
    length = valid.shape[1]
    position = torch.arange(length, device=valid.device)[None, :]
    result = {'all': valid}
    for q in range(4):
        result[f'quarter_{q+1}'] = valid & (position >= q*length//4) & (position < (q+1)*length//4)
    result['tail_128'] = valid & (position >= max(0, length-128))
    result['unsettled_suffix'] = valid & (position >= total_pass-1)
    return result


def _vocabulary_scalars(core, hidden, batch, ce_mask, chunk):
    """Full-vocabulary CE and entropy, position-chunked with separate raw sums."""
    ce = torch.zeros_like(batch.valid_mask, dtype=torch.float32)
    entropy = torch.zeros_like(ce)
    locations = ce_mask.nonzero(as_tuple=False)
    for start in range(0, locations.shape[0], chunk):
        rows, positions = locations[start:start+chunk].unbind(1)
        logits = core.project_logits(hidden[rows, positions]).float()
        log_probs = F.log_softmax(logits, dim=-1)
        targets = batch.input_ids[rows, positions+1]
        ce[rows, positions] = -log_probs.gather(1, targets[:, None]).squeeze(1)
        entropy[rows, positions] = -(log_probs.exp()*log_probs).sum(-1)
    return ce, entropy


def probe_batch(model, batch, recipe, *, passes, vocab_position_chunk=256, beta=1.):
    """Return only JSON additive statistics; do not average rank-local means."""
    _guard(model, batch, recipe)
    if type(vocab_position_chunk) is not int or not 1 <= vocab_position_chunk <= 2048:
        raise ValueError('Choose a bounded positive vocabulary position chunk')
    before = {name: None if value is None else tensor_metadata(value) for name, value in vars(batch).items()}
    ce_mask = torch.zeros_like(batch.valid_mask)
    ce_mask[:, :-1] = build_nextlat_masks(batch, document_policy=recipe.document_policy)['ce']
    previous = None; records = []
    for total_pass, hidden, scales in stream_pass_states(model, batch, recipe, passes=passes, beta=beta):
        hidden_square = hidden.float().square().mean(-1)
        ce, entropy = _vocabulary_scalars(model.backbone, hidden, batch, ce_mask, vocab_position_chunk)
        if previous is not None:
            delta_square = (hidden.float()-previous.float()).square().mean(-1)
            previous_square = previous.float().square().mean(-1)
            cosine = F.cosine_similarity(hidden.float(), previous.float(), dim=-1, eps=1e-12)
        regions = {}
        for name, active in region_masks(batch.valid_mask, total_pass).items():
            targets = active & ce_mask
            values = {'positions': int(active.sum()), 'ce_targets': int(targets.sum()),
                'difference_positions': 0 if previous is None else int(active.sum()),
                'hidden_square_sum': float(hidden_square[active].double().sum()),
                'pre_norm_square_sum': float(scales['pre_norm_square'][active].double().sum()),
                'input_square_sum': float(scales['input_square'][active].double().sum()),
                'delta_square_sum': 0. if previous is None else float(delta_square[active].double().sum()),
                'previous_square_sum': 0. if previous is None else float(previous_square[active].double().sum()),
                'cosine_sum': 0. if previous is None else float(cosine[active].double().sum()),
                'ce_sum': float(ce[targets].double().sum()), 'entropy_sum': float(entropy[targets].double().sum())}
            if not all(math.isfinite(values[key]) for key in SUM_FIELDS):
                raise ValueError('Nonfinite raw stability statistic')
            regions[name] = values
        records.append({'pass': total_pass, 'regions': regions})
        previous = hidden
    if before != {name: None if value is None else tensor_metadata(value) for name, value in vars(batch).items()}:
        raise ValueError('Stability probe modified its held-out input tensors')
    return {'schema': SCHEMA, 'policy': 'common_fp32_no_jitter_v1', 'passes': records,
        'beta': float(beta), 'input_tokens': int(batch.valid_mask.sum()),
        'ce_targets': int(ce_mask.sum())}


def summarize(rows, *, expected_tokens, expected_ce_targets):
    """Sum sufficient statistics over physical batches/ranks before deriving ratios."""
    if not rows:
        raise ValueError('Probe requires physical batch contributions')
    count = len(rows[0]['passes']); beta = rows[0]['beta']
    if not 1 <= count <= 32:
        raise ValueError('Invalid pass count')
    for row in rows:
        if (row['schema'] != SCHEMA or row['policy'] != 'common_fp32_no_jitter_v1'
                or row['beta'] != beta or len(row['passes']) != count
                or [p['pass'] for p in row['passes']] != list(range(1, count+1))):
            raise ValueError('Probe batch policies or pass order differ')
    tokens = sum(row['input_tokens'] for row in rows)
    targets = sum(row['ce_targets'] for row in rows)
    if tokens != expected_tokens or targets != expected_ce_targets:
        raise ValueError('Probe targets/tokens differ from independently planned membership')
    output = []
    for index in range(count):
        regions = {}
        for name in REGIONS:
            values = [row['passes'][index]['regions'][name] for row in rows]
            for value in values:
                if (set(value) != set(SUM_FIELDS) | {'positions', 'ce_targets', 'difference_positions'}
                        or any(type(value[key]) is not int or value[key] < 0 for key in
                            ('positions', 'ce_targets', 'difference_positions'))
                        or value['ce_targets'] > value['positions']
                        or value['difference_positions'] != (0 if index == 0 else value['positions'])
                        or any(type(value[key]) not in (int, float) or not math.isfinite(value[key]) for key in SUM_FIELDS)):
                    raise ValueError('Malformed raw probe region statistics')
            total = {key: math.fsum(value[key] for value in values) for key in SUM_FIELDS}
            total.update({key: sum(value[key] for value in values)
                for key in ('positions', 'ce_targets', 'difference_positions')})
            n, d, c = total['positions'], total['difference_positions'], total['ce_targets']
            if any(total[key] < 0 for key in SUM_FIELDS if key != 'cosine_sum'):
                raise ValueError('Negative squared scale, CE or entropy')
            metrics = {key+'_rms': math.sqrt(total[key+'_square_sum']/n) if n else None
                       for key in ('hidden', 'pre_norm', 'input')}
            metrics.update(delta_rms=math.sqrt(total['delta_square_sum']/d) if d else None,
                relative_delta_rms=math.sqrt(total['delta_square_sum']/max(total['previous_square_sum'], d*1e-24)) if d else None,
                cosine=total['cosine_sum']/d if d else None,
                ce=total['ce_sum']/c if c else None, entropy=total['entropy_sum']/c if c else None)
            if any(value is not None and not math.isfinite(value) for value in metrics.values()):
                raise ValueError('Nonfinite derived stability metric')
            regions[name] = {'sums': total, 'metrics': metrics}
        if regions['all']['sums']['positions'] != tokens or regions['all']['sums']['ce_targets'] != targets:
            raise ValueError('All-position probe accounting differs')
        output.append({'pass': index+1, 'regions': regions})
    return {'schema': SCHEMA, 'passes': output, 'input_tokens': tokens,
            'ce_targets': targets, 'beta': beta, 'policy': 'common_fp32_no_jitter_v1'}


class StabilityEvaluationController:
    """Compose ordinary dev evaluation and Figure3-style probes at clean boundaries."""
    def __init__(self, ordered_controller, probe_plan):
        self.ordered = ordered_controller; self.probe_plan = probe_plan
        self.probe_published = set()
        if (probe_plan['schema'] != SCHEMA or self.recipe.arm != 'F'
                or probe_plan['world_size'] != self.coordinator.world_size
                or probe_plan['length'] != self.recipe.sequence_length):
            raise ValueError('Probe controller model/partition/length mismatch')
        self.report['stability_probe_policy'] = probe_plan
        self.report['stability_probes'] = []

    def __getattr__(self, name):
        return getattr(self.ordered, name)

    def metrics_for(self, completed):
        metrics = self.ordered.metrics_for(completed)
        entries = [row for row in self.report['stability_probes']
                   if row['after_update'] == completed and row['status'] == 'completed']
        if entries:
            for p in entries[-1]['result']['passes']:
                for region, values in p['regions'].items():
                    for key, value in values['metrics'].items():
                        if value is not None:
                            metrics[f'dev/stability/{region}/pass_{p["pass"]}/{key}'] = value
        return metrics

    def run_if_due(self, completed, *, model, runner, generators, training_data,
                   boundary, persist, log_metrics=True):
        self.ordered.run_if_due(completed, model=model, runner=runner, generators=generators,
            training_data=training_data, boundary=boundary, persist=persist, log_metrics=log_metrics)
        plan = self.probe_plan
        if completed not in plan['scheduled_updates'] or completed in self.probe_published:
            return
        c = self.coordinator; started = time.perf_counter()
        passes = plan['deep_passes'] if completed in plan['deep_updates'] else plan['shallow_passes']
        def state():
            return {'cursor': asdict(training_data.cursor()), 'graph': graph_boundary(runner),
                    'complete': boundary() if self.acceptance else None}
        before = c.call('stability probe entry boundary', state)
        entry = {'schema': SCHEMA, 'after_update': completed, 'num_passes': passes,
                 'status': 'running', 'panel': 'dev-main'}
        self.report['stability_probes'].append(entry)
        c.call('stability probe initial evidence', persist, rank_zero=True)
        try:
            panel = plan['panel']; planned = panel['fixed_plan']['updates'][0]
            def local():
                with ordered.OrderedCampaignData(legacy.local_path(self.data_spec['corpus']),
                        legacy.local_path(panel['index'])) as dev:
                    if (dev.manifest_sha256 != panel['index_manifest_sha256']
                            or dev.manifest['panel'] != 'dev-main' or dev.split != 'dev'):
                        raise ValueError('Probe dev-main authority changed')
                    cursor = dev.cursor(); logical = dev.peek_update(cursor, panel['target_valid_tokens'])
                    if logical is None:
                        raise ValueError('Probe dev prefix exhausted')
                    actual = {'start_cursor': asdict(cursor), 'next_cursor': asdict(logical.next_cursor),
                        'counts': asdict(logical.counts), 'target_valid_tokens': panel['target_valid_tokens'],
                        'overshoot_tokens': logical.overshoot_tokens,
                        'unique_documents_in_this_update': logical.unique_document_count,
                        'membership_sha256': legacy.digest([asdict(row) for row in logical.rows])}
                    if any(planned[key] != value for key, value in actual.items()):
                        raise ValueError('Probe membership differs from fixed metadata plan')
                    packed = dev.rank_batches(logical, rank=c.rank, world_size=c.world_size,
                                              physical_batch_size=plan['physical_batch_per_rank'])
                    with evaluation_runtime(model, device=self.device, generators=generators) as preservation:
                        rows = [probe_batch(model, batch.to(self.device), self.recipe, passes=passes,
                                    vocab_position_chunk=plan['vocab_position_chunk']) for batch in packed.batches]
                    if dev.cursor() != cursor:
                        raise ValueError('Stability probe advanced its dev reader')
                    dev.validate_integrity()
                    return {'rows': rows, 'preservation': preservation, 'accounting': packed.accounting}
            local_result = c.call('stability probe local passes and restoration', local)
            all_results = c.gather(local_result)
            result = c.call('stability probe global accounting', lambda: summarize(
                [row for rank in all_results for row in rank['rows']],
                expected_tokens=planned['counts']['valid_tokens'],
                expected_ce_targets=planned['counts']['ce_targets']))
            exact = c.gather(before == c.call('stability probe exit boundary', state))
            if not all(exact):
                raise RuntimeError('Stability probe changed captured training boundary')
            entry.update(status='completed', result=result, by_rank=all_results,
                membership_sha256=planned['membership_sha256'],
                index_manifest_sha256=panel['index_manifest_sha256'],
                training_boundary_exact_by_rank=exact, total_seconds=time.perf_counter()-started)
            def publish():
                write_json(self.output_dir/f'stability-update-{completed:06d}.json', entry)
                persist()
                if self.tracker is not None and log_metrics:
                    self.tracker.log({'update': completed, **self.metrics_for(completed)}, step=completed)
            c.call('stability probe publication', publish, rank_zero=True)
            self.probe_published.add(completed)
        except BaseException:
            entry['status'] = 'failed'
            raise
