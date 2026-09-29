"""Named ordered dev panels using the accepted common-FP32 evaluator.

Membership is a declared finite prefix of each named panel. Evaluation batching
is explicit and independent of training batching. Overlapping main/source
panels are reported separately and never pooled. Confirmation is not routine dev.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.campaign_recipe import ARMS
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_pilot_ordered_data as ordered
from scripts.olmo_campaign_eval_control import TERMS, due, graph_boundary, summarize
from scripts.olmo_campaign_evaluation import evaluation_runtime, per_pass_sums

SCHEMA = 'olmo-pilot-evaluation-control-v1'
KIND = 'ordered_named_dev_panels_v1'
DEV_PANELS = tuple(name for name in ordered.panel_names() if name.startswith('dev-'))


def validate_policy(policy):
    if policy.get('kind') == 'deferred':
        legacy.exact_fields(policy, ('kind', 'reason'), 'evaluation')
        if not isinstance(policy['reason'], str) or not policy['reason'].strip():
            raise ValueError('Deferred evaluation requires a reason')
        return
    legacy.exact_fields(policy, ('kind', 'panels', 'physical_batch_by_arm', 'every_updates',
        'precision', 'feedback_jitter', 'report_passes', 'generation'), 'ordered evaluation')
    if (policy['kind'] != KIND or policy['precision'] != 'fp32'
            or type(policy['feedback_jitter']) not in (int, float) or policy['feedback_jitter'] != 0
            or policy['report_passes'] != 'all_trained_passes' or policy['generation'] != 'not_implemented'):
        raise ValueError('Require explicit common-FP32/no-jitter/all-pass dev evaluation')
    legacy.integer(policy['every_updates'], 'every_updates')
    panels = policy['panels']
    if not isinstance(panels, list) or not 1 <= len(panels) <= len(DEV_PANELS):
        raise ValueError('Declare a bounded nonempty list of named dev panels')
    seen = set()
    for panel in panels:
        legacy.exact_fields(panel, ('name', 'target_valid_tokens'), 'evaluation panel')
        if panel['name'] not in DEV_PANELS or panel['name'] in seen:
            raise ValueError('Routine evaluation accepts distinct named dev panels only; no confirmation')
        seen.add(panel['name'])
        legacy.integer(panel['target_valid_tokens'], 'target_valid_tokens')
    batches = policy['physical_batch_by_arm']
    if not isinstance(batches, dict) or not batches or set(batches)-set(ARMS):
        raise ValueError('Declare evaluation physical batches for known arms')
    for arm, batch in batches.items():
        legacy.integer(batch, 'evaluation physical batch for '+arm)


def _suite(data_spec):
    root = legacy.local_path(data_spec['suite'])
    manifest, _ = ordered._pinned_json(root/'manifest.json', data_spec['suite_manifest_sha256'])
    if (manifest.get('schema') != ordered.SUITE_SCHEMA
            or manifest.get('identity_sha256') != ordered._digest({k:v for k,v in manifest.items() if k != 'identity_sha256'})
            or set(manifest.get('panels', {})) != set(ordered.panel_names())):
        raise ValueError('Ordered evaluation suite identity or panel inventory differs')
    return root, manifest


def _panel_path(root, suite, name):
    record = suite['panels'][name]
    if record['path'] != 'panels/'+name:
        raise ValueError('Ordered named-panel path differs')
    return root/record['path']


def resolve_evaluation(data_spec, policy, *, length, updates, partitions):
    """Resolve membership and allocation from metadata; never open token payloads."""
    validate_policy(policy)
    result = {'declaration': policy, 'execution': 'not_run'}
    if policy['kind'] == 'deferred':
        return result
    if set(partitions) != set(policy['physical_batch_by_arm']):
        raise ValueError('Evaluation physical batches must cover exactly the declared training arms')
    legacy.integer(updates, 'updates')
    eval_partitions = {}
    for arm, (world, _training_batch) in partitions.items():
        legacy.integer(world, 'evaluation world size')
        eval_partitions[arm] = (world, policy['physical_batch_by_arm'][arm])
    corpus = legacy.local_path(data_spec['corpus'])
    root, suite = _suite(data_spec)
    train_path = _panel_path(root, suite, 'train')
    if (train_path.resolve() != legacy.local_path(data_spec['index']).resolve()
            or suite['panels']['train']['manifest_sha256'] != data_spec['index_manifest_sha256']):
        raise ValueError('Training index differs from named suite authority')
    panels = {}
    with ordered.OrderedCampaignData(corpus, train_path) as train:
        if (train.split != 'train' or train.length != length
                or train.manifest_sha256 != data_spec['index_manifest_sha256']):
            raise ValueError('Ordered training split/context/authority differs')
        for declaration in policy['panels']:
            name = declaration['name']; index = _panel_path(root, suite, name)
            with ordered.OrderedCampaignData(corpus, index) as dev:
                if (dev.split != 'dev' or dev.length != length or dev.manifest['panel'] != name
                        or dev.manifest_sha256 != suite['panels'][name]['manifest_sha256']
                        or dev.manifest['identity_sha256'] != suite['panels'][name]['identity_sha256']
                        or any(train.manifest[key] != dev.manifest[key] or dev.manifest[key] != suite[key]
                            for key in ('corpus_manifest_sha256', 'corpus_config_sha256', 'corpus_files',
                                'vocab_size', 'eos_id', 'pad_id', 'tokenizer', 'token_dtype', 'recipe',
                                'recipe_sha256', 'round_id', 'selection_authority', 'acquisition_authority',
                                'source_selection', 'exclusion_authority'))
                        or train.manifest['policy'] != dev.manifest['policy']):
                    raise ValueError('Named dev panel differs from training/suite authority')
                fixed = legacy.plan_updates(dev,
                    {'updates': 1, 'target_valid_tokens_per_update': declaration['target_valid_tokens']},
                    eval_partitions)
                panels[name] = {'index': str(index), 'index_manifest_sha256': dev.manifest_sha256,
                    'panel_identity_sha256': dev.manifest['identity_sha256'], 'fixed_plan': fixed,
                    'selection': 'fixed_ordered_prefix_from_chunk_zero',
                    'target_valid_tokens': declaration['target_valid_tokens'],
                    'available_panel_tokens': dev.total_tokens}
    result.update(schema=SCHEMA, panels=panels,
        scheduled_updates=list(range(policy['every_updates'], updates+1, policy['every_updates'])),
        partition_by_arm={arm: {'world_size': world, 'physical_batch_per_rank': batch}
                          for arm, (world, batch) in eval_partitions.items()},
        overlap_policy='Report each panel separately; main/source overlap is not independent replication')
    return result


class EvaluationController:
    """Shared-engine interface with evaluation-owned physical allocation."""
    def __init__(self, plan, data_spec, recipe, *, coordinator, device, batch_size,
                 tracker, report, output_dir, acceptance):
        self.plan, self.data_spec, self.recipe = plan, data_spec, recipe
        self.coordinator, self.device = coordinator, device
        self.tracker, self.report, self.output_dir = tracker, report, Path(output_dir)
        self.acceptance = acceptance; self.published = set()
        self.batch_size = None
        if plan.get('declaration', {}).get('kind') != 'deferred':
            partition = plan['partition_by_arm'][recipe.arm]
            if partition['world_size'] != coordinator.world_size:
                raise ValueError('Live evaluation rank count differs from resolved partition')
            self.batch_size = partition['physical_batch_per_rank']
        report['evaluation_policy'] = {'plan': plan,
            'training_physical_batch_per_rank': batch_size,
            'evaluation_physical_batch_per_rank': self.batch_size,
            'resume': 'repeat_scheduled_restored_boundary_once_per_segment',
            'failure': 'earlier committed checkpoint remains authoritative; no save of unverified state'}
        report['evaluations'] = []

    def metrics_for(self, completed):
        entries = [entry for entry in self.report['evaluations']
            if entry['after_update'] == completed and entry['status'] == 'completed']
        if not entries:
            return {}
        metrics = {}
        for name, panel in entries[-1]['panels'].items():
            prefix = 'dev/'+name.removeprefix('dev-'); result = panel['result']
            metrics[prefix+'/input_tokens'] = result['input_tokens']
            for row in result['passes']:
                for term in TERMS:
                    value = row['means'][term]
                    if value is not None:
                        metrics[f'{prefix}/pass_{row["index"]+1}/{term}'] = value
                    metrics[f'{prefix}/pass_{row["index"]+1}/{term}_targets'] = row['counts'][term]
            for term, value in result['aggregate']['means'].items():
                if value is not None:
                    metrics[f'{prefix}/aggregate/{term}'] = value
            metrics[prefix+'/aggregate/objective'] = result['objective']
        return metrics

    def run_if_due(self, completed, *, model, runner, generators, training_data,
                   boundary, persist, log_metrics=True):
        if not due(completed, self.plan.get('scheduled_updates', []), self.published):
            return
        c = self.coordinator; started = time.perf_counter()
        def state():
            return {'cursor': asdict(training_data.cursor()), 'graph': graph_boundary(runner),
                    'complete': boundary() if self.acceptance else None}
        before = c.call('evaluation entry state', state)
        entry = {'schema': SCHEMA, 'after_update': completed, 'status': 'running', 'panels': {}}
        self.report['evaluations'].append(entry)
        c.call('evaluation started evidence', persist, rank_zero=True)
        try:
            for name, panel in self.plan['panels'].items():
                panel_started = time.perf_counter()
                def local():
                    with ordered.OrderedCampaignData(legacy.local_path(self.data_spec['corpus']),
                            legacy.local_path(panel['index'])) as dev:
                        if (dev.manifest_sha256 != panel['index_manifest_sha256']
                                or dev.manifest['panel'] != name or dev.split != 'dev'):
                            raise ValueError('Named dev authority changed')
                        cursor = dev.cursor(); logical = dev.peek_update(cursor, panel['target_valid_tokens'])
                        if logical is None:
                            raise ValueError('Declared dev prefix exhausted')
                        planned = panel['fixed_plan']['updates'][0]
                        actual = {'start_cursor': asdict(cursor), 'next_cursor': asdict(logical.next_cursor),
                            'counts': asdict(logical.counts), 'target_valid_tokens': panel['target_valid_tokens'],
                            'overshoot_tokens': logical.overshoot_tokens,
                            'unique_documents_in_this_update': logical.unique_document_count,
                            'membership_sha256': legacy.digest([asdict(row) for row in logical.rows])}
                        if any(planned[key] != value for key, value in actual.items()):
                            raise ValueError('Materialized dev prefix differs from fixed metadata plan')
                        packed = dev.rank_batches(logical, rank=c.rank, world_size=c.world_size,
                                                  physical_batch_size=self.batch_size)
                        with evaluation_runtime(model, device=self.device, generators=generators) as preservation:
                            rows = [per_pass_sums(model, batch.to(self.device), self.recipe)
                                    for batch in packed.batches]
                        if dev.cursor() != cursor:
                            raise ValueError('Evaluation advanced its dev reader')
                        dev.validate_integrity()
                        return {'rows': rows, 'preservation': preservation, 'accounting': packed.accounting,
                                'cursor': asdict(cursor), 'seconds': time.perf_counter()-panel_started}
                local_result = c.call('named heldout local evaluation and restoration: '+name, local)
                all_results = c.gather(local_result)
                raw = panel['fixed_plan']['updates'][0]['counts']
                expected = {'ce': raw['ce_targets'],
                    'latent': raw['latent_pairs'] if self.recipe.nextlat else 0,
                    'kl': raw['kl_triples'] if self.recipe.nextlat else 0}
                result = c.call('named global evaluation accounting: '+name, lambda: summarize(
                    [row for rank in all_results for row in rank['rows']],
                    expected_counts=expected, expected_tokens=raw['valid_tokens']))
                entry['panels'][name] = {'result': result, 'by_rank': all_results,
                    'index_manifest_sha256': panel['index_manifest_sha256'],
                    'membership_sha256': panel['fixed_plan']['updates'][0]['membership_sha256'],
                    'total_seconds': time.perf_counter()-panel_started}
            after = c.call('evaluation exit state', state)
            exact = c.gather(before == after)
            if not all(exact):
                raise RuntimeError('Evaluation changed live captured training boundary')
            entry.update(status='completed', training_boundary_exact_by_rank=exact,
                         total_seconds=time.perf_counter()-started)
            def publish():
                write_json(self.output_dir/f'evaluation-update-{completed:06d}.json', entry)
                persist()
                if self.tracker is not None and log_metrics:
                    self.tracker.log({'update': completed, **self.metrics_for(completed)}, step=completed)
            c.call('evaluation publication', publish, rank_zero=True)
            self.published.add(completed)
        except BaseException:
            entry['status'] = 'failed'
            raise
