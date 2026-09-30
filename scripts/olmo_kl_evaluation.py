"""Named dev evaluation for the explicit KL1 versus KL0.1 continuation.

The accepted common-FP32 per-pass model evaluation is reused unchanged. This
module retains the existing ordered controller and scalar reducer checks; the
only objective extension is positive KL weight 0.1 with latent weight 1. Raw
CE/latent/KL sums and counts are never rescaled to disguise the changed objective.
No historical source file is patched or monkeypatched.
"""
from dataclasses import asdict
import math
from pathlib import Path
import time

from cdrm.pretrained.artifacts import write_json
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_pilot_ordered_data as ordered
from scripts.olmo_campaign_eval_control import TERMS, due, graph_boundary
from scripts.olmo_campaign_evaluation import evaluation_runtime, per_pass_sums
from scripts.olmo_pilot_eval_control import EvaluationController as OrderedEvaluationController

SCHEMA = 'olmo-kl-continuation-evaluation-v1'


def _kl_weight(value):
    if type(value) not in (int, float) or value not in (1., .1):
        raise ValueError('KL continuation supports only explicit weights 1 or 0.1')
    return float(value)


def summarize(rows,*,expected_counts,expected_tokens,kl_weight):
    """Sum numerators and separate integer counts globally before dividing."""
    kl_weight = _kl_weight(kl_weight)
    if not rows:raise ValueError('Evaluation must include physical rows, including dummies')
    first=rows[0];count=len(first['passes'])
    coefficients={t:([1.] if count==1 else ([.5]+[.5/(count-1)]*(count-1) if t=='ce' else [1./count]*count)) for t in TERMS} if count else {}
    if (first['schema']!='olmo-campaign-local-evaluation-v1' or first['policy']!='common_fp32_no_jitter_v1'
            or count != 4 or first['mode'].get('num_passes')!=count
            or first['mode'].get('feedback_jitter')!=0
            or set(first['weights'])!=set(TERMS) or set(first['enabled'])!=set(TERMS)
            or any(type(v) not in (int,float) for v in first['weights'].values())
            or first['weights'] != {'ce':1., 'latent':1., 'kl':kl_weight}
            or any(type(first['enabled'][t]) is not bool or first['enabled'][t]!=bool(first['weights'][t]) for t in TERMS)
            or first['term_pass_coefficients']!=coefficients):
        raise ValueError('Invalid canonical evaluation metadata or coefficients')
    metadata=('schema','weights','enabled','mode','policy','term_pass_coefficients')
    for row in rows:
        if (any(row[key]!=first[key] for key in metadata) or len(row['passes'])!=count
                or [p['index'] for p in row['passes']]!=list(range(count))
                or type(row['input_tokens']) is not int or row['input_tokens']<0):
            raise ValueError('Per-pass evaluation contract differs across physical batches/ranks')
        for values in [row,*row['passes']]:
            if set(values['counts'])!=set(TERMS) or any(type(v) is not int or v<0 for v in values['counts'].values()):
                raise ValueError('Target counts must be nonnegative integers')
        if any(p['counts']!=row['counts'] for p in row['passes']):
            raise ValueError('Pass denominators differ')
    counts={t:sum(row['counts'][t] for row in rows) for t in TERMS}
    tokens=sum(row['input_tokens'] for row in rows)
    if counts!=expected_counts or tokens!=expected_tokens:
        raise ValueError('Observed global targets/tokens differ from independent packed plan')
    def reduce_sums(values):
        if any(set(v)!=set(TERMS) or any(type(x) not in (int,float) or not math.isfinite(x) for x in v.values()) for v in values):
            raise ValueError('Invalid or nonfinite loss sums')
        sums={t:math.fsum(v[t] for v in values) for t in TERMS}
        for t in TERMS:
            if not first['enabled'][t] and (counts[t]!=0 or sums[t]!=0):
                raise ValueError('Disabled auxiliary term has targets or loss')
            if first['enabled'][t] and counts[t]<=0:
                raise ValueError('Enabled evaluation term has no global targets')
        return {'sums':sums,'counts':dict(counts),
                'means':{t:sums[t]/counts[t] if counts[t] else None for t in TERMS}}
    passes=[{'index':i,**reduce_sums([row['passes'][i]['sums'] for row in rows])} for i in range(count)]
    aggregate=reduce_sums([row['aggregate_sums'] for row in rows])
    # Canonical aggregation performs FP32 arithmetic per physical batch, so this
    # diagnostic comparison permits only its accumulated rounding, never an
    # alternative denominator or loss weighting.
    reconstructed={t:math.fsum(first['term_pass_coefficients'][t][i]*passes[i]['sums'][t]
                              for i in range(count)) for t in TERMS}
    if any(not math.isclose(reconstructed[t],aggregate['sums'][t],rel_tol=2e-6,abs_tol=1e-6) for t in TERMS):
        raise ValueError('Canonical aggregate differs from declared per-pass coefficients')
    objective=math.fsum(first['weights'][t]*(aggregate['means'][t] or 0.) for t in TERMS)
    if not math.isfinite(objective):raise ValueError('Nonfinite global evaluation objective')
    return {'schema':SCHEMA,'passes':passes,'aggregate':aggregate,
        'objective':objective,
        'input_tokens':tokens,'enabled':first['enabled'],'weights':first['weights'],
        'term_pass_coefficients':first['term_pass_coefficients'],
        'reconstructed_aggregate_sums':reconstructed,'policy':first['policy']}

class EvaluationController(OrderedEvaluationController):
    """Keep accepted scheduling, metrics and preservation for a declared objective.

    The parent has no reducer-injection seam. Its run_if_due body is copied
    here with a live objective check and the explicit KL-aware reducer call;
    data materialization, runtime restoration and publication stay unchanged.
    """
    def __init__(self, *args, kl_weight, **kwargs):
        self.kl_weight = _kl_weight(kl_weight)
        super().__init__(*args, **kwargs)
        if self.recipe.arm not in ('NF', 'NFR') or self.recipe.mode().num_passes != 4:
            raise ValueError('KL continuation evaluation requires a K4 NF/NFR model')
        self.report['evaluation_policy']['objective'] = {
            'schema': SCHEMA, 'weights': {'ce':1., 'latent':1., 'kl':self.kl_weight},
            'raw_terms': 'Existing unweighted per-pass sums divided by separate counts',
            'weighted_total': 'Branch-specific objective; not a cross-branch quality metric'}

    def run_if_due(self, completed, *, model, runner, generators, training_data,
                   boundary, persist, log_metrics=True):
        if not due(completed, self.plan.get('scheduled_updates', []), self.published):
            return
        if (model.objective_weights() != {'ce':1., 'latent':1., 'kl':self.kl_weight}
                or model.predictor.config.lambda_kl != self.kl_weight):
            raise ValueError('Live model objective differs from declared KL continuation')
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
                    expected_counts=expected, expected_tokens=raw['valid_tokens'], kl_weight=self.kl_weight))
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
