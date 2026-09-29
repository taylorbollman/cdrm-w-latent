"""Pinned dev membership, per-pass global reduction and evaluation scheduling."""
from dataclasses import asdict
import math
from pathlib import Path
import time

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from scripts import olmo_campaign_manifest as legacy
from scripts.olmo_campaign_evaluation import evaluation_runtime, per_pass_sums
from scripts.olmo_campaign_graph_probe import pointer_snapshot
from scripts.olmo_lm_common import tree_digests

TERMS=('ce','latent','kl')
SCHEMA='olmo-campaign-evaluation-control-v1'


def validate_policy(policy):
    if policy.get('kind')=='deferred':
        legacy.exact_fields(policy,('kind','reason'),'evaluation')
        if not isinstance(policy['reason'],str) or not policy['reason'].strip():
            raise ValueError('Deferred evaluation requires a reason')
        return
    legacy.exact_fields(policy,('kind','index','index_manifest_sha256','split','target_valid_tokens',
        'every_updates','precision','feedback_jitter','report_passes','generation'),'evaluation')
    if (policy['kind']!='finite_pass_teacher_forced' or policy['split']!='dev'
            or policy['precision']!='fp32' or type(policy['feedback_jitter']) not in (int,float)
            or policy['feedback_jitter']!=0 or policy['report_passes']!='all_trained_passes'
            or policy['generation']!='not_implemented'):
        raise ValueError('Only explicit common-FP32/no-jitter finite-pass dev evaluation is supported')
    legacy.local_path(policy['index']);legacy.pin(policy['index_manifest_sha256'])
    for key in ('every_updates','target_valid_tokens'):legacy.integer(policy[key],key)


def resolve_evaluation(data_spec,policy,*,length,updates,partitions):
    validate_policy(policy)
    result={'declaration':policy,'execution':'not_run'}
    if policy['kind']=='deferred':return result
    corpus=legacy.local_path(data_spec['corpus'])
    with PackedCampaignData(corpus,legacy.local_path(data_spec['index'])) as train:
        with PackedCampaignData(corpus,legacy.local_path(policy['index'])) as dev:
            if (train.split!='train' or dev.split!='dev' or train.length!=length or dev.length!=length
                    or train.manifest_sha256!=data_spec['index_manifest_sha256']
                    or dev.manifest_sha256!=policy['index_manifest_sha256']
                    or any(train.manifest[key]!=dev.manifest[key] for key in
                           ('corpus_manifest_sha256','vocab_size','eos_id','policy'))
                    or train.pad_id!=dev.pad_id):
                raise ValueError('Dev split/context/vocabulary/corpus/policy differs from training authority')
            result['fixed_plan']=legacy.plan_updates(dev,
                {'updates':1,'target_valid_tokens_per_update':policy['target_valid_tokens']},partitions)
    result['scheduled_updates']=list(range(policy['every_updates'],updates+1,policy['every_updates']))
    return result


def due(completed,scheduled,published):
    if type(completed) is not int or completed<0:raise ValueError('Invalid completed update')
    return completed in scheduled and completed not in published


def summarize(rows,*,expected_counts,expected_tokens):
    """Sum numerators and separate integer counts globally before dividing."""
    if not rows:raise ValueError('Evaluation must include physical rows, including dummies')
    first=rows[0];count=len(first['passes'])
    coefficients={t:([1.] if count==1 else ([.5]+[.5/(count-1)]*(count-1) if t=='ce' else [1./count]*count)) for t in TERMS} if count else {}
    if (first['schema']!='olmo-campaign-local-evaluation-v1' or first['policy']!='common_fp32_no_jitter_v1'
            or count not in (1,4) or first['mode'].get('num_passes')!=count
            or first['mode'].get('feedback_jitter')!=0
            or set(first['weights'])!=set(TERMS) or set(first['enabled'])!=set(TERMS)
            or any(type(v) not in (int,float) or v not in (0.,1.) for v in first['weights'].values())
            or first['weights']['ce']!=1 or first['weights']['latent']!=first['weights']['kl']
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


def graph_boundary(runner):
    if runner is None:return None
    runner.validate_execution()
    adapter=runner.adapter
    inputs={f'input/{i}':v for i,v in enumerate(adapter.owned_inputs())}
    inputs.update({f'loss/{n}':v for n,v in vars(adapter.loss_layout).items()
                   if hasattr(v,'data_ptr')})
    inputs.update({f'forward/{i}':v for i,v in enumerate(adapter.forward_layout._owned_tensors()) if v is not None})
    return {'pointers':pointer_snapshot(runner),'contents':tree_digests(inputs),'metadata':runner.metadata,
            'graph_objects':(id(runner.local_graph),id(runner.sync_graph))}


class EvaluationController:
    def __init__(self,plan,data_spec,recipe,*,coordinator,device,batch_size,tracker,report,output_dir,acceptance):
        self.plan,self.data_spec,self.recipe=plan,data_spec,recipe
        self.coordinator,self.device,self.batch_size=coordinator,device,batch_size
        self.tracker,self.report,self.output_dir=tracker,report,Path(output_dir)
        self.acceptance=acceptance;self.published=set()
        report['evaluation_policy']={'plan':plan,'resume':'repeat_scheduled_restored_boundary_once_per_segment',
            'failure':'earlier committed checkpoint remains authoritative; no save of unverified state'}
        report['evaluations']=[]

    def metrics_for(self,completed):
        entries=[entry for entry in self.report['evaluations'] if entry['after_update']==completed and entry['status']=='completed']
        if not entries:return {}
        result=entries[-1]['result'];metrics={'dev/input_tokens':result['input_tokens']}
        for row in result['passes']:
            for term in TERMS:
                value=row['means'][term]
                if value is not None:metrics[f'dev/pass_{row["index"]+1}/{term}']=value
                metrics[f'dev/pass_{row["index"]+1}/{term}_targets']=row['counts'][term]
        for term,value in result['aggregate']['means'].items():
            if value is not None:metrics[f'dev/aggregate/{term}']=value
        metrics['dev/aggregate/objective']=result['objective']
        return metrics

    def run_if_due(self,completed,*,model,runner,generators,training_data,boundary,persist,log_metrics=True):
        if not due(completed,self.plan.get('scheduled_updates',[]),self.published):return
        c=self.coordinator;policy=self.plan['declaration'];started=time.perf_counter()
        before=c.call('evaluation entry state',lambda:{'cursor':asdict(training_data.cursor()),
            'graph':graph_boundary(runner),'complete':boundary() if self.acceptance else None})
        entry={'after_update':completed,'status':'running'}
        self.report['evaluations'].append(entry)
        c.call('evaluation started evidence',persist,rank_zero=True)
        def local():
            with PackedCampaignData(legacy.local_path(self.data_spec['corpus']),legacy.local_path(policy['index'])) as dev:
                if dev.manifest_sha256!=policy['index_manifest_sha256']:raise ValueError('Dev authority changed')
                cursor=dev.cursor();logical=dev.peek_update(cursor,policy['target_valid_tokens'])
                if logical is None:raise ValueError('Declared dev prefix exhausted')
                planned=self.plan['fixed_plan']['updates'][0]
                actual={'start_cursor':asdict(cursor),'next_cursor':asdict(logical.next_cursor),
                    'counts':asdict(logical.counts),'target_valid_tokens':policy['target_valid_tokens'],
                    'overshoot_tokens':logical.overshoot_tokens,'unique_documents_in_this_update':logical.unique_document_count,
                    'membership_sha256':legacy.digest([asdict(row) for row in logical.rows])}
                if any(planned[key]!=value for key,value in actual.items()):
                    raise ValueError('Materialized dev membership/cursor/count differs from fixed metadata plan')
                packed=dev.rank_batches(logical,rank=c.rank,world_size=c.world_size,physical_batch_size=self.batch_size)
                with evaluation_runtime(model,device=self.device,generators=generators) as preservation:
                    rows=[per_pass_sums(model,batch.to(self.device),self.recipe) for batch in packed.batches]
                if dev.cursor()!=cursor:raise ValueError('Evaluation committed or advanced its dev reader')
                dev.validate_integrity()
                return {'rows':rows,'preservation':preservation,'accounting':packed.accounting,
                        'cursor':asdict(cursor),'seconds':time.perf_counter()-started}
        try:
            local_result=c.call('heldout local evaluation and restoration',local)
            all_results=c.gather(local_result)
            after=c.call('evaluation exit state',lambda:{'cursor':asdict(training_data.cursor()),
                'graph':graph_boundary(runner),'complete':boundary() if self.acceptance else None})
            exact=c.gather(before==after)
            if not all(exact):raise RuntimeError('Evaluation changed live captured training boundary')
            raw=self.plan['fixed_plan']['updates'][0]['counts']
            expected={'ce':raw['ce_targets'],'latent':raw['latent_pairs'] if self.recipe.nextlat else 0,
                      'kl':raw['kl_triples'] if self.recipe.nextlat else 0}
            result=c.call('global evaluation accounting',lambda:summarize(
                [row for rank in all_results for row in rank['rows']],expected_counts=expected,
                expected_tokens=raw['valid_tokens']))
            entry.update(status='completed',result=result,by_rank=all_results,
                         training_boundary_exact_by_rank=exact,total_seconds=time.perf_counter()-started)
            def publish():
                write_json(self.output_dir/f'evaluation-update-{completed:06d}.json',entry)
                persist()
                if self.tracker is not None and log_metrics:
                    self.tracker.log({'update':completed,**self.metrics_for(completed)},step=completed)
            c.call('evaluation publication',publish,rank_zero=True)
            self.published.add(completed)
        except BaseException:
            entry['status']='failed'
            # All persistent checkpoints predate this observer. Unknown CUDA or
            # restoration errors are handled by the enclosing external launcher.
            raise
