"""Global denominator and pinned real dev-plan tests; CPU and local files only."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest

from cdrm.pretrained import document_shards as shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts import olmo_campaign_eval_control as control
from scripts import olmo_campaign_evaluation as core


@pytest.fixture
def evaluation_paths(tmp_path,monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[int(value) for value in text.split()])
    monkeypatch.setattr(shards,'_load_tokenizer',lambda _:Tokenizer())
    raw=b''.join((json.dumps({'id':str(i),'text':' '.join(map(str,[100+i,*range(3,12+i%7)]))})+'\n').encode()
                 for i in range(64))
    path=tmp_path/'source.jsonl';path.write_bytes(raw)
    source=LocalJSONLSource(SourcePin('fixture','https://example.invalid/fixed','pinned',hashlib.sha256(raw).hexdigest()),path)
    corpus=tmp_path/'corpus'
    shards.prepare_document_shards([source],corpus,tokenizer_path='fixture',
        split_policy=SplitPolicy(21,(('train',3),('dev',1))),max_documents_per_shard=8)
    indices={}
    for split in ('train','dev'):
        index=tmp_path/('index-'+split)
        build_packed_index(corpus,index,split=split,length=16)
        indices[split]=index
    return corpus,indices


def policy(index,**updates):
    return {'kind':'finite_pass_teacher_forced','index':str(index),
        'index_manifest_sha256':control.legacy.sha256_file(index/'manifest.json'),
        'split':'dev','target_valid_tokens':48,'every_updates':1,'precision':'fp32',
        'feedback_jitter':0.,'report_passes':'all_trained_passes','generation':'not_implemented',**updates}


def synthetic_row(counts,tokens,scale=1.,*,enabled=True,passes=4):
    weights={'ce':1.,'latent':float(enabled),'kl':float(enabled)}
    coefficients={'ce':[1.] if passes==1 else [.5]+[1/6]*3,
                  'latent':[1/passes]*passes,'kl':[1/passes]*passes}
    records=[]
    for index in range(passes):
        sums={term:counts[term]*(scale+index+(0.25 if term=='latent' else 0.)) for term in control.TERMS}
        records.append({'index':index,'sums':sums,'counts':dict(counts)})
    return {'schema':core.SCHEMA,'passes':records,
        'aggregate_sums':{term:sum(coefficients[term][i]*records[i]['sums'][term] for i in range(passes)) for term in control.TERMS},
        'counts':dict(counts),'weights':weights,'term_pass_coefficients':coefficients,
        'enabled':{term:bool(value) for term,value in weights.items()},'input_tokens':tokens,
        'mode':{'enabled':passes==4,'num_passes':passes,'feedback_jitter':0.},
        'policy':'common_fp32_no_jitter_v1'}


def samples(enabled=True):
    first={'ce':2,'latent':1 if enabled else 0,'kl':1 if enabled else 0}
    second={'ce':6,'latent':3 if enabled else 0,'kl':2 if enabled else 0}
    return [synthetic_row(first,4,1.,enabled=enabled),synthetic_row(second,8,3.,enabled=enabled),
            synthetic_row(dict.fromkeys(control.TERMS,0),0,enabled=enabled)]


@pytest.mark.parametrize('enabled',[False,True])
def test_global_aggregation_uses_target_sums_not_rank_means_and_keeps_dummy(enabled):
    rows=samples(enabled)
    counts={'ce':8,'latent':4 if enabled else 0,'kl':3 if enabled else 0}
    result=control.summarize(rows,expected_counts=counts,expected_tokens=12)
    assert result['passes'][0]['means']['ce']==2.5
    assert result['passes'][0]['means']['ce']!=(1.+3.)/2
    for index,row in enumerate(result['passes']):
        assert row['counts']==counts and row['means']['ce']==2.5+index
    assert result['aggregate']['means']['ce']==pytest.approx(3.5)
    assert result['input_tokens']==12
    if enabled:
        expected=sum(result['aggregate']['means'].values())
        assert result['objective']==pytest.approx(expected)
    else:
        assert result['aggregate']['means']['latent'] is result['aggregate']['means']['kl'] is None
        assert result['objective']==result['aggregate']['means']['ce']
        assert result['aggregate']['sums']['latent']==result['aggregate']['sums']['kl']==0


@pytest.mark.parametrize('mutation',['count_float','negative_count','pass_count','pass_order','rank_contract',
    'nonfinite_sum','missing_term','wrong_aggregate','disabled_loss','global_counts','global_tokens',
    'bad_schema','nan_weight','negative_weight','bad_coefficients'])
def test_malformed_scalar_contract_or_accounting_is_rejected(mutation):
    enabled=mutation!='disabled_loss';rows=samples(enabled)
    counts={'ce':8,'latent':4 if enabled else 0,'kl':3 if enabled else 0};tokens=12
    if mutation=='count_float':rows[0]['counts']['ce']=2.
    elif mutation=='negative_count':rows[0]['counts']['ce']=-2
    elif mutation=='pass_count':rows[0]['passes'].pop()
    elif mutation=='pass_order':rows[0]['passes'][0]['index']=1
    elif mutation=='rank_contract':rows[1]['weights']['ce']=2.
    elif mutation=='nonfinite_sum':rows[0]['passes'][0]['sums']['ce']=float('nan')
    elif mutation=='missing_term':del rows[0]['aggregate_sums']['kl']
    elif mutation=='wrong_aggregate':rows[0]['aggregate_sums']['ce']+=1
    elif mutation=='disabled_loss':rows[0]['passes'][0]['sums']['latent']=1.
    elif mutation=='global_counts':counts['ce']+=1
    elif mutation=='global_tokens':tokens+=1
    elif mutation=='bad_schema':
        for row in rows:row['schema']='wrong'
    elif mutation=='nan_weight':
        for row in rows:row['weights']['ce']=float('nan')
    elif mutation=='negative_weight':
        for row in rows:row['weights']['ce']=-1.
    else:
        for row in rows:
            row['term_pass_coefficients']['ce']=[1.,0.,0.,0.]
            row['aggregate_sums']['ce']=row['passes'][0]['sums']['ce']
    with pytest.raises(ValueError):control.summarize(rows,expected_counts=counts,expected_tokens=tokens)


def test_actual_dev_plan_is_disjoint_pure_pinned_and_counts_match_materialized_masks(evaluation_paths):
    corpus,indices=evaluation_paths
    spec={'corpus':str(corpus),'index':str(indices['train']),
          'index_manifest_sha256':control.legacy.sha256_file(indices['train']/'manifest.json')}
    declared=policy(indices['dev'])
    resolved=control.resolve_evaluation(spec,declared,length=16,updates=3,partitions={'NFR':(2,1)})
    assert resolved['scheduled_updates']==[1,2,3]
    planned=resolved['fixed_plan']['updates'][0]
    counts=dict.fromkeys(control.TERMS,0)
    with PackedCampaignData(corpus,indices['dev']) as data:
        origin=data.cursor();logical=data.peek_update(origin,48)
        assert planned['start_cursor']==asdict(origin)
        assert planned['next_cursor']==asdict(logical.next_cursor)
        assert logical.counts.cross_document_ce_targets>0
        for rank in range(2):
            packed=data.rank_batches(logical,rank=rank,world_size=2,physical_batch_size=1)
            for batch in packed.batches:
                masks=build_nextlat_masks(batch,document_policy='continuous-stream-v1')
                for term in counts:counts[term]+=int(masks[term].sum())
            if rank==1:assert packed.empty_rows==1
        assert data.cursor()==origin
    assert counts=={term:planned['counts'][field] for term,field in
                   [('ce','ce_targets'),('latent','latent_pairs'),('kl','kl_triples')]}
    assert planned['counts']['valid_tokens']==48


@pytest.mark.parametrize('mutation',['dev_pin','train_pin','wrong_split','length','budget'])
def test_resolver_rejects_different_authority_or_exhausted_plan(evaluation_paths,mutation):
    corpus,indices=evaluation_paths
    spec={'corpus':str(corpus),'index':str(indices['train']),
          'index_manifest_sha256':control.legacy.sha256_file(indices['train']/'manifest.json')}
    declared=policy(indices['dev']);length=16
    if mutation=='dev_pin':declared['index_manifest_sha256']='0'*64
    elif mutation=='train_pin':spec['index_manifest_sha256']='0'*64
    elif mutation=='wrong_split':declared=policy(indices['train'])
    elif mutation=='length':length=32
    else:declared['target_valid_tokens']=10**9
    with pytest.raises(ValueError):control.resolve_evaluation(spec,declared,length=length,updates=3,partitions={'B':(2,1)})


@pytest.mark.parametrize('mutation',['precision','jitter','boolean_jitter','interval','target','split','passes','generation','extra'])
def test_policy_is_explicit_and_fail_closed(evaluation_paths,mutation):
    _,indices=evaluation_paths;declared=policy(indices['dev'])
    if mutation=='precision':declared['precision']='bf16_mixed'
    elif mutation=='jitter':declared['feedback_jitter']=.02
    elif mutation=='boolean_jitter':declared['feedback_jitter']=False
    elif mutation=='interval':declared['every_updates']=0
    elif mutation=='target':declared['target_valid_tokens']=0
    elif mutation=='split':declared['split']='train'
    elif mutation=='passes':declared['report_passes']='final'
    elif mutation=='generation':declared['generation']='greedy'
    else:declared['extra']=True
    with pytest.raises(ValueError):control.validate_policy(declared)


def test_deferred_and_repeated_segment_scheduling_do_not_invent_evaluation():
    deferred={'kind':'deferred','reason':'fixed acceptance reference'}
    assert control.resolve_evaluation({},deferred,length=16,updates=3,partitions={})=={'declaration':deferred,'execution':'not_run'}
    assert not control.due(0,[1,3],set()) and control.due(1,[1,3],set())
    assert not control.due(1,[1,3],{1}) and control.due(1,[1,3],set())
    assert control.due(3,[1,3],set())  # Completed-plan resume can report its scheduled boundary.


def test_wandb_metrics_only_include_completed_requested_boundary(tmp_path):
    report={}
    controller=control.EvaluationController({},None,None,coordinator=None,device='cpu',
        batch_size=1,tracker=None,report=report,output_dir=tmp_path,acceptance=False)
    result=control.summarize(samples(False),expected_counts={'ce':8,'latent':0,'kl':0},expected_tokens=12)
    report['evaluations']=[{'after_update':1,'status':'completed','result':result},
                          {'after_update':2,'status':'running'},
                          {'after_update':3,'status':'failed'}]
    metrics=controller.metrics_for(1)
    assert metrics['dev/input_tokens']==12 and metrics['dev/pass_1/ce']==2.5
    assert metrics['dev/pass_4/latent_targets']==0
    assert 'dev/pass_1/latent' not in metrics and 'dev/aggregate/kl' not in metrics
    assert all(controller.metrics_for(index)=={} for index in (0,2,3,4))
    assert all(key.startswith('dev/') for key in metrics)
