"""Named ordered dev-prefix identity, allocation and live-state preservation."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model, build_campaign_adamw
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_pilot_eval_control as control
from scripts.olmo_lm_common import tree_digests
from test_pilot_ordered_data import fixture, cpu_tokenizer, raw_json


@pytest.fixture
def data(tmp_path):
    return fixture(tmp_path)


def spec(data):
    return {'corpus': str(data.corpus), 'index': str(data.output/'panels/train'),
        'index_manifest_sha256': data.manifest['panels']['train']['manifest_sha256'],
        'suite': str(data.output),
        'suite_manifest_sha256': control.legacy.sha256_file(data.output/'manifest.json')}


def policy(*, arms=('NFR',), panels=None, **changes):
    return {'kind': control.KIND, 'panels': panels or [{'name':'dev-main', 'target_valid_tokens':12}],
        'physical_batch_by_arm': dict.fromkeys(arms, 1), 'every_updates':1, 'precision':'fp32',
        'feedback_jitter':0., 'report_passes':'all_trained_passes', 'generation':'not_implemented', **changes}


def resolve(data, declared=None, *, partitions=None):
    return control.resolve_evaluation(spec(data), policy() if declared is None else declared,
        length=4, updates=3, partitions={'NFR':(2,8)} if partitions is None else partitions)


def test_named_prefix_uses_independent_batch_and_literal_masks_with_uneven_ranks(data):
    declared = policy(arms=('B','NFR'), physical_batch_by_arm={'B':2, 'NFR':1})
    planned = resolve(data, declared, partitions={'B':(2,8), 'NFR':(2,12)})
    assert planned['scheduled_updates'] == [1,2,3]
    assert planned['partition_by_arm'] == {'B':{'world_size':2,'physical_batch_per_rank':2},
                                         'NFR':{'world_size':2,'physical_batch_per_rank':1}}
    panel = planned['panels']['dev-main']
    fixed = panel['fixed_plan']['updates'][0]
    assert panel['selection'] == 'fixed_ordered_prefix_from_chunk_zero'
    assert panel['target_valid_tokens'] == 12 < panel['available_panel_tokens']
    assert fixed['allocation_by_arm']['NFR'][1]['dummy_rows'] == 1
    observed = dict.fromkeys(control.TERMS,0)
    with control.ordered.OrderedCampaignData(data.corpus, panel['index']) as dev:
        cursor = dev.cursor(); logical = dev.peek_update(cursor,12)
        for rank in (0,1):
            batches = dev.rank_batches(logical,rank=rank,world_size=2,physical_batch_size=1)
            for batch in batches.batches:
                for name, mask in build_nextlat_masks(batch,document_policy='continuous-stream-v1').items():
                    observed[name] += int(mask.sum())
        assert dev.cursor() == cursor
    assert observed == {term:fixed['counts'][field] for term,field in
                        [('ce','ce_targets'),('latent','latent_pairs'),('kl','kl_triples')]}


def test_resolution_is_metadata_only_and_named_panels_remain_separate(data,monkeypatch):
    monkeypatch.setattr(control.ordered.OrderedCampaignData,'_token_slice',
        lambda *_: (_ for _ in ()).throw(AssertionError('Unexpected token materialization')))
    declared = policy(panels=[{'name':'dev-main','target_valid_tokens':12},
                             {'name':'dev-source/cc_en_tail','target_valid_tokens':8}])
    planned = resolve(data,declared)
    assert list(planned['panels']) == ['dev-main','dev-source/cc_en_tail']
    assert [p['fixed_plan']['totals']['valid_tokens'] for p in planned['panels'].values()] == [12,8]
    assert 'totals' not in planned  # Overlapping panels have no pooled estimate.


@pytest.mark.parametrize('mutation', ['confirmation','train','unknown','duplicate','empty','path',
    'zero_target','boolean_target','zero_batch','boolean_batch','unknown_arm','zero_interval',
    'precision','jitter','boolean_jitter','passes','generation','extra'])
def test_policy_fails_closed(mutation):
    value = policy()
    if mutation == 'confirmation': value['panels'][0]['name']='confirmation-main'
    elif mutation == 'train': value['panels'][0]['name']='train'
    elif mutation == 'unknown': value['panels'][0]['name']='dev-unregistered'
    elif mutation == 'duplicate': value['panels'] *= 2
    elif mutation == 'empty': value['panels']=[]
    elif mutation == 'path': value['panels'][0]['name']='dev-source/../../train'
    elif mutation == 'zero_target': value['panels'][0]['target_valid_tokens']=0
    elif mutation == 'boolean_target': value['panels'][0]['target_valid_tokens']=True
    elif mutation == 'zero_batch': value['physical_batch_by_arm']['NFR']=0
    elif mutation == 'boolean_batch': value['physical_batch_by_arm']['NFR']=True
    elif mutation == 'unknown_arm': value['physical_batch_by_arm']={'other':1}
    elif mutation == 'zero_interval': value['every_updates']=0
    elif mutation == 'precision': value['precision']='bf16_mixed'
    elif mutation == 'jitter': value['feedback_jitter']=.02
    elif mutation == 'boolean_jitter': value['feedback_jitter']=False
    elif mutation == 'passes': value['report_passes']='last_only'
    elif mutation == 'generation': value['generation']='enabled'
    else: value['extra']='unrecognized'
    with pytest.raises(ValueError): control.validate_policy(value)


@pytest.mark.parametrize('mutation', ['suite_pin','train_pin','wrong_train','missing_arm','extra_arm',
    'exhausted','length','suite_panel_path','suite_identity','panel_pin','dev_bytes'])
def test_resolver_rejects_authority_and_allocation_changes(data,mutation):
    data_spec=spec(data); declared=policy(); partitions={'NFR':(2,8)}; length=4
    if mutation=='suite_pin': data_spec['suite_manifest_sha256']='0'*64
    elif mutation=='train_pin': data_spec['index_manifest_sha256']='0'*64
    elif mutation=='wrong_train': data_spec['index']=str(data.output/'panels/dev-main')
    elif mutation=='missing_arm': partitions={}
    elif mutation=='extra_arm': partitions['B']=(2,8)
    elif mutation=='exhausted': declared['panels'][0]['target_valid_tokens']=10**6
    elif mutation=='length': length=8
    elif mutation=='dev_bytes':
        target=data.output/'panels/dev-main/manifest.json'; target.write_bytes(target.read_bytes()+b' ')
    else:
        suite=deepcopy(data.manifest)
        if mutation=='suite_panel_path': suite['panels']['dev-main']['path']='panels/train'
        elif mutation=='panel_pin': suite['panels']['dev-main']['manifest_sha256']='0'*64
        if mutation!='suite_identity':
            suite['identity_sha256']=control.ordered._digest({k:v for k,v in suite.items() if k!='identity_sha256'})
        else: suite['identity_sha256']='0'*64
        data_spec['suite_manifest_sha256']=raw_json(data.output/'manifest.json',suite)
    with pytest.raises(ValueError):
        control.resolve_evaluation(data_spec,declared,length=length,updates=3,partitions=partitions)


class Coordinator:
    rank=0
    world_size=1
    def call(self,name,callback,*,rank_zero=False):
        return callback()
    def gather(self,value):
        return [value]


def live_setup(data,tmp_path,arm):
    declared=policy(arms=(arm,))
    planned=resolve(data,declared,partitions={arm:(1,7)})
    recipe=CampaignRecipe(arm,sequence_length=4,rt_layers=(0,1),document_policy='continuous-stream-v1')
    config=replace(OLMoConfig.tiny(),vocab_size=50280,tokenizer_vocab_size=50280,eos_token_id=50279,pad_token_id=1)
    base=OLMoTiledRTForCausalLM(config,attention_backend='sdpa',attention_precision='mixed',
        tile_backend='eager',backward_tile_backend='eager',ordinary_activation_checkpointing=True,
        cast_weights_once=True,reuse_rope=True,kv_only_writes=True,backward_memory='recompute')
    model=build_campaign_model(base,recipe).train()
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.grad=torch.zeros_like(parameter)
    optimizer.step()  # Populate real Adam moments/step state before preservation.
    report={}; tracker=SimpleNamespace(logs=[],log=lambda value,step:tracker.logs.append((step,value)))
    controller=control.EvaluationController(planned,spec(data),recipe,coordinator=Coordinator(),device='cpu',
        batch_size=7,tracker=tracker,report=report,output_dir=tmp_path,acceptance=True)
    return controller,model,optimizer,report,tracker


@pytest.mark.parametrize('arm',['B','NFR'])
def test_live_evaluation_preserves_model_adam_gradient_rng_and_cursor(data,tmp_path,arm):
    controller,model,optimizer,report,tracker=live_setup(data,tmp_path,arm)
    rng=torch.Generator().manual_seed(111)
    def boundary():
        return tree_digests({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'rng':torch.get_rng_state(),'custom_rng':rng.get_state(),
                            'gradients':{n:p.grad for n,p in model.named_parameters()}})
    before=boundary(); persisted=[]
    pointers=[(id(p),p.data_ptr(),None if p.grad is None else p.grad.data_ptr()) for p in model.parameters()]
    with control.ordered.OrderedCampaignData(data.corpus,data.output/'panels/train') as training:
        origin=asdict(training.cursor())
        kwargs=dict(model=model,runner=None,generators={'custom':rng},training_data=training,
                    boundary=boundary,persist=lambda:persisted.append(True))
        controller.run_if_due(0,**kwargs)
        assert not persisted
        controller.run_if_due(1,**kwargs)
        controller.run_if_due(1,**kwargs)
        assert asdict(training.cursor())==origin
    assert boundary()==before
    assert pointers==[(id(p),p.data_ptr(),None if p.grad is None else p.grad.data_ptr()) for p in model.parameters()]
    assert all(module.training for module in model.modules())
    assert model.backbone.backbone.attention_precision=='mixed'
    assert report['evaluation_policy']['training_physical_batch_per_rank']==7
    assert report['evaluation_policy']['evaluation_physical_batch_per_rank']==1
    assert len(report['evaluations'])==1 and len(tracker.logs)==1 and len(persisted)==2
    entry=report['evaluations'][0]
    assert entry['training_boundary_exact_by_rank']==[True]
    result=entry['panels']['dev-main']['result']
    assert result['input_tokens']==12 and len(result['passes'])==(4 if arm=='NFR' else 1)
    assert controller.metrics_for(0)==controller.metrics_for(2)=={}
    metrics=controller.metrics_for(1)
    assert metrics['dev/main/input_tokens']==12 and 'dev/main/pass_1/ce' in metrics
    assert not any(key.startswith('dev/aggregate') for key in metrics)
    if arm=='B': assert 'dev/main/aggregate/latent' not in metrics
    else: assert metrics['dev/main/pass_4/kl_targets']>0


def test_local_evaluation_exception_restores_runtime_and_never_publishes(data,tmp_path,monkeypatch):
    controller,model,optimizer,report,tracker=live_setup(data,tmp_path,'B')
    before=tree_digests(model.state_dict()); rng=torch.get_rng_state().clone()
    def fail(*args):
        torch.rand(10)
        raise ValueError('Injected local evaluation failure')
    monkeypatch.setattr(control,'per_pass_sums',fail)
    with control.ordered.OrderedCampaignData(data.corpus,data.output/'panels/train') as training:
        cursor=training.cursor()
        with pytest.raises(ValueError,match='Injected'):
            controller.run_if_due(1,model=model,runner=None,generators={},training_data=training,
                boundary=lambda:tree_digests(model.state_dict()),persist=lambda:None)
        assert training.cursor()==cursor
    assert tree_digests(model.state_dict())==before and torch.equal(torch.get_rng_state(),rng)
    assert model.training and model.backbone.backbone.attention_precision=='mixed'
    assert report['evaluations'][0]['status']=='failed'
    assert not tracker.logs and controller.metrics_for(1)=={} and not controller.published
    assert not (tmp_path/'evaluation-update-000001.json').exists()


def test_live_training_cursor_mutation_is_detected_before_publication(data,tmp_path,monkeypatch):
    controller,model,optimizer,report,tracker=live_setup(data,tmp_path,'B')
    original=control.per_pass_sums
    with control.ordered.OrderedCampaignData(data.corpus,data.output/'panels/train') as training:
        def mutate(*args):
            cursor=training.cursor()
            if cursor.next_chunk==0:
                training.commit(cursor,training.peek_update(cursor,4))
            return original(*args)
        monkeypatch.setattr(control,'per_pass_sums',mutate)
        with pytest.raises(RuntimeError,match='live captured training boundary'):
            controller.run_if_due(1,model=model,runner=None,generators={},training_data=training,
                boundary=lambda:tree_digests(optimizer.state_dict()),persist=lambda:None)
    assert report['evaluations'][0]['status']=='failed' and not tracker.logs
    assert controller.metrics_for(1)=={} and not controller.published
    assert not (tmp_path/'evaluation-update-000001.json').exists()


def test_deferred_policy_does_not_require_data_and_rank_mismatch_fails(data,tmp_path):
    deferred={'kind':'deferred','reason':'Capacity timing only'}
    assert control.resolve_evaluation({},deferred,length=4,updates=3,partitions={})=={
        'declaration':deferred,'execution':'not_run'}
    plan=resolve(data)
    with pytest.raises(ValueError,match='rank count'):
        control.EvaluationController(plan,spec(data),SimpleNamespace(arm='NFR'),
            coordinator=Coordinator(),device='cpu',batch_size=8,tracker=None,report={},
            output_dir=tmp_path,acceptance=True)
