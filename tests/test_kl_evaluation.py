"""Objective-fork evaluation keeps raw statistics, masks and live-state integrity."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from scripts import olmo_kl_evaluation as control
from scripts import olmo_campaign_eval_control as accepted
from scripts.olmo_lm_common import tree_digests
from test_campaign_eval_control import samples
from test_pilot_eval_control import data, live_setup
from test_pilot_ordered_data import cpu_tokenizer


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def summarize(rows, kl_weight):
    return control.summarize(rows, expected_counts={'ce':8, 'latent':4, 'kl':3},
                             expected_tokens=12, kl_weight=kl_weight)


def weighted_rows(kl_weight):
    rows=samples()
    for row in rows:
        row['weights']['kl']=kl_weight
    return rows


def test_control_equals_accepted_reducer_and_reduced_kl_keeps_all_raw_statistics():
    rows=weighted_rows(1.)
    previous=deepcopy(rows)
    ordinary=accepted.summarize(rows,expected_counts={'ce':8,'latent':4,'kl':3},expected_tokens=12)
    actual=summarize(rows,1.)
    assert rows==previous
    assert {k:v for k,v in actual.items() if k!='schema'}=={
        k:v for k,v in ordinary.items() if k!='schema'}
    reduced=summarize(weighted_rows(.1),.1)
    for key in ('passes','aggregate','input_tokens','enabled','term_pass_coefficients',
                'reconstructed_aggregate_sums','policy'):
        assert reduced[key]==actual[key]
    assert actual['passes'][0]['means']['ce']==2.5  # Unequal counts, plus dummy row.
    means=actual['aggregate']['means']
    assert reduced['objective']==pytest.approx(means['ce']+means['latent']+.1*means['kl'])
    assert reduced['objective']!=actual['objective']


@pytest.mark.parametrize('weight',[True,False,0.,-.1,.2,float('nan'),float('inf'),'0.1'])
def test_undeclared_objective_weights_are_rejected(weight):
    with pytest.raises(ValueError,match='weights 1 or 0.1'):
        summarize(weighted_rows(.1),weight)


@pytest.mark.parametrize('mutation',['branch_mismatch','mixed_rank_weights','latent_changed',
    'boolean_weight','pass_denominator','global_count','scaled_aggregate','scaled_passes',
    'nonfinite','wrong_pass_coefficients','jitter','wrong_pass_count'])
def test_fork_does_not_relax_existing_accounting_guards(mutation):
    rows=weighted_rows(.1)
    if mutation=='branch_mismatch':
        for row in rows:row['weights']['kl']=1.
    elif mutation=='mixed_rank_weights':rows[1]['weights']['kl']=1.
    elif mutation=='latent_changed':
        for row in rows:row['weights']['latent']=.1
    elif mutation=='boolean_weight':
        for row in rows:row['weights']['ce']=True
    elif mutation=='pass_denominator':rows[0]['passes'][1]['counts']['kl']+=1
    elif mutation=='global_count':
        rows[0]['counts']['ce']+=1
        for p in rows[0]['passes']:p['counts']['ce']+=1
    elif mutation=='scaled_aggregate':rows[0]['aggregate_sums']['kl']*=.1
    elif mutation=='scaled_passes':rows[0]['passes'][0]['sums']['kl']*=.1
    elif mutation=='nonfinite':rows[0]['passes'][2]['sums']['latent']=float('nan')
    elif mutation=='wrong_pass_coefficients':
        for row in rows:row['term_pass_coefficients']['kl']=[.025]*4
    elif mutation=='jitter':
        for row in rows:row['mode']['feedback_jitter']=.02
    else:rows[0]['passes'].pop()
    with pytest.raises(ValueError):summarize(rows,.1)


def fork_controller(old,model,report,kl_weight):
    model.config=replace(model.config,lambda_kl=kl_weight)
    model.predictor.config=replace(model.predictor.config,lambda_kl=kl_weight)
    return control.EvaluationController(old.plan,old.data_spec,old.recipe,
        coordinator=old.coordinator,device=old.device,batch_size=7,tracker=old.tracker,
        report=report,output_dir=old.output_dir,acceptance=True,kl_weight=kl_weight)


@pytest.mark.parametrize('arm',['NF','NFR'])
def test_same_model_controller_keeps_raw_losses_and_populated_adam_across_kl_change(data,tmp_path,arm):
    old,model,optimizer,report,tracker=live_setup(data,tmp_path,arm)
    # Imported historical predictor configuration can retain isolated metadata;
    # its forward does not own the current wrapper's document loss masks.
    model.predictor.config=replace(model.predictor.config,document_policy='isolated-v1')
    generator=torch.Generator().manual_seed(41)
    def boundary():
        return tree_digests({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'custom_rng':generator.get_state(),
            'gradients':{n:p.grad for n,p in model.named_parameters()}})
    before=boundary()
    pointers=[(id(p),p.data_ptr(),p.grad.data_ptr()) for p in model.parameters()]
    results=[]
    with control.ordered.OrderedCampaignData(data.corpus,data.output/'panels/train') as training:
        initial=asdict(training.cursor())
        for weight in (1.,.1):
            branch_report={}
            controller=fork_controller(old,model,branch_report,weight)
            kwargs=dict(model=model,runner=None,generators={'custom':generator},training_data=training,
                        boundary=boundary,persist=lambda:None)
            controller.run_if_due(0,**kwargs)
            assert not branch_report['evaluations']
            controller.run_if_due(1,**kwargs)
            controller.run_if_due(1,**kwargs)
            assert len(branch_report['evaluations'])==1
            entry=branch_report['evaluations'][0]
            assert entry['training_boundary_exact_by_rank']==[True]
            result=entry['panels']['dev-main']['result']
            assert result['weights']=={'ce':1.,'latent':1.,'kl':weight}
            assert controller.metrics_for(1)['dev/main/aggregate/kl']==result['aggregate']['means']['kl']
            results.append(result)
            assert asdict(training.cursor())==initial and boundary()==before
            assert model.predictor.config.document_policy=='isolated-v1'
    for key in ('passes','aggregate','reconstructed_aggregate_sums'):
        assert results[0][key]==results[1][key]
    assert results[1]['objective']==pytest.approx(results[0]['objective']-.9*results[0]['aggregate']['means']['kl'])
    assert pointers==[(id(p),p.data_ptr(),p.grad.data_ptr()) for p in model.parameters()]
    assert model.training and model.backbone.backbone.attention_precision=='mixed'


def test_wrong_live_branch_rejected_before_data_or_publication(data,tmp_path):
    old,model,optimizer,report,tracker=live_setup(data,tmp_path,'NF')
    controller=fork_controller(old,model,report,.1)
    model.config=replace(model.config,lambda_kl=1.)
    model.predictor.config=model.config
    with pytest.raises(ValueError,match='Live model objective'):
        controller.run_if_due(1,model=model,runner=None,generators={},training_data=None,
                              boundary=lambda:None,persist=lambda:None)
    assert not report['evaluations'] and not tracker.logs


def test_evaluation_failure_restores_live_runtime_and_never_publishes(data,tmp_path,monkeypatch):
    old,model,optimizer,report,tracker=live_setup(data,tmp_path,'NF')
    controller=fork_controller(old,model,report,.1)
    before=tree_digests(model.state_dict());rng=torch.get_rng_state().clone()
    def fail(*args):
        torch.rand(7)
        raise ValueError('Injected KL evaluation failure')
    monkeypatch.setattr(control,'per_pass_sums',fail)
    with control.ordered.OrderedCampaignData(data.corpus,data.output/'panels/train') as training:
        cursor=training.cursor()
        with pytest.raises(ValueError,match='Injected'):
            controller.run_if_due(1,model=model,runner=None,generators={},training_data=training,
                boundary=lambda:tree_digests(optimizer.state_dict()),persist=lambda:None)
        assert training.cursor()==cursor
    assert tree_digests(model.state_dict())==before and torch.equal(torch.get_rng_state(),rng)
    assert model.training and model.backbone.backbone.attention_precision=='mixed'
    assert report['evaluations'][0]['status']=='failed'
    assert not tracker.logs and not controller.published
    assert not (tmp_path/'evaluation-update-000001.json').exists()
