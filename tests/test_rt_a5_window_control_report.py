"""CE-only scope, exact backbone pairing, fixed budgets and unclipped shared plots."""
import copy
import json

import pytest
import torch

from scripts import rt_a5_window_control_report as report


def history_fixture():
    rows = [{"update":i,"examples_seen":1024*i,"order_chain":f"{i:064x}","seconds":.01,
             "loss":.8,"state_loss":.8,"token_accuracy":.5,"whole_word_exact":.2,"grad_norm":.7}
            for i in range(1,1001)]
    packet = {"order_chain":rows[-1]["order_chain"],"train_seconds":sum(row["seconds"] for row in rows),"elapsed_seconds":20.}
    return rows,packet


def test_pure_ce_training_curve_has_no_invented_latent_loss():
    history,packet = history_fixture()
    report.validate_ce_history(history,packet,1000)
    curve = report.training_curve(history,False)
    assert len(curve)==10 and curve[-1]['update']==1000
    assert all(row['state_ce']==pytest.approx(.8) for row in curve)
    assert all('latent_loss' not in row and 'weighted_latent_loss' not in row for row in curve)


@pytest.mark.parametrize('mutation',['latent','state_alias','nonfinite','order','missing'])
def test_ce_history_rejects_changed_objective_nonfinite_or_missing_updates(mutation):
    rows,packet = history_fixture()
    if mutation=='latent': rows[4]['latent_loss']=0.
    elif mutation=='state_alias': rows[4]['state_loss']=.7
    elif mutation=='nonfinite': rows[4]['grad_norm']=float('nan')
    elif mutation=='order': rows[4]['update']=5_000
    else: rows.pop()
    with pytest.raises(ValueError): report.validate_ce_history(rows,packet,1000)


def contract_fixture():
    common={'model_config':{'n_layers':2,'d_model':512,'alibi':True},'evaluation':'same evaluator',
            'experiment_config':{'attention':{'window_layer':0}},'precision':'fp32','batch_size':1024}
    control={**copy.deepcopy(common),'optimizer':'AdamW','nextlat_enabled':False,'predictor_registered':False,
        'objective':{'state_loss':'same-position CE mean over B*T','latent_weight':0.,'kl_weight':0.,'predicted_state_ce_weight':0.}}
    nextlat={**copy.deepcopy(common),'optimizer':'AdamW-all-hybrid',
        'objective':{'horizon':1,'state_loss':'same-position CE mean over B*T',
            'latent_loss':'SmoothL1 beta=1 mean over B*(T-1)*D','latent_weight':1.,'kl_weight':0.,
            'predicted_state_ce_weight':0.,'source_and_embedding_attached':True,'target_detached':True}}
    return control,nextlat


def test_control_keeps_shared_architecture_runtime_and_backbone_optimizer():
    control,nextlat=contract_fixture()
    report.compare_shared_contract(control,nextlat)
    before=copy.deepcopy(control)
    assert report.shared_backbone_contract(control)==report.shared_backbone_contract(nextlat)
    assert control==before
    for path,value in [(('model_config','alibi'),False),(('objective','latent_weight'),.1),
                       (('experiment_config','attention'),{'window_layer':1})]:
        bad=copy.deepcopy(control);bad[path[0]][path[1]]=value
        with pytest.raises(ValueError): report.compare_shared_contract(bad,nextlat)
    control['optimizer']='different'
    with pytest.raises(ValueError,match='optimizer'): report.compare_shared_contract(control,nextlat)


def initial_fixture():
    weights={'transformer.matrix':torch.ones(2,2),'transformer.norm':torch.ones(2)}
    options={'lr':1e-4,'betas':(.9,.95),'eps':1e-8,'weight_decay':.01}
    control={'completed_updates':0,'model':weights,'optimizer':{'state':{},'param_groups':[
        {**options,'params':[0]},{**options,'params':[1],'weight_decay':0.}]},
        'optimizer_parameter_names':[['transformer.matrix'],['transformer.norm']]}
    nextlat={'completed_updates':0,'model':{**{'backbone.'+k:v.clone() for k,v in weights.items()},'predictor.matrix':torch.ones(3,3)},
        'optimizer':{'state':{},'param_groups':[{**options,'params':[0,2]},{**options,'params':[1],'weight_decay':0.}]},
        'optimizer_parameter_names':[['backbone.transformer.matrix','predictor.matrix'],['backbone.transformer.norm']]}
    return control,nextlat


def test_saved_backbone_mapping_is_exact_with_optimizer_hyperparameters():
    control,nextlat=initial_fixture()
    proof=report.compare_initial_packets(control,nextlat)
    assert proof['exact_backbone_tensor_identity'] and proof['control_predictor_registered'] is False
    assert (proof['backbone_parameters'],proof['backbone_parameter_tensors'])==(6,2)
    assert proof['nextlat_predictor_parameters']==9


@pytest.mark.parametrize('mutation',['tensor','predictor','precision','group','duplicate_id','trained'])
def test_saved_mapping_rejects_tensor_predictor_optimizer_or_initial_state_changes(mutation):
    control,nextlat=initial_fixture()
    if mutation=='tensor': control['model']['transformer.matrix'][0,0]=2.
    elif mutation=='predictor': control['model']['predictor.matrix']=torch.ones(3,3)
    elif mutation=='precision': nextlat['model']['backbone.transformer.matrix']=nextlat['model']['backbone.transformer.matrix'].half()
    elif mutation=='group': control['optimizer']['param_groups'][0]['lr']=.01
    elif mutation=='duplicate_id': control['optimizer']['param_groups'][1]['params']=[0]
    else: control['completed_updates']=10000
    with pytest.raises(ValueError): report.compare_initial_packets(control,nextlat)


@pytest.mark.parametrize('field,value',[('status','failed'),('start_update',10000),('parent_checkpoint',{'sha256':'resume'}),
    ('completed_updates',79999),('completed_updates',80001),('endpoint',100000),('wandb',{'status':'offline'})])
def test_final_report_never_accepts_unfinished_resumed_failed_or_unsynced_control(tmp_path,field,value):
    packet={'schema':report.TRAIN_SCHEMA,'status':'complete','start_update':0,'parent_checkpoint':None,
            'endpoint':80000,'completed_updates':80000,'wandb':{'status':'synced'}}
    packet[field]=value
    (tmp_path/'report.json').write_text(json.dumps(packet))
    with pytest.raises(ValueError): report.read_control({'training_directory':str(tmp_path)})


def metric(step,role,perfect=False):
    length=12 if role=='dev' else 36
    exact=([1.]*36 if perfect else [.5]*12+[.25]*2+[0.]*22)[:length]
    accuracy=1. if perfect else .5
    return {'update':step,'role':role,'rows':102400,'length':length,'tokens':102400*length,'route':'backbone_only',
        'isolated_state_accuracy':[accuracy]*length,'cumulative_prefix_exactness':exact,'per_position_ce':[1.]*length,
        'ce':1.,'token_accuracy':accuracy,'whole_word_exact_match':exact[-1],'final_state_accuracy':accuracy}


def summary_fixture():
    summary={'arms':{}}
    history,_=history_fixture()
    for arm in report.ARMS:
        metrics={str(step):{role:metric(step,role,arm==report.NEXTLAT and step>=30000) for role in report.ROLES} for step in report.STEPS}
        summary['arms'][arm]={'parameter_count':6357504 if arm==report.CONTROL else 7407104,'metrics':metrics,
            'curves':{step:{role:report.metric_rows(m,arm) for role,m in roles.items()} for step,roles in metrics.items()},
            'training_curve':report.training_curve(history,False)}
    summary['checkpoint_summary']=report.checkpoint_summary(summary)
    return summary


def test_full_boundary_and_e36_trajectory_share_rows_and_tick_labels_fit(monkeypatch,tmp_path):
    import matplotlib.figure
    original=matplotlib.figure.Figure.savefig
    checked=[]
    def savefig(fig,*args,**kwargs):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for axis in fig.axes:
            for label in axis.get_yticklabels():
                if label.get_visible():
                    bounds=label.get_window_extent(renderer)
                    assert bounds.x0>=0 and bounds.x1<=fig.bbox.width
        assert kwargs['bbox_inches']=='tight' and kwargs['pad_inches']>=.1
        checked.append(args[0])
        return original(fig,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',savefig)
    summary=summary_fixture();rows=report.metric_table(summary)
    assert len(rows)==1056 and len(summary['checkpoint_summary'])==22
    for arm in report.ARMS:
        plotted=report.plot_rows(summary,arm,80000)
        selected=[r for r in rows if (r['arm'],r['update'],r['role'])==(arm,80000,'ood_dev')]
        assert plotted==selected and plotted[9:18]==selected[9:18]
    text=report.markdown_report(summary)
    for phrase in ('CE-only E(36)','6,357,504','7,407,104','global norm','same development words','unevaluated'):
        assert phrase in text
    figures=report.plot_results(summary,tmp_path)
    assert 'whole-word-vs-updates' in figures and len(figures)==5 and len(checked)==10
    for files in figures.values():
        assert (tmp_path/files['png']).read_bytes().startswith(b'\x89PNG')
        assert (tmp_path/files['pdf']).read_bytes().startswith(b'%PDF')


def test_metric_exports_reject_wrong_arms_or_unavailable_checkpoints():
    summary=summary_fixture()
    with pytest.raises(ValueError): report.plot_rows(summary,report.CONTROL,100000)
    summary['arms'][report.CONTROL]['curves']['80000']['ood_dev'][0]['arm']='rt_full'
    with pytest.raises(ValueError): report.metric_table(summary)
    with pytest.raises(ValueError): report.plot_rows(summary,report.CONTROL,80000)


def test_ce_evaluation_requires_shared_rows_and_no_predictor_diagnostics():
    packet={'evaluations':[metric(step,role) for step in (500,1000) for role in report.ROLES]}
    for item in packet['evaluations']:
        if item['update']==500:
            item['rows']=4096;item['tokens']=item['length']*4096
    report.validate_ce_evaluations(packet,1000)
    packet['one_step_diagnostics']=[{'rows':1024}]
    with pytest.raises(ValueError,match='predictor diagnostics'): report.validate_ce_evaluations(packet,1000)
    packet.pop('one_step_diagnostics');packet['evaluations'][-1]['rows']=4096
    with pytest.raises(ValueError): report.validate_ce_evaluations(packet,1000)


def test_preflight_can_include_validation_harness_without_changing_training_source_identity():
    sources={'train.py':'frozen'}
    preflight={'status':'passed','wandb':{'status':'synced'},'source_files':{**sources,'validator.py':'separate'}}
    report.validate_preflight(preflight,sources)
    preflight['source_files']['train.py']='changed'
    with pytest.raises(ValueError,match='frozen training sources'): report.validate_preflight(preflight,sources)
