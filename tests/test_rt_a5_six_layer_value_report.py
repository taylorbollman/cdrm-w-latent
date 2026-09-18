"""Small saved-report endpoint, global schedule and plotting checks; no models."""
import copy
import hashlib

import pytest

from scripts import rt_a5_six_layer_value_report as report


def terminal_fixture(start=0, endpoint=10000, end=None, variant='constant'):
    end = endpoint if end is None else end
    protocol={'start_update':start,'endpoint':endpoint,'variant':variant}
    value={'schema':report.TRAIN_SCHEMA,'start_update':start,'endpoint':endpoint,'completed_updates':end,
        'status':'complete','wandb':{'status':'synced'},'requested_endpoint_reached':True,
        'confirmation_evaluated':False,'latent_rollout_evaluated':False,
        'injection_coefficient':report.coefficient(end,variant)}
    return value,protocol


def test_complete_requires_actual_pilot_or_continuation_endpoint():
    for start,end in ((0,10000),(10000,50000)):
        value,protocol=terminal_fixture(start,end)
        assert report.terminal(value,protocol)==end
        value['completed_updates']-=1
        with pytest.raises(ValueError,match='actual requested endpoint'):
            report.terminal(value,protocol)


def test_graceful_stopped_continuation_binds_stop_file_and_global_lambda(tmp_path):
    value,protocol=terminal_fixture(10000,50000,15000,'linear')
    stop=tmp_path/'STOP';stop.write_text('User stop\n')
    protocol['resolved_args']={'stop_file':str(stop)}
    value.update(status='stopped',requested_endpoint_reached=False,stop_request={
        'reason':'user_stop_file','observed_after_update':15000,'path':str(stop),
        'sha256':hashlib.sha256(stop.read_bytes()).hexdigest()})
    assert report.terminal(value,protocol)==15000
    assert value['injection_coefficient']==.0075
    value['injection_coefficient']=report.coefficient(5000,'linear')
    with pytest.raises(ValueError,match='coefficient differs'):
        report.terminal(value,protocol)


def test_combined_history_never_restarts_linear_schedule_after_10k():
    rows=[{'update':step,'order_chain':str(step),'examples_seen':step*1024,
        'injection_coefficient':report.coefficient(step,'linear'),'loss':.3,'state_loss':.2,
        'latent_loss':.1,'weighted_latent_loss':.1,'grad_norm':1.,'seconds':.1} for step in range(1,10003)]
    original=copy.deepcopy(rows)
    report.validate_history(rows,10002,original,'linear')
    rows[10000]['injection_coefficient']=report.coefficient(1,'linear')
    with pytest.raises(ValueError,match='Global history'):
        report.validate_history(rows,10002,original,'linear')


@pytest.mark.parametrize('variant,end', [('constant',10000),('linear',15000)])
def test_shared_prefix_rows_and_separate_unequal_terminal_plots(tmp_path,variant,end):
    def curve(step):
        return [{'update':step,'role':'ood_dev','length':t,'E':1/t,'A':.8,'M':.9} for t in range(1,37)]
    baseline={str(step):{'ood_dev':curve(step)} for step in (1000,5000,10000)}
    current={**baseline,str(end):{'ood_dev':curve(end)}}
    bins=[{'update':end,'state_ce':.2,'latent_loss':.1}]
    summary={'variant':variant,'through_update':end,'matched_update':10000,'curves':current,
        'checkpoint_updates':[0,*sorted(map(int,current))],'training_curve':bins,
        'baseline':{'curves':baseline,'checkpoint_updates':[0,1000,5000,10000],'training_curve':bins}}
    assert report.plot_rows(summary,10000) is current['10000']['ood_dev']
    names=report.plots(summary,tmp_path)
    assert names[:2]==['length-full','length-boundary']
    assert ('length-unequal-terminals' in names)==(end != 10000)
    assert ('coefficient' in names)==(variant=='linear')
    assert all((tmp_path/f'{name}.{suffix}').stat().st_size>1000 for name in names for suffix in ('png','pdf'))
    current['10000']['ood_dev'][0]['length']=2
    with pytest.raises(ValueError,match='identical 36-prefix'):
        report.plot_rows(summary,10000)
