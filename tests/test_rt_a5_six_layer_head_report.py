"""Bounded CPU evidence/figure checks for the existing-head reassignment report."""
import copy
import hashlib
from pathlib import Path

import pytest

from scripts import rt_a5_six_layer_head_report as report


def terminal_fixture(start=0, endpoint=10000, end=None):
    end = endpoint if end is None else end
    protocol={'start_update':start,'endpoint':endpoint}
    value={'schema':report.TRAIN_SCHEMA,'start_update':start,'endpoint':endpoint,'completed_updates':end,
        'status':'complete','wandb':{'status':'synced'},'requested_endpoint_reached':True,
        'confirmation_evaluated':False,'latent_rollout_evaluated':False,'embedding_head_enabled':True}
    return value,protocol


def test_complete_requires_actual_pilot_or_continuation_and_head_route():
    for start,end in ((0,10000),(10000,50000)):
        value,protocol=terminal_fixture(start,end)
        assert report.terminal(value,protocol)==end
        value['completed_updates']-=1
        with pytest.raises(ValueError,match='actual requested endpoint'):
            report.terminal(value,protocol)
    for key,value in (('embedding_head_enabled',False),('injection_coefficient',.01),('schedule',{})):
        data,protocol=terminal_fixture();data[key]=value
        with pytest.raises(ValueError,match='head route differs'):
            report.terminal(data,protocol)


def test_graceful_stopped_continuation_requires_retained_sentinel(tmp_path):
    value,protocol=terminal_fixture(10000,50000,15000)
    stop=tmp_path/'STOP';stop.write_text('User stop\n')
    protocol['resolved_args']={'stop_file':str(stop)}
    value.update(status='stopped',requested_endpoint_reached=False,stop_request={
        'reason':'user_stop_file','observed_after_update':15000,'path':str(stop),
        'sha256':hashlib.sha256(stop.read_bytes()).hexdigest()})
    assert report.terminal(value,protocol)==15000
    stop.write_text('Changed\n')
    with pytest.raises(ValueError,match='sentinel changed'):
        report.terminal(value,protocol)


def test_combined_history_preserves_global_order_and_has_no_bypass_scalar():
    rows=[{'update':step,'order_chain':str(step),'examples_seen':step*1024,'embedding_head_enabled':True,
        'loss':.3,'state_loss':.2,'latent_loss':.1,'weighted_latent_loss':.1,'grad_norm':1.,'seconds':.1}
        for step in range(1,10003)]
    original=copy.deepcopy(rows)
    report.validate_history(rows,10002,original)
    for key,value in (('update',1),('order_chain','wrong'),('injection_coefficient',.01),('latent_loss',float('nan'))):
        changed=copy.deepcopy(rows);changed[10000][key]=value
        with pytest.raises(ValueError):
            report.validate_history(changed,10002,original)


@pytest.mark.parametrize('end', [10000,15000])
def test_full_boundary_render_identical_data_and_no_coefficient_claims(tmp_path,monkeypatch,end):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    def curve(step,head=False):
        return [{'update':step,'role':'ood_dev','length':t,'E':(.5 if head else 1)/t,
                 'A':.7 if head else .8,'M':.85 if head else .9} for t in range(1,37)]
    baseline={str(step):{'ood_dev':curve(step)} for step in (1000,5000,10000)}
    current={str(step):{'ood_dev':curve(step,head=True)} for step in (1000,5000,10000,end)}
    bins=[{'update':end,'state_ce':.2,'latent_loss':.1}]
    summary={'variant':'embedding_head','through_update':end,'matched_update':10000,'curves':current,
        'checkpoint_updates':[0,*sorted(map(int,current))],'training_curve':bins,
        'training_status':'complete' if end==10000 else 'stopped','requested_endpoint':10000 if end==10000 else 50000,
        'qualification':'One seed and reused development pools.','training_wandb':{'run_url':'https://example.test/run'},
        'baseline':{'curves':baseline,'checkpoint_updates':[0,1000,5000,10000],'training_curve':bins,
                    'comparisons':[{'update':10000,'baseline':{'E':.8,'M':.9},'head':{'E':.7,'M':.85}}]}}
    captured={};save=Figure.savefig
    def inspect_and_save(fig,path,*args,**kwargs):
        if Path(path).suffix=='.png':
            captured[Path(path).stem]={'lines':[
                [(line.get_label(),list(line.get_xdata()),list(line.get_ydata())) for line in axis.lines]
                for axis in fig.axes], 'xlims':[axis.get_xlim() for axis in fig.axes]}
        return save(fig,path,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',inspect_and_save)
    assert report.plot_rows(summary,10000) is current['10000']['ood_dev']
    names=report.plots(summary,tmp_path)
    assert names[:2]==['length-full','length-boundary']
    assert captured['length-full']['lines']==captured['length-boundary']['lines']
    assert captured['length-full']['xlims']==[(1.,36.)]*3
    assert captured['length-boundary']['xlims']==[(10.,18.)]*3
    assert captured['length-full']['lines'][0][0][2]==[r['E'] for r in baseline['10000']['ood_dev']]
    assert captured['length-full']['lines'][0][1][2]==[r['E'] for r in current['10000']['ood_dev']]
    assert ('length-unequal-terminals' in names)==(end != 10000)
    assert 'coefficient' not in names and 'injection_coefficient' not in report.COLUMNS
    assert all((tmp_path/f'{name}.{suffix}').stat().st_size>1000 for name in names for suffix in ('png','pdf'))
    text=report.markdown(summary)
    assert 'Both models have 19,998,208 parameters.' in text
    assert 'existing eighth head (index 7)' in text and 'There is no added parameter or scalar gate.' in text
    assert all(word not in text for word in ('λ','P_e','warmup','coefficient','262,144','20,260,352'))
    assert ('Actual endpoints differ' in text)==(end != 10000)
    current['10000']['ood_dev'][0]['length']=2
    with pytest.raises(ValueError,match='identical 36-prefix'):
        report.plot_rows(summary,10000)
