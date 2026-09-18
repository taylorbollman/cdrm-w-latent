"""Bounded saved-report checks; no model, inference, checkpoint loading, or W&B calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from scripts import rt_a5_six_layer_input_extension_report as report


def extension_fixture(end=20000):
    parent={'strict_contract':{'n_layers':6},'initialization':{'shared':'same'},'source_files':{'file.py':'same'}}
    protocol={**deepcopy(parent),'schema':report.EXT_SCHEMA,'start_update':10000,'endpoint':20000,
        'parent_checkpoint':{'completed_updates':10000,'sha256':'parent'},'coefficient':.01}
    current={'schema':report.base.TRAIN_SCHEMA,'contract':deepcopy(parent['strict_contract']),
        'initialization':deepcopy(parent['initialization']),'source_files':deepcopy(parent['source_files']),
        'start_update':10000,'endpoint':20000,'parent_checkpoint':{'sha256':'parent'},
        'wandb':{'status':'synced'},'completed_updates':end,'status':'complete' if end==20000 else 'stopped',
        'injection_coefficient':.01,'coefficient_learned':False,'requested_endpoint_reached':end==20000,
        'confirmation_evaluated':False,'latent_rollout_evaluated':False}
    return protocol,current,parent


@pytest.mark.parametrize('end',[20000,16666])
def test_accept_terminal_scope(end):
    assert report.validate_extension(*extension_fixture(end),schema=report.EXT_SCHEMA)==end


@pytest.mark.parametrize('mutation', ['parent','source','failed','coefficient'])
def test_reject_invalid_continuation(mutation):
    p,r,parent=extension_fixture()
    if mutation=='parent':r['parent_checkpoint']['sha256']='different'
    if mutation=='source':p['source_files']['extra.py']='changed';r['source_files']=deepcopy(p['source_files'])
    if mutation=='failed':r['status']='failed'
    if mutation=='coefficient':r['injection_coefficient']=.02
    with pytest.raises(ValueError):report.validate_extension(p,r,parent,schema=report.EXT_SCHEMA)


def test_actual_closed_six_layer_baseline_contract():
    root=Path(__file__).resolve().parents[1]
    lineage=root/'.runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k'
    p=json.loads((lineage/'protocol.json').read_text())
    r=json.loads((lineage/'train-depth/report.json').read_text())
    parent=json.loads((root/p['parent_protocol']['path']).read_text())
    assert report.validate_extension(p,r,parent,schema='rt-a5-l1r-depth-extension-protocol-v1')==20000
    assert {(e['update'],e['role'],e['rows']) for e in r['evaluations'] if e['update'] in (15000,20000)} == {
        (step,role,102400) for step in (15000,20000) for role in ('dev','ood_dev')}


def histories(end=20000):
    return ([{'update':i} for i in range(1,10001)], [{'update':i} for i in range(10001,end+1)])


def test_stitch_preserves_parent_child_and_exact_boundary():
    parent,child=histories(16666);combined=report.stitch_history(parent,child,16666)
    assert len(combined)==16666 and combined[9999] is parent[-1] and combined[10000] is child[0]
    assert len(parent)==10000 and len(child)==6666


@pytest.mark.parametrize('mutation',['gap','overlap','incomplete_parent'])
def test_stitch_rejects_missing_or_overlapping_history(mutation):
    parent,child=histories()
    if mutation=='gap':child.pop(123)
    if mutation=='overlap':child[0]['update']=10000
    if mutation=='incomplete_parent':parent.pop()
    with pytest.raises(ValueError):report.stitch_history(parent,child,20000)


def test_common_full_checkpoints_include_15k_but_not_unmatched_endpoint():
    assert report.common_checkpoint_steps({'1000':{},'5000':{},'10000':{},'15000':{},'16666':{}},
        {'1000':{},'5000':{},'10000':{},'15000':{},'20000':{}})==[1000,5000,10000,15000]


def synthetic_summary(end):
    def rows(step,arm):
        return [{'update':step,'role':'ood_dev','length':t,
            'E':1-t/100+arm/100+step/1e7,'A':.8-t/200+arm/100+step/1e7,
            'M':.9-t/300+arm/100+step/1e7} for t in range(1,37)]
    current_steps=[0,1000,5000,10000,15000,end]
    baseline_steps=[0,1000,5000,10000,15000,20000]
    def view(steps,arm):
        return {'completed_updates':steps[-1],'checkpoint_updates':steps,
            'curves':{str(s):{'ood_dev':rows(s,arm)} for s in steps[1:]},
            'training_curve':[{'update':s,'state_ce':1/(s+1),'latent_loss':.1/(s+1)} for s in steps[1:]]}
    summary=view(current_steps,1);summary['baseline']=view(baseline_steps,0)
    summary['through_update']=end
    summary['matched_update']=max(report.common_checkpoint_steps(summary['curves'],summary['baseline']['curves']))
    return summary


@pytest.mark.parametrize('end,matched',[ (20000,20000),(16666,15000) ])
def test_plots_use_same36_rows_and_separate_unequal_endpoint(end,matched,monkeypatch,tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    saved={}
    def capture(figure,path,**kwargs):
        saved[Path(path).name]={'title':figure._suptitle.get_text() if figure._suptitle else '',
            'axes':[[{'x':list(line.get_xdata()),'y':list(line.get_ydata()),'label':line.get_label()}
                for line in axis.lines] for axis in figure.axes]}
    monkeypatch.setattr(Figure,'savefig',capture)
    summary=synthetic_summary(end);names=report.plots(summary,tmp_path)
    assert saved['length-full.png']['axes']==saved['length-boundary.png']['axes']
    assert f'{matched:,}' in saved['length-full.png']['title']
    for axis,key in enumerate(('E','A','M')):
        for arm_index,arm in enumerate(('baseline','injection')):
            expected=report.plot_rows(summary,matched,arm)
            actual=saved['length-full.png']['axes'][axis][arm_index]
            assert actual['x']==list(range(1,37)) and actual['y']==[r[key] for r in expected]
    if end==20000:
        assert len(names)==4 and 'length-unequal-terminals' not in names
    else:
        assert len(names)==5 and 'length-unequal-terminals' in names
        assert '20,000' in saved['length-unequal-terminals.png']['title']
        assert '16,666' in saved['length-unequal-terminals.png']['title']
        for axis,key in enumerate(('E','A','M')):
            assert saved['length-unequal-terminals.png']['axes'][axis][0]['y']==[
                r[key] for r in report.plot_rows(summary,20000,'baseline')]
            assert saved['length-unequal-terminals.png']['axes'][axis][1]['y']==[
                r[key] for r in report.plot_rows(summary,end,'injection')]
    assert set(saved)=={f'{name}.{suffix}' for name in names for suffix in ('png','pdf')}
