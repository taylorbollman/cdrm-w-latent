import copy
import json
from pathlib import Path

import pytest

from scripts import rt_nextlat_a5_fuzzy_depth_extend_report as reporter
from test_rt_nextlat_a5_fuzzy_depth_resume_report import build_stage, write_json


@pytest.fixture
def lineage(tmp_path):
    a,b,c = [tmp_path/name for name in ('original','recovered','extended')]
    prior = build_stage(a,start=0,endpoint=3)
    with (a/'history.jsonl').open('ab') as stream:
        stream.write(b'{"update":4,"seconds":999}\n\0\0')
    middle = build_stage(b,start=3,endpoint=5,parent=prior['checkpoints'][-1],legacy_boundary=True)
    final = build_stage(c,start=5,endpoint=7,parent=middle['checkpoints'][-1],legacy_boundary=True)
    final['requested_endpoint'] = 7500
    write_json(c/'report.json',final)
    return a,b,c,final


def test_three_stage_lineage_counts_each_update_once(lineage):
    a,b,c,final=lineage
    old_hashes={str(path):reporter.sha(path/'history.jsonl') for path in (a,b)}
    run=reporter.load_run(c)
    assert [row['update'] for row in run['history']]==list(range(1,8))
    assert sum(row['seconds'] for row in run['history'])==7
    assert [stage['start_update'] for stage in run['lineage']]==[0,3,5]
    assert all(stage['boundary_state_audit']['entire_packet_exact'] for stage in run['lineage'][1:])
    assert run['lineage'][0]['history']['discarded_valid_update_range']==[4,4]
    assert len([row for row in run['evaluations'] if row['update']==5])==3
    assert old_hashes=={str(path):reporter.sha(path/'history.jsonl') for path in (a,b)}


@pytest.mark.parametrize('damage',['wrong_endpoint','restarted_contract','terminal_tail'])
def test_invalid_extension_rejected(lineage,damage):
    _,_,path,final=lineage
    if damage=='wrong_endpoint':final['requested_endpoint']=8000
    if damage=='restarted_contract':final['contract']['learning_rate_schedule']['start_lr']=.0003
    if damage=='terminal_tail':
        with (path/'history.jsonl').open('ab') as stream:stream.write(b'\0')
    write_json(path/'report.json',final)
    with pytest.raises(ValueError):reporter.load_run(path)


def test_stop_at_extension_boundary_retains_full_metrics(lineage):
    _,_,path,final=lineage
    final['completed_updates']=5
    final['checkpoints']=[record for record in final['checkpoints'] if record['completed_updates']==5]
    final['evaluations']=[row for row in final['evaluations'] if row['update']==5]
    write_json(path/'report.json',final)
    (path/'history.jsonl').write_text('')
    run=reporter.load_run(path)
    assert len(run['history'])==5 and run['endpoint']==5
    assert run['lineage'][-1]['endpoint_boundary_reevaluation_used']


def test_budget_summary_never_extends_reference(monkeypatch):
    # This checks reporting semantics independently of checkpoint parsing tests:
    # new training may reach 7500, but the saved reference remains exactly 5000.
    base={'history':[{'update':step,'seconds':1} for step in range(1,5001)]}
    parent={'endpoint':5000}
    candidate={'endpoint':7500,'report':{'requested_endpoint':7500,'status':'complete','contract':{'learning_rate_schedule':{}}},
               'history':[{'update':step,'seconds':2} for step in range(1,7501)],'evaluations':[], 'lineage':[]}
    matched={'arms':{'two_layers':{'endpoint':5000},'three_layers':{'endpoint':5000}},
             'matched_observations':[{'update':5000}], 'matched_training_seconds':{'two_layers':5000,'three_layers':10000},
             'common_full_updates':[1000,2500,5000], 'parameter_difference':197120, 'microbatch':{'two_layers':2560,'three_layers':2560}}
    monkeypatch.setattr(reporter,'matched_ancestor',lambda value:parent)
    monkeypatch.setattr(reporter.depth,'compare',lambda left,right:copy.deepcopy(matched))
    monkeypatch.setattr(reporter.depth,'check_histories',lambda left,right:None)
    monkeypatch.setattr(reporter.paired,'_packet',lambda run,update:{})
    monkeypatch.setattr(reporter.lr,'check_saved_lr',lambda packet,step,schedule:None)
    monkeypatch.setattr(reporter.paired,'arm_summary',lambda label,run:{'endpoint':run['endpoint']})
    monkeypatch.setattr(reporter.lr,'training_bins',lambda rows,constant:[])
    summary=reporter.compare(base,candidate)
    assert summary['matched_through_update']==5000
    assert summary['arms']['two_layers']['endpoint']==5000
    assert summary['arms']['three_layers']['endpoint']==7500
    assert summary['matched_training_seconds']=={'two_layers':5000,'three_layers':10000}
    assert summary['continuation']['additional_training_seconds']==5000
    assert summary['continuation']['cumulative_training_seconds']==15000
    assert summary['continuation']['paired_reference_available_after_5000'] is False
    assert summary['common_full_updates']==[1000,2500,5000]
    assert max(row['update'] for row in summary['matched_observations'])==5000
