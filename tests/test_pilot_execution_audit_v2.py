"""Narrow post-runtime audit fixes, preserving every scientific field."""
from copy import deepcopy
import math
import subprocess
import sys

import pytest
from scripts import olmo_pilot_execution_audit_v2 as audit
from test_pilot_execution_audit import pair, stopped_at_two
from test_campaign_eval_audit import resumed
from test_campaign_ssd_audit import storage


def controls():
    _,reference=pair()
    for event in reference['evaluations']:
        for panel in event['panels'].values():
            for part in panel['by_rank']:
                part['preservation'].update(elapsed_seconds=.1,integrity_scope='complete',rope_scope='fixed',timing_scope='wall')
    storage(reference);actual=resumed(reference)
    publication=deepcopy(reference['published_checkpoints'][2]);actual['resume']['manifest_sha256']=publication['manifest_sha256']
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/resume'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/pilot-resume'
    evidence=storage(actual)
    return reference,actual,publication,evidence


def run(reference,actual,publication,evidence):
    return audit.compare(reference,actual,kind='resume',resume_publication=publication,storage_evidence=evidence)


def test_only_preservation_elapsed_is_ignored_in_cross_process_equality():
    reference,actual,publication,evidence=controls()
    actual['evaluations'][0]['panels']['dev-main']['by_rank'][0]['preservation']['elapsed_seconds']=123.456
    before=deepcopy((reference,actual,publication,evidence))
    result=run(reference,actual,publication,evidence)
    assert result['passed'],result['failures']
    assert (reference,actual,publication,evidence)==before
    assert not audit.original.compare(reference,actual,kind='resume',resume_publication=publication,storage_evidence=evidence)['passed']


@pytest.mark.parametrize('field',['restored','integrity_passed','precision','feedback_jitter','integrity_scope','rope_scope','timing_scope',
    'tensor_metadata_unchanged','module_ownership_unchanged','cache_generations_unchanged','gradient_identity_unchanged',
    'gradient_values_remained_zero','runtime_restored','modes_restored','rng_restored','autocast_cache_restored',
    'elapsed_missing','elapsed_negative','elapsed_nan','elapsed_infinity','elapsed_bool','new_field','removed_field'])
def test_every_meaningful_preservation_change_or_bad_elapsed_is_rejected(field):
    reference,actual,publication,evidence=controls()
    p=actual['evaluations'][0]['panels']['dev-main']['by_rank'][0]['preservation']
    if field in p['checks']:p['checks'][field]=False
    elif field in ('restored','integrity_passed'):p[field]=False
    elif field=='feedback_jitter':p[field]=.02
    elif field in ('precision','integrity_scope','rope_scope','timing_scope'):p[field]='different'
    elif field=='elapsed_missing':p.pop('elapsed_seconds')
    elif field=='elapsed_negative':p['elapsed_seconds']=-1.
    elif field=='elapsed_nan':p['elapsed_seconds']=math.nan
    elif field=='elapsed_infinity':p['elapsed_seconds']=math.inf
    elif field=='elapsed_bool':p['elapsed_seconds']=True
    elif field=='new_field':p['hidden']=True
    elif field=='removed_field':p.pop('rope_scope')
    assert not run(reference,actual,publication,evidence)['passed'],field


def test_engine_omitted_maps_are_valid_only_for_terminal_no_update_segment():
    reference,_,_,_=controls();stopped=stopped_at_two(reference);storage(stopped)
    publication=deepcopy(stopped['published_checkpoints'][-1]);actual=stopped_at_two(reference)
    actual['resume']={'completed_update':2,'reference_report_required':False,'manifest_sha256':publication['manifest_sha256']}
    actual['loop']['start_update']=2;actual['origin_boundary_by_rank']=deepcopy(actual['final_boundary_by_rank'])
    actual['origin_clocks']=deepcopy(actual['final_clocks'])
    actual.pop('updates');actual.pop('observations');actual['graph_prepared']=False;actual['runner_by_rank']=[None,None]
    actual.pop('preparation_boundary_exact')
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/terminal'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/pilot-terminal'
    for part in actual['evaluations'][0]['panels']['dev-main']['by_rank']:part['preservation']['elapsed_seconds']=.5
    evidence=storage(actual)
    result=run(stopped,actual,publication,evidence)
    assert result['passed'],result['failures']
    actual['final_boundary_by_rank'][0]['rng']['python']=[111]
    assert not run(stopped,actual,publication,evidence)['passed']


def test_nonempty_segment_cannot_omit_updates():
    reference,actual,publication,evidence=controls();actual.pop('updates')
    assert not run(reference,actual,publication,evidence)['passed']


def test_new_module_remains_json_only():
    subprocess.run([sys.executable,'-c','import sys;import scripts.olmo_pilot_execution_audit_v2;assert "torch" not in sys.modules;assert "google.cloud.storage" not in sys.modules'],
        cwd=audit.ROOT,check=True,capture_output=True,text=True)
