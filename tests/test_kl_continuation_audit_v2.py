"""Regression: the actual accepted tiny payload has no native LR lookup table."""
from copy import deepcopy
import subprocess
import sys

import pytest

from scripts import olmo_kl_continuation_audit_v2 as audit
from test_kl_continuation_audit import pair as v1_pair, PARENT_PIN
from test_pilot_async_audit import add_transport
from test_campaign_eval_audit import reidentify, resumed


def pair():
    values=v1_pair();parent=values[0]
    schedule={'schema':'campaign-token-schedule-v1','planned_updates':3,'planned_tokens':240,
        'warmup_tokens':240,'start_fraction':.1,'plan_sha256':'c'*64}
    parent['configuration']['execution_identity']['payload'].pop('schedule')
    parent['configuration']['schedule']=deepcopy(schedule)
    reidentify(parent);add_transport(parent,'async')
    manifest=parent['local_checkpoints'][1]['receipt']['manifest_sha256']
    for report in values[1:]:
        config=report['configuration'];payload=config['execution_identity']['payload']
        payload.pop('schedule');config['schedule']=deepcopy(schedule)
        report['original_configuration']=deepcopy(parent['configuration'])
        report['branch']['parent_identity_sha256']=parent['configuration']['execution_identity']['sha256']
        report['branch']['parent_manifest_sha256']=manifest
        config['recipe']['objective_transition']['parent_manifest_sha256']=manifest
        report['parent_manifest_sha256']=manifest;report['resume']['manifest_sha256']=manifest
        report['objective_transition']['parent_manifest_sha256']=manifest
        reidentify(report);add_transport(report,'async');report['schema']=audit.original.REPORT_SCHEMA
    return values


def run(values):
    return audit.audit_pair(*values,expected_stop=3,parent_report_sha256=PARENT_PIN)


def test_actual_tiny_schedule_shape_passes_without_rewriting_frozen_v1():
    values=pair();before=deepcopy(values)
    old=audit.original.audit_pair(*values,expected_stop=3,parent_report_sha256=PARENT_PIN)
    assert not old['passed'] and 'schedule' in old['failures'][0]
    result=run(values)
    assert result['passed'],result['failures']
    assert result['schema']==audit.SCHEMA and set(result['audit_sources'])==set(audit.AUDIT_SOURCES)
    assert values==before


@pytest.mark.parametrize('mutation',['invented_native_schedule','changed_schedule','planned_tokens',
    'planned_updates','reduced_lr_used','reduced_lr_next','both_lr_reset','control_gradient','input_noise',
    'raw_first_forward','fresh_Adam','eval_weights','eval_mean','parent_pin','missing_dependency'])
def test_regression_does_not_weaken_clocks_state_data_or_objective_guards(mutation):
    values=pair();parent,control,reduced=values
    if mutation=='invented_native_schedule':reduced['configuration']['execution_identity']['payload']['schedule']={}
    elif mutation=='changed_schedule':reduced['configuration']['schedule']['warmup_tokens']=80
    elif mutation=='planned_tokens':reduced['configuration']['schedule']['planned_tokens']=160
    elif mutation=='planned_updates':reduced['configuration']['schedule']['planned_updates']=2
    elif mutation in ('reduced_lr_used','reduced_lr_next'):
        for row in reduced['updates']['2']:row['metrics'][mutation.removeprefix('reduced_')]=[.002]
    elif mutation=='both_lr_reset':
        for report in (control,reduced):
            for row in report['updates']['2']:row['metrics']['lr_used']=[.002]
    elif mutation=='control_gradient':
        for row in control['updates']['3']:row['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='input_noise':reduced['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='raw_first_forward':
        for row in reduced['updates']['2']:
            row['metrics']['loss_sums']['ce']+=1;row['loss_means']['ce']=row['metrics']['loss_sums']['ce']/75
    elif mutation=='fresh_Adam':
        for row in reduced['origin_boundary_by_rank']:row['state']['optimizer']['state']={}
    elif mutation=='eval_weights':reduced['evaluations'][0]['panels']['dev-main']['result']['weights']['kl']=1.
    elif mutation=='eval_mean':reduced['evaluations'][0]['panels']['dev-main']['result']['passes'][0]['means']['ce']+=1
    elif mutation=='parent_pin':reduced['branch']['parent_manifest_sha256']='0'*64
    elif mutation=='missing_dependency':reduced['resume']['parent_files_required']=False
    reidentify(reduced)
    assert not run(values)['passed'],mutation


def test_actual_shape_same_branch_restart_has_exact_next_update():
    _,_,reference=pair()
    publication=deepcopy(next(row for row in reference['published_checkpoints'] if row['counters']['optimizer_updates']==2))
    actual=resumed(reference);actual['branch_resume']=True;actual.pop('objective_transition')
    actual['resume'].update(manifest_sha256=publication['manifest_sha256'],reference_report_required=True,
                            parent_files_required=True,parent_boundary_compared=False)
    actual['storage']['checkpoint_root']+='/resume';actual['storage']['evidence_dir']+='/resume'
    reidentify(actual);add_transport(actual,'async');actual['schema']=audit.original.REPORT_SCHEMA
    result=audit.audit_restart(reference,actual,source_publication=publication,expected_stop=3)
    assert result['passed'],result['failures']
    actual['updates']['3'][0]['input']['batches']['sha256']='0'*64
    assert not audit.audit_restart(reference,actual,source_publication=publication,expected_stop=3)['passed']


def test_import_remains_json_only():
    subprocess.run([sys.executable,'-c','import sys;import scripts.olmo_kl_continuation_audit_v2;assert "torch" not in sys.modules;assert "google.cloud.storage" not in sys.modules'],
        cwd=audit.ROOT,check=True,capture_output=True,text=True)
