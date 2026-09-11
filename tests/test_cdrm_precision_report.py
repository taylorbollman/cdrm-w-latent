"""Structural checks keep failed execution and numerical flags distinct."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cdrm_precision_report as report


def comparison():
    return {'reference':'current_fp32','actual_ce_gradients':{
        'global_parameter_relative_l2':.04,'global_parameter_fp32_floor_l2':.001,
        'global_parameter_l2_pass':False,'pass':False,'per_tensor_l2_failures':['weight'],
        'rows':{'weight':{'finite':True,'fp32_floor_l2':.002,'fp32_floor_max':.0003,
            'prospective_l2_pass':False,'prospective_maximum_pass':True,
            'elementwise_pass':False,'worst_coordinate':{'large':'dump'}}}},
        'adam':{'global_delta_relative_l2':.05,'guardrail_pass':False,'rows':{}}}


def coarse_doc():
    return {'checkpoint':{'sha256':'weights','completed_updates':1000},
            'source_case':{'fixture':{'sha256':'data','shape':[64,256],'example_offset':0,'native_targets':6144}},
            'comparisons':{'current_bf16':comparison()}}


def attempt(kind,name,accepted):
    return {'kind':kind,'name':name,'artifact':{'sha256':name},
            'accepted_for_diagnostic_summary':accepted,'exclusion_reasons':[]}


def test_flags_and_fp32_floors_survive_compaction():
    original=comparison(); compact=report.compact_comparison(original)
    assert compact['actual_ce_gradients']['pass'] is False
    assert compact['actual_ce_gradients']['per_tensor_l2_failures']==['weight']
    row=compact['actual_ce_gradients']['rows']['weight']
    assert row['fp32_floor_l2']==.002 and row['fp32_floor_max']==.0003
    assert row['prospective_l2_pass'] is False and row['prospective_maximum_pass'] is True
    assert 'worst_coordinate' not in row


def test_failed_attempt_stays_visible_and_excluded_even_with_numbers(monkeypatch):
    values={'failed':(attempt('coarse','failed',False),coarse_doc()),
            'clean':(attempt('coarse','clean',True),coarse_doc())}
    monkeypatch.setattr(report,'load_attempt',lambda kind,path,*_:copy.deepcopy(values[path]))
    result=report.build_summary([('coarse','failed'),('coarse','clean')],report.Verifier())
    assert result['excluded_attempts']==1 and result['accepted_attempts']==1
    assert result['coarse'][0]['gradient_relative_l2']==.04
    assert result['coarse'][0]['accepted_for_diagnostic_summary'] is False
    assert result['coarse'][1]['gradient_screen_pass'] is False
    assert result['numerical_clearance'] is False and result['pending_stages']==report.PENDING


def test_attention_cannot_use_failed_probe_as_clean_evidence(monkeypatch):
    rows={key:{'interval_pass':True,'relative_l2':.001} for key in ('output','q_gradient','k_gradient','v_gradient')}
    local={'probe':{'report_sha256':'failed'},'observations':{'current_bf16/block0':{
        'native_against_analytic':rows,'local_fp32_against_analytic':rows,
        'fp32_bias_intervention_same_qkv_dy':rows,'native_local_interval_pass':True,'reference_floor_supported':True}}}
    values={'failed':(attempt('coarse','failed',False),coarse_doc()),
            'attention':(attempt('attention','attention',True),local)}
    monkeypatch.setattr(report,'load_attempt',lambda kind,path,*_:copy.deepcopy(values[path]))
    result=report.build_summary([('coarse','failed'),('attention','attention')],report.Verifier())
    assert result['accepted_attempts']==0 and result['excluded_attempts']==2
    assert all(not row['accepted_for_diagnostic_summary'] for row in result['attention'])
    assert result['attention'][0]['native']['interval_pass'] is True


def test_compiler_guard_rejects_fallback_and_missing_coverage():
    clean={'required':True,'fail_on_recompile_limit_hit':True,'counters':{'stats':{'unique_graphs':2}}}
    assert report.compiler_errors(clean)==[]
    assert report.compiler_errors({**clean,'fail_on_recompile_limit_hit':False})
    failed=copy.deepcopy(clean);failed['counters']['unimplemented']={'fallback':1}
    assert report.compiler_errors(failed)
    doc={'controls':[{'name':'current_fp32'},{'name':'current_bf16'}],
         'compiler_by_arm':{'unobserved/current_fp32':clean},
         'compiler':{'all_arms_pass':True,'failed_arms':[],'audited_arms':1,'counters':{'stats':{'unique_graphs':2}}}}
    assert 'Per-arm compiler coverage differs from requested controls' in report.coarse_errors(doc)


def test_hash_recheck_detects_late_mutation(tmp_path):
    path=tmp_path/'packet.pt';path.write_bytes(b'original')
    verifier=report.Verifier();verifier.check(path)
    path.write_bytes(b'altered')
    with pytest.raises(ValueError,match='changed during summary'):verifier.recheck()


def test_source_snapshot_mismatch_fails_closed(tmp_path):
    path=tmp_path/'report.json';path.write_text('{}')
    source=tmp_path/'source/scripts/probe.py';source.parent.mkdir(parents=True);source.write_text('original')
    doc={'source_sha256':{'scripts/probe.py':'incorrect'}}
    with pytest.raises(ValueError,match='SHA256 mismatch'):report.snapshot_audit(doc,path,report.Verifier())


def test_adam_local_pass_never_clears_original_policy_failure(monkeypatch):
    global_metric={'relative_l2':.04}
    check={'pass':True,'failed_tensors':[],'norm_screen_pass':True,
           **{key:{'global':global_metric,'rows':{}} for key in ('native_delta_vs_ideal_increment','native_delta_vs_fp32_write_realized','fp32_write_effect')}}
    case={'source_case':'old/slot0','checkpoint':{'sha256':'weights'},'fixture':{'sha256':'fixture'},
          'local_optimizer_checks':{'tiled_bf16':check},'local_optimizer_checks_pass':True,
          'original_machine_screens_pass':False,'original_adam_summary_reproduced':True,
          'gradient_comparisons':{'raw':{'global':global_metric}},
          'adam_policy_comparisons':{'native':{'global':global_metric,'original_trained_adam_screen_pass_if_applied':False}},
          'common_clipping_coefficient':.1}
    monkeypatch.setattr(report,'load_attempt',lambda *args:(attempt('adam','oracle',True),{'cases':[case]}))
    result=report.build_summary([('adam','oracle')],report.Verifier())
    assert result['adam'][0]['local_optimizer_checks_pass'] is True
    assert result['adam'][0]['original_machine_screens_pass'] is False
    assert result['adam'][0]['policy_comparisons']['native']['original_trained_adam_screen_pass_if_applied'] is False
    assert result['numerical_clearance'] is False


def test_final_sync_failure_still_retains_summary(tmp_path,monkeypatch):
    output=tmp_path/'summary'
    mock={'scope':'test','attempts':[],'accepted_attempts':0,'excluded_attempts':0,
          'coarse':[],'attention':[],'adam':[],'embedding':[],'confirmation':[],'numerical_clearance':False}
    monkeypatch.setattr(report,'build_summary',lambda *args:copy.deepcopy(mock))
    monkeypatch.setattr(report,'write_csvs',lambda *args:None)
    monkeypatch.setattr(report,'make_plot',lambda *args:None)
    monkeypatch.setitem(sys.modules,'cdrm_tiled_validate',SimpleNamespace(preserve_rng=nullcontext))
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Image=lambda path:path))
    class Tracker:
        def __init__(self,**kwargs):self.record={'status':'running'}
        def start(self,*args):pass
        def log(self,*args):pass
        def summary(self,*args):pass
        def finish(self,**kwargs):raise RuntimeError('injected final synchronization failure')
    monkeypatch.setattr(report,'OnlineTracker',Tracker)
    with pytest.raises(RuntimeError,match='injected'):
        report.main(['--coarse-report','unused','--output-dir',str(output),'--wandb-project','unit-test'])
    retained=json.loads((output/'report.json').read_text())
    assert retained['status']=='failed' and retained['numerical_clearance'] is False
    assert retained['final_sync_error_type']=='RuntimeError'


def test_graph_free_pure_bypass_does_not_clear_nonbypass_fallback():
    audit={'required':True,'fail_on_recompile_limit_hit':True,'counters':{}}
    assert report.compiler_errors(audit,require_graphs=False)==[]
    assert report.compiler_errors(audit,require_graphs=True)==['Compiler audit recorded no compiled graphs']
    audit['counters']['unimplemented']={'fallback':1}
    assert report.compiler_errors(audit,require_graphs=False)==['Compiler audit contains unsupported/fallback paths']


def test_extra_unobserved_controls_require_complete_parity_audits():
    one={'required':True,'fail_on_recompile_limit_hit':True,'require_graphs':True,
         'counters':{'stats':{'unique_graphs':1}}}
    names=['current_fp32','current_bf16','bf16_attention_fp32']
    doc={'controls':[{'name':name,'bypass':False} for name in names],
         'arguments':{'verify_observer_arm':['bf16_attention_fp32']},
         'compiler_by_arm':{f'{phase}/{name}':copy.deepcopy(one) for phase in ('unobserved','observed') for name in names},
         'compiler':{'all_arms_pass':True,'failed_arms':[],'audited_arms':6,'counters':{'stats':{'unique_graphs':6}}},
         'baseline_replay':{name:{'bitwise_equal':True} for name in names[:2]},
         'observation_invariance':{name:{'bitwise_equal':True} for name in names}}
    assert report.coarse_errors(doc)==[]
    doc['observation_invariance'].pop('bf16_attention_fp32')
    assert 'Missing exact observation_invariance/bf16_attention_fp32' in report.coarse_errors(doc)


def test_effectiveness_exclusion_only_removes_the_linked_arm(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'PROJECT',tmp_path)
    tensor=tmp_path/'tensors.pt';tensor.write_bytes(b'original packet')
    producer=tmp_path/'original.json';producer.write_text(json.dumps({'tensor_artifact':{'sha256':report.digest(tensor)}}))
    source=tmp_path/'source.py';source.write_text('# retained source')
    sidecar=tmp_path/'effectiveness.json';sidecar.write_text(json.dumps({
        'schema':'cdrm-alibi-control-effectiveness-v1','eligible_as_effective_fp32_alibi_observation':False,
        'other_narrow_controls_unchanged':True,'packet_differences':[],'step_differences':[],'scope':'ineffective bias control',
        'source_sha256':{str(source):report.digest(source)},
        'original_report':{'path':str(producer),'sha256':report.digest(producer)},
        'original_tensor':{'path':str(tensor),'sha256':report.digest(tensor)}}))
    rows=[{'arm':name,'input_report_sha256':report.digest(producer),'accepted_for_diagnostic_summary':True}
          for name in ('bf16_alibi_fp32','bf16_attention_fp32')]
    summary={'coarse':rows}
    result=report.apply_arm_exclusion(summary,sidecar,report.Verifier())
    assert result['other_arms_unchanged'] is True
    assert rows[0]['accepted_for_diagnostic_summary'] is False
    assert rows[1]['accepted_for_diagnostic_summary'] is True


def confirmation_doc():
    names=('naive_fp32','tiled_fp32','current_bf16','candidate')
    compare={name:{**comparison(),'machine_screens_pass':name!='current_bf16_vs_tiled_fp32'} for name in
             ('tiled_fp32_vs_naive_fp32','current_bf16_vs_tiled_fp32','candidate_vs_tiled_fp32')}
    return {'status':'diagnostics_complete','role':'fp32_trained_u1000','candidate':{'arm':'bf16_attention_fp32'},
        'checkpoint':{'sha256':'weights','completed_updates':1000},'fixture':{'sha256':'fixture'},
        'decision':{'sha256':'decision'},'fixtures':{'sha256':'fixtures'},
        'comparisons':compare,'per_arm_side_scaling_pass':{name:True for name in names},
        'side_scaling':{name:{scale:{'weight':{'bitwise_normalized_equal':True}} for scale in ('0.03125','32.0')} for name in names},
        'full_output_scaling':{name:{'diagnostic_pass':True,'fixed_forward_logits_bitwise_equal':True,
            'scaling':{scale:{'weight':{'bitwise_normalized_equal':True}} for scale in ('0.03125','32.0')}} for name in names},
        'candidate_machine_screens_pass':True,'candidate_additional_diagnostics_pass':True,
        'candidate_prospective_checks_pass':True,'machine_screens_pass':False,
        'compiler':{name:{'required':True,'require_graphs':name!='naive_fp32','validation_status':'passed',
                         'fail_on_recompile_limit_hit':True,'counters':{'stats':{'unique_graphs':int(name!='naive_fp32')}}}
                    for name in names}}


def test_fresh_candidate_success_preserves_original_failure(monkeypatch):
    doc=confirmation_doc()
    assert report.confirmation_execution_errors(doc)==[]
    monkeypatch.setattr(report,'load_attempt',lambda *args:(attempt('confirmation','fresh',True),copy.deepcopy(doc)))
    result=report.build_summary([('confirmation','fresh')],report.Verifier())
    row=result['confirmation'][0]
    assert row['candidate_prospective_checks_pass'] is True and row['machine_screens_pass'] is False
    assert row['comparisons']['current_bf16_vs_tiled_fp32']['machine_screens_pass'] is False
    assert row['comparisons']['candidate_vs_tiled_fp32']['actual_ce_gradients']['rows']['weight']['fp32_floor_l2']==.002
    assert result['numerical_clearance'] is False


def test_fresh_flags_require_scaling_and_every_compiled_arm():
    doc=confirmation_doc();doc['full_output_scaling']['candidate']['diagnostic_pass']=False
    assert 'Combined candidate flags do not match saved original/scaling flags' in report.confirmation_execution_errors(doc)
    doc=confirmation_doc();doc['compiler']['candidate']['required']=False
    assert any('Missing required compiled-path audit' in error for error in report.confirmation_execution_errors(doc))
    doc=confirmation_doc();doc['compiler']['current_bf16']['counters']['graph_break']={'fallback':1}
    assert any('unsupported/fallback' in error for error in report.confirmation_execution_errors(doc))
    doc=confirmation_doc();doc['machine_screens_pass']=True
    assert any('Combined original machine flag' in error for error in report.confirmation_execution_errors(doc))


def test_fresh_scaling_summary_must_match_saved_tensor_leaves():
    doc=confirmation_doc();doc['side_scaling']['candidate']['0.03125']['weight']['bitwise_normalized_equal']=False
    assert any('Side-scaling summary differs' in error for error in report.confirmation_execution_errors(doc))
    doc=confirmation_doc();doc['full_output_scaling']['candidate']['fixed_forward_logits_bitwise_equal']=False
    assert any('Output-scaling summary differs' in error for error in report.confirmation_execution_errors(doc))


def test_completed_confirmation_requires_unique_roles_and_same_anchors():
    rows=[{'role':name,'accepted_for_diagnostic_summary':True,'candidate_prospective_checks_pass':False,
           'decision':{'sha256':'decision'},'fixtures':{'sha256':'fixtures'}}
          for name in ('fp32_trained_u1000','bf16_trained_u1000','distinct_initialization')]
    summary={'confirmation':copy.deepcopy(rows)}
    report.apply_completed_stages(summary,report.PENDING)
    assert summary['pending_stages']==[]  # Completed observation need not be a numerical pass.
    for invalid in (rows[:2],rows+[rows[0]]):
        with pytest.raises(ValueError,match='exactly three'):report.apply_completed_stages({'confirmation':invalid},['fresh_confirmation'])
    summary={'confirmation':copy.deepcopy(rows)};summary['confirmation'][1]['fixtures']['sha256']='different'
    with pytest.raises(ValueError,match='anchors'):report.apply_completed_stages(summary,['fresh_confirmation'])
