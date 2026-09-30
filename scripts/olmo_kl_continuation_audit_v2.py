#!/usr/bin/env python3
"""Post-freeze audit correction for the accepted tiny identity's schedule shape.

The frozen v1/runtime remain untouched. Native identities contain an explicit
LR lookup table; tiny identities bind their schedule in configuration instead.
Require that declared token schedule for tiny, paired LR equality, and exact
KL1 control-versus-parent next-update parity. Reuse all other accepted checks.
This auditor is independently source-pinned, outside the 210 training sources.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_kl_continuation_audit as original
from scripts.olmo_kl_continuation_audit import (Audit, legacy, publication_metadata,
    training_check, _branch_check, _parent_check, _same_except_kl, evaluation_check,
    _raw_evaluation, expected_counts, transport_check, training_parity,
    file_sha, bounded_json)

SCHEMA='olmo-kl-continuation-audit-v2'
AUDIT_SOURCES=('scripts/olmo_kl_continuation_audit_v2.py',
    'scripts/olmo_kl_continuation_audit.py','scripts/olmo_pilot_async_audit.py',
    'scripts/olmo_pilot_execution_audit.py','scripts/olmo_pilot_execution_audit_v2.py',
    'scripts/olmo_campaign_execution_audit.py')


def validate_report(audit,report,label,*,expected_stop,source_root=None,check_transport=True):
    payload=training_check(audit,report,label,source_root)
    _branch_check(audit,report,label,expected_stop=expected_stop)
    evaluation_check(audit,report,label+'/evaluation')
    objective=report['evaluation_policy']['objective']
    audit.equal(label+'/evaluation_weights',objective['weights'],payload['model_contract']['weights'])
    audit.equal(label+'/evaluation_objective_schema',objective['schema'],'olmo-kl-continuation-evaluation-v1')
    prefix=[expected_counts(payload,n)['input_tokens'] for n in range(len(payload['plan']['updates'])+1)]
    if report['scale']=='native':
        schedule=payload['schedule']
        audit.equal(label+'/schedule_count',len(schedule['lr_at_completed_boundaries']),len(prefix))
        audit.equal(label+'/token_prefix',schedule['valid_token_prefix'],prefix)
        for step,rows in report.get('updates',{}).items():
            for key,index in (('lr_used',int(step)-1),('lr_next',int(step))):
                actual=rows[0]['metrics'][key]
                audit.equal(label+f'/update{step}/'+key,actual,[schedule['lr_at_completed_boundaries'][index]]*len(actual))
    else:
        schedule=report['configuration']['schedule']
        audit.equal(label+'/tiny_schedule_schema',schedule['schema'],'campaign-token-schedule-v1')
        audit.equal(label+'/tiny_planned_tokens',schedule['planned_tokens'],prefix[-1])
        audit.equal(label+'/tiny_planned_updates',schedule['planned_updates'],len(prefix)-1)
        audit.equal(label+'/tiny_exact_inherited_schedule',schedule,report['original_configuration']['schedule'])
        # Tiny has no LR lookup in payload; it does have exact saved schedule
        # state and a KL1 accepted trajectory. Pair/restart parity below checks
        # every actual LR value as well as token/optimizer clocks.
        audit.require(label+'/tiny_no_invented_payload_schedule','schedule' not in payload)
    if check_transport:transport_check(audit,report,label+'/transport')
    return payload


def _result(audit,error=None,**extra):
    result=original._result(audit,error,**extra)
    result['schema']=SCHEMA
    result['audit_sources']={name:file_sha(ROOT/name) for name in AUDIT_SOURCES}
    result['audit_correction']='Tiny has configuration.schedule, not native payload.schedule; frozen v1/runtime unchanged.'
    return result


def audit_pair(parent_report, control_report, reduced_report, *, expected_stop,
               source_roots=None, check_transport=True, parent_report_sha256=None):
    """Audit two new fork reports against the exact retained common parent."""
    audit = Audit(); source_roots = source_roots or {}
    try:
        reports = {'control':control_report, 'reduced':reduced_report}
        for label, report in reports.items():
            validate_report(audit, report, label, expected_stop=expected_stop,
                            source_root=source_roots.get(label), check_transport=check_transport)
            audit.require(label+'/is_first_fork_segment', not report.get('branch_resume',False))
            _parent_check(audit, parent_report, report, label, parent_report_sha256)
        a,b = control_report,reduced_report
        audit.equal('pair/weights', (a['branch']['kl_weight'],b['branch']['kl_weight']), (1.,.1))
        for field in a['branch']:
            if field not in ('kl_weight','recipe_as_declared'):
                audit.equal('pair/branch/'+field, a['branch'][field], b['branch'][field])
        _same_except_kl(audit, a['configuration'], b['configuration'], 'pair/only_declared_KL_change')
        audit.equal('pair/origin_complete_state', a['origin_boundary_by_rank'], b['origin_boundary_by_rank'])
        ap,bp = (r['configuration']['execution_identity']['payload'] for r in (a,b))
        for key in ('data','partition','plan','schedule','execution','sources','runtime','determinism','storage_policy'):
            audit.equal('pair/'+key, (key in ap,ap.get(key)), (key in bp,bp.get(key)))
        audit.equal('pair/evaluation_plan', a['evaluation_policy']['plan'], b['evaluation_policy']['plan'])
        first = a['branch']['parent_update']+1
        for key in ('loss_sums','counts'):
            audit.equal('pair/first_forward_'+key, a['updates'][str(first)][0]['metrics'][key],
                        b['updates'][str(first)][0]['metrics'][key])
        for step in range(first,expected_stop+1):
            for key in ('lr_used','lr_next'):
                audit.equal(f'pair/update{step}/'+key,a['updates'][str(step)][0]['metrics'][key],
                            b['updates'][str(step)][0]['metrics'][key])
            audit.equal(f'pair/update{step}/accounting', a['observations'][str(step)]['rank_data'],
                        b['observations'][str(step)]['rank_data'])
            if a['observation_mode'] == b['observation_mode'] == 'acceptance':
                audit.equal(f'pair/update{step}/inputs_noise', [r['input'] for r in a['updates'][str(step)]],
                            [r['input'] for r in b['updates'][str(step)]])
        if a['scale']=='tiny':
            audit.equal('control/acceptance',a['observation_mode'],'acceptance')
            training_parity(audit,parent_report,a)
            previous={row['after_update']:row for row in parent_report['evaluations']}
            for row in a['evaluations']:
                audit.equal('control/raw_evaluation'+str(row['after_update']),_raw_evaluation(row),
                            _raw_evaluation(previous[row['after_update']]))
        origin = first-1
        ae = {r['after_update']:r for r in a['evaluations']}
        be = {r['after_update']:r for r in b['evaluations']}
        if origin in ae or origin in be:
            audit.require('pair/origin_eval_both', origin in ae and origin in be)
            audit.equal('pair/origin_raw_evaluation', _raw_evaluation(ae[origin]), _raw_evaluation(be[origin]))
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        return _result(audit, f'{type(error).__name__}: {error}', comparison_kind='paired_fork')
    return _result(audit, comparison_kind='paired_fork')


def audit_restart(reference, resumed, *, source_publication, expected_stop,
                  source_roots=None, check_transport=True):
    """Require exact tiny continuation after a committed child-branch checkpoint."""
    audit = Audit(); source_roots = source_roots or {}
    try:
        for label, report in (('reference',reference),('resumed',resumed)):
            validate_report(audit,report,label,expected_stop=expected_stop,
                            source_root=source_roots.get(label),check_transport=check_transport)
            audit.equal(label+'/tiny_acceptance',(report['scale'],report['observation_mode']),('tiny','acceptance'))
        audit.equal('restart/child_flag',resumed.get('branch_resume'),True)
        for field in ('configuration','branch','original_configuration','sources','source_fingerprint','evaluation_policy'):
            audit.equal('restart/'+field,reference[field],resumed[field])
        audit.require('restart/source_publication',source_publication is not None)
        manifest,_,identity = publication_metadata(source_publication)
        audit.equal('restart/manifest_pin',resumed['resume']['manifest_sha256'],source_publication['manifest_sha256'])
        audit.equal('restart/identity',identity,resumed['configuration']['execution_identity'])
        audit.equal('restart/counters',manifest['counters'],resumed['origin_boundary_by_rank'][0]['state']['counters'])
        audit.equal('restart/cursors',manifest['rank_cursors'],[r['cursor'] for r in resumed['origin_boundary_by_rank']])
        training_parity(audit, reference, resumed)
        previous={r['after_update']:r for r in reference['evaluations']}
        for row in resumed['evaluations']:
            audit.equal('restart/raw_eval'+str(row['after_update']),_raw_evaluation(row),
                        _raw_evaluation(previous[row['after_update']]))
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        return _result(audit,f'{type(error).__name__}: {error}',comparison_kind='same_branch_restart')
    return _result(audit,comparison_kind='same_branch_restart')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('pair','restart'),required=True)
    parser.add_argument('--input',action='append',nargs=3,metavar=('LABEL','PATH','SHA256'),required=True)
    parser.add_argument('--expected-stop',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();reports={};pins={}
    for label,path,pin in args.input:
        if label in reports:parser.error('Duplicate input label')
        reports[label],pins[label]=bounded_json(Path(path),pin)
    required={'parent','control','reduced'} if args.kind=='pair' else {'reference','resumed','publication'}
    if set(reports)!=required:parser.error('Require exact input labels '+str(sorted(required)))
    result=(audit_pair(reports['parent'],reports['control'],reports['reduced'],expected_stop=args.expected_stop,
                      parent_report_sha256=pins['parent']['sha256'])
        if args.kind=='pair' else audit_restart(reports['reference'],reports['resumed'],
            source_publication=reports['publication'],expected_stop=args.expected_stop))
    for pin in pins.values():
        if file_sha(pin['path'])!=pin['sha256']:raise ValueError('Audit input changed')
    result['inputs']=pins
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
