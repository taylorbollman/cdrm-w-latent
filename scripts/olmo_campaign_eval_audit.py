#!/usr/bin/env python3
"""JSON-only evaluation insertion/recovery audit; no training imports or transfers.

New reports retain their real schema and lineage. The training-evidence rules
are a versioned copy of the frozen execution auditor, sharing its pure helpers.
Only explicitly named declaration/source/schema differences are permitted for
cross-version insertion; generic same-lineage resume permits none.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_campaign_execution_audit as legacy
from scripts.olmo_campaign_execution_audit import (Audit,digest,file_sha,finite_json,expected_counts,
    expected_cursor,boundary_check,OBSERVATION_SCHEMA,COUNTERS,TERMS,HEAVY)

SCHEMA='olmo-campaign-evaluation-audit-v1'
NEW_REPORT_SCHEMA='olmo-campaign-eval-execute-report-v1'
NEW_CONFIG_SCHEMA='olmo-campaign-evaluation-execution-v1'
NEW_TINY_SCHEMA='olmo-campaign-evaluation-tiny-acceptance-v1'
NEW_SOURCES={
    'scripts/olmo_campaign_eval_engine.py','scripts/olmo_campaign_eval_execute.py',
    'scripts/olmo_campaign_eval_control.py','scripts/olmo_campaign_evaluation.py',
    'tests/test_campaign_eval_control.py','tests/test_campaign_eval_execute.py','tests/test_campaign_evaluation.py',
    'docs/reports/olmo-campaign-evaluation/evaluator-contract.md','docs/reports/olmo-campaign-evaluation/protocol.md'}

def _audit_new_training(audit, report, label, source_root=None):
    check = lambda name, condition: audit.require(label+'/'+name, condition)
    equal = lambda name, actual, expected: audit.equal(label+'/'+name, actual, expected)
    check('finite_json', finite_json(report))
    equal('schema', report['schema'], NEW_REPORT_SCHEMA)
    check('closed_success', report['status'] in ('completed_plan', 'stopped_at_boundary'))
    configuration = report['configuration']; identity = configuration['execution_identity']; payload = identity['payload']
    equal('identity_schema', identity['schema'], 'olmo-campaign-execution-identity-v1')
    equal('identity_digest', identity['sha256'], digest(payload))
    equal('arm', report['arm'], payload['arm'])
    check('known_arm', payload['arm'] in ('B','N','F','R','NF','NR','FR','NFR'))
    equal('two_ranks', payload['partition']['world_size'], 2)
    equal('configuration_recipe', configuration['recipe'], payload['recipe'])
    equal('configuration_ownership', configuration['parameters'], payload['model_contract'])
    equal('configuration_precision', configuration['training']['precision'], payload['execution']['precision'])
    equal('sources', report['sources'], payload['sources'])
    check('nonempty_sources', isinstance(report['sources'], dict) and bool(report['sources']))
    fingerprint = report['source_fingerprint']
    equal('fingerprint_identity', fingerprint['execution_identity_sha256'], identity['sha256'])
    equal('fingerprint_digest', fingerprint['sha256'], identity['sha256'])
    equal('fingerprint_scope', fingerprint['scope'], 'execution_identity')
    equal('fingerprint_sources', fingerprint['sources'], report['sources'])
    equal('fingerprint_source', fingerprint['source_checkpoint'], configuration['source_checkpoint'])
    for name, pin in report['sources'].items():
        path = Path(name)
        check('source_safe/'+name, not path.is_absolute() and '..' not in path.parts and bool(name)
              and isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin) is not None)
        if source_root is not None:
            candidate = Path(source_root)/path
            check('source_snapshot/'+name, candidate.is_file() and not candidate.is_symlink() and file_sha(candidate) == pin)
    final = report['final_counters']['optimizer_updates']; start = report.get('resume', {}).get('completed_update', 0)
    check('integer_final_counters', all(type(v) is int for v in report['final_counters'].values()))
    expected = expected_counts(payload, final)
    equal('final_counters', report['final_counters'], expected)
    check('valid_segment', type(start) is int and 0 <= start <= final <= len(payload['plan']['updates']))
    equal('loop_start', report['loop']['start_update'], start)
    equal('loop_final', report['loop']['completed_update'], final)
    equal('plan_completed', report['plan_completed'], final == len(payload['plan']['updates']))
    equal('status_matches_plan', report['status'] == 'completed_plan', report['plan_completed'])
    updates = report.get('updates', {})
    equal('complete_update_membership', set(updates), {str(i) for i in range(start+1, final+1)})
    equal('graph_prepared', report['graph_prepared'], final > start)
    if report['graph_prepared']:
        equal('preparation_exact', report['preparation_boundary_exact'], [True]*2)
        check('runner_metadata', len(report['runner_by_rank']) == 2 and all(isinstance(x, dict) and x for x in report['runner_by_rank']))
    else:
        check('no_preparation_evidence', 'preparation_boundary_exact' not in report)
        equal('no_captured_runners', report['runner_by_rank'], [None, None])
    if 'resume' in report:
        equal('resume_no_reference_dependency', report['resume']['reference_report_required'], False)
        check('resume_pin', re.fullmatch('[0-9a-f]{64}', report['resume']['manifest_sha256']) is not None)
    boundary_check(audit, label+'/origin', report['origin_boundary_by_rank'], payload, start)
    boundary_check(audit, label+'/final', report['final_boundary_by_rank'], payload, final)
    mode = report['observation_mode']; check('observer_mode', mode in ('lean', 'acceptance'))
    active = {p['name'] for p in payload['model_contract']['parameter_layout'] if p['requires_grad']}
    check('active_parameters', bool(active))
    for label_clock, completed in (('origin_clocks',start),('final_clocks',final)):
        equal(label_clock, report[label_clock], {'adam_parameters':len(active) if completed else 0,
              'input_tokens':expected_counts(payload,completed)['input_tokens'],
              'optimizer_updates':completed,'scheduler_epoch':completed})
    equal('complete_accounting_membership', set(report.get('observations',{})), set(updates))
    for step in range(start+1, final+1):
        rank_data=report['observations'][str(step)]['rank_data']
        check(f'update{step}/accounting_ranks', isinstance(rank_data,list) and len(rank_data)==2)
        plan_row=payload['plan']['updates'][step-1]
        for key, value in plan_row['counts'].items():
            equal(f'update{step}/data_total/{key}',sum(row[key] for row in rank_data),value)
        for rank, allocated in enumerate(plan_row['allocation_by_arm'][payload['arm']]):
            for actual_key, planned_key in (('valid_tokens','valid_tokens'),('packed_rows','packed_rows'),
                    ('physical_rows','physical_rows'),('padding_tokens','padding_tokens'),('empty_rows','dummy_rows')):
                equal(f'update{step}/allocation/rank{rank}/{actual_key}',rank_data[rank][actual_key],allocated[planned_key])
        rows = updates[str(step)]; check(f'update{step}/rank_count', isinstance(rows, list) and len(rows) == 2)
        if mode == 'acceptance':
            boundary_check(audit, label+f'/update{step}/boundary', [row['boundary'] for row in rows], payload, step)
        before, after = expected_counts(payload, step-1), expected_counts(payload, step)
        for rank, row in enumerate(rows):
            prefix = f'update{step}/rank{rank}/'
            equal(prefix+'observation_fields',set(row), {'schema','mode','metrics','cursor','loss_means','clipping','observation_seconds'}
                  | (set(HEAVY) if mode=='acceptance' else set()))
            equal(prefix+'schema', row['schema'], OBSERVATION_SCHEMA)
            equal(prefix+'mode', row['mode'], mode)
            metrics = row['metrics']
            check(prefix+'complete_metrics', isinstance(metrics,dict) and
                  {'loss_sums','counts','objective','microbatches','documents','input_tokens',
                   'gradient_norm_before_clip','lr_used','lr_next','counters'} <= metrics.keys())
            check(prefix+'lr_groups', isinstance(metrics['lr_used'],list) and bool(metrics['lr_used'])
                  and isinstance(metrics['lr_next'],list) and len(metrics['lr_used']) == len(metrics['lr_next'])
                  and all(type(v) in (int,float) and v >= 0 for key in ('lr_used','lr_next') for v in metrics[key]))
            equal(prefix+'replica_metrics', metrics, rows[0]['metrics'])
            equal(prefix+'counters', metrics['counters'], after)
            equal(prefix+'cursor', row['cursor'], expected_cursor(payload, step, rank))
            for key in ('input_tokens','documents','microbatches'):
                equal(prefix+key, metrics[key], after[key]-before[key])
            counts = {term: after[key]-before[key] for term,key in zip(TERMS,('ce_positions','latent_pairs','kl_triples'))}
            equal(prefix+'loss_counts', metrics['counts'], counts)
            equal(prefix+'loss_terms', set(metrics['loss_sums']), set(TERMS))
            equal(prefix+'loss_means', row['loss_means'], {term: metrics['loss_sums'][term]/counts[term] if counts[term] else None for term in TERMS})
            limit = configuration['training']['max_grad_norm']; norm = metrics['gradient_norm_before_clip']
            check(prefix+'gradient_norm', type(norm) in (int,float) and norm >= 0)
            clipping = row['clipping']
            equal(prefix+'clip_limit', clipping['configured_limit'], limit)
            equal(prefix+'clip_exceeded', clipping['norm_exceeds_limit'], limit is not None and norm > limit)
            equal(prefix+'clip_coefficient', clipping['coefficient_estimate'], 1.0 if limit is None else min(1.0,limit/(norm+1e-6)))
            if mode == 'lean':
                check(prefix+'no_heavy_evidence', not set(HEAVY)&row.keys())
            else:
                check(prefix+'input_evidence', isinstance(row['input'], dict) and bool(row['input']))
                equal(prefix+'all_active_gradients', set(row['raw_gradients']), active)
                check(prefix+'gradient_digests', all(isinstance(v, dict) and {'shape','dtype','sha256'} == set(v)
                      and re.fullmatch('[0-9a-f]{64}', v['sha256']) is not None for v in row['raw_gradients'].values()))
                equal(prefix+'replica_gradients', row['raw_gradients'], rows[0]['raw_gradients'])
    if mode == 'acceptance' and final > start:
        equal('last_update_final_boundary', [row['boundary'] for row in updates[str(final)]], report['final_boundary_by_rank'])
    if final == start:
        equal('no_update_state_change', report['final_boundary_by_rank'], report['origin_boundary_by_rank'])
    return payload




def evaluation_check(audit,report,label):
    policy=report['evaluation_policy'];plan=policy['plan'];declaration=plan['declaration']
    payload=report['configuration']['execution_identity']['payload']
    audit.equal(label+'/resume_policy',policy['resume'],'repeat_scheduled_restored_boundary_once_per_segment')
    audit.equal(label+'/failure_policy',policy['failure'],'earlier committed checkpoint remains authoritative; no save of unverified state')
    if report['scale']=='native':
        audit.equal(label+'/bound_evaluation_plan',payload['evaluation'],plan)
    else:
        audit.equal(label+'/bound_tiny_declaration',payload['declaration']['evaluation'],declaration)
        audit.equal(label+'/tiny_schema',payload['declaration']['schema'],NEW_TINY_SCHEMA)
        audit.equal(label+'/tiny_declaration_digest',payload['resolved_contract_sha256'],digest(payload['declaration']))
    start=report.get('resume',{}).get('completed_update',0);final=report['final_counters']['optimizer_updates']
    rows=report['evaluations']
    if declaration['kind']=='deferred':
        audit.equal(label+'/deferred_plan',set(plan),{'declaration','execution'})
        audit.equal(label+'/deferred_evaluations',rows,[])
        return
    audit.equal(label+'/policy_fields',set(declaration),{'kind','index','index_manifest_sha256','split',
        'target_valid_tokens','every_updates','precision','feedback_jitter','report_passes','generation'})
    for key,value in {'kind':'finite_pass_teacher_forced','split':'dev','precision':'fp32','feedback_jitter':0,
            'report_passes':'all_trained_passes','generation':'not_implemented'}.items():
        audit.equal(label+'/'+key,declaration[key],value)
    audit.require(label+'/index_pin',re.fullmatch('[0-9a-f]{64}',declaration['index_manifest_sha256']) is not None)
    audit.require(label+'/positive_budget_interval',all(type(declaration[k]) is int and declaration[k]>0
        for k in ('target_valid_tokens','every_updates')))
    scheduled=list(range(declaration['every_updates'],len(payload['plan']['updates'])+1,declaration['every_updates']))
    audit.equal(label+'/fixed_schedule',plan['scheduled_updates'],scheduled)
    audit.equal(label+'/actual_evaluations',[r['after_update'] for r in rows],[s for s in scheduled if start<=s<=final])
    audit.equal(label+'/one_fixed_dev_update',len(plan['fixed_plan']['updates']),1)
    fixed=plan['fixed_plan']['updates'][0];raw=fixed['counts'];arm=payload['arm'];recipe=payload['recipe']
    audit.equal(label+'/dev_origin',plan['fixed_plan']['first_cursor'],{'manifest_sha256':declaration['index_manifest_sha256'],
        'split':'dev','next_chunk':0,'next_update':0})
    audit.equal(label+'/dev_declared_budget',fixed['target_valid_tokens'],declaration['target_valid_tokens'])
    expected={'ce':raw['ce_targets'],'latent':raw['latent_pairs'] if 'N' in arm else 0,'kl':raw['kl_triples'] if 'N' in arm else 0}
    count=recipe['fbt_passes'] if 'F' in arm else 1
    coefficients={'ce':[1.] if count==1 else [.5]+[.5/(count-1)]*(count-1),
        'latent':[1./count]*count,'kl':[1./count]*count}
    weights={'ce':1.,'latent':1. if 'N' in arm else 0.,'kl':1. if 'N' in arm else 0.}
    mode={'enabled':'F' in arm,'num_passes':count,'beta':1.,'feedback_jitter':0.,
        'first_pass_policy':recipe['first_pass_policy'],'document_policy':recipe.get('document_policy','isolated-v1'),
        'rt_mode':{'selected_layers':recipe['rt_layers'] if 'R' in arm else [],'alpha':1.}}
    for entry in rows:
        scope=label+'/update'+str(entry['after_update'])
        audit.equal(scope+'/completed',entry['status'],'completed')
        audit.equal(scope+'/training_boundary_preserved',entry['training_boundary_exact_by_rank'],[True,True])
        audit.equal(scope+'/rank_count',len(entry['by_rank']),2)
        local=[]
        for rank,record in enumerate(entry['by_rank']):
            evidence=record['preservation'];checks=evidence['checks']
            required={'tensor_metadata_unchanged','module_ownership_unchanged','cache_generations_unchanged',
                'gradient_identity_unchanged','gradient_values_remained_zero','runtime_restored','modes_restored',
                'rng_restored','autocast_cache_restored'}
            audit.equal(scope+f'/rank{rank}/check_membership',set(checks),required)
            audit.require(scope+f'/rank{rank}/preserved',evidence['restored'] is True and evidence['integrity_passed'] is True
                and all(value is True for value in checks.values()))
            audit.equal(scope+f'/rank{rank}/precision',evidence['precision'],'fp32_math_eager')
            audit.equal(scope+f'/rank{rank}/noise',evidence['feedback_jitter'],0.)
            audit.equal(scope+f'/rank{rank}/dev_cursor',record['cursor'],plan['fixed_plan']['first_cursor'])
            allocation=fixed['allocation_by_arm'][arm][rank]
            audit.equal(scope+f'/rank{rank}/physical_calls',len(record['rows']),allocation['microbatches'])
            for actual_key,planned_key in (('valid_tokens','valid_tokens'),('packed_rows','packed_rows'),
                    ('physical_rows','physical_rows'),('padding_tokens','padding_tokens'),('empty_rows','dummy_rows')):
                audit.equal(scope+f'/rank{rank}/allocation/'+actual_key,record['accounting'][actual_key],allocation[planned_key])
            audit.equal(scope+f'/rank{rank}/local_input_sum',sum(row['input_tokens'] for row in record['rows']),record['accounting']['valid_tokens'])
            for term,key in zip(TERMS,('ce_targets','latent_pairs','kl_triples')):
                audit.equal(scope+f'/rank{rank}/local_target_sum/'+term,sum(row['counts'][term] for row in record['rows']),
                    record['accounting'][key] if weights[term] else 0)
            local.extend(record['rows'])
        for key,value in raw.items():
            audit.equal(scope+'/global_accounting/'+key,sum(r['accounting'][key] for r in entry['by_rank']),value)
        for i,row in enumerate(local):
            prefix=scope+'/physical'+str(i)
            audit.equal(prefix+'/schema',row['schema'],'olmo-campaign-local-evaluation-v1')
            audit.equal(prefix+'/policy',row['policy'],'common_fp32_no_jitter_v1')
            audit.equal(prefix+'/mode',row['mode'],mode)
            audit.equal(prefix+'/weights',row['weights'],weights)
            audit.equal(prefix+'/enabled',row['enabled'],{k:bool(v) for k,v in weights.items()})
            audit.equal(prefix+'/coefficients',row['term_pass_coefficients'],coefficients)
            audit.equal(prefix+'/pass_membership',[p['index'] for p in row['passes']],list(range(count)))
            audit.equal(prefix+'/count_terms',set(row['counts']),set(TERMS))
            audit.require(prefix+'/nonnegative_counts',all(type(v) is int and v>=0 for v in row['counts'].values()))
            audit.require(prefix+'/nonnegative_tokens',type(row['input_tokens']) is int and row['input_tokens']>=0)
            audit.equal(prefix+'/aggregate_terms',set(row['aggregate_sums']),set(TERMS))
            for p in row['passes']:
                audit.equal(prefix+f'/pass{p["index"]}/counts',p['counts'],row['counts'])
                audit.equal(prefix+f'/pass{p["index"]}/sum_terms',set(p['sums']),set(TERMS))
            for sums in [row['aggregate_sums'],*[p['sums'] for p in row['passes']]]:
                audit.require(prefix+'/numeric_sums',all(type(v) in (int,float) and math.isfinite(v) for v in sums.values()))
                for term in TERMS:
                    if not weights[term]:audit.equal(prefix+'/disabled_'+term,(sums[term],row['counts'][term]),(0.,0))
        counts={t:sum(r['counts'][t] for r in local) for t in TERMS}
        audit.equal(scope+'/global_targets',counts,expected)
        audit.equal(scope+'/global_inputs',sum(r['input_tokens'] for r in local),raw['valid_tokens'])
        def reduce_sums(values):
            sums={t:math.fsum(v[t] for v in values) for t in TERMS}
            for term in TERMS:
                if not weights[term]:audit.equal(scope+'/disabled_'+term,(sums[term],counts[term]),(0.,0))
                else:audit.require(scope+'/positive_'+term,counts[term]>0)
            return {'sums':sums,'counts':counts,'means':{t:sums[t]/counts[t] if counts[t] else None for t in TERMS}}
        passes=[{'index':i,**reduce_sums([r['passes'][i]['sums'] for r in local])} for i in range(count)]
        aggregate=reduce_sums([r['aggregate_sums'] for r in local])
        reconstructed={t:math.fsum(coefficients[t][i]*passes[i]['sums'][t] for i in range(count)) for t in TERMS}
        for term in TERMS:
            audit.require(scope+'/weighted_rounding/'+term,math.isclose(reconstructed[term],aggregate['sums'][term],rel_tol=2e-6,abs_tol=1e-6))
        wanted={'schema':'olmo-campaign-evaluation-control-v1','passes':passes,'aggregate':aggregate,
            'objective':math.fsum(weights[t]*(aggregate['means'][t] or 0.) for t in TERMS),
            'input_tokens':raw['valid_tokens'],'enabled':{t:bool(weights[t]) for t in TERMS},'weights':weights,
            'term_pass_coefficients':coefficients,'reconstructed_aggregate_sums':reconstructed,'policy':'common_fp32_no_jitter_v1'}
        audit.equal(scope+'/independent_global_reduction',entry['result'],wanted)


def validate_report(audit,report,label,source_root):
    if report['schema']==legacy.REPORT_SCHEMA:
        return legacy._audit_report(audit,report,label,source_root)
    audit.equal(label+'/actual_new_schema',report['schema'],NEW_REPORT_SCHEMA)
    payload=_audit_new_training(audit,report,label,source_root)
    audit.equal(label+'/actual_configuration_schema',report['configuration']['schema'],NEW_CONFIG_SCHEMA)
    evaluation_check(audit,report,label+'/evaluation')
    return payload


def allowed_insertion_contract(audit,reference,actual,rp,ap):
    audit.equal('insertion/report_schema',actual['schema'],NEW_REPORT_SCHEMA)
    audit.equal('insertion/arm',actual['arm'],reference['arm'])
    audit.equal('insertion/scale',actual['scale'],reference['scale'])
    audit.equal('insertion/observer',actual['observation_mode'],'acceptance')
    audit.equal('insertion/reference_observer',reference['observation_mode'],'acceptance')
    old=reference['schema']==legacy.REPORT_SCHEMA
    rs,ns=reference['sources'],actual['sources']
    audit.equal('insertion/source_membership',set(ns),set(rs)|NEW_SOURCES if old else set(rs))
    for name,pin in rs.items():audit.equal('insertion/inherited_source/'+name,ns[name],pin)
    rc,ac=reference['configuration'],actual['configuration']
    audit.equal('insertion/configuration_fields',set(rc),set(ac))
    for key in rc:
        if key not in ('schema','execution_identity'):audit.equal('insertion/configuration/'+key,ac[key],rc[key])
    audit.equal('insertion/reference_config_schema',rc['schema'],'olmo-campaign-execution-v1' if old else NEW_CONFIG_SCHEMA)
    audit.equal('insertion/payload_fields',set(rp),set(ap))
    allowed={'resolved_contract_sha256','sources','evaluation','declaration'}
    for key in rp:
        if key not in allowed:audit.equal('insertion/payload/'+key,ap[key],rp[key])
    for name,payload in (('reference',rp),('actual',ap)):
        audit.require('insertion/'+name+'/resolved_pin',re.fullmatch('[0-9a-f]{64}',payload['resolved_contract_sha256']) is not None)
    if actual['scale']=='tiny':
        rd,ad=rp['declaration'],ap['declaration']
        audit.equal('insertion/tiny_declaration_fields',set(ad),set(rd)|{'evaluation'})
        for key in rd:
            if key not in ('schema','evaluation','storage_prefix'):audit.equal('insertion/tiny_declaration/'+key,ad[key],rd[key])
        audit.equal('insertion/actual_tiny_schema',ad['schema'],NEW_TINY_SCHEMA)
    else:
        audit.equal('insertion/reference_deferred',rp['evaluation']['declaration']['kind'],'deferred')
    audit.equal('insertion/active_evaluation',actual['evaluation_policy']['plan']['declaration']['kind'],'finite_pass_teacher_forced')
    audit.equal('insertion/startup_import',actual.get('startup_import'),reference.get('startup_import'))
    audit.equal('insertion/reference_source_weights',actual['source_fingerprint']['source_checkpoint'],reference['source_fingerprint']['source_checkpoint'])


def compare(reference,actual,*,kind,reference_source_root=None,actual_source_root=None):
    audit=Audit()
    try:
        if kind not in ('insertion','resume'):raise ValueError('Unknown audit comparison kind')
        rp=validate_report(audit,reference,'reference',reference_source_root)
        ap=validate_report(audit,actual,'actual',actual_source_root)
        if kind=='insertion':allowed_insertion_contract(audit,reference,actual,rp,ap)
        else:
            audit.equal('resume/reference_schema',reference['schema'],NEW_REPORT_SCHEMA)
            audit.equal('resume/actual_schema',actual['schema'],NEW_REPORT_SCHEMA)
            audit.require('resume/present','resume' in actual)
            for key in ('configuration','sources','source_fingerprint','arm','scale','declaration_sha256','resolved_sha256','startup_import','evaluation_policy'):
                audit.equal('resume/'+key,actual.get(key),reference.get(key))
        start=actual.get('resume',{}).get('completed_update',0);final=actual['final_counters']['optimizer_updates']
        audit.equal('training/origin',actual['origin_boundary_by_rank'],legacy.boundary_at(reference,start))
        audit.equal('training/final',actual['final_boundary_by_rank'],legacy.boundary_at(reference,final))
        for step,rows in actual.get('updates',{}).items():
            audit.require('training/update'+step+'/reference_coverage',step in reference['updates'])
            audit.equal('training/update'+step+'/rank_accounting',actual['observations'][step]['rank_data'],reference['observations'][step]['rank_data'])
            for rank,row in enumerate(rows):
                want=reference['updates'][step][rank]
                audit.equal(f'training/update{step}/rank{rank}/row_fields',set(row),set(want))
                for key in row:
                    if key!='observation_seconds':audit.equal(f'training/update{step}/rank{rank}/'+key,row[key],want[key])
        if kind=='resume':
            previous={row['after_update']:row for row in reference['evaluations']}
            for row in actual['evaluations']:
                step=row['after_update'];audit.require('resume/eval_coverage/'+str(step),step in previous)
                audit.equal('resume/eval_result/'+str(step),row['result'],previous[step]['result'])
                for rank,part in enumerate(row['by_rank']):
                    for key in ('rows','accounting','cursor'):
                        audit.equal(f'resume/eval{step}/rank{rank}/'+key,part[key],previous[step]['by_rank'][rank][key])
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        result=audit.result(f'{type(error).__name__}: {error}')
    else:result=audit.result()
    result.update(schema=SCHEMA,comparison_kind=kind,scope='Actual report lineage and exact training evidence, independent per-pass dev arithmetic; no tensor reload, GPU, precision equivalence or quality claim')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=('insertion','resume'),required=True)
    for name in ('reference','actual'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
        parser.add_argument('--'+name+'-source-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args(argv)
    reports,pins={},{}
    for name in ('reference','actual'):
        path=getattr(args,name);pin=getattr(args,name+'_sha256')
        if path.is_symlink() or path.stat().st_size>128*1024*1024 or file_sha(path)!=pin:raise ValueError(name+' report differs from bounded authority')
        reports[name]=json.loads(path.read_text());pins[name]={'path':str(path),'sha256':pin,'size_bytes':path.stat().st_size}
    result=compare(reports['reference'],reports['actual'],kind=args.kind,
        reference_source_root=args.reference_source_root,actual_source_root=args.actual_source_root)
    for name in reports:
        if file_sha(getattr(args,name))!=pins[name]['sha256']:raise ValueError(name+' report changed during audit')
    result['inputs']=pins;args.output_dir.mkdir(parents=True,exist_ok=False)
    snapshot=args.output_dir/'source-snapshot';snapshot.mkdir()
    files=(Path(__file__),ROOT/'scripts/olmo_campaign_execution_audit.py',ROOT/'tests/test_campaign_eval_audit.py',
           ROOT/'tests/test_campaign_execution_audit.py')
    sources={}
    for path in files:
        relative=path.relative_to(ROOT);destination=snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        sources[str(relative)]=file_sha(destination)
    result['audit_sources']=sources;result['sources']=sources
    with tempfile.NamedTemporaryFile(mode='w',dir=args.output_dir,delete=False) as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno());temporary=stream.name
    os.replace(temporary,args.output_dir/'report.json')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
