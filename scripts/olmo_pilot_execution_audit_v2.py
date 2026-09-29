#!/usr/bin/env python3
"""Post-runtime audit correction for nondeterministic evaluation wall time.

The accepted runtime and v1 auditor remain immutable. Reuse all individual
report, numerical and storage validators. Exclude only elapsed_seconds from
cross-process preservation equality and accept the engine's absent empty update
map at a terminal boundary; no scientific tolerance or schema conversion.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_pilot_execution_audit as original
from scripts.olmo_pilot_execution_audit import (Audit, legacy, storage_audit,
    publication_metadata, expected_counts, expected_cursor, file_sha, bounded_json,
    storage_check)

SCHEMA='olmo-pilot-execution-audit-v2'


def validate_report(audit, report, label, source_root=None):
    payload=original.validate_report(audit,report,label,source_root)
    for event in report['evaluations']:
        for name,panel in event['panels'].items():
            for rank,part in enumerate(panel['by_rank']):
                seconds=part['preservation'].get('elapsed_seconds')
                audit.require(f'{label}/eval{event["after_update"]}/{name}/rank{rank}/finite_nonnegative_elapsed',
                    type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0)
    return payload


def training_parity(audit, reference, actual):
    if 'updates' in actual:
        return original.training_parity(audit,reference,actual)
    # The unchanged engine omits update maps when no update was executed.
    # The original individual validator already enforces this exact zero-step
    # condition; repeat it here before comparing terminal boundary evidence.
    start=actual.get('resume',{}).get('completed_update',0)
    final=actual['final_counters']['optimizer_updates']
    audit.equal('training/absent_updates_only_at_terminal',final,start)
    audit.equal('training/absent_update_observations',actual.get('observations',{}),{})
    audit.equal('training/origin',actual['origin_boundary_by_rank'],legacy.boundary_at(reference,start))
    audit.equal('training/final',actual['final_boundary_by_rank'],legacy.boundary_at(reference,final))


def compare(reference, actual, *, kind, reference_source_root=None, actual_source_root=None,
            storage_evidence=None, resume_publication=None):
    audit=Audit()
    try:
        if kind not in ('insertion','resume'):raise ValueError('Unknown ordered audit kind')
        if reference.get('scale')!='tiny' or actual.get('scale')!='tiny':
            raise ValueError('Ordered comparison audit is bounded to tiny acceptance')
        rp=validate_report(audit,reference,'reference',reference_source_root)
        ap=validate_report(audit,actual,'actual',actual_source_root)
        audit.equal('comparison/acceptance',(reference['observation_mode'],actual['observation_mode']),('acceptance','acceptance'))
        if kind=='insertion':
            audit.require('insertion/no_resume','resume' not in reference and 'resume' not in actual)
            audit.equal('insertion/deferred_reference',reference['evaluation_policy']['plan']['declaration']['kind'],'deferred')
            audit.equal('insertion/evaluated_actual',actual['evaluation_policy']['plan']['declaration']['kind'],'ordered_named_dev_panels_v1')
            audit.equal('insertion/payload_fields',set(rp),set(ap))
            for key in rp:
                if key not in ('resolved_contract_sha256','declaration','evaluation'):
                    audit.equal('insertion/payload/'+key,ap[key],rp[key])
            rd,ad=rp['declaration'],ap['declaration']
            audit.equal('insertion/declaration_fields',set(rd),set(ad))
            for key in rd:
                if key not in ('evaluation','storage_prefix'):
                    audit.equal('insertion/declaration/'+key,ad[key],rd[key])
            rc,ac=reference['configuration'],actual['configuration']
            audit.equal('insertion/configuration_fields',set(rc),set(ac))
            for key in rc:
                if key!='execution_identity':audit.equal('insertion/configuration/'+key,ac[key],rc[key])
            for key in ('sources','arm','scale','startup_import'):
                audit.equal('insertion/'+key,actual.get(key),reference.get(key))
        else:
            audit.require('resume/present','resume' in actual)
            for key in ('configuration','sources','source_fingerprint','arm','scale','declaration_sha256',
                        'resolved_sha256','startup_import','evaluation_policy'):
                audit.equal('resume/'+key,actual.get(key),reference.get(key))
            if resume_publication is None:raise ValueError('Resume audit requires exact pinned source publication')
            manifest,objects,identity=publication_metadata(resume_publication)
            audit.equal('resume/published_identity',identity,actual['configuration']['execution_identity'])
            audit.equal('resume/published_configuration',manifest['metadata']['configuration'],actual['configuration'])
            completed=actual['resume']['completed_update']
            audit.equal('resume/published_counters',manifest['counters'],expected_counts(ap,completed))
            audit.equal('resume/published_cursors',manifest['rank_cursors'],[expected_cursor(ap,completed,rank) for rank in range(2)])
            audit.equal('resume/exact_manifest',actual['resume']['manifest_sha256'],resume_publication['manifest_sha256'])
            by_step={row['after_update']:row for row in reference['evaluations']}
            for row in actual['evaluations']:
                previous=by_step[row['after_update']]
                for name,panel in row['panels'].items():
                    other=previous['panels'][name]
                    for key in ('result','index_manifest_sha256','membership_sha256'):
                        audit.equal(f'resume/eval{row["after_update"]}/{name}/'+key,panel[key],other[key])
                    for rank,part in enumerate(panel['by_rank']):
                        for key in ('rows','accounting','cursor'):
                            audit.equal(f'resume/eval{row["after_update"]}/{name}/rank{rank}/'+key,part[key],other['by_rank'][rank][key])
                        audit.equal(f'resume/eval{row["after_update"]}/{name}/rank{rank}/preservation_except_elapsed',
                            {k:v for k,v in part['preservation'].items() if k != 'elapsed_seconds'},
                            {k:v for k,v in other['by_rank'][rank]['preservation'].items() if k != 'elapsed_seconds'})
        training_parity(audit,reference,actual)
        if storage_evidence is not None:storage_check(audit,actual,storage_evidence)
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        result=audit.result(f'{type(error).__name__}: {error}')
    else:result=audit.result()
    result.update(schema=SCHEMA,comparison_kind=kind,storage_evidence_checked=storage_evidence is not None,
        scope='JSON-only exact ordered training/evaluation and declared storage evidence; no tensor reload, new cloud verification or precision clearance')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('insertion','resume'),required=True)
    for name in ('reference','actual'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
        parser.add_argument('--'+name+'-source-root',type=Path)
    parser.add_argument('--storage-evidence-root',type=Path,required=True)
    parser.add_argument('--resume-publication',type=Path)
    parser.add_argument('--resume-publication-sha256')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if (args.resume_publication is None)!=(args.resume_publication_sha256 is None):parser.error('Pair publication and SHA')
    if (args.kind=='resume')!=(args.resume_publication is not None):parser.error('Publication required for resume only')
    reports={};inputs={}
    for name in ('reference','actual'):
        reports[name],inputs[name]=bounded_json(getattr(args,name),getattr(args,name+'_sha256'))
    publication=None
    if args.resume_publication is not None:
        publication,inputs['resume_publication']=bounded_json(args.resume_publication,args.resume_publication_sha256)
    evidence,storage_pins=storage_audit.load_storage_evidence(args.storage_evidence_root)
    result=compare(reports['reference'],reports['actual'],kind=args.kind,reference_source_root=args.reference_source_root,
        actual_source_root=args.actual_source_root,storage_evidence=evidence,resume_publication=publication)
    for pin in [*inputs.values(),*storage_pins.values()]:
        if file_sha(pin['path'])!=pin['sha256']:raise ValueError('Audit input changed')
    result['inputs']=inputs;result['storage_inputs']=storage_pins
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ('scripts/olmo_pilot_execution_audit_v2.py','tests/test_pilot_execution_audit_v2.py',
        'scripts/olmo_pilot_execution_audit.py','tests/test_pilot_execution_audit.py',
        'scripts/olmo_campaign_ssd_audit.py','scripts/olmo_campaign_execution_audit.py',
        'scripts/olmo_campaign_execution_restore.py','scripts/olmo_campaign_recovery_bundle.py',
        'scripts/olmo_pilot_execution_restore.py','tests/test_pilot_execution_restore.py',
        'scripts/olmo_campaign_ssd_storage.py'):
        destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination);sources[name]=file_sha(destination)
    result['sources']=result['audit_sources']=sources
    (args.output_dir/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
