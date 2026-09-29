#!/usr/bin/env python3
"""JSON-only tiny SSD execution/recovery audit; no tensor loading, cloud or GPU.

Incoming reports keep their real schemas and identities. Training rules below
are the frozen evaluation-auditor rules with an explicit SSD schema; pure
boundary, accounting and evaluation validators are reused unchanged.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_campaign_eval_audit as evaluation
from scripts import olmo_campaign_execution_audit as legacy
from scripts.olmo_campaign_execution_audit import (Audit, digest, file_sha, finite_json,
    expected_counts, expected_cursor, boundary_check, OBSERVATION_SCHEMA, COUNTERS, TERMS, HEAVY)
from scripts.olmo_campaign_execution_restore import publication_metadata

SCHEMA = 'olmo-campaign-ssd-audit-v1'
SSD_REPORT_SCHEMA = 'olmo-campaign-ssd-execute-report-v1'
SSD_CONFIG_SCHEMA = 'olmo-campaign-ssd-execution-v1'
STORAGE_POLICY = {'schema': 'olmo-campaign-ssd-policy-v1',
                  'location': 'local_ssd_verified_gcs', 'keep_local_completed': 2}
NEW_SOURCES = {
    'scripts/olmo_campaign_ssd_engine.py', 'scripts/olmo_campaign_ssd_execute.py',
    'scripts/olmo_campaign_ssd_storage.py', 'scripts/olmo_campaign_ssd_restore.py',
    'tests/test_campaign_ssd_execute.py', 'tests/test_campaign_ssd_storage.py',
    'tests/test_campaign_ssd_restore.py', 'docs/reports/olmo-campaign-storage/protocol.md'}

def training_check(audit, report, label, source_root=None):
    check = lambda name, condition: audit.require(label+'/'+name, condition)
    equal = lambda name, actual, expected: audit.equal(label+'/'+name, actual, expected)
    check('finite_json', finite_json(report))
    equal('schema', report['schema'], SSD_REPORT_SCHEMA)
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



def validate_report(audit, report, label, source_root=None):
    if report['schema'] == evaluation.NEW_REPORT_SCHEMA:
        return evaluation.validate_report(audit, report, label, source_root)
    payload = training_check(audit, report, label, source_root)
    audit.equal(label+'/configuration_schema', report['configuration']['schema'], SSD_CONFIG_SCHEMA)
    audit.equal(label+'/storage_policy', payload['storage_policy'], STORAGE_POLICY)
    audit.equal(label+'/reported_storage_policy', report['storage']['policy'], STORAGE_POLICY)
    evaluation.evaluation_check(audit, report, label+'/evaluation')
    return payload


def transition_check(audit, reference, actual, rp, ap):
    """Allow only the new storage implementation/identity, never training drift."""
    audit.equal('transition/reference_schema', reference['schema'], evaluation.NEW_REPORT_SCHEMA)
    audit.equal('transition/actual_schema', actual['schema'], SSD_REPORT_SCHEMA)
    audit.equal('transition/tiny_scope', (reference['scale'], actual['scale']), ('tiny', 'tiny'))
    audit.equal('transition/arm', reference['arm'], actual['arm'])
    audit.equal('transition/acceptance', (reference['observation_mode'], actual['observation_mode']),
                ('acceptance', 'acceptance'))
    rs, ns = reference['sources'], actual['sources']
    audit.equal('transition/source_membership', set(ns), set(rs) | NEW_SOURCES)
    for name, pin in rs.items():
        audit.equal('transition/inherited_source/'+name, ns[name], pin)
    rc, ac = reference['configuration'], actual['configuration']
    audit.equal('transition/configuration_fields', set(rc), set(ac))
    for key in rc:
        if key not in ('schema', 'execution_identity'):
            audit.equal('transition/configuration/'+key, ac[key], rc[key])
    audit.equal('transition/payload_fields', set(ap), set(rp) | {'storage_policy'})
    for key in rp:
        if key not in ('sources', 'declaration', 'resolved_contract_sha256'):
            audit.equal('transition/payload/'+key, ap[key], rp[key])
    rd, ad = rp['declaration'], ap['declaration']
    audit.equal('transition/declaration_fields', set(ad), set(rd))
    for key in rd:
        if key != 'storage_prefix':
            audit.equal('transition/declaration/'+key, ad[key], rd[key])
    for name, payload in (('reference', rp), ('actual', ap)):
        audit.equal('transition/'+name+'/declared_identity', payload['resolved_contract_sha256'],
                    digest(payload['declaration']))
    audit.equal('transition/startup_import', actual.get('startup_import'), reference.get('startup_import'))
    audit.equal('transition/evaluation_policy', actual['evaluation_policy'], reference['evaluation_policy'])
    audit.equal('transition/source_weights', actual['source_fingerprint']['source_checkpoint'],
                reference['source_fingerprint']['source_checkpoint'])


def training_parity(audit, reference, actual):
    start = actual.get('resume', {}).get('completed_update', 0)
    final = actual['final_counters']['optimizer_updates']
    audit.equal('training/origin', actual['origin_boundary_by_rank'], legacy.boundary_at(reference, start))
    audit.equal('training/final', actual['final_boundary_by_rank'], legacy.boundary_at(reference, final))
    for step, rows in actual.get('updates', {}).items():
        audit.require('training/update'+step+'/reference_coverage', step in reference.get('updates', {}))
        audit.equal('training/update'+step+'/rank_accounting', actual['observations'][step]['rank_data'],
                    reference['observations'][step]['rank_data'])
        for rank, row in enumerate(rows):
            wanted = reference['updates'][step][rank]
            audit.equal(f'training/update{step}/rank{rank}/fields', set(row), set(wanted))
            for key in row:
                if key != 'observation_seconds':
                    audit.equal(f'training/update{step}/rank{rank}/'+key, row[key], wanted[key])
    previous = {row['after_update']: row for row in reference['evaluations']}
    for row in actual['evaluations']:
        step = row['after_update']
        audit.require('evaluation/reference_coverage/'+str(step), step in previous)
        audit.equal('evaluation/result/'+str(step), row['result'], previous[step]['result'])
        for rank, part in enumerate(row['by_rank']):
            for key in ('rows', 'accounting', 'cursor'):
                audit.equal(f'evaluation/update{step}/rank{rank}/'+key,
                            part[key], previous[step]['by_rank'][rank][key])


def compare(reference, actual, *, kind, reference_source_root=None, actual_source_root=None,
            storage_evidence=None, resume_publication=None):
    audit = Audit()
    try:
        if kind not in ('storage_transition', 'resume'):
            raise ValueError('Unknown SSD audit kind')
        if reference.get('scale') != 'tiny' or actual.get('scale') != 'tiny':
            raise ValueError('SSD comparison audit supports tiny acceptance only; native assets require a separate restore audit')
        rp = validate_report(audit, reference, 'reference', reference_source_root)
        ap = validate_report(audit, actual, 'actual', actual_source_root)
        if kind == 'storage_transition':
            transition_check(audit, reference, actual, rp, ap)
        else:
            audit.equal('resume/schemas', (reference['schema'], actual['schema']),
                        (SSD_REPORT_SCHEMA, SSD_REPORT_SCHEMA))
            audit.require('resume/present', 'resume' in actual)
            for key in ('configuration', 'sources', 'source_fingerprint', 'arm', 'scale',
                        'declaration_sha256', 'resolved_sha256', 'startup_import', 'evaluation_policy'):
                audit.equal('resume/'+key, actual.get(key), reference.get(key))
            if resume_publication is None:
                raise ValueError('Resume audit requires the exact pinned source publication')
            manifest, objects, identity = publication_metadata(resume_publication)
            audit.equal('resume/published_identity', identity, actual['configuration']['execution_identity'])
            audit.equal('resume/published_configuration', manifest['metadata']['configuration'], actual['configuration'])
            audit.equal('resume/published_fingerprint', manifest['metadata']['source_fingerprint'], actual['source_fingerprint'])
            completed = actual['resume']['completed_update']
            audit.equal('resume/published_counters', manifest['counters'], expected_counts(ap, completed))
            audit.equal('resume/published_cursors', manifest['rank_cursors'],
                        [expected_cursor(ap, completed, rank) for rank in range(2)])
            audit.equal('resume/exact_published_manifest', actual['resume']['manifest_sha256'],
                        resume_publication['manifest_sha256'])
        training_parity(audit, reference, actual)
        if storage_evidence is not None:
            storage_check(audit, actual, storage_evidence)
    except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        result = audit.result(f'{type(error).__name__}: {error}')
    else:
        result = audit.result()
    result.update(schema=SCHEMA, comparison_kind=kind, storage_evidence_checked=storage_evidence is not None,
        scope='JSON-only exact training/evaluation evidence and declared storage receipts/journal; '
              'no checkpoint tensor reload, new cloud verification, hardware or precision clearance')
    return result


def bounded_json(path, pin=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024**2:
        raise ValueError('Require bounded regular JSON authority')
    actual = file_sha(path)
    if pin is not None and actual != pin:
        raise ValueError('JSON authority bytes differ')
    return json.loads(path.read_text()), {'path': str(path), 'sha256': actual, 'size_bytes': path.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('storage_transition', 'resume'), required=True)
    for name in ('reference', 'actual'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
        parser.add_argument('--'+name+'-source-root', type=Path)
    parser.add_argument('--storage-evidence-root', type=Path, required=True)
    parser.add_argument('--resume-publication', type=Path)
    parser.add_argument('--resume-publication-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.resume_publication is None) != (args.resume_publication_sha256 is None):
        parser.error('Resume publication path and SHA256 must be paired')
    if (args.kind == 'resume') != (args.resume_publication is not None):
        parser.error('Provide an exact source publication for resume audits only')
    reports, inputs = {}, {}
    for name in ('reference', 'actual'):
        reports[name], inputs[name] = bounded_json(getattr(args, name), getattr(args, name+'_sha256'))
    resume_publication = None
    if args.resume_publication is not None:
        resume_publication, inputs['resume_publication'] = bounded_json(
            args.resume_publication, args.resume_publication_sha256)
    storage_evidence, storage_pins = load_storage_evidence(args.storage_evidence_root)
    result = compare(reports['reference'], reports['actual'], kind=args.kind,
        reference_source_root=args.reference_source_root, actual_source_root=args.actual_source_root,
        storage_evidence=storage_evidence, resume_publication=resume_publication)
    for value in [*inputs.values(), *storage_pins.values()]:
        if file_sha(value['path']) != value['sha256']:
            raise ValueError('Audit input changed during verification')
    result['inputs'] = inputs
    result['storage_inputs'] = storage_pins
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = {}
    names = ('scripts/olmo_campaign_ssd_audit.py', 'tests/test_campaign_ssd_audit.py',
             'scripts/olmo_campaign_eval_audit.py', 'tests/test_campaign_eval_audit.py',
             'scripts/olmo_campaign_execution_audit.py', 'tests/test_campaign_execution_audit.py',
             'scripts/olmo_campaign_execution_restore.py', 'scripts/olmo_campaign_recovery_bundle.py')
    for name in names:
        destination = args.output_dir/'source-snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        sources[name] = file_sha(destination)
    result['sources'] = result['audit_sources'] = sources
    with tempfile.NamedTemporaryFile(mode='w', dir=args.output_dir, delete=False) as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno()); temporary = stream.name
    os.replace(temporary, args.output_dir/'report.json')
    print(json.dumps({'passed': result['passed'], 'checks': len(result['checks']), 'failures': result['failures']}))
    return 0 if result['passed'] else 1


# Storage evidence functions are defined below before command-line invocation.
def load_storage_evidence(root):
    """Read only persistent JSON records; original SSD directories are not opened."""
    root = Path(root)
    journal, jp = bounded_json(root/'ssd-journal.json')
    pins = {'journal': jp}
    receipts = {}
    for entry in journal['published']:
        number = entry['update']
        relative = f'checkpoint-publications/update-{number:06d}.json'
        if entry['receipt_path'] != relative or relative in receipts:
            raise ValueError('Receipt path or unique update differs from journal')
        value, pin = bounded_json(root/relative, entry['receipt_sha256'])
        receipts[relative] = value
        pins[relative] = pin
    latest = None
    if journal['published']:
        latest, lp = bounded_json(root/'latest-checkpoint.json')
        pins['latest'] = lp
    elif (root/'latest-checkpoint.json').exists():
        raise ValueError('Unpublished journal has an unexpected latest receipt')
    return {'journal': journal, 'receipts': receipts, 'latest': latest,
            'journal_sha256': jp['sha256']}, pins


def storage_check(audit, report, evidence):
    journal, receipts = evidence['journal'], evidence['receipts']
    payload = report['configuration']['execution_identity']['payload']
    identity = report['configuration']['execution_identity']
    audit.equal('storage/journal_fields', set(journal),
                {'schema', 'ownership', 'destinations', 'published', 'prune_operations'})
    audit.equal('storage/journal_schema', journal['schema'], 'olmo-campaign-ssd-journal-v1')
    owned = journal['ownership']
    audit.equal('storage/ownership_fields', set(owned), {'schema', 'checkpoint_root', 'evidence_dir',
        'ssd_mount', 'namespace', 'segment', 'durability', 'execution_identity_sha256', 'storage_prefix',
        'keep_local_completed', 'resume_source'})
    audit.equal('storage/ownership_schema', owned['schema'], 'olmo-campaign-ssd-ownership-v1')
    audit.equal('storage/execution_identity', owned['execution_identity_sha256'], identity['sha256'])
    audit.equal('storage/keep', owned['keep_local_completed'], payload['storage_policy']['keep_local_completed'])
    audit.equal('storage/root', owned['checkpoint_root'], report['storage']['checkpoint_root'])
    audit.equal('storage/evidence', owned['evidence_dir'], report['storage']['evidence_dir'])
    root, persistent = Path(owned['checkpoint_root']), Path(owned['evidence_dir'])
    audit.equal('storage/mount', owned['ssd_mount'], '/mnt/localssd')
    audit.equal('storage/root_components', root.parts,
                ('/', 'mnt', 'localssd', 'cdrm-checkpoints', owned['namespace'], owned['segment']))
    audit.require('storage/simple_names', all(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', owned[k]) is not None
                                            for k in ('namespace', 'segment')))
    audit.require('storage/persistent_path', persistent.is_absolute() and '..' not in persistent.parts
                  and not persistent.is_relative_to(Path('/mnt/localssd')))
    audit.equal('storage/durability', owned['durability'], 'volatile SSD; persistent verified-GCS receipts')
    audit.equal('storage/prefix', owned['storage_prefix'],
                payload['declaration']['storage_prefix']+'/'+report['arm']+'/'+persistent.name)
    if 'resume' in report:
        audit.require('storage/protected_resume_source', isinstance(owned['resume_source'], str)
                      and Path(owned['resume_source']).is_absolute())
    else:
        audit.equal('storage/no_resume_source', owned['resume_source'], None)
    published = journal['published']
    updates = [row['update'] for row in published]
    audit.require('storage/publication_order', all(type(n) is int and n >= 0 for n in updates)
                  and updates == sorted(set(updates)))
    audit.equal('storage/report_publication_count', len(report.get('published_checkpoints', [])), len(updates))
    audit.equal('storage/report_storage_count', len(report.get('storage_publications', [])), len(updates))
    local = report.get('local_checkpoints', [])
    audit.equal('storage/complete_local_publication_order', [r['optimizer_update'] for r in local], updates)
    audit.equal('storage/destination_membership', set(journal['destinations']), {str(n) for n in updates})
    audit.equal('storage/receipt_membership', set(receipts),
                {f'checkpoint-publications/update-{n:06d}.json' for n in updates})
    keep = owned['keep_local_completed']
    expected_prunes = []
    for i, entry in enumerate(published):
        number = entry['update']
        prefix = 'storage/update'+str(number)
        directory = str(root/f'update-{number:06d}')
        relative = f'checkpoint-publications/update-{number:06d}.json'
        audit.equal(prefix+'/entry_fields', set(entry),
                    {'update', 'directory', 'receipt_path', 'receipt_sha256', 'local_status'})
        audit.equal(prefix+'/directory', entry['directory'], directory)
        audit.equal(prefix+'/registered', journal['destinations'][str(number)], directory)
        audit.require(prefix+'/resume_source_not_owned', directory != owned['resume_source'])
        audit.equal(prefix+'/relative_receipt', entry['receipt_path'], relative)
        receipt = receipts[relative]
        manifest, objects, found_identity = publication_metadata(receipt)
        audit.equal(prefix+'/identity', found_identity, identity)
        audit.equal(prefix+'/configuration', manifest['metadata']['configuration'], report['configuration'])
        audit.equal(prefix+'/source_fingerprint', manifest['metadata']['source_fingerprint'], report['source_fingerprint'])
        audit.equal(prefix+'/counters', manifest['counters'], expected_counts(payload, number))
        audit.equal(prefix+'/cursors', manifest['rank_cursors'], [expected_cursor(payload, number, rank) for rank in range(2)])
        audit.equal(prefix+'/receipt_directory', receipt['directory'], directory)
        for filename, obj in objects.items():
            audit.equal(prefix+'/object_uri/'+filename, obj['uri'],
                        owned['storage_prefix']+f'/update-{number:06d}/'+filename)
        audit.equal(prefix+'/reported_publication', report['published_checkpoints'][i], receipt)
        audit.equal(prefix+'/reported_local_receipt', local[i]['receipt'],
                    {key: value for key, value in receipt.items() if key != 'retention'})
        audit.equal(prefix+'/committed_live_boundary', local[i]['boundary_by_rank'], legacy.boundary_at(report, number))
        retained = number in updates[-keep:]
        audit.equal(prefix+'/final_local_status', entry['local_status'], 'retained' if retained else 'pruned')
        observation = report['storage_publications'][i]
        audit.equal(prefix+'/observation_fields', set(observation), {'status', 'update', 'receipt_path',
            'receipt_sha256', 'journal_path', 'journal_sha256', 'kept_updates', 'pruned_updates'})
        audit.equal(prefix+'/successful_publication', observation['status'], 'published_and_local_retention_applied')
        audit.equal(prefix+'/observation_update', observation['update'], number)
        audit.equal(prefix+'/observation_receipt', observation['receipt_path'], str(persistent/relative))
        audit.equal(prefix+'/observation_receipt_pin', observation['receipt_sha256'], entry['receipt_sha256'])
        audit.require(prefix+'/receipt_pin', re.fullmatch('[0-9a-f]{64}', entry['receipt_sha256']) is not None)
        audit.equal(prefix+'/observation_journal', observation['journal_path'], str(persistent/'ssd-journal.json'))
        audit.require(prefix+'/journal_pin', re.fullmatch('[0-9a-f]{64}', observation['journal_sha256']) is not None)
        audit.equal(prefix+'/kept_at_publication', observation['kept_updates'], updates[max(0, i+1-keep):i+1])
        audit.equal(prefix+'/pruned_at_publication', observation['pruned_updates'], updates[:max(0, i+1-keep)])
        if not retained:
            expected_prunes.append({'update': number, 'directory': directory,
                'receipt_sha256': entry['receipt_sha256'], 'replaced_by_update': updates[i+keep], 'status': 'completed'})
    audit.equal('storage/exact_prune_history', journal['prune_operations'], expected_prunes)
    if updates:
        audit.equal('storage/latest', evidence['latest'], report['published_checkpoints'][-1])
        audit.equal('storage/final_journal_pin', report['storage_publications'][-1]['journal_sha256'], evidence['journal_sha256'])
        audit.equal('storage/final_boundary_published', updates[-1], report['final_counters']['optimizer_updates'])
    else:
        audit.equal('storage/no_latest', evidence['latest'], None)
        audit.require('storage/no_new_checkpoint_only_on_resume', 'resume' in report
                      and report['resume']['completed_update'] == report['final_counters']['optimizer_updates'])


if __name__ == '__main__':
    raise SystemExit(main())
