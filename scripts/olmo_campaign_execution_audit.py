#!/usr/bin/env python3
"""Independent JSON-only execution/recovery audit; never imports training code.

Timing, W&B identifiers, memory samples and storage paths are operational fields.
All recorded scientific/update evidence is compared without tolerances or masks.
This verifies evidence consistency, not checkpoint tensor bytes or BF16 fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

SCHEMA = 'olmo-campaign-execution-audit-v1'
REPORT_SCHEMA = 'olmo-campaign-execute-report-v1'
OBSERVATION_SCHEMA = 'olmo-campaign-execution-observation-v1'
COUNTERS = ('optimizer_updates', 'input_tokens', 'documents', 'microbatches',
            'ce_positions', 'latent_pairs', 'kl_triples')
TERMS = ('ce', 'latent', 'kl')
HEAVY = ('input', 'raw_gradients', 'boundary')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []

    def require(self, name, condition):
        self.checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    def equal(self, name, actual, expected):
        self.require(name, actual == expected)

    def result(self, failure=None):
        return {'schema': SCHEMA, 'passed': failure is None, 'checks': self.checks,
                'failures': [] if failure is None else [str(failure)],
                'scope': 'Exact JSON evidence consistency; no tensor reload, numerical equivalence or throughput claim'}


def finite_json(value):
    if isinstance(value, dict):
        return all(type(key) is str and finite_json(item) for key, item in value.items())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return value is None or type(value) in (str, bool, int) or (type(value) is float and math.isfinite(value))


def expected_counts(payload, completed):
    rows = payload['plan']['updates']
    if type(completed) is not int or not 0 <= completed <= len(rows):
        raise ValueError('Completed update outside finite plan')
    result = dict.fromkeys(COUNTERS, 0)
    for row in rows[:completed]:
        counts = row['counts']
        result['optimizer_updates'] += 1
        result['input_tokens'] += counts['valid_tokens']
        result['documents'] += counts['packed_rows']
        result['microbatches'] += sum(rank['microbatches'] for rank in row['allocation_by_arm'][payload['arm']])
        result['ce_positions'] += counts['ce_targets']
        if 'N' in payload['arm']:
            result['latent_pairs'] += counts['latent_pairs']
            result['kl_triples'] += counts['kl_triples']
    return result


def expected_cursor(payload, completed, rank):
    plan = payload['plan']
    cursor = plan['first_cursor'] if completed == 0 else plan['updates'][completed-1]['next_cursor']
    return {'schema': payload['cursor_schema'], 'rank': rank,
            'world_size': payload['partition']['world_size'],
            'physical_batch_per_rank': payload['partition']['physical_batch_per_rank'], 'cursor': cursor}


def boundary_check(audit, name, rows, payload, completed):
    world = payload['partition']['world_size']
    audit.require(name+'/rank_count', isinstance(rows, list) and len(rows) == world)
    for rank, row in enumerate(rows):
        label = f'{name}/rank{rank}'
        audit.require(label+'/complete', isinstance(row, dict) and set(row) == {'state', 'cursor', 'rng'})
        state = row['state']
        audit.require(label+'/state_complete', isinstance(state, dict) and set(state) == {'model', 'optimizer', 'scheduler', 'counters'}
                      and all(isinstance(state[key], dict) and state[key] for key in ('model', 'optimizer', 'scheduler')))
        audit.equal(label+'/scheduler_epoch', state['scheduler']['last_epoch'], completed)
        audit.equal(label+'/counters', state['counters'], expected_counts(payload, completed))
        audit.require(label+'/integer_counters', all(type(v) is int for v in state['counters'].values()))
        audit.equal(label+'/cursor', row['cursor'], expected_cursor(payload, completed, rank))
        audit.require(label+'/rng_complete', isinstance(row['rng'], dict) and
                      {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'generators', 'generator_devices', 'device'} <= row['rng'].keys())
        audit.equal(label+'/replica_state', state, rows[0]['state'])


def _audit_report(audit, report, label, source_root=None):
    check = lambda name, condition: audit.require(label+'/'+name, condition)
    equal = lambda name, actual, expected: audit.equal(label+'/'+name, actual, expected)
    check('finite_json', finite_json(report))
    equal('schema', report['schema'], REPORT_SCHEMA)
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


def audit_report(report, *, source_root=None):
    audit = Audit()
    try:
        _audit_report(audit, report, 'report', source_root)
    except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        return audit.result(f'{type(exc).__name__}: {exc}')
    return audit.result()


def boundary_at(report, completed):
    start = report.get('resume', {}).get('completed_update', 0)
    if completed == start:
        return report['origin_boundary_by_rank']
    if completed == report['final_counters']['optimizer_updates']:
        return report['final_boundary_by_rank']
    rows = report.get('updates', {}).get(str(completed), [])
    if rows and all('boundary' in row for row in rows):
        return [row['boundary'] for row in rows]
    for checkpoint in report.get('local_checkpoints', []):
        if checkpoint['optimizer_update'] == completed:
            return checkpoint['boundary_by_rank']
    raise ValueError(f'Reference lacks complete boundary evidence at update {completed}')


def compare(reference, actual, *, reference_source_root=None, actual_source_root=None):
    audit = Audit()
    try:
        _audit_report(audit, reference, 'reference', reference_source_root)
        _audit_report(audit, actual, 'actual', actual_source_root)
        for key in ('configuration','sources','source_fingerprint','arm','scale','declaration_sha256','resolved_sha256'):
            audit.equal('pair/'+key, actual[key], reference[key])
        start = actual.get('resume', {}).get('completed_update', 0)
        final = actual['final_counters']['optimizer_updates']
        audit.equal('pair/origin_boundary', actual['origin_boundary_by_rank'], boundary_at(reference, start))
        audit.equal('pair/final_boundary', actual['final_boundary_by_rank'], boundary_at(reference, final))
        for step, rows in actual.get('updates', {}).items():
            audit.require('pair/update'+step+'/reference_coverage', step in reference.get('updates', {}))
            audit.equal('pair/update'+step+'/rank_data',actual['observations'][step]['rank_data'],reference['observations'][step]['rank_data'])
            for rank, row in enumerate(rows):
                expected = reference['updates'][step][rank]
                for key in ('metrics','cursor','loss_means','clipping'):
                    audit.equal(f'pair/update{step}/rank{rank}/{key}', row[key], expected[key])
                if row['mode'] == expected['mode'] == 'acceptance':
                    for key in HEAVY:
                        audit.equal(f'pair/update{step}/rank{rank}/{key}', row[key], expected[key])
    except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        return audit.result(f'{type(exc).__name__}: {exc}')
    return audit.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--actual-sha256', required=True)
    parser.add_argument('--reference-source-root', type=Path)
    parser.add_argument('--actual-source-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    pins = {}
    for name in ('reference','actual'):
        path = getattr(args,name); expected = getattr(args,name+'_sha256')
        if file_sha(path) != expected:
            raise ValueError(name+' report bytes differ from explicit pin')
        pins[name] = {'path':str(path),'sha256':expected,'size_bytes':path.stat().st_size}
    reference, actual = (json.loads(getattr(args,name).read_text()) for name in ('reference','actual'))
    result = compare(reference,actual,reference_source_root=args.reference_source_root,actual_source_root=args.actual_source_root)
    result['inputs'] = pins
    # The payload was parsed after the first hash. Revalidate at publication so
    # a concurrent writer cannot attach this result to different pinned bytes.
    for name in ('reference','actual'):
        if file_sha(getattr(args,name)) != pins[name]['sha256']:
            raise ValueError(name+' report changed during JSON audit')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    snapshot = args.output_dir/'source-snapshot'; snapshot.mkdir()
    for path in (Path(__file__),Path(__file__).resolve().parents[1]/'tests/test_campaign_execution_audit.py'):
        shutil.copyfile(path,snapshot/path.name)
    result['audit_sources'] = {p.name:file_sha(p) for p in snapshot.iterdir()}
    with tempfile.NamedTemporaryFile(mode='w',dir=args.output_dir,delete=False) as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary,args.output_dir/'report.json')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
