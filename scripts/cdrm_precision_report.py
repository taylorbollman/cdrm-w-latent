#!/usr/bin/env python3
"""Compact, provenance-checked numerical diagnostics; never grants clearance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

from experiment_tracking import OnlineTracker, add_wandb_arguments

PROJECT = Path(__file__).resolve().parents[1]
SCHEMAS = {'coarse': 'cdrm-fixed-state-precision-probe-v1',
           'attention': 'cdrm-same-operand-attention-v1',
           'adam': 'cdrm-precision-adam-reference-v1',
           'embedding': 'cdrm-embedding-accumulation-reference-v1',
           'confirmation': 'cdrm-precision-confirmation-v1'}
ORIGINAL_SCREEN_TERM = 2 ** -6
PENDING = ['causal_localization', 'candidate_selection', 'fresh_confirmation']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def resolve_path(name, directory):
    path = Path(name)
    if str(path).startswith('/workspace/cdrm-w-latent/'):
        path = PROJECT / path.relative_to('/workspace/cdrm-w-latent')
    elif not path.is_absolute():
        path = PROJECT / path
    if not path.is_file():
        path = Path(directory) / Path(name).name
    if not path.is_file() or not path.resolve().is_relative_to(PROJECT.resolve()):
        raise ValueError(f'Missing or out-of-project input: {name}')
    return path.resolve()


class Verifier:
    def __init__(self):
        self.files = {}

    def check(self, path, expected=None):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing regular input: {path}')
        path = path.resolve()
        actual = self.files.setdefault(str(path), digest(path)) if str(path) not in self.files else self.files[str(path)]
        if expected is not None and actual != expected:
            raise ValueError(f'Input SHA256 mismatch: {path}')
        return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}

    def record(self, record, directory):
        return self.check(resolve_path(record['path'], directory), record['sha256'])

    def recheck(self):
        for name, expected in self.files.items():
            if digest(name) != expected:
                raise ValueError(f'Input changed during summary: {name}')


def snapshot_audit(doc, path, verifier):
    sources = doc.get('source_sha256', {})
    if not sources:
        raise ValueError('Missing immutable source snapshot identity')
    for index, (name, expected) in enumerate(sources.items()):
        relative = Path(name)
        candidates = []
        if not relative.is_absolute() and '..' not in relative.parts:
            candidates.append(path.parent / 'source' / relative)
        # Adam oracle also retains external installed-Torch sources by index.
        candidates.append(path.parent / 'source' / f'{index:02d}-{relative.name}')
        saved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if saved is None and doc.get('schema') == SCHEMAS['embedding']:
            saved = resolve_path(name, path.parent)
        if saved is None:
            raise ValueError(f'Missing source snapshot: {name}')
        verifier.check(saved, expected)


def compact_metric(row):
    keep = {'status', 'finite', 'reference_present', 'actual_present', 'presence_match', 'shape_match',
            'ref_shape', 'actual_shape', 'ref_dtype', 'actual_dtype', 'ref_l2', 'error_l2',
            'rel_l2', 'max_abs_error', 'maximum_over_reference_maximum', 'mean_signed_error',
            'atol', 'rtol', 'norm_floor', 'elementwise_failure_count', 'elementwise_failure_fraction'}
    return {key: value for key, value in row.items()
            if key in keep or key.endswith('_pass') or key.startswith('fp32_floor_')}


def compact_comparison(value):
    """Retain flags and floors, excluding coordinate lists and tensor dumps."""
    result = {}
    for key, item in value.items():
        if key == 'rows':
            result[key] = {name: compact_metric(row) for name, row in item.items()}
        elif key in {'actual_ce_gradients', 'independent_side_gradients', 'adam'}:
            result[key] = compact_comparison(item)
        elif key in {'logits', 'hat_m', 'candidate', 'bridge_unscaled'}:
            result[key] = compact_metric(item)
        elif key in {'unscaled_side_states', 'actual_forward_unscaled_side_states'}:
            result[key] = {name: compact_metric(row) for name, row in item.items()}
        elif not isinstance(item, dict):
            result[key] = item
        elif key in {'clipping', 'gradient_near_zero_buckets'}:
            result[key] = item
    return result


def compiler_errors(audit, *, require_graphs=True):
    if not isinstance(audit, dict) or audit.get('required') is not True:
        return ['Missing required compiled-path audit']
    counters = audit.get('counters', {})
    errors = []
    if audit.get('fail_on_recompile_limit_hit') is not True:
        errors.append('Compiler fallback guard was not enabled')
    if any(v for group in ('unimplemented', 'graph_break') for v in counters.get(group, {}).values()):
        errors.append('Compiler audit contains unsupported/fallback paths')
    if require_graphs and counters.get('stats', {}).get('unique_graphs', 0) < 1:
        errors.append('Compiler audit recorded no compiled graphs')
    return errors


def coarse_errors(doc):
    if doc.get('compiler_by_arm'):
        audits = doc['compiler_by_arm']
        controls={row['name']:row for row in doc['controls']}
        errors=[]
        for name,audit in audits.items():
            expected_graphs=not controls.get(name.split('/',1)[-1],{}).get('bypass',False)
            errors.extend(f'{name}: {error}' for error in compiler_errors(audit,require_graphs=expected_graphs))
            if audit.get('require_graphs',True) != expected_graphs:
                errors.append(f'{name}: Compiled-graph requirement differs from declared bypass control')
        aggregate = doc.get('compiler', {})
        if aggregate.get('all_arms_pass') is not True or aggregate.get('failed_arms') or aggregate.get('audited_arms') != len(audits):
            errors.append('Per-arm compiler summary is inconsistent')
        extra=set(doc.get('arguments',{}).get('verify_observer_arm') or [])-{'current_fp32','current_bf16'}
        if not extra.issubset(controls):errors.append('Undeclared extra observer control')
        expected_arms = {'unobserved/current_fp32', 'unobserved/current_bf16'} | {f'unobserved/{name}' for name in extra} | {f'observed/{row["name"]}' for row in doc['controls']}
        if set(audits) != expected_arms:
            errors.append('Per-arm compiler coverage differs from requested controls')
        total_graphs = sum(a['counters'].get('stats', {}).get('unique_graphs', 0) for a in audits.values())
        if aggregate.get('counters', {}).get('stats', {}).get('unique_graphs') != total_graphs:
            errors.append('Aggregated compiled graph count differs from per-arm audits')
    else:
        errors = compiler_errors(doc.get('compiler'))
    for group in ('baseline_replay', 'observation_invariance'):
        names={'current_fp32','current_bf16'}
        if group=='observation_invariance':names.update(doc.get('arguments',{}).get('verify_observer_arm') or [])
        for arm in names:
            if doc.get(group, {}).get(arm, {}).get('bitwise_equal') is not True:
                errors.append(f'Missing exact {group}/{arm}')
    for name, row in doc.get('bypass_parity', {}).items():
        if row.get('bitwise_shared_computation_and_normalized_step_equal') is not True:
            errors.append(f'Bypass semantic parity failed: {name}')
    return errors


def brief_context(checkpoint, fixture):
    return {'checkpoint_sha256': checkpoint.get('sha256'),
            'completed_updates': checkpoint.get('completed_updates'),
            'fixture_sha256': fixture.get('sha256'), 'fixture_shape': fixture.get('shape'),
            'example_offset': fixture.get('example_offset'), 'native_targets': fixture.get('native_targets')}


def original_case(case_dir, verifier, originals):
    directory = resolve_path(str(Path(case_dir) / 'report.json'), PROJECT).parent
    ref = verifier.check(directory / 'report.json')
    if ref['sha256'] in originals:
        return originals[ref['sha256']]
    doc = json.loads((directory / 'report.json').read_text())
    verifier.record(doc['checkpoint'], directory)
    verifier.record(doc['tensor_artifact'], directory)
    record = {'artifact': ref, 'context': brief_context(doc['checkpoint'], doc['fixture']),
              'machine_screens_pass': doc['machine_screens_pass'], 'criteria': doc['criteria'],
              'checkpoint': doc['checkpoint'], 'tensor_artifact': doc['tensor_artifact'],
              'comparisons': {name: compact_comparison(row) for name, row in doc['comparisons'].items()}}
    originals[ref['sha256']] = record
    return record


def load_attempt(kind, path, verifier, originals):
    path = Path(path).resolve()
    artifact = verifier.check(path)
    doc = json.loads(path.read_text())
    if doc.get('schema') != SCHEMAS[kind]:
        raise ValueError(f'Unexpected {kind} report schema: {path}')
    snapshot_audit(doc, path, verifier)
    contract = doc.get('reference_contract', doc.get('contract'))
    if contract:
        verifier.record(contract, path.parent)
    for key in ('checkpoint', 'tensor_artifact'):
        if doc.get(key):
            verifier.record(doc[key], path.parent)
    if doc.get('source_case'):
        original = original_case(doc['source_case']['path'], verifier, originals)
        if original['artifact']['sha256'] != doc['source_case']['report_sha256']:
            raise ValueError('Coarse source-case report identity differs')
        if (original['tensor_artifact']['sha256'] != doc['source_case']['tensor_sha256']
                or original['checkpoint']['sha256'] != doc['checkpoint']['sha256']
                or original['context'] != brief_context(doc['checkpoint'],doc['source_case']['fixture'])):
            raise ValueError('Coarse checkpoint/tensor/fixture role identity differs')
    if doc.get('probe'):
        probe_path = resolve_path(str(Path(doc['probe']['path']) / 'report.json'), path.parent)
        verifier.check(probe_path, doc['probe']['report_sha256'])
        verifier.check(probe_path.parent / 'tensors.pt', doc['probe']['tensors_sha256'])
        if json.loads(probe_path.read_text()).get('status') != 'diagnostics_complete':
            raise ValueError('Local reference depends on an unsuccessful probe')
    if kind == 'confirmation':
        confirmation_identity_audit(doc,path,verifier)
    if kind in {'adam', 'embedding'}:
        for index, case in enumerate(doc.get('cases', [])):
            original = original_case(case['source_case'], verifier, originals)
            if original['artifact']['sha256'] != case['report_sha256']:
                raise ValueError('CPU reference source-case report identity differs')
            verifier.record(case['checkpoint'],path.parent)
            if (original['checkpoint']['sha256'] != case['checkpoint']['sha256']
                    or original['tensor_artifact']['sha256'] != case['tensor_sha256']):
                raise ValueError('CPU reference checkpoint/tensor role identity differs')
            if case.get('fixture') and original['context'] != brief_context(case['checkpoint'],case['fixture']):
                raise ValueError('CPU reference fixture identity differs')
            if kind == 'adam':
                verifier.record(case['reference_tensors'], path.parent)
                verifier.check(path.parent / f'case-{index}' / 'report.json')
                saved = json.loads((path.parent / f'case-{index}' / 'report.json').read_text())
                if saved != case:
                    raise ValueError('Embedded Adam case and retained case report differ')
    reasons = []
    complete = doc.get('status') == 'diagnostics_complete'
    if not complete:
        reasons.append(f"Producer status {doc.get('status')}; {doc.get('error_type', 'incomplete execution')}")
        if doc.get('error'):
            reasons.append(str(doc['error'])[:800])
    if complete and kind == 'coarse':
        reasons.extend(coarse_errors(doc))
    if complete and kind == 'confirmation':
        reasons.extend(confirmation_execution_errors(doc))
    if complete and kind != 'embedding' and doc.get('wandb', {}).get('status') != 'synced':
        reasons.append('Required producer W&B synchronization did not complete')
    attempt = {'kind': kind, 'artifact': artifact, 'name': path.parent.name, 'status': doc.get('status'),
               'accepted_for_diagnostic_summary': not reasons, 'exclusion_reasons': reasons,
               'numerical_clearance': False, 'producer_numerical_clearance': doc.get('numerical_clearance'),
               'contract': contract, 'wandb': doc.get('wandb'),
               'compiler': doc.get('compiler'), 'compiler_by_arm': doc.get('compiler_by_arm'),
               'baseline_replay': doc.get('baseline_replay'), 'observation_invariance': doc.get('observation_invariance')}
    return attempt, doc


def build_summary(inputs, verifier):
    report = {'schema': 'cdrm-precision-summary-v1', 'status': 'running',
              'scope': 'Numerical diagnosis only; completed execution is provenance acceptance, never numerical clearance.',
              'numerical_clearance': False, 'original_bf16_screen_term': ORIGINAL_SCREEN_TERM,
              'screen_note': '1.5625% is the original BF16 term. Saved global/per-tensor flags and FP32 floors remain authoritative.',
              'attempts': [], 'coarse': [], 'attention': [], 'adam': [], 'embedding': [], 'confirmation': [], 'original_cases': [],
              'pending_stages': list(PENDING), 'audit_errors': []}
    originals, loaded, seen = {}, [], set()
    for kind, path in inputs:
        attempt, doc = load_attempt(kind, path, verifier, originals)
        key=(kind,attempt['artifact']['sha256'])
        if key in seen:raise ValueError('Duplicate diagnostic report supplied')
        seen.add(key)
        report['attempts'].append(attempt); loaded.append((attempt, doc))
    accepted_probe_hashes = {a['artifact']['sha256'] for a, _ in loaded
                            if a['kind'] == 'coarse' and a['accepted_for_diagnostic_summary']}
    for attempt, doc in loaded:
        eligible = attempt['accepted_for_diagnostic_summary']
        base = {'attempt': attempt['name'], 'input_report_sha256': attempt['artifact']['sha256'],
                'accepted_for_diagnostic_summary': eligible}
        if attempt['kind'] == 'coarse':
            for arm, comparison in doc.get('comparisons', {}).items():
                row = {**base, 'arm': arm, 'reference': comparison['reference'],
                       'context': brief_context(doc['checkpoint'], doc['source_case']['fixture']),
                       'gradient_relative_l2': comparison['actual_ce_gradients']['global_parameter_relative_l2'],
                       'gradient_screen_pass': comparison['actual_ce_gradients']['pass'],
                       'adam_relative_l2': comparison['adam']['global_delta_relative_l2'],
                       'adam_screen_pass': comparison['adam']['guardrail_pass'],
                       'metrics': compact_comparison(comparison)}
                report['coarse'].append(row)
        elif attempt['kind'] == 'attention':
            if eligible and doc['probe']['report_sha256'] not in accepted_probe_hashes:
                attempt['accepted_for_diagnostic_summary'] = False
                attempt['exclusion_reasons'].append('Matching accepted coarse probe must be supplied explicitly')
                base['accepted_for_diagnostic_summary'] = False
            for name, obs in doc.get('observations', {}).items():
                arm, block = name.rsplit('/block', 1)
                for tensor in ('output', 'q_gradient', 'k_gradient', 'v_gradient'):
                    report['attention'].append({**base, 'case': name, 'arm': arm, 'block': int(block), 'tensor': tensor,
                        'native': obs['native_against_analytic'][tensor],
                        'local_fp32': obs['local_fp32_against_analytic'][tensor],
                        'fp32_bias_intervention': obs['fp32_bias_intervention_same_qkv_dy'][tensor],
                        'native_local_interval_pass': obs['native_local_interval_pass'],
                        'reference_floor_supported': obs['reference_floor_supported']})
        elif attempt['kind'] == 'confirmation':
            report['confirmation'].append({**base,'role':doc['role'],'candidate':doc['candidate'],
                'context':brief_context(doc['checkpoint'],doc['fixture']),'decision':doc['decision'],'fixtures':doc['fixtures'],
                'compiler':doc.get('compiler'),
                'comparisons':{key:compact_comparison(value) for key,value in doc.get('comparisons',{}).items()},
                **{key:doc.get(key) for key in ('machine_screens_pass','candidate_machine_screens_pass',
                    'candidate_additional_diagnostics_pass','candidate_prospective_checks_pass',
                    'side_scaling','per_arm_side_scaling_pass','side_scaling_pass',
                    'full_output_scaling','full_output_scaling_pass','disposition')}})
        elif attempt['kind'] == 'embedding':
            for case in doc.get('cases', []):
                report['embedding'].append({**base, 'case': Path(case['source_case']).name,
                    'checkpoint': case['checkpoint'],
                    'same_operand_checks': {arm: row['native_vs_same_operand_fp64']['global'] for arm,row in case['arms'].items()},
                    'incoming_adjoint_policy_difference': case['incoming_adjoint_policy_difference']['global'],
                    'fp64_accumulated_policy_difference': case['fp64_accumulated_policy_difference']['global'],
                    'native_policy_difference': case['native_policy_difference']['global'],
                    'native_vs_predicted': case['native_policy_error_vs_predicted_incoming_adjoint_error']['global'],
                    'scope': 'Descriptive CPU same-operand scatter accumulation; no new threshold or numerical clearance.'})
        else:
            for index, case in enumerate(doc.get('cases', [])):
                local = {}
                for arm, checks in case['local_optimizer_checks'].items():
                    local[arm] = {key: value for key, value in checks.items()
                                  if key not in {'rows', 'native_delta_vs_ideal_increment', 'native_delta_vs_fp32_write_realized', 'fp32_write_effect'}}
                    for key in ('native_delta_vs_ideal_increment', 'native_delta_vs_fp32_write_realized', 'fp32_write_effect'):
                        local[arm][key] = checks[key]['global']
                report['adam'].append({**base, 'case': Path(case['source_case']).name, 'case_index': index,
                    'context': brief_context(case['checkpoint'], case['fixture']),
                    'original_machine_screens_pass': case['original_machine_screens_pass'],
                    'local_optimizer_checks_pass': case['local_optimizer_checks_pass'], 'local_checks': local,
                    'original_adam_summary_reproduced': case['original_adam_summary_reproduced'],
                    'gradient_comparisons': {key: row['global'] for key, row in case['gradient_comparisons'].items()},
                    'policy_comparisons': {key: {k: v for k, v in row.items() if k != 'rows'}
                                           for key, row in case['adam_policy_comparisons'].items()},
                    'common_clipping_coefficient': case['common_clipping_coefficient'],
                    'common_clip_scope': 'Counterfactual diagnostic only; original optimizer policy remains unchanged.'})
    report['original_cases'] = list(originals.values())
    accepted_local=[r for r in report['attention'] if r['accepted_for_diagnostic_summary']]
    report['attention_interval_flags']={group: {field: sum(r[field].get('interval_failures',0) for r in accepted_local
        if (r['tensor']=='output') == (group=='output')) for field in ('native','local_fp32')}
        for group in ('output','qkv_gradients')}
    report['attention_interval_flags']['all_native_tensor_intervals_pass'] = bool(accepted_local) and all(r['native']['interval_pass'] for r in accepted_local)
    report['attention_interval_flags']['all_local_fp32_tensor_intervals_pass'] = bool(accepted_local) and all(r['local_fp32']['interval_pass'] for r in accepted_local)
    report['accepted_attempts'] = sum(a['accepted_for_diagnostic_summary'] for a in report['attempts'])
    report['excluded_attempts'] = len(report['attempts']) - report['accepted_attempts']
    report['audit_pass'] = True
    return report


def write_csvs(output, report):
    tables = {
        'coarse.csv': [{k: row[k] for k in ('attempt','arm','reference','accepted_for_diagnostic_summary','gradient_relative_l2','gradient_screen_pass','adam_relative_l2','adam_screen_pass')} for row in report['coarse']],
        'attention.csv': [{'attempt':row['attempt'],'arm':row['arm'],'block':row['block'],'tensor':row['tensor'],
            'accepted_for_diagnostic_summary':row['accepted_for_diagnostic_summary'],
            'native_relative_l2':row['native'].get('relative_l2'),'native_interval_pass':row['native']['interval_pass'],
            'fp32_relative_l2':row['local_fp32'].get('relative_l2'),'fp32_interval_pass':row['local_fp32']['interval_pass']} for row in report['attention']],
        'adam.csv': [{'attempt':row['attempt'],'case':row['case'],'policy':policy,
            'accepted_for_diagnostic_summary':row['accepted_for_diagnostic_summary'],
            'relative_l2':value['global']['relative_l2'],
            'original_screen_pass_if_applied':value['original_trained_adam_screen_pass_if_applied'],
            'local_optimizer_checks_pass':row['local_optimizer_checks_pass']}
            for row in report['adam'] for policy,value in row['policy_comparisons'].items()]}
    tables['embedding.csv'] = [{'attempt':row['attempt'],'case':row['case'],
        'accepted_for_diagnostic_summary':row['accepted_for_diagnostic_summary'],
        'incoming_adjoint_relative_l2':row['incoming_adjoint_policy_difference']['relative_l2'],
        'native_embedding_relative_l2':row['native_policy_difference']['relative_l2'],
        'fp64_predicted_relative_l2':row['fp64_accumulated_policy_difference']['relative_l2']} for row in report['embedding']]
    tables['confirmation.csv']=[{'role':row['role'],'comparison':name,
        'accepted_for_diagnostic_summary':row['accepted_for_diagnostic_summary'],
        'gradient_relative_l2':value['actual_ce_gradients']['global_parameter_relative_l2'],
        'gradient_screen_pass':value['actual_ce_gradients']['pass'],
        'side_relative_l2':value['independent_side_gradients']['global_parameter_relative_l2'],
        'adam_relative_l2':value['adam']['global_delta_relative_l2'],
        'adam_cosine':value['adam']['global_delta_cosine'],
        'adam_guardrail':value['adam']['guardrail'],'adam_guardrail_pass':value['adam']['guardrail_pass'],
        'comparison_machine_screens_pass':value['machine_screens_pass'],
        'candidate_prospective_checks_pass':row['candidate_prospective_checks_pass'],
        'all_original_machine_screens_pass':row['machine_screens_pass']}
        for row in report['confirmation'] for name,value in row['comparisons'].items()]
    for name, rows in tables.items():
        with (output/name).open('w', newline='') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_plot(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    coarse = [r for r in report['coarse'] if r['accepted_for_diagnostic_summary']]
    for axis, metric, flag, title in ((axes[0,0],'gradient_relative_l2','gradient_screen_pass','Full CE parameter-gradient error'),
                                      (axes[0,1],'adam_relative_l2','adam_screen_pass','Saved-state Adam-update error')):
        if coarse:
            axis.barh(range(len(coarse)),[100*r[metric] for r in coarse],
                      color=['#20887c' if r[flag] else '#c95442' for r in coarse])
            axis.set_yticks(range(len(coarse)),[short_label(r) for r in coarse],fontsize=8)
            axis.invert_yaxis()
        else:
            axis.text(.5,.5,'Awaiting completed, audited coarse controls',ha='center',transform=axis.transAxes)
        axis.axvline(100*ORIGINAL_SCREEN_TERM,color='#7b2636',linestyle='--',label='Original BF16 term: 1.5625%')
        axis.set_xlabel('Relative L2 error (%)');axis.set_title(title);axis.legend(fontsize=8)
    axis=axes[1,0]
    local=[r for r in report['attention'] if r['accepted_for_diagnostic_summary'] and r['tensor']!='output']
    groups={}
    for row in local:
        groups.setdefault((row['attempt'],row['arm']),{}).setdefault(row['block'],[]).append(row)
    for (attempt,arm), blocks in groups.items():
        xs=sorted(blocks)
        for field,style in [('native','-o'),('local_fp32','--x')]:
            ys=[max((r[field].get('relative_l2') or 0.) for r in blocks[x]) for x in xs]
            axis.semilogy(xs,[max(y*100,1e-12) for y in ys],style,label=f'{attempt}/{arm}: {field}',markersize=4)
    if not groups:axis.text(.5,.5,'Local attention references pending',ha='center',transform=axis.transAxes)
    else:axis.legend(fontsize=7)
    axis.set_xlabel('Ordinary block');axis.set_ylabel('Max q/k/v relative L2 (%)')
    native_fails=sum(not r['native']['interval_pass'] for r in local)
    fp32_fails=sum(not r['local_fp32']['interval_pass'] for r in local)
    output_flags=report['attention_interval_flags']['output']
    axis.set_title(f'QKV tensors failing intervals: native={native_fails}, FP32={fp32_fails}\n'
        f'Output coordinate flags: native={output_flags["native"]}, FP32={output_flags["local_fp32"]}',fontsize=10)
    axis=axes[1,1]
    adams=[r for r in report['adam'] if r['accepted_for_diagnostic_summary']]
    policies=['native','fp64_own_clip','fp64_common_clip','fp32_write_own_clip']
    for index,row in enumerate(adams):
        xs=[x+index*.8/max(len(adams),1) for x in range(len(policies))]
        axis.bar(xs,[100*row['policy_comparisons'][p]['global']['relative_l2'] for p in policies],
                 width=.8/max(len(adams),1),label=row['case'])
    if adams:
        axis.set_xticks([x+.4-.4/len(adams) for x in range(4)],['Native','FP64 own clip','FP64 common clip*','FP32 weight write'],rotation=12)
        axis.legend(fontsize=8)
    else:axis.text(.5,.5,'CPU Adam references pending',ha='center',transform=axis.transAxes)
    axis.axhline(100*ORIGINAL_SCREEN_TERM,color='#7b2636',linestyle='--')
    axis.set_ylabel('BF16-vs-FP32 update error (%)')
    axis.set_title('Optimizer arithmetic / input sensitivity (*counterfactual)')
    failed=', '.join(a['name'] for a in report['attempts'] if not a['accepted_for_diagnostic_summary']) or 'none'
    fig.suptitle('CDRM numerical resolution — scoped evidence; clearance requires reviewed disposition',fontsize=15)
    excluded_arms=', '.join(r['arm']+' (ineffective)' for r in report.get('arm_exclusions',[])) or 'none'
    fig.text(.02,.018,f'Excluded attempts ({report["excluded_attempts"]}): {failed}; excluded interventions: {excluded_arms}\n'
             'Saved flags/floors remain authoritative. Pending stages: '+(', '.join(report['pending_stages']) or 'none (explicit reviewed completion)'),
             fontsize=9,color='#972f32')
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(output/'summary.png',dpi=160);fig.savefig(output/'summary.svg');plt.close(fig)


def confirmation_execution_errors(doc):
    names={'naive_fp32','tiled_fp32','current_bf16','candidate'}
    if set(doc.get('compiler',{}))!=names:return ['Missing per-arm confirmation compiler audits']
    errors=[]
    for name,audit in doc['compiler'].items():
        expected=name!='naive_fp32'
        errors.extend(f'{name}: {error}' for error in compiler_errors(audit,require_graphs=expected))
        if audit.get('require_graphs')!=expected or audit.get('validation_status')!='passed':
            errors.append(f'{name}: Incomplete or incompatible required compiler audit')
    if doc.get('status')=='diagnostics_complete':
        comparisons={'tiled_fp32_vs_naive_fp32','current_bf16_vs_tiled_fp32','candidate_vs_tiled_fp32'}
        if set(doc.get('comparisons',{}))!=comparisons:
            return errors+['Missing required fresh reference/original/candidate comparisons']
        if set(doc.get('per_arm_side_scaling_pass',{}))!=names or set(doc.get('full_output_scaling',{}))!=names:
            return errors+['Missing per-arm side/full-output scaling diagnostics']
        def scaling_pass(scales):
            if set(scales)!={'0.03125','32.0'} or any(not rows for rows in scales.values()):
                errors.append('Missing declared normalized scale leaves')
                return False
            return all(row['bitwise_normalized_equal'] is True for rows in scales.values() for row in rows.values())
        for name in names:
            if doc['per_arm_side_scaling_pass'][name]!=scaling_pass(doc.get('side_scaling',{}).get(name,{})):
                errors.append(f'{name}: Side-scaling summary differs from saved tensor leaves')
            full=doc['full_output_scaling'][name]
            full_pass=scaling_pass(full.get('scaling',{})) and full.get('fixed_forward_logits_bitwise_equal') is True
            if full.get('diagnostic_pass')!=full_pass:
                errors.append(f'{name}: Output-scaling summary differs from saved tensor leaves/logits')
        candidate=doc['comparisons']['candidate_vs_tiled_fp32']['machine_screens_pass']
        candidate = candidate and all(doc['per_arm_side_scaling_pass'][n] for n in ('candidate','tiled_fp32'))
        additional=all(doc['full_output_scaling'][n]['diagnostic_pass'] for n in ('candidate','tiled_fp32'))
        if (doc.get('candidate_machine_screens_pass')!=candidate
                or doc.get('candidate_additional_diagnostics_pass')!=additional
                or doc.get('candidate_prospective_checks_pass')!=(candidate and additional)):
            errors.append('Combined candidate flags do not match saved original/scaling flags')
        machine=all(row['machine_screens_pass'] for row in doc['comparisons'].values()) and all(doc['per_arm_side_scaling_pass'].values())
        if doc.get('machine_screens_pass')!=machine:
            errors.append('Combined original machine flag does not match saved comparison/scaling flags')
    return errors


def apply_completed_stages(report,stages):
    report['completed_stages']=list(dict.fromkeys(stages))
    report['pending_stages']=[stage for stage in PENDING if stage not in stages]
    if 'fresh_confirmation' in stages:
        valid=[row for row in report['confirmation'] if row['accepted_for_diagnostic_summary']]
        expected={'fp32_trained_u1000','bf16_trained_u1000','distinct_initialization'}
        if len(valid)!=3 or {row['role'] for row in valid}!=expected:
            raise ValueError('Completed fresh confirmation requires exactly three audited roles')
        if len({row['decision']['sha256'] for row in valid})!=1 or len({row['fixtures']['sha256'] for row in valid})!=1:
            raise ValueError('Fresh role reports do not share candidate/fixture anchors')


def confirmation_identity_audit(doc,path,verifier):
    decision_ref=verifier.record(doc['decision'],path.parent)
    fixture_ref=verifier.record(doc['fixtures'],path.parent)
    decision=json.loads(Path(decision_ref['path']).read_text())
    fixtures=json.loads(Path(fixture_ref['path']).read_text())
    role_ref=verifier.record(decision['roles'],path.parent)
    roles=json.loads(Path(role_ref['path']).read_text())
    if (decision.get('schema')!='cdrm-precision-candidate-freeze-v1' or decision.get('status')!='frozen_for_confirmation'
            or fixtures.get('schema')!='cdrm-precision-fresh-fixtures-v1' or fixtures.get('status')!='generated'
            or fixtures['candidate_freeze_sha256']!=decision_ref['sha256']
            or fixtures['roles_sha256']!=role_ref['sha256']):
        raise ValueError('Fresh confirmation candidate/fixture/role anchors differ')
    if doc['candidate']!=decision['candidate'] or doc['source_sha256']!=decision['source_sha256']:
        raise ValueError('Fresh confirmation candidate or frozen source map differs')
    for field,record in (('criteria',decision['original_criteria']),('reference_contract',decision['reference_contract'])):
        verifier.record(record,path.parent)
        if doc[field]['sha256']!=record['sha256']:raise ValueError('Fresh numerical criterion identity differs')
    expected={'fp32_trained_u1000':0,'bf16_trained_u1000':64,'distinct_initialization':128}
    allocated={row['name']:row for row in roles['roles']}
    actual={row['name']:row for row in fixtures['roles']}
    if set(actual)!=set(expected) or set(allocated)!=set(expected) or len(fixtures['roles'])!=3 or len(roles['roles'])!=3:
        raise ValueError('Fresh confirmation role set differs')
    corpus=fixtures['corpus']
    if corpus['seed']!=925903 or corpus['examples']!=192 or corpus['split']!='dev':
        raise ValueError('Fresh corpus dimensions/seed differ')
    from cdrm.mad_data import load_dataset
    data_root=resolve_path(str(Path(corpus['root'])/'selective-copying/dev.manifest.json'),path.parent).parent.parent
    manifest_path=data_root/'selective-copying/dev.manifest.json'
    verifier.check(manifest_path,corpus['manifest_sha256'])
    disk=json.loads(manifest_path.read_text())
    for row in disk['files'].values():verifier.check(manifest_path.parent/row['name'],row['sha256'])
    dataset=load_dataset(data_root,'selective-copying','dev',verify=True)
    if len(dataset)!=192 or dataset.sha256!=corpus['dataset_sha256']:raise ValueError('Fresh native arrays differ')
    for name,offset in expected.items():
        row=actual[name];declared=allocated[name]
        if row['example_offset']!=offset or row['example_stop']!=offset+64 or declared['example_offset']!=offset:
            raise ValueError('Fresh role offset differs')
        verifier.record(row['checkpoint'],path.parent)
        if declared.get('checkpoint_sha256') and row['checkpoint']['sha256']!=declared['checkpoint_sha256']:
            raise ValueError('Fresh trained checkpoint role differs')
        if dataset.take(slice(offset,offset+64)).sha256!=row['batch_sha256']:
            raise ValueError('Fresh role batch arrays differ')
    row=actual[doc['role']]
    if (doc['checkpoint']['sha256']!=row['checkpoint']['sha256'] or doc['fixture']['sha256']!=row['batch_sha256']
            or doc['fixture']['shape']!=[64,256] or doc['fixture']['native_targets']!=6144
            or doc['fixture']['example_offset']!=expected[doc['role']]
            or doc['fixture']['dataset_sha256']!=corpus['dataset_sha256']):
        raise ValueError('Observed fresh confirmation checkpoint/fixture role differs')


def short_label(row):
    role='S' if row['context'].get('example_offset')==64 else 'P'
    names={'current_bf16':'Current BF16','bf16_lambda_zero':'Lambda zero','bf16_bypass':'Bypass',
           'bf16_reduction_off':'Reduction off','bf16_side_fp32_backbone':'FP32 backbone',
           'bf16_attention_fp32':'FP32 attention','bf16_alibi_fp32':'FP32 ALiBi'}
    arm=names.get(row['arm'],row['arm'].replace('bf16_block_','Block ').replace('_fp32',' FP32'))
    return role+' / '+arm+' / '+row['attempt'].replace('primary-','').replace('secondary-','')


def make_confirmation_plot(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['fp32_trained_u1000','bf16_trained_u1000','distinct_initialization']
    rows={r['role']:r for r in report['confirmation'] if r['accepted_for_diagnostic_summary']}
    names=[name for name in order if name in rows]
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    labels={'fp32_trained_u1000':'FP32-trained u1000','bf16_trained_u1000':'BF16-trained u1000','distinct_initialization':'Initialization7502'}
    for axis,group,key,title in ((axes[0],'actual_ce_gradients','global_parameter_relative_l2','Fresh full-CE gradients'),
                                (axes[1],'adam','global_delta_relative_l2','Fresh trained-state Adam updates')):
        selected=names if group=='actual_ce_gradients' else [n for n in names if n!='distinct_initialization']
        for shift,comparison,label,color in [(-.2,'current_bf16_vs_tiled_fp32','Original BF16','#d57a64'),
                                             (.2,'candidate_vs_tiled_fp32','FP32-attention candidate','#20887c')]:
            values=[100*rows[name]['comparisons'][comparison][group][key] for name in selected]
            bars=axis.bar([i+shift for i in range(len(selected))],values,width=.36,label=label,color=color)
            axis.bar_label(bars,labels=[f'{value:.3f}%' for value in values],fontsize=8,padding=3)
        axis.axhline(100*ORIGINAL_SCREEN_TERM,color='#7b2636',linestyle='--',linewidth=1)
        axis.set_xticks(range(len(selected)),[labels[n] for n in selected],rotation=10,fontsize=9)
        axis.set_ylabel('Relative L2 (%)');axis.set_title(title);axis.legend(fontsize=8)
    axis=axes[2]
    if 'distinct_initialization' in rows:
        initial=rows['distinct_initialization']['comparisons']
        keys=['current_bf16_vs_tiled_fp32','candidate_vs_tiled_fp32']
        values=[initial[key]['adam']['global_delta_cosine'] for key in keys]
        bars=axis.bar(range(2),values,color=['#d57a64','#20887c'])
        axis.bar_label(bars,labels=[f'{value:.6f}' for value in values],padding=3,fontsize=9)
        axis.set_xticks(range(2),['Original BF16','FP32-attention candidate'],rotation=10,fontsize=9)
        axis.set_ylim(min(.985,min(values)-.002),1.002)
    else:axis.text(.5,.5,'Fresh initialization pending',ha='center',transform=axis.transAxes)
    axis.axhline(.99,color='#7b2636',linestyle='--',label='Original initial Adam gate: cosine ≥ .99')
    axis.set_title('Fresh initialization Adam agreement');axis.set_ylabel('Update cosine');axis.legend(fontsize=8)
    qualification=''
    for extra in report.get('extra_artifacts',[]):
        if extra.get('schema')=='cdrm-initial-adam-reference-v1' and extra['accepted_for_diagnostic_summary']:
            summary=extra['attribution_summary']
            qualification=(f' Initialization qualification: candidate update L2 distance {100*summary["candidate_native_relative_l2"]:.2f}%; '
                f'{100*summary["candidate_near_zero_error_energy_fraction"]:.1f}% of error energy in the declared near-zero bucket.')
    passed=sum(rows[n]['candidate_prospective_checks_pass'] is True for n in names)
    fig.suptitle(f'Fresh numerical confirmation: candidate prospective checks {passed}/{len(names)}')
    fig.text(.03,.025,'Each role uses distinct fresh B64/T256 data and identical weights/moments within its comparison.\n'
             'Gradient/trained-Adam dashed1.5625% term is illustrative; saved floors remain authoritative. Initialization uses its original cosine gate.\n'
             +qualification+' No automatic broad clearance.',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.92));fig.savefig(output/'fresh-confirmation.png',dpi=160)
    fig.savefig(output/'fresh-confirmation.svg');plt.close(fig)


def apply_arm_exclusion(report, path, verifier):
    path=Path(path).resolve(); ref=verifier.check(path); doc=json.loads(path.read_text())
    if (doc.get('schema')!='cdrm-alibi-control-effectiveness-v1'
            or doc.get('eligible_as_effective_fp32_alibi_observation') is not False
            or doc.get('other_narrow_controls_unchanged') is not True
            or doc.get('packet_differences') or doc.get('step_differences')):
        raise ValueError('Unsupported or inconsistent intervention-effectiveness exclusion')
    for name,expected in doc['source_sha256'].items():
        saved=path.parent/(path.stem+'-source')/name
        if not saved.is_file():saved=resolve_path(name,path.parent)
        verifier.check(saved,expected)
    original=verifier.record(doc['original_report'],path.parent)
    verifier.record(doc['original_tensor'],path.parent)
    producer=json.loads(Path(original['path']).read_text())
    if producer['tensor_artifact']['sha256']!=doc['original_tensor']['sha256']:
        raise ValueError('Intervention exclusion refers to a different tensor packet')
    selected=[row for row in report['coarse'] if row['input_report_sha256']==original['sha256'] and row['arm']=='bf16_alibi_fp32']
    if len(selected)!=1:raise ValueError('Supply exactly the coarse report named by the intervention exclusion')
    row=selected[0]
    row.update(accepted_for_diagnostic_summary=False,intervention_effective=False,
               comparison_exclusion_reason=doc['scope'],effectiveness_audit=ref)
    return {'artifact':ref,'input_report_sha256':original['sha256'],'arm':'bf16_alibi_fp32',
            'excluded_from_effective_intervention_summary':True,'other_arms_unchanged':True,'reason':doc['scope']}


def extra_artifact(path, verifier):
    """Retain independent descriptive artifacts without extending acceptance rules."""
    path=Path(path).resolve(); artifact=verifier.check(path); doc=json.loads(path.read_text())
    for name,expected in doc.get('source_sha256',{}).items():
        saved=path.parent/'source'/name
        if not saved.is_file():saved=resolve_path(name,path.parent)
        verifier.check(saved,expected)
    for name,expected in doc.get('input_sha256',{}).items():
        verifier.check(resolve_path(name,path.parent),expected)
    for key in ('checkpoint','tensor_artifact'):
        if doc.get(key):verifier.record(doc[key],path.parent)
    if doc.get('schema')=='cdrm-initial-adam-reference-v1':
        for key in ('source_report','source_tensors','reference_tensors','contract'):
            verifier.record(doc[key],path.parent)
        source=json.loads(resolve_path(doc['source_report']['path'],path.parent).read_text())
        if (source.get('status')!='diagnostics_complete' or source.get('role')!='distinct_initialization'
                or source['checkpoint']['sha256']!=doc['checkpoint']['sha256']
                or source['tensor_artifact']['sha256']!=doc['source_tensors']['sha256']
                or source['reference_contract']['sha256']!=doc['contract']['sha256']):
            raise ValueError('Initial Adam reference/source/checkpoint/contract identity differs')
    if doc.get('probe'):
        ref=doc['probe']; p=resolve_path(str(Path(ref['path'])/'report.json'),path.parent)
        verifier.check(p,ref['report_sha256']);verifier.check(p.parent/'tensors.pt',ref['tensors_sha256'])
        if json.loads(p.read_text()).get('status')!='diagnostics_complete':
            raise ValueError('Extra attribution depends on a failed probe')
    accepted=doc.get('status')=='diagnostics_complete' and (not doc.get('wandb') or doc['wandb']['status']=='synced')
    result={'artifact':artifact,'name':path.parent.name,'schema':doc.get('schema'),'status':doc.get('status'),
            'accepted_for_diagnostic_summary':accepted,'numerical_clearance':False,'wandb':doc.get('wandb')}
    if doc.get('schema')=='cdrm-attention-storage-attribution-v1':
        rows=[row for obs in doc['observations'].values() for row in obs['rows'].values()]
        result['attribution_summary']={'tensor_comparisons':len(rows),'coordinates':sum(r['elements'] for r in rows),
            'bf16_coordinates':sum(r['elements'] for r in rows if r['native_dtype']=='torch.bfloat16'),
            'all_native_equal_cast_local_fp32':all(r['native_equals_cast_local_fp32'] for r in rows),
            'native_cast_mismatches':sum(r['native_cast_local_fp32_mismatches'] for r in rows),
            'native_extra_energy':sum(r['components']['native_extra']['energy'] for r in rows),
            'maximum_vector_identity_residual':max(r['vector_identity_maximum_residual'] for r in rows),
            'inherited_native_interval_failing_blocks':sum(not obs['inherited_native_local_interval_pass'] for obs in doc['observations'].values()),
            'inherited_fp32_floor_failing_blocks':sum(not obs['inherited_reference_floor_supported'] for obs in doc['observations'].values())}
        result['observations']=doc['observations']
        result['scope']=doc['scope']
    if doc.get('schema')=='cdrm-block-zero-cotangent-attribution-v1':
        result['replay_parity']=doc['replay_parity']
        result['attribution']=doc['attribution']
        result['scope']=doc['scope']
        if any(row['bitwise_equal'] is not True for row in doc['replay_parity'].values()):
            result['accepted_for_diagnostic_summary']=False
    if doc.get('schema')=='cdrm-initial-adam-reference-v1':
        result.update(scope=doc['scope'],old_adam_guard=doc['old_adam_guard'],zero_prior=doc['zero_prior'],
            local_optimizer_checks_pass=doc['local_optimizer_checks_pass'],
            original_comparisons={key:compact_comparison(row) for key,row in doc['original_comparisons'].items()},
            comparisons={arm:{policy:{key:value for key,value in values.items() if key!='rows'}
                              for policy,values in policies.items()} for arm,policies in doc['comparisons'].items()},
            local_optimizer_checks={arm:{key:(value['global'] if isinstance(value,dict) and 'global' in value else value)
                                         for key,value in checks.items() if key!='rows'}
                                    for arm,checks in doc['local_optimizer_checks'].items()})
        native=doc['comparisons']['candidate']['native'];ideal=doc['comparisons']['candidate']['fp64_own_clip']
        result['attribution_summary']={'local_optimizer_checks_pass':doc['local_optimizer_checks_pass'],
            'candidate_native_relative_l2':native['global']['relative_l2'],
            'candidate_fp64_own_clip_relative_l2':ideal['global']['relative_l2'],
            'candidate_native_cosine':native['global']['cosine'],
            'candidate_initial_cosine_guard_pass':native['initial_cosine_guard_pass'],
            'candidate_near_zero_error_energy_fraction':native['global']['near_zero']['error_energy_fraction']}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for kind in SCHEMAS:parser.add_argument(f'--{kind}-report',type=Path,action='append',default=[])
    parser.add_argument('--extra-report',type=Path,action='append',default=[])
    parser.add_argument('--arm-exclusion-report',type=Path,action='append',default=[])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--completed-stage',choices=PENDING,action='append',default=[])
    parser.add_argument('--reviewed-disposition',default='Awaiting causal localization, candidate selection and fresh numerical confirmation.')
    add_wandb_arguments(parser)
    args=parser.parse_args(argv)
    inputs=[(kind,path) for kind in SCHEMAS for path in getattr(args,kind+'_report')]
    if not inputs:parser.error('Supply at least one diagnostic report')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    verifier,tracker=Verifier(),None
    report={'schema':'cdrm-precision-summary-v1','status':'running','numerical_clearance':False}
    started=time.monotonic()
    try:
        report=build_summary(inputs,verifier)
        apply_completed_stages(report,args.completed_stage)
        report['arm_exclusions']=[apply_arm_exclusion(report,path,verifier) for path in args.arm_exclusion_report]
        report['extra_artifacts']=[extra_artifact(path,verifier) for path in args.extra_report]
        report['reviewed_disposition']=args.reviewed_disposition
        report['reporter_source']={'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}
        (args.output_dir/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        write_csvs(args.output_dir,report);make_plot(args.output_dir,report)
        if report['confirmation']:make_confirmation_plot(args.output_dir,report)
        verifier.recheck()
        if args.wandb_project:
            from cdrm_tiled_validate import preserve_rng
            tracker=OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,
                group=args.wandb_group,name=args.wandb_run_name,output_dir=args.output_dir,preserve_state=preserve_rng)
            report['wandb']=tracker.record
            tracker.start({'scope':report['scope'],'reporter_source':report['reporter_source'],
                'attempts':[a['artifact'] for a in report['attempts']], 'numerical_clearance':False})
            import wandb
            tracker.log({'summary_plot':wandb.Image(str(args.output_dir/'summary.png')),
                'accepted_attempts':report['accepted_attempts'],'excluded_attempts':report['excluded_attempts']})
            if report['confirmation']:
                tracker.log({'fresh_confirmation_plot':wandb.Image(str(args.output_dir/'fresh-confirmation.png'))})
            for row in report['confirmation']:
                if row['accepted_for_diagnostic_summary']:
                    tracker.log({f'confirmation/{row["role"]}/candidate_prospective_checks_pass':row['candidate_prospective_checks_pass'],
                        f'confirmation/{row["role"]}/original_machine_screens_pass':row['machine_screens_pass'],
                        **{f'confirmation/{row["role"]}/{name}/gradient_relative_l2':value['actual_ce_gradients']['global_parameter_relative_l2']
                           for name,value in row['comparisons'].items()},
                        **{f'confirmation/{row["role"]}/{name}/adam_relative_l2':value['adam']['global_delta_relative_l2']
                           for name,value in row['comparisons'].items()},
                        **{f'confirmation/{row["role"]}/{name}/adam_cosine':value['adam']['global_delta_cosine']
                           for name,value in row['comparisons'].items()}})
            for row in report['coarse']:
                if row['accepted_for_diagnostic_summary']:
                    tracker.log({f'coarse/{row["attempt"]}/{row["arm"]}/gradient_relative_l2':row['gradient_relative_l2'],
                                 f'coarse/{row["attempt"]}/{row["arm"]}/adam_relative_l2':row['adam_relative_l2']})
            for row in report['extra_artifacts']:
                if row['accepted_for_diagnostic_summary'] and row.get('attribution_summary'):
                    tracker.log({f'extra/{row["name"]}/{key}':value for key,value in row['attribution_summary'].items() if isinstance(value,(bool,int,float))})
            for row in report['embedding']:
                if row['accepted_for_diagnostic_summary']:
                    tracker.log({f'embedding/{row["case"]}/{key}/relative_l2':row[key]['relative_l2']
                        for key in ('incoming_adjoint_policy_difference','native_policy_difference','fp64_accumulated_policy_difference')})
            tracker.summary({'numerical_clearance':False,'accepted_attempts':report['accepted_attempts'],
                'excluded_attempts':report['excluded_attempts'],'reviewed_disposition':args.reviewed_disposition})
        if digest(__file__) != report['reporter_source']['sha256']:raise ValueError('Reporter source changed')
        verifier.recheck()
        report['verified_input_files']=len(verifier.files)
        report['status']='summary_complete'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        try:
            if tracker:tracker.finish(succeeded=report['status']=='summary_complete')
        except Exception as error:
            report.update(status='failed',final_sync_error_type=type(error).__name__)
            raise
        finally:save_json(args.output_dir/'report.json',report)


if __name__=='__main__':main()
