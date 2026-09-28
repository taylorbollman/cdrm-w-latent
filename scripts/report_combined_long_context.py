#!/usr/bin/env python3
"""Read-only combined T2048 evidence audit and optional CPU figures.

Adapted from the ordinary long-context local audit. Raw stage files are never
modified. Retention checks inspect saved receipts/local archives, not the cloud.
No torch, model loading or CUDA; plotting requires the project CPU container.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'docs/reports/olmo-combined-long-context/protocol.md'
PARAMETERS = dict(backbone=1176764416, fusion=8388608, nextlat_training_only=82726912,
                  training_architecture=1267879936, deployable_inference=1185153024)
REQUIRED_SOURCES = {'scripts/olmo_two_gpu_single_reference.py', 'scripts/olmo_two_gpu_validate.py',
    'scripts/olmo_rt_large_batch.py', 'cdrm/pretrained/olmo_tiled.py',
    'cdrm/pretrained/static_training.py', 'cdrm/pretrained/static_nextlat.py',
    'cdrm/pretrained/nextlat.py', 'cdrm/pretrained/olmo_fbt.py', 'cdrm/pretrained/fbt_training.py'}
OWN_CHECKS = {'pre_capture_eager_vs_graph', 'changed_weight_graph_vs_eager'}
NEW_CHECKS = OWN_CHECKS | {'combined_dispatch_native_rt_and_ordinary',
                         'dependencies_unchanged', 'sources_unchanged'}
CHECKPOINT_SHA = 'ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def source_check(directory, report, historical=False):
    pins = report.get('sources', {})
    required = REQUIRED_SOURCES | (set() if historical else {PROTOCOL})
    issues = [f'missing source pin: {name}' for name in sorted(required - pins.keys())]
    verified = 0
    for name, sha in pins.items():
        base = directory/'source-snapshot'
        path = base/name
        if not path.resolve().is_relative_to(base.resolve()):
            issues.append(f'unsafe source path: {name}')
        elif not path.is_file() or digest(path) != sha:
            issues.append(f'missing or mismatched source: {name}')
        else:
            verified += 1
    return dict(pairs=len(pins), verified_pairs=verified, issues=issues)


def dependency_check(directory, report):
    issues, pairs, verified = [], 0, 0
    dependencies = report.get('dependencies', {})
    if not dependencies.get('packages'):
        issues.append('missing dependency package inventory')
    for key, subdir in (('dao_sources', 'flash_attn'), ('fa4_sources', 'flash_attn/cute')):
        for name, row in dependencies.get(key, {}).items():
            pairs += 1
            base = directory/'dependency-snapshot'/subdir
            path = base/name
            if (path.resolve().is_relative_to(base.resolve()) and path.is_file()
                    and digest(path) == row.get('sha256')):
                verified += 1
            else:
                issues.append(f'missing, unsafe or mismatched dependency: {key}/{name}')
    return dict(pairs=pairs, verified_pairs=verified, issues=issues)


def retention_check(directory):
    base = directory.parent/'retention'
    path = base/(directory.name+'.json')
    if not path.is_file():
        return dict(verified=False, status='missing', issues=[])
    receipt = read(path)
    artifacts = base/(directory.name+'.artifacts')
    issues = []
    manifest = artifacts/'retention-manifest.json'
    matched = None
    if manifest.is_file():
        member = next((row for row in read(manifest).get('members', [])
                       if row.get('path') == 'evidence/report.json'), None)
        matched = bool(member and digest(directory/'report.json') == member.get('sha256'))
        if not matched:
            issues.append('report differs from retained manifest')
    objects = receipt.get('objects', [])
    for obj in objects:
        if not all(obj.get('verification', {}).get(key) is True
                   for key in ('server_md5', 'server_size', 'sha256_metadata')):
            issues.append('incomplete saved remote verification')
        local = artifacts/str(obj.get('uri', '')).rsplit('/', 1)[-1]
        if local.name in ('retention-manifest.json', 'evidence.tar.gz') and local.is_file():
            if local.stat().st_size != obj.get('size_bytes') or digest(local) != obj.get('sha256'):
                issues.append(f'local archive/manifest differs: {local.name}')
    return dict(verified=receipt.get('status') == 'verified' and bool(objects) and not issues,
                status=receipt.get('status'), issues=issues, report_matches_manifest=matched,
                objects=objects, prefix=receipt.get('prefix'),
                scope='Saved verification receipt and available local hashes; no live cloud request.')


def backend_evidence(directory, report, historical=False):
    result = {}
    for key, expected in (('ordinary_attention_backend', 'sdpa'), ('ordinary_rope_backend', 'native')):
        values = [report.get(section, {}).get(key) for section in ('configuration', 'execution_options')
                  if key in report.get(section, {})]
        result[key] = dict(value=values[0] if values and len(set(values)) == 1 else None,
                           verified=bool(values) and set(values) == {expected}, basis='explicit configuration')
    if not historical or all(row['verified'] for row in result.values()):
        return result
    # This exact older harness predates explicit performance selectors. Inspect
    # frozen construct/arm declarations; do not infer execution from run names.
    try:
        scripts = directory/'source-snapshot/scripts'
        tree = ast.parse((scripts/'olmo_two_gpu_validate.py').read_text())
        construct = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'construct')
        calls = [n for n in ast.walk(construct) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'set_arm']
        tree = ast.parse((scripts/'olmo_rt_large_batch.py').read_text())
        arms = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
        harness = ast.parse((scripts/'olmo_two_gpu_single_reference.py').read_text())
        writes = {n.attr for tree in (construct, harness) for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
        direct = [ast.literal_eval(n.args[1]) for n in calls] == ['compiled-native']
        for key, arm_key, expected in (('ordinary_attention_backend', 'attention', 'sdpa'),
                                       ('ordinary_rope_backend', 'rope', 'native')):
            if (result[key]['value'] is None and direct and key not in writes
                    and arms['compiled-native'][arm_key] == expected):
                result[key] = dict(value=expected, verified=True,
                    basis='Frozen historical construct selects compiled-native; single harness has no backend assignment')
    except (OSError, SyntaxError, ValueError, StopIteration, KeyError, IndexError):
        pass
    return result


def contract_check(report, historical=False):
    case, mode, cfg = (report.get(key, {}) for key in ('case', 'mode', 'configuration'))
    resources = report.get('resources', {})
    ledger, observed = resources.get('analytic_matrix_work', {}), resources.get('observed_parameters', {})
    batch, length = case.get('batch_size'), case.get('length')
    shape_ok = type(batch) is int and batch >= 2 and batch % 2 == 0 and length == (512 if historical else 2048)
    expected = dict(ce=batch*(length-1), latent=batch*(length-1), kl=batch*(length//2)) if shape_ok else {}
    checks = dict(shape=shape_ok, case=case.get('name') == 'combined',
        features=case.get('fbt') is True and case.get('nextlat') is True and mode.get('enabled') is True,
        passes=case.get('passes') == mode.get('num_passes') == 2,
        rt_layers=case.get('rt_layers') == mode.get('rt_mode', {}).get('selected_layers') == [0, 15],
        alpha_beta=case.get('alpha') == mode.get('rt_mode', {}).get('alpha') == case.get('beta') == mode.get('beta') == 1.,
        single_gpu=cfg.get('world_size') == 1 and cfg.get('ddp_buckets') is False,
        no_accumulation=cfg.get('accumulation_steps') == 1,
        full_model=cfg.get('tiny') is False,
        blocks=ledger.get('ordinary_block_calls_per_microbatch') == 30 and ledger.get('rt_block_calls_per_microbatch') == 2,
        parameters=ledger.get('parameter_counts') == PARAMETERS,
        active_parameters=all(observed.get(key) == PARAMETERS['training_architecture'] for key in
            ('registered_unique', 'trainable', 'executed_declared', 'gradient_participating', 'optimizer_owned')),
        inference_parameters=observed.get('deployable_inference_declared') == PARAMETERS['deployable_inference'],
        checkpoint=report.get('checkpoint', {}).get('sha256') == CHECKPOINT_SHA)
    checks['counts'] = bool(expected) and ledger.get('objective_positions_per_update') == {**expected, 'predictor': expected['latent']}
    checks['pass_work'] = shape_ok and ledger.get('input_tokens_per_update') == batch*length and ledger.get('pass_token_work_per_update') == 2*batch*length
    checks['across_pass_counts'] = bool(expected) and ledger.get('objective_positions_across_passes') == {**{k:2*v for k,v in expected.items()}, 'predictor': 2*expected['latent']}
    records = report.get('preparation_updates', []) + [r.get('metrics', {}) for r in report.get('timed_updates', [])]
    checks['actual_update_objectives'] = len(records) == 8 and all(
        r.get('counts') == expected and r.get('objective_weights') == dict(ce=1., latent=1., kl=1.)
        and r.get('update_completed') is True and all(math.isfinite(r.get('loss_means', {}).get(k, math.nan)) for k in expected)
        for r in records)
    if not historical:
        model = report.get('model_contract', {})
        checks['explicit_objectives'] = (model.get('nextlat_enabled') is True and model.get('counts_per_pass') == expected
            and model.get('objective_weights') == dict(ce=1., latent=1., kl=1.) and model.get('pass_loss_gamma') == 1.)
        checks['backbone'] = all(model.get('backbone', {}).get(k) == v for k,v in
            dict(model_dim=2048, num_layers=16, num_heads=16, mlp_intermediate_size=8192, vocab_size=50304, max_context_length=2048).items())
        checks['nextlat_recipe'] = all(model.get('nextlat', {}).get(k) == v for k,v in
            dict(model_dim=2048, proj_factor=1.6, lambda_latent=1., lambda_kl=1., vocab_chunk_size=128, ce_chunk_size=2048).items())
        checks['ordinary_execution'] = all(model.get('ordinary', {}).get(k) == v for k,v in
            dict(ordinary_attention_backend='sdpa', ordinary_rope_backend='native', ordinary_pointwise_backend='compiled',
                 ordinary_activation_checkpointing=True, ordinary_checkpoint_layers=None).items())
        checks['native_rt_execution'] = all(model.get('native_rt', {}).get(k) == v for k,v in
            dict(rt_implementation='native', attention_precision='mixed', tile_backend='triton', backward_tile_backend='triton',
                 backward_memory='recompute', cast_weights_once=True, reuse_rope=True, kv_only_writes=True).items())
    return dict(passed=all(checks.values()), checks=checks, expected_counts_per_pass=expected)


def dispatch_check(report, historical=False):
    if historical:
        return dict(passed=True, scope='Historical dispatch inferred from verified frozen source; no new instrumentation.')
    row = next((c for c in report.get('checks', [])
                if c.get('name') == 'combined_dispatch_native_rt_and_ordinary'), {})
    rt = row.get('native_rt', {})
    shapes = rt.get('forward_tiles_by_shape', {})
    triton = rt.get('forward_triton_tiles_by_shape', {})
    expected = dict(Counter(f'{boundary & -boundary}x{boundary & -boundary}'
                           for boundary in range(1, 2048)))
    expected = {k:2*v for k,v in expected.items()}
    checks = dict(ordinary=row.get('ordinary', {}).get('passed') is True,
        native_rt=rt.get('passed') is True, forward=shapes == expected,
        triton=triton == {k:v for k,v in expected.items() if int(k.split('x')[0]) <= 256},
        eager=rt.get('forward_eager_tiles_by_shape') == {'512x512': 4, '1024x1024': 2},
        backward=rt.get('backward_recomputed_triton_calls') == rt.get('expected_backward_recomputed_triton_calls') == 4094)
    return dict(passed=all(checks.values()), checks=checks, observed=row)



def timing_check(report):
    rows = report.get('timed_updates', [])
    checks = dict(five_timed_updates=len(rows) == 5, eight_physical_updates=report.get('physical_updates') == 8)
    valid = bool(rows) and all(type(r.get('input_tokens')) is int and r['input_tokens'] > 0
        and isinstance(r.get('seconds'), (int, float)) and math.isfinite(r['seconds']) and r['seconds'] > 0 for r in rows)
    seconds = sum(r['seconds'] for r in rows) if valid else 0.
    tokens = sum(r['input_tokens'] for r in rows) if valid else 0
    rate = tokens/seconds if seconds else None
    throughput = report.get('throughput', {})
    checks['timing_records'] = valid
    checks['rate_matches_records'] = rate is not None and math.isclose(rate, throughput.get('global_tokens_per_second', math.nan), rel_tol=1e-10)
    checks['update_seconds_match'] = valid and math.isclose(seconds/len(rows), throughput.get('seconds_per_update', math.nan), rel_tol=1e-10)
    case = report.get('case', {})
    checks['input_tokens_match_shape'] = valid and all(r['input_tokens'] == case.get('batch_size', 0)*case.get('length', 0) for r in rows)
    return dict(passed=all(checks.values()), checks=checks, total_timed_tokens=tokens, total_timed_seconds=seconds, tokens_per_second=rate)


def flatten_stage(directory, historical=False):
    report = read(directory/'report.json')
    source, deps = source_check(directory, report, historical), dependency_check(directory, report)
    retention = retention_check(directory)
    backend, contract, timing = backend_evidence(directory, report, historical), contract_check(report, historical), timing_check(report)
    checks = report.get('checks', [])
    names = [r.get('name') for r in checks]
    missing = sorted((OWN_CHECKS if historical else NEW_CHECKS) - set(names))
    gates = dict(recorded=len(checks), passed=sum(c.get('passed') is True for c in checks),
        failed=sum(c.get('passed') is False for c in checks), missing=missing, duplicate_names=len(names) != len(set(names)))
    gates['complete_and_passed'] = bool(checks) and not missing and not gates['duplicate_names'] and gates['passed'] == len(checks)
    config, case = report.get('configuration', {}), report.get('case', {})
    batch, length = case.get('batch_size'), case.get('length')
    diagnostic = not historical and (batch == 2 or config.get('stage') == 'correctness')
    dispatch = dispatch_check(report, historical)
    setup, steady = report.get('setup_memory', {}), report.get('steady_memory', {})
    memory_valid = all(isinstance(container.get(k), (int, float)) and math.isfinite(container[k]) and container[k] >= 0
        for container, keys in ((setup, ('peak_allocated_gib', 'peak_reserved_gib')),
                                (steady, ('allocated_gib', 'reserved_gib', 'device_free_gib'))) for k in keys)
    reasons = []
    for condition, reason in ((report.get('status') != 'passed', 'status is not passed'),
        (not gates['complete_and_passed'], 'operational gates incomplete or failed'),
        (diagnostic, 'B2/correctness diagnostic'), (not contract['passed'], 'combined execution contract failed'),
        (not timing['passed'], 'timing/update audit failed'),
        (not dispatch['passed'], 'dispatch evidence audit failed'), (not memory_valid, 'memory inventory unavailable'),
        (bool(source['issues']), 'source integrity failed'),
        (bool(deps['issues']), 'dependency integrity failed'),
        (not all(r['verified'] for r in backend.values()), 'native RoPE / SDPA backend unverified')):
        if condition: reasons.append(reason)
    setup, steady = report.get('setup_memory', {}), report.get('steady_memory', {})
    return dict(stage=directory.name, origin='historical_T512_reference' if historical else 'new_T2048_experiment',
        report=str(directory/'report.json'), report_sha256=digest(directory/'report.json'),
        status=report.get('status', 'incomplete'), error=report.get('error'), diagnostic=diagnostic,
        batch_size=batch, length=length, input_tokens_per_update=batch*length if type(batch) is int and type(length) is int else None,
        total_timed_tokens=timing['total_timed_tokens'], total_timed_seconds=timing['total_timed_seconds'],
        tokens_per_second=timing['tokens_per_second'], seconds_per_update=report.get('throughput', {}).get('seconds_per_update'),
        physical_updates=report.get('physical_updates', 0), gates=gates, combined_contract=contract, timing_check=timing,
        source_check=source, dependency_check=deps, retention=retention, backend_evidence=backend, dispatch=dispatch,
        setup_peak_allocated_gib=setup.get('peak_allocated_gib'), setup_peak_reserved_gib=setup.get('peak_reserved_gib'),
        steady_allocated_gib=steady.get('allocated_gib'), steady_reserved_gib=steady.get('reserved_gib'),
        sampled_free_gib=steady.get('device_free_gib'), device_used_gib=steady.get('device_used_gib'),
        plot_eligible=not reasons, plot_exclusion_reasons=reasons, git_head=report.get('git_head'),
        runtime=report.get('runtime', {}), dependency_packages=report.get('dependencies', {}).get('packages', {}),
        runtime_sources={k:v for k,v in report.get('sources', {}).items() if k.startswith('cdrm/pretrained/')},
        wandb=report.get('wandb', {}).get('run_url'))


def pooled_measurements(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['plot_eligible']:
            groups[(row['origin'], row['length'], row['batch_size'])].append(row)
    result = []
    for (origin, length, batch), group in sorted(groups.items()):
        tokens = sum(r['total_timed_tokens'] for r in group)
        seconds = sum(r['total_timed_seconds'] for r in group)
        result.append(dict(origin=origin, length=length, batch_size=batch, input_tokens_per_update=length*batch,
            runs=len(group), stages=[r['stage'] for r in group], total_timed_tokens=tokens, total_timed_seconds=seconds,
            tokens_per_second=tokens/seconds, seconds_per_update=seconds/(5*len(group)),
            min_run_tokens_per_second=min(r['tokens_per_second'] for r in group), max_run_tokens_per_second=max(r['tokens_per_second'] for r in group),
            setup_peak_allocated_gib=max(r['setup_peak_allocated_gib'] for r in group),
            setup_peak_reserved_gib=max(r['setup_peak_reserved_gib'] for r in group),
            steady_reserved_gib=max(r['steady_reserved_gib'] for r in group),
            sampled_free_gib=min(r['sampled_free_gib'] for r in group)))
    historical = next((r for r in result if r['origin'] == 'historical_T512_reference'), None)
    for row in result:
        if historical and row['origin'] != 'historical_T512_reference':
            row['rate_ratio_to_historical'] = row['tokens_per_second']/historical['tokens_per_second']
            row['same_input_tokens_per_update'] = row['input_tokens_per_update'] == historical['input_tokens_per_update']
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows)


def audit(evidence, historical):
    rows, unreadable = [], []
    candidates = [(p.parent, False) for p in sorted(evidence.glob('*/report.json'))]
    if (historical/'report.json').is_file(): candidates.append((historical, True))
    for directory, old in candidates:
        try: rows.append(flatten_stage(directory, historical=old))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            unreadable.append(dict(stage=directory.name, historical=old, error=str(exc)))
    old = next((r for r in rows if r['origin'] == 'historical_T512_reference'), None)
    new = [r for r in rows if r['origin'] == 'new_T2048_experiment']
    if old:
        for row in new:
            before, after = old['runtime_sources'], row['runtime_sources']
            row['historical_comparison'] = dict(all_pretrained_runtime_sources_equal=bool(before) and before == after,
                unchanged_source_pairs=sum(after.get(k) == v for k,v in before.items()),
                changed_or_missing_sources=sorted(k for k,v in before.items() if after.get(k) != v),
                added_sources=sorted(after.keys()-before.keys()),
                torch_cuda_gpu_equal=all(row['runtime'].get(k) == old['runtime'].get(k) for k in ('torch', 'cuda', 'gpu')),
                package_changes={k:dict(historical=v, current=row['dependency_packages'].get(k))
                    for k,v in old['dependency_packages'].items() if row['dependency_packages'].get(k) != v})
    final = [r for r in new if r['status'] in ('passed', 'failed', 'oom')]
    return dict(schema='olmo-combined-long-context-audit-v1', generated_utc=datetime.now(timezone.utc).isoformat(),
        rows=rows, unreadable_reports=unreadable, pooled=pooled_measurements(rows),
        summary=dict(new_reports=len(new), new_final_reports=len(final), new_statuses=dict(Counter(r['status'] for r in new)),
            new_physical_updates=sum(r['physical_updates'] for r in final),
            new_source_pairs=sum(r['source_check']['verified_pairs'] for r in final),
            new_dependency_pairs=sum(r['dependency_check']['verified_pairs'] for r in final),
            new_verified_retention=sum(r['retention']['verified'] for r in final),
            new_plot_eligible=sum(r['plot_eligible'] for r in new),
            new_operational_checks=sum(r['gates']['recorded'] for r in final),
            new_operational_checks_passed=sum(r['gates']['passed'] for r in final)),
        notes=['Historical T512 is explicitly selected; it is not a new or interleaved measurement.',
            'SDPA only. No FA4 comparison/adoption or numerical equivalence claim.',
            'K2 ordinary bootstrap + feedback with native RT0/15; NextLat on both passes. Tokens count input once.',
            'Pooled rates are total timed tokens / total timed wall seconds, not averages of rates.',
            'B2 diagnostic and every failed/incomplete report stay in the ledger but not performance figures.',
            'Saved receipt verification is not a fresh cloud request; pending retention does not invalidate measured performance.',
            'Free memory is sampled, not continuous; steady allocated omits reserved graph pools.',
            'Equal input tokens does not equal physical batch, recurrence depth or model quality.'])


def plot(result, output):
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Plot in the project CPU container with CDRM_DOCKER_GPUS=none')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['pooled']
    if not rows: return
    labels = [f'T{r["length"]} / B{r["batch_size"]}'+('\nHistorical' if r['origin'].startswith('historical') else f'\n{r["runs"]} run(s)') for r in rows]
    colors = ['#777777' if r['origin'].startswith('historical') else '#2166ac' for r in rows]
    for name, fields in [('throughput', [('tokens_per_second', 'Input tokens / second', 1)]),
        ('memory', [('setup_peak_allocated_gib', 'Setup peak allocated (GiB)', 1),
                    ('steady_reserved_gib', 'Steady reserved (GiB)', 1), ('sampled_free_gib', 'Sampled free (GiB)', 1)])]:
        fig, axes = plt.subplots(1, len(fields), figsize=(max(7, len(fields)*5), 4.9), squeeze=False)
        for ax, (field, title, scale) in zip(axes[0], fields):
            values = [r[field]/scale for r in rows]
            bars = ax.bar(labels, values, color=colors)
            ax.bar_label(bars, labels=[f'{v:,.1f}' for v in values], padding=3, fontsize=8)
            ax.set_ylabel(title); ax.set_ylim(0, max(values)*1.16); ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Combined FBT + native RT + NextLat: one H100, Flash SDPA')
        fig.text(.5, .012, 'Graphs, ordinary-layer checkpointing, native RoPE, fused Adam. Historical T512 is not a new measurement.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94))
        for suffix in ('pdf', 'png'): fig.savefig(output/f'{name}.{suffix}', dpi=170, bbox_inches='tight')
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT/'.runtime/olmo-combined-long-context')
    parser.add_argument('--historical-dir', type=Path, default=ROOT/'.runtime/olmo-two-gpu/single-combined-b128-01')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(argv)
    evidence, historical, output = (p.resolve() for p in (args.evidence_dir, args.historical_dir, args.output_dir))
    if output == evidence or output.is_relative_to(historical) or any(output.is_relative_to(p.parent) for p in evidence.glob('*/report.json')):
        parser.error('Derived output must not overlap raw stage evidence')
    output.mkdir(parents=True, exist_ok=True)
    result = audit(evidence, historical)
    (output/'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    fields = ('stage', 'origin', 'status', 'diagnostic', 'length', 'batch_size', 'input_tokens_per_update',
        'tokens_per_second', 'seconds_per_update', 'setup_peak_allocated_gib', 'setup_peak_reserved_gib',
        'steady_reserved_gib', 'sampled_free_gib', 'physical_updates', 'plot_eligible', 'plot_exclusion_reasons', 'gates', 'wandb')
    write_csv(output/'stages.csv', [{k:r[k] for k in fields} for r in result['rows']])
    write_csv(output/'performance.csv', result['pooled'])
    lines = ['# Combined OLMo T2048 evidence summary', '', *result['notes'], '',
        '| Origin | T / B | Runs | Input tokens/s | Setup allocated / reserved GiB | Steady reserved / sampled free GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['pooled']:
        lines.append(f'| {r["origin"]} | {r["length"]} / {r["batch_size"]} | {r["runs"]} | {r["tokens_per_second"]:,.2f} | '
            f'{r["setup_peak_allocated_gib"]:.3f} / {r["setup_peak_reserved_gib"]:.3f} | {r["steady_reserved_gib"]:.3f} / {r["sampled_free_gib"]:.3f} |')
    lines += ['', 'New-run totals: `'+json.dumps(result['summary'], sort_keys=True)+'`', '']
    (output/'summary.md').write_text('\n'.join(lines))
    if args.plots: plot(result, output)
    print(json.dumps(dict(summary=result['summary'], unreadable=result['unreadable_reports']), indent=2))


if __name__ == '__main__': main()
