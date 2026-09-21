#!/usr/bin/env python3
"""CPU-only 7.5k three-layer continuation with an explicitly bounded 5k reference.

Historical training and reporting sources remain unchanged. Comparisons between
architectures stop at their common 5000-update budget. Later three-layer results
are continuation context, never an extrapolated two-layer comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts import rt_nextlat_a5_fuzzy_depth_resume_report as resume
from scripts import rt_nextlat_a5_fuzzy_depth_report as depth
from scripts import rt_nextlat_a5_fuzzy_lr_report as lr
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired
from scripts import rt_nextlat_a5_fuzzy_report as saved

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-depth-extension-comparison-v1"
QUALIFICATION = (
    depth.QUALIFICATION + " " + resume.RECOVERY_QUALIFICATION + " "
    "Only the three-layer model continues beyond 5000 updates. The two-layer reference ends at its "
    "actual saved 5000-update checkpoint; all matched-update comparisons and matched training times "
    "stop there. Three-layer results from 5001 through its actual endpoint (at most 7500) are "
    "unequal-budget continuation context. There is no extrapolated reference curve or reference "
    "training time beyond 5000. The 5000-update result and any continuation result are disclosed "
    "separately, and no learning-rate, optimizer, model or data recipe changes at the extension."
)
require, read_json, sha, json_sha = saved.require, saved.read_json, saved.sha, saved.json_sha
local_path, read_history_prefix = saved.local_path, saved.read_history_prefix
_evaluations, selected_evaluations = saved._evaluations, saved.selected_evaluations
boundary_state_proof = resume.boundary_state_proof


def _load_lineage(directory, *, through=None, ancestors=()):
    directory = local_path(directory)
    require(directory not in ancestors and len(ancestors) < 16, "Cyclic or excessively deep checkpoint lineage")
    report_path = directory / "report.json"
    report = read_json(report_path)
    terminal = through is None
    require(report.get("schema") == lr.TRAIN_SCHEMA and report.get("requested_endpoint") in (5000, 7500),
            "Expected the LR-training schema and an approved 5000 or 7500 endpoint")
    require(not terminal or report["requested_endpoint"] == 7500, "Terminal extension must request 7500 updates")
    require(report["contract"].get("schema") == lr.TRAIN_SCHEMA
            and report["contract"].get("mode") == "mixed", "Expected a mixed LR-training contract")
    require(report.get("status") in (("complete", "stopped") if terminal else ("running", "complete", "stopped")),
            "Require a completed or explicitly stopped training report; running is allowed only for a bound ancestor")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Final confirmation and autonomous latent rollout must remain unused")
    reported_endpoint, start = report.get("completed_updates"), report.get("start_update", 0)
    endpoint = reported_endpoint if terminal else through
    require(type(reported_endpoint) is int and type(start) is int and type(endpoint) is int
            and 0 <= start <= endpoint <= reported_endpoint <= report["requested_endpoint"] <= 7500 and endpoint > 0, "Invalid actual endpoint or restored boundary")
    require(not terminal or report["status"] != "complete" or endpoint == report["requested_endpoint"],
            "Complete status must reach the requested endpoint")
    contract = report["contract"]
    config_path, identity_path = directory / "model-config.json", directory / "data-identity.json"
    identity = read_json(identity_path)
    require(read_json(config_path) == contract["model_config"]
            and sha(config_path) == contract["configuration_file_sha256"], "Saved model configuration differs")
    require(json_sha(identity) == contract["data_sha256"], "Saved dataset identity differs")
    require(report["initialization"] == contract["initialization"], "Initialization record differs")
    sources = read_json(directory / "source-manifest.json")
    require(json_sha(sources) == contract["source_sha256"], "Source manifest differs")
    for relative, digest in sources.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Invalid frozen source path")
        require(sha(directory / "source" / relative) == digest, f"Frozen source differs: {relative}")

    parent_record, parent = report.get("parent_checkpoint"), None
    if start:
        require(isinstance(parent_record, dict), "Continuation is missing its parent checkpoint identity")
        parent_path = local_path(parent_record["path"])
        require(parent_path.parent.name == "checkpoints" and parent_path.name == f"step-{start:06d}.pt",
                "Parent checkpoint path does not identify the restored boundary")
        require(parent_path.is_file() and sha(parent_path) == parent_record["sha256"], "Parent checkpoint hash mismatch")
        parent = _load_lineage(parent_path.parent.parent, through=start, ancestors=(*ancestors, directory))
        require(parent["checkpoints"][start]["sha256"] == parent_record["sha256"],
                "Parent report records a different boundary checkpoint")
        require(parent["report"]["contract"] == contract and parent["report"]["initialization"] == report["initialization"]
                and parent["data_identity"] == identity and parent["sources"] == sources,
                "Exact continuation contract, source, initialization or data identity differs")
    else:
        require(parent_record is None, "Fresh training cannot claim a parent checkpoint")

    checkpoints = dict(parent["checkpoints"]) if parent else {}
    seen, duplicate_boundary, exact_boundary_audit, discarded_checkpoints = set(), None, None, 0
    for record in report["checkpoints"]:
        update = record["completed_updates"]
        require(type(update) is int and start <= update <= reported_endpoint and update not in seen,
                "Invalid or duplicate checkpoint update")
        seen.add(update)
        if update > endpoint:
            discarded_checkpoints += 1
            continue
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        require(path.is_file() and sha(path) == record["sha256"], "Checkpoint hash mismatch")
        require(path.stat().st_size == record["bytes"], "Checkpoint byte count mismatch")
        if update in checkpoints:
            require(update == start, "Only the restored boundary checkpoint can be duplicated")
            byte_identical = record["sha256"] == checkpoints[update]["sha256"] and record["bytes"] == checkpoints[update]["bytes"]
            exact_boundary_audit = boundary_state_proof(directory, parent, record, path, update)
            duplicate_boundary = {**record, "verified_local_path": str(path), "byte_identical": byte_identical,
                                  "exact_state_verified": byte_identical or exact_boundary_audit is not None}
        else:
            checkpoints[update] = {**record, "verified_local_path": str(path)}
    require(0 in checkpoints and endpoint in checkpoints and start in seen, "Initial, restored or endpoint checkpoint is missing")
    require(not start or duplicate_boundary is not None, "Continuation did not save its restored boundary")
    own_evaluations = [m for m in report["evaluations"] if start < m["update"] <= endpoint]
    if not start:
        own_evaluations = [m for m in report["evaluations"] if 0 <= m["update"] <= endpoint]
    evaluations = list(parent["evaluations"]) if parent else []
    boundary_only_stop = bool(start and terminal and endpoint == start)
    if boundary_only_stop:
        # A stop immediately after restoration has no new updates, but its fresh
        # full endpoint evaluation must supersede the ancestor's monitoring subset.
        checkpoints[start] = duplicate_boundary
        evaluations = [metric for metric in evaluations if metric["update"] != start]
        own_evaluations = [metric for metric in report["evaluations"] if metric["update"] == start]
    evaluations += _evaluations({**report, "evaluations": own_evaluations})
    keys = [(m["update"], m["task"], m.get("role", "dev"), m.get("rows", m.get("examples"))) for m in evaluations]
    require(len(set(keys)) == len(keys), "Duplicate evaluation task/role/update")
    require(all(type(k[0]) is int and 0 <= k[0] <= endpoint for k in keys), "Evaluation beyond actual endpoint")
    tasks = {m["task"] for m in evaluations}
    expected_tasks = {"a5", "fuzzy"}
    require(tasks == expected_tasks, "Evaluation tasks differ from the training mode")
    for metric in evaluations:
        if report.get("schema") != "rt-nextlat-fuzzy-training-v1":
            cp = metric.get("checkpoint", {})
            require(metric["update"] in checkpoints and cp.get("completed_updates") == metric["update"]
                    and cp.get("sha256") == checkpoints[metric["update"]]["sha256"],
                    "Evaluation is bound to a different checkpoint")
    if terminal:
        required = {("a5", "dev"), ("a5", "ood_dev")} if "a5" in tasks else set()
        if "fuzzy" in tasks:
            required.add(("fuzzy", "dev"))
        actual = {(task, role) for update, task, role, _ in keys if update == endpoint}
        require(required and actual == required, "Endpoint must contain all task evaluations at the same update")
        for metric in selected_evaluations(evaluations):
            if metric["update"] == endpoint:
                require(metric.get("rows", metric.get("examples")) == (102400 if metric["task"] == "a5" else 1280),
                        "Endpoint requires the full development evaluation")
    history_path = directory / "history.jsonl"
    own_history, history_audit = read_history_prefix(history_path, start=start, through=endpoint, allow_tail=not terminal)
    history = (list(parent["history"]) if parent else []) + own_history
    require([r["update"] for r in history] == list(range(1, endpoint + 1)), "Stitched history has a gap or duplicate")
    input_hashes = {"report": sha(report_path), "history": sha(history_path),
                    "model_config": sha(config_path), "data_identity": sha(identity_path)}
    stage = {"directory": str(directory), "status": report["status"], "start_update": start,
             "reported_completed_updates": reported_endpoint, "used_through_update": endpoint,
             "parent_checkpoint": parent_record, "input_hashes": input_hashes,
             "history": history_audit, "discarded_post_boundary_checkpoints": discarded_checkpoints,
             "discarded_post_boundary_evaluations": sum(m["update"] > endpoint for m in report["evaluations"]),
             "omitted_child_boundary_evaluations": sum(m["update"] == start for m in report["evaluations"])
                if start and not boundary_only_stop else 0,
             "endpoint_boundary_reevaluation_used": boundary_only_stop,
             "boundary_checkpoint_copy": duplicate_boundary, "boundary_state_audit": exact_boundary_audit,
             "training_wandb": report.get("wandb")}
    lineage = (list(parent["lineage"]) if parent else []) + [stage]
    return {"directory": str(directory), "report": report, "endpoint": endpoint,
            "checkpoints": checkpoints, "evaluations": evaluations, "history": history,
            "data_identity": identity, "sources": sources,
            "input_hashes": input_hashes, "lineage": lineage, "tasks": sorted(tasks)}


def load_run(directory):
    return _load_lineage(directory)


def matched_ancestor(candidate):
    """Load the actual completed 5k ancestor, without manufacturing a terminal report."""
    for stage in reversed(candidate['lineage']):
        if stage['used_through_update'] != 5000:
            continue
        report = read_json(Path(stage['directory']) / 'report.json')
        if report['requested_endpoint'] == 5000 and report['status'] == 'complete' and report['completed_updates'] == 5000:
            parent = resume.load_run(stage['directory'])
            require(parent['history'] == candidate['history'][:5000], 'The completed 5k ancestor prefix changed')
            require(parent['report']['contract'] == candidate['report']['contract']
                    and parent['sources'] == candidate['sources']
                    and parent['data_identity'] == candidate['data_identity'], 'Continuation recipe or identity changed')
            return parent
    raise ValueError('The extension requires its actual completed 5000-update ancestor')


def compare(base, candidate):
    require(5000 <= candidate['endpoint'] <= 7500 and candidate['report']['requested_endpoint'] == 7500,
            'Expected a three-layer extension from 5k toward 7.5k')
    require(candidate['report']['status'] in ('complete', 'stopped'), 'Extension must be terminal')
    parent = matched_ancestor(candidate)
    matched = depth.compare(base, parent)
    # A self-check validates global update, exposure, LR and finite-value rules
    # after 5k without pretending an unobserved two-layer stream exists there.
    depth.check_histories(candidate['history'], candidate['history'])
    schedule = candidate['report']['contract']['learning_rate_schedule']
    full_steps = sorted({metric['update'] for metric in candidate['evaluations']
                        if metric['task'] == 'a5' and metric['role'] == 'ood_dev'
                        and metric['rows'] == 102400 and metric['update'] >= 5000})
    for step in sorted({5000, candidate['endpoint'], *full_steps}):
        lr.check_saved_lr(paired._packet(candidate, step), step, schedule)
    arms = {'two_layers': matched['arms']['two_layers'],
            'three_layers': paired.arm_summary('three_layers', candidate)}
    times = paired.cumulative_training_seconds(candidate['history'])
    return {'schema': SCHEMA, 'status': 'complete', 'qualification': QUALIFICATION,
        'endpoint': candidate['endpoint'], 'schedule': schedule, 'arms': arms,
        'matched_comparison': matched, 'matched_through_update': 5000,
        'matched_observations': matched['matched_observations'],
        'matched_training_seconds': matched['matched_training_seconds'],
        'common_full_updates': matched['common_full_updates'],
        'parameter_difference': matched['parameter_difference'], 'microbatch': matched['microbatch'],
        'continuation': {'start_update': 5000, 'endpoint': candidate['endpoint'],
            'updates': candidate['endpoint'] - 5000,
            'additional_training_seconds': times[candidate['endpoint']] - times[5000],
            'cumulative_training_seconds': times[candidate['endpoint']],
            'full_evaluation_updates': full_steps, 'paired_reference_available_after_5000': False},
        'training_bins': {'two_layers': lr.training_bins(base['history'], constant=False),
                          'three_layers': lr.training_bins(candidate['history'], constant=False)},
        'recovery': {'stages': len(candidate['lineage']),
            'restored_updates': [stage['start_update'] for stage in candidate['lineage'] if stage['start_update']],
            'discarded_tail_bytes': sum(stage['history']['discarded_tail_bytes'] for stage in candidate['lineage']),
            'boundary_full_state_exact': all(stage['boundary_state_audit']['passed']
                for stage in candidate['lineage'] if stage['start_update'])}}


def figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Unchanged matched figures use only the actual completed 5k runs.
    names = depth.figures(summary['matched_comparison'], output)
    colors = {'two_layers': '#222222', 'three_layers': '#137c8b'}
    labels = {'two_layers': '2 layers (ends at 5k)', 'three_layers': '3 layers (continued)'}
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{suffix}', dpi=150, bbox_inches='tight')
        plt.close(fig)
        names.append(name)
    panels = [('a5', 'dev', 'whole_word_exact_match', 'A5 length12 whole word'),
              ('a5', 'ood_dev', 'whole_word_exact_match', 'A5 length36 whole word'),
              ('a5', 'ood_dev', 'token_accuracy', 'A5 length36 token'),
              ('fuzzy', 'dev', 'answer_accuracy', 'Fuzzy answer tokens'),
              ('fuzzy', 'dev', 'first_value_token_accuracy', 'Fuzzy first value'),
              ('fuzzy', 'dev', 'sequence_exact_match', 'Fuzzy all-answer sequence')]
    for clock in ('updates', 'training-time'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
        for axis, (task, role, metric, title) in zip(axes.flat, panels):
            for label, arm in summary['arms'].items():
                rows = [row for row in arm['evaluations'] if row['task'] == task and row.get('role','dev') == role]
                x = [row['update'] if clock == 'updates' else row['cumulative_training_seconds']/3600 for row in rows]
                axis.plot(x, [100*row[metric] for row in rows], color=colors[label], label=labels[label])
                full = [index for index,row in enumerate(rows) if task == 'a5' and row['rows'] == 102400]
                axis.scatter([x[index] for index in full], [100*rows[index][metric] for index in full],
                             color=colors[label], s=14)
            if clock == 'updates':
                axis.axvline(5000, color='.5', ls=':')
                axis.axvspan(5000, summary['endpoint'], color='#137c8b', alpha=.06)
            axis.set(title=title, xlabel='Optimizer updates' if clock == 'updates' else 'Cumulative training hours',
                     ylabel='Accuracy (%)', ylim=(-2,102))
            axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle('Unequal-budget continuation context: only 3 layers train beyond 5k\n'
                     'Curves end at actual observations; A5 dots mark full 102,400-word evaluations')
        save(fig, 'continuation-context-' + clock)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    arm = summary['arms']['three_layers']
    selected = sorted({5000, summary['endpoint']})
    for step in selected:
        for axis, role in zip(axes, ('dev','ood_dev')):
            rows = [row for row in arm['evaluations'] if row['task'] == 'a5' and row['role'] == role
                    and row['update'] == step and row['rows'] == 102400]
            require(len(rows) == 1, 'Continuation prefix plot requires one full checkpoint observation')
            row = rows[0]
            axis.plot(range(1,row['length']+1), [100*x for x in row['cumulative_prefix_exactness']], label=f'3 layers at {step:,}')
            axis.set(title=f"A5 length{row['length']}", xlabel='Prefix length', ylabel='Whole prefix correct (%)', ylim=(-2,102))
            axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle('Three-layer continuation only: full-development prefix curves')
    save(fig, 'continuation-prefix')
    fig, axes = plt.subplots(2, 3, figsize=(14,8), layout='constrained')
    diagnostics = [('a5_ce','A5 state CE'), ('fuzzy_ce','Fuzzy dense-label CE'), ('grad_norm','Mean gradient norm before clipping'),
                   ('a5_latent','A5 NextLat loss'), ('fuzzy_latent','Fuzzy NextLat loss'), ('clip_fraction','Fraction of updates clipped at norm1')]
    for axis,(key,title) in zip(axes.flat,diagnostics):
        for label,rows in summary['training_bins'].items():
            axis.plot([row['update'] for row in rows], [row[key] for row in rows],label=labels[label],color=colors[label])
        axis.axvline(5000,color='.5',ls=':'); axis.set(title=title,xlabel='Optimizer updates')
        axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.suptitle('Training diagnostics: 100-update means; unchanged LR3e-4 after warmup\nOnly 3 layers continue beyond 5k')
    save(fig,'continuation-diagnostics')
    return names


def markdown(summary):
    endpoint = summary['endpoint']
    arm = summary['arms']['three_layers']
    lines = ['# Three-layer mixed A5/Fuzzy continuation', '',
        f'The three-layer model {arm["training_status"]} at **{endpoint:,} total optimizer updates**. '
        'Its recipe remains restricted RT → full RT → full RT, NextLat, FP32, 2,560 examples per task '
        'and LR3e-4 following the original 100-update warmup. Model, Adam, RNG and data streams resume exactly.', '',
        '**Matched architecture comparisons end at 5,000 updates.** The two-layer reference has no later observations. '
        'The continuation results below are additional-budget context.', '',
        '| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |',
        '|---:|---|---:|---:|---:|']
    for step in (1000,2500,5000):
        for label, reference in summary['matched_comparison']['arms'].items():
            if step not in summary['common_full_updates']: continue
            metrics = {f"{row['task']}/{row.get('role','dev')}":row for row in reference['evaluations'] if row['update']==step}
            a,b,f=[metrics[key] for key in ('a5/dev','a5/ood_dev','fuzzy/dev')]
            lines.append(f'| {step:,} | {depth.LABELS[label]} | {100*a["whole_word_exact_match"]:.4f}% | '
                f'{100*b["whole_word_exact_match"]:.4f}% | {100*f["answer_accuracy"]:.4f}% / '
                f'{100*f["first_value_token_accuracy"]:.4f}% / {100*f["sequence_exact_match"]:.4f}% |')
    lines += ['', '## Three-layer continuation only', '',
        '| Update | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |',
        '|---:|---:|---:|---:|']
    selected = sorted({step for step in (5000,6000,6500,7000,7500,endpoint) if step <= endpoint})
    for step in selected:
        metrics = {f"{row['task']}/{row.get('role','dev')}":row for row in arm['evaluations'] if row['update']==step}
        if not all(key in metrics for key in ('a5/dev','a5/ood_dev','fuzzy/dev')): continue
        a,b,f = [metrics[key] for key in ('a5/dev','a5/ood_dev','fuzzy/dev')]
        if a['rows'] != 102400 or b['rows'] != 102400: continue
        lines.append(f'| {step:,} | {100*a["whole_word_exact_match"]:.4f}% | {100*b["whole_word_exact_match"]:.4f}% | '
            f'{100*f["answer_accuracy"]:.4f}% / {100*f["first_value_token_accuracy"]:.4f}% / {100*f["sequence_exact_match"]:.4f}% |')
    cont = summary['continuation']
    lines += ['', f'Three-layer cumulative committed training time: **{cont["cumulative_training_seconds"]/3600:.3f} hours**; '
        f'additional training after 5k: **{cont["additional_training_seconds"]/60:.2f} minutes**. '
        'Evaluation, checkpointing, reporting, shutdown downtime and discarded work are excluded.', '',
        'Matched training time through 5k: '+', '.join(f'{depth.LABELS[name]} {seconds/3600:.3f} hours'
             for name,seconds in summary['matched_training_seconds'].items())+'.', '', QUALIFICATION, '',
        '## Figures', '', *[f'[{name}]({name}.pdf)' for name in summary['figures']]]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-train', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    import torch
    require(Path('/.dockerenv').is_file() and not torch.cuda.is_initialized(), 'Use the GPU-disabled project container')
    base, candidate = lr.read_fresh(args.baseline_train), load_run(args.train)
    summary = compare(base,candidate)
    args.output.mkdir(parents=True,exist_ok=False)
    summary['figures'] = figures(summary,args.output)
    (args.output/'report.md').write_text(markdown(summary))
    for name in ('scripts/rt_nextlat_a5_fuzzy_depth_extend_report.py','scripts/rt_nextlat_a5_fuzzy_depth_resume_report.py',
                 'scripts/rt_nextlat_a5_fuzzy_depth_report.py','scripts/rt_nextlat_a5_fuzzy_lr_report.py',
                 'scripts/rt_nextlat_a5_fuzzy_report.py','scripts/rt_nextlat_a5_fuzzy_embedding_compare.py'):
        target=args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    (args.output/'evidence.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker=OnlineTracker(project='rt-nextlat-fuzzy-a5',entity='taylorbollman',output_dir=args.output,
                              name='mixed-b2560-three-rt-layers-7500-continuation-reference5000')
        try:
            tracker.start({'schema':SCHEMA,'qualification':QUALIFICATION,'schedule':summary['schedule'],
                           'matched_through_update':5000,'continuation':summary['continuation']})
            tracker.log({f'report/{name}':wandb.Image(str(args.output/f'{name}.png')) for name in summary['figures']})
            tracker.summary({'candidate_endpoint':candidate['endpoint'],'candidate_final':summary['arms']['three_layers']['final'],
                             'matched_through_update':5000,'continuation':summary['continuation']})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False);raise
        summary['report_wandb']=tracker.record
    (args.output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','endpoint':candidate['endpoint'],'output':str(args.output)}))


if __name__ == '__main__':
    main()
