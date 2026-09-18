#!/usr/bin/env python3
"""Saved six-layer fixed-input20k continuation versus its exact six-layer baseline."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
from scripts import rt_a5_six_layer_input_report as base
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import ROLES, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / '.runtime/rt-a5/20260915T173000Z-six-layer-input001-20k'
PARENT = ROOT / '.runtime/rt-a5/20260915T171500Z-six-layer-input001-10k'
BASELINE_EXTENSION = ROOT / '.runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k'
SCHEMA = 'rt-a5-six-layer-input-extension-report-v1'
EXT_SCHEMA = 'rt-a5-six-layer-input-extension-protocol-v1'
COLUMNS = base.COLUMNS
require, bound, local_path = base.require, base.bound, base.local_path
REPORTING_SOURCES = ('scripts/rt_a5_six_layer_input_extension_report.py', *base.REPORTING_SOURCES)


def validate_extension(protocol, report, parent_protocol, *, schema, end_limit=20000):
    require(protocol['schema'] == schema and protocol['start_update'] == 10000 and protocol['endpoint'] == end_limit,
        'Wrong continuation protocol/budget')
    require(protocol['strict_contract'] == parent_protocol['strict_contract']
        and protocol['initialization'] == parent_protocol['initialization']
        and protocol['source_files'] == parent_protocol['source_files'], 'Continuation changes parent contract/initialization/source')
    require(report['contract'] == protocol['strict_contract'] and report['initialization'] == protocol['initialization']
        and report['source_files'] == protocol['source_files'] and report['start_update'] == 10000
        and report['endpoint'] == end_limit and report['parent_checkpoint']['sha256'] == protocol['parent_checkpoint']['sha256']
        and protocol['parent_checkpoint']['completed_updates'] == 10000 and report['wandb']['status'] == 'synced',
        'Continuation did not restore the declared10k parent')
    end = report['completed_updates']
    require(type(end) is int and 10000 < end <= end_limit, 'Actual continuation endpoint differs')
    require((report['status'] == 'complete' and end == end_limit)
        or (report['status'] == 'stopped' and end < end_limit), 'Unaccepted continuation state')
    if schema == EXT_SCHEMA:
        require(report['schema'] == base.TRAIN_SCHEMA and protocol['coefficient'] == .01
            and report['injection_coefficient'] == .01 and report['coefficient_learned'] is False
            and report['requested_endpoint_reached'] == (end == end_limit), 'Fixed input coefficient/endpoint differs')
    else:
        require(report['schema'] == 'rt-a5-l1r-depth-training-v1' and report['status'] == 'complete', 'Baseline extension must be completed')
    require(report['confirmation_evaluated'] is False and report['latent_rollout_evaluated'] is False, 'Unexpected confirmation/rollout')
    return end


def stitch_history(parent_rows, child_rows, end):
    require(len(parent_rows) == 10000 and [r['update'] for r in parent_rows] == list(range(1, 10001)), 'Parent history must contain1..10000')
    require(len(child_rows) == end - 10000 and [r['update'] for r in child_rows] == list(range(10001, end + 1)),
        'Child history has a missing/overlapping update')
    return parent_rows + child_rows


def common_checkpoint_steps(injection_curves, baseline_curves):
    return sorted(set(map(int, injection_curves)) & set(map(int, baseline_curves)))


def load_evidence(lineage=LINEAGE, through_update=None):
    require(through_update is None, 'This continuation report requires an actual terminal endpoint')
    lineage = local_path(lineage).resolve(); require(lineage == LINEAGE, 'Unexpected continuation lineage')
    parent_summary, parent_payloads = base.load_evidence(PARENT)
    require(parent_summary['through_update'] == 10000 and not parent_summary['partial'], 'Parent10k evidence is not closed')
    inputs = {'parent_' + key: value for key, value in parent_summary['inputs'].items()}
    payloads = {'parent_' + key: value for key, value in parent_payloads.items()}
    def capture(name, path):
        raw, record = read_input(path); inputs[name], payloads[name] = record, raw; return raw
    parent_current = json.loads(parent_payloads['training_report'])
    parent_original = json.loads(parent_payloads['baseline_report'])
    parent_original_protocol = json.loads(parent_payloads['baseline_protocol'])
    histories = {'injection': [json.loads(s) for s in parent_payloads['training_history'].splitlines()],
                 'baseline': [json.loads(s) for s in parent_payloads['baseline_history'].splitlines()]}
    extensions = {}
    for arm, run, parent_protocol, parent_report, schema in (
        ('injection', lineage, parent_summary['protocol'], parent_current, EXT_SCHEMA),
        ('baseline', BASELINE_EXTENSION, parent_original_protocol, parent_original, 'rt-a5-l1r-depth-extension-protocol-v1')):
        protocol = json.loads(capture(arm + '_protocol', run / 'protocol.json'))
        directory = local_path(protocol['training_directory']).resolve()
        require(directory == run / ('train-injection' if arm == 'injection' else 'train-depth'), 'Unexpected training directory')
        report = json.loads(capture(arm + '_report', directory / 'report.json'))
        end = validate_extension(protocol, report, parent_protocol, schema=schema)
        config = json.loads(capture(arm + '_config', directory / 'config.json'))
        require(config == protocol['resolved_args'], 'Executed continuation config differs')
        require(capture(arm + '_exit', run / 'training-exit-code.txt').strip() == b'0', 'Continuation process failed')
        launch = json.loads(capture(arm + '_launch', run / 'launch-status.json'))
        require(launch['status'] == report['status'], 'Launcher did not accept continuation')
        bound(protocol['parent_protocol']); bound(protocol['parent_report'])
        expected_parent_key = 'protocol' if arm == 'injection' else 'baseline_protocol'
        expected_report_key = 'training_report' if arm == 'injection' else 'baseline_report'
        require(protocol['parent_protocol']['sha256'] == parent_summary['inputs'][expected_parent_key]['sha256']
            and protocol['parent_report']['sha256'] == parent_summary['inputs'][expected_report_key]['sha256'], 'Wrong parent evidence')
        resume = bound(protocol['parent_checkpoint'])
        prior_checkpoint = next(c for c in parent_report['checkpoints'] if c['completed_updates'] == 10000)
        require(resume['sha256'] == prior_checkpoint['sha256'], 'Continuation restored a different10k checkpoint')
        if report['status'] == 'stopped':
            stop = report['stop_request']; stop_path = local_path(stop['path']).resolve()
            require(stop['reason'] == 'user_stop_file' and stop['observed_after_update'] == end
                and stop_path == local_path(config['stop_file']).resolve(), 'Invalid earlier stop')
            capture(arm + '_stop', stop_path); require(hash_file(stop_path)['sha256'] == stop['sha256'], 'Stop sentinel changed')
        sources = protocol['source_files']
        require(base._digest_dict(sources) == protocol['source_sha256'] == report['contract']['source_sha256'], 'Source identity differs')
        for name, digest in sources.items():
            require(hash_file(ROOT / name)['sha256'] == hash_file(directory / 'source' / name)['sha256'] == digest, 'Source/current snapshot changed')
        child_rows = [json.loads(s) for s in capture(arm + '_history', directory / 'history.jsonl').splitlines()]
        histories[arm] = stitch_history(histories[arm], child_rows, end)
        require(histories[arm][-1]['order_chain'] == report['order_chain'], 'History/report order differs')
        steps = sorted(set(s for s in protocol['checkpoint_steps'] if s <= end) | {end})
        require([c['completed_updates'] for c in report['checkpoints']] == steps, 'Continuation checkpoints differ')
        checkpoints = {str(c['completed_updates']): bound(c, directory / f"checkpoints/step-{c['completed_updates']:06d}.pt") for c in report['checkpoints']}
        state = json.loads(capture(arm + '_state', run / 'final-state-validation.json'))
        expected_state = 'rt-a5-six-layer-input-extension-final-state-validation-v1' if arm == 'injection' else 'rt-a5-l1r-depth-extension-final-state-validation-v1'
        require(state['schema'] == expected_state and state['passed'] is True and state['completed_updates'] == end
            and state['model_parameter_tensors'] == state['final_active_adam_states'] == (62 if arm == 'injection' else 61)
            and state['protocol']['sha256'] == inputs[arm + '_protocol']['sha256']
            and state['report']['sha256'] == inputs[arm + '_report']['sha256']
            and state['source_sha256'] == protocol['source_sha256'], 'Bound saved-state proof differs')
        require(all(state['checkpoint_inputs'][step]['sha256'] == record['sha256'] for step, record in checkpoints.items()), 'Checkpoint proof differs')
        if arm == 'injection':
            require(state['parent_checkpoint']['sha256'] == resume['sha256']
                and state['all61_baseline_initial_tensors_bitwise_equal'] is True
                and state['shared_six_layer_model_sha256'] == parent_summary['protocol']['initialization']['shared_six_layer_model_sha256'], 'Paired initialization/resume proof differs')
        curves, metrics, rows, seen = {}, {}, [], set()
        for evaluation in report['evaluations']:
            step, role = evaluation['update'], evaluation['role']
            require((step, role) not in seen and 10000 < step <= end and role in ROLES
                and evaluation['rows'] == (102400 if step in steps else 4096)
                and evaluation['route'] == 'backbone_only', 'Unexpected continuation evaluation')
            seen.add((step, role))
            if arm == 'injection': require(evaluation['injection_coefficient'] == .01, 'Evaluation coefficient differs')
            converted = metric_rows(evaluation, 'six_layer_input' if arm == 'injection' else 'baseline')
            for row in converted: row['injection_coefficient'] = .01 if arm == 'injection' else 0.
            rows.extend(converted)
            if step in steps:
                curves.setdefault(str(step), {})[role] = converted
                metrics.setdefault(str(step), {})[role] = evaluation
        expected_evaluations = set(range(10500, end + 1, 500)) | set(steps)
        require(seen == {(step, role) for step in expected_evaluations for role in ROLES}, 'Missing full/routine continuation evaluation')
        extensions[arm] = {'protocol': protocol, 'report': report, 'state': state, 'completed_updates': end,
            'checkpoint_updates': steps, 'checkpoints': checkpoints, 'curves': curves, 'metrics': metrics, 'metric_rows': rows}
    reference_extension = extensions['injection']['protocol']['reference_extension']
    for key in ('protocol', 'report', 'config', 'history'):
        require(bound(reference_extension[key])['sha256'] == inputs['baseline_' + key]['sha256'], 'Different bound baseline extension evidence')
    for step, binding in reference_extension['checkpoints'].items():
        require(bound(binding)['sha256'] == extensions['baseline']['checkpoints'][step]['sha256'], 'Different bound baseline extension checkpoint')
    end = extensions['injection']['completed_updates']
    base.validate_history(histories['injection'], end, histories['baseline'], exact=True)
    # The original baseline has no injection instrumentation; add a local constant only for shared loss/order validation.
    base.validate_history([{**r, 'injection_coefficient': .01} for r in histories['baseline']], 20000,
        histories['baseline'], exact=True)
    baseline = copy.deepcopy(parent_summary['baseline']); be = extensions['baseline']
    baseline.update(completed_updates=20000, saved_state=be['state'], wandb=be['report']['wandb'])
    for key in ('curves', 'checkpoints'): baseline[key].update(be[key])
    baseline['checkpoint_updates'] += be['checkpoint_updates']; baseline['metric_rows'] += be['metric_rows']
    baseline['training_curve'] = training_curve(histories['baseline'], augmented=True)
    current_curves = {**parent_summary['curves'], **extensions['injection']['curves']}
    common = common_checkpoint_steps(current_curves, baseline['curves'])
    comparisons = []
    for step in common:
        old, new = baseline['curves'][str(step)]['ood_dev'][-1], current_curves[str(step)]['ood_dev'][-1]
        comparisons.append({'update': step, 'words': 102400,
            'baseline': {'coefficient': 0., **{k: old[k] for k in ('E', 'A', 'M')}},
            'injection': {'coefficient': .01, **{k: new[k] for k in ('E', 'A', 'M')}}})
    baseline.update(common_checkpoint_updates=common, comparisons=comparisons)
    current = extensions['injection']; curve = training_curve(histories['injection'], augmented=True)
    for row in curve: row['injection_coefficient'] = .01
    rows = [row for row in parent_summary['metric_rows'] if row['arm'] == 'six_layer_input'] + current['metric_rows'] + baseline['metric_rows']
    summary = {**parent_summary, 'schema': SCHEMA, 'report_scope': 'terminal exact10k-to20k continuation',
        'through_update': end, 'requested_endpoint': 20000, 'requested_endpoint_reached': current['report']['requested_endpoint_reached'],
        'training_status_at_snapshot': current['report']['status'], 'partial': False, 'inputs': inputs,
        'protocol': current['protocol'], 'saved_state': current['state'], 'parent_protocol': parent_summary['protocol'],
        'extensions': extensions, 'baseline': baseline, 'common_checkpoint_updates': common, 'matched_update': max(common),
        'actual_endpoints': {'injection': end, 'baseline': 20000}, 'curves': current_curves,
        'metrics': {**parent_summary['metrics'], **current['metrics']},
        'checkpoints': {**parent_summary['checkpoints'], **current['checkpoints']},
        'checkpoint_updates': parent_summary['checkpoint_updates'] + current['checkpoint_updates'],
        'metric_rows': rows, 'training_curve': curve, 'matched_minibatch_order_hashes': end,
        'observed_complete_history_rows': end, 'excluded_later_history_rows': 0,
        'training_wandb': current['report']['wandb'],
        'qualification': 'One seed and repeatedly inspected development pools. The user extended the current six-layer input run while it was training. This continues the exact10k model/Adam/RNG/data order at fixedlambda0.01 through at most20k. All61 original baseline initial tensors are shared; Pe adds262144 parameters. The baseline also continues its own exact10k state to20k. Comparisons use shared full102400-word checkpoints; a user-selected earlier stop is retrospective and its unmatched terminal is shown separately. No new inference, confirmation or autonomous latent rollout is included.'}
    return summary, payloads


def plot_rows(summary, step, arm='injection'):
    return base.plot_rows(summary, step, arm)


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figures = []
    labels = {"baseline": "Original six-layer baseline", "injection": "Six layers + input injection λ=0.01"}
    colors = {"baseline": "#3574B2", "injection": "#D65B35"}
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        figures.append(name)
        plt.close(fig)
    def prefix_figure(name, limits, selected, title):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.78, wspace=.4)
        for axis, key, label in zip(axes, ("E", "A", "M"), ("All states through t correct", "Only state t correct", "Mean token accuracy through t")):
            for arm, step in selected:
                rows = plot_rows(summary, step, arm)
                axis.plot([r["length"] for r in rows], [r[key] for r in rows], color=colors[arm],
                          label=f"{labels[arm]} ({step:,})")
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of the same 36-token words", title=f"{key}(t): {label}")
            axis.axvline(12, color="gray", linestyle=":")
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle(title + " · 102,400 words", y=.97)
        save(fig, name)
    end, matched = summary["through_update"], summary["matched_update"]
    baseline_end = summary['baseline']['completed_updates']
    selected = [("baseline", matched), ("injection", matched)] if matched is not None else [("injection", end)]
    title = f"Same training budget: {matched:,} updates" if matched is not None else "Injection endpoint; no shared retained checkpoint yet"
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        prefix_figure(name, limits, selected, title)
    if end != baseline_end:
        prefix_figure("length-unequal-terminals", (1, 36), [("baseline", baseline_end), ("injection", end)],
                      f"Unequal terminal budgets: baseline {baseline_end:,} versus injection {end:,}; descriptive only")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), layout="constrained")
    for axis, key in zip(axes, ("E", "A", "M")):
        for arm in labels:
            view = summary if arm == "injection" else summary["baseline"]
            steps = view["checkpoint_updates"][1:]
            axis.plot(steps, [plot_rows(summary, step, arm)[-1][key] for step in steps], marker="o", color=colors[arm], label=labels[arm])
        axis.set(xlabel="Completed training updates", ylabel=f"{key}(36)", ylim=(-.025, 1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle("Full 102,400-word development evaluations at each arm's retained checkpoints")
    save(fig, "length36-vs-updates")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    for axis, key, title in zip(axes, ("state_ce", "latent_loss"), ("State cross entropy", "NextLat SmoothL1")):
        for arm in labels:
            view = summary if arm == "injection" else summary["baseline"]
            bins = view["training_curve"]
            axis.plot([b["update"] for b in bins], [b[key] for b in bins], color=colors[arm], label=labels[arm])
        axis.set(xlabel="Completed training updates", ylabel="Loss", title=title + ", 100-update means")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    save(fig, "training-losses")
    return figures




def run(args):
    summary, payloads = load_evidence(args.lineage, args.through_update)
    output = local_path(args.output_dir).resolve()
    require(not output.exists(), "Use a new report output directory")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for name, raw in payloads.items():
        suffix = Path(summary["inputs"][name]["path"]).suffix
        (output / "inputs" / f"{name}{suffix}").write_bytes(raw)
    training_directory = local_path(summary["protocol"]["training_directory"])
    for relative in summary["protocol"]["source_files"]:
        destination = output / "training-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_directory / "source" / relative, destination)
    for relative in REPORTING_SOURCES:
        destination = output / "reporting-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    write_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS); writer.writeheader(); writer.writerows(summary["metric_rows"])
    with (output / "training-curve.csv").open("w") as stream:
        bins = [{**row, "arm": "injection", "injection_coefficient": .01} for row in summary["training_curve"]]
        bins += [{**row, "arm": "baseline", "injection_coefficient": 0.} for row in summary["baseline"]["training_curve"]]
        columns = sorted({key for row in bins for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(bins)
    figures = plots(summary, output)
    (output / "README.md").write_text(markdown(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
        group=args.wandb_group or local_path(args.lineage).name,
        name=f"six-layer-input-extension-report-through-{summary['through_update']:06d}")
    result = {"schema": SCHEMA, "status": "running", "through_update": summary["through_update"],
              "partial": summary["partial"], "matched_update": summary["matched_update"],
              "actual_endpoints": summary["actual_endpoints"], "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "injection_coefficient": .01, "through_update": summary["through_update"],
                       "partial": summary["partial"], "qualification": summary["qualification"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(COLUMNS), data=[[row[k] for k in COLUMNS] for row in summary["metric_rows"]]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for row in summary["training_curve"]:
            tracker.log({"update": row["update"], "injection/coefficient": row["injection_coefficient"],
                         **{f"train/{key}": row[key] for key in ("state_ce", "latent_loss", "loss")}})
        for step in summary["checkpoint_updates"][1:]:
            row = plot_rows(summary, step)[-1]
            tracker.log({"update": step, **{f"dev/ood_dev/{key}36": row[key] for key in ("E", "A", "M")}})
        tracker.summary({"through_update": summary["through_update"], "partial": summary["partial"],
                         "injection_coefficient": summary["injection_coefficient"]})
        tracker.finish(succeeded=True); result["status"] = "complete"
        (output / "README.md").write_text(markdown(summary) + f"\n[Report W&B]({tracker.record['run_url']})\n")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(path.relative_to(output)): hash_file(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and "wandb" not in path.relative_to(output).parts and path != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}))
    return result


def markdown(summary):
    end = summary['through_update']; matched = summary['matched_update']
    lines = [f'# Six-layer fixed input injection through {end:,} updates', '',
        f'Exact continuation from 10,000 to {end:,}; requested endpoint 20,000. Fixed λ=0.01. Latest matched full checkpoint: {matched:,}.', '',
        'The first layer uses window-2 attention and the next five use full recurrent attention. Only the second block input receives 0.01 × Pe(raw token embedding), before its existing normalization. NextLat is unchanged.', '',
        'E(t): every state through t correct; A(t): state t alone correct; M(t): mean token accuracy through t. Full and boundary plots use identical36-prefix rows.', '',
        '| Update | Arm | E(36) | A(36) | M(36) |', '|---:|---|---:|---:|---:|']
    for point in summary['baseline']['comparisons']:
        for arm in ('baseline', 'injection'):
            values = point[arm]
            lines.append(f"| {point['update']:,} | {arm} | {values['E']:.4%} | {values['A']:.4%} | {values['M']:.4%} |")
    if end != 20000:
        lines += ['', f'Actual endpoints are injection {end:,} and baseline 20,000: unequal budgets. [Descriptive terminal curves](length-unequal-terminals.pdf).']
    lines += ['', summary['qualification'], '',
        '[Matched full curve](length-full.pdf) · [Matched boundary](length-boundary.pdf) · [Length36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)', '',
        f"[Injection continuation W&B]({summary['training_wandb']['run_url']}) · [Baseline continuation W&B]({summary['extensions']['baseline']['report']['wandb']['run_url']})", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lineage', default=str(LINEAGE))
    parser.add_argument('--output-dir', default=str(ROOT / 'docs/reports/rt-a5/six-layer-input001-20k'))
    parser.add_argument('--wandb-group')
    args = parser.parse_args(); args.through_update = None; run(args)


if __name__ == '__main__': main()
