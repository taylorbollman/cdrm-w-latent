#!/usr/bin/env python3
"""Compare completed, paired mixed-task embedding pilots using retained evidence.

No model inference is performed. The first --run is the uninjected baseline;
later arms may be added as they finish. Training time is summed from individual
committed updates across checkpoint continuations, never from a restarted clock.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from scripts import rt_nextlat_a5_fuzzy_report as saved


SCHEMA = "rt-nextlat-a5-fuzzy-embedding-comparison-v1"
PROJECTION = "backbone.embedding_projection.weight"
QUALIFICATION = (
    "One initialization, fixed reused development pools, and no final confirmation. "
    "These are directional architecture comparisons, not replicated effects. "
    "All training uses NextLat; evaluation uses the backbone without latent rollout. "
    "A5 monitoring subsets and full evaluations have different sample sizes. "
    "Threshold times are first observed crossings, not sustained success or the exact onset of learning. "
    "Near-ceiling Fuzzy answer accuracy can conceal differences; motif, sequence, first-value and terminal "
    "metrics are shown without claiming that sequence exactness removes the underlying task ceiling. "
    "Training seconds exclude evaluation, checkpointing and reporting, and include initialization-period "
    "training updates; sequential runs may also differ because of system conditions."
)


def require(condition, message):
    saved.require(condition, message)


def parse_run(value):
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use --run label=training-directory")
    return label.strip(), saved.local_path(path)


def cumulative_training_seconds(history):
    """Stitched histories already contain one row per committed optimizer update."""
    result, total = {0: 0.0}, 0.0
    for expected, row in enumerate(history, 1):
        seconds = row.get("seconds")
        require(row.get("update") == expected, "Training-time history has a gap or duplicate")
        require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                and math.isfinite(seconds) and seconds > 0, "Invalid per-update training seconds")
        total += seconds
        result[expected] = total
    return result


def mechanism(config):
    injection = config.get("embedding_injection")
    if injection is None:
        return {"variant": "baseline", "description": "No embedding injection", "added_parameters": 0}
    width = config["backbone"]["d_model"]
    require(injection.get("layer_index") == 1, "Only the upper block may receive embedding injection")
    variant = injection.get("variant")
    if variant in ("input", "value"):
        require(injection == {"variant": variant, "layer_index": 1,
                              "coefficient": 0.01, "projection_seed": 1237},
                "Input/value arm must use the approved constant 0.01 projection")
        description = ("Upper input u + 0.01 P_e e" if variant == "input" else
                       "Upper permanent values W_V h + 0.01 P_e e; keys unchanged")
        return {**injection, "description": description, "added_parameters": width * width}
    last_head = config["backbone"]["n_heads"] - 1
    require(injection == {"variant": "head", "layer_index": 1, "head_index": last_head, "enabled": True},
            "Head arm must reassign only the existing upper last head")
    return {**injection, "description": f"Upper head {last_head} stores embedding-derived values; contextual keys/query, unchanged temporary values",
            "added_parameters": 0, "head_width": width // config["backbone"]["n_heads"]}


def _packet(run, update):
    import torch
    require(update in run["checkpoints"], "Required audit checkpoint is missing")
    record = run["checkpoints"][update]
    path = Path(record["verified_local_path"])
    require(saved.sha(path) == record["sha256"], "Audit checkpoint bytes changed")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    contract = run["report"]["contract"]
    require(packet.get("completed_updates") == update
            and packet.get("contract") == contract
            and packet.get("initialization") == run["report"]["initialization"],
            "Checkpoint contract, initialization or update differs")
    expected_seen = {task: update * contract["batch_per_task"] for task in run["tasks"]}
    expected_cursors = {task: {"absolute_example_offset": n,
                              "epoch": n // contract["streams"][task]["train_rows"],
                              "position": n % contract["streams"][task]["train_rows"]}
                        for task, n in expected_seen.items()}
    require(packet.get("examples_seen") == expected_seen and packet.get("next_cursors") == expected_cursors,
            "Checkpoint task exposure or stream cursors differ")
    if update:
        require(packet.get("order_chains") == run["history"][update - 1]["order_chains"],
                "Checkpoint data order differs from committed history")
    require(all(torch.isfinite(value).all().item() for value in packet["model"].values()),
            "Nonfinite model checkpoint")
    return packet


def verify_initial_tensors(base, other, variant):
    import torch
    left, right = _packet(base, 0), _packet(other, 0)
    a, b = left["model"], right["model"]
    allowed = {PROJECTION} if variant["variant"] in ("input", "value") else set()
    require(set(a) <= set(b) and set(b) - set(a) == allowed,
            "Variant initial state has unapproved extra or missing tensors")
    require(all(a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
                and torch.equal(a[name], b[name]) for name in a),
            "Shared initial tensors differ")
    require(left["order_chains"] == right["order_chains"], "Initial data order chains differ")
    require(not left["optimizer"]["state"] and not right["optimizer"]["state"],
            "Fresh comparison cannot inherit optimizer state")
    if allowed:
        require(b[PROJECTION].shape == (base["report"]["contract"]["model_config"]["backbone"]["d_model"],) * 2,
                "Embedding projection shape differs")
    return {"passed": True, "shared_tensor_count": len(a), "shared_tensors_exact": True,
            "extra_tensors": sorted(allowed), "empty_initial_optimizer_states": True,
            "baseline_checkpoint": base["checkpoints"][0], "variant_checkpoint": other["checkpoints"][0]}


def compatibility(base, other):
    """Permit only the explicit architecture addition and its additive source closure."""
    from cdrm.rt_nextlat_task_embeddings import SOURCE_PATHS
    a, b = base["report"]["contract"], other["report"]["contract"]
    require(a.get("mode") == b.get("mode") == "mixed" and base["tasks"] == other["tasks"] == ["a5", "fuzzy"],
            "Compare mixed A5/Fuzzy arms only")
    require("embedding_injection" not in a["model_config"], "First arm must be the uninjected baseline")
    variant = mechanism(b["model_config"])
    require(variant["variant"] != "baseline", "Later arms must be embedding variants")
    plain = copy.deepcopy(b["model_config"])
    plain.pop("embedding_injection")
    require(plain == a["model_config"], "Non-injection model configuration differs")
    ignored = {"model_config", "configuration_file_sha256", "source_sha256", "initialization"}
    require({k: v for k, v in a.items() if k not in ignored}
            == {k: v for k, v in b.items() if k not in ignored},
            "Training contract differs beyond the approved embedding variant")
    require(other["report"]["initialization"].get("baseline_initialization") == base["report"]["initialization"],
            "Canonical baseline initialization metadata differs")
    require(base["data_identity"] == other["data_identity"], "Dataset identities differ")
    extra_sources = set(other["sources"]) - set(base["sources"])
    allowed_sources = set(SOURCE_PATHS) | {"scripts/rt_nextlat_a5_fuzzy_embedding_train.py"}
    require(set(base["sources"]) <= set(other["sources"])
            and all(other["sources"][name] == digest for name, digest in base["sources"].items()),
            "Frozen baseline source closure changed")
    require(extra_sources <= allowed_sources, "Unexpected additional source dependency")
    require("cdrm/rt_nextlat_task_embeddings.py" in extra_sources, "Missing frozen embedding adapter")
    delta = other["report"]["parameter_count"] - base["report"]["parameter_count"]
    require(delta == variant["added_parameters"], "Unapproved parameter-count difference")
    through = min(base["endpoint"], other["endpoint"])
    for left, right in zip(base["history"], other["history"]):
        require(left["order_chains"] == right["order_chains"],
                f"Task data order differs at update {left['update']}")
        expected = {task: left["update"] * a["batch_per_task"] for task in base["tasks"]}
        require(left.get("examples_seen") == right.get("examples_seen") == expected,
                f"Task exposure differs at update {left['update']}")
    initialization = verify_initial_tensors(base, other, variant)
    full_a = {m["update"] for m in saved.selected_evaluations(base["evaluations"])
              if m["task"] == "a5" and m["role"] == "ood_dev" and m["rows"] == 102400}
    full_b = {m["update"] for m in saved.selected_evaluations(other["evaluations"])
              if m["task"] == "a5" and m["role"] == "ood_dev" and m["rows"] == 102400}
    milestones = sorted(full_a & full_b - {0})
    for update in milestones:
        left, right = _packet(base, update), _packet(other, update)
        require(left["order_chains"] == right["order_chains"] and left["next_cursors"] == right["next_cursors"],
                "Matched milestone checkpoint data streams differ")
    # Audit each actual terminal checkpoint even if one arm was explicitly stopped early.
    _packet(base, base["endpoint"])
    _packet(other, other["endpoint"])
    return {"passed": True, "variant": variant, "initialization": initialization,
            "matched_data_order_updates": through, "audited_full_milestone_updates": milestones,
            "additional_frozen_sources": sorted(extra_sources), "parameter_difference": delta}


def observed_crossings(evaluations, times):
    specifications = (("a5", "ood_dev", "whole_word_exact_match", "a5_l36_whole_word"),
                      ("fuzzy", "dev", "answer_accuracy", "fuzzy_answer"))
    result = {}
    for task, role, key, name in specifications:
        rows = [m for m in evaluations if m["task"] == task and m.get("role", "dev") == role
                and m["update"] > 0 and (task != "a5" or m["rows"] == 102400)]
        rows.sort(key=lambda row: row["update"])
        thresholds = ("positive", .1, .5, .9, .99) if task == "a5" else (.5, .9, .99)
        result[name] = {}
        for threshold in thresholds:
            first = next((m for m in rows if (m["whole_word_correct"] > 0 if threshold == "positive"
                                            else m[key] >= threshold)), None)
            result[name][str(threshold)] = (None if first is None else {
                "update": first["update"], "cumulative_training_seconds": times[first["update"]],
                "accuracy": first[key], "evaluated_rows": first.get("rows", first.get("examples")),
                "checkpoint_sha256": first["checkpoint"]["sha256"]})
    return result


def arm_summary(label, run):
    times = cumulative_training_seconds(run["history"])
    selected = saved.selected_evaluations(run["evaluations"])
    observations = [{**m, "cumulative_training_seconds": times[m["update"]]} for m in selected]
    contract = run["report"]["contract"]
    return {"label": label, "directory": run["directory"], "endpoint": run["endpoint"],
            "training_status": run["report"]["status"], "requested_endpoint": run["report"]["requested_endpoint"],
            "mechanism": mechanism(contract["model_config"]), "parameters": run["report"]["parameter_count"],
            "contract": contract, "checkpoint": run["checkpoints"][run["endpoint"]],
            "final": saved.endpoint_metrics(run), "evaluations": observations,
            "cumulative_training_seconds": times[run["endpoint"]],
            "examples_per_task": run["endpoint"] * contract["batch_per_task"],
            "observed_threshold_crossings": observed_crossings(selected, times),
            "lineage": run["lineage"], "input_hashes": run["input_hashes"],
            "training_wandb": run["report"].get("wandb")}


def matched_observations(base, other):
    """No endpoint interpolation and no comparison of full pools to subsets."""
    def indexed(arm):
        return {(m["update"], m["task"], m.get("role", "dev"), m.get("rows", m.get("examples"))): m
                for m in arm["evaluations"]}
    a, b = indexed(base), indexed(other)
    return [{"update": key[0], "task": key[1], "role": key[2], "evaluated_rows": key[3],
             "baseline": a[key], "variant": b[key]} for key in sorted(a.keys() & b.keys())]


def make_plots(arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=155)
        plt.close(fig)
        figures.append(name)
    colors = dict(zip(arms, ("#202020", "#d95f02", "#1b9e77", "#7570b3")))
    def draw(axis, arm, task, role, key, clock):
        rows = [m for m in arm["evaluations"] if m["task"] == task and m.get("role", "dev") == role]
        x = lambda row: row["update"] if clock == "updates" else row["cumulative_training_seconds"] / 3600
        y = lambda row: row[key] if isinstance(key, str) else row["cumulative_prefix_exactness"][key - 1]
        color = colors[arm["label"]]
        axis.plot([x(m) for m in rows], [100*y(m) for m in rows], color=color, label=arm["label"])
        if task == "a5":
            for full, marker in ((True, "o"), (False, "x")):
                chosen = [m for m in rows if (m["rows"] == 102400) == full]
                axis.scatter([x(m) for m in chosen], [100*y(m) for m in chosen], s=14, marker=marker, color=color)
    for clock in ("updates", "training-time"):
        xlabel = "Optimizer updates" if clock == "updates" else "Cumulative training hours (all continuation stages)"
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for row, role in enumerate(("dev", "ood_dev")):
            for axis, key, title in zip(axes[row], ("token_accuracy", "final_state_accuracy", "whole_word_exact_match"),
                                        ("Token", "Final-state", "Whole-word")):
                for arm in arms.values():
                    draw(axis, arm, "a5", role, key, clock)
                axis.set(title=f"A5 L{12 if role == 'dev' else 36}: {title}", xlabel=xlabel, ylabel="Accuracy (%)", ylim=(-2, 102))
                axis.legend(fontsize=7)
                axis.grid(alpha=.2)
        fig.suptitle("One seed, development only — circles full 102,400; crosses monitoring subset")
        save(fig, f"a5-learning-{clock}")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        fuzzy_keys = (("answer_accuracy", "Answer tokens"), ("answer_motif_exact_match", "Answer motifs"),
                      ("sequence_exact_match", "All answers in sequence"), ("first_value_token_accuracy", "First value token"),
                      ("terminal_probe_accuracy", "Terminal probe"))
        for axis, (key, title) in zip(axes.flat, fuzzy_keys):
            for arm in arms.values():
                draw(axis, arm, "fuzzy", "dev", key, clock)
            axis.set(title=f"Fuzzy: {title}", xlabel=xlabel, ylabel="Accuracy / exactness (%)", ylim=(-2, 102))
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
        axes.flat[-1].axis("off")
        fig.suptitle("1,280 shared Fuzzy development examples — teacher-forced answers")
        save(fig, f"fuzzy-learning-{clock}")
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
        for axis, position in zip(axes.flat, (12, 13, 14, 15, 16, 18, 24, 36)):
            for arm in arms.values():
                draw(axis, arm, "a5", "ood_dev", position, clock)
            axis.set(title=f"L36 cumulative E({position})", xlabel=xlabel, ylabel="Exact prefixes (%)", ylim=(-2, 102))
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
        fig.suptitle("Prefixes of the same L36 words; circle/full vs cross/subset evaluation")
        save(fig, f"a5-prefix-learning-{clock}")
    for name, limits in (("a5-endpoint-prefix-full", (1, 36)), ("a5-endpoint-prefix-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for label, arm in arms.items():
            metric = arm["final"]["a5/ood_dev"]
            state = metric["isolated_state_accuracy"]
            for axis, curve in zip(axes, (metric["cumulative_prefix_exactness"], state,
                                          [sum(state[:n])/n for n in range(1, len(state)+1)])):
                axis.plot(range(1, len(curve)+1), [100*v for v in curve], color=colors[label],
                          label=f"{label} @ {arm['endpoint']:,}")
        for axis, title in zip(axes, ("E(t): exact through t", "A(t): state at t", "M(t): mean through t")):
            axis.set(title=title, xlim=limits, ylim=(-2, 102), xlabel="Position in L36 word", ylabel="Accuracy (%)")
            axis.axvline(12, color=".6", ls=":")
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle("Same endpoint source rows in full and boundary views; unequal endpoints are descriptive")
        save(fig, name)
    return figures


def markdown(summary):
    lines = ["# Mixed A5 + Fuzzy embedding comparison", "", QUALIFICATION, "",
             "Each arm starts from the same original backbone and NextLat predictor tensors, with the same "
             "ordered examples per update. No arm is warm-started from the trained baseline. "
             "Only upper block index 1 changes; the first block retains its two-position window.", "",
             "| Arm | Upper-block change | Parameters | Actual updates | Train hours |", "| --- | --- | ---: | ---: | ---: |"]
    for label, arm in summary["arms"].items():
        lines.append(f"| {label} | {arm['mechanism']['description']} | {arm['parameters']:,} | {arm['endpoint']:,} | {arm['cumulative_training_seconds']/3600:.3f} |")
    lines += ["", "Input and value variants use a fixed coefficient of 0.01 and a learned D×D projection. "
              "The head variant reassigns one existing 8-dimensional head, adding no parameters or gate. "
              "Its gain is not directly matched to the 0.01 additive mechanisms.", "",
              "## Actual endpoint results", "", "All metrics in one arm's row belong to its same saved checkpoint. "
              "Different terminal update counts are descriptive, not matched-budget comparisons.", "",
              "| Arm | A5 L12 token / whole | A5 L36 token / whole | Fuzzy answer / motif / sequence | Fuzzy first value / terminal |",
              "| --- | ---: | ---: | ---: | ---: |"]
    percent = lambda value: f"{100*value:.4f}%"
    for label, arm in summary["arms"].items():
        a, b, f = [arm["final"][key] for key in ("a5/dev", "a5/ood_dev", "fuzzy/dev")]
        lines.append(f"| {label} | {percent(a['token_accuracy'])} / {percent(a['whole_word_exact_match'])} | "
                     f"{percent(b['token_accuracy'])} / {percent(b['whole_word_exact_match'])} | "
                     f"{percent(f['answer_accuracy'])} / {percent(f['answer_motif_exact_match'])} / {percent(f['sequence_exact_match'])} | "
                     f"{percent(f['first_value_token_accuracy'])} / {percent(f['terminal_probe_accuracy'])} |")
    lines += ["", "## First observed threshold crossings", "",
              "Only full evaluations establish the crossings below. No interpolation or best-checkpoint selection is used. "
              "Full A5 evaluations need not have identical schedules across resumed and fresh runs; use the matched "
              "observations in evidence.json for equal-update, equal-pool comparisons.", "",
              "| Arm | Metric | Threshold | First observed update | Training hours |", "| --- | --- | --- | ---: | ---: |"]
    for label, arm in summary["arms"].items():
        for key, thresholds in arm["observed_threshold_crossings"].items():
            for threshold, observation in thresholds.items():
                label_threshold = threshold if threshold == "positive" else f"{100*float(threshold):g}%"
                update = "Not observed" if observation is None else f"{observation['update']:,}"
                seconds = "—" if observation is None else f"{observation['cumulative_training_seconds']/3600:.3f}"
                lines.append(f"| {label} | {key} | {label_threshold} | {update} | {seconds} |")
    lines += ["", "## Evidence and interpretation", "",
              "Training time sums every committed per-update duration through all baseline continuation stages, "
              "including the original 0–2.5k and 2.5k–5k portions. It excludes eval/report/storage overhead. "
              "Accuracy-versus-time plots therefore compare measured training efficiency; accuracy-versus-update "
              "plots compare equal per-task data exposure. Neither axis alone establishes a general architectural advantage.", "",
              "Compatibility checks require unchanged original source files, task data, optimization, precision and "
              "loss settings; exact shared initial tensors; empty initial optimizer state; identical per-update "
              "data-order chains; and checkpoint-bound matching exposure/cursors at common full milestones. "
              "The input/value projection increases parameter count by 16,384 (about 3.4%). "
              "No automatic winner is chosen from these single-seed, potentially fluctuating trajectories.", ""]
    for label, arm in summary["arms"].items():
        link = (arm.get("training_wandb") or {}).get("run_url")
        lines.append(f"- {label}: `{arm['directory']}`; endpoint checkpoint `{arm['checkpoint']['sha256']}`."
                     + (f" [W&B training]({link})." if link else ""))
    lines += ["", "## Figures", ""]
    for name in summary["figures"]:
        lines += [f"![{name}]({name}.png)", f"[PDF]({name}.pdf)", ""]
    return "\n".join(lines)


def build_report(args):
    require(2 <= len(args.run) <= 4, "Provide the baseline and one to three completed variants")
    labels = [label for label, _ in args.run]
    require(len(set(labels)) == len(labels), "Duplicate arm labels")
    runs = {label: saved.load_run(directory) for label, directory in args.run}
    base = runs[labels[0]]
    require(base["report"]["contract"]["batch_per_task"] == 2560, "Expected the B2560 mixed comparison")
    require(base["report"]["contract"]["model_config"]["backbone"]["d_model"] == 128,
            "Expected the D128 mixed comparison")
    audits = {label: compatibility(base, runs[label]) for label in labels[1:]}
    kinds = [audit["variant"]["variant"] for audit in audits.values()]
    require(len(set(kinds)) == len(kinds), "Duplicate embedding mechanisms")
    arms = {label: arm_summary(label, run) for label, run in runs.items()}
    summary = {"schema": SCHEMA, "status": "complete", "baseline": labels[0],
               "qualification": QUALIFICATION, "arms": arms, "compatibility": audits,
               "matched_observations": {label: matched_observations(arms[labels[0]], arms[label]) for label in labels[1:]},
               "source_code": str(Path(__file__).resolve()), "source_sha256": saved.sha(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = make_plots(arms, args.output)
    summary["figure_sha256"] = {f"{name}.{suffix}": saved.sha(args.output / f"{name}.{suffix}")
                               for name in summary["figures"] for suffix in ("png", "pdf")}
    (args.output / "report.md").write_text(markdown(summary))
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def publish_wandb(summary, output):
    import wandb
    from scripts.experiment_tracking import OnlineTracker
    tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=output,
                            name=f"mixed-b2560-embedding-comparison-{len(summary['arms'])}-arms")
    try:
        tracker.start({"scope": QUALIFICATION, "arms": list(summary["arms"]), "baseline": summary["baseline"]})
        tracker.log({f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in summary["figures"]})
        tracker.summary({"arms": {label: {key: arm[key] for key in
                          ("endpoint", "parameters", "mechanism", "final", "cumulative_training_seconds", "observed_threshold_crossings")}
                          for label, arm in summary["arms"].items()}, "qualification": QUALIFICATION})
        artifact = wandb.Artifact(f"mixed-embedding-comparison-{tracker.record['run_id']}", type="development-report")
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in (".json", ".md", ".png", ".pdf"):
                artifact.add_file(str(path), name=path.name)
        tracker._call("artifact logging", lambda: tracker._run.log_artifact(artifact))
        tracker.finish(succeeded=True)
        summary["report_wandb"] = tracker.record
    except BaseException:
        tracker.finish(succeeded=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    summary = build_report(args)
    if args.wandb:
        publish_wandb(summary, args.output)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "arms": list(summary["arms"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
