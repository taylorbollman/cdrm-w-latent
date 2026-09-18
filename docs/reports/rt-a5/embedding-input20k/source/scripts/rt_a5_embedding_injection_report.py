#!/usr/bin/env python3
"""Compare saved four-layer input/value embedding-injection diagnostics at 10k.

CPU reporting only; no checkpoint tensors, model construction, or inference.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

from scripts import rt_a5_l1r_depth_report as base
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT = base.ROOT
LINEAGES = {arm: ROOT / f".runtime/rt-a5/20260915T150000Z-embedding-{arm}10k" for arm in ("input", "value")}
OUTPUT = ROOT / "docs/reports/rt-a5/embedding-injection10k"
SCHEMA = "rt-a5-embedding-injection-comparison-v1"
ENDPOINT, STEPS, ARMS = 10000, (1000, 5000, 10000), ("input", "value")
LABELS = {"input": "4 layers: input to second block", "value": "4 layers: permanent V in second block"}
COLORS = {"input": "#236B9E", "value": "#AF5141"}
REPORTING_SOURCES = ("scripts/rt_a5_embedding_injection_report.py", *base.REPORTING_SOURCES)


def validate_shared_contracts(new, reference, variant):
    shared = ("architecture", "width", "seed", "predictor_seed", "data_order_seed", "batch_size", "length", "train_rows",
              "data_manifest_sha256", "precision", "tf32", "compile", "cuda_graphs", "torch", "cuda", "device_capability",
              "optimizer", "word_order", "evaluation_route", "latent_rollout_evaluated", "objective", "nextlat_config")
    if variant not in ARMS or any(new.get(key) != reference.get(key) for key in shared):
        raise ValueError("Injection objective, data, optimizer, predictor or runtime differs from the shared reference")
    expected = {"architecture": "rt", "width": 512, "batch_size": 1024, "length": 12, "train_rows": 800000,
                "seed": 1234, "predictor_seed": 1235, "data_order_seed": 1234, "precision": "fp32", "tf32": False,
                "compile": False, "cuda_graphs": False, "evaluation_route": "backbone_only", "latent_rollout_evaluated": False,
                "n_layers": 4, "variant": variant, "projection_seed": 1236, "injection_coefficient": .02}
    if any(new.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected four-layer injection recipe")
    if new["objective"].get("latent_weight") != 1 or new["objective"].get("target_detached") is not True:
        raise ValueError("Original weight-one NextLat objective required")
    left, right = (copy.deepcopy(c["model_config"]) for c in (new, reference))
    if left.pop("n_layers") != 4 or right.pop("n_layers") != 2 or left != right:
        raise ValueError("Core transformer configuration may differ only in layer count")
    experiment = new["experiment_config"]
    expected_experiment = {"n_layers": 4, "window_layer": 0, "window_length": 2, "injection_layer": 1,
        "position_encoding": "alibi", "learned_position_parameters": False, "variant": variant,
        "coefficient": .02, "coefficient_learned": False, "projection_seed": 1236}
    if any(experiment.get(key) != value for key, value in expected_experiment.items()):
        raise ValueError("Injection location, projection or coefficient differs")
    ref_layers = reference["experiment_config"]["attention"]["layers"]
    if (experiment["attention"].get("gradient_truncation") is not False
            or experiment["attention"]["layers"] != [ref_layers[0]] + [ref_layers[1]] * 3):
        raise ValueError("First layer must use attached window-2; the remaining three use full recurrence")


def validate_initializations(input_init, value_init):
    hashes = ("model_parameter_sha256", "canonical_sha256", "predictor_sha256", "projection_sha256",
              "shared_four_layer_model_sha256", "shared_transformer_sha256")
    for variant, initial in zip(ARMS, (input_init, value_init)):
        if (initial.get("schema") != "rt-a5-embedding-injection-initialization-v1" or initial.get("variant") != variant
                or initial.get("parameter_count") != 13964800 or initial.get("parameter_tensors") != 44
                or initial.get("projection_seed") != 1236 or initial.get("projection_parameter_count") != 512 * 512
                or initial.get("predictor_seed") != 1235 or initial.get("coefficient") != .02
                or initial.get("coefficient_learned") is not False or initial.get("baseline_parameter_tensors_changed") != []
                or initial.get("predictor_initialization_paired") is not True):
            raise ValueError("Unexpected injection initialization or learned-parameter ownership")
        if any(not _sha(initial.get(key)) for key in hashes):
            raise ValueError("Missing initialized-tensor identity hashes")
    if any(input_init[key] != value_init[key] for key in hashes):
        raise ValueError("The two injection variants must start from identical learned tensor values")


def compare_orders(arms, reference_history):
    base.validate_history(reference_history)
    for arm in ARMS:
        base.validate_history(arms[arm]["history"])
        base.compare_minibatch_orders(arms[arm]["history"], reference_history)


def read_injection(variant):
    lineage = LINEAGES[variant]
    raw, protocol_file = read_input(lineage / "protocol.json"); protocol = json.loads(raw)
    expected = {"schema": "rt-a5-embedding-injection-protocol-v1", "start_update": 0, "endpoint": ENDPOINT,
                "variant": variant, "n_layers": 4, "injection_layer": 1, "coefficient": .02,
                "checkpoint_steps": [0, *STEPS], "full_evaluation_rows": 102400}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Require the frozen prospective four-layer injection protocol")
    directory = local_path(protocol["training_directory"]).resolve()
    if directory != lineage / "train-injection" or local_path(protocol["report_directory"]).resolve() != OUTPUT:
        raise ValueError("Unexpected injection lineage/training directory")
    raw, report_file = read_input(directory / "report.json"); report = json.loads(raw)
    if (report.get("schema") != "rt-a5-embedding-injection-training-v1" or report.get("status") != "complete"
            or report.get("start_update") != 0 or report.get("endpoint") != ENDPOINT or report.get("completed_updates") != ENDPOINT
            or report.get("parent_checkpoint") is not None or report.get("wandb", {}).get("status") != "synced"
            or report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False
            or (lineage / "training-exit-code.txt").read_text().strip() != "0"):
        raise ValueError("Require successful fresh, complete, synced 10k training")
    sources = report["source_files"]
    if (report["contract"] != protocol["strict_contract"] or report["initialization"] != protocol["initialization"]
            or sources != protocol["source_files"] or len(sources) != 62
            or _digest_dict(sources) != protocol["source_sha256"] or protocol["source_sha256"] != report["contract"]["source_sha256"]):
        raise ValueError("Injection training record differs from its frozen 62-source protocol")
    for name, wanted in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not _sha(wanted):
            raise ValueError("Invalid source path/hash")
        if hash_file(ROOT / relative)["sha256"] != wanted or hash_file(directory / "source" / relative)["sha256"] != wanted:
            raise ValueError(f"Frozen injection source changed: {name}")
    raw, config_file = read_input(directory / "config.json"); config = json.loads(raw)
    if config != protocol["resolved_args"] or config.get("resume") is not None or config.get("variant") != variant:
        raise ValueError("Resolved fresh injection configuration differs")
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != report["contract"]["data_manifest_sha256"]:
        raise ValueError("Training data manifest changed")
    base.bound_file(protocol["preflight"])
    if json.loads(local_path(protocol["preflight"]["path"]).read_text()).get("passed") is not True:
        raise ValueError("Injection preflight did not pass")
    if base.bound_file(protocol["data_manifest"])["sha256"] != manifest["sha256"]:
        raise ValueError("Protocol and actual data manifests differ")
    history_path = directory / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    base.validate_history(history)
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("Report/history endpoint order differs")
    if sorted(c["completed_updates"] for c in report["checkpoints"]) != [0, *STEPS]:
        raise ValueError("Require exactly 0/1k/5k/10k saved checkpoints")
    checkpoints = {}
    for item in report["checkpoints"]:
        step = item["completed_updates"]
        if local_path(item["path"]).resolve() != directory / "checkpoints" / f"step-{step:06d}.pt":
            raise ValueError("Checkpoint outside its declared lineage")
        checkpoints[str(step)] = base.bound_file(item)
    raw, state_file = read_input(lineage / "final-state-validation.json"); state = json.loads(raw)
    if (state.get("schema") != "rt-a5-embedding-injection-final-state-validation-v1"
            or state.get("passed") is not True or state.get("variant") != variant
            or state.get("completed_updates") != ENDPOINT or state.get("model_parameters") != 13964800
            or state.get("model_parameter_tensors") != 44 or state.get("final_active_adam_states") != 44):
        raise ValueError("Require passing saved-state validation before comparison")
    # The checker owns tensor/moment inspection; this report binds its saved inputs.
    if state.get("protocol", {}).get("sha256") != protocol_file["sha256"] or state.get("report", {}).get("sha256") != report_file["sha256"]:
        raise ValueError("Saved-state validation is bound to different protocol/report evidence")
    if state.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("Saved-state validation uses different training source")
    for step in (0, *STEPS):
        if state["checkpoint_inputs"][str(step)]["sha256"] != checkpoints[str(step)]["sha256"]:
            raise ValueError("Saved-state validation checked different checkpoint tensors")
    for checked, recorded in (("initial_model_sha256", "model_parameter_sha256"),
                              ("shared_four_layer_model_sha256", "shared_four_layer_model_sha256"),
                              ("projection_sha256", "projection_sha256"), ("predictor_sha256", "predictor_sha256")):
        if state.get(checked) != report["initialization"].get(recorded):
            raise ValueError("Saved initialized-tensor identity differs from the training record")
    curves, metrics = {}, {}
    for step in STEPS:
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            selected = [e for e in report["evaluations"] if e["update"] == step and e["role"] == role]
            if len(selected) != 1 or selected[0].get("rows") != 102400 or selected[0].get("route") != "backbone_only":
                raise ValueError("Primary evaluation requires the same full 102400 words at 1k/5k/10k")
            metrics[str(step)][role] = selected[0]; curves[str(step)][role] = metric_rows(selected[0], variant)
    return {"directory": str(directory), "protocol": protocol, "contract": report["contract"], "initialization": report["initialization"],
            "source_files": sources, "checkpoints": checkpoints, "curves": curves, "metrics": metrics,
            "history": history, "wandb": report["wandb"], "saved_state": state, "inputs": {"protocol": protocol_file, "report": report_file,
            "config": config_file, "history": hash_file(history_path), "data_manifest": manifest, "saved_state": state_file}}


def make_summary():
    # Preserve both closed baseline readers and limit the old 80k reference to its first 10k.
    context = base.make_summary(base.LINEAGE / "protocol.json")
    arms = {arm: read_injection(arm) for arm in ARMS}
    reference = context["arms"]["two_layers"]
    with local_path(reference["inputs"]["history"]["path"]).open() as stream:
        reference_history = [json.loads(next(stream)) for _ in range(ENDPOINT)]
    for arm in ARMS:
        validate_shared_contracts(arms[arm]["contract"], reference["contract"], arm)
        binding = arms[arm]["protocol"]["reference"]
        if (local_path(binding["training_directory"]).resolve() != base.REFERENCE
                or binding["source_files"] != reference["source_files"]
                or binding["source_sha256"] != reference["contract"]["source_sha256"]):
            raise ValueError("Injection protocol declares a different contextual reference")
        for name in ("report", "config"):
            if base.bound_file(binding[name])["sha256"] != reference["inputs"][name]["sha256"]:
                raise ValueError("Context reference report/config binding differs")
        for step in (0, *STEPS):
            if base.bound_file(binding["checkpoints"][str(step)])["sha256"] != reference["checkpoints"][str(step)]["sha256"]:
                raise ValueError("Context reference checkpoint binding differs")
        for name, digest in context["arms"]["six_layers"]["source_files"].items():
            if arms[arm]["source_files"].get(name) != digest:
                raise ValueError("Historical 58-source training identity changed")
    if arms["input"]["source_files"] != arms["value"]["source_files"]:
        raise ValueError("The two injection variants must share all source files")
    validate_initializations(*(arms[arm]["initialization"] for arm in ARMS))
    compare_orders(arms, reference_history)
    for arm in ARMS:
        arms[arm]["training_curve"] = training_curve(arms[arm].pop("history"), augmented=True)
    return {"schema": SCHEMA, "primary_update": ENDPOINT, "checkpoint_updates": list(STEPS), "arms": arms,
            "context_arms": context["arms"], "context_protocol": context["protocol_input"],
            "matched_minibatch_order_hashes": ENDPOINT, "same_initial_tensors_between_injection_arms": True,
            "same_initial_tensors_against_depth_baselines": False, "four_layer_uninjected_control_available": False,
            "confirmation_evaluated": False, "latent_rollout_evaluated": False,
            "metric_definitions": context["metric_definitions"],
            "scope": "Two fresh four-layer injection diagnostics, one seed, fixed10k; two/six-layer10k context only"}


def plot_rows(summary, arm, step=ENDPOINT):
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Primary plots require input/value and matched 1k/5k/10k checkpoints")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [(row["arm"], row["update"], row["role"], row["length"]) for row in rows] != [
            (arm, step, "ood_dev", length) for length in range(1, 37)]:
        raise ValueError("Full and boundary plots require identical length-36 metric rows")
    return rows


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figures = []
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        figures.append(name); plt.close(fig)
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.80, wspace=.35)
        for axis, key, title in zip(axes, ("E", "A", "M"), ("Every state through t correct", "Only state t correct", "Mean token accuracy through t")):
            for arm in ARMS:
                rows = plot_rows(summary, arm); x = [row["length"] for row in rows]
                axis.plot(x, [row[key] for row in rows], color=COLORS[arm], label=LABELS[arm])
                if key != "M":
                    axis.fill_between(x, [r[f"{key}_low95"] for r in rows], [r[f"{key}_high95"] for r in rows], color=COLORS[arm], alpha=.12)
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of same length-36 words", title=f"{key}(t): {title}")
            axis.axvline(12, color="gray", linestyle=":"); axis.grid(alpha=.2); axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.legend(fontsize=7)
        fig.suptitle("Four-layer RT + NextLat embedding injection · matched 10,000 updates · 102,400 words", y=.97)
        save(fig, name)
    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for arm in ARMS:
        axis.plot(STEPS, [plot_rows(summary, arm, step)[-1]["E"] for step in STEPS], marker="o", color=COLORS[arm], label=LABELS[arm])
    axis.set(xlabel="Optimizer updates", ylabel="E(36): whole-word accuracy", ylim=(-.025, 1.025), xticks=STEPS)
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Matched length-36 whole-word accuracy · primary injection variants")
    save(fig, "whole-word-vs-updates")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.80, wspace=.35)
    for axis, key, title in zip(axes, ("state_ce", "latent_loss", "loss"), ("State CE", "Latent SmoothL1", "Joint objective")):
        for arm in ARMS:
            rows = summary["arms"][arm]["training_curve"]
            axis.plot([r["update"] for r in rows], [r[key] for r in rows], color=COLORS[arm], label=LABELS[arm])
        axis.set(xlabel="Optimizer updates", ylabel=title, xlim=(0, ENDPOINT)); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle("Training objective · nonoverlapping 100-update means", y=.97)
    save(fig, "training-losses")
    return figures


def markdown(summary, *, brief=False):
    first, second = (plot_rows(summary, arm)[-1] for arm in ARMS)
    lines = ["# Four-layer RT + NextLat: embedding-injection diagnostics", "",
             f"At **10,000 updates**, length-36 whole-word accuracy E(36) is **{first['E']:.4%}** with input injection and "
             f"**{second['E']:.4%}** with permanent-value injection.", "",
             "Both models start with identical learned tensor values, including the independently initialized embedding projection. "
             "Only the projection's route differs: the second block's input versus its permanent value write. "
             "The fixed coefficient is 0.02. The first of four RT layers uses window-2; the remaining three use full recurrence.", "",
             "| Model at 10k | L12 whole word | E(13) | E(14) | E(36) | A(36) | M(36) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, packet, label in [(arm, summary["arms"][arm], LABELS[arm]) for arm in ARMS] + [
            (arm, summary["context_arms"][arm], "Context: " + base.LABELS[arm]) for arm in base.ARMS]:
        rows = packet["curves"][str(ENDPOINT)]["ood_dev"]
        values = [packet["metrics"][str(ENDPOINT)]["dev"]["whole_word_exact_match"], rows[12]["E"], rows[13]["E"],
                  rows[-1]["E"], rows[-1]["A"], rows[-1]["M"]]
        lines.append(f"| {label} | " + " | ".join(f"{v:.4%}" for v in values) + " |")
    lines += ["", "E(t) requires every state through t correct; A(t) checks only state t; M(t) averages correctness through t. "
              "Full and boundary plots use the identical 102,400 length-36 development words and saved 10k rows.", "",
              "The two- and six-layer baselines are contextual comparisons at 10k. There is **no uninjected four-layer control**, "
              "so differences against those baselines cannot isolate an injection benefit. Depth changes initialization and compute. "
              "The paired input-versus-value comparison holds depth, learned tensor initialization, optimizer, all 10k minibatch orders, "
              "and original joint NextLat objective fixed.", "",
              "This is a one-seed development diagnostic selected after earlier experiments. Width512, batch1024, fullFP32, "
              "noTF32/compile/CUDA graphs; 13,964,800 parameters per injected model. No final confirmation, autonomous latent rollout, "
              "or convergence claim. Saved-state checkers inspect tensors/Adam separately; this reporter reads saved metadata and counts."]
    if not brief:
        for name in ("whole-word-vs-updates", "length-full", "length-boundary", "training-losses"):
            lines += ["", f"![{name}]({name}.png)"]
        lines += ["", "Pointwise Wilson 95% E/A intervals describe variation over words, not uncertainty across training seeds.", "",
                  "[Exact primary metric counts](metrics.csv) · [Context metrics](context-metrics.csv) · [Provenance](summary.json)"]
    return "\n".join(lines) + "\n"


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh embedding-injection report directory")
    summary = make_summary(); output.mkdir(parents=True)
    rows = [r for arm in ARMS for step in STEPS for role in ROLES for r in summary["arms"][arm]["curves"][str(step)][role]]
    context_rows = [r for arm in base.ARMS for role in ROLES for r in summary["context_arms"][arm]["curves"][str(ENDPOINT)][role]]
    for name, records in (("metrics.csv", rows), ("context-metrics.csv", context_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS); writer.writeheader(); writer.writerows(records)
    for name in REPORTING_SOURCES:
        target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    summary["reporting_sources"] = {name: hash_file(output / "source" / name) for name in REPORTING_SOURCES}
    write_json(output / "summary.json", summary)
    figures = plots(summary, output)
    (output / "report.md").write_text(markdown(summary)); (output / "outcome.md").write_text(markdown(summary, brief=True))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="four-layer-rt-nextlat-input-vs-value-injection10k")
    result = {"schema": SCHEMA, "status": "running", "primary_update": ENDPOINT, "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "scope": summary["scope"], "four_layer_uninjected_control": False})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for index in range(ENDPOINT // 100):
            update = (index + 1) * 100
            values = {"update": update, **{f"train/{arm}/{key}": summary["arms"][arm]["training_curve"][index][key]
                       for arm in ARMS for key in ("state_ce", "latent_loss", "loss")}}
            if update in STEPS:
                values.update({f"dev/{arm}/E36": plot_rows(summary, arm, update)[-1]["E"] for arm in ARMS})
            tracker.log(values)
        tracker.summary({"primary_update": ENDPOINT, **{f"{arm}/E36": plot_rows(summary, arm)[-1]["E"] for arm in ARMS}})
        tracker.finish(succeeded=True); result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try: tracker.finish(succeeded=False)
        except Exception: pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(p.relative_to(output)): hash_file(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--wandb-group", default="20260915T150000Z-embedding-injection10k")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
