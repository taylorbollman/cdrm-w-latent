#!/usr/bin/env python3
"""Report the completed, frozen-backbone O5c fusion adaptation comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_o4_report as paired
from scripts import olmo_o5b_report as fbt

ARMS = ("code", "mixed")
DOMAINS = ("code", "general")
SPLITS = ("dev", "retention_dev")
SCHEMA = "olmo-o5c-final-comparison-v1"
FIGURES = ("learning-curves.pdf", "learning-curves.png", "domain-exposure.pdf", "domain-exposure.png")
TRAINABLE_NAMES = {"backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight"}
require, sha = paired.require, paired.sha


def _hashes(value, label):
    require(isinstance(value, dict) and value and all(isinstance(name, str) and name and
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in value.items()),
            f"Missing or malformed {label} tensor/source hashes")
    return value


def _frozen(initial, final, authority):
    for value in (initial, final, authority):
        _hashes(value, "frozen state")
    require(initial == final == authority, "Frozen native tensors or buffers changed or differ from source endpoint")
    require(not TRAINABLE_NAMES.intersection(initial), "Trainable fusion matrices cannot be counted as frozen")
    require("backbone.fusion.output_scale" in initial, "Fixed fusion output scale is missing from frozen inventory")


def _pass0_equal(actual, expected):
    require(actual["ce_count"] == expected["ce_count"] and actual["next_token_correct"] == expected["next_token_correct"],
            "Frozen ordinary pass target/correct-token counts changed")
    # Both use the same BF16 evaluator and batches. Allow only a tiny aggregate
    # reduction-order allowance; bytewise frozen-state hashes are checked too.
    require(abs(actual["mean_nll"] - expected["mean_nll"]) <= 2e-6,
            "Frozen ordinary pass NLL changed beyond roundoff")


def _domain_counts(value, arm, update, quota):
    require(isinstance(value, dict) and set(value) == set(DOMAINS), "Missing domain exposure counters")
    shares = {"code": 1.0, "general": 0.0} if arm == "code" else {"code": .5, "general": .5}
    for domain in DOMAINS:
        counts = value[domain]
        require(all(paired._integer(counts.get(key)) for key in ("input_tokens", "ce_positions", "documents")),
                "Domain exposure counters must be nonnegative integers")
        require(counts["ce_positions"] == int(update * quota * shares[domain]), "Domain CE quota differs from frozen mixture")
        require(counts["input_tokens"] == counts["ce_positions"] + counts["documents"],
                "Domain input/CE/context segment accounting differs")
        require((counts["ce_positions"] == 0) == (counts["documents"] == 0), "Empty domain has phantom segments")
    return {key: sum(value[domain][key] for domain in DOMAINS) for key in ("input_tokens", "ce_positions", "documents")}


def _validate_plan(plan, schedule, arm):
    updates, quota = schedule["total_updates"], schedule["ce_per_update"]
    require(plan.get("total_updates") == updates and plan.get("ce_per_update") == quota,
            "Arm plan optimizer/CE budget differs")
    prefixes = {key: plan.get(key) for key in ("batch_ce_prefix", "batch_token_prefix", "batch_row_prefix")}
    domains = plan.get("domain_prefixes", {})
    require(set(domains) == set(DOMAINS), "Arm plan omits domain prefixes")
    for domain in DOMAINS:
        require(set(domains[domain]) == {"input_tokens", "ce_positions", "documents"}, "Domain prefix keys differ")
        prefixes.update({domain + "/" + key: value for key, value in domains[domain].items()})
    for values in prefixes.values():
        require(isinstance(values, list) and len(values) == updates + 1 and values[0] == 0
                and all(paired._integer(v) for v in values) and all(a <= b for a, b in zip(values, values[1:])),
                "Arm exposure prefixes are malformed or nonmonotonic")
    for update in range(updates + 1):
        counts = {domain: {key: values[update] for key, values in domains[domain].items()} for domain in DOMAINS}
        total = _domain_counts(counts, arm, update, quota)
        for name, key in (("batch_ce_prefix", "ce_positions"), ("batch_token_prefix", "input_tokens"), ("batch_row_prefix", "documents")):
            require(plan[name][update] == total[key], "Arm plan/domain cumulative exposure differs")
    return plan


def _event(event, arm, configuration, *, full, baseline):
    schedule = configuration["schedule"]
    update = event.get("update")
    require(paired._integer(update) and update <= schedule["total_updates"], "Evaluation update is outside frozen budget")
    require(event.get("full") is full and event.get("rows_requested") == (512 if full else 128),
            "Evaluation mixes full and curve selections")
    require(event.get("beta") == 1.0, "O5c evaluates the fixed beta-one endpoint")
    totals = _domain_counts(event.get("domain_counts"), arm, update, schedule["ce_per_update"])
    require(event.get("ce_positions") == totals["ce_positions"] and event.get("input_tokens") == totals["input_tokens"],
            "Evaluation exposure differs from domain counters")
    plan = schedule["arms"][arm]
    require(event["input_tokens"] == plan["batch_token_prefix"][update] and totals["documents"] == plan["batch_row_prefix"][update],
            "Evaluation differs from frozen segment/token plan")
    require(event["domain_counts"] == {domain: {key: values[update] for key, values in plan["domain_prefixes"][domain].items()} for domain in DOMAINS},
            "Evaluation domain exposure differs from frozen plan")
    require(set(event.get("metrics", {})) == set(SPLITS), "Evaluation splits differ")
    for split, container in event["metrics"].items():
        fbt.validate_container(container, beta=1.0, rows=512 if full else 128, documents=full)
        for index, metric in enumerate(container["passes"]):
            reference = baseline[split]["passes"][index]
            require(metric["ce_count"] == reference["ce_count"], "Evaluation targets differ from source endpoint")
            if full:
                require(paired._window_signature(metric) == paired._window_signature(reference),
                        "Final window/document selection differs from source endpoint")
        _pass0_equal(container["passes"][0], baseline[split]["passes"][0])
    return event


def _checkpoint(report, configuration):
    total_updates = configuration["schedule"]["total_updates"]
    records = [row for row in report.get("checkpoints", []) if row.get("optimizer_updates") == total_updates]
    require(len(records) == 1, "Require exactly one retained endpoint checkpoint")
    record = records[0]
    storage = record.get("storage", {})
    require(record.get("input_tokens") == report["counters"]["input_tokens"], "Checkpoint input exposure differs")
    require(isinstance(record.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Invalid checkpoint SHA256")
    require(paired._integer(record.get("size_bytes"), 1), "Invalid checkpoint byte count")
    require(storage.get("sha256") == record["sha256"] and storage.get("size_bytes") == record["size_bytes"]
            and storage.get("generation") and storage.get("md5_base64") and storage.get("verification"),
            "Endpoint lacks a matching verified storage receipt")
    prefix = report.get("storage_prefix", "")
    require(prefix.startswith("gs://fast-chunks/") and storage.get("uri", "").startswith(prefix + "/" + report["arm"] + "/"),
            "Checkpoint URI differs from selected arm")
    return {key: record[key] for key in ("sha256", "size_bytes", "optimizer_updates", "input_tokens", "storage")}


def build_comparison(preflight, configuration, arm_reports):
    require(preflight.get("schema") == "olmo-o5c-preflight-v1" and preflight.get("status") == "passed" and preflight.get("finished_utc"),
            "Require completed passing O5c preflight")
    require(configuration.get("schema") == "olmo-o5c-pilot-config-v1", "Unexpected O5c configuration schema")
    require(set(arm_reports) == set(ARMS), "Require both completed O5c arms")
    sources = _hashes(configuration.get("source_hashes"), "source")
    require(sources == preflight.get("source_hashes"), "Preflight/training source hashes differ")
    require(configuration.get("precision") == "bf16_mixed" and configuration.get("eval_rows") == 128
            and configuration.get("final_eval_rows") == 512, "Frozen precision/evaluation selections differ")
    require(configuration.get("num_passes") == 2 and configuration.get("gamma") == 1.0 and configuration.get("beta") == 1.0
            and configuration.get("rt_layers") == [] and configuration.get("nextlat_enabled") is False
            and configuration.get("prefix_mixin") is False and configuration.get("hidden_jitter") == 0.0,
            "O5c requires fixed beta1/K2/gamma1 with no RT/NextLat/prefix mixing/jitter")
    require(configuration.get("native_backbone_frozen") is True, "Native backbone must be explicitly frozen")
    schedule = configuration.get("schedule", {})
    require(paired._integer(schedule.get("total_updates"), 1) and paired._integer(schedule.get("ce_per_update"), 2)
            and schedule["ce_per_update"] % 2 == 0 and schedule.get("total_ce") == schedule["total_updates"] * schedule["ce_per_update"],
            "Frozen CE quota schedule is inconsistent")
    require(preflight.get("schedule") == schedule, "Preflight CE schedule differs")
    require(set(schedule.get("arms", {})) == set(ARMS), "Require separate code/mixed segment plans")
    for arm in ARMS:
        _validate_plan(schedule["arms"][arm], schedule, arm)
    require(preflight.get("source_checkpoint_sha256") == configuration.get("source_checkpoint_sha256")
            and isinstance(configuration.get("source_checkpoint_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", configuration["source_checkpoint_sha256"]), "Adapted source checkpoint differs")
    require(preflight.get("data_manifest_sha256") == configuration.get("data_manifest_sha256")
            and isinstance(configuration.get("data_manifest_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", configuration["data_manifest_sha256"]), "Prepared O5c data identity differs")
    frozen = _hashes(preflight.get("frozen_state_initial"), "frozen source")
    layout = preflight.get("trainable_layout")
    require(isinstance(layout, list) and len(layout) == 2 and all(isinstance(row, dict) for row in layout)
            and {row.get("name") for row in layout} == TRAINABLE_NAMES, "Only the two fusion matrices may be trainable")
    width = configuration.get("model_config", {}).get("model_dim")
    layers = configuration.get("model_config", {}).get("num_layers")
    require(paired._integer(width, 1) and paired._integer(layers, 1), "Native model dimensions are missing")
    require(all(row.get("shape") == [width, width] and row.get("numel") == width * width for row in layout),
            "Trainable fusion matrix shape/size differs")
    native_names = {"backbone.backbone.transformer.wte.weight", "backbone.fusion.output_scale"}
    native_names.update(f"backbone.backbone.transformer.blocks.{i}.{name}.weight" for i in range(layers)
                        for name in ("att_proj", "attn_out", "ff_proj", "ff_out"))
    require(set(frozen) == native_names, "Frozen inventory must cover every native tensor and fixed fusion scale")
    require(configuration.get("frozen_state_initial") == frozen and configuration.get("trainable_layout") == layout,
            "Frozen source state/trainable layout differs from configuration")
    endpoint = preflight.get("endpoint", {})
    require(endpoint.get("checkpoint_sha256") == configuration["source_checkpoint_sha256"]
            and endpoint.get("frozen_state_digests") == frozen and endpoint.get("trainable_parameters") == layout,
            "Preflight endpoint tensor ownership/provenance differs")
    initial_all = _hashes(endpoint.get("state_digests"), "initial endpoint")
    require(set(initial_all) == set(frozen) | TRAINABLE_NAMES and all(initial_all[key] == value for key, value in frozen.items()),
            "Source endpoint must identify frozen and initial trainable tensors")
    require(endpoint.get("optimizer_state") == "fresh; endpoint optimizer/scheduler/counters/RNG not resumed",
            "O5c must reset the endpoint optimizer state")
    baseline, small = preflight.get("baseline", {}), preflight.get("baseline_small", {})
    require(set(baseline) == set(small) == set(SPLITS), "Require full and small source-endpoint baselines")
    for split in SPLITS:
        fbt.validate_container(baseline[split], beta=1.0, rows=512, documents=True)
        fbt.validate_container(small[split], beta=1.0, rows=128, documents=False)
    summaries, curves = {}, {}
    for arm in ARMS:
        report = arm_reports[arm]
        require(report.get("schema") == "olmo-o5c-arm-v1" and report.get("status") == "completed"
                and report.get("arm") == arm and report.get("finished_utc"), f"Arm {arm} is not completed")
        require(report.get("configuration") == configuration, f"Arm {arm} frozen configuration differs")
        require(report.get("endpoint") == endpoint, "Arms must import the identical source fusion/native state and reset optimizer")
        source = report.get("source_fingerprint", {})
        require(source.get("source_checkpoint_sha256") == configuration["source_checkpoint_sha256"]
                and source.get("checkpoint_sha256") == configuration["source_checkpoint_sha256"]
                and source.get("data_manifest_sha256") == configuration["data_manifest_sha256"]
                and source.get("code") == sources and source.get("runtime") == preflight.get("runtime"),
                "Arm source/data/runtime provenance differs")
        _frozen(report.get("frozen_state_initial"), report.get("frozen_state_final"), frozen)
        require(report.get("trainable_layout") == layout, "Arm trainable ownership differs from fusion-only preflight")
        counters = report.get("counters", {})
        require(counters.get("optimizer_updates") == schedule["total_updates"], "Arm stopped before fixed CE budget")
        require(report.get("data_cursor") == schedule["total_updates"], "Arm next-update cursor differs from endpoint")
        totals = _domain_counts(report.get("domain_counts"), arm, schedule["total_updates"], schedule["ce_per_update"])
        require(all(counters.get(key) == value for key, value in totals.items()), "Final domain/total exposures differ")
        require(counters.get("latent_pairs") == counters.get("kl_triples") == 0 and paired._integer(counters.get("microbatches"), 1),
                "Inactive auxiliary or missing microbatch counts")
        plan = schedule["arms"][arm]
        require(counters["input_tokens"] == plan["batch_token_prefix"][-1] and counters["documents"] == plan["batch_row_prefix"][-1],
                "Final exposure differs from frozen segment plan")
        physical = configuration.get("physical_batch_size")
        require(paired._integer(physical, 1), "Physical microbatch bound is missing")
        expected_microbatches = sum((right - left + physical - 1) // physical for left, right in
                                   zip(plan["batch_row_prefix"], plan["batch_row_prefix"][1:]))
        require(counters["microbatches"] == expected_microbatches, "Completed physical microbatches differ from segment plan")
        full = [row for row in report.get("evaluations", []) if row.get("full") is True]
        require(len(full) == 1, "Require one authoritative full endpoint evaluation")
        final = _event(full[0], arm, configuration, full=True, baseline=baseline)
        require(final["update"] == schedule["total_updates"] and final["domain_counts"] == report["domain_counts"],
                "Final evaluation is not at the arm endpoint")
        by_update = {}
        for event in report["evaluations"]:
            if event.get("full") is True:
                continue
            _event(event, arm, configuration, full=False, baseline=small)
            require(event["update"] not in by_update, "Duplicate authoritative curve boundary")
            by_update[event["update"]] = event
        require({0, schedule["total_updates"]} <= set(by_update), "Curves omit initial/final boundary")
        curves[arm] = [by_update[index] for index in sorted(by_update)]
        for split in SPLITS:
            for index in (0, 1):
                require(paired._close(by_update[0]["metrics"][split]["passes"][index]["mean_nll"], small[split]["passes"][index]["mean_nll"]),
                        "Arm does not start from the source-endpoint baseline")
        link = report.get("wandb", {}).get("run_url")
        require(isinstance(link, str) and link.startswith("https://wandb.ai/"), "Missing online run link")
        summaries[arm] = {"metrics": final["metrics"], "counters": counters, "domain_counts": report["domain_counts"],
            "checkpoint": _checkpoint(report, configuration), "wandb_url": link,
            "frozen_state_initial": report["frozen_state_initial"], "frozen_state_final": report["frozen_state_final"],
            "trainable_layout": layout, "started_utc": report.get("started_utc"), "finished_utc": report["finished_utc"]}
    require([e["update"] for e in curves["code"]] == [e["update"] for e in curves["mixed"]], "Curve update/CE boundaries differ")
    comparisons = {}
    for split in SPLITS:
        code, mixed = (summaries[arm]["metrics"][split]["passes"] for arm in ARMS)
        comparisons[split] = {"mixed_pass1_minus_code_pass1": fbt.contrast(mixed[1], code[1]),
            "versus_source": {arm: fbt.contrast(summaries[arm]["metrics"][split]["passes"][1], baseline[split]["passes"][1]) for arm in ARMS},
            "feedback_gap": {arm: fbt.contrast(summaries[arm]["metrics"][split]["passes"][1], summaries[arm]["metrics"][split]["passes"][0]) for arm in ARMS}}
    return {"schema": SCHEMA, "status": "completed", "configuration": configuration,
        "matched_ce_exposure": schedule["total_ce"], "optimizer_updates_per_arm": schedule["total_updates"],
        "source_baseline": baseline, "source_baseline_small": small, "arms": summaries, "curves": curves, "comparisons": comparisons,
        "difference_convention": "left minus right token NLL; negative favors left",
        "qualification": "One seed and development subsets only; reserved tests untouched. Both arms start from the same adapted O5b endpoint, "
        "not the original pretrained checkpoint. Only fusion matrices train; frozen-state tensor hashes and ordinary-pass metrics are checked. "
        "The budget matches supervised CE targets and optimizer updates, not exact input tokens, number of segments, or measured compute. "
        "Mixed has half the code CE exposure plus fresh general-text exposure, so differences include that allocation tradeoff. "
        "Documents counters count independent reset segments/windows, not unique documents. Paired intervals resample original evaluation-document "
        "clusters and exclude training-seed variability and unknown pretraining overlap. Curves use128 windows; final comparisons use512. "
        "These are finite K2 teacher-forced NLL measurements, not exact-online or generated-code capability results."}


def _read_inputs(preflight, runs):
    directory = Path(preflight)
    directory = directory.parent if directory.is_file() else directory
    paths = {"preflight": directory / "report.json", "configuration": directory / "configuration.json",
             **{arm: Path(runs) / arm / "report.json" for arm in ARMS}}
    require(all(path.is_file() for path in paths.values()), "Require preflight and both completed arm reports")
    return paths, {key: json.loads(path.read_text()) for key, path in paths.items()}


def validate_runs(preflight, runs):
    _, values = _read_inputs(preflight, runs)
    return build_comparison(values["preflight"], values["configuration"], {arm: values[arm] for arm in ARMS})


def write_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = ("Code continuation development", "WikiText retention development")
    colors = {"code": "#2878b5", "mixed": "#bb5c13"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, split, title in zip(axes, SPLITS, titles):
        base = summary["source_baseline_small"][split]["passes"]
        ax.axhline(base[0]["mean_nll"], color="gray", linestyle=":", label="Frozen ordinary pass0")
        ax.axhline(base[1]["mean_nll"], color="gray", linestyle="--", label="Source feedback pass1")
        for arm in ARMS:
            rows = summary["curves"][arm]
            ax.plot([r["ce_positions"] / 1e6 for r in rows], [r["metrics"][split]["passes"][1]["mean_nll"] for r in rows],
                    marker="o", markersize=3, color=colors[arm], label=arm)
        ax.set(title=title, xlabel="Additional supervised CE targets (millions)", ylabel="Token-weighted NLL (nats)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fusion-only adaptation · fixed128-window curves · one seed")
    for suffix in ("pdf", "png"): fig.savefig(output / f"learning-curves.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, arm in zip(axes, ARMS):
        rows = summary["curves"][arm]
        x = [r["update"] for r in rows]
        for domain, style in (("code", "-"), ("general", "--")):
            ax.plot(x, [r["domain_counts"][domain]["ce_positions"] / 1e6 for r in rows], linestyle=style, label=f"{domain} CE")
        ax.plot(x, [r["input_tokens"] / 1e6 for r in rows], color="gray", linestyle=":", label="All input tokens including context")
        ax.set(title=arm, xlabel="Optimizer updates", ylabel="Additional exposure (millions)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Matched CE budgets; unequal domain allocation and segment-context overhead")
    for suffix in ("pdf", "png"): fig.savefig(output / f"domain-exposure.{suffix}", dpi=180)
    plt.close(fig)


def markdown(summary):
    lines = ["# O5c: frozen-backbone fusion adaptation", "",
        f"Both arms completed **{summary['optimizer_updates_per_arm']:,} updates** and **{summary['matched_ce_exposure']:,} additional CE targets** "
        "from the same O5b FBT endpoint. Only the two fusion matrices trained; every recorded frozen tensor and buffer remained byte-identical.", "",
        "| Arm | Code CE targets | General CE targets | Total input tokens | Reset segments |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for arm, row in summary["arms"].items():
        lines.append(f"| {arm} | {row['domain_counts']['code']['ce_positions']:,} | {row['domain_counts']['general']['ce_positions']:,} | "
                     f"{row['counters']['input_tokens']:,} | {row['counters']['documents']:,} |")
    lines += ["", "Final512-window metrics; lower NLL is better.", "",
        "| Model / pass | Code NLL | Code accuracy | Retention NLL | Retention accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    rows = [(f"Source O5b pass{p}", {s: summary["source_baseline"][s]["passes"][p] for s in SPLITS}) for p in (0, 1)]
    rows += [(f"{arm} pass{p}", {s: summary["arms"][arm]["metrics"][s]["passes"][p] for s in SPLITS}) for arm in ARMS for p in (0, 1)]
    for label, values in rows:
        lines.append(f"| {label} | " + " | ".join(f"{values[s]['mean_nll']:.6f} | {values[s]['next_token_accuracy']*100:.3f}%" for s in SPLITS) + " |")
    lines += ["", "Differences are left minus right NLL. 95% intervals use1,000 paired resamples of original-document clusters, "
        "combining their windows before token weighting (seed20260922).", "",
        "| Feedback-pass contrast | Code difference [95% interval] | Retention difference [95% interval] |", "| --- | ---: | ---: |"]
    for label, selector in (("mixed − code", lambda c: c["mixed_pass1_minus_code_pass1"]),
        ("code − source", lambda c: c["versus_source"]["code"]), ("mixed − source", lambda c: c["versus_source"]["mixed"])):
        values = [selector(summary["comparisons"][s]) for s in SPLITS]
        lines.append(f"| {label} | " + " | ".join(f"{v['estimate']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" for v in values) + " |")
    lines += ["", "![Learning curves](learning-curves.png)", "", "[Curves PDF](learning-curves.pdf)", "",
              "![Domain exposure](domain-exposure.png)", "", "[Exposure PDF](domain-exposure.pdf) · [Full comparison](final-comparison.json)",
              "", summary["qualification"], "", "Run and endpoint records:", ""]
    for arm, row in summary["arms"].items():
        checkpoint = row["checkpoint"]["storage"]
        lines.append(f"- [{arm} on W&B]({row['wandb_url']}); [checkpoint]({checkpoint['uri']}), generation `{checkpoint['generation']}`, SHA256 `{checkpoint['sha256']}`.")
    return "\n".join(lines) + "\n"


def build_report(args):
    paths, values = _read_inputs(args.preflight, args.runs)
    result = build_comparison(values["preflight"], values["configuration"], {arm: values[arm] for arm in ARMS})
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), report_source_sha256=sha(__file__),
        helper_source_sha256={"scripts/olmo_o4_report.py": sha(paired.__file__), "scripts/olmo_o5b_report.py": sha(fbt.__file__)},
        input_sha256={name: sha(path) for name, path in paths.items()})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_figures(result, output)
    result["figure_sha256"] = {name: sha(output / name) for name in FIGURES}
    (output / "final-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preflight", "runs", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = build_report(parser.parse_args())
    print(json.dumps({"status": result["status"], "ce_targets_per_arm": result["matched_ce_exposure"]}))


if __name__ == "__main__":
    main()
