#!/usr/bin/env python3
"""Report four completed, matched O4 arms without opening held-out test splits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.lm_schedule import alpha_for_update

ARMS = ("ordinary", "ordinary-nextlat", "rt", "rt-nextlat")
SPLITS = ("dev", "retention_dev")
LABELS = {"ordinary": "Ordinary", "ordinary-nextlat": "Ordinary + NextLat",
          "rt": "RT", "rt-nextlat": "RT + NextLat"}
SCHEMA = "olmo-o4-comparison-v1"
BOOTSTRAP_SEED = 20260922
BOOTSTRAP_RESAMPLES = 1000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _close(actual, expected):
    return _finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9)


def validate_metric(metric, *, documents=False):
    require(metric.get("schema") == "olmo-lm-evaluation-v1", "Unexpected evaluation schema")
    count, total = metric.get("ce_count"), metric.get("ce_sum")
    require(_integer(count, 1) and _finite(total) and total >= 0, "Invalid evaluation NLL sum/count")
    require(_close(metric.get("mean_nll"), total / count), "Evaluation NLL mean differs from token sums")
    require(_close(metric.get("perplexity"), math.exp(total / count)), "Evaluation perplexity differs from NLL")
    correct = metric.get("next_token_correct")
    require(_integer(correct) and correct <= count, "Invalid evaluation correct-token count")
    require(_close(metric.get("next_token_accuracy"), correct / count), "Evaluation accuracy differs from token counts")
    require(_integer(metric.get("documents"), 1), "Evaluation needs nonempty windows")
    if documents:
        records = metric.get("document_records")
        require(isinstance(records, list) and len(records) == metric["documents"], "Missing final document/window records")
        identities = set()
        for row in records:
            require(all(_integer(row.get(k)) for k in ("batch_index", "row_index", "document_id", "ce_count", "next_token_correct")),
                    "Invalid document/window identity or count")
            identity = (row["batch_index"], row["row_index"])
            require(identity not in identities, "Duplicate evaluation window occurrence")
            identities.add(identity)
            require(_finite(row.get("ce_sum")) and row["ce_sum"] >= 0, "Invalid per-window NLL")
            require(row["next_token_correct"] <= row["ce_count"], "Invalid per-window accuracy count")
            if row["ce_count"]:
                require(_close(row.get("mean_nll"), row["ce_sum"] / row["ce_count"]), "Per-window NLL mean differs")
            else:
                require(row["ce_sum"] == 0 and row.get("mean_nll") is None, "Empty window must contribute zero NLL")
        require(sum(row["ce_count"] for row in records) == count, "Document records disagree with total targets")
        require(sum(row["next_token_correct"] for row in records) == correct, "Document records disagree with correct targets")
        require(_close(math.fsum(row["ce_sum"] for row in records), total), "Document records disagree with total NLL")
    return metric


def _window_signature(metric):
    return [(row["batch_index"], row["row_index"], row["document_id"], row["ce_count"])
            for row in metric["document_records"]]


def _clusters(metric):
    totals = {}
    for row in metric["document_records"]:
        document = row["document_id"]
        if document not in totals:
            totals[document] = [0.0, 0]
        totals[document][0] += row["ce_sum"]
        totals[document][1] += row["ce_count"]
    return {document: values for document, values in sorted(totals.items()) if values[1]}


def document_cluster_contrast(metrics, coefficients, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Paired cluster bootstrap of a linear contrast of token-weighted NLLs.

    Merge every selected window of an original document, sample documents with
    replacement, then recompute summed NLL / summed target count. The same
    sampled document multiplicities apply to every arm. This is neither a
    bootstrap of independent windows nor a mean of document-average NLLs.
    """
    require(metrics and set(metrics) == set(coefficients), "Contrast metrics/coefficients must match")
    require(_integer(resamples, 1) and _integer(seed), "Invalid bootstrap settings")
    require(all(_finite(v) for v in coefficients.values()), "Invalid contrast coefficients")
    first = next(iter(metrics.values()))
    for metric in metrics.values():
        validate_metric(metric, documents=True)
        require(_window_signature(metric) == _window_signature(first), "Paired evaluation window/document/target alignment differs")
    groups = {name: _clusters(metric) for name, metric in metrics.items()}
    documents = list(next(iter(groups.values())))
    require(documents, "No scored document clusters")
    counts = np.asarray([next(iter(groups.values()))[doc][1] for doc in documents], dtype=np.float64)
    difference = np.zeros(len(documents), dtype=np.float64)
    for name, values in groups.items():
        require(list(values) == documents and all(values[doc][1] == counts[i] for i, doc in enumerate(documents)),
                "Paired document-cluster target counts differ")
        difference += coefficients[name] * np.asarray([values[doc][0] for doc in documents], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, dtype=np.float64)
    # Bound temporary bootstrap memory independently of document count.
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(documents), size=(min(100, resamples - start), len(documents)))
        sampled[start:start + len(indices)] = difference[indices].sum(1) / counts[indices].sum(1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"estimate": float(difference.sum() / counts.sum()), "ci95": [float(low), float(high)],
            "document_clusters": len(documents), "windows": first["documents"], "ce_count": int(counts.sum()),
            "resamples": resamples, "seed": seed, "coefficients": coefficients,
            "method": "paired original-document cluster bootstrap; token-weighted ratio; percentile 95% interval",
            "scope": "Evaluation-document variability only; excludes training-seed and pretraining-overlap uncertainty"}


def paired_document_bootstrap(left, right, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    return document_cluster_contrast({"left": left, "right": right}, {"left": 1.0, "right": -1.0},
                                     resamples=resamples, seed=seed)


def _preflight_sources(preflight, configuration):
    old, new = preflight.get("source_hashes", {}), configuration.get("source_hashes", {})
    require(old and set(old) == set(new), "Preflight/training source inventories differ")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in [*old.values(), *new.values()]),
            "Malformed source hashes")
    changed = {key for key in old if old[key] != new[key]}
    amendment = configuration.get("provenance_amendment")
    if changed:
        require(changed == {"scripts/olmo_o4_train.py"}, "Unapproved preflight/training source changes")
        require(isinstance(amendment, dict), "Changed training source requires an explicit preflight amendment")
        path = "scripts/olmo_o4_train.py"
        require(amendment.get("changed_file") == path and amendment.get("old_sha256") == old[path]
                and amendment.get("new_sha256") == new[path] and bool(amendment.get("reason"))
                and isinstance(amendment.get("previous_source_snapshot"), str)
                and bool(amendment["previous_source_snapshot"]), "Invalid preflight source amendment")
    else:
        require(amendment is None, "A source amendment must correspond to an actual recorded source difference")
    return amendment


def _checkpoint(report, configuration):
    schedule = configuration["schedule"]
    records = [row for row in report.get("checkpoints", []) if row.get("optimizer_updates") == schedule["total_updates"]]
    require(len(records) == 1, "Require exactly one retained endpoint checkpoint")
    record = records[0]
    storage = record.get("storage", {})
    require(record.get("input_tokens") == schedule["total_tokens"], "Checkpoint exposure differs from endpoint")
    require(isinstance(record.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Invalid checkpoint SHA256")
    require(_integer(record.get("size_bytes"), 1), "Invalid checkpoint byte count")
    require(storage.get("sha256") == record["sha256"] and storage.get("size_bytes") == record["size_bytes"]
            and storage.get("generation") and storage.get("md5_base64") and storage.get("verification"),
            "Endpoint checkpoint lacks a matching verified storage receipt")
    prefix = report["configuration"].get("storage_prefix", "")
    require(prefix.startswith("gs://fast-chunks/") and storage.get("uri", "").startswith(prefix + "/" + report["arm"] + "/"),
            "Endpoint checkpoint URI differs from the recorded arm namespace")
    return {key: record[key] for key in ("sha256", "size_bytes", "optimizer_updates", "input_tokens", "storage")}


def build_comparison(preflight, configuration, arm_reports):
    """Validate complete lineages before computing any comparative conclusion."""
    require(preflight.get("schema") == "olmo-o4-preflight-v1" and preflight.get("status") == "passed"
            and preflight.get("finished_utc"), "Require a completed passing O4 preflight")
    require(configuration.get("schema") == "olmo-o4-pilot-config-v1", "Unexpected frozen O4 configuration")
    require(set(arm_reports) == set(ARMS), "Require all four O4 arms")
    amendment = _preflight_sources(preflight, configuration)
    schedule = configuration["schedule"]
    require(preflight.get("schedule") == schedule, "Preflight and training schedules differ")
    alpha_for_update(schedule, schedule["total_updates"], schedule["total_tokens"])
    require(preflight.get("selected_batch_size") == schedule["batch_size"], "Selected batch differs from schedule")
    require(preflight.get("data_manifest_sha256") == configuration.get("data_manifest_sha256"), "Preflight data identity differs")
    require(preflight.get("checkpoint", {}).get("sha256") == configuration.get("checkpoint_sha256"), "Preflight native checkpoint differs")
    require(configuration.get("final_eval_rows") == 512 and configuration.get("eval_rows") == 128,
            "This report requires the frozen 128-window curves and 512-window final protocol")
    baseline = preflight.get("baseline", {}).get("ordinary", {})
    require(set(baseline) == set(SPLITS), "Missing ordinary preflight baselines")
    for metric in baseline.values():
        validate_metric(metric, documents=True)
        require(metric["documents"] <= configuration["final_eval_rows"] and metric.get("mode") == {
            "selected_layers": [], "alpha": 0.0}, "Ordinary preflight baseline window count/mode differs")
    summaries, curves, shared_counts = {}, {}, None
    for arm in ARMS:
        report = arm_reports[arm]
        require(report.get("schema") == "olmo-o4-arm-v1" and report.get("status") == "completed"
                and report.get("arm") == arm and report.get("finished_utc"), f"Arm {arm} is not completed")
        resolved = report.get("configuration", {})
        require(resolved.get("arm") == arm and {key: value for key, value in resolved.items() if key not in ("arm", "storage_prefix")} == configuration,
                f"Arm {arm} configuration differs")
        source = report.get("source_fingerprint", {})
        require(source.get("checkpoint_sha256") == configuration["checkpoint_sha256"]
                and source.get("code") == configuration["source_hashes"]
                and source.get("data_manifest_sha256") == configuration["data_manifest_sha256"]
                and source.get("runtime") == preflight.get("runtime"), f"Arm {arm} source/data/runtime identity differs")
        counters = report.get("counters", {})
        require(counters.get("optimizer_updates") == schedule["total_updates"]
                and counters.get("input_tokens") == schedule["total_tokens"]
                and report.get("data_cursor") == schedule["used_windows"], f"Arm {arm} has incomplete or mismatched exposure")
        counts = {key: counters.get(key) for key in ("optimizer_updates", "microbatches", "documents", "input_tokens", "ce_positions")}
        require(all(_integer(value, 1) for value in counts.values()), f"Arm {arm} is missing positive exposure counters")
        require(counts["microbatches"] == schedule["total_updates"] and counts["documents"] == schedule["used_windows"]
                and counts["ce_positions"] == schedule["total_tokens"] - schedule["used_windows"], "Exposure does not match full one-window batches/all-target CE")
        require(shared_counts is None or counts == shared_counts, "Four-arm exposure counters differ")
        shared_counts = counts
        if "nextlat" in arm:
            require(counters.get("latent_pairs") == counts["ce_positions"] and _integer(counters.get("kl_triples"), 1), "Enabled NextLat exposure is missing")
        else:
            require(counters.get("latent_pairs") == counters.get("kl_triples") == 0, "Disabled NextLat has auxiliary exposure")
        full = [row for row in report.get("evaluations", []) if row.get("full") is True]
        require(len(full) == 1, f"Arm {arm} requires exactly one full final evaluation")
        final = full[0]
        expected_alpha = 1.0 if arm.startswith("rt") else 0.0
        require(final.get("update") == schedule["total_updates"] and final.get("input_tokens") == schedule["total_tokens"]
                and final.get("alpha") == expected_alpha and final.get("rows_requested") == configuration["final_eval_rows"],
                f"Arm {arm} final evaluation differs from endpoint mode/exposure")
        require(set(final.get("metrics", {})) == set(SPLITS), "Final evaluation splits differ")
        for split, metric in final["metrics"].items():
            validate_metric(metric, documents=True)
            require(_window_signature(metric) == _window_signature(baseline[split]), "Final windows/document identities differ from preflight")
            require(metric.get("precision") == configuration["precision"] and metric.get("mode") == {
                "selected_layers": [0] if arm.startswith("rt") else [], "alpha": expected_alpha}, "Final inference mode/precision differs")
        by_update = {}
        for event in report["evaluations"]:
            if event.get("full") is True:
                continue
            require(event.get("full") is False and event.get("rows_requested") == configuration["eval_rows"], "Learning curve mixes evaluation sample sizes")
            update = event.get("update")
            require(_integer(update) and update <= schedule["total_updates"], "Invalid curve update")
            alpha = alpha_for_update(schedule, update, event["input_tokens"]) if arm.startswith("rt") else 0.0
            require(event.get("alpha") == alpha, "Curve alpha differs from frozen schedule")
            # Check token accounting for ordinary curves as well.
            alpha_for_update(schedule, update, event["input_tokens"])
            require(set(event.get("metrics", {})) == set(SPLITS), "Curve splits differ")
            for metric in event["metrics"].values():
                validate_metric(metric)
                require(metric["documents"] <= configuration["eval_rows"], "Curve includes more than the fixed small prefix")
            require(update not in by_update or by_update[update] == event, "Conflicting duplicate evaluation after resume")
            by_update[update] = event
        required_updates = {0, schedule["warmup_updates"], schedule["ramp_end_update"], schedule["total_updates"]}
        require(required_updates <= set(by_update), "Learning curves omit initial/phase endpoints")
        curves[arm] = [by_update[key] for key in sorted(by_update)]
        for event in curves[arm]:
            for split in SPLITS:
                first = curves[arm][0]["metrics"][split]
                require(event["metrics"][split]["ce_count"] == first["ce_count"]
                        and event["metrics"][split]["documents"] == first["documents"],
                        "Curve sample target/window counts change over time")
        link = report.get("wandb", {}).get("run_url")
        require(isinstance(link, str) and link.startswith("https://wandb.ai/"), "Missing online run link")
        summaries[arm] = {"metrics": final["metrics"], "counters": counters, "checkpoint": _checkpoint(report, configuration),
                          "wandb_url": link, "started_utc": report.get("started_utc"), "finished_utc": report["finished_utc"]}
    require(arm_reports["ordinary-nextlat"]["counters"]["kl_triples"] == arm_reports["rt-nextlat"]["counters"]["kl_triples"],
            "NextLat arms have different triple exposure")
    ordinary_curve = curves["ordinary"]
    for arm, rows in curves.items():
        require([row["update"] for row in rows] == [row["update"] for row in ordinary_curve], "Learning curve evaluation boundaries differ across arms")
        for left, right in zip(rows, ordinary_curve):
            for split in SPLITS:
                require(left["metrics"][split]["ce_count"] == right["metrics"][split]["ce_count"]
                        and left["metrics"][split]["documents"] == right["metrics"][split]["documents"], "Learning curve target/window counts differ across arms")
    comparisons = {}
    for split in SPLITS:
        metrics = {arm: summaries[arm]["metrics"][split] for arm in ARMS}
        comparisons[split] = {"versus_preflight_ordinary": {
            arm: paired_document_bootstrap(metric, baseline[split]) for arm, metric in metrics.items()},
            "between_arms": {f"{left}_minus_{right}": paired_document_bootstrap(metrics[left], metrics[right])
                             for right, left in itertools.combinations(ARMS, 2)},
            "interaction": document_cluster_contrast(metrics, {
                "ordinary": 1.0, "ordinary-nextlat": -1.0, "rt": -1.0, "rt-nextlat": 1.0})}
    return {"schema": SCHEMA, "status": "completed", "configuration": configuration, "exposure": shared_counts,
            "preflight_ordinary": baseline, "preflight_wandb": preflight.get("wandb", {}).get("run_url"),
            "preflight_provenance_amendment": amendment, "arms": summaries, "curves": curves, "comparisons": comparisons,
            "difference_convention": "left minus right NLL; negative favors left. Interaction=(RT+NextLat-RT)-(NextLat-ordinary).",
            "qualification": "One seed, approximately 20–22M valid-input-token recovery pilot; equal data exposure, not equal compute. "
                "Development subsets only; tests untouched. All valid same-document targets, reset context per window, unknown OLMo pretraining overlap. "
                "Cluster intervals describe sampled evaluation documents, not seed variability or research efficacy."}


def write_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ("#4b5563", "#2878b5", "#e07a24", "#7a3ba3")
    schedule = summary["configuration"]["schedule"]
    for axis, split, title in zip(axes, SPLITS, ("CodeSearchNet Python development", "WikiText-2 language retention")):
        for arm, color in zip(ARMS, colors):
            rows = summary["curves"][arm]
            axis.plot([row["input_tokens"] / 1e6 for row in rows],
                      [row["metrics"][split]["mean_nll"] for row in rows],
                      label=LABELS[arm], color=color, linewidth=1.8, marker="o", markersize=3)
        start, end = schedule["ramp_start_tokens"] / 1e6, schedule["ramp_end_tokens"] / 1e6
        axis.axvspan(start, end, color="#64748b", alpha=.08)
        axis.axvline(start, color="#64748b", linestyle=":", linewidth=1)
        axis.axvline(end, color="#64748b", linestyle="--", linewidth=1)
        axis.text((start + end) / 2, .98, "RT alpha ramp", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        axis.set(title=title, xlabel="Valid input tokens seen (millions)", ylabel="Token-weighted NLL (nats)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Fixed 128-window development curves · one seed · equal data exposure", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"learning-curves.{suffix}", dpi=180)
    plt.close(fig)


def markdown(summary):
    exposure, comparisons = summary["exposure"], summary["comparisons"]
    lines = ["# OLMo O4: matched Python continuation pilot", "",
             f"All four arms completed **{exposure['optimizer_updates']:,} updates** and **{exposure['input_tokens']:,} valid input tokens** "
             f"({exposure['ce_positions']:,} CE targets) from the same original checkpoint and prepared stream.", "",
             "Final results use the fixed 512-window prefixes (capped by available data). Curves use the separate 128-window prefixes; "
             "the larger final evaluation is never inserted into those curves.", "",
             "| Model | Code NLL | Code perplexity | Code token accuracy | Retention NLL | Retention perplexity | Retention token accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [("Original ordinary checkpoint", summary["preflight_ordinary"])] + [
        (LABELS[arm], summary["arms"][arm]["metrics"]) for arm in ARMS]
    for label, metrics in rows:
        code, retention = metrics["dev"], metrics["retention_dev"]
        lines.append(f"| {label} | {code['mean_nll']:.6f} | {code['perplexity']:.4f} | {100*code['next_token_accuracy']:.3f}% "
                     f"| {retention['mean_nll']:.6f} | {retention['perplexity']:.4f} | {100*retention['next_token_accuracy']:.3f}% |")
    lines.extend(["", "Differences below are **left minus right** token NLL: negative favors the left model. "
                  "Intervals resample original documents, combining their windows first, with 1,000 paired resamples (seed20260922).", "",
                  "| Comparison | Code NLL difference [95% interval] | Retention NLL difference [95% interval] |",
                  "| --- | ---: | ---: |"])
    for name in comparisons["dev"]["between_arms"]:
        left, right = name.split("_minus_")
        entries = [comparisons[split]["between_arms"][name] for split in SPLITS]
        formatted = [f"{item['estimate']:+.6f} [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]" for item in entries]
        lines.append(f"| {LABELS[left]} − {LABELS[right]} | {formatted[0]} | {formatted[1]} |")
    for split, label in (("dev", "Code"), ("retention_dev", "Retention")):
        item = comparisons[split]["interaction"]
        lines.extend(["", f"{label} interaction `(RT+NextLat − RT) − (NextLat − ordinary)`: "
                      f"{item['estimate']:+.6f} nats [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}], "
                      f"from {item['document_clusters']} original-document clusters. Negative means a more favorable "
                      "NextLat effect with RT in this pilot; it is not a replicated interaction finding."])
    lines.extend(["", "![Development learning curves](learning-curves.png)", "",
                  "[Standalone PDF](learning-curves.pdf) · [Full comparisons and paired intervals](final-comparison.json)", "",
                  summary["qualification"], "",
                  "The general-language set is a retention check, not a published WikiText perplexity comparison. "
                  "Code is continued as text rather than executed; token NLL does not establish programming-task success. "
                  "NextLat adds training-only parameters and compute; its predictor is absent from evaluation. "
                  "No FBT or autonomous latent rollout is tested."])
    if summary["preflight_provenance_amendment"]:
        amendment = summary["preflight_provenance_amendment"]
        lines.extend(["", "Preflight provenance amendment: " + amendment["reason"] + " "
                      "The authentic preflight hashes and explicit old/new training-source hashes remain recorded; "
                      "all four learning arms use the same amended frozen configuration."])
    lines.extend(["", "Run and checkpoint records:", ""])
    for arm in ARMS:
        record = summary["arms"][arm]
        checkpoint = record["checkpoint"]["storage"]
        lines.append(f"- [{LABELS[arm]} on W&B]({record['wandb_url']}); "
                     f"[endpoint checkpoint]({checkpoint['uri']}), GCS generation `{checkpoint['generation']}`, "
                     f"SHA256 `{checkpoint['sha256']}`.")
    return "\n".join(lines) + "\n"


def build_report(args):
    path = Path(args.preflight)
    preflight_path = path / "report.json" if path.is_dir() else path
    configuration_path = preflight_path.parent / "configuration.json"
    preflight = json.loads(preflight_path.read_text())
    configuration = json.loads(configuration_path.read_text())
    paths = {arm: Path(args.runs) / arm / "report.json" for arm in ARMS}
    require(all(path.is_file() for path in paths.values()), "Four completed arm report files are required")
    reports = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    summary = build_comparison(preflight, configuration, reports)
    amendment = summary["preflight_provenance_amendment"]
    if amendment:
        snapshot = Path(amendment["previous_source_snapshot"])
        if not snapshot.is_absolute():
            snapshot = ROOT / snapshot
        require(snapshot.is_file() and sha(snapshot) == amendment["old_sha256"], "Amended preflight's previous source snapshot is missing or changed")
    summary.update(created_utc=datetime.now(timezone.utc).isoformat(), report_source_sha256=sha(__file__),
                   input_sha256={"preflight": sha(preflight_path), "configuration": sha(configuration_path),
                                 **{arm: sha(path) for arm, path in paths.items()}})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_figures(summary, output)
    summary["figure_sha256"] = {name: sha(output / name) for name in ("learning-curves.pdf", "learning-curves.png")}
    (output / "final-comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    summary = build_report(parser.parse_args())
    print(json.dumps({"status": summary["status"], "updates": summary["exposure"]["optimizer_updates"],
                      "input_tokens": summary["exposure"]["input_tokens"]}))


if __name__ == "__main__":
    main()
