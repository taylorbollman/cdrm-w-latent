#!/usr/bin/env python3
"""Compare ordinary continuation with retained fusion-only adaptation results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_o4_report as paired
from scripts import olmo_o5c_report as fusion_report
from scripts import olmo_o5d_report as online_report

SCHEMA = "olmo-o5e-final-comparison-v1"
SPLITS = ("dev", "retention_dev")
FROZEN_NAMES = online_report.FUSION_NAMES | {"backbone.fusion.output_scale"}
FIGURES = ("learning-curves.pdf", "learning-curves.png", "endpoint-comparison.pdf", "endpoint-comparison.png")
O5C_PREFLIGHT = ROOT / ".runtime/olmo1b-step60000/o5c-preflight-01"
O5C_RUNS = ROOT / ".runtime/olmo1b-step60000/o5c-pilot-01"
O5D_REPORT = ROOT / "docs/reports/olmo1b-o5d/report.json"
require = paired.require
QUALIFICATION = (
    "One seed and development subsets only; reserved tests are untouched. The ordinary "
    "and fusion-only runs start from the same O5b native backbone and consume the exact "
    "same mixed-data targets, order, reset contexts and update budget. Ordinary trains "
    "all native parameters with LR 1e-5; O5c trains only 8,388,608 fusion parameters "
    "with LR 1e-4 over a frozen backbone. This is an equal-data practical continuation "
    "control, not a pure FBT ablation or a match of trainable capacity, FLOPs, memory "
    "or wall time. Ordinary uses one CE term; O5c's ordinary CE term was constant "
    "with respect to its trainable fusion. Curves use 128 windows; endpoint comparisons "
    "use 512. Online FBT is teacher-forced sequential feedback, not free-running "
    "generation. Bootstrap intervals cover evaluation-document variability only, "
    "excluding training-seed variance and unknown pretraining overlap. WikiText is "
    "a narrow general-text proxy."
)


def _numeric(metric):
    return {**metric, "schema": "olmo-lm-evaluation-v1"}


def validate_container(container, *, rows, documents):
    require(container.get("schema") == "olmo-fbt-evaluation-v1", "Unexpected ordinary evaluation container")
    passes = container.get("passes")
    require(isinstance(passes, list) and len(passes) == 1, "Ordinary evaluation requires exactly one pass")
    metric = passes[0]
    mode = {"enabled": False, "num_passes": 1, "beta": 0., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    require(metric.get("schema") == "olmo-fbt-pass-evaluation-v1" and metric.get("execution") == "finite"
            and metric.get("pass_index") == 0 and metric.get("mode") == mode,
            "Ordinary execution must disable FBT and use K1/beta0/empty RT")
    require(metric.get("precision") == "bf16_mixed", "Ordinary evaluation precision differs")
    for key in ("mode", "precision", "execution", "positions_per_chunk", "batches", "documents", "input_tokens"):
        require(container.get(key) == metric.get(key), "Container/pass metadata differs")
    paired.validate_metric(_numeric(metric), documents=documents)
    require(metric["documents"] == rows and metric.get("batches") == math.ceil(rows / 8)
            and metric.get("positions_per_chunk") == 128, "Frozen evaluation selection/batching differs")
    require(metric["input_tokens"] == metric["ce_count"] + rows and metric["input_tokens"] <= rows * 512,
            "Ordinary evaluation is not bounded all-target one-document-per-row CE")
    if documents:
        for i, row in enumerate(metric["document_records"]):
            require(row["batch_index"] == i // 8 and row["row_index"] == i % 8 and row["ce_count"] < 512,
                    "Evaluation record order or context length differs")
    return metric


def _ordinary_equal(actual, expected):
    require(actual["ce_count"] == expected["ce_count"] and actual["next_token_correct"] == expected["next_token_correct"]
            and abs(actual["mean_nll"] - expected["mean_nll"]) <= 2e-6,
            "Ordinary starting scores differ from the shared frozen backbone")


def _ownership(config, initial, frozen, layout):
    model = config["model_config"]
    width, hidden, layers, vocab = (model[k] for k in ("model_dim", "mlp_intermediate_size", "num_layers", "vocab_size"))
    expected = {"backbone.backbone.transformer.wte.weight": [vocab, width]}
    for layer in range(layers):
        prefix = f"backbone.backbone.transformer.blocks.{layer}."
        expected.update({prefix + "att_proj.weight": [3 * width, width], prefix + "attn_out.weight": [width, width],
                         prefix + "ff_proj.weight": [2 * hidden, width], prefix + "ff_out.weight": [width, hidden]})
    require(set(frozen) == FROZEN_NAMES and set(initial) == set(expected) | FROZEN_NAMES,
            "Frozen/trainable state ownership differs from native backbone plus unused fusion")
    require(all(initial[name] == value for name, value in frozen.items()), "Initial frozen fusion hashes differ")
    require(isinstance(layout, list) and len(layout) == len(expected), "Native trainable parameter count differs")
    require({row.get("name") for row in layout} == set(expected), "Only all native parameters must train")
    for row in layout:
        shape = expected[row["name"]]
        require(row.get("shape") == shape and row.get("numel") == math.prod(shape), "Native parameter shape/size differs")
    return sum(math.prod(shape) for shape in expected.values())


def _event(event, config, *, full, source):
    update = event.get("update")
    plan = config["schedule"]["arms"]["mixed"]
    require(paired._integer(update) and update <= plan["total_updates"], "Evaluation update outside fixed budget")
    require(event.get("full") is full and event.get("rows_requested") == (512 if full else 128)
            and event.get("beta") == 0., "Ordinary evaluation selection or beta differs")
    expected = {domain: {key: values[update] for key, values in fields.items()}
                for domain, fields in plan["domain_prefixes"].items()}
    require(event.get("domain_counts") == expected, "Evaluation domain exposure differs from exact mixed plan")
    require(event.get("ce_positions") == plan["batch_ce_prefix"][update]
            and event.get("input_tokens") == plan["batch_token_prefix"][update], "Evaluation exposure differs from plan")
    require(set(event.get("metrics", {})) == set(SPLITS), "Evaluation domains differ")
    for split, container in event["metrics"].items():
        metric = validate_container(container, rows=512 if full else 128, documents=full)
        require(metric["ce_count"] == source[split]["passes"][0]["ce_count"], "Evaluation target counts differ from source")
        if full:
            require(paired._window_signature(metric) == paired._window_signature(source[split]["passes"][0]),
                    "Final ordinary and reference target selections differ")
    return event


def build_comparison(preflight, config, ordinary, fusion, online):
    require(preflight.get("schema") == "olmo-o5e-preflight-v1" and preflight.get("status") == "passed"
            and preflight.get("finished_utc"), "Require passing completed O5e preflight")
    require(config.get("schema") == "olmo-o5e-config-v1", "Unexpected O5e configuration")
    require(ordinary.get("schema") == "olmo-o5e-arm-v1" and ordinary.get("status") == "completed"
            and ordinary.get("arm") == "ordinary" and ordinary.get("finished_utc"), "Ordinary arm is incomplete")
    require(fusion.get("schema") == fusion_report.SCHEMA and fusion.get("status") == "completed",
            "Require completed validated O5c comparison")
    online_report.validate_report(online)
    reference = fusion["configuration"]
    schedule = config["schedule"]
    require(config.get("data_plan_arm") == "mixed" and set(schedule.get("arms", {})) == {"mixed"}
            and schedule["arms"]["mixed"] == reference["schedule"]["arms"]["mixed"],
            "Ordinary must reuse the exact O5c mixed segment/target plan")
    for name in ("total_updates", "ce_per_update", "total_ce"):
        require(schedule.get(name) == reference["schedule"][name], "Shared mixed budget differs")
    require(schedule["total_updates"] == 512 and schedule["ce_per_update"] == 8192
            and schedule["total_ce"] == 4194304, "Approved update/CE budget differs")
    for name in ("model_config", "source_checkpoint_sha256", "data_manifest_sha256", "base_data_manifest_sha256",
                 "precision", "attention_backend", "eval_batch_size", "physical_batch_size", "seed", "eval_rows", "final_eval_rows"):
        require(config.get(name) == reference.get(name), f"Reference {name} differs")
    require(config.get("fbt_enabled") is False and config.get("beta") == 0. and config.get("num_passes") == 1
            and config.get("rt_layers") == [] and config.get("nextlat_enabled") is False
            and config.get("native_backbone_frozen") is False and config.get("objective") == "single_ordinary_ce"
            and config.get("gamma") == 1., "Ordinary must train native parameters with one CE and feedback/RT/NextLat disabled")
    require(config.get("lr") == 1e-5 and config.get("warmup_updates") == 50 and config.get("betas") == [.9, .95]
            and config.get("eps") == 1e-8 and config.get("weight_decay") == .1 and config.get("max_grad_norm") == 1.,
            "Ordinary optimizer differs from the fixed native continuation settings")
    require(ordinary.get("configuration") == config, "Ordinary arm configuration differs from preflight")
    sources = fusion_report._hashes(config.get("source_hashes"), "source")
    require(preflight.get("source_hashes") == sources and all(sources.get(k) == v for k, v in reference["source_hashes"].items()),
            "Frozen runtime/source lineage differs")
    fingerprint = ordinary.get("source_fingerprint", {})
    require(fingerprint.get("code") == sources and fingerprint.get("runtime") == preflight.get("runtime")
            and fingerprint.get("source_checkpoint_sha256") == config["source_checkpoint_sha256"]
            and fingerprint.get("checkpoint_sha256") == config["source_checkpoint_sha256"]
            and fingerprint.get("data_manifest_sha256") == config["data_manifest_sha256"], "Ordinary source/data/runtime fingerprint differs")
    initial = fusion_report._hashes(ordinary.get("full_state_initial"), "initial state")
    require(initial == config.get("full_state_initial") == preflight.get("full_state_initial") == online["endpoints"]["source"]["state_before"],
            "Ordinary does not start from the exact shared O5b model state")
    frozen = fusion_report._hashes(config.get("frozen_state_initial"), "frozen fusion")
    require(frozen == preflight.get("frozen_state_initial") == ordinary.get("frozen_state_initial") == ordinary.get("frozen_state_final"),
            "Unused fusion weights or fixed scale changed")
    final_state = fusion_report._hashes(ordinary.get("full_state_final"), "final state")
    require(set(final_state) == set(initial) and all(final_state[name] == value for name, value in frozen.items()),
            "Final model ownership or unused fusion differs")
    layout = config.get("trainable_layout")
    require(layout == preflight.get("trainable_layout") == ordinary.get("trainable_layout"), "Trainable ownership differs")
    trainable = _ownership(config, initial, frozen, layout)
    require(online["endpoints"]["source"]["checkpoint"]["sha256"] == config["source_checkpoint_sha256"]
            and online["endpoints"]["mixed"]["checkpoint"]["sha256"] == fusion["arms"]["mixed"]["checkpoint"]["sha256"],
            "Online reference checkpoint differs from the source or O5c mixed endpoint")
    baseline, small = preflight.get("baseline", {}), preflight.get("baseline_small", {})
    require(set(baseline) == set(small) == set(SPLITS), "Missing full and curve ordinary preflight baselines")
    for split in SPLITS:
        full_metric = validate_container(baseline[split], rows=512, documents=True)
        small_metric = validate_container(small[split], rows=128, documents=False)
        _ordinary_equal(full_metric, fusion["source_baseline"][split]["passes"][0])
        _ordinary_equal(small_metric, fusion["source_baseline_small"][split]["passes"][0])
        require(paired._window_signature(full_metric) == paired._window_signature(fusion["source_baseline"][split]["passes"][0]),
                "Ordinary preflight window signatures differ from source")
    plan = schedule["arms"]["mixed"]
    counters = ordinary.get("counters", {})
    require(counters.get("optimizer_updates") == 512 and ordinary.get("data_cursor") == 512,
            "Ordinary arm stopped before the full mixed budget")
    expected_counts = {domain: {key: values[512] for key, values in fields.items()} for domain, fields in plan["domain_prefixes"].items()}
    require(ordinary.get("domain_counts") == expected_counts, "Final domain exposure differs from shared mixed plan")
    for name, prefix in (("ce_positions", "batch_ce_prefix"), ("input_tokens", "batch_token_prefix"), ("documents", "batch_row_prefix")):
        require(counters.get(name) == plan[prefix][-1], "Final native continuation exposure differs")
    physical = config["physical_batch_size"]
    expected_microbatches = sum(math.ceil((right - left) / physical) for left, right in zip(plan["batch_row_prefix"], plan["batch_row_prefix"][1:]))
    require(counters.get("microbatches") == expected_microbatches and counters.get("latent_pairs") == counters.get("kl_triples") == 0,
            "Auxiliary or microbatch counters differ")
    full_events = [event for event in ordinary.get("evaluations", []) if event.get("full") is True]
    require(len(full_events) == 1 and full_events[0].get("update") == 512, "Require one full ordinary endpoint evaluation")
    final = _event(full_events[0], config, full=True, source=fusion["source_baseline"])
    curves = {}
    for event in ordinary["evaluations"]:
        if event.get("full") is True:
            continue
        _event(event, config, full=False, source=fusion["source_baseline_small"])
        require(event["update"] not in curves, "Duplicate ordinary curve boundary")
        curves[event["update"]] = event
    expected_updates = [event["update"] for event in fusion["curves"]["mixed"]]
    require(sorted(curves) == expected_updates == [0, 50, *range(64, 513, 64)], "Ordinary/fusion curve boundaries differ")
    for split in SPLITS:
        _ordinary_equal(curves[0]["metrics"][split]["passes"][0], small[split]["passes"][0])
    endpoint_metrics = {"ordinary": final["metrics"], "fusion_K2": fusion["arms"]["mixed"]["metrics"]}
    online_cases = [row for row in online["cases"] if row["endpoint"] == "mixed" and row["case"]["section"] == "full_context" and row["case"]["passes"] is None]
    require(len(online_cases) == 1, "Missing full-context repaired online endpoint")
    endpoint_metrics["fusion_online"] = online_cases[0]["metrics"]
    comparisons = {}
    for split in SPLITS:
        native = final["metrics"][split]["passes"][0]
        references = {"source_ordinary": fusion["source_baseline"][split]["passes"][0],
                      "fusion_K2": endpoint_metrics["fusion_K2"][split]["passes"][-1],
                      "fusion_online": endpoint_metrics["fusion_online"][split]["passes"][-1]}
        comparisons[split] = {"ordinary_minus_" + name: paired.paired_document_bootstrap(_numeric(native), _numeric(metric))
                              for name, metric in references.items()}
    link = ordinary.get("wandb", {}).get("run_url")
    require(isinstance(link, str) and link.startswith("https://wandb.ai/"), "Missing ordinary W&B run")
    return {"schema": SCHEMA, "status": "completed", "configuration": config, "qualification": QUALIFICATION,
        "optimizer_updates": 512, "matched_ce_targets": 4194304, "trainable_parameters": {"ordinary": trainable,
            "fusion_only": sum(row["numel"] for row in reference["trainable_layout"])},
        "source_baseline": fusion["source_baseline"], "source_baseline_small": fusion["source_baseline_small"],
        "endpoint_metrics": endpoint_metrics, "comparisons": comparisons,
        "curves": {"ordinary": [curves[u] for u in expected_updates], "fusion_K2": fusion["curves"]["mixed"]},
        "ordinary": {"checkpoint": fusion_report._checkpoint(ordinary, config), "counters": counters,
            "domain_counts": expected_counts, "wandb_url": link, "started_utc": ordinary.get("started_utc"),
            "finished_utc": ordinary["finished_utc"], "frozen_state_initial": frozen, "frozen_state_final": ordinary["frozen_state_final"],
            "full_state_initial": initial, "trainable_layout": layout},
        "reference_checkpoints": {name: online["endpoints"][name]["checkpoint"] for name in ("source", "mixed")},
        "reference_wandb": {"fusion": fusion["arms"]["mixed"]["wandb_url"], "online": online.get("wandb", {}).get("run_url")}}


def markdown(result):
    lines = ["# O5e: ordinary continuation on the same mixed data", "",
        f"Completed **{result['optimizer_updates']:,} updates / {result['matched_ce_targets']:,} supervised CE targets** from the same O5b native backbone as O5c. Data targets, order and contexts match the fusion-only mixed run exactly.", "",
        "| Training path | Trainable parameters | Peak learning rate | Training CE |", "| --- | ---: | ---: | --- |",
        f"| O5e ordinary | {result['trainable_parameters']['ordinary']:,} | 1e-5 | One ordinary CE |",
        f"| O5c fusion only | {result['trainable_parameters']['fusion_only']:,} | 1e-4 | Trainable feedback CE plus constant ordinary CE |", "",
        "Final 512-window evaluation, maximum512 tokens/window, batch8. Lower NLL is better.", "",
        "| Path | Code NLL | WikiText NLL | Code accuracy | WikiText accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    rows = [("Shared starting ordinary", {s: result["source_baseline"][s]["passes"][0] for s in SPLITS})]
    rows += [(label, {s: result["endpoint_metrics"][name][s]["passes"][-1] for s in SPLITS})
             for name, label in (("ordinary", "O5e trained ordinary"), ("fusion_K2", "O5c fusion, K2"), ("fusion_online", "O5c fusion, exact online"))]
    for label, values in rows:
        lines.append(f"| {label} | {values['dev']['mean_nll']:.6f} | {values['retention_dev']['mean_nll']:.6f} | {values['dev']['next_token_accuracy']:.3%} | {values['retention_dev']['next_token_accuracy']:.3%} |")
    lines += ["", "Paired original-document bootstrap intervals: 1,000 resamples, seed20260922. Negative NLL differences favor O5e ordinary.", "",
        "| Contrast | Code NLL difference [95% interval] | WikiText NLL difference [95% interval] |", "| --- | ---: | ---: |"]
    for name in ("source_ordinary", "fusion_K2", "fusion_online"):
        values = [result["comparisons"][split]["ordinary_minus_" + name] for split in SPLITS]
        cells = [f"{v['estimate']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" for v in values]
        lines.append(f"| Ordinary − {name.replace('_', ' ')} | {' | '.join(cells)} |")
    lines += ["", QUALIFICATION, "", "![Matched learning curves](learning-curves.png)", "",
        "[Curves PDF](learning-curves.pdf) · [Endpoint comparison PDF](endpoint-comparison.pdf) · [Full comparison](final-comparison.json)", "",
        f"[Ordinary run on W&B]({result['ordinary']['wandb_url']}) · [Fusion run on W&B]({result['reference_wandb']['fusion']})", "",
        f"Retained ordinary checkpoint: `{result['ordinary']['checkpoint']['storage']['uri']}`; SHA256 `{result['ordinary']['checkpoint']['sha256']}`.", ""]
    return "\n".join(lines)


def write_figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, split in zip(axes, SPLITS):
        for name, index, label, color in (("ordinary", 0, "Ordinary: native weights train", "#904d18"),
                                         ("fusion_K2", 1, "Fusion only: K2 feedback pass", "#187b9f")):
            rows = result["curves"][name]
            axis.plot([r["ce_positions"] / 1e6 for r in rows], [r["metrics"][split]["passes"][index]["mean_nll"] for r in rows],
                      "o-", markersize=3, label=label, color=color)
        axis.axhline(result["source_baseline_small"][split]["passes"][0]["mean_nll"], ls="--", color="gray", label="Shared starting ordinary")
        axis.set(title="Code" if split == "dev" else "WikiText", xlabel="Additional supervised CE targets (millions)", ylabel="Token NLL (nats)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Same mixed data and update budget · 128-window curves · different trainable capacity and LR")
    for suffix in ("pdf", "png"): fig.savefig(Path(output) / f"learning-curves.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    names = ("source_ordinary", "fusion_K2", "fusion_online")
    for axis, split in zip(axes, SPLITS):
        values = [result["comparisons"][split]["ordinary_minus_" + name] for name in names]
        estimates = [v["estimate"] for v in values]
        errors = [[max(0., v["estimate"] - v["ci95"][0]) for v in values], [max(0., v["ci95"][1] - v["estimate"]) for v in values]]
        axis.errorbar(range(3), estimates, yerr=errors, fmt="o", color="#904d18", capsize=5)
        axis.axhline(0, ls="--", color="gray")
        axis.set(xticks=range(3), xticklabels=("Starting ordinary", "Fusion K2", "Fusion online"),
            title="Code" if split == "dev" else "WikiText", ylabel="O5e ordinary − reference NLL (nats)")
        axis.grid(alpha=.2)
    fig.suptitle("512-window endpoints · paired original-document 95% intervals\nNegative favors trained ordinary; equal data, different capacity and compute", fontsize=11)
    for suffix in ("pdf", "png"): fig.savefig(Path(output) / f"endpoint-comparison.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preflight", "run-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {"preflight": args.preflight / "report.json", "configuration": args.preflight / "configuration.json",
             "ordinary": args.run_dir / "report.json", "online_reference": O5D_REPORT,
             "fusion_preflight": O5C_PREFLIGHT / "report.json", "fusion_config": O5C_PREFLIGHT / "configuration.json",
             "fusion_code": O5C_RUNS / "code/report.json", "fusion_mixed": O5C_RUNS / "mixed/report.json"}
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fusion = fusion_report.build_comparison(values["fusion_preflight"], values["fusion_config"],
        {"code": values["fusion_code"], "mixed": values["fusion_mixed"]})
    result = build_comparison(values["preflight"], values["configuration"], values["ordinary"], fusion, values["online_reference"])
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), input_sha256={k: paired.sha(v) for k, v in paths.items()},
        report_source_sha256=paired.sha(__file__), helper_source_sha256={"scripts/olmo_o4_report.py": paired.sha(paired.__file__),
            "scripts/olmo_o5c_report.py": paired.sha(fusion_report.__file__), "scripts/olmo_o5d_report.py": paired.sha(online_report.__file__)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_figures(result, args.output_dir)
    result["figure_sha256"] = {name: paired.sha(args.output_dir / name) for name in FIGURES}
    (args.output_dir / "final-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "results.md").write_text(markdown(result))
    print(json.dumps({"status": result["status"], "ordinary_parameters": result["trainable_parameters"]["ordinary"]}))


if __name__ == "__main__":
    main()
