#!/usr/bin/env python3
"""Localize eager/graph gradient differences without changing model equations.

Every comparison holds weights, inputs, precision and the selected backend fixed.
Default-stream eager, capture-stream eager and repeated graph replay are separate
controls. Disagreement is retained as an observation rather than stopping the
matrix at its first failed pair. This script does not clear graph correctness,
loosen the original probe's budgets, train, or time a language-model objective.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_f1_observe import summarize_profiler
from scripts.olmo_f2_graph_probe import (
    SOURCE_FILES as PROBE_SOURCES, allocate_gradients, gradient_addresses,
    make_static_inputs, parse_layers, stack_step, tensor_comparison, validate_options,
)
from scripts.olmo_validation import require_container_gpu


SOURCE_FILES = tuple(sorted(set(PROBE_SOURCES) | {
    "scripts/olmo_f2_graph_localize.py", "scripts/olmo_f1_observe.py",
}))


def backend_context(name):
    if name == "auto":
        return nullcontext()
    selected = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION}
    if name not in selected:
        raise ValueError("Unknown diagnostic attention backend")
    return sdpa_kernel(selected[name])


def snapshot(model, hidden):
    """At most three CPU gradient copies; no persistent model-sized GPU clones."""
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Persistent gradient buffer disappeared: {name}")
        gradients[name] = parameter.grad.detach().to("cpu", copy=True)
    return {"hidden": hidden.detach().to("cpu", copy=True), "gradients": gradients}


def compare_snapshots(actual, reference, *, name):
    if actual["gradients"].keys() != reference["gradients"].keys():
        raise ValueError("Compared snapshots have different parameter ownership")
    hidden = tensor_comparison(actual["hidden"], reference["hidden"])
    gradients = {key: tensor_comparison(actual["gradients"][key], expected)
                 for key, expected in reference["gradients"].items()}
    changed, failed, by_layer = [], [], {}
    for key, values in gradients.items():
        match = re.search(r"(?:^|\.)blocks\.(\d+)\.", key)
        if match:
            layer = int(match.group(1))
            by_layer[layer] = max(by_layer.get(layer, 0.), values["relative_l2"])
            if not values["bitwise_equal"]:
                changed.append(layer)
            if not values["passed"]:
                failed.append(layer)
    return {"name": name, "hidden": hidden, "gradients": gradients,
            "all_bitwise_equal": hidden["bitwise_equal"] and all(v["bitwise_equal"] for v in gradients.values()),
            "passed": hidden["passed"] and all(v["passed"] for v in gradients.values()),
            "max_gradient_relative_l2": max((v["relative_l2"] for v in gradients.values()), default=0.),
            "layer_max_gradient_relative_l2": {str(k): by_layer[k] for k in sorted(by_layer)},
            "highest_layer_with_any_gradient_difference": max(changed) if changed else None,
            "highest_layer_outside_original_budget": max(failed) if failed else None,
            "comparison_reductions": "CPU FP32 using the original graph probe tensor_comparison budgets"}


def run_controls(model, args, report, on_comparison):
    mode = RTMode(args.rt_layers, args.alpha)
    ids, cotangent = make_static_inputs(model.config, args.batch_size, args.length, device="cuda")
    expected_addresses = allocate_gradients(model)
    expected_versions = {name: (p.data_ptr(), p._version) for name, p in model.named_parameters()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def step():
        with backend_context(args.backend):
            return stack_step(model, ids, cotangent, mode, args.precision)

    def execute_eager(*, side_stream=False):
        if side_stream:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                hidden = step()
            torch.cuda.current_stream().wait_stream(stream)
        else:
            hidden = step()
        torch.cuda.synchronize()
        return snapshot(model, hidden)

    def record(actual, reference, name):
        row = compare_snapshots(actual, reference, name=name)
        row["gradient_addresses_unchanged"] = gradient_addresses(model) == expected_addresses
        row["weights_storage_versions_unchanged"] = expected_versions == {
            key: (p.data_ptr(), p._version) for key, p in model.named_parameters()}
        row["passed"] &= row["gradient_addresses_unchanged"] and row["weights_storage_versions_unchanged"]
        report["comparisons"].append(row)
        on_comparison(row)

    report["stage"] = "warmup"
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    baseline = execute_eager()
    report["stage"] = "eager_controls"
    for index in range(args.repeats):
        current = execute_eager()
        record(current, baseline, f"eager_default_repeat_{index+1}")
        del current
    current = execute_eager(side_stream=True)
    record(current, baseline, "eager_capture_stream_vs_default_stream")
    del current

    report["stage"] = "capture"
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        graph_hidden = step()
    torch.cuda.synchronize()
    report["capture_succeeded"] = True
    report["stage"] = "graph_controls"
    graph.replay()
    torch.cuda.synchronize()
    graph_baseline = snapshot(model, graph_hidden)
    record(graph_baseline, baseline, "graph_vs_eager_default")
    for index in range(args.repeats):
        graph.replay()
        torch.cuda.synchronize()
        current = snapshot(model, graph_hidden)
        record(current, graph_baseline, f"graph_repeat_{index+1}")
        del current
    current = execute_eager(side_stream=True)
    record(graph_baseline, current, "graph_vs_eager_capture_stream")
    del current
    current = execute_eager()
    record(current, baseline, "eager_default_after_capture_vs_before")
    record(graph_baseline, current, "graph_vs_eager_default_after_capture")
    del current, graph_baseline, baseline
    gc.collect()

    # A separate trace records the actually executed ordinary SDPA backend.
    # It is excluded from all numerical comparisons and does not measure TPS.
    report["stage"] = "dispatch_trace"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
            profile_memory=False, with_stack=False) as profiler:
        step()
        torch.cuda.synchronize()
    report["dispatch"] = summarize_profiler(profiler, rt_selected_layers=args.rt_layers)
    report["gradient_addresses_unchanged"] = gradient_addresses(model) == expected_addresses
    report["weights_storage_versions_unchanged"] = expected_versions == {
        name: (p.data_ptr(), p._version) for name, p in model.named_parameters()}
    report["structural_invariants_passed"] = report["gradient_addresses_unchanged"] and report["weights_storage_versions_unchanged"]
    report["all_comparisons_within_original_budget"] = report["structural_invariants_passed"] and all(row["passed"] for row in report["comparisons"])
    report["all_comparisons_bitwise_equal"] = all(row["all_bitwise_equal"] for row in report["comparisons"])
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    report["stage"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rt-layers", type=parse_layers, default=(0,))
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--precision", choices=("fp32", "bf16_mixed"), default="bf16_mixed")
    parser.add_argument("--attention-precision", choices=("mixed", "fp32"), default="mixed")
    parser.add_argument("--backend", choices=("auto", "math", "flash", "cudnn"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    config = OLMoConfig.native_1b()
    validate_options(args, config)
    runtime = {key: str(value) for key, value in require_container_gpu().items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-f2-graph-localization-v1", "status": "running", "stage": "load",
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_hashes": {name: sha256_file(ROOT/name) for name in SOURCE_FILES},
        "comparisons": [], "capture_succeeded": False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "scope": "Fixed native weights/tokens/cotangent; eager-repeat, side-stream and graph-repeat controls",
        "limitations": ["No CE/NextLat/FBT/optimizer or changing-token/weight correctness clearance",
            "Forced backends are separate numerical executions, compared only with themselves",
            "Observed discrepancy is retained without changing the original graph probe's budgets",
            "Two or a few repeat pairs do not establish universal determinism"]}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f2-graph-localization", name="olmo-1b-f2-localize-"+args.output_dir.name)
    began = time.perf_counter()
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"scope": report["scope"], "configuration": report["configuration"],
                       "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
        state = load_native_state_dict(args.artifacts)
        model = OLMoTiledRTForCausalLM(config, attention_backend="sdpa",
            attention_precision=args.attention_precision, device="meta", dtype=torch.float32)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to("cuda").eval()
        del state

        def on_comparison(row):
            write_json(args.output_dir/"report.json", report)
            tracker.log({"localization/within_original_budget": row["passed"],
                         "localization/bitwise_equal": row["all_bitwise_equal"],
                         "localization/max_gradient_relative_l2": row["max_gradient_relative_l2"]},
                        step=len(report["comparisons"]))
            print({"comparison": row["name"], "passed": row["passed"],
                   "max_gradient_relative_l2": row["max_gradient_relative_l2"]}, flush=True)

        run_controls(model, args, report, on_comparison)
        if report["source_hashes"] != {name: sha256_file(ROOT/name) for name in SOURCE_FILES}:
            raise AssertionError("Runtime source changed during localization")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="capture_blocked" if report["stage"] == "capture" else "failed",
                      error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-began
        try:
            tracker.summary({"localization/status": report["status"],
                "localization/all_within_original_budget": report.get("all_comparisons_within_original_budget", False)})
            tracker.finish(succeeded=report["status"] == "completed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)
    print({"status": report["status"], "report": str(args.output_dir/"report.json"),
           "all_within_original_budget": report["all_comparisons_within_original_budget"]}, flush=True)


if __name__ == "__main__":
    main()
