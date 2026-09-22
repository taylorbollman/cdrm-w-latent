#!/usr/bin/env python3
"""Run the unchanged F2 graph checks under an explicit ordinary SDPA backend."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scripts import olmo_f2_graph_probe as base

SOURCE_FILES = (*base.SOURCE_FILES, "scripts/olmo_f2_graph_backend_probe.py")


def configure_determinism(enabled):
    """Configure cuBLAS before any CUDA initialization; never silently set it late."""
    if enabled and torch.cuda.is_initialized():
        raise RuntimeError("Deterministic backend probe must configure cuBLAS before CUDA initialization")
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def backend_context(backend):
    if backend == "auto": return nullcontext()
    mapping = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
               "math": SDPBackend.MATH}
    if backend not in mapping: raise ValueError("Unknown SDPA backend")
    return sdpa_kernel(mapping[backend])


def run_backend_probe(model, args, report, tracker):
    # The existing helper owns all original changed-token/weight, overwrite,
    # equivalence and timing checks; no tolerance or arithmetic is changed here.
    with backend_context(args.backend):
        base.run_probe(model, args, report, tracker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rt-layers", type=base.parse_layers, default=(0,))
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--precision", choices=("fp32", "bf16_mixed"), default="bf16_mixed")
    parser.add_argument("--attention-precision", choices=("fp32", "mixed"), default="mixed")
    parser.add_argument("--backend", choices=("flash", "cudnn", "math", "auto"), required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    config = base.OLMoConfig.native_1b()
    base.validate_options(args, config)
    deterministic = configure_determinism(args.deterministic)
    runtime = {key: str(value) for key, value in base.require_container_gpu().items()}
    runtime.update(requested_sdpa_backend=args.backend, **deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-f2-graph-probe-v1", "status": "running", "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(), "model_config": config.to_dict(),
        "configuration": {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            **deterministic, "attention_backend": "sdpa"},
        "source_hashes": {name: base.sha256_file(ROOT/name) for name in SOURCE_FILES},
        "comparisons": [], "capture_succeeded": False, "stage": "load",
        "scope": "Static unpadded public native stack forward + fixed hidden cotangent backward; explicit ordinary SDPA backend",
        "limitations": ["No CE, readout logits, NextLat or FBT", "No optimizer or input copies in graph/timing",
            "One external disposable SGD update only; not a learning experiment", "No KV cache or changing shape/mask/mode",
            "No full training throughput, compiler or multi-GPU clearance",
            "Forced ordinary SDPA does not replace native RT dyadic attention or its custom backward",
            "Passing here does not clear automatic dispatch or other backend/shape/precision combinations"],
        "autocast_cache_enabled": False, "persistent_grads_zeroed_inside_graph": True,
        "parameters_gradients_fp32": True, "explicit_position_validation_bypassed": False,
        "comparison_policy": "Unchanged original graph probe budgets and changed-state checks"}
    tracker = base.OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f2-graph-backend-probe", name="olmo-1b-f2-graph-"+args.output_dir.name)
    try:
        manifest = base.validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": report["configuration"], "scope": report["scope"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        base.write_json(args.output_dir/"report.json", report)
        state = base.load_native_state_dict(args.artifacts)
        model = base.OLMoTiledRTForCausalLM(config, attention_backend="sdpa",
            attention_precision=args.attention_precision, device="meta", dtype=torch.float32)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to("cuda").eval()
        del state
        run_backend_probe(model, args, report, tracker)
        if report["source_hashes"] != {name: base.sha256_file(ROOT/name) for name in SOURCE_FILES}:
            raise ValueError("Probe runtime source changed")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="capture_blocked" if report["stage"] == "capture" else "failed",
            error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        try:
            tracker.summary({"graph_probe/status": report["status"], "graph_probe/capture_succeeded": report["capture_succeeded"]})
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            base.write_json(args.output_dir/"report.json", report)
    print({"status": report["status"], "backend": args.backend, "deterministic": args.deterministic,
        "replay_speedup_wall": report.get("replay_speedup_wall"), "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
