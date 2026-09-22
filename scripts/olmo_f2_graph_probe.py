#!/usr/bin/env python3
"""CUDA-graph feasibility for native tiled RT stack and fixed hidden-state VJP.

This is a static-shape microbenchmark, not graphed language-model training.
The public stack forward is unchanged. Implicit positions and unpadded rows
avoid the host validation of explicit positions; no cache or loss wrapper is
used. Gradients have persistent buffers, zeroed inside the graph. A disposable
SGD update outside capture verifies that replay reads changed native weights.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu

SOURCE_FILES = (
    "scripts/olmo_f2_graph_probe.py", "scripts/experiment_tracking.py", "scripts/olmo_validation.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_recurrent.py", "cdrm/pretrained/olmo.py",
    "cdrm/pretrained/recurrent.py", "cdrm/pretrained/openelm.py", "cdrm/pretrained/artifacts.py",
    "cdrm/pretrained/olmo_artifacts.py",
)


def parse_layers(value):
    if value.lower() == "none":
        return ()
    try:
        indices = tuple(int(part) for part in value.split(","))
        return RTMode(indices, 1).selected_layers
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Use comma-separated distinct layer indices, or none") from error


def validate_options(args, config):
    for name in ("batch_size", "length", "warmup", "repeats"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.length < 2 or args.length > min(config.max_context_length, 512):
        raise ValueError("This bounded probe supports lengths 2 through 512 within native context")
    if args.precision not in ("fp32", "bf16_mixed"):
        raise ValueError("Unsupported precision")
    if any(index >= config.num_layers for index in args.rt_layers):
        raise ValueError("Selected RT layer is outside the model")
    RTMode(args.rt_layers, args.alpha)


def make_static_inputs(config, batch, length, *, device):
    """No RNG in captured work; changed tokens preserve shape and addresses."""
    ids = (torch.arange(batch*length, device=device).reshape(batch, length)*13+7) % config.tokenizer_vocab_size
    index = torch.arange(batch*length*config.model_dim, device=device, dtype=torch.float32)
    cotangent = index.add(1).sin().reshape(batch, length, config.model_dim)
    cotangent /= cotangent.numel()**.5
    return ids, cotangent


def allocate_gradients(model):
    parameters = tuple(model.parameters())
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("The native stack probe expects all native parameters trainable")
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    return {name: parameter.grad.data_ptr() for name, parameter in model.named_parameters()}


def gradient_addresses(model):
    return {name: None if parameter.grad is None else parameter.grad.data_ptr()
            for name, parameter in model.named_parameters()}


def stack_step(model, ids, cotangent, mode, precision):
    """Only tensor work after static validation; suitable for capture or eager."""
    for parameter in model.parameters():
        parameter.grad.zero_()
    context = (torch.autocast(ids.device.type, dtype=torch.bfloat16, cache_enabled=False)
               if precision == "bf16_mixed" else nullcontext())
    with context:
        hidden = model(ids, mode=mode, return_logits=False, use_cache=False).last_hidden_state
    hidden.backward(cotangent)
    return hidden.detach()


def tensor_comparison(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    error = torch.linalg.vector_norm(a-b).item()
    norm = torch.linalg.vector_norm(b).item()
    maximum = (a-b).abs().max().item()
    reference_max = b.abs().max().item()
    # Same arithmetic/precision/backend. Preserve exactness separately; a tiny
    # joint tensor budget admits reduction-order noise without BF16-vs-FP32 claims.
    relative = error/max(norm, 1e-30)
    limit = 1e-6 + 1e-5*reference_max
    return {"bitwise_equal": torch.equal(a, b), "finite": finite,
        "relative_l2": relative, "max_abs": maximum, "reference_max_abs": reference_max,
        "relative_l2_limit": 1e-5, "max_abs_limit": limit,
        "passed": finite and relative <= 1e-5 and maximum <= limit}


def compare_replay(model, step, replay, graph_hidden, *, expected_addresses, name):
    eager_hidden = step().clone()
    expected = {key: parameter.grad.detach().clone() for key, parameter in model.named_parameters()}
    replay()
    torch.cuda.synchronize()
    hidden = tensor_comparison(graph_hidden, eager_hidden)
    gradients = {key: tensor_comparison(parameter.grad, expected[key])
                 for key, parameter in model.named_parameters()}
    addresses = gradient_addresses(model) == expected_addresses
    row = {"name": name, "hidden": hidden, "gradients": gradients,
        "gradient_buffers_unchanged": addresses,
        "all_bitwise_equal": hidden["bitwise_equal"] and all(value["bitwise_equal"] for value in gradients.values()),
        "passed": addresses and hidden["passed"] and all(value["passed"] for value in gradients.values())}
    del expected, eager_hidden
    return row


def time_steps(step, *, repeats, tokens):
    wall, device = [], []
    for _ in range(repeats):
        start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record(); step(); finish.record(); finish.synchronize()
        wall.append(time.perf_counter()-began)
        device.append(start.elapsed_time(finish)/1000)
    return {"repeats": repeats, "wall_seconds": wall, "cuda_seconds": device,
        "median_wall_seconds": statistics.median(wall), "median_cuda_seconds": statistics.median(device),
        "input_tokens_per_second": tokens/statistics.median(wall)}


def run_probe(model, args, report, tracker):
    config = model.config
    mode = RTMode(args.rt_layers, args.alpha)
    ids, cotangent = make_static_inputs(config, args.batch_size, args.length, device="cuda")
    original_ids = ids.clone()
    addresses = allocate_gradients(model)
    parameter_addresses = {key: parameter.data_ptr() for key, parameter in model.named_parameters()}
    report["persistent_gradient_bytes"] = sum(parameter.grad.numel()*parameter.grad.element_size() for parameter in model.parameters())
    step = lambda: stack_step(model, ids, cotangent, mode, args.precision)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    report["stage"] = "warmup"
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if gradient_addresses(model) != addresses:
        raise AssertionError("Warmup replaced a persistent gradient buffer")
    graph = torch.cuda.CUDAGraph()
    report["stage"] = "capture"
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        graph_hidden = step()
    torch.cuda.synchronize()
    report["capture_seconds"] = time.perf_counter()-started
    report["capture_succeeded"] = True
    report["stage"] = "equivalence"
    for name in ("original_tokens_weights", "changed_tokens", "changed_tokens_and_weights"):
        if name == "changed_tokens":
            ids.copy_((original_ids+17) % config.tokenizer_vocab_size)
        elif name == "changed_tokens_and_weights":
            # A real optimizer update, deliberately outside the graph/timings.
            # The model is disposable; there is no checkpoint write or learning run.
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
            before = model.layers[args.rt_layers[0] if args.rt_layers else 0].ff_out.weight.detach().clone()
            optimizer.step()
            report["weight_update_changed_probe_tensor"] = not torch.equal(before,
                model.layers[args.rt_layers[0] if args.rt_layers else 0].ff_out.weight)
            del before, optimizer
            if not report["weight_update_changed_probe_tensor"]:
                raise AssertionError("External weight update did not change the probed tensor")
        row = compare_replay(model, step, graph.replay, graph_hidden, expected_addresses=addresses, name=name)
        row["parameter_addresses_unchanged"] = parameter_addresses == {
            key: parameter.data_ptr() for key, parameter in model.named_parameters()}
        row["passed"] &= row["parameter_addresses_unchanged"]
        report["comparisons"].append(row)
        tracker.log({"graph_probe/comparison": len(report["comparisons"]),
            "graph_probe/hidden_relative_l2": row["hidden"]["relative_l2"],
            "graph_probe/max_gradient_relative_l2": max(value["relative_l2"] for value in row["gradients"].values()),
            "graph_probe/equivalent": int(row["passed"])}, step=len(report["comparisons"]))
        if not row["passed"]:
            raise AssertionError(f"Graph replay differs from eager: {name}")
    # Repeated replay must overwrite, rather than accumulate, persistent grads.
    report["comparisons"].append(compare_replay(model, step, lambda: (graph.replay(), graph.replay()),
        graph_hidden, expected_addresses=addresses, name="two_replays_do_not_accumulate"))
    if not report["comparisons"][-1]["passed"]:
        raise AssertionError("Repeated replay changed the gradient overwrite contract")
    report["stage"] = "timing"
    torch.cuda.reset_peak_memory_stats()
    tokens = args.batch_size*args.length
    report["timings"] = {"eager": time_steps(step, repeats=args.repeats, tokens=tokens),
        "graph_replay": time_steps(graph.replay, repeats=args.repeats, tokens=tokens)}
    report["replay_speedup_wall"] = report["timings"]["eager"]["median_wall_seconds"]/report["timings"]["graph_replay"]["median_wall_seconds"]
    report["peak_allocated_gib_timing"] = torch.cuda.max_memory_allocated()/2**30
    report["reserved_gib_after_capture"] = torch.cuda.memory_reserved()/2**30
    tracker.log({"graph_probe/eager_tokens_per_second": report["timings"]["eager"]["input_tokens_per_second"],
        "graph_probe/replay_tokens_per_second": report["timings"]["graph_replay"]["input_tokens_per_second"],
        "graph_probe/replay_speedup_wall": report["replay_speedup_wall"]}, step=5)
    report["stage"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--rt-layers", type=parse_layers, default=(0,))
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--precision", choices=("fp32", "bf16_mixed"), default="bf16_mixed")
    parser.add_argument("--attention-precision", choices=("fp32", "mixed"), default="mixed")
    parser.add_argument("--attention-backend", choices=("math", "sdpa"), default="sdpa")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
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
    report = {"schema": "olmo-f2-graph-probe-v1", "status": "running", "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(), "model_config": config.to_dict(),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_hashes": {name: sha256_file(ROOT/name) for name in SOURCE_FILES}, "comparisons": [],
        "capture_succeeded": False, "stage": "load", "scope": "Static unpadded public native stack forward + fixed hidden cotangent backward",
        "limitations": ["No CE, readout logits, NextLat or FBT", "No optimizer or input copies in graph/timing",
            "One external disposable SGD update only; not a learning experiment", "No KV cache or changing shape/mask/mode",
            "No full training throughput, compiler or multi-GPU clearance"],
        "autocast_cache_enabled": False, "persistent_grads_zeroed_inside_graph": True,
        "parameters_gradients_fp32": True, "explicit_position_validation_bypassed": False}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f2-graph-probe", name="olmo-1b-f2-graph-"+args.output_dir.name)
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": report["configuration"], "scope": report["scope"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
        state = load_native_state_dict(args.artifacts)
        model = OLMoTiledRTForCausalLM(config, attention_backend=args.attention_backend,
            attention_precision=args.attention_precision, device="meta", dtype=torch.float32)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to("cuda").eval()
        del state
        run_probe(model, args, report, tracker)
        if report["source_hashes"] != {name: sha256_file(ROOT/name) for name in SOURCE_FILES}:
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
            write_json(args.output_dir/"report.json", report)
    print({"status": report["status"], "report": str(args.output_dir/"report.json"),
        "replay_speedup_wall": report.get("replay_speedup_wall")}, flush=True)


if __name__ == "__main__":
    main()
