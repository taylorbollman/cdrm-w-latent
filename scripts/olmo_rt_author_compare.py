#!/usr/bin/env python3
"""Matched native and author-derived RT block-stack correctness and throughput."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_recurrent import recurrent_layer_reference
from cdrm.pretrained.olmo_rope import build_rope_tables
from cdrm.pretrained.olmo_tiled import tiled_recurrent_layer
from cdrm.pretrained.olmo_author import AUTHOR_SOURCE, COMPILED_HELPER_BOUNDARIES
from cdrm.pretrained.rt_block_resources import estimate_rt_block_resources
from scripts.olmo_f1_common import build_optimizer, boundary_digests, state_health
from scripts.olmo_f3_graph_training import (timed, configure_determinism, require_container_gpu,
    validate_prepared_manifest, load_native_state_dict, OnlineTracker)
from scripts.olmo_f3d_validate import global_gradient_l2
from scripts.olmo_rt_efficiency import (SOURCES as STAGE_A_SOURCES, ARMS, memory_snapshot,
    track_optimizer_steps, operator_profile)

PROTOCOL = ROOT / "docs/reports/olmo-rt-author-comparison/protocol.md"
AUDIT = ROOT / "docs/reports/olmo-rt-author-comparison/author-port-audit.md"
SOURCES = tuple(sorted(set(STAGE_A_SOURCES) | {
    "scripts/olmo_rt_author_compare.py", "scripts/olmo_rt_author_event_probe.py",
    "cdrm/pretrained/olmo_author.py", "cdrm/pretrained/rt_block_resources.py"}))
SEED = 20260924
BF16_BUDGETS = {"global_parameter_relative_l2": 1 / 64, "gradient_relative_l2": 1 / 32,
    "gradient_max_relative": 1 / 16, "output_relative_l2": 1 / 64,
    "output_max_relative": 1 / 16, "mse_relative_l2": 1e-5}
FP32_BUDGETS = {"relative_l2": 1e-4, "absolute_l2": 2e-6,
                "max_absolute_floor": 2e-6, "max_relative": 1e-4}


@dataclass(frozen=True)
class Execution:
    backend: str = "native"
    precision: str = "bf16_mixed"
    native_arm: str = "both"
    native_backward: str = "recompute"
    author_precision: str = "author_legacy"
    bwd_mlp_chunks: int = 4
    compiled_helpers: bool = True
    native_tiles: str = "triton"

    def __post_init__(self):
        if self.backend not in ("native", "author", "native_scan", "author_scan"):
            raise ValueError("Unknown block execution backend")
        if self.precision not in ("fp32", "bf16_mixed") or self.native_arm not in ARMS:
            raise ValueError("Unknown precision or native arm")
        if self.native_backward not in ("recompute", "materialized"):
            raise ValueError("Unknown native backward policy")
        if self.author_precision not in ("author_legacy", "fp32_state"):
            raise ValueError("Unknown author precision policy")
        if type(self.bwd_mlp_chunks) is not int or self.bwd_mlp_chunks < 1:
            raise ValueError("bwd_mlp_chunks must be a positive integer")
        if type(self.compiled_helpers) is not bool or self.native_tiles not in ("eager", "triton"):
            raise ValueError("Invalid helper compilation or native tile choice")


@dataclass(frozen=True)
class BlockFixture:
    inputs: torch.Tensor
    targets: torch.Tensor


def make_fixture(batch, length, width, *, seed=SEED):
    if any(type(value) is not int or value < 1 for value in (batch, length, width)):
        raise ValueError("Fixture dimensions must be positive integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (batch, length, width)
    return BlockFixture(torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))


def make_cotangent(shape, *, seed=SEED + 100):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator) / math.sqrt(math.prod(shape))


def fixture_record(fixture, *, seed):
    return {"seed": seed, "shape": list(fixture.inputs.shape), "dtype": str(fixture.inputs.dtype),
        "input_l2": float(fixture.inputs.double().norm()), "target_l2": float(fixture.targets.double().norm()),
        "input_rms": float(fixture.inputs.double().square().mean().sqrt()),
        "target_rms": float(fixture.targets.double().square().mean().sqrt()),
        "distribution": "independent standard-normal CPU inputs and targets; separate fixed generator"}


class BlockStack(nn.Module):
    """Native packed parameters only; neither backend owns a copied parameter."""
    def __init__(self, config, *, device=None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(OLMoBlock(config, device=device, dtype=torch.float32)
                                   for _ in range(config.num_layers))

    def forward(self, x, positions, valid, tables, execution):
        for layer in self.layers:
            if execution.backend == "native_scan":
                x, _ = recurrent_layer_reference(layer, x, alpha=1.0, past=None,
                    query_positions=positions, key_positions=positions, key_valid=valid, attention_backend="math")
            elif execution.backend == "native":
                reuse_rope, kv_only = ARMS[execution.native_arm]
                x, _ = tiled_recurrent_layer(layer, x, alpha=1.0, past=None,
                    query_positions=positions, key_positions=positions, key_valid=valid,
                    attention_precision="mixed", cast_weights_once=True,
                    tile_backend=execution.native_tiles, backward_tile_backend=execution.native_tiles,
                    backward_memory=execution.native_backward, reuse_rope=reuse_rope, kv_only_writes=kv_only,
                    query_rope=tables if reuse_rope else None, key_rope=tables if reuse_rope else None)
            else:
                from cdrm.pretrained.olmo_author import author_recurrent_reference, author_tiled_recurrent_layer
                if execution.backend == "author_scan":
                    x = author_recurrent_reference(layer, x, tables,
                        precision_policy=execution.author_precision, autocast_cache=True)
                else:
                    x = author_tiled_recurrent_layer(layer, x, tables,
                        compiled_helpers=execution.compiled_helpers, bwd_mlp_chunks=execution.bwd_mlp_chunks,
                        precision_policy=execution.author_precision, autocast_cache=True)
        return x


def build_stack(state, config, *, device):
    model = BlockStack(config, device="meta")
    expected = set(model.state_dict())
    subset = {name: state[name] for name in expected}
    # CPU fixtures must not mutate a caller-owned checkpoint through assign=True.
    if torch.device(device).type == "cpu":
        subset = {name: value.clone() for name, value in subset.items()}
    model.load_state_dict(subset, strict=True, assign=True)
    return model.to(device).train()


def mse_objective(output, targets):
    return (output.float() - targets.float()).square().mean()


def prepare_positions(fixture, config, device):
    batch, length, _ = fixture.inputs.shape
    positions = torch.arange(length, device=device).expand(batch, -1).clone()
    valid = torch.ones((batch, length), dtype=torch.bool, device=device)
    tables = build_rope_tables(positions, config.head_dim, config.rope_freq_constant)
    return positions, valid, tables


def autocast(execution, device):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
        enabled=execution.precision == "bf16_mixed", cache_enabled=False)


def raw_snapshot(model, fixture, probe, execution):
    device = next(model.parameters()).device
    if execution.precision == "bf16_mixed" and device.type != "cuda":
        raise ValueError("Mixed-precision validation requires CUDA, without CPU fallback")
    model.zero_grad(set_to_none=True)
    x = fixture.inputs.to(device, copy=True).requires_grad_(True)
    targets, cotangent = fixture.targets.to(device), probe.to(device)
    positions, valid, tables = prepare_positions(fixture, model.config, device)
    with autocast(execution, device):
        output = model(x, positions, valid, tables, execution)
    output.backward(cotangent)
    snapshot = {"output": output.detach().cpu().clone(), "input_gradient": x.grad.detach().cpu().clone(),
        "parameters": {name: parameter.grad.detach().cpu().clone()
                       for name, parameter in model.named_parameters() if parameter.grad is not None},
        "mse": mse_objective(output.detach(), targets).cpu(),
        "expected_parameters": sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad),
        "execution": asdict(execution)}
    model.zero_grad(set_to_none=True)
    return snapshot


def tensor_metrics(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError("Comparison tensor shape differs")
    a, b = candidate.double(), reference.double()
    delta = a - b
    error, norm = float(delta.square().sum()), float(b.square().sum())
    peak, difference = float(b.abs().max()), float(delta.abs().max())
    return {"delta_sq": error, "reference_sq": norm,
        "relative_l2": global_gradient_l2([{"delta_sq": error, "reference_sq": norm}]),
        "difference_l2": math.sqrt(error), "reference_max_abs": peak, "max_abs": difference,
        "max_relative": difference / peak if peak else (0.0 if difference == 0 else float("inf")),
        "bitwise_equal": torch.equal(candidate, reference),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def fp32_tensor_passes(row):
    return (row["finite"] and (row["relative_l2"] <= FP32_BUDGETS["relative_l2"]
        or row["difference_l2"] <= FP32_BUDGETS["absolute_l2"])
        and row["max_abs"] <= FP32_BUDGETS["max_absolute_floor"]
        + FP32_BUDGETS["max_relative"] * row["reference_max_abs"])


def compare_snapshots(candidate, reference, *, name, policy, gate=True):
    if policy not in ("exact", "fp32", "bf16", "diagnostic"):
        raise ValueError("Unknown comparison screen")
    names = set(reference["parameters"])
    ownership = names == set(candidate["parameters"]) == set(candidate["expected_parameters"])
    ownership = ownership and names == set(reference["expected_parameters"])
    gradients = {key: tensor_metrics(candidate["parameters"][key], reference["parameters"][key])
                 for key in sorted(names & set(candidate["parameters"]))}
    output = tensor_metrics(candidate["output"], reference["output"])
    input_gradient = tensor_metrics(candidate["input_gradient"], reference["input_gradient"])
    mse = tensor_metrics(candidate["mse"], reference["mse"])
    rows = [output, input_gradient, mse, *gradients.values()]
    finite = all(row["finite"] for row in rows)
    exact = all(row["bitwise_equal"] for row in rows)
    global_l2 = global_gradient_l2(gradients.values())
    global_error = math.sqrt(sum(row["delta_sq"] for row in gradients.values()))
    passed = ownership and bool(gradients) and finite
    if policy == "exact":
        passed = passed and exact
    elif policy == "fp32":
        passed = (passed and all(fp32_tensor_passes(row) for row in rows)
            and (global_l2 <= FP32_BUDGETS["relative_l2"] or global_error <= FP32_BUDGETS["absolute_l2"]))
    elif policy == "bf16":
        passed = (passed and global_l2 <= BF16_BUDGETS["global_parameter_relative_l2"]
            and all(row["relative_l2"] <= BF16_BUDGETS["gradient_relative_l2"]
                    and row["max_relative"] <= BF16_BUDGETS["gradient_max_relative"]
                    for row in [input_gradient, *gradients.values()])
            and output["relative_l2"] <= BF16_BUDGETS["output_relative_l2"]
            and output["max_relative"] <= BF16_BUDGETS["output_max_relative"]
            and mse["relative_l2"] <= BF16_BUDGETS["mse_relative_l2"])
    return {"name": name, "passed": passed, "gate": gate, "policy": policy,
        "ownership_matches": ownership, "finite": finite, "all_bitwise_equal": exact,
        "global_parameter_relative_l2": global_l2, "global_parameter_absolute_l2": global_error,
        "output": output, "input_gradient": input_gradient, "mse": mse, "gradients": gradients,
        "budgets": FP32_BUDGETS if policy == "fp32" else BF16_BUDGETS if policy == "bf16" else None,
        "reference_execution": reference["execution"], "candidate_execution": candidate["execution"]}


def configure_compiler():
    torch._dynamo.config.recompile_limit = 64
    if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        raise RuntimeError("Compiler runtime lacks fail-on-recompile-limit enforcement")
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.utils.counters.clear()
    return compiler_audit(required=False)


def compiler_audit(*, required, require_graphs=True):
    counters = {str(group): {str(key): int(value) for key, value in entries.items()}
                for group, entries in torch._dynamo.utils.counters.items()}
    failures = {group: {key: value for key, value in counters.get(group, {}).items() if value}
                for group in ("unimplemented", "graph_break")}
    failures = {key: value for key, value in failures.items() if value}
    if required and (failures or not torch._dynamo.config.fail_on_recompile_limit_hit):
        raise RuntimeError(f"Author helper compilation broke/fell back: {failures}")
    if required and require_graphs and counters.get("stats", {}).get("unique_graphs", 0) == 0:
        raise RuntimeError("Author helpers requested compilation but no graph was observed")
    return {"required": required, "counters": counters, "recompile_limit": torch._dynamo.config.recompile_limit,
        "accumulated_recompile_limit": torch._dynamo.config.accumulated_recompile_limit,
        "fail_on_recompile_limit_hit": torch._dynamo.config.fail_on_recompile_limit_hit}


class PreparedBlockTraining:
    """One shared capture boundary for block forward, scalar MSE and backward."""
    def __init__(self, model, fixture, execution, *, region_events=False):
        self.model, self.execution = model, execution
        device = next(model.parameters()).device
        if execution.precision == "bf16_mixed" and device.type != "cuda":
            raise ValueError("Mixed-precision graph training requires CUDA")
        if not model.training or not torch.is_grad_enabled():
            raise ValueError("Prepare in grad-enabled training mode")
        self.inputs = fixture.inputs.to(device, copy=True).requires_grad_(True)
        self.targets = fixture.targets.to(device, copy=True)
        self.positions, self.valid, self.tables = prepare_positions(fixture, model.config, device)
        self._shape = fixture.inputs.shape
        self._static = (self.positions, self.valid, self.tables.cos, self.tables.sin)
        self._static_signature = tuple((id(value), value.data_ptr(), value._version) for value in self._static)
        self._parameter_signature = self.parameter_signature()
        self._mutable_signature = (id(self.inputs), self.inputs.data_ptr(), id(self.targets), self.targets.data_ptr())
        self._execution = execution
        self._configuration = model.config
        self._module_signature = tuple((name, id(module), module.training) for name, module in model.named_modules())
        self._math_signature = self.math_signature()
        self.graph = self.graph_result = None
        self.gradient_addresses = None
        self.warmup_backward_calls = self.capture_backward_calls = self.replay_calls = 0
        self.input_tokens = fixture.inputs.shape[0] * fixture.inputs.shape[1]
        self.events = None
        if region_events:
            if device.type != "cuda":
                raise ValueError("Region events require CUDA")
            self.events = tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(3))
            for event in self.events:
                event.record()
            torch.cuda.synchronize()

    def parameter_signature(self):
        return tuple((name, id(parameter), parameter.data_ptr(), parameter.shape, parameter.dtype,
                      parameter.device, parameter.requires_grad) for name, parameter in self.model.named_parameters())

    @staticmethod
    def math_signature():
        return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled())

    def validate(self):
        if not self.model.training or self._execution != self.execution or self._configuration != self.model.config:
            raise ValueError("Prepared block configuration changed")
        if (not torch.is_grad_enabled() or self.math_signature() != self._math_signature
                or tuple((name, id(module), module.training) for name, module in self.model.named_modules()) != self._module_signature):
            raise ValueError("Prepared block runtime/module settings changed")
        if self.parameter_signature() != self._parameter_signature:
            raise ValueError("Prepared block parameter storage/ownership changed")
        current_static = (self.positions, self.valid, self.tables.cos, self.tables.sin)
        if tuple((id(value), value.data_ptr(), value._version) for value in current_static) != self._static_signature:
            raise ValueError("Prepared positions/RoPE tables changed")
        if (id(self.inputs), self.inputs.data_ptr(), id(self.targets), self.targets.data_ptr()) != self._mutable_signature:
            raise ValueError("Prepared input/target storage changed")
        if self.gradient_addresses is not None and self.grad_addresses() != self.gradient_addresses:
            raise ValueError("Persistent gradients changed")

    def grad_addresses(self):
        return {"input": None if self.inputs.grad is None else self.inputs.grad.data_ptr(),
            **{name: None if parameter.grad is None else parameter.grad.data_ptr()
               for name, parameter in self.model.named_parameters()}}

    def load_fixture(self, fixture):
        self.validate()
        if (fixture.inputs.shape != self._shape or fixture.targets.shape != self._shape
                or fixture.inputs.dtype != torch.float32 or fixture.targets.dtype != torch.float32):
            raise ValueError("Fixture layout/dtype changed")
        with torch.no_grad():
            self.inputs.copy_(fixture.inputs)
            self.targets.copy_(fixture.targets)

    def _tensor_backward(self):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        if self.inputs.grad is not None:
            self.inputs.grad.zero_()
        if self.events:
            self.events[0].record()
        with autocast(self.execution, self.inputs.device):
            output = self.model(self.inputs, self.positions, self.valid, self.tables, self.execution)
            loss = mse_objective(output, self.targets)
        if self.events:
            self.events[1].record()
        loss.backward()
        if self.events:
            self.events[2].record()
        return output, loss

    def initialize(self):
        self.validate()
        if self.gradient_addresses is not None:
            return
        if self.inputs.grad is not None or any(parameter.grad is not None for parameter in self.model.parameters()):
            raise ValueError("Prepare at a cleared gradient boundary")
        self._tensor_backward()
        self.warmup_backward_calls += 1
        self.gradient_addresses = self.grad_addresses()
        if any(address is None for address in self.gradient_addresses.values()):
            raise AssertionError("Missing block/input gradient")

    def backward(self, *, replay=False):
        self.initialize()
        self.validate()
        if replay:
            if self.graph is None:
                raise ValueError("Capture before graph replay")
            self.graph.replay()
            self.replay_calls += 1
            return self.graph_result
        return self._tensor_backward()

    def capture(self, *, warmup=10):
        if self.inputs.device.type != "cuda":
            raise ValueError("Capture requires CUDA, without CPU fallback")
        if self.graph is not None or type(warmup) is not int or warmup < 10:
            raise ValueError("Fresh capture requires at least ten backward warmups")
        self.initialize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._tensor_backward()
        self.warmup_backward_calls += warmup
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.validate()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = self._tensor_backward()
        torch.cuda.synchronize()
        self.graph, self.graph_result = graph, result
        self.capture_backward_calls += 1
        self.validate()

    def optimizer_step(self, optimizer, fixture, *, replay, scheduler, counters):
        self.load_fixture(fixture)
        learning_rate = [group["lr"] for group in optimizer.param_groups]
        _, loss = self.backward(replay=replay)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError("Nonfinite block MSE")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,
                                             foreach=False, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        counters.optimizer_updates += 1
        counters.microbatches += 1
        counters.documents += self.inputs.shape[0]
        counters.input_tokens += self.input_tokens
        return {"mse": loss_value, "gradient_norm_before_clip": float(norm), "max_grad_norm": 1.0,
            "lr_used": learning_rate, "lr_next": [group["lr"] for group in optimizer.param_groups],
            "counters": asdict(counters)}

    def region_seconds(self):
        if self.events is None:
            return None
        return {"forward_objective_seconds": self.events[0].elapsed_time(self.events[1]) / 1000,
                "backward_seconds": self.events[1].elapsed_time(self.events[2]) / 1000}


def plan_snapshot(plan, result):
    output, loss = result
    return {"output": output.detach().cpu().clone(), "input_gradient": plan.inputs.grad.detach().cpu().clone(),
        "mse": loss.detach().cpu().clone(), "execution": asdict(plan.execution),
        "parameters": {name: parameter.grad.detach().cpu().clone() for name, parameter in plan.model.named_parameters()
                       if parameter.grad is not None},
        "expected_parameters": sorted(name for name, parameter in plan.model.named_parameters() if parameter.requires_grad)}


def graph_check(plan, name, *, replays=1):
    reference = plan_snapshot(plan, plan.backward(replay=False))
    for _ in range(replays):
        result = plan.backward(replay=True)
    return compare_snapshots(plan_snapshot(plan, result), reference, name=name, policy="exact")


def update_parity(plan, fixtures, report, persist):
    model = plan.model
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer, scheduler = build_optimizer(model)
    initial_optimizer, initial_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    hook = track_optimizer_steps(optimizer, report)
    outcomes = []
    report["update_parity_progress"] = []
    try:
        for replay in (False, True):
            if replay:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(initial[name])
                optimizer.load_state_dict(initial_optimizer)
                scheduler.load_state_dict(initial_scheduler)
            counters, records = TrainingCounters(), []
            for index in range(3):
                record = plan.optimizer_step(optimizer, fixtures[index % len(fixtures)], replay=replay,
                                             scheduler=scheduler, counters=counters)
                records.append(record)
                report["update_parity_progress"].append({"replay": replay, "record": record})
                persist()
            outcomes.append({"replay": replay, "metrics": records,
                "boundary": boundary_digests(model, optimizer, scheduler, counters),
                "health": state_health(model, optimizer)})
    finally:
        hook.remove()
    changed = any(not torch.equal(parameter.detach().cpu(), initial[name])
                  for name, parameter in model.named_parameters())
    exact_metrics, exact_boundary = outcomes[0]["metrics"] == outcomes[1]["metrics"], outcomes[0]["boundary"] == outcomes[1]["boundary"]
    return {"name": "complete_adamw_update_parity", "gate": True, "updates_per_arm": 3,
        "physical_optimizer_updates": 6, "metrics_exact": exact_metrics,
        "model_optimizer_scheduler_counters_exact": exact_boundary, "weights_changed": changed,
        "arms": outcomes, "passed": exact_metrics and exact_boundary and changed
        and all(outcome["health"]["passed"] for outcome in outcomes)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "capacity"), required=True)
    parser.add_argument("--backend", choices=("native", "author"), required=True)
    parser.add_argument("--layers", type=int, choices=(1, 2, 6), default=1)
    parser.add_argument("--batch-size", type=int, choices=(1, 8, 32, 128, 256, 512), required=True)
    parser.add_argument("--length", type=int, choices=(32, 512), default=512)
    parser.add_argument("--precision", choices=("fp32", "bf16_mixed"), default="bf16_mixed")
    parser.add_argument("--native-arm", choices=tuple(ARMS), default="both")
    parser.add_argument("--native-backward", choices=("recompute", "materialized"), default="recompute")
    parser.add_argument("--author-precision", choices=("author_legacy", "fp32_state"), default="author_legacy")
    parser.add_argument("--bwd-mlp-chunks", type=int, default=4)
    parser.add_argument("--region-events", action=argparse.BooleanOptionalAction, default=True,
                        help="Validated external event nodes inside the common training graph; --no-region-events is a labeled diagnostic")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.bwd_mlp_chunks not in (1, 2, 4, 8, 16):
        parser.error("Bound bwd_mlp_chunks to 1/2/4/8/16 and record every change")
    if args.stage == "verify":
        if (args.batch_size, args.length) not in ((1, 32), (8, 512)):
            parser.error("Verification is bounded to B1/T32 or B8/T512")
        if args.precision == "fp32" and (args.batch_size, args.length) != (1, 32):
            parser.error("Strict FP32 sequential reference is bounded to B1/T32")
        if args.profile:
            parser.error("Profiling is separate from verification")
    else:
        if args.length != 512 or args.batch_size == 8:
            parser.error("Capacity uses T512 and physical batches1/32/128/256/512")
        if args.precision != "bf16_mixed":
            parser.error("Primary capacity is BF16 mixed; FP32 is a bounded verification diagnostic")
        if args.profile and (args.batch_size != 32 or args.layers != 1):
            parser.error("Separate profile is bounded to one block at B32/T512")
    return args


def main(argv=None):
    args = parse_args(argv)
    overall_began = time.perf_counter()
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compilation = configure_compiler()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        destination = args.output_dir / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    shutil.copyfile(AUDIT, args.output_dir / "author-port-audit.md")
    config = replace(OLMoConfig.native_1b(), num_layers=args.layers)
    execution = Execution(backend=args.backend, precision=args.precision, native_arm=args.native_arm,
        native_backward=args.native_backward, author_precision=args.author_precision, bwd_mlp_chunks=args.bwd_mlp_chunks)
    configuration = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": config.to_dict(), "execution": asdict(execution), "seed": SEED,
        "checkpoint_blocks": list(range(args.layers)), "input_distribution": "fixed standard-normal FP32",
        "training_objective": "FP32 mean squared error against fixed Gaussian targets",
        "raw_gradient_cotangent": "Gaussian seed20261024 divided by sqrt(B*T*D); expected norm1; no clipping",
        "capture_boundary": "gradient reset + block forward + FP32 scalar MSE + backward",
        "optimizer_boundary": "input/target copy, graph replay, clipping1.0, AdamW, scheduler",
        "ordinary_layers": 0, "fbt": False, "nextlat": False, "embedding_readout": False,
        "physical_batch": args.batch_size, "accumulation": 1, "world_size": 1,
        "native_autocast_cache": False, "native_explicit_cast_reuse": True,
        "author_autocast_cache": "fresh internal scopes, enabled", "author_compiled_helpers": True,
        "author_backward_memory": "materialized", "tf32": False,
        "preparation_updates": 3 if args.stage == "capacity" else 0,
        "capture_warmup_backwards": 10, "timed_updates": 5 if args.stage == "capacity" else 0,
        "graph_timing_samples": 3 if args.stage == "capacity" else 0,
        "scope": "Block-stack operational fixture, not LM training throughput"}
    report = {"schema": "olmo-rt-author-comparison-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "author_source": dict(AUTHOR_SOURCE), "compiled_helper_boundaries": copy.deepcopy(COMPILED_HELPER_BOUNDARIES),
        "compiler_configuration": compilation, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [], "physical_optimizer_updates": 0,
        "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "audit_sha256": sha256_file(AUDIT)}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-rt-author-comparison",
        name=args.output_dir.name, output_dir=args.output_dir)

    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    def publish(check, *, blocking=True):
        report["checks"].append(check)
        save()
        tracker.log({"correctness/passed": int(check["passed"])}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"], "gate": check.get("gate", True),
               "global_parameter_relative_l2": check.get("global_parameter_relative_l2")}, flush=True)
        if blocking and not check["passed"]:
            raise AssertionError(check["name"])

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        load_began = time.perf_counter()
        state = load_native_state_dict(args.artifacts)
        model = build_stack(state, config, device="cuda")
        del state
        report["checkpoint_load_and_stack_setup_seconds"] = time.perf_counter() - load_began
        report["parameters"] = {"active": sum(parameter.numel() for parameter in model.parameters()),
            "resident": sum(parameter.numel() for parameter in model.parameters()),
            "owned_parameter_tensors": len(list(model.parameters())),
            "named_shapes": {name: list(parameter.shape) for name, parameter in model.named_parameters()}}
        report["resources"] = estimate_rt_block_resources(config, batch_size=args.batch_size,
            sequence_length=args.length, backend=args.backend, native_arm=args.native_arm,
            native_backward=args.native_backward)
        if report["resources"]["unique_parameters"] != report["parameters"]["active"]:
            raise AssertionError("Analytic and observed block parameter counts differ")
        fixtures = [make_fixture(args.batch_size, args.length, config.model_dim, seed=SEED + index)
                    for index in range(2)]
        report["fixtures"] = [fixture_record(fixture, seed=SEED + index) for index, fixture in enumerate(fixtures)]
        report["fixture_order"] = "Two fixed changed input/target batches, alternating by optimizer update"
        if args.stage == "verify":
            save("raw_gradient_comparisons")
            comparison_began = time.perf_counter()
            probe = make_cotangent(fixtures[0].inputs.shape)
            report["cotangent"] = {"seed": SEED + 100, "normalization": "divide by sqrt(numel)",
                "l2": float(probe.double().norm()), "max_abs": float(probe.abs().max())}
            snapshots = {}
            if args.length == 32:
                fp32_reference = raw_snapshot(model, fixtures[0], probe,
                    replace(execution, backend="native_scan", precision="fp32", compiled_helpers=False))
                for backend in ("author_scan", "author", "native"):
                    snapshot = raw_snapshot(model, fixtures[0], probe,
                        replace(execution, backend=backend, precision="fp32"))
                    publish(compare_snapshots(snapshot, fp32_reference,
                        name=backend + "_fp32_vs_native_scan", policy="fp32"))
                    if backend in ("author", "native"):
                        snapshots[(backend, "fp32")] = snapshot
                if args.precision == "bf16_mixed":
                    for backend in ("native", "author"):
                        snapshot = raw_snapshot(model, fixtures[0], probe, replace(execution, backend=backend))
                        snapshots[(backend, args.precision)] = snapshot
                        publish(compare_snapshots(snapshot, fp32_reference,
                            name=backend + "_bf16_vs_fp32_diagnostic", policy="diagnostic", gate=False), blocking=False)
            else:
                for backend in ("native", "author"):
                    snapshots[(backend, args.precision)] = raw_snapshot(model, fixtures[0], probe,
                        replace(execution, backend=backend))
            publish(compare_snapshots(snapshots[("author", args.precision)], snapshots[("native", args.precision)],
                name="author_vs_native_" + args.precision,
                policy="bf16" if args.precision == "bf16_mixed" else "fp32"), blocking=args.precision == "fp32")
            report["raw_comparison_seconds"] = time.perf_counter() - comparison_began
            if any(not check.get("finite", True) for check in report["checks"]):
                raise FloatingPointError("Nonfinite raw comparison prevents graph/update continuation")
            del snapshots, probe
            if args.length == 32:
                del fp32_reference
            model.zero_grad(set_to_none=True)
            plan = PreparedBlockTraining(model, fixtures[0], execution, region_events=args.region_events)
            save("capture")
            began = time.perf_counter()
            plan.capture(warmup=10)
            report["capture_seconds"] = time.perf_counter() - began
            publish(graph_check(plan, "candidate_initial_graph"))
            plan.load_fixture(fixtures[1])
            publish(graph_check(plan, "candidate_changed_inputs_overwrite", replays=2))
            save("complete_update_parity")
            publish(update_parity(plan, fixtures, report, save))
            plan.load_fixture(fixtures[1])
            publish(graph_check(plan, "candidate_changed_weights"))
            report["memory"] = memory_snapshot()
        else:
            plan = PreparedBlockTraining(model, fixtures[0], execution, region_events=args.region_events)
            optimizer, scheduler = build_optimizer(model)
            hook = track_optimizer_steps(optimizer, report)
            counters, records = TrainingCounters(), []
            initial_probe = model.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
            torch.cuda.reset_peak_memory_stats()
            save("preparation_updates")
            preparation_began = time.perf_counter()
            report["preparation_records"] = []
            for index in range(3):
                report["preparation_records"].append(plan.optimizer_step(optimizer, fixtures[index % 2],
                    replay=False, scheduler=scheduler, counters=counters))
                save()
            report["preparation_seconds"] = time.perf_counter() - preparation_began
            save("capture")
            began = time.perf_counter()
            plan.capture(warmup=10)
            report["capture_seconds"] = time.perf_counter() - began
            report["compiler_after_warmup"] = compiler_audit(required=args.backend == "author")
            save("capacity_graph_validation")
            publish(graph_check(plan, "capacity_initial_graph"))
            plan.load_fixture(fixtures[1])
            publish(graph_check(plan, "capacity_changed_inputs_overwrite", replays=2))
            report["setup_memory"] = memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
            regions = []
            report["timed_records"] = records

            def complete():
                records.append(plan.optimizer_step(optimizer, fixtures[(3 + len(records)) % 2],
                    replay=True, scheduler=scheduler, counters=counters))
                if args.region_events:
                    regions.append(plan.region_seconds())

            save("timed_complete_updates")
            report["full_update"] = timed(complete, 5)
            report["steady_memory"] = memory_snapshot()
            report["region_timings"] = regions
            report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
            report["layer_token_work_per_second"] = args.layers * report["input_tokens_per_second"]
            report["estimated_matrix_tflops_per_second"] = (report["resources"]["total_matrix_flops"]
                / report["full_update"]["median_wall_seconds"] / 1e12)
            save("timed_graph_replays")
            report["forward_loss_backward"] = timed(lambda: plan.backward(replay=True), 3)
            report["forward_loss_backward_tokens_per_second"] = plan.input_tokens / report["forward_loss_backward"]["median_wall_seconds"]
            save("capacity_changed_weights_validation")
            publish(graph_check(plan, "capacity_changed_weights"))
            report["health"] = state_health(model, optimizer)
            changed = not torch.equal(initial_probe, model.layers[0].ff_out.weight.detach().flatten()[:4096])
            finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                   for parameter in model.parameters()) and bool(torch.isfinite(plan.inputs.grad).all())
            publish({"name": "finite_complete_updates", "passed": report["health"]["passed"] and changed and finite_gradients,
                "trainable_weight_changed": changed, "finite_parameter_and_input_gradients": finite_gradients})
            hook.remove()
            tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                "benchmark/layer_token_work_per_second": report["layer_token_work_per_second"],
                "benchmark/forward_loss_backward_tokens_per_second": report["forward_loss_backward_tokens_per_second"],
                "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]}, step=len(report["checks"]) + 1)
            if args.profile:
                save("untimed_operator_profile")
                report["profile"] = operator_profile(plan, args.output_dir)
        report["compiler_final"] = compiler_audit(required=args.backend == "author" or args.stage == "verify")
        report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
            "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
        if report["source_hashes"] != {name: sha256_file(ROOT / name) for name in SOURCES}:
            raise AssertionError("Runtime source changed during execution")
        if report["protocol_sha256"] != sha256_file(PROTOCOL) or report["audit_sha256"] != sha256_file(AUDIT):
            raise AssertionError("Frozen protocol/audit changed during execution")
        failed = [check["name"] for check in report["checks"] if check.get("gate", True) and not check["passed"]]
        if failed:
            raise AssertionError("Retained numerical/operational gate failures: " + ", ".join(failed))
        report["status"] = "passed"
        tracker.summary({"result_status": "passed", "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        report["compiler_on_exit"] = compiler_audit(required=False)
        save()
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["elapsed_seconds"] = time.perf_counter() - overall_began
            save()


if __name__ == "__main__":
    main()
