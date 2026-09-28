#!/usr/bin/env python3
"""Bounded actual-checkpoint changing-layout CUDA-graph accumulation probe.

One graph, one model, two logical updates, three physical microbatches per
update. CPU snapshots support same-state gradient and two-step Adam comparisons;
no diagnostic billion-parameter checkpoint is written. This is not throughput,
training quality, independent FP32, distributed, or restart qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import (CampaignRecipe, CampaignTokenSchedule,
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer, validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_probe import gradient_record, memory
from scripts.olmo_validation import require_container_gpu


TERMS = ("ce", "latent", "kl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(16, 32), default=16)
    return parser.parse_args(argv)


def fixture_batches(token_ids, eos_id, *, length, update):
    """Right-padded B2 fixtures, including short and entirely empty slots."""
    if length not in (16, 32) or update not in (0, 1) or len(token_ids) < length:
        raise ValueError("Probe requires T16/T32, update 0/1 and enough source tokens")
    lengths = ((length, 7), (2, 0), (0, 0)) if update == 0 else ((length-3, 5), (6, 0), (0, 0))
    batches, keys = [], []
    for microbatch, sizes in enumerate(lengths):
        ids = torch.full((2, length), eos_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, -1)
        row_keys = []
        for row, size in enumerate(sizes):
            if not size:
                continue
            offset = (update*7+microbatch*3+row) % len(token_ids)
            content = [token_ids[(offset+i) % len(token_ids)] for i in range(size-1)] + [eos_id]
            ids[row, :size] = torch.tensor(content)
            valid[row, :size] = True
            docs[row, :size] = update*100+microbatch*2+row
            row_keys.append(f"probe-update-{update}-microbatch-{microbatch}-row-{row}")
        ce, latent, kl = (valid.clone() for _ in TERMS)
        if update == 1:
            ce[:, 4::3] = False
            latent[:, 2::3] = False
            kl[:, 3::4] = False
        batches.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
        keys.append(row_keys)
    return tuple(batches), tuple(keys)


def fixture_record(batches):
    return [{name: value.tolist() for name, value in vars(batch).items()} for batch in batches]


def source_hashes():
    files = set((ROOT / "cdrm/pretrained").rglob("*.py")) | {
        Path(__file__), ROOT / "scripts/olmo_campaign_probe.py",
        ROOT / "scripts/experiment_tracking.py", ROOT / "scripts/olmo_validation.py"}
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(files)}


def rng_snapshot():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def rng_unchanged(before):
    return (torch.equal(before[0], torch.get_rng_state())
            and torch.equal(before[1], torch.cuda.get_rng_state()))


def pointer_snapshot(runner):
    adapter = runner.adapter
    pointers = {f"parameter/{name}": parameter.data_ptr() for name, parameter in runner.model.named_parameters()}
    pointers.update({f"gradient/{name}": None if parameter.grad is None else parameter.grad.data_ptr()
                     for name, parameter in runner.model.named_parameters()})
    pointers.update({f"input/{index}": value.data_ptr() for index, value in enumerate(adapter.owned_inputs())})
    pointers.update({f"loss/{name}": value.data_ptr() for name, value in vars(adapter.loss_layout).items()
                     if isinstance(value, torch.Tensor)})
    # Host validation snapshots can be replaced. Only the tensor storage used
    # by the captured forward belongs to the graph pointer contract.
    pointers.update({f"forward/{index}": value.data_ptr()
                     for index, value in enumerate(adapter.forward_layout._owned_tensors())
                     if value is not None})
    return pointers


@torch.no_grad()
def gradients_are_zero(model):
    return all(parameter.grad is not None and bool((parameter.grad == 0).all())
               for parameter in model.parameters() if parameter.requires_grad)


def compare_metrics(actual, expected):
    count_keys = ("counts", "microbatches", "documents", "input_tokens")
    exact = all(actual[key] == expected[key] for key in count_keys)
    rows = {key: {"actual": actual["loss_sums"][key], "expected": expected["loss_sums"][key],
                  "close": math.isclose(actual["loss_sums"][key], expected["loss_sums"][key],
                                       abs_tol=1e-5, rel_tol=3e-6)} for key in TERMS}
    objective_close = math.isclose(actual["objective"], expected["objective"], abs_tol=1e-6, rel_tol=3e-6)
    return {"counts_match": exact, "terms": rows,
            "objective_abs_difference": abs(actual["objective"]-expected["objective"]),
            "objective_close": objective_close,
            "passed": exact and objective_close and all(row["close"] for row in rows.values())}


@torch.no_grad()
def parameter_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


@torch.no_grad()
def restore_parameters_in_place(model, values):
    """Preserve captured parameter storage and leave immutable buffers alone."""
    named = dict(model.named_parameters())
    if named.keys() != values.keys() or any(named[n].shape != values[n].shape
            or named[n].dtype != values[n].dtype for n in named):
        raise ValueError("Parameter snapshot differs from model ownership/shapes/dtypes")
    for name, parameter in named.items():
        parameter.copy_(values[name])


@torch.no_grad()
def optimizer_snapshot(model, optimizer):
    return {name: {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                   for key, value in optimizer.state.get(parameter, {}).items()}
            for name, parameter in model.named_parameters()}


@torch.no_grad()
def compare_tensor_mappings(actual, expected, *, atol, rtol, initial=None):
    if actual.keys() != expected.keys():
        return {"passed": False, "error": "Tensor names differ"}
    if initial is not None and initial.keys() != actual.keys():
        return {"passed": False, "error": "Initial tensor names differ"}
    rows = {}
    error_squared = reference_squared = update_squared = 0.0
    for name, value in actual.items():
        want = expected[name]
        if value.shape != want.shape or value.dtype != want.dtype:
            return {"passed": False, "error": f"Tensor shape/dtype differs: {name}"}
        want = want.to(value.device)
        difference = value-want
        err = float(difference.square().sum(dtype=torch.float64))
        ref = float(want.square().sum(dtype=torch.float64))
        if initial is not None:
            update = want-initial[name].to(value.device)
            update_squared += float(update.square().sum(dtype=torch.float64))
        rows[name] = {"max_abs": float(difference.abs().max()) if difference.numel() else 0.,
                      "relative_l2": math.sqrt(err/max(ref, 1e-60)),
                      "close": bool(torch.allclose(value, want, atol=atol, rtol=rtol))}
        error_squared += err
        reference_squared += ref
    record = {"passed": all(row["close"] for row in rows.values()), "parameter_rows": rows,
              "relative_l2": math.sqrt(error_squared/max(reference_squared, 1e-60)),
              "atol": atol, "rtol": rtol}
    if initial is not None:
        record.update(update_relative_l2=math.sqrt(error_squared/max(update_squared, 1e-60)),
                      reference_update_norm=math.sqrt(update_squared), update_relative_l2_budget=1e-3)
        record["passed"] &= record["update_relative_l2"] <= 1e-3
    return record


def compare_optimizer(model, optimizer, reference):
    actual, expected = {}, {}
    names = dict(model.named_parameters())
    if names.keys() != reference.keys():
        return {"passed": False, "error": "Optimizer parameter names differ"}
    for name, parameter in names.items():
        state = optimizer.state.get(parameter, {})
        if state.keys() != reference[name].keys():
            return {"passed": False, "error": f"Optimizer state keys differ: {name}"}
        for key, value in state.items():
            want = reference[name][key]
            if isinstance(value, torch.Tensor) and isinstance(want, torch.Tensor):
                if key == "step" and not torch.equal(value.cpu(), want.cpu()):
                    return {"passed": False, "error": f"Optimizer step differs: {name}"}
                if key != "step":
                    actual[name+"/"+key], expected[name+"/"+key] = value, want
            elif value != want:
                return {"passed": False, "error": f"Optimizer scalar differs: {name}/{key}"}
    record = compare_tensor_mappings(actual, expected, atol=3e-5, rtol=3e-4)
    record["relative_l2_budget"] = 1e-3
    record["passed"] &= record.get("relative_l2", float("inf")) <= 1e-3
    return record


def main(argv=None):
    args = parse_args(argv)
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    recipe = CampaignRecipe("NFR", sequence_length=1024)
    report = {"schema": "olmo-campaign-dynamic-graph-probe-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
        "fixture_length": args.length, "physical_batch": 2, "microbatches_per_update": 3,
        "sources": source_hashes(), "rows": [],
        "scope": "One-device dynamic-mask graph/accumulation and two-step same-BF16 Adam parity; no performance, independent FP32, distributed, restart or training-quality claim",
        "execution": {"ordinary_attention": "forced Flash SDPA", "native_rt_tiles": "Triton",
            "backward": "recompute", "ordinary_checkpointing": True, "cuda_graphs": True,
            "ordinary_pointwise": "eager", "rope": "native reused", "torch_compile": False,
            "precision": "BF16 mixed, FP32 master parameters/gradients/Adam, TF32 off, autocast cache off"},
        "checkpoints": "Disposable snapshots in CPU RAM only; each bounded probe restarts from retained pinned source"}
    for relative in report["sources"]:
        destination = args.output_dir / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-campaign-dynamic-graph-readiness", name="campaign-graph-" + args.output_dir.name)
    started = time.monotonic()

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)
        tracker.log(scalar_metrics(row, "diagnostic/"+stage), step=len(report["rows"]))
        print({"stage": stage, "passed": row.get("passed"), "elapsed_seconds": time.monotonic()-started}, flush=True)
        if row.get("passed") is False:
            raise AssertionError(f"Campaign graph probe stage failed: {stage}")

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"recipe": report["recipe"], "fixture_length": args.length,
                       "checkpoint": report["checkpoint"], "scope": report["scope"]})
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        base = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), device="meta", dtype=torch.float32,
            attention_backend="sdpa", attention_precision="mixed", ordinary_activation_checkpointing=True,
            cast_weights_once=True, tile_backend="triton", backward_tile_backend="triton",
            backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
        base.load_state_dict(state, strict=True, assign=True)
        model = build_campaign_model(base, recipe).to("cuda").train()
        del state, base
        gc.collect()
        tokenizer = load_native_tokenizer(args.artifacts)
        ids = tokenizer.encode("The model keeps a record of earlier tokens and predicts the next token. "*20)
        fixture = [fixture_batches(ids, tokenizer.eos_token_id, length=args.length, update=u) for u in range(2)]
        batches = [value[0] for value in fixture]
        noises = [[feedback_noise_for_rows(recipe, keys, logical_update=u,
                    sequence_length=args.length, width=model.config.model_dim, physical_batch_size=2)
                   for keys in fixture[u][1]] for u in range(2)]
        global_counts = [sum_objective_counts([model.counts(batch) for batch in update]) for update in batches]
        token_counts = [sum(int(batch.valid_mask.sum()) for batch in update) for update in batches]
        report["fixture"] = {"updates": [fixture_record(update) for update in batches],
            "global_counts": global_counts, "valid_tokens_per_update": token_counts,
            "source": "Fixed operational prose, no benchmark or corpus sample"}
        report["parameters"] = {"resident": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            adapter = CampaignObjective(model, batches[0][0], mode=recipe.mode(), global_counts=global_counts[0],
                feedback_noise=noises[0][0], config=LMTrainingConfig(precision="bf16_mixed", max_grad_norm=recipe.max_grad_norm))
            runner = CampaignGraphTraining(adapter)
            before = rng_snapshot()
            eager = runner.backward(batches[0], feedback_noises=noises[0], replay=False)
            first_gradients, first_reference = gradient_record(model, save_cpu=True)
            pointers = pointer_snapshot(runner)
            publish("eager-first-update", {"metrics": eager, "gradients": first_gradients,
                "rng_unchanged": rng_unchanged(before), "memory": memory(),
                "passed": first_gradients["finite"] and rng_unchanged(before)})
            runner.discard_backward()
            # Capture on a nonempty fixture; local dummy selection is exercised
            # later using the same graph with nonzero global denominators.
            adapter.load_batch(batches[0][0], feedback_noise=noises[0][0], global_counts=global_counts[0])
            before = rng_snapshot()
            runner.capture(warmup=10)
            zeros = gradients_are_zero(model)
            publish("capture", {"warmup_backward_calls": runner.warmup_backward_calls,
                "capture_backward_calls": runner.capture_backward_calls, "gradients_zero": zeros,
                "rng_unchanged": rng_unchanged(before), "pointers_stable": pointers == pointer_snapshot(runner),
                "memory": memory(), "passed": zeros and rng_unchanged(before) and pointers == pointer_snapshot(runner)})
            before = rng_snapshot()
            adapter.load_batch(batches[0][-1], feedback_noise=noises[0][-1], global_counts=global_counts[0])
            runner.zero_grad()
            runner.graph.replay()
            runner.replay_calls += 1
            torch.cuda.synchronize()
            dummy_sums = {t: float(runner.graph_result["loss_sums"][t]) for t in TERMS}
            dummy_objective = float(runner.graph_result["objective"])
            zeros = gradients_are_zero(model)
            publish("empty-local-slot", {"loss_sums": dummy_sums, "objective": dummy_objective,
                "gradients_zero": zeros, "rng_unchanged": rng_unchanged(before),
                "pointers_stable": pointers == pointer_snapshot(runner),
                "passed": zeros and all(value == 0 for value in dummy_sums.values()) and dummy_objective == 0
                    and rng_unchanged(before) and pointers == pointer_snapshot(runner)})
            before = rng_snapshot()
            replay = runner.backward(batches[0], feedback_noises=noises[0], replay=True)
            gradient_comparison, _ = gradient_record(model, first_reference)
            gradient_comparison["comparison"]["scope"] = "Same BF16 weights/inputs: eager accumulation versus CUDA graph replay"
            metric_comparison = compare_metrics(replay, eager)
            publish("replay-first-update", {"metrics": replay, "gradient_comparison": gradient_comparison,
                "metric_comparison": metric_comparison, "rng_unchanged": rng_unchanged(before),
                "pointers_stable": pointers == pointer_snapshot(runner),
                "passed": gradient_comparison["finite"] and gradient_comparison["comparison"]["all_parameters_close"]
                    and metric_comparison["passed"] and rng_unchanged(before) and pointers == pointer_snapshot(runner)})
            before = rng_snapshot()
            changed_eager = runner.backward(batches[1], feedback_noises=noises[1], replay=False)
            changed, second_reference = gradient_record(model, first_reference, save_cpu=True)
            changed["comparison"]["scope"] = "Different fixture versus first fixture; gradients are expected to change"
            different = changed["comparison"]["relative_l2"] > 1e-8
            publish("eager-changed-update", {"metrics": changed_eager, "gradients": changed,
                "inputs_changed_gradients": different, "global_counts_changed": global_counts[0] != global_counts[1],
                "rng_unchanged": rng_unchanged(before), "passed": changed["finite"] and different
                    and global_counts[0] != global_counts[1] and rng_unchanged(before)})
            del first_reference
            before = rng_snapshot()
            changed_replay = runner.backward(batches[1], feedback_noises=noises[1], replay=True)
            changed_comparison, _ = gradient_record(model, second_reference)
            changed_comparison["comparison"]["scope"] = "Changed inputs/noise/counts through the original captured graph versus eager"
            metric_comparison = compare_metrics(changed_replay, changed_eager)
            publish("replay-changed-update", {"metrics": changed_replay, "gradients": changed_comparison,
                "metric_comparison": metric_comparison, "rng_unchanged": rng_unchanged(before),
                "pointers_stable": pointers == pointer_snapshot(runner),
                "passed": changed_comparison["finite"] and changed_comparison["comparison"]["all_parameters_close"]
                    and metric_comparison["passed"] and rng_unchanged(before) and pointers == pointer_snapshot(runner)})
            del second_reference
            runner.discard_backward()
            gc.collect()
            initial_parameters = parameter_snapshot(model)

            def run_updates(replay):
                optimizer = build_campaign_adamw(model, recipe, fused=True)
                scheduler = CampaignTokenSchedule(optimizer, token_counts, warmup_tokens=recipe.warmup_tokens,
                                                  start_fraction=recipe.warmup_start_fraction)
                counters = TrainingCounters()
                rows = []
                for u in range(2):
                    before = rng_snapshot()
                    row = runner.optimizer_step(optimizer, batches[u], feedback_noises=noises[u], replay=replay,
                                                scheduler=scheduler, counters=counters)
                    torch.cuda.synchronize()
                    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
                    row.update(parameters_finite=finite, rng_unchanged=rng_unchanged(before),
                        gradients_zero=gradients_are_zero(model), pointers_stable=pointers == pointer_snapshot(runner),
                        memory=memory(), completed_valid_tokens=scheduler.completed_tokens)
                    row["passed"] = finite and row["rng_unchanged"] and row["gradients_zero"] and row["pointers_stable"]
                    rows.append(row)
                    publish(("graph" if replay else "eager") + f"-adam-update-{u+1}", row)
                return optimizer, scheduler, counters, rows

            eager_optimizer, eager_scheduler, eager_counters, eager_updates = run_updates(False)
            expected_parameters = parameter_snapshot(model)
            expected_optimizer = optimizer_snapshot(model, eager_optimizer)
            expected_scheduler = eager_scheduler.state_dict()
            expected_counters = asdict(eager_counters)
            del eager_optimizer, eager_scheduler, eager_counters
            restore_parameters_in_place(model, initial_parameters)
            gc.collect()
            runner.validate_execution()
            actual_optimizer, actual_scheduler, actual_counters, actual_updates = run_updates(True)
            parameters = compare_tensor_mappings(dict(model.named_parameters()), expected_parameters,
                                                  atol=3e-6, rtol=3e-5, initial=initial_parameters)
            optimizer_state = compare_optimizer(model, actual_optimizer, expected_optimizer)
            schedule_equal = actual_scheduler.state_dict() == expected_scheduler
            counters_equal = asdict(actual_counters) == expected_counters
            update_comparisons = [compare_metrics(actual, expected)
                                  for actual, expected in zip(actual_updates, eager_updates)]
            publish("two-update-adam-parity", {"parameters": parameters, "optimizer": optimizer_state,
                "scheduler_equal": schedule_equal, "counters_equal": counters_equal,
                "updates": update_comparisons, "pointers_stable": pointers == pointer_snapshot(runner),
                "memory": memory(), "passed": parameters["passed"] and optimizer_state["passed"]
                    and schedule_equal and counters_equal and all(row["passed"] for row in update_comparisons)
                    and pointers == pointer_snapshot(runner)})
            del expected_parameters, expected_optimizer, initial_parameters
        if report["sources"] != source_hashes():
            raise AssertionError("Runtime source changed during probe")
        report["graph"] = {"warmup_backward_calls": runner.warmup_backward_calls,
            "capture_backward_calls": runner.capture_backward_calls, "replay_calls": runner.replay_calls,
            "captured_graphs": 1}
        report["status"] = "passed"
        tracker.summary({"diagnostic/status": "passed", "diagnostic/replay_calls": runner.replay_calls})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
