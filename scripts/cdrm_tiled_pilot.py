#!/usr/bin/env python3
"""Bounded three-arm learning pilot with immutable numerical sources and ancestry.

CUDA execution is required. The new child lineage can continue an original
100-update CDRM checkpoint or its own checkpoints without resetting Adam,
epoch-cosine scheduling, RNG, or the shared example permutation. SEQ starts
from exactly the corresponding backbone, removing only the two CDRM adapters.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
from pathlib import Path
import shlex
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import cdrm_tiled_common as common
from cdrm_tiled_operational import OPTIMIZER, SCHEDULE, SEEDS
from cdrm_train import compare_resumed
from experiment_tracking import OnlineTracker, add_wandb_arguments, scalar_metrics

PILOT_SCHEMA = "cdrm-tiled-pilot-v1"
FORMAT = common.FORMAT
ARMS = {"seq-fp32": "fp32", "cdrm-fp32": "fp32", "cdrm-bf16": "bf16_fp32_state"}
ADAPTERS = {"cdrm.deep_adapter.weight", "cdrm.bridge_adapter.weight"}
MAX_UPDATES = 2500
BATCH = 64
UPDATES_PER_EPOCH = 200
PROTOCOL = Path("docs/reports/cdrm-tiled-pilot/protocol.md")
RUNNER = Path("scripts/cdrm_tiled_pilot.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-final", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, default=common.PRESET)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--stop-updates", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-updates", type=int, nargs="+", default=[190, 210, 1000, 2500])
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-tiled-learning-pilot")
    args = parser.parse_args(argv)
    if not 1 <= args.stop_updates <= MAX_UPDATES:
        parser.error("The bounded pilot requires 1..2500 total updates")
    if args.eval_every != 200:
        parser.error("The frozen pilot evaluates every 200 updates plus the endpoint")
    if any(not 1 <= value <= MAX_UPDATES for value in args.save_updates):
        parser.error("Checkpoint calendar must lie within 1..2500")
    args.save_updates = sorted(set(args.save_updates) | {190, 210, 1000, 2500})
    if args.reference_final and not args.checkpoint:
        parser.error("--reference-final requires a retained --checkpoint")
    if not args.wandb_project:
        parser.error("The learning pilot requires online W&B tracking")
    return args


def reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": common.file_digest(path), "bytes": path.stat().st_size}


def verify_reference(record):
    if common.file_digest(Path(record["path"])) != record["sha256"]:
        raise ValueError("An immutable ancestry artifact changed")


def validate_initial(initial, preset):
    if (initial.get("format") != common.INIT_FORMAT or initial.get("completed_updates") != 0
            or initial.get("completed_epochs") != 0 or initial.get("batch_in_epoch") != 0
            or initial["optimizer"]["state"] or initial["identity_sha256"] != common.json_digest(initial["identity"])):
        raise ValueError("Require the retained five-block initialization and empty Adam state")
    common.verify_sources(initial["identity"]["source_sha256"])
    if initial["identity"]["preset_sha256"] != common.file_digest(preset):
        raise ValueError("Initialization preset identity changed")
    expected = dataclasses.asdict(common.config("naive", "fp32", preset=preset))
    if initial["model_config"] != expected or initial["identity"]["model_config"] != expected:
        raise ValueError("Initialization differs from the validated five-block architecture")
    if initial["identity"]["optimizer"] != OPTIMIZER or initial["identity"]["schedule"] != SCHEDULE:
        raise ValueError("Initialization optimizer or epoch schedule changed")
    if initial["identity"]["seeds"]["shuffle"] != SEEDS["shuffle"]:
        raise ValueError("The pilot retains the original shared data-order seed")
    if initial["identity"]["initialization"] != initial["initialization"]:
        raise ValueError("Initialization provenance disagrees with its identity")
    if common.state_digest(initial["model"]) != initial["initialization"]["full_initialization_sha256"]:
        raise ValueError("Saved full initialization weights changed")
    backbone = {name: value for name, value in initial["model"].items() if name not in ADAPTERS}
    if (set(initial["model"]) - set(backbone) != ADAPTERS
            or common.state_digest(backbone) != initial["initialization"]["backbone_initialization_sha256"]):
        raise ValueError("Saved common backbone or adapter ownership changed")


def build_arm_model(initial, arm, *, device="cuda", preset=common.PRESET):
    """CPU construction is exposed for tests; the training entry point is CUDA-only."""
    if arm not in ARMS:
        raise ValueError("Unsupported pilot arm")
    if arm != "seq-fp32":
        return common.build_model("tiled", ARMS[arm], checkpoint=initial, device=device, preset=preset)
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    raw = copy.deepcopy(initial["model_config"])
    raw.update(cdrm_enabled=False, cdrm_backend="tiled", cdrm_precision_policy="fp32", reference_eager=False)
    model = OLMo(ModelConfig(**raw)).to(device=device, dtype=torch.float32)
    weights = {name: value for name, value in initial["model"].items() if name not in ADAPTERS}
    if set(initial["model"]) - set(weights) != ADAPTERS or set(model.state_dict()) != set(weights):
        raise ValueError("SEQ must remove exactly the two CDRM adapters")
    model.load_state_dict(weights, strict=True)
    digest = common.state_digest(model.state_dict())
    if digest != initial["initialization"]["backbone_initialization_sha256"]:
        raise AssertionError("SEQ did not preserve the corresponding backbone exactly")
    if model.config.cdrm_enabled or getattr(model, "cdrm", None) is not None or any(name.startswith("cdrm.") for name, _ in model.named_parameters()):
        raise AssertionError("SEQ still has active CDRM ownership")
    parameters = list(model.parameters())
    if len(parameters) != len({id(parameter) for parameter in parameters}):
        raise AssertionError("SEQ optimizer would have duplicate owners")
    return model, {"model_config": dataclasses.asdict(model.config), "starting_weights_sha256": digest,
                   "backbone_initialization_sha256": digest, "removed_parameter_names": sorted(ADAPTERS),
                   "parameter_count": sum(parameter.numel() for parameter in parameters),
                   "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters)}


def cursor_at(completed):
    if not 0 <= completed <= MAX_UPDATES:
        raise ValueError("Update position lies outside the bounded pilot")
    return divmod(completed, UPDATES_PER_EPOCH)


def expected_rate(update):
    epoch = (update - 1) // UPDATES_PER_EPOCH
    return SCHEDULE["eta_min"] + (OPTIMIZER["lr"] - SCHEDULE["eta_min"]) * (1 + math.cos(math.pi * epoch / 200)) / 2


def batch_indices(completed, train_size, shuffle_seed):
    epoch, position = cursor_at(completed)
    return common.epoch_indices(train_size, epoch, shuffle_seed)[position * BATCH:(position + 1) * BATCH]


def advance_cursor(epochs, position, scheduler):
    position += 1
    if position == UPDATES_PER_EPOCH:
        epochs += 1
        position = 0
        scheduler.step()
    return epochs, position


def validate_position(payload, train, shuffle_seed):
    completed = payload["completed_updates"]
    if cursor_at(completed) != (payload["completed_epochs"], payload["batch_in_epoch"]):
        raise ValueError("Checkpoint update/epoch/batch cursor disagrees")
    history = payload["history"]
    if len(history) != completed:
        raise ValueError("Checkpoint is missing its full inherited training history")
    permutations = {}
    for update, row in enumerate(history, 1):
        epoch, position = cursor_at(update - 1)
        if (row["update"], row["epoch"], row["batch_in_epoch"]) != (update, epoch + 1, position):
            raise ValueError("History update/epoch/batch ordering changed")
        if not math.isclose(row["learning_rate"], expected_rate(update), rel_tol=1e-12, abs_tol=1e-16):
            raise ValueError("History violates the retained 200-epoch learning-rate schedule")
        if epoch not in permutations:
            permutations = {epoch: common.epoch_indices(len(train), epoch, shuffle_seed)}
        indices = permutations[epoch][position * BATCH:(position + 1) * BATCH]
        if row["indices_sha256"] != common.state_digest(indices) or row["batch_sha256"] != train.take(indices).sha256:
            raise ValueError("History no longer matches the frozen shared data order")
    if payload["next_batch_sha256"] != train.take(batch_indices(completed, len(train), shuffle_seed)).sha256:
        raise ValueError("The restored next batch differs")
    if payload["scheduler"]["last_epoch"] != payload["completed_epochs"]:
        raise ValueError("Scheduler does not match the completed epoch count")
    next_rate = expected_rate(completed + 1)
    if any(not math.isclose(group["lr"], next_rate, rel_tol=1e-12, abs_tol=1e-16)
           for group in payload["optimizer"]["param_groups"]):
        raise ValueError("Saved optimizer LR does not match its restored epoch")


def pilot_sources(initial, preset, protocol):
    tracked = common.sources([preset, protocol, RUNNER])
    for name, expected in initial["identity"]["source_sha256"].items():
        if name in tracked and tracked[name] != expected:
            raise ValueError("A parent source differs from the new child source inventory")
        tracked[name] = expected
    common.verify_sources(tracked)
    return dict(sorted(tracked.items()))


def legacy_identity(initial, initial_ref, cfg, precision, runtime, data):
    return {"schema": FORMAT, "initial_checkpoint_sha256": initial_ref["sha256"],
            "initial_identity_sha256": initial["identity_sha256"], "source_sha256": common.sources([common.PRESET]),
            "model_config": cfg, "precision": precision, "execution_contract": runtime,
            "initialization": initial["initialization"], "data": data, "physical_batch": BATCH,
            "shuffle_seed": SEEDS["shuffle"], "data_order": "shared_epoch_permutation", "accumulation": False,
            "optimizer": OPTIMIZER, "schedule": SCHEDULE, "updates_per_epoch": UPDATES_PER_EPOCH,
            "stop_updates": 100, "midpoint": 50, "purpose": "paired_training"}


def check_payload_identity(payload):
    if payload.get("format") != FORMAT or payload.get("identity_sha256") != common.json_digest(payload["identity"]):
        raise ValueError("Checkpoint format or identity digest mismatch")
    if payload["model_config"] != payload["identity"]["model_config"] or payload["precision"] != payload["identity"]["precision"]:
        raise ValueError("Checkpoint config or precision disagrees with its identity")
    common.verify_sources(payload["identity"]["source_sha256"])


def validate_legacy_parent(payload, expected, arm):
    check_payload_identity(payload)
    if (arm == "seq-fp32" or payload.get("pilot_schema") is not None
            or payload["completed_updates"] != 100 or payload["identity"] != expected):
        raise ValueError("Only the compatible original CDRM update-100 checkpoint may enter this child lineage")


def validate_resume_target(payload, identity, target, *, recovery=False):
    check_payload_identity(payload)
    if payload.get("pilot_schema") != PILOT_SCHEMA or payload["identity"] != identity:
        raise ValueError("Pilot source/config/runtime/data/arm/ancestry identity changed")
    if target <= payload["completed_updates"]:
        raise ValueError("Continuation must execute at least one new update")
    if target < payload["run_target_updates"] and not recovery:
        raise ValueError("Only an explicit exact-recovery control may shorten the declared target")


def optimizer_and_scheduler(model, initial, arm):
    optimizer = common.optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    if arm != "seq-fp32":
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    else:
        a, b = [copy.deepcopy(value["param_groups"]) for value in (initial["optimizer"], optimizer.state_dict())]
        for groups in (a, b):
            for group in groups:
                group.pop("params")
        if a != b:
            raise AssertionError("SEQ optimizer settings differ from the corresponding CDRM initialization")
    scheduler.load_state_dict(copy.deepcopy(initial["scheduler"]))
    return optimizer, scheduler


def make_identity(args, initial, initial_ref, construction, runtime, data, tracked, ancestry):
    return {"schema": PILOT_SCHEMA, "arm": args.arm, "precision": ARMS[args.arm],
            "initial_checkpoint_sha256": initial_ref["sha256"], "initial_identity_sha256": initial["identity_sha256"],
            "initialization": initial["initialization"],
            "backbone_initialization_sha256": initial["initialization"]["backbone_initialization_sha256"],
            "source_sha256": tracked, "parent_source_sha256": initial["identity"]["source_sha256"],
            "model_config": construction["model_config"], "execution_contract": runtime, "data": data,
            "physical_batch": BATCH, "shuffle_seed": SEEDS["shuffle"], "data_order": "shared_epoch_permutation",
            "accumulation": False, "optimizer": OPTIMIZER, "schedule": SCHEDULE,
            "updates_per_epoch": UPDATES_PER_EPOCH, "maximum_updates": MAX_UPDATES,
            "eval_every": args.eval_every, "checkpoint_updates": args.save_updates,
            "ablation_updates": [1000, 2500], "protocol_sha256": common.file_digest(args.protocol),
            "origin": {key: ancestry["origin"][key] for key in ("sha256", "identity_sha256", "completed_updates", "kind")}}


def seq_update(model, optimizer, ids, labels, answers):
    """The common FP32 update, with CDRM diagnostics explicitly disabled for SEQ."""
    if model.config.cdrm_enabled or getattr(model, "cdrm", None) is not None:
        raise ValueError("SEQ update requires an actually disabled CDRM")
    model.train()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        logits = model(ids, output_cdrm_states=False).logits
    if logits.dtype != torch.float32:
        raise AssertionError("SEQ must produce FP32 logits")
    summed, count = common.loss_sum(logits, labels)
    loss = summed / count
    loss.backward()
    if any(parameter.dtype != torch.float32 or parameter.grad is None or parameter.grad.dtype != torch.float32
           for parameter in model.parameters()):
        raise AssertionError("Every SEQ parameter must receive an FP32 gradient")
    signals = {name: {"l2": float(parameter.grad.detach().double().norm().item()),
                      "nonzero": bool(torch.count_nonzero(parameter.grad).item())}
               for name, parameter in model.named_parameters()
               if name.startswith(f"transformer.blocks.{model.config.cdrm_early_layer}.")}
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item())
    optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not math.isfinite(loss.item()) or loss.dtype != torch.float32:
        raise FloatingPointError("SEQ native CE must be finite FP32")
    return {"native_loss": float(loss.item()), "native_targets": count, "seconds": seconds,
            "gradient_norm": norm, "clipped": norm > 1., "clip_coefficient": min(1., 1. / (norm + 1e-6)),
            "input_tokens": ids.numel(), "learning_rate": optimizer.param_groups[0]["lr"],
            "logits_dtype": str(logits.dtype), "ce_dtype": str(loss.dtype),
            "precision": common.effective_precision(model, optimizer, require_gradients=True),
            "state_summaries": {}, "parameter_gradient_signals": signals,
            "update_l2": float(torch.stack([(parameter.detach().double() - old.double()).square().sum()
                                            for parameter, old in zip(model.parameters(), before)]).sum().sqrt().item()),
            "answer": common.finish_metrics(common.metric_counts(logits.detach(), answers))}


def evaluate_preserving_state(model, optimizer, data, precision, *, ablate=False):
    modes = {module: module.training for module in model.modules()}
    before = {"model": common.state_digest(model.state_dict()), "optimizer": common.state_digest(optimizer.state_dict()),
              "gradients": common.state_digest({name: parameter.grad for name, parameter in model.named_parameters()})}
    original_lambda = model.config.cdrm_lambda
    if ablate and (not model.config.cdrm_enabled or model.cdrm.config is not model.config):
        raise ValueError("Lambda-zero evaluation requires the active shared CDRM configuration")
    result = {}
    try:
        with common.preserve_tracking_rng():
            result["active"] = common.evaluate(model, data, precision=precision)
            if ablate:
                model.config.cdrm_lambda = 0.
                result["lambda_zero"] = common.evaluate(model, data, precision=precision)
    finally:
        model.config.cdrm_lambda = original_lambda
        for module, training in modes.items():
            module.training = training
    after = {"model": common.state_digest(model.state_dict()), "optimizer": common.state_digest(optimizer.state_dict()),
             "gradients": common.state_digest({name: parameter.grad for name, parameter in model.named_parameters()})}
    if before != after:
        raise AssertionError("Development or lambda-zero evaluation changed training state")
    result["training_state_unchanged"] = True
    return result


def without_times(value):
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


def compare_pilot_recovery(reference_payload, actual):
    result = compare_resumed(reference_payload, actual)
    for key in ("ancestry", "model_config", "precision", "initialization", "ablation_development"):
        if common.state_digest(without_times(reference_payload.get(key))) != common.state_digest(without_times(actual.get(key))):
            result["differences"][key] = {"exact_state_or_nontiming_metrics_differ": True}
    if set(reference_payload["development"]) != set(actual["development"]):
        result["differences"]["development/keys"] = {"evaluation_scope_differs": True}
    result["bitwise_state_and_nontiming_metrics_equal"] = not result["differences"]
    result["excluded"].extend(["declared invocation stopping target", "immediate resume checkpoint reference", "execution role"])
    return result


def run(args, report, tracker):
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    validate_initial(initial, args.preset)
    initial_ref = reference(args.initial_checkpoint)
    report.update(common.setup(initial["initialization"]["training_seed"]))
    runtime = report["execution_contract"]
    if not runtime["inductor_cache_directory"]:
        raise ValueError("A retained shared compiler cache is required")
    tracked = pilot_sources(initial, args.preset, args.protocol)
    common.snapshot_sources(args.output_dir, tracked)
    train, dev = (common.load_dataset(args.data_root, common.TASK, split) for split in ("train", "dev"))
    data = {"train": common.dataset_identity(train), "dev": common.dataset_identity(dev)}
    for name, dataset in (("train", train), ("dev", dev)):
        common.validate_data(dataset)
        if data[name] != initial["identity"]["data"][name]:
            raise ValueError("The pilot must retain its original train/development arrays and manifests")
    if len(train) != 12800 or len(dev) != 256:
        raise ValueError("The bounded pilot retains 12800 train and 256 development examples")
    model, construction = build_arm_model(initial, args.arm, preset=args.preset)
    optimizer, scheduler = optimizer_and_scheduler(model, initial, args.arm)
    common.seed_all(initial["initialization"]["training_seed"], deterministic=True)
    restored = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    origin = {**initial_ref, "identity_sha256": initial["identity_sha256"], "completed_updates": 0, "kind": "initialization"}
    ancestry = {"initial": initial_ref, "origin": origin}
    old_expected = legacy_identity(initial, initial_ref, construction["model_config"], ARMS[args.arm], runtime, data)
    if restored is not None:
        check_payload_identity(restored)
        if restored.get("pilot_schema") == PILOT_SCHEMA:
            ancestry = copy.deepcopy(restored["ancestry"])
            if ancestry["initial"]["sha256"] != initial_ref["sha256"]:
                raise ValueError("Pilot ancestry refers to a different initialization")
            verify_reference(ancestry["initial"])
            verify_reference(ancestry["origin"])
        else:
            validate_legacy_parent(restored, old_expected, args.arm)
            ancestry["origin"] = {**reference(args.checkpoint), "identity_sha256": restored["identity_sha256"],
                                  "completed_updates": 100, "kind": "legacy_operational"}
    identity = make_identity(args, initial, initial_ref, construction, runtime, data, tracked, ancestry)
    if restored is not None and restored.get("pilot_schema") == PILOT_SCHEMA:
        validate_resume_target(restored, identity, args.stop_updates, recovery=bool(args.reference_final))
    reference_payload = None
    if args.reference_final:
        reference_payload = torch.load(args.reference_final, map_location="cpu", weights_only=False)
        check_payload_identity(reference_payload)
        if (reference_payload.get("pilot_schema") != PILOT_SCHEMA or reference_payload["identity"] != identity
                or reference_payload["completed_updates"] != args.stop_updates):
            raise ValueError("Exact recovery requires the same child lineage and stopping update")
    history, development, ablations = [], {}, {}
    completed = epochs = position = 0
    if restored is not None:
        if args.stop_updates <= restored["completed_updates"]:
            raise ValueError("The new target must exceed the checkpoint update")
        validate_position(restored, train, SEEDS["shuffle"])
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        common.restore_rng(restored["rng"])
        for name, current in (("model", model.state_dict()), ("optimizer", optimizer.state_dict()),
                              ("scheduler", scheduler.state_dict()), ("rng", common.rng_state())):
            if common.state_digest(current) != common.state_digest(restored[name]):
                raise AssertionError(f"{name} did not restore exactly")
        completed, epochs, position = (restored[key] for key in ("completed_updates", "completed_epochs", "batch_in_epoch"))
        history = copy.deepcopy(restored["history"])
        development = copy.deepcopy(restored["development"])
        ablations = copy.deepcopy(restored.get("ablation_development", {}))
    report.update(identity=identity, identity_sha256=common.json_digest(identity), construction=construction,
                  model_config=construction["model_config"], precision=ARMS[args.arm], initialization=initial["initialization"],
                  ancestry=ancestry, initial_checkpoint=initial_ref,
                  resume={"checkpoint": reference(args.checkpoint) if args.checkpoint else None,
                          "loaded_state_exact": restored is not None, "starting_update": completed,
                          "cold_process": True, "warmup_updates": 0},
                  history=history, development=development, ablation_development=ablations, checkpoints={},
                  effective_precision_at_start=common.effective_precision(model, optimizer, require_gradients=False))
    common.atomic_json(args.output_dir / "resolved-config.json", identity)
    tracker.start({"evidence_class": "bounded_learning_pilot", "arm": args.arm, "seed": initial["initialization"]["seed"],
                   "identity": identity, "target_updates": args.stop_updates})
    common.atomic_json(args.output_dir / "wandb-run.json", tracker.record)
    print(f"W&B: {tracker.record['run_url']}", flush=True)

    def snapshot():
        return {"format": FORMAT, "pilot_schema": PILOT_SCHEMA, "identity": identity,
                "identity_sha256": common.json_digest(identity), "model_config": construction["model_config"],
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": common.rng_state(), "completed_updates": completed, "completed_epochs": epochs,
                "batch_in_epoch": position,
                "next_batch_sha256": train.take(batch_indices(completed, len(train), SEEDS["shuffle"])).sha256,
                "history": history, "development": development, "ablation_development": ablations,
                "initialization": initial["initialization"], "initial_checkpoint": initial_ref, "ancestry": ancestry,
                "precision": ARMS[args.arm], "memory_state": "Rebuilt for each independent forward",
                "run_target_updates": args.stop_updates}

    if completed == 0:
        development["0"] = evaluate_preserving_state(model, optimizer, dev, ARMS[args.arm])["active"]
        report["checkpoints"]["0"] = common.save_checkpoint(args.output_dir / "u0000.pt", snapshot())
    # Inherited observations retain their original global update coordinates.
    for update in range(completed + 1):
        logged = {"update": update}
        if update:
            logged.update(scalar_metrics(history[update - 1], "train"))
        if str(update) in development:
            logged.update(scalar_metrics(development[str(update)], "dev"))
        if str(update) in ablations:
            logged.update(scalar_metrics(ablations[str(update)], "dev/ablation"))
        if len(logged) > 1:
            tracker.log(logged)
    initial_completed = completed
    while completed < args.stop_updates:
        indices = batch_indices(completed, len(train), SEEDS["shuffle"])
        batch = train.take(indices)
        ids, labels, answers = (torch.as_tensor(array, device="cuda") for array in (batch.input_ids, batch.labels, batch.answer_labels))
        if args.arm == "seq-fp32":
            row = seq_update(model, optimizer, ids, labels, answers)
        else:
            row = common.update(model, optimizer, ids, labels, answers, precision=ARMS[args.arm])
        row.update(update=completed + 1, epoch=epochs + 1, batch_in_epoch=position,
                   batch_sha256=batch.sha256, indices_sha256=common.state_digest(indices))
        completed += 1
        epochs, position = advance_cursor(epochs, position, scheduler)
        history.append(row)
        common.append_jsonl(args.output_dir / "learning-curve.jsonl", row)
        logged = {"update": completed, **scalar_metrics(row, "train")}
        terminal_evaluation = completed == args.stop_updates and (
            reference_payload is None or str(completed) in reference_payload["development"])
        if completed % args.eval_every == 0 or terminal_evaluation:
            result = evaluate_preserving_state(model, optimizer, dev, ARMS[args.arm],
                                                ablate=args.arm != "seq-fp32" and completed in (1000, 2500))
            development[str(completed)] = result["active"]
            logged.update(scalar_metrics(result["active"], "dev"))
            if "lambda_zero" in result:
                ablations[str(completed)] = result
                logged.update(scalar_metrics(result, "dev/ablation"))
        if completed in args.save_updates or completed == args.stop_updates:
            common.verify_sources(tracked)
            report["checkpoints"][str(completed)] = common.save_checkpoint(args.output_dir / f"u{completed:04d}.pt", snapshot())
        tracker.log(logged)
        if completed % 100 == 0 or completed == args.stop_updates:
            print(f"{args.arm} update {completed}: native CE {row['native_loss']:.7f}", flush=True)
    report.update(completed_updates=completed, completed_epochs=epochs, batch_in_epoch=position,
                  training_seconds=sum(row["seconds"] for row in history),
                  invocation_training_seconds=sum(row["seconds"] for row in history[initial_completed:]),
                  clipping_count=sum(row["clipped"] for row in history),
                  final_precision=common.effective_precision(model, optimizer, require_gradients=True),
                  compiler=common.compiler_audit(True, require_graphs=args.arm != "seq-fp32"))
    final = common.cpu_tree(snapshot())
    validate_position(final, train, SEEDS["shuffle"])
    if reference_payload is not None:
        report["recovery_comparison"] = compare_pilot_recovery(reference_payload, final)
        common.atomic_json(args.output_dir / "recovery-comparison.json", report["recovery_comparison"])
        if not report["recovery_comparison"]["bitwise_state_and_nontiming_metrics_equal"]:
            raise AssertionError("Exact recovery failed; all differences are retained")
    common.verify_sources(tracked)
    if pilot_sources(initial, args.preset, args.protocol) != tracked:
        raise RuntimeError("Source inventory changed during the pilot")
    verify_reference(initial_ref)
    verify_reference(ancestry["origin"])
    report["status"] = "complete"


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    forbidden = [Path(".runtime") / name for name in ("stage-b", "r3-backward", "r3-bf16", "cdrm-naive",
                                                     "r3-bf16-tiled-resolution", "cdrm-tiled-bf16")]
    if any(path.resolve() in output.parents for path in forbidden):
        raise ValueError("Original experiment lineages are immutable; create a new pilot lineage")
    output.mkdir(parents=True)
    report = {"schema": PILOT_SCHEMA, "format": FORMAT, "evidence": "bounded_learning_pilot", "status": "running",
              "arm": args.arm, "command": shlex.join([sys.executable, *sys.argv]),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                            name=args.wandb_run_name, output_dir=output, preserve_state=common.preserve_tracking_rng)
    report["wandb"] = tracker.record
    started = time.monotonic()
    try:
        run(args, report, tracker)
        tracker.summary({"experiment_status": report["status"],
                         **scalar_metrics({key: report[key] for key in ("completed_updates", "completed_epochs", "training_seconds",
                                                                       "invocation_training_seconds", "clipping_count", "final_precision")}),
                         **scalar_metrics(report.get("recovery_comparison", {}), "recovery")})
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "complete")
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            common.atomic_json(output / "report.json", report)


if __name__ == "__main__":
    main()
