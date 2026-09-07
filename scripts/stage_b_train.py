#!/usr/bin/env python3
"""GPU-only paired symbolic training with aligned answer loss and resumable state.

The CLI must run from /workspace/cdrm-w-latent inside the project container.
Pure loss/schedule helpers can be imported in explicit CPU-container tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cdrm.synthetic.common import SyntheticBatch, load_batch
from stage_a_common import (
    configure_compiled_helpers, config_dict, provenance, require_cuda_container,
    restore_rng, rng_state, seed_all, unique_parameters,
)

FORMAT = "stage-b-training-v1"


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum CE only at already-aligned labels; never perform a next-token shift."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("Expected logits [B,T,V] and aligned int64 labels [B,T]")
    mask = labels != -100
    count = int(mask.sum().item())
    if not count:
        return logits.sum() * 0.0, 0
    selected = logits[mask]
    if selected.dtype in {torch.bfloat16, torch.float16}:
        selected = selected.float()
    return F.cross_entropy(selected, labels[mask], reduction="sum"), count


def learning_rate_at_update(completed_updates: int, settings: dict) -> float:
    """LR for update completed_updates+1, always using the planned full horizon."""
    horizon, warmup = int(settings["updates"]), int(settings["warmup_updates"])
    if horizon < 1 or not 0 <= warmup < horizon or not 0 <= completed_updates < horizon:
        raise ValueError("Invalid update, warmup, or schedule horizon")
    base, final = float(settings["learning_rate"]), float(settings["final_lr_multiplier"])
    if base <= 0 or not 0 <= final <= 1:
        raise ValueError("Invalid learning rate or final multiplier")
    update = completed_updates + 1
    if warmup and update <= warmup:
        return base * update / warmup
    progress = (update - warmup) / (horizon - warmup)
    return base * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress)))


def training_settings(plan: dict, fixed_batch: bool) -> dict:
    settings = copy.deepcopy(plan["training"])
    if fixed_batch:
        settings.update(updates=plan["calibration"]["updates"],
                        warmup_updates=plan["calibration"]["warmup_updates"],
                        global_batch=plan["calibration"]["fixed_examples"])
    for key in ("global_batch", "microbatch", "eval_microbatch", "eval_interval", "checkpoint_interval"):
        if int(settings[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if settings["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Precision must be explicitly fp32 or bf16")
    learning_rate_at_update(0, settings)
    return settings


def compiler_audit(required: bool, *, require_graphs: bool = True) -> dict:
    counters = {str(group): {str(key): int(value) for key, value in entries.items()}
                for group, entries in torch._dynamo.utils.counters.items()}
    if required:
        failures = {group: {key: value for key, value in counters.get(group, {}).items() if value}
                    for group in ("unimplemented", "graph_break")}
        failures = {group: values for group, values in failures.items() if values}
        if failures:
            raise RuntimeError(f"Compiled recurrent helpers encountered unsupported/fallback paths: {failures}")
        if not getattr(torch._dynamo.config, "fail_on_recompile_limit_hit", False):
            raise RuntimeError("Compiled execution must fail on recompile-limit fallback")
        if require_graphs and counters.get("stats", {}).get("unique_graphs", 0) == 0:
            raise RuntimeError("R3 requested compiled helpers but no compiled graph was observed")
    return {"required": required, "counters": counters,
            "recompile_limit": torch._dynamo.config.recompile_limit,
            "accumulated_recompile_limit": torch._dynamo.config.accumulated_recompile_limit,
            "fail_on_recompile_limit_hit": getattr(torch._dynamo.config, "fail_on_recompile_limit_hit", None)}


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), PROJECT_ROOT / "scripts/stage_a_common.py"]
    paths.extend(sorted((PROJECT_ROOT / "cdrm/synthetic").glob("*.py")))
    paths.extend(PROJECT_ROOT / "recurrent-transformer/olmo" / name for name in
                 ("model.py", "config.py", "checkpoint_conversion.py", "efficient_utils.py",
                  "initialization.py", "torch_util.py"))
    paths.append(PROJECT_ROOT / "vendors/mad-lab/mad/data/instances.py")
    return {str(path.relative_to(PROJECT_ROOT)): file_digest(path) for path in paths}


def atomic_json(path: Path, value, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite retained artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def build_model(plan: dict, topology: str, vocab_size: int):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model

    raw = copy.deepcopy(plan["model"])
    raw.update(vocab_size=vocab_size, embedding_size=vocab_size, init_device="cpu",
               recurrent_layers=[], recurrent_backend="tiled", recurrent_write_rho=1.0,
               reference_eager=False)
    seed_all(int(plan["seed"]), deterministic=True)
    source = OLMo(ModelConfig(**raw))
    if topology == "r3":
        target_config = copy.deepcopy(raw)
        target_config["recurrent_layers"] = [3]
        model = OLMo(ModelConfig(**target_config))
        conversion = convert_model(source, model).to_dict()
        del source
    else:
        model, conversion = source, None
    model = model.to(device="cuda", dtype=torch.float32)
    unique_parameters(model)
    # Corresponding parameter initialization and data are paired. Reset runtime
    # RNG equally despite R3's additional target construction before conversion.
    seed_all(int(plan["seed"]) + 1, deterministic=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return model, conversion


def load_fixtures(fixtures_dir: Path, task: str, plan: dict, split: str):
    path = fixtures_dir / task / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["task"] != task:
        raise ValueError("Fixture task mismatch")
    if manifest["primary_dev_conditions"] != plan["tasks"][task]["primary_dev_conditions"]:
        raise ValueError("Primary development conditions disagree with the plan")
    conditions = {}
    for name, condition in manifest["conditions"].items():
        if condition["config"] != plan["tasks"][task]["evaluation_conditions"][name]:
            raise ValueError(f"Fixture condition {name} differs from the frozen plan")
        entry = condition[split]
        fixture_path = Path(entry["path"])
        if not fixture_path.is_absolute():
            fixture_path = PROJECT_ROOT / fixture_path
        batch = load_batch(fixture_path)
        if batch.sha256 != entry["sha256"] or len(batch.input_ids) != entry["examples"]:
            raise ValueError(f"Fixture integrity failure: {fixture_path}")
        conditions[name] = batch
    if set(conditions) != set(plan["tasks"][task]["evaluation_conditions"]):
        raise ValueError("Missing or unexpected evaluation conditions")
    return manifest, conditions


def load_training_arrays(manifest: dict, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read the frozen, fully audited stream once, before any optimizer update."""
    entry = manifest["training"]
    path = Path(entry["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if file_digest(path) != entry["sha256_file"]:
        raise ValueError("Frozen training array file failed its SHA-256 check")
    if entry["global_batch"] != settings["global_batch"]:
        raise ValueError("Frozen training global batch differs from the plan")
    with np.load(path, allow_pickle=False) as arrays:
        inputs, labels = arrays["input_ids"], arrays["labels"]
    expected = settings["updates"] * settings["global_batch"]
    if inputs.shape != labels.shape or len(inputs) != expected or len(inputs) != entry["examples"]:
        raise ValueError("Frozen training arrays do not cover the exact planned horizon")
    stream = hashlib.sha256()
    for begin in range(0, expected, settings["global_batch"]):
        end = begin + settings["global_batch"]
        batch = SyntheticBatch(inputs[begin:end], labels[begin:end], [{}] * settings["global_batch"])
        stream.update(batch.sha256.encode())
    if stream.hexdigest() != entry["training_stream_sha256"]:
        raise ValueError("Frozen training stream digest differs from the audited stream")
    if manifest["audit"]["status"] != "passed" or stream.hexdigest() != manifest["audit"]["training_stream_sha256"]:
        raise ValueError("Frozen training stream lacks a successful matching split audit")
    return inputs, labels


def _counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    mask = labels != -100
    predictions = torch.zeros_like(labels)
    predictions[mask] = logits.detach()[mask].argmax(dim=-1)
    correct = (predictions == labels) & mask
    return int(correct.sum().item()), int((correct | ~mask).all(dim=1).sum().item())


def train_update(model, optimizer, batch: SyntheticBatch, settings: dict, completed_updates: int) -> dict:
    model.train()
    target_count = int((batch.labels != -100).sum())
    if target_count == 0:
        raise ValueError("A training update must contain scored answers")
    learning_rate = learning_rate_at_update(completed_updates, settings)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss_total = 0.0
    correct_answers = correct_sequences = 0
    for begin in range(0, len(batch.input_ids), int(settings["microbatch"])):
        ids = torch.as_tensor(batch.input_ids[begin:begin + settings["microbatch"]], device="cuda")
        labels = torch.as_tensor(batch.labels[begin:begin + settings["microbatch"]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
            logits = model(ids).logits
            loss_sum, _ = aligned_ce_sum(logits, labels)
        (loss_sum / target_count).backward()
        correct, sequences = _counts(logits, labels)
        correct_answers += correct
        correct_sequences += sequences
        loss_total += float(loss_sum.detach().item())
        del logits, loss_sum
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(settings["gradient_clip"]), error_if_nonfinite=True
    ).item())
    optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    loss = loss_total / target_count
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    return {"update": completed_updates + 1, "loss": loss,
            "answer_accuracy": correct_answers / target_count,
            "sequence_accuracy": correct_sequences / len(batch.input_ids),
            "correct_answers": correct_answers, "supervised_targets": target_count,
            "input_tokens": int(batch.input_ids.size), "examples": len(batch.input_ids),
            "learning_rate": learning_rate, "gradient_norm_before_clipping": gradient_norm,
            "update_seconds": seconds, "data_sha256": batch.sha256}


@torch.no_grad()
def evaluate(model, conditions: dict[str, SyntheticBatch], settings: dict, output: Path,
             *, completed_updates: int, split: str, write_predictions: bool,
             compiled_required: bool, primary: list[str]) -> dict:
    model.eval()
    result = {"update": completed_updates, "split": split, "conditions": {}}
    for name, batch in conditions.items():
        loss_total = 0.0
        correct_answers = correct_sequences = target_count = 0
        prediction_path = output / f"{split}-u{completed_updates:04d}-{name}-predictions.jsonl"
        if write_predictions and prediction_path.exists():
            raise FileExistsError(f"Refusing to overwrite retained predictions {prediction_path}")
        prediction_handle = None
        if write_predictions:
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_handle = prediction_path.open("x")
        try:
            for begin in range(0, len(batch.input_ids), int(settings["eval_microbatch"])):
                end = begin + settings["eval_microbatch"]
                ids = torch.as_tensor(batch.input_ids[begin:end], device="cuda")
                labels = torch.as_tensor(batch.labels[begin:end], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings["precision"] == "bf16"):
                    logits = model(ids).logits
                    loss_sum, count = aligned_ce_sum(logits, labels)
                correct, sequences = _counts(logits, labels)
                loss_total += float(loss_sum.item())
                target_count += count
                correct_answers += correct
                correct_sequences += sequences
                if prediction_handle:
                    mask = labels != -100
                    positions = mask.nonzero(as_tuple=False).cpu().tolist()
                    selected = logits[mask].float()
                    predictions = selected.argmax(dim=-1).cpu().tolist()
                    gold = labels[mask]
                    log_probabilities = (selected.gather(1, gold[:, None]).squeeze(1) - selected.logsumexp(dim=-1)).cpu().tolist()
                    gold = gold.cpu().tolist()
                    rows = []
                    for local_index in range(ids.shape[0]):
                        absolute = begin + local_index
                        rows.append({"example_index": absolute,
                                     "metadata": batch.metadata[absolute], "answers": []})
                    for (local_index, position), prediction, answer, log_probability in zip(
                        positions, predictions, gold, log_probabilities
                    ):
                        rows[local_index]["answers"].append({"position": position,
                            "prediction": prediction, "gold": answer,
                            "gold_log_probability": log_probability})
                    for row in rows:
                        prediction_handle.write(json.dumps(row, allow_nan=False) + "\n")
                del logits, loss_sum
        finally:
            if prediction_handle:
                prediction_handle.close()
        result["conditions"][name] = {
            "answer_ce": loss_total / target_count,
            "answer_accuracy": correct_answers / target_count,
            "sequence_accuracy": correct_sequences / len(batch.input_ids),
            "correct_answers": correct_answers, "supervised_targets": target_count,
            "examples": len(batch.input_ids), "input_tokens": int(batch.input_ids.size),
            "fixture_sha256": batch.sha256,
            "predictions": str(prediction_path) if write_predictions else None,
        }
    result["primary_macro_answer_ce"] = float(np.mean([
        result["conditions"][name]["answer_ce"] for name in primary
    ]))
    result["compiler_audit"] = compiler_audit(compiled_required)
    atomic_json(output / f"{split}-u{completed_updates:04d}-metrics.json", result)
    return result


def checkpoint_payload(model, optimizer, manifest: dict, completed: int,
                       counters: dict, best: dict, metrics: dict | None, next_data_sha: str) -> dict:
    return {"format": FORMAT, "run_id": manifest["run_id"], "identity": manifest["identity"],
            "identity_sha256": manifest["identity_sha256"], "model_config": config_dict(model.config),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
            "completed_updates": completed, "data_offset_examples": counters["examples"],
            "training_counters": copy.deepcopy(counters), "best_development": copy.deepcopy(best),
            "metrics": metrics, "schedule": manifest["settings"],
            "next_data_sha256": next_data_sha, "rho": model.config.recurrent_write_rho,
            "manifest": copy.deepcopy(manifest), "optimizer_policy": "exact_resume"}


def save_checkpoint(path: Path, payload: dict, *, replace: bool = False) -> dict:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite retained checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size,
            "completed_updates": payload["completed_updates"],
            "evidence_class": payload["manifest"]["evidence_class"]}


def validate_resume(payload: dict, identity: dict, model_config: dict, next_data_sha: str) -> None:
    if payload.get("format") != FORMAT:
        raise ValueError("Not a Stage B training checkpoint")
    if payload["identity"] != identity or payload["identity_sha256"] != json_digest(identity):
        raise ValueError("Exact-resume identity mismatch (plan, task, topology, fixtures, or source hashes)")
    if payload["model_config"] != model_config:
        raise ValueError("Exact-resume model configuration mismatch")
    if payload["next_data_sha256"] != next_data_sha:
        raise ValueError("Exact-resume data stream mismatch")
    if payload["data_offset_examples"] != payload["training_counters"]["examples"]:
        raise ValueError("Checkpoint data offset and counters disagree")


def validate_resume_boundary(topology: str, completed_updates: int) -> None:
    if topology == "r3" and completed_updates == 0:
        raise ValueError(
            "Compiled R3 resume from update 0 is disabled: the cold initialization resume path "
            "has not passed bitwise validation. Start a fresh paired R3 initialization, or resume "
            "a checkpoint after completed training updates; midpoint recovery passed exact validation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task", choices=("mqar", "noisy_recall", "state_tracking"), required=True)
    parser.add_argument("--topology", choices=("seq", "r3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--updates", "--stop", dest="updates", type=int,
                        help="Stop after this many total completed updates; preserve the full schedule horizon")
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--fixed-batch", action="store_true", help="Separate repeated-batch OPS calibration")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume", type=Path)
    checkpoints.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args()

    hardware = require_cuda_container()
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled_required = args.topology == "r3"
    configure_compiled_helpers(compiled_required)
    plan = json.loads(args.plan.read_text())
    evidence_class = "OPS" if args.fixed_batch else plan.get("evidence_class", "SYN")
    if evidence_class not in {"NUM", "OPS", "SYN"}:
        raise ValueError("Stage B evidence_class must be NUM, OPS, or SYN")
    settings = training_settings(plan, args.fixed_batch)
    if args.eval_interval is not None:
        if args.eval_interval < 1:
            raise ValueError("eval interval must be positive")
        settings["eval_interval"] = args.eval_interval
    stop = settings["updates"] if args.updates is None else args.updates
    if not 0 <= stop <= settings["updates"]:
        raise ValueError("Stop update must lie within the fixed schedule horizon")
    from cdrm.synthetic.experiment import generate_training_batch

    data_plan = copy.deepcopy(plan)
    if args.fixed_batch:
        data_plan["seed"] = plan["calibration"]["seed"]
        data_plan["training"]["global_batch"] = plan["calibration"]["fixed_examples"]
    fixed = generate_training_batch(data_plan, args.task, 0, split="calibration") if args.fixed_batch else None

    fixture_manifest_path = args.fixtures_dir / args.task / "manifest.json"
    fixture_manifest = json.loads(fixture_manifest_path.read_text())
    if fixture_manifest["plan_sha256"] not in {json_digest(plan), file_digest(args.plan)}:
        raise ValueError("Fixture manifest was generated from a different plan")
    training_arrays = None if args.fixed_batch else load_training_arrays(fixture_manifest, settings)

    def next_batch(completed):
        if fixed is not None:
            return fixed
        if completed < settings["updates"]:
            begin = completed * settings["global_batch"]
            end = begin + settings["global_batch"]
            return SyntheticBatch(training_arrays[0][begin:end], training_arrays[1][begin:end],
                                  [{}] * settings["global_batch"])
        # A completed run retains the next counter's digest for an explicit
        # stream-consistency check, although no extra update is performed.
        return generate_training_batch(data_plan, args.task, completed)
    identity = {"plan_sha256": file_digest(args.plan), "plan": plan, "task": args.task,
                "topology": args.topology, "fixed_batch": args.fixed_batch, "settings": settings,
                "fixture_manifest_sha256": file_digest(fixture_manifest_path),
                "source_sha256": source_hashes()}
    model, conversion = build_model(plan, args.topology, int(fixture_manifest["vocab_size"]))
    optimizer = torch.optim.AdamW(unique_parameters(model), lr=settings["learning_rate"],
        betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"],
        foreach=False, fused=False)
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    checkpoint = args.resume or args.evaluate_only
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
    completed = int(payload["completed_updates"]) if payload else 0
    if args.resume:
        validate_resume_boundary(args.topology, completed)
    if payload:
        validate_resume(payload, identity, config_dict(model.config), next_batch(completed).sha256)
        model.load_state_dict(payload["model"], strict=True)
        model.set_recurrent_write_rho(payload["rho"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
    if args.evaluate_only:
        if args.fixed_batch:
            raise ValueError("Fixed-batch OPS checkpoints cannot produce held-out SYN evaluation")
        if args.split == "test" and completed != settings["updates"]:
            raise ValueError("Final test fixtures may only evaluate the completed frozen-horizon checkpoint")
        _, conditions = load_fixtures(args.fixtures_dir, args.task, plan, args.split)
        with sdpa_kernel(SDPBackend.MATH):
            result = evaluate(model, conditions, settings, output, completed_updates=completed,
                split=args.split, write_predictions=True, compiled_required=compiled_required,
                primary=plan["tasks"][args.task]["primary_dev_conditions"])
        result.update(checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
                      identity_sha256=json_digest(identity), checkpoint_role="final" if completed == settings["updates"] else "partial")
        atomic_json(output / "evaluation-manifest.json", result)
        print(json.dumps({"status": "evaluated", "manifest": str(output / "evaluation-manifest.json")}))
        return
    if completed > stop:
        raise ValueError("Resume checkpoint is already beyond requested stop")
    if args.resume and completed == stop:
        raise ValueError("No remaining updates after this checkpoint; use --evaluate-only or a later --stop")
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text())
        if not args.resume or prior_manifest["identity_sha256"] != json_digest(identity):
            raise FileExistsError("Output directory belongs to an existing trajectory; choose a new path or exact resume")
        if prior_manifest["run_id"] != payload["run_id"]:
            raise ValueError("Resume checkpoint and output directory have different run owners")
        metric_path = output / "learning-curve.jsonl"
        if metric_path.exists():
            rows = [json.loads(line) for line in metric_path.read_text().splitlines() if line]
            if rows and max(row["update"] for row in rows) > completed:
                raise ValueError("Output has later updates than the checkpoint; resume into a fresh directory to preserve history")
        manifest = prior_manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing to claim a nonempty output directory")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"format": FORMAT, "run_id": payload["run_id"] if payload else uuid.uuid4().hex,
            "identity": identity, "identity_sha256": json_digest(identity),
            "evidence_class": evidence_class, "task": args.task,
            "topology": args.topology, "seed": plan["seed"], "settings": settings,
            "artifact_warning": "Debugging artifact; not held-out SYN learning evidence" if evidence_class != "SYN" else
                "Small symbolic diagnostic model; not a pretrained language-model checkpoint",
            "provenance": provenance(hardware), "conversion": conversion,
            "model_config": config_dict(model.config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parent_checkpoint": str(args.resume) if args.resume else None,
            "parent_checkpoint_sha256": file_digest(args.resume) if args.resume else None,
            "checkpoint_records": {}, "cuda_graphs": False, "model_compilation": False,
            "helper_compilation": compiled_required, "attention_backend": "deterministic_math_sdpa",
            "paired_data": "Counter-based training stream shared between SEQ and R3",
            "gradient_accumulation": {
                "global_batch": settings["global_batch"], "microbatch": settings["microbatch"],
                "used": settings["microbatch"] < settings["global_batch"],
                "r3_parameter_equivalence_validated": False,
                "note": "The frozen pilot uses global_batch=microbatch=64. Optional R3 accumulation "
                        "failed a parameter-equivalence check for a near-zero-gradient embedding element; "
                        "that failure is outside the pilot configuration, not waived."},
            "label_alignment": "already_aligned_with_logits_no_shift",
            "optimizer_policy": "exact_resume" if args.resume else "fresh_adamw",
            "initialization": "Corresponding random SEQ weights; exhaustive weights-only conversion for R3"}
        atomic_json(output / "resolved-config.json", {"plan": plan, "settings": settings,
                    "model": config_dict(model.config), "identity_sha256": json_digest(identity)})
    best = copy.deepcopy(payload["best_development"]) if payload else {"score": None, "update": None}
    counters = copy.deepcopy(payload["training_counters"]) if payload else {
        "examples": 0, "input_tokens": 0, "supervised_targets": 0, "update_seconds": 0.0,
        "elapsed_seconds": 0.0}
    prefix = f"{evidence_class}-{args.task}-{args.topology.upper()}-seed{plan['seed']}"
    primary = ["fixed_batch"] if args.fixed_batch else plan["tasks"][args.task]["primary_dev_conditions"]
    conditions = {"fixed_batch": fixed} if args.fixed_batch else load_fixtures(args.fixtures_dir, args.task, plan, "dev")[1]
    evaluation_output = output / "evaluations"
    last_evaluation = payload["metrics"] if payload else None
    invocation_started = time.perf_counter()
    previous_elapsed = counters["elapsed_seconds"]
    stop_requested = {"value": False}

    def request_stop(signum, frame):
        stop_requested["value"] = True
        print(f"Signal {signum}: will checkpoint at the next complete update boundary", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def retain(role, *, replace=False):
        counters["elapsed_seconds"] = previous_elapsed + time.perf_counter() - invocation_started
        snapshot = checkpoint_payload(model, optimizer, manifest, completed, counters, best,
                                      last_evaluation, next_batch(completed).sha256)
        path = output / f"{prefix}-{role}.pt"
        record = save_checkpoint(path, snapshot, replace=replace)
        manifest["checkpoint_records"][role] = record
        manifest.update(completed_updates=completed, training_counters=copy.deepcopy(counters),
                        best_development=copy.deepcopy(best))
        atomic_json(manifest_path, manifest, replace=True)

    with sdpa_kernel(SDPBackend.MATH):
        if not payload:
            retain("init")
        if last_evaluation is None:
            if completed != 0:
                raise ValueError("A noninitial checkpoint is missing its development evaluation state")
            initial_metrics_path = evaluation_output / "dev-u0000-metrics.json"
            if initial_metrics_path.exists() and args.resume:
                last_evaluation = json.loads(initial_metrics_path.read_text())
                if any(last_evaluation["conditions"][name]["fixture_sha256"] != batch.sha256
                       for name, batch in conditions.items()):
                    raise ValueError("Existing initial development metrics use different fixtures")
            else:
                last_evaluation = evaluate(model, conditions, settings, evaluation_output,
                    completed_updates=0, split="dev", write_predictions=False,
                    compiled_required=compiled_required, primary=primary)
            best = {"score": last_evaluation["primary_macro_answer_ce"], "update": 0}
            retain("bestdev", replace=True)
        for update in range(completed, stop):
            if stop_requested["value"] or previous_elapsed + time.perf_counter() - invocation_started >= settings["max_wall_seconds_per_run"]:
                break
            data_started = time.perf_counter()
            batch = next_batch(update)
            data_seconds = time.perf_counter() - data_started
            metrics = train_update(model, optimizer, batch, settings, update)
            completed = update + 1
            for key in ("examples", "input_tokens", "supervised_targets", "update_seconds"):
                counters[key] += metrics[key]
            metrics.update(data_generation_seconds=data_seconds, evidence_class=manifest["evidence_class"])
            append_jsonl(output / "learning-curve.jsonl", metrics)
            if completed % 20 == 0 or completed == stop:
                print(json.dumps({"update": completed, "loss": metrics["loss"],
                    "answer_accuracy": metrics["answer_accuracy"], "seconds": metrics["update_seconds"]}), flush=True)
            if completed % settings["eval_interval"] == 0 or completed == stop:
                last_evaluation = evaluate(model, conditions, settings, evaluation_output,
                    completed_updates=completed, split="dev", write_predictions=False,
                    compiled_required=compiled_required, primary=primary)
                score = last_evaluation["primary_macro_answer_ce"]
                if best["score"] is None or score < best["score"]:
                    best = {"score": score, "update": completed}
                    retain("bestdev", replace=True)
            if completed % settings["checkpoint_interval"] == 0:
                retain("latest", replace=True)
        if last_evaluation is None or last_evaluation["update"] != completed:
            last_evaluation = evaluate(model, conditions, settings, evaluation_output,
                completed_updates=completed, split="dev", write_predictions=False,
                compiled_required=compiled_required, primary=primary)
            score = last_evaluation["primary_macro_answer_ce"]
            if best["score"] is None or score < best["score"]:
                best = {"score": score, "update": completed}
                retain("bestdev", replace=True)
        manifest["compiler_audit"] = compiler_audit(compiled_required)
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Nonfinite final parameter {name}")
        role = "final" if completed == settings["updates"] else f"u{completed:04d}"
        manifest["status"] = "complete" if role == "final" else "paused_at_complete_update"
        manifest["requested_stop_update"] = stop
        retain(role)
        retain("latest", replace=True)
    print(json.dumps({"status": manifest["status"], "completed_updates": completed,
                      "manifest": str(manifest_path), "best_development": best}), flush=True)


if __name__ == "__main__":
    main()
