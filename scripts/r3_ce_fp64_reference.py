#!/usr/bin/env python3
"""Naive-only FP64 mean-CE fallback on an already retained R3 NUM fixture.

This diagnostic reconstructs the exact pre-step FP32 weights from saved post-
step weights and FP64 deltas. It performs no optimizer step and never executes
tiled recurrence in FP64. Every model operation must run in the GPU container.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from r3_backward_validate import backward_packet, compare_maps, forward_graph, tensors
from r3_validation_metrics import compare_tensors
from stage_a_common import provenance, require_cuda_container, seed_all, unique_parameters


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reconstruct_initial_weights(saved: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """Reverse each saved FP64 delta and verify an exact FP32-representable start."""
    reconstructed = {}
    audit = {}
    for backend in ("naive", "tiled"):
        step = saved["steps"][backend]
        after, deltas = step["weights"], step["deltas"]
        if list(after) != list(deltas):
            raise AssertionError(f"{backend}: post-step weights/deltas have different keys or order")
        initial = {}
        for name, weight in after.items():
            delta = deltas[name]
            if weight.dtype != torch.float32 or delta.dtype != torch.float64 or weight.shape != delta.shape:
                raise AssertionError(f"{backend}/{name}: expected FP32 weights and shape-matched FP64 deltas")
            if not torch.isfinite(weight).all() or not torch.isfinite(delta).all():
                raise AssertionError(f"{backend}/{name}: nonfinite saved weight or delta")
            before_double = weight.double() - delta
            before_float = before_double.float()
            if not torch.equal(before_float.double(), before_double):
                raise AssertionError(f"{backend}/{name}: reconstructed initial weight is not exactly FP32 representable")
            if not torch.equal(before_double + delta, weight.double()):
                raise AssertionError(f"{backend}/{name}: reconstruction does not recover the saved post-step weight")
            initial[name] = before_double
        reconstructed[backend] = initial
        audit[backend] = {"parameter_tensors": len(initial), "elements": sum(x.numel() for x in initial.values()),
                          "fp32_roundtrip_exact": True, "post_step_reconstruction_exact": True}
    if list(reconstructed["naive"]) != list(reconstructed["tiled"]):
        raise AssertionError("Reconstructed initial parameter names/order differ between arms")
    if any(not torch.equal(value, reconstructed["tiled"][name])
           for name, value in reconstructed["naive"].items()):
        raise AssertionError("Reconstructed initial weights were not identical between arms")
    audit["between_arms_exact"] = True
    audit["formula"] = "saved steps[arm].weights.double() - saved steps[arm].deltas (FP64)"
    return reconstructed["naive"], audit


def adapt_mixed_reference(source_report: dict, saved: dict) -> tuple[dict, dict, dict]:
    """Map the two saved FP32 arms of a mixed audit to the original CE schema.

    No values are cast or recomputed. BF16 arms are deliberately excluded from
    the reference reconstruction. Preserve the original model/configuration and
    fixture identities while supplying the legacy field names used below.
    """
    if source_report.get("schema") != "r3-mixed-validation-v1" or source_report.get("status") != "diagnostics_complete":
        raise ValueError("Mixed reference must be a completed r3-mixed-validation-v1 diagnostic")
    settings = source_report["settings"]
    for key, expected in {"parameters": "FP32", "tf32": False, "sdpa": "math", "deterministic": True}.items():
        if settings.get(key) != expected:
            raise ValueError(f"Mixed reference uses incompatible {key}: {settings.get(key)!r}")
    fixture = source_report["fixture"]
    if fixture.get("alignment") != "Stage B aligned answer-only CE / total scored answers, no shift":
        raise ValueError("Mixed reference does not declare the authoritative aligned answer-only mean CE")
    if fixture.get("accumulation") is not False:
        raise ValueError("Mixed reference must retain a physical batch without accumulation")
    if not source_report.get("arguments", {}).get("include_tiled_fp32"):
        raise ValueError("Mixed reference did not declare execution of a tiled FP32 arm")
    if "model_config" not in saved:
        raise ValueError("Mixed reference has no saved authoritative model configuration")
    gradients, steps, contracts = {}, {}, {}
    for backend in ("naive", "tiled"):
        arm = backend + "_fp32"
        if arm not in saved.get("packets", {}) or arm not in saved.get("steps", {}):
            raise ValueError(f"Mixed reference lacks required saved arm: {arm}")
        packet, step = saved["packets"][arm], saved["steps"][arm]
        order = list(step["weights"])
        if list(packet["parameters"]) != order:
            raise AssertionError(f"{arm}: parameter-gradient order differs from saved weights")
        if list(step["deltas"]) != order:
            raise AssertionError(f"{arm}: saved parameter delta order differs from weights")
        if set(packet["inputs"]) != {"embedding_output", "block3_input"}:
            raise AssertionError(f"{arm}: expected the actual embedding and recurrent-block input gradients")
        checks = {**packet["parameters"],
                  **{f"input/{name}": value for name, value in packet["inputs"].items()},
                  "logits": packet["logits"], "cotangent": packet["cotangent"]}
        for name, value in checks.items():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise AssertionError(f"{arm}/{name}: saved FP32 arm must be present, finite and FP32")
        if packet.get("loss") is None or not isinstance(packet["loss"], (int, float)):
            raise AssertionError(f"{arm}: missing actual mean CE")
        gradients[backend] = {**packet, "parameter_order": order}
        steps[backend] = step
        contracts[arm] = {"parameter_tensors": len(order), "gradient_tensors_including_inputs": len(order) + len(packet["inputs"]),
                          "parameters_inputs_logits_cotangent_finite_fp32": True,
                          "canonical_order_matches_step_weights": True}
    normalized_report = {**source_report,
        "arguments": {**source_report["arguments"], "case": "ce"},
        "settings": {**settings, "autocast": False},
        "model_config": copy.deepcopy(saved["model_config"]),
        "fixture": {**fixture, "answer_count": fixture["answers"], "used_batch_sha256": fixture["sha256"]}}
    normalized_saved = {"tokens": saved["tokens"], "labels": saved["labels"],
                        "gradients": gradients, "steps": steps}
    audit = {"adapter": "r3-mixed-validation-v1 FP32 arms to retained CE reference schema",
             "source_arms": {"naive": "naive_fp32", "tiled": "tiled_fp32"},
             "source_autocast_description": settings.get("autocast"),
             "reference_autocast": False,
             "reference_autocast_basis": "The mixed validator's explicit *_fp32 arms plus verified saved FP32 logits, cotangents and all gradients; BF16 arms excluded.",
             "fixture_aliases": {"answer_count": "answers", "used_batch_sha256": "sha256"},
             "source_fixture": fixture, "arm_contracts": contracts}
    return normalized_report, normalized_saved, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--mixed-reference", action="store_true",
                        help="Read tensors.pt from a completed mixed audit; use its naive_fp32 and tiled_fp32 arms only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve retained diagnostics; select a new FP64 output directory")
    hardware = require_cuda_container()
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": "r3-naive-fp64-ce-reference-v1", "evidence_class": "NUM",
              "status": "running", "arguments": {key: str(value) for key, value in vars(args).items()},
              "scope": "Naive FP64 reference only; identical pre-step FP32-representable weights and saved CE fixture. No optimizer step, no research continuation, no tiled FP64 claim."}
    try:
        source_report_path = args.reference_run / "report.json"
        source_tensor_path = args.reference_run / ("tensors.pt" if args.mixed_reference else "ce-and-update-tensors.pt")
        source_report = json.loads(source_report_path.read_text())
        saved = torch.load(source_tensor_path, map_location="cpu", weights_only=False)
        if args.mixed_reference:
            source_report, saved, adapter_audit = adapt_mixed_reference(source_report, saved)
            report["mixed_reference_adapter"] = adapter_audit
        if source_report["arguments"]["case"] != "ce" or source_report["status"] != "diagnostics_complete":
            raise ValueError("Reference run must be a completed actual-CE diagnostic")
        if source_report["settings"]["parameters"] != "FP32" or source_report["settings"]["autocast"]:
            raise ValueError("Reference run must use FP32 weights and disabled autocast")
        initial_weights, reconstruction = reconstruct_initial_weights(saved)
        tokens, labels = saved["tokens"], saved["labels"]
        if tokens.dtype != torch.int64 or labels.dtype != torch.int64 or tokens.shape != labels.shape:
            raise AssertionError("Expected saved equally shaped int64 tokens/labels")
        if not bool((labels != -100).any().item()):
            raise AssertionError("The saved actual CE fixture has no scored answers")
        if list(tokens.shape) != source_report["fixture"]["shape"]:
            raise AssertionError("Saved CE tensor shape differs from the reference report")
        if int((labels != -100).sum().item()) != source_report["fixture"]["answer_count"]:
            raise AssertionError("Saved CE answer count differs from the reference report")
        from cdrm.synthetic.common import SyntheticBatch
        fixture = SyntheticBatch(tokens.numpy(), labels.numpy(), [{}] * len(tokens))
        if fixture.sha256 != source_report["fixture"]["used_batch_sha256"]:
            raise AssertionError("Saved token/label arrays differ from the original CE fixture digest")
        # Source-identical model definitions matter when changing only precision.
        checked_model_sources = {}
        for name, expected in source_report["provenance"].get("source_sha256", {}).items():
            if name.startswith("recurrent-transformer/olmo/"):
                actual = digest_file(Path(name))
                if actual != expected:
                    raise AssertionError(f"Model source changed since FP32 reference: {name}")
                checked_model_sources[name] = actual
        seed_all(937, deterministic=True)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        from olmo.config import ModelConfig
        from olmo.model import OLMo
        cfg = copy.deepcopy(source_report["model_config"])
        if cfg["recurrent_layers"] != [3] or cfg["recurrent_write_rho"] != 1.0:
            raise ValueError("Reference configuration must be R3 at rho=1")
        if cfg["layer_norm_type"] != "default":
            raise ValueError("This FP64 fallback has only audited dtype-preserving default layer norm")
        cfg.update(init_device="cpu", recurrent_backend="naive", reference_eager=True)
        if args.mixed_reference:
            # Match build(..., backend, policy="legacy", bf16=False) used for
            # both saved FP32 arms; do not inherit the opt-in BF16 state policy.
            cfg.update(precision=None, recurrent_precision_policy="legacy")
        model = OLMo(ModelConfig(**cfg)).double().cuda().train()
        model.load_state_dict(initial_weights, strict=True)
        unique_parameters(model)
        if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
            raise AssertionError("Every oracle parameter must be FP64")
        expected_names = list(initial_weights)
        actual_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if actual_names != expected_names:
            raise AssertionError("Oracle's intended parameter names/order differ from retained weights")
        for name, parameter in model.named_parameters():
            if not torch.equal(parameter.detach().cpu(), initial_weights[name]):
                raise AssertionError(f"Oracle initial value differs: {name}")
        report.update(provenance=provenance(hardware), reconstruction=reconstruction,
            model_config=dataclasses.asdict(model.config),
            source_reference={"report": str(source_report_path), "report_sha256": digest_file(source_report_path),
                "tensors": str(source_tensor_path), "tensors_sha256": digest_file(source_tensor_path),
                "fixture": source_report["fixture"], "checkpoint": source_report.get("checkpoint"),
                "model_source_hashes_verified": checked_model_sources},
            settings={"parameters": "FP64", "computation": "FP64 naive operations; existing fixed ALiBi/mask constants retain their source construction precision",
                "autocast": False, "tf32": False, "deterministic": True,
                "sdpa": "explicit math", "recurrent_backend": "naive", "helper_compilation": False,
                "whole_model_compilation": False, "cuda_graphs": False,
                "loss": "same aligned_ce_sum(logits, saved_labels) / total scored answers; no shift",
                "input_token_dtype": str(tokens.dtype), "answer_count": int((labels != -100).sum().item())})
        tracked_sources = [Path(__file__), Path(__file__).with_name("r3_backward_validate.py"),
                           Path(__file__).with_name("r3_validation_metrics.py"), Path(__file__).with_name("stage_b_train.py")]
        report["diagnostic_source_sha256"] = {str(path): digest_file(path) for path in tracked_sources}
        print(f"Naive FP64 CE: {list(tokens.shape)}, D{model.config.d_model}, {len(actual_names)} trainable parameter tensors", flush=True)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            packet = backward_packet(model, forward_graph(model, tokens.cuda()), labels=labels.cuda())
        if packet["logits"].dtype != torch.float64 or packet["cotangent"].dtype != torch.float64:
            raise AssertionError("Oracle logits and CE cotangent must be FP64")
        if any(value is None or value.dtype != torch.float64 or not torch.isfinite(value).all()
               for value in tensors(packet).values()):
            raise AssertionError("Every intended oracle gradient must be present, finite and FP64")
        if packet["parameter_order"] != expected_names:
            raise AssertionError("Oracle parameter-gradient order changed")
        torch.save({"format": "r3-fp64-ce-oracle-tensors-v1", "oracle": packet,
                    "tokens": tokens, "labels": labels, "initial_fp64_weights": initial_weights,
                    "initial_fp32_weights": {name: value.float() for name, value in initial_weights.items()},
                    "source_fp32_tensors_path": str(source_tensor_path),
                    "source_fp32_tensors_sha256": digest_file(source_tensor_path)},
                   args.output_dir / "fp64-ce-tensors.pt")
        report["comparisons"] = {}
        for backend in ("naive", "tiled"):
            fp32 = saved["gradients"][backend]
            if fp32["parameter_order"] != expected_names:
                raise AssertionError(f"{backend}: saved gradient order differs from the oracle")
            report["comparisons"][backend] = {
                "reference": "naive_fp64", "actual": backend + "_fp32",
                "logits": compare_tensors(packet["logits"], fp32["logits"]),
                "loss": compare_tensors(torch.tensor(packet["loss"], dtype=torch.float64),
                                        torch.tensor(fp32["loss"], dtype=torch.float64)),
                "ce_cotangent": compare_tensors(packet["cotangent"], fp32["cotangent"]),
                "gradients": compare_maps(tensors(packet), tensors(fp32))}
        counters = {str(group): {str(key): int(value) for key, value in entries.items()}
                    for group, entries in torch._dynamo.utils.counters.items()}
        if counters.get("stats", {}).get("unique_graphs", 0):
            raise AssertionError("Unexpected compiled graph in the naive-only FP64 fallback")
        report["compiler_counters"] = counters
        report["oracle_loss"] = packet["loss"]
        report["oracle_cotangent_l2"] = float(packet["cotangent"].double().norm().item())
        report["parameter_tensor_count"] = len(packet["parameters"])
        report["gradient_tensor_count_including_inputs"] = len(tensors(packet))
        report["tensor_artifact"] = {"path": str(args.output_dir / "fp64-ce-tensors.pt"),
                                    "sha256": digest_file(args.output_dir / "fp64-ce-tensors.pt")}
        report["status"] = "diagnostics_complete"
    except BaseException as error:
        report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
