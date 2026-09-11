#!/usr/bin/env python3
"""Frozen-policy, fresh-fixture CDRM numerical confirmation; never training.

Fixture and initialization generation are intentionally separate. Both input
manifests require externally supplied SHA256 anchors before any CUDA work.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm_precision_probe import ARMS, intervention
from cdrm_tiled_common import (ARM_FIELDS, FORMAT, INIT_FORMAT, build_model, config,
                               json_digest, loss_sum, optimizer_for, setup,
                               snapshot_sources, sources, validate_data, verify_sources)
from cdrm_tiled_validate import (ATOL, RTOL, EPS, adam_comparison, comparison, cpu,
                                 digest, full_packet, gradients, metric, preserve_rng,
                                 save_json, side_packet)
from experiment_tracking import OnlineTracker, add_wandb_arguments
from r3_backward_validate import step_packet
from stage_a_common import configure_compiled_helpers
from stage_b_train import compiler_audit

LINEAGE = Path(".runtime/cdrm-numerical-resolution/20260907T232931Z")
OLD_CONTRACT = "26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20"
LOCAL_CONTRACT = "6660b964a2f5a678d4324ac2799c814ad617ce9acceabbed5e0b0e23101714cc"
ROLE_SPECS = {
    "fp32_trained_u1000": (0, "ef505d31586003b60b1252bd315d605ee241b11bcde21697c1f17c1db91bb4cf", 7500, 1000, "fp32"),
    "bf16_trained_u1000": (64, "66fcce189407151a194503502aeeb20b535b00b97a682125888eb6b41640aadf", 7500, 1000, "bf16_fp32_state"),
    "distinct_initialization": (128, None, 7502, 0, None),
}


def json_value(value):
    return json.loads(json.dumps(value))


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Immutable JSON SHA mismatch: {path}")
    return json.loads(Path(path).read_text())


def confirmation_sources():
    return sources(("scripts/cdrm_precision_confirm.py", "tests/test_cdrm_precision_confirm.py",
                    "scripts/cdrm_precision_probe.py", "scripts/cdrm_tiled_validate.py",
                    "scripts/r3_backward_validate.py", "docs/reports/cdrm-numerical-resolution/reference-contract.md"))


def validate_roles(roles):
    expected = {"schema": "cdrm-numerical-resolution-future-roles-v1", "prospective_dataset_seed": 925903,
                "prospective_examples": 192, "physical_batch": 64, "sequence_length": 256,
                "copy_tokens": 96, "vocabulary_size": 16}
    if any(roles.get(key) != value for key, value in expected.items()):
        raise ValueError("Prospective role corpus dimensions or identity differ")
    rows = {row["name"]: row for row in roles["roles"]}
    if len(roles["roles"]) != 3 or set(rows) != set(ROLE_SPECS):
        raise ValueError("Expected exactly the three declared roles")
    for name, (offset, sha, seed, _, _) in ROLE_SPECS.items():
        row = rows[name]
        if row["example_offset"] != offset or row["example_stop"] != offset+64:
            raise ValueError("Prospective role offset changed")
        if sha is not None and row.get("checkpoint_sha256") != sha:
            raise ValueError("Prospective trained checkpoint changed")
        if sha is None and row.get("initialization_seed") != seed:
            raise ValueError("Prospective distinct initialization seed changed")
    return rows


def validate_decision(decision, roles):
    if decision.get("schema") != "cdrm-precision-candidate-freeze-v1" or decision.get("status") != "frozen_for_confirmation":
        raise ValueError("A selected, frozen candidate decision is required")
    name = decision["candidate"]["arm"]
    arm = ARMS.get(name)
    if arm is None or not arm.bf16 or name == "current_bf16" or arm.bypass or arm.lambda_zero or arm.reduction_off:
        raise ValueError("Candidate must preserve the complete active mixed CDRM path")
    if json_value(asdict(arm)) != decision["candidate"]["specification"]:
        raise ValueError("Candidate policy specification changed after selection")
    if decision["original_criteria"]["sha256"] != OLD_CONTRACT or decision["reference_contract"]["sha256"] != LOCAL_CONTRACT:
        raise ValueError("Numerical criteria changed")
    validate_roles(roles)
    return arm


def validate_fixture_roles(fixtures, decision_sha, roles_sha):
    if (fixtures.get("schema") != "cdrm-precision-fresh-fixtures-v1" or fixtures.get("status") != "generated"
            or fixtures.get("candidate_freeze_sha256") != decision_sha or fixtures.get("roles_sha256") != roles_sha):
        raise ValueError("Fresh fixtures must bind the prior candidate/role freeze")
    corpus = fixtures["corpus"]
    if corpus["seed"] != 925903 or corpus["examples"] != 192 or corpus["split"] != "dev":
        raise ValueError("Fresh corpus allocation differs")
    rows = {row["name"]: row for row in fixtures["roles"]}
    if len(fixtures["roles"]) != 3 or set(rows) != set(ROLE_SPECS):
        raise ValueError("Fresh fixtures must contain exactly three roles")
    for name, (offset, known_sha, _, _, _) in ROLE_SPECS.items():
        row = rows[name]
        if row["example_offset"] != offset or row["example_stop"] != offset+64:
            raise ValueError("Fresh fixture assigned to wrong role offset")
        if known_sha is not None and row["checkpoint"]["sha256"] != known_sha:
            raise ValueError("Fresh fixture assigned to wrong trained checkpoint")
    return rows


def expected_parameter_shapes():
    shapes = {"transformer.wte.weight": (16,128), "transformer.ln_f.weight": (128,)}
    for index in range(5):
        for name, shape in (("k_norm", (128,)), ("q_norm", (128,)), ("attn_out", (128,128)),
                            ("ff_out", (128,512)), ("att_proj", (384,128)), ("ff_proj", (512,128)),
                            ("attn_norm", (128,)), ("ff_norm", (128,))):
            shapes[f"transformer.blocks.{index}.{name}.weight"] = shape
    shapes.update({"transformer.ff_out.weight": (16,128), "cdrm.deep_adapter.weight": (128,128),
                   "cdrm.bridge_adapter.weight": (128,128)})
    return shapes


def validate_checkpoint(checkpoint, role):
    _, _, seed, updates, precision = ROLE_SPECS[role]
    if (checkpoint.get("format") != (INIT_FORMAT if updates == 0 else FORMAT)
            or checkpoint.get("completed_updates") != updates
            or checkpoint.get("initialization", {}).get("seed") != seed):
        raise ValueError("Checkpoint format, initialization seed or update does not match role")
    if updates and checkpoint.get("precision") != precision:
        raise ValueError("Trained checkpoint precision does not match role")
    if checkpoint.get("identity_sha256") != json_digest(checkpoint["identity"]):
        raise ValueError("Checkpoint identity digest mismatch")
    expected_config = asdict(config("tiled", "fp32"))
    if any(checkpoint["model_config"].get(key) != value for key,value in expected_config.items() if key not in ARM_FIELDS):
        raise ValueError("Checkpoint architecture/loss/precision context differs from the five-block scope")
    shapes = expected_parameter_shapes()
    if list(checkpoint["model"]) != list(shapes):
        raise ValueError("Expected the exact 45 canonical parameter names and optimizer order")
    for name, value in checkpoint["model"].items():
        if tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("Master parameter shape/dtype/finiteness failure")
    if updates == 0 and any(not bool(checkpoint["model"][name].count_nonzero()) for name in
                            ("cdrm.deep_adapter.weight", "cdrm.bridge_adapter.weight")):
        raise ValueError("Distinct initialization requires both nonzero adapters")
    optimizer = checkpoint["optimizer"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Expected one canonical AdamW parameter group")
    group = optimizer["param_groups"][0]
    if (len(group["params"]) != 45 or len(set(group["params"])) != 45
            or tuple(group["betas"]) != (.9,.98) or group["eps"] != 1e-8 or group["weight_decay"] != 0
            or not math.isfinite(group["lr"]) or group["lr"] <= 0
            or any(group.get(key, False) for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"))):
        raise ValueError("Canonical optimizer ownership/settings differ")
    if updates == 0 and group["lr"] != 5e-4:
        raise ValueError("Distinct initialization must preserve the original initial learning rate")
    if updates == 0 and optimizer["state"]:
        raise ValueError("Distinct initialization must have empty Adam state")
    if updates:
        if set(optimizer["state"]) != set(group["params"]):
            raise ValueError("Every trained canonical parameter requires Adam state")
        for name,pid in zip(shapes, group["params"]):
            state = optimizer["state"][pid]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != updates:
                raise ValueError("Adam state/step mismatch")
            if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError("Adam state must be finite FP32")
            if any(tuple(state[key].shape) != shapes[name] for key in ("exp_avg", "exp_avg_sq")) or not bool((state["exp_avg_sq"] >= 0).all()):
                raise ValueError("Adam moment shape/value mismatch")
    return {"canonical_parameter_tensors": 45, "completed_updates": updates, "initialization_seed": seed,
            "optimizer_lr": group["lr"], "initialization": updates == 0}


def validate_dataset(dataset, corpus, rows):
    validate_data(dataset)
    if (len(dataset) != 192 or dataset.manifest["seed"] != 925903 or dataset.sha256 != corpus["dataset_sha256"]
            or dataset.manifest["manifest_sha256"] != corpus["manifest_sha256"]):
        raise ValueError("Actual fresh corpus arrays/manifest differ")
    for row in rows.values():
        if dataset.take(slice(row["example_offset"], row["example_stop"])).sha256 != row["batch_sha256"]:
            raise ValueError("Actual fresh minibatch differs from frozen role")


def preflight(args):
    decision = checked_json(args.decision, args.decision_sha256)
    roles = checked_json(decision["roles"]["path"], decision["roles"]["sha256"])
    arm = validate_decision(decision, roles)
    for key in ("original_criteria", "reference_contract"):
        if digest(decision[key]["path"]) != decision[key]["sha256"]:
            raise ValueError("Frozen criteria content changed")
    tracked = decision["source_sha256"]
    required = confirmation_sources()
    if any(tracked.get(name) != value for name,value in required.items()):
        raise ValueError("Candidate freeze does not cover current confirmation dependencies")
    verify_sources(tracked)
    fixtures = checked_json(args.fixtures, args.fixtures_sha256)
    rows = validate_fixture_roles(fixtures, args.decision_sha256, decision["roles"]["sha256"])
    if args.role not in rows:
        raise ValueError("Undeclared confirmation role")
    from cdrm.mad_data import load_dataset
    corpus = fixtures["corpus"]
    dataset = load_dataset(corpus["root"], "selective-copying", "dev", verify=True)
    validate_dataset(dataset, corpus, rows)
    for row in rows.values():
        if digest(row["checkpoint"]["path"]) != row["checkpoint"]["sha256"]:
            raise ValueError("Fresh role checkpoint bytes changed")
    row = rows[args.role]
    checkpoint = torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=False)
    checkpoint_record = validate_checkpoint(checkpoint, args.role)
    verify_sources(checkpoint["identity"]["source_sha256"])
    batch = dataset.take(slice(row["example_offset"], row["example_stop"]))
    return decision, fixtures, arm, checkpoint, checkpoint_record, batch


def output_scaling_packet(model, tokens, bf16, fixed_cotangent, expected_logits):
    """One additional fixed forward, three VJPs; no parameter .grad writes."""
    embeddings = model.transformer.wte(tokens)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(None, input_embeddings=embeddings, output_cdrm_states=True)
    targets = {**dict(model.named_parameters()), "input/embedding_output": embeddings,
               **{f"input/{name}": output.cdrm_states[name] for name in ("p3", "p8", "hat_m")}}
    cotangent = fixed_cotangent.to(device=output.logits.device, dtype=output.logits.dtype)
    baseline = dict(zip(targets, map(cpu, torch.autograd.grad(output.logits, tuple(targets.values()), cotangent, retain_graph=True))))
    if len(baseline) != 49 or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in baseline.values()):
        raise ValueError("Fixed-output VJP must retain 45 finite FP32 parameter and four input gradients")
    rows = {}
    for index, scale in enumerate((1/32, 32.)):
        values = torch.autograd.grad(output.logits, tuple(targets.values()), cotangent*scale, retain_graph=index == 0)
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Scaled full-output VJP must remain finite FP32")
        rows[str(scale)] = {name: {"bitwise_normalized_equal": torch.equal(cpu(value).double()/scale, baseline[name].double()),
                                   "maximum_normalized_error": float((cpu(value).double()/scale-baseline[name].double()).abs().max())}
                            for name,value in zip(targets, values)}
    logits_equal = torch.equal(cpu(output.logits), expected_logits)
    return {"fixed_forward_logits_bitwise_equal": logits_equal,
            "scope": "Same graph and raw saved FP32-CE logit cotangent without additional normalization, cast once to the native output dtype; linear derivative probe, not a second CE loss.",
            "baseline_gradients": baseline, "cotangent": cpu(cotangent), "scaling": rows,
            "diagnostic_pass": logits_equal and all(row["bitwise_normalized_equal"] for scale in rows.values() for row in scale.values())}


def compare_pair(reference, actual, side_reference, side_actual, step_reference, step_actual, checkpoint, *, fp32):
    """The old validator's comparisons and thresholds, preserved verbatim in scope."""
    full = comparison(gradients(reference), gradients(actual), list(reference["parameters"]))
    local = comparison(gradients(side_reference), gradients(side_actual), list(side_reference["parameters"]))
    logits = metric(reference["logits"], actual["logits"])
    logits["global_logit_l2_pass"] = logits["error_l2"] <= 2*EPS*logits["ref_l2"]+logits["fp32_floor_l2"]
    states = {name: metric(side_reference[name], side_actual[name]) for name in ("hat_m", "candidate", "bridge_unscaled")}
    gain = checkpoint["model_config"]["cdrm_lambda"]
    full_states = {name: metric(reference["states"][name], actual["states"][name]) for name in ("hat_m", "candidate")}
    full_states["bridge_unscaled"] = metric(reference["states"]["bridge_correction"]/gain, actual["states"]["bridge_correction"]/gain)
    ce_error = abs(actual["loss"]-reference["loss"])
    fp32_ce_pass = ce_error <= ATOL+RTOL*abs(reference["loss"])
    adam = adam_comparison(step_reference, step_actual, reference["parameters"], actual["parameters"],
                           checkpoint["completed_updates"] == 0, checkpoint["optimizer"]["param_groups"][0]["lr"])
    passed = ((not full["fp32_elementwise_failures"] and not local["fp32_elementwise_failures"] and logits["elementwise_pass"] and fp32_ce_pass
               and all(row["elementwise_pass"] for row in (*states.values(), *full_states.values()))) if fp32 else
              (full["pass"] and local["pass"] and logits["global_logit_l2_pass"] and ce_error <= .01 and adam["guardrail_pass"]
               and all(row["prospective_l2_pass"] and row["prospective_maximum_pass"] for row in (*states.values(), *full_states.values()))))
    return {"actual_ce_gradients": full, "independent_side_gradients": local, "logits": logits,
            "absolute_ce_difference": ce_error, "fp32_ce_elementwise_pass": fp32_ce_pass,
            "unscaled_side_states": states, "actual_forward_unscaled_side_states": full_states,
            "adam": adam, "machine_screens_pass": passed}


def save_partial(path, artifact):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(artifact, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixtures-sha256", required=True)
    parser.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(LINEAGE.resolve()):
        raise ValueError("Use a new output directory in the numerical-resolution lineage")
    if not args.wandb_project:
        parser.error("Fresh numerical confirmation requires online W&B")
    decision, fixtures, candidate, checkpoint, ck_record, batch = preflight(args)
    fixture_role = next(row for row in fixtures["roles"] if row["name"] == args.role)
    report = {"schema": "cdrm-precision-confirmation-v1", "status": "running", "numerical_clearance": False,
              "scope": "Fresh numerical confirmation only; no task-performance testing or carried optimizer update",
              "role": args.role, "candidate": decision["candidate"], "checkpoint": {**fixture_role["checkpoint"], **ck_record},
              "decision": {"path": str(args.decision), "sha256": args.decision_sha256},
              "fixtures": {"path": str(args.fixtures), "sha256": args.fixtures_sha256},
              "criteria": decision["original_criteria"], "reference_contract": decision["reference_contract"],
              "source_sha256": decision["source_sha256"], "fixture": {"sha256": batch.sha256, "shape": [64,256],
                  "example_offset": ROLE_SPECS[args.role][0], "dataset_sha256": fixtures["corpus"]["dataset_sha256"],
                  "manifest": batch.manifest, "native_targets": 6144, "loss_semantics": "Native aligned MAD masked CE; no shift; sum/6144"}}
    artifact = {"tokens": torch.as_tensor(batch.input_ids), "labels": torch.as_tensor(batch.labels),
                "answer_labels": torch.as_tensor(batch.answer_labels), "model_config": checkpoint["model_config"],
                "packets": {}, "side_packets": {}, "steps": {}, "full_output_scaling": {}}
    tracker = None; started = time.monotonic()
    args.output_dir.mkdir(parents=True)
    try:
        snapshot_sources(args.output_dir, decision["source_sha256"])
        for name,path in (("candidate-decision.json",args.decision), ("fixture-manifest.json",args.fixtures),
                          ("original-criteria.md",Path(decision["original_criteria"]["path"])),
                          ("reference-contract.md",Path(decision["reference_contract"]["path"]))):
            (args.output_dir/name).write_bytes(path.read_bytes())
        report["runtime"] = setup(0)
        tokens, labels = artifact["tokens"].cuda(), artifact["labels"].cuda()
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir, preserve_state=preserve_rng)
        report["wandb"] = tracker.record
        tracker.start({"evidence": "NUM-fresh", "role": args.role, "candidate": decision["candidate"],
                       "decision_sha256": args.decision_sha256, "fixture": report["fixture"]})
        names = ("naive_fp32", "tiled_fp32", "current_bf16", "candidate")
        for name in names:
            print(f"Fresh {args.role}: {name}, B64/T256", flush=True)
            torch._dynamo.reset(); torch._dynamo.utils.counters.clear(); configure_compiled_helpers(True)
            mixed = name in ("current_bf16", "candidate")
            backend = "naive" if name == "naive_fp32" else "tiled"
            model, construction = build_model(backend, "bf16_fp32_state" if mixed else "fp32", checkpoint=checkpoint)
            optimizer = optimizer_for(model); optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
            report.setdefault("construction", {})[name] = construction
            policy = candidate if name == "candidate" else ARMS["current_bf16" if mixed else "current_fp32"]
            with intervention(model, policy):
                packet = full_packet(model, tokens, labels, mixed, loss_sum)
                artifact["packets"][name] = packet
                if list(packet["parameters"]) != list(expected_parameter_shapes()):
                    raise ValueError("Full actual-CE canonical gradient coverage differs")
                if "fixed_side_fixture" not in artifact:
                    from olmo.model import causal_attention_bias
                    bias = model.get_alibi_attention_bias(256,tokens.device)[:,:,:256,:256]+causal_attention_bias(256,tokens.device)
                    artifact["fixed_side_fixture"] = {"p3": packet["states"]["p3"], "p8": packet["states"]["p8"], "attention_bias": cpu(bias),
                        "cotangent": torch.randn(packet["states"]["hat_m"].shape, generator=torch.Generator().manual_seed(724193)),
                        "source_arm": name, "cotangent_seed": 724193}
                    artifact["fixed_output_cotangent"] = packet["cotangent"]
                artifact["side_packets"][name] = side_packet(model, artifact["fixed_side_fixture"], mixed, True)
                if any(not math.isfinite(row["maximum_normalized_error"]) for scale in artifact["side_packets"][name]["scaling"].values() for row in scale.values()):
                    raise ValueError("Scaled side VJP must remain finite")
                artifact["full_output_scaling"][name] = output_scaling_packet(model, tokens, mixed, artifact["fixed_output_cotangent"], packet["logits"])
                for pname,parameter in model.named_parameters():
                    if not torch.equal(cpu(parameter.grad), packet["parameters"][pname]):
                        raise ValueError("Derivative diagnostic mutated actual-CE parameter gradients")
            artifact["steps"][name] = step_packet(model, optimizer, 1.)
            for group in ("weights", "clipped_gradients"):
                if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in artifact["steps"][name][group].values()):
                    raise ValueError("Nonfinite/non-FP32 post-step weights or gradients")
            for state in artifact["steps"][name]["state"].values():
                if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in state.values() if torch.is_tensor(value)):
                    raise ValueError("Nonfinite/non-FP32 post-step Adam state")
            audit = compiler_audit(False, require_graphs=False)
            report.setdefault("compiler", {})[name] = {**audit, "require_graphs": backend != "naive", "validation_status": "pending"}
            save_partial(args.output_dir/"partial-tensors.pt", artifact)
            required_audit = compiler_audit(True, require_graphs=backend != "naive")
            report["compiler"][name] = {**required_audit, "require_graphs": backend != "naive", "validation_status": "passed"}
            tracker.log({f"ce/{name}": packet["loss"], f"clip_norm/{name}": artifact["steps"][name]["clip_norm"]})
            del model, optimizer
        (args.output_dir/"partial-tensors.pt").replace(args.output_dir/"tensors.pt")
        report["tensor_artifact"] = {"path": str(args.output_dir/"tensors.pt"), "sha256": digest(args.output_dir/"tensors.pt")}
        report["comparisons"] = {}
        for actual, reference in (("tiled_fp32","naive_fp32"), ("current_bf16","tiled_fp32"), ("candidate","tiled_fp32")):
            row = compare_pair(artifact["packets"][reference], artifact["packets"][actual], artifact["side_packets"][reference],
                               artifact["side_packets"][actual], artifact["steps"][reference], artifact["steps"][actual], checkpoint, fp32=actual == "tiled_fp32")
            report["comparisons"][f"{actual}_vs_{reference}"] = row
            tracker.summary({f"{actual}/full_gradient_relative_l2": row["actual_ce_gradients"]["global_parameter_relative_l2"],
                             f"{actual}/side_gradient_relative_l2": row["independent_side_gradients"]["global_parameter_relative_l2"],
                             f"{actual}/adam_relative_l2": row["adam"]["global_delta_relative_l2"], f"{actual}/machine_screens_pass": row["machine_screens_pass"]})
        report["side_scaling"] = {name: packet["scaling"] for name,packet in artifact["side_packets"].items()}
        per_arm_side_pass = {name: all(row["bitwise_normalized_equal"] for scale in arm.values() for row in scale.values()) for name,arm in report["side_scaling"].items()}
        side_pass = all(per_arm_side_pass.values())
        report["per_arm_side_scaling_pass"] = per_arm_side_pass
        report["side_scaling_pass"] = side_pass
        report["full_output_scaling"] = {name: {key:value for key,value in packet.items() if key not in ("baseline_gradients","cotangent")}
                                          for name,packet in artifact["full_output_scaling"].items()}
        report["candidate_machine_screens_pass"] = report["comparisons"]["candidate_vs_tiled_fp32"]["machine_screens_pass"] and per_arm_side_pass["candidate"] and per_arm_side_pass["tiled_fp32"]
        report["full_output_scaling_pass"] = all(value["diagnostic_pass"] for value in report["full_output_scaling"].values())
        report["candidate_additional_diagnostics_pass"] = all(report["full_output_scaling"][name]["diagnostic_pass"] for name in ("candidate","tiled_fp32"))
        report["candidate_prospective_checks_pass"] = report["candidate_machine_screens_pass"] and report["candidate_additional_diagnostics_pass"]
        report["machine_screens_pass"] = all(row["machine_screens_pass"] for row in report["comparisons"].values()) and side_pass
        report["disposition"] = "Raw original flags retained, including FP32 raw-side near-floor flags. Full-output scaling is an additional derivative diagnostic. No automatic precision-policy clearance."
        preflight(args)  # Verify the same sources, anchors, array files and checkpoint again.
        report["status"] = "diagnostics_complete"
    except BaseException as error:
        report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        if artifact["packets"] and not (args.output_dir/"tensors.pt").exists():
            save_partial(args.output_dir/"partial-tensors.pt", artifact)
            report["partial_tensor_artifact"] = {"path": str(args.output_dir/"partial-tensors.pt"), "sha256": digest(args.output_dir/"partial-tensors.pt")}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        try:
            if tracker:
                tracker.finish(succeeded=report["status"] == "diagnostics_complete")
        except BaseException as error:
            if report.get("error_type"):
                report["prior_execution_error"] = {key:report[key] for key in ("error_type","error")}
            report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            save_json(args.output_dir/"report.json",report)


if __name__ == "__main__":
    main()
