#!/usr/bin/env python3
"""Three-arm CDRM numerical validation on a frozen native MAD minibatch.

Full-model actual-CE gradients and a separate unscaled side probe are retained.
The latter uses identical independent preview leaves and one saved cotangent,
so the ordinary preview path and small bridge gain cannot hide scan errors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiment_tracking import OnlineTracker, add_wandb_arguments
from r3_backward_validate import step_packet
from r3_validation_metrics import compare_tensors
from stage_a_common import rng_state, restore_rng

EPS, ATOL, RTOL = 2**-7, 2e-6, 2e-5
ARMS = (("naive_fp32", "naive", "fp32"),
        ("tiled_fp32", "tiled", "fp32"),
        ("tiled_bf16", "tiled", "bf16_fp32_state"))


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(item) for item in value)
    return copy.deepcopy(value)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def gradients(packet):
    return {**packet["parameters"], **{f"input/{name}": value for name, value in packet["inputs"].items()}}


def assert_gradients(packet):
    for name, value in gradients(packet).items():
        if value is None or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"Missing/nonfinite/non-FP32 intended gradient: {name}")


def metric(reference, actual):
    row = compare_tensors(reference, actual)
    if not row["finite"]:
        row.update(prospective_l2_pass=False, prospective_maximum_pass=False)
        return row
    r, a = reference.double(), actual.double()
    floor = ATOL + RTOL * r.abs()
    row.update(fp32_floor_l2=float(floor.norm()), fp32_floor_max=float(floor.max()),
               prospective_l2_pass=row["error_l2"] <= 4 * EPS * row["ref_l2"] + float(floor.norm()),
               prospective_maximum_pass=row["max_abs_error"] <= 8 * EPS * row["ref_max_abs"] + float(floor.max()),
               maximum_over_reference_maximum=row["max_abs_error"] / max(row["ref_max_abs"], 1e-12),
               mean_signed_error=float((a-r).mean()))
    near = r.abs() <= 2 * EPS * row["ref_rms"] + ATOL
    for name, mask in (("adam_near_zero", near), ("outside_adam_near_zero", ~near)):
        error = a-r
        row[name] = {"count": int(mask.sum()), "error_energy": float(error[mask].square().sum()),
                     "strict_sign_flip_count": int(((r*a < 0) & mask).sum()),
                     "sign_change_including_zero_count": int(((r.sign() != a.sign()) & mask).sum())}
    return row


def comparison(reference, actual, parameter_names):
    if list(reference) != list(actual):
        raise AssertionError("Gradient coverage/order differs")
    rows = {name: metric(reference[name], actual[name]) for name in reference}
    selected = [rows[name] for name in parameter_names]
    if not all(row["finite"] for row in rows.values()):
        return {"rows": rows, "pass": False, "invalid_tensors": [name for name, row in rows.items() if not row["finite"]]}
    rnorm = math.sqrt(sum(row["ref_l2"]**2 for row in selected))
    enorm = math.sqrt(sum(row["error_l2"]**2 for row in selected))
    fnorm = math.sqrt(sum(row["fp32_floor_l2"]**2 for row in selected))
    l2_failed = [name for name, row in rows.items() if not row["prospective_l2_pass"]]
    maximum_failed = [name for name, row in rows.items() if not row["prospective_maximum_pass"]]
    global_pass = enorm <= 2 * EPS * rnorm + fnorm
    return {"rows": rows, "global_parameter_relative_l2": enorm/max(rnorm, 1e-12),
            "global_parameter_reference_l2": rnorm, "global_parameter_error_l2": enorm,
            "global_parameter_fp32_floor_l2": fnorm, "global_parameter_l2_pass": global_pass,
            "parameter_tensor_count": len(parameter_names), "total_tensor_count": len(rows),
            "per_tensor_l2_failures": l2_failed, "per_tensor_maximum_failures": maximum_failed,
            "fp32_elementwise_failures": [name for name, row in rows.items() if not row["elementwise_pass"]],
            "fp32_scale_aware_diagnostic_failures": [name for name, row in rows.items() if not row["scale_aware_pass"]],
            "pass": global_pass and not l2_failed and not maximum_failed}


def full_packet(model, tokens, labels, bf16, loss_sum):
    model.zero_grad(set_to_none=True)
    # Ordinary embedding lookup is explicit solely to retain its derivative;
    # the standard input_embeddings path applies the same remaining operations.
    embeddings = model.transformer.wte(tokens)
    embeddings.retain_grad()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(None, input_embeddings=embeddings, output_cdrm_states=True)
        logits, states = output.logits, output.cdrm_states
        summed, count = loss_sum(logits, labels)
        loss = summed/count
    logits.retain_grad()
    for name in ("p3", "p8", "hat_m"):
        states[name].retain_grad()
    loss.backward()
    parameter_names = dict(model.named_parameters())
    packet = {"parameters": {name: cpu(value.grad) for name, value in parameter_names.items()},
              "inputs": {"embedding_output": cpu(embeddings.grad),
                         **{name: cpu(states[name].grad) for name in ("p3", "p8", "hat_m")}},
              "input_gradient_scope": "Actual full-model intermediates; independent side leaves are in side_packet, not inferred here.",
              "logits": cpu(logits), "cotangent": cpu(logits.grad), "loss": loss.item(),
              "states": {name: cpu(states[name]) for name in
                         ("p3", "p8", "candidate", "hat_m", "m", "v8", "deep_correction", "bridge_correction",
                          "query", "temporary_k", "temporary_v")},
              "native_targets": count}
    assert_gradients(packet)
    return packet


def side_packet(model, fixture, bf16, scale_check):
    early, late = (value.cuda().detach().requires_grad_() for value in (fixture["p3"], fixture["p8"]))
    bias, cotangent = fixture["attention_bias"].cuda(), fixture["cotangent"].cuda()
    owner = model.transformer.blocks[model.config.cdrm_early_layer]
    parameters = {f"owner/{name}": value for name, value in owner.named_parameters()}
    parameters["cdrm.deep_adapter.weight"] = model.cdrm.deep_adapter.weight
    targets = {**parameters, "input/p3": early, "input/p8": late}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        _, states = model.cdrm(early, late, owner, bias, output_states=True)
    values = torch.autograd.grad(states["hat_m"], tuple(targets.values()), cotangent, retain_graph=scale_check)
    saved = dict(zip(targets, map(cpu, values)))
    packet = {"parameters": {name: saved[name] for name in parameters},
              "inputs": {name: saved[f"input/{name}"] for name in ("p3", "p8")},
              "hat_m": cpu(states["hat_m"]), "candidate": cpu(states["candidate"]),
              "bridge_unscaled": cpu(states["bridge_correction"] / model.config.cdrm_lambda),
              "cotangent": cpu(cotangent), "loss": None,
              "unused_parameters": ["cdrm.bridge_adapter.weight"],
              "scope": "Identical independent preview leaves and fixed unnormalized direct-hat_m cotangent; no bridge lambda in this derivative objective.",
              "scaling": {}}
    assert_gradients(packet)
    if scale_check:
        for scale in (1/32, 32.0):
            scaled = torch.autograd.grad(states["hat_m"], tuple(targets.values()), cotangent*scale, retain_graph=True)
            packet["scaling"][str(scale)] = {
                name: {"bitwise_normalized_equal": torch.equal(cpu(value).double()/scale, saved[name].double()),
                       "maximum_normalized_error": float((cpu(value).double()/scale-saved[name].double()).abs().max())}
                for name, value in zip(targets, scaled)}
    return packet


def adam_comparison(reference, actual, ref_grads, actual_grads, initialized, lr):
    rows = {}; ref_energy = actual_energy = error_energy = dot = 0.0
    buckets = {name: {"count": 0, "delta_error_energy": 0.0, "gradient_sign_flips": 0,
                      "delta_sign_flips": 0} for name in ("near_zero", "outside_near_zero")}
    for name in reference["deltas"]:
        r, a = reference["deltas"][name].double(), actual["deltas"][name].double()
        g, ga = ref_grads[name].double(), actual_grads[name].double()
        row = metric(r, a); rows[name] = row
        ref_energy += float(r.square().sum()); actual_energy += float(a.square().sum())
        error_energy += float((a-r).square().sum()); dot += float((a*r).sum())
        near = g.abs() <= 2*EPS*float(g.square().mean().sqrt())+ATOL
        row["gradient_near_zero_classification"] = {}
        for bucket, mask in (("near_zero", near), ("outside_near_zero", ~near)):
            values = {"count": int(mask.sum()), "delta_error_energy": float((a-r)[mask].square().sum()),
                      "gradient_sign_flips": int(((g*ga < 0)&mask).sum()),
                      "delta_sign_flips": int(((r*a < 0)&mask).sum())}
            row["gradient_near_zero_classification"][bucket] = values
            for key, value in values.items():
                buckets[bucket][key] += value
    relative = math.sqrt(error_energy)/max(math.sqrt(ref_energy),1e-12)
    cosine = dot/math.sqrt(ref_energy*actual_energy) if ref_energy*actual_energy>1e-24 else None
    passed = cosine is not None and cosine >= .99 if initialized else relative <= 2*EPS
    for row in buckets.values():
        row["fraction_of_delta_error_energy"] = row["delta_error_energy"]/error_energy if error_energy else 0.0
    return {"rows": rows, "global_delta_relative_l2": relative, "global_delta_cosine": cosine,
            "maximum_error_over_lr": max(row["max_abs_error"] for row in rows.values())/lr,
            "gradient_near_zero_buckets": buckets, "guardrail_pass": passed,
            "guardrail": "initial delta cosine >=.99" if initialized else "trained delta relative L2 <=.015625",
            "clipping": {name: {key: packet[key] for key in ("clip_norm", "clip_coefficient")}
                         for name, packet in (("reference", reference), ("actual", actual))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", default="selective-copying", choices=["selective-copying"])
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--example-offset", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--preset", type=Path, default=Path("configs/cdrm/base_d128_5.json"))
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side-cotangent-seed", type=int, default=724193)
    parser.add_argument("--scale-check", action="store_true")
    parser.add_argument("--eager", action="store_true", help="Diagnostic helper bypass only; omit for actual compiled confirmation")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    for archived in ("cdrm-naive", "r3-backward", "r3-bf16", "r3-bf16-tiled-resolution", "stage-a", "stage-b"):
        if args.output_dir.resolve().is_relative_to((Path(".runtime")/archived).resolve()):
            raise ValueError("Prior experiment lineages are read-only; choose the new CDRM tiled lineage")
    if args.batch < 1 or args.example_offset < 0:
        parser.error("Positive batch and nonnegative example offset required")
    if not args.wandb_project:
        parser.error("--wandb-project is required for graphable numerical runs")
    from cdrm_tiled_common import setup, build_model, optimizer_for, loss_sum, sources, validate_data, snapshot_sources
    report = {"schema": "cdrm-tiled-numerical-v1", "evidence": "NUM", "status": "running",
              "numerical_clearance": False,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    tracker = None; started = time.monotonic()
    args.output_dir.mkdir(parents=True)
    try:
        report["runtime"] = setup(args.seed)
        tracked_extras = (Path(__file__).resolve().relative_to(Path.cwd()),
                          Path("scripts/r3_backward_validate.py"))
        source_identity = sources(tracked_extras); report["source_sha256"] = source_identity
        snapshot_sources(args.output_dir, source_identity)
        report["source_snapshot"] = str(args.output_dir/"source")
        report["validator_sha256"] = digest(__file__)
        report["criteria"] = {"path": str(args.criteria), "sha256": digest(args.criteria)}
        (args.output_dir/"validation-contract.md").write_bytes(args.criteria.read_bytes())
        if digest(args.output_dir/"validation-contract.md") != report["criteria"]["sha256"]:
            raise RuntimeError("Criteria changed while preserving their exact contents")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        report["checkpoint"] = {"path": str(args.checkpoint), "sha256": digest(args.checkpoint),
                                "completed_updates": checkpoint.get("completed_updates", 0)}
        from cdrm.mad_data import load_dataset
        dataset = load_dataset(args.data_root, args.task, args.split)
        validate_data(dataset)
        data = dataset.take(slice(args.example_offset,args.example_offset+args.batch))
        if len(data) != args.batch:
            raise ValueError("Requested full physical minibatch is unavailable")
        tokens, labels = [torch.as_tensor(value, device="cuda") for value in (data.input_ids,data.labels)]
        report["fixture"] = {"sha256": data.sha256, "dataset_sha256": dataset.sha256,
                             "manifest": data.manifest, "shape": list(tokens.shape),
                             "native_targets": int((labels != -100).sum()),
                             "answer_targets": int((data.answer_labels != -100).sum()),
                             "example_offset": args.example_offset, "split": args.split,
                             "loss_semantics": "Native MAD labels already aligned; sum CE / all native scored targets; no additional shift",
                             "accumulation": False}
        if list(tokens.shape)[1] != 256 or data.manifest["vocab_size"] != 16:
            raise ValueError("This milestone expects native MAD V16/T256")
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name,
                                output_dir=args.output_dir, preserve_state=preserve_rng)
        tracker.start({"evidence": "NUM", "checkpoint": report["checkpoint"], "fixture": report["fixture"],
                       "criteria": report["criteria"], "source": source_identity})
        report["wandb"] = tracker.record
        print(json.dumps({"wandb_run_url": tracker.record["run_url"]}), flush=True)
        packets, side_packets, steps = {}, {}, {}
        fixed_side = None
        for arm, backend, precision in ARMS:
            print(f"{arm}: B{args.batch}/T256 actual native CE and independent side probe", flush=True)
            net, construction = build_model(backend, precision, args.seed, checkpoint=checkpoint,
                                             eager=args.eager if backend == "tiled" else True,
                                             preset=args.preset)
            if (net.config.n_layers != 5 or net.config.d_model != 128 or net.config.n_heads != 16
                    or net.config.mlp_hidden_size != 512
                    or net.config.cdrm_early_layer != 1 or net.config.cdrm_late_layer != 3
                    or net.config.cdrm_rho != 1 or net.config.cdrm_lambda == 0):
                raise ValueError("Expected active five-block rho-one CDRM with sites 1/3")
            opt = optimizer_for(net)
            opt.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
            if any(value.dtype != torch.float32 for value in net.parameters()):
                raise AssertionError("Master parameters must stay FP32")
            report.setdefault("construction", {})[arm] = construction
            packets[arm] = full_packet(net, tokens, labels, arm == "tiled_bf16", loss_sum)
            if fixed_side is None:
                from olmo.model import causal_attention_bias
                length = tokens.shape[1]
                bias = (net.get_alibi_attention_bias(length, tokens.device)[:, :, :length, :length]
                        + causal_attention_bias(length, tokens.device))
                fixed_side = {"p3": packets[arm]["states"]["p3"], "p8": packets[arm]["states"]["p8"],
                              "attention_bias": cpu(bias),
                              "cotangent": torch.randn(packets[arm]["states"]["hat_m"].shape,
                                                       generator=torch.Generator().manual_seed(args.side_cotangent_seed)),
                              "source_arm": arm, "cotangent_seed": args.side_cotangent_seed}
            side_packets[arm] = side_packet(net, fixed_side, arm == "tiled_bf16", args.scale_check)
            for name, parameter in net.named_parameters():
                if not torch.equal(cpu(parameter.grad), packets[arm]["parameters"][name]):
                    raise AssertionError(f"Side autograd.grad mutated the actual-CE parameter gradient: {name}")
            steps[arm] = step_packet(net, opt, 1.0)
            for group in ("weights", "clipped_gradients"):
                for name, value in steps[arm][group].items():
                    if value is None or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                        raise AssertionError(f"Post-step {group}/{name} must remain present finite FP32")
            for state in opt.state.values():
                if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                       for value in state.values() if torch.is_tensor(value)):
                    raise AssertionError("Adam state must stay finite FP32")
            tracker.log({f"ce/{arm}": packets[arm]["loss"], f"clip_norm/{arm}": steps[arm]["clip_norm"]})
            del net, opt
        artifact = {"tokens": cpu(tokens), "labels": cpu(labels), "answer_labels": torch.as_tensor(data.answer_labels),
                    "packets": packets, "side_packets": side_packets, "steps": steps, "fixed_side_fixture": fixed_side,
                    "model_config": checkpoint["model_config"]}
        torch.save(artifact,args.output_dir/"tensors.pt")
        report["tensor_artifact"] = {"path": str(args.output_dir/"tensors.pt"), "sha256": digest(args.output_dir/"tensors.pt")}
        report["comparisons"] = {}
        for actual, reference in (("tiled_fp32", "naive_fp32"), ("tiled_bf16", "tiled_fp32")):
            full = comparison(gradients(packets[reference]), gradients(packets[actual]), list(packets[reference]["parameters"]))
            local = comparison(gradients(side_packets[reference]), gradients(side_packets[actual]), list(side_packets[reference]["parameters"]))
            logits = metric(packets[reference]["logits"], packets[actual]["logits"])
            logits["global_logit_l2_pass"] = logits["error_l2"] <= 2*EPS*logits["ref_l2"]+logits["fp32_floor_l2"]
            state_comparisons = {name: metric(side_packets[reference][name], side_packets[actual][name])
                                 for name in ("hat_m", "candidate", "bridge_unscaled")}
            gain = float(checkpoint["model_config"]["cdrm_lambda"])
            full_states = {name: metric(packets[reference]["states"][name], packets[actual]["states"][name])
                           for name in ("hat_m", "candidate")}
            full_states["bridge_unscaled"] = metric(packets[reference]["states"]["bridge_correction"]/gain,
                                                      packets[actual]["states"]["bridge_correction"]/gain)
            ce_error = abs(packets[actual]["loss"]-packets[reference]["loss"])
            fp32_ce_pass = ce_error <= ATOL + RTOL * abs(packets[reference]["loss"])
            step = adam_comparison(steps[reference], steps[actual], packets[reference]["parameters"], packets[actual]["parameters"],
                                  report["checkpoint"]["completed_updates"] == 0,
                                  float(checkpoint["optimizer"]["param_groups"][0]["lr"]))
            fp32 = actual == "tiled_fp32"
            passed = ((not full["fp32_elementwise_failures"] and not local["fp32_elementwise_failures"]
                       and logits["elementwise_pass"] and fp32_ce_pass
                       and all(row["elementwise_pass"] for row in (*state_comparisons.values(), *full_states.values())))
                      if fp32 else (full["pass"] and local["pass"] and logits["global_logit_l2_pass"]
                                    and ce_error <= .01 and step["guardrail_pass"]
                                    and all(row["prospective_l2_pass"] and row["prospective_maximum_pass"]
                                            for row in (*state_comparisons.values(), *full_states.values()))))
            label = f"{actual}_vs_{reference}"
            report["comparisons"][label] = {"actual_ce_gradients": full, "independent_side_gradients": local,
                "logits": logits, "absolute_ce_difference": ce_error, "unscaled_side_states": state_comparisons,
                "actual_forward_unscaled_side_states": full_states, "fp32_ce_elementwise_pass": fp32_ce_pass,
                "adam": step, "machine_screens_pass": passed}
            tracker.summary({f"{label}/full_gradient_relative_l2": full["global_parameter_relative_l2"],
                             f"{label}/side_gradient_relative_l2": local["global_parameter_relative_l2"],
                             f"{label}/hat_m_relative_l2": state_comparisons["hat_m"]["rel_l2"],
                             f"{label}/actual_forward_hat_m_relative_l2": full_states["hat_m"]["rel_l2"],
                             f"{label}/ce_difference": ce_error, f"{label}/machine_screens_pass": passed})
            for index, (name, row) in enumerate(full["rows"].items()):
                tracker.log({"tensor_index": index, f"{label}/gradient_relative_l2": row["rel_l2"],
                             f"{label}/gradient_maximum_over_reference_maximum": row["maximum_over_reference_maximum"]})
        report["scaling"] = {arm: value["scaling"] for arm, value in side_packets.items()}
        scaling_pass = all(row["bitwise_normalized_equal"] for arm in report["scaling"].values()
                           for scale in arm.values() for row in scale.values())
        report["scaling_requested"] = args.scale_check
        report["scaling_pass"] = scaling_pass if args.scale_check else None
        report["machine_screens_pass"] = all(row["machine_screens_pass"] for row in report["comparisons"].values()) and scaling_pass
        report["disposition"] = "Numerical fixture screens only; semantic tests, operational/recovery evidence and cost are separate."
        if sources(tracked_extras) != source_identity:
            raise RuntimeError("Source changed during numerical validation")
        if digest(args.criteria) != report["criteria"]["sha256"]:
            raise RuntimeError("Criteria changed during numerical validation")
        report["status"] = "diagnostics_complete"
    except BaseException as error:
        report.update(status="execution_failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        try:
            if tracker:
                tracker.finish(succeeded=report["status"] == "diagnostics_complete")
        except BaseException as error:
            if report.get("error_type"):
                report["prior_execution_error"] = {"error_type": report["error_type"], "error": report["error"]}
            report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            save_json(args.output_dir/"report.json",report)
    print(json.dumps({"status": report["status"], "output": str(args.output_dir),
                      "machine_screens_pass": report["machine_screens_pass"]}), flush=True)


if __name__ == "__main__":
    main()
