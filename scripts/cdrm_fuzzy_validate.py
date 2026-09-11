#!/usr/bin/env python3
"""Same-state numerical checks for the explicitly configured fuzzy-recall models.

No checkpoint or optimizer update is carried between arms. The numerical metrics
and budgets are the retained CDRM contract, with FP32 coordinate flags visible.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import cdrm_fuzzy_common as common
from cdrm.mad_data import load_dataset
from cdrm_precision_confirm import compare_pair
from cdrm_tiled_validate import (ATOL, RTOL, EPS, adam_comparison, assert_gradients,
                                 comparison, cpu, digest, full_packet, gradients,
                                 metric, preserve_rng, side_packet)
from experiment_tracking import OnlineTracker, add_wandb_arguments
from r3_backward_validate import step_packet
from stage_a_common import configure_compiled_helpers
from stage_b_train import atomic_json, compiler_audit

CONTRACT = Path("docs/reports/cdrm-tiled-bf16/validation-contract.md")
CONTRACT_SHA = "26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20"


def reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def save_packet(path, packet):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(packet, temporary)
    temporary.replace(path)
    return reference(path)


def seq_packet(model, tokens, labels, mixed):
    model.zero_grad(set_to_none=True)
    embeddings = model.transformer.wte(tokens)
    embeddings.retain_grad()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        logits = model(None, input_embeddings=embeddings).logits
        total, count = common.native_loss_sum(logits, labels)
        loss = total / count
    logits.retain_grad()
    loss.backward()
    packet = {"parameters": {name: cpu(value.grad) for name, value in model.named_parameters()},
              "inputs": {"embedding_output": cpu(embeddings.grad)},
              "logits": cpu(logits), "cotangent": cpu(logits.grad),
              "loss": loss.item(), "native_targets": count}
    assert_gradients(packet)
    return packet


def output_scale_check(model, tokens, mixed, cotangent, expected_logits):
    """Recompute one fixed graph and verify VJP scaling without writing .grad."""
    embeddings = model.transformer.wte(tokens)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        output = model(None, input_embeddings=embeddings, output_cdrm_states=model.config.cdrm_enabled)
    targets = {**dict(model.named_parameters()), "input/embedding_output": embeddings}
    if model.config.cdrm_enabled:
        targets.update({f"input/{name}": output.cdrm_states[name] for name in ("p3", "p8", "hat_m")})
    cot = cotangent.to(device=output.logits.device, dtype=output.logits.dtype)
    baseline = dict(zip(targets, map(cpu, torch.autograd.grad(output.logits, tuple(targets.values()), cot, retain_graph=True))))
    rows = {}
    for index, scale in enumerate((1/32, 32.)):
        values = torch.autograd.grad(output.logits, tuple(targets.values()), cot*scale, retain_graph=index == 0)
        rows[str(scale)] = {}
        for name, value in zip(targets, values):
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise AssertionError(f"Invalid scaled VJP: {name}")
            normalized = cpu(value).double()/scale
            rows[str(scale)][name] = {
                "bitwise_normalized_equal": torch.equal(normalized, baseline[name].double()),
                "maximum_normalized_error": float((normalized-baseline[name].double()).abs().max())}
    equal = torch.equal(cpu(output.logits), expected_logits)
    return {"baseline_gradients": baseline, "cotangent": cpu(cot), "scaling": rows,
            "fixed_forward_logits_bitwise_equal": equal,
            "pass": equal and all(row["bitwise_normalized_equal"] for group in rows.values() for row in group.values())}


def seq_compare(ref, actual, checkpoint):
    full = comparison(gradients(ref["full"]), gradients(actual["full"]), list(ref["full"]["parameters"]))
    logits = metric(ref["full"]["logits"], actual["full"]["logits"])
    logits["global_logit_l2_pass"] = logits["error_l2"] <= 2*EPS*logits["ref_l2"]+logits["fp32_floor_l2"]
    ce = abs(ref["full"]["loss"]-actual["full"]["loss"])
    adam = adam_comparison(ref["step"], actual["step"], ref["full"]["parameters"], actual["full"]["parameters"],
                           checkpoint["completed_updates"] == 0, checkpoint["optimizer"]["param_groups"][0]["lr"])
    return {"actual_ce_gradients": full, "logits": logits, "absolute_ce_difference": ce,
            "adam": adam, "machine_screens_pass": full["pass"] and logits["global_logit_l2_pass"] and ce <= .01 and adam["guardrail_pass"]}


def causality_check(model, tokens, mixed, baseline):
    boundary = tokens.shape[1]//2
    changed = tokens.clone()
    changed[:,boundary:] = (changed[:,boundary:]+1)%model.config.vocab_size
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        altered = model(changed).logits
    # Match the grad-enabled mode of the saved actual-CE forward. Sequence shape
    # and every causal prefix are fixed; later symbols may not affect this prefix.
    before,after = baseline[:,:boundary],cpu(altered[:,:boundary])
    return {"boundary":boundary,"changed_input_count":int((changed!=tokens).sum()),
            "prefix_logits_bitwise_equal":torch.equal(before,after),
            "prefix_maximum_error":float((before.float()-after.float()).abs().max())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("cdrm", "seq"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-naive", action="store_true")
    parser.add_argument("--scale-check", action="store_true")
    parser.add_argument("--causality-check", action="store_true")
    parser.add_argument("--side-seed", type=int, default=963101)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-150m-fuzzy-recall", wandb_group="20260908T031418Z")
    args = parser.parse_args(argv)
    if args.batch < 1 or args.offset < 0 or args.length < 4:
        parser.error("Require a positive batch, nonnegative offset and length >=4")
    if args.include_naive and args.arm != "cdrm":
        parser.error("Naive scan is a CDRM reference only")
    if not args.wandb_project:
        parser.error("Online W&B is required")
    return args


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new numerical case directory")
    if digest(CONTRACT) != CONTRACT_SHA:
        raise ValueError("Original numerical criteria changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = load_dataset(args.data_root, "fuzzy-in-context-recall", args.split, verify=True)
    common.validate_data(dataset,args.split,length=args.length)
    batch = dataset.take(slice(args.offset, args.offset+args.batch))
    if len(batch) != args.batch or batch.input_ids.shape != (args.batch, args.length):
        raise ValueError("Numerical fixture shape differs")
    args.output_dir.mkdir(parents=True)
    source_map = common.sources(("scripts/cdrm_fuzzy_validate.py", "scripts/cdrm_precision_confirm.py",
                                 "scripts/cdrm_tiled_validate.py", "scripts/r3_backward_validate.py",
                                 "scripts/cdrm_precision_probe.py", str(CONTRACT)))
    report = {"schema": "cdrm-fuzzy-numerical-v1", "status": "running", "arm": args.arm,
              "checkpoint": reference(args.checkpoint), "completed_updates": checkpoint["completed_updates"],
              "criteria": reference(CONTRACT), "source_sha256": source_map,
              "fixture": {"root": str(args.data_root), "split": args.split, "offset": args.offset,
                  "shape": list(batch.input_ids.shape), "sha256": batch.sha256,
                  "dataset_sha256": dataset.sha256, "manifest": dataset.manifest,
                  "native_targets": int((batch.labels != -100).sum()),
                  "loss": "native aligned labels; dense training or native masked development; no additional shift"},
              "packets": {}, "compiler": {}, "construction": {}, "comparisons": {}}
    common.snapshot_sources(args.output_dir, source_map)
    (args.output_dir/"criteria.md").write_bytes(CONTRACT.read_bytes())
    tracker = None
    start = time.monotonic()
    fixed_side = fixed_cotangent = None
    try:
        report["runtime"] = common.setup(0)
        tokens = torch.as_tensor(batch.input_ids, device="cuda")
        labels = torch.as_tensor(batch.labels, device="cuda")
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir, preserve_state=preserve_rng)
        report["wandb"] = tracker.record
        tracker.start({"evidence": "NUM", "arm": args.arm, "fixture": report["fixture"],
                       "checkpoint_sha256": report["checkpoint"]["sha256"], "criteria_sha256": CONTRACT_SHA})
        names = (["naive_fp32"] if args.include_naive else []) + ["tiled_fp32", "mixed"]
        for name in names:
            print(f"Numerical {args.arm} {name}: B{args.batch}/T{args.length}", flush=True)
            torch._dynamo.reset(); torch._dynamo.utils.counters.clear(); configure_compiled_helpers(True)
            mixed, backend = name == "mixed", "naive" if name == "naive_fp32" else "tiled"
            model, construction = common.build_model(args.arm, "bf16" if mixed else "fp32", backend,
                checkpoint=checkpoint, device="cuda", length=args.length)
            report["construction"][name] = construction
            rate = checkpoint.get("optimizer", {}).get("param_groups", [{}])[0].get("lr", 5e-4)
            optimizer = common.optimizer_for(model, rate)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
            compare_checkpoint = {**checkpoint, "optimizer": cpu(optimizer.state_dict())}
            packet = full_packet(model, tokens, labels, mixed, common.native_loss_sum) if args.arm == "cdrm" else seq_packet(model, tokens, labels, mixed)
            current = {"full": packet}
            if fixed_cotangent is None:
                fixed_cotangent = packet["cotangent"]
            if args.arm == "cdrm":
                if fixed_side is None:
                    from olmo.model import causal_attention_bias
                    bias = model.get_alibi_attention_bias(args.length,tokens.device)[:,:,:args.length,:args.length]+causal_attention_bias(args.length,tokens.device)
                    fixed_side = {"p3": packet["states"]["p3"], "p8": packet["states"]["p8"],
                                  "attention_bias": cpu(bias), "cotangent": torch.randn(packet["states"]["hat_m"].shape,
                                  generator=torch.Generator().manual_seed(args.side_seed)), "cotangent_seed": args.side_seed,
                                  "source_arm": name}
                    report["side_fixture"] = save_packet(args.output_dir/"side-fixture.pt", fixed_side)
                current["side"] = side_packet(model, fixed_side, mixed, args.scale_check)
            if args.scale_check:
                current["output_scaling"] = output_scale_check(model, tokens, mixed, fixed_cotangent, packet["logits"])
            if args.causality_check:
                current["causality"] = causality_check(model,tokens,mixed,packet["logits"])
                report.setdefault("causality",{})[name] = current["causality"]
            for pname, param in model.named_parameters():
                if not torch.equal(cpu(param.grad), packet["parameters"][pname]):
                    raise AssertionError(f"Derivative probe changed actual-CE gradient: {pname}")
            current["step"] = step_packet(model, optimizer, 1.)
            required = args.arm == "cdrm" and backend != "naive"
            report["compiler"][name] = {**compiler_audit(False, require_graphs=False), "require_graphs": required,
                                        "validation_status": "pending"}
            report["packets"][name] = save_packet(args.output_dir/(name+".pt"), current)
            report["compiler"][name] = {**compiler_audit(True, require_graphs=required), "require_graphs": required,
                                        "validation_status": "passed"}
            tracker.log({f"ce/{name}": packet["loss"], f"clip_norm/{name}": current["step"]["clip_norm"]})
            atomic_json(args.output_dir/"progress.json", report, replace=True)
            del model, optimizer, current, packet
            gc.collect(); torch.cuda.empty_cache()
        pairs = ([("tiled_fp32", "naive_fp32")] if args.include_naive else []) + [("mixed", "tiled_fp32")]
        for actual_name, ref_name in pairs:
            ref = torch.load(report["packets"][ref_name]["path"], map_location="cpu", weights_only=False)
            actual = torch.load(report["packets"][actual_name]["path"], map_location="cpu", weights_only=False)
            if args.arm == "cdrm":
                row = compare_pair(ref["full"], actual["full"], ref["side"], actual["side"], ref["step"], actual["step"],
                                   compare_checkpoint, fp32=actual_name == "tiled_fp32")
            else:
                row = seq_compare(ref, actual, compare_checkpoint)
            if args.scale_check:
                row["scaling_pass"] = all(packet["output_scaling"]["pass"] and
                    all(r["bitwise_normalized_equal"] for group in packet.get("side", {}).get("scaling", {}).values() for r in group.values())
                    for packet in (ref, actual))
                row["output_scaling"] = {name: {k:v for k,v in packet["output_scaling"].items() if k not in ("baseline_gradients", "cotangent")}
                                         for name,packet in ((ref_name,ref),(actual_name,actual))}
            report["comparisons"][f"{actual_name}_vs_{ref_name}"] = row
            tracker.summary({f"{actual_name}/gradient_relative_l2": row["actual_ce_gradients"]["global_parameter_relative_l2"],
                             f"{actual_name}/adam_relative_l2": row["adam"]["global_delta_relative_l2"],
                             f"{actual_name}/adam_cosine": row["adam"]["global_delta_cosine"],
                             f"{actual_name}/machine_screens_pass": row["machine_screens_pass"]})
            del ref,actual
            gc.collect()
        candidate = report["comparisons"]["mixed_vs_tiled_fp32"]
        report["candidate_checks_pass"] = candidate["machine_screens_pass"] and candidate.get("scaling_pass", True)
        if args.causality_check:
            report["causality_checks_pass"] = all(row["prefix_logits_bitwise_equal"] for row in report["causality"].values())
            report["candidate_checks_pass"] = report["candidate_checks_pass"] and report["causality_checks_pass"]
        report["raw_machine_screens_pass"] = all(x["machine_screens_pass"] and x.get("scaling_pass",True) for x in report["comparisons"].values())
        report["scale_check_performed"] = args.scale_check
        report["disposition"] = "Bounded same-state observation; retain strict FP32 coordinate flags and initial Adam distance separately. No automatic training-policy clearance."
        if reference(args.checkpoint) != report["checkpoint"] or load_dataset(args.data_root,"fuzzy-in-context-recall",args.split,verify=True).sha256 != dataset.sha256:
            raise AssertionError("Scientific input changed during validation")
        common.verify_sources(source_map)
        common.verify_sources(checkpoint["identity"]["source_sha256"])
        report["status"] = "diagnostics_complete"
    except BaseException as error:
        report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        try:
            if tracker:
                tracker.finish(succeeded=report["status"] == "diagnostics_complete")
        except BaseException as error:
            report.update(status="execution_failed",final_sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir/"report.json", report)
    print(json.dumps({"status":report["status"],"candidate_checks_pass":report["candidate_checks_pass"],
                      "report":str(args.output_dir/"report.json")}),flush=True)


if __name__ == "__main__":
    main()
