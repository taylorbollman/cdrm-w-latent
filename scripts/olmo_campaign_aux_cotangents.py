#!/usr/bin/env python3
"""Bounded fixed-state NextLat cotangents, separate from backbone numerics.

A pinned JSON fixture contains detached BF16-production hidden/embedding
states, masks and weight hashes, never model weights. The pinned initial model
is reconstructed on CPU solely to verify/extract its readout and predictor.
Eight loss-only cases compare latent/KL at FP32/BF16 and sparse/dense layouts,
with one backward per physical fixture record in each case. Other objective
cotangents are zero; canonical detachments remain.
No backbone forward/backward, optimizer, DDP, graph capture or training occurs.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_losses import DynamicNextLatLayout, compute_dynamic_nextlat_loss_sums
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, build_nextlat_masks, compute_nextlat_loss_sums
from cdrm.pretrained.static_nextlat import _cpu_batch
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, source_hashes as campaign_sources
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_components import TERMS, pass_sums
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-campaign-aux-cotangents-fixture-v1"
OBJECTIVES = ("latent", "kl")
MAX_FIXTURE_BYTES = 64 * 1024 * 1024
DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool)}
BATCH_FIELDS = ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")


def source_hashes():
    result = campaign_sources()
    for path in (Path(__file__), ROOT / "scripts/olmo_campaign_precision_components.py",
                 ROOT / "scripts/olmo_f2_graph_backend_probe.py",
                 ROOT / "docs/reports/olmo-precision-localization/protocol.md"):
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(result.items()))


def _encode_tensor(value):
    if not isinstance(value, torch.Tensor) or str(value.dtype) not in DTYPES:
        raise ValueError("Fixture tensor dtype must be FP32, BF16, int64 or bool")
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "data_base64": base64.b64encode(raw).decode("ascii")}


def _decode_tensor(value):
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_base64"}:
        raise ValueError("Invalid encoded tensor fields")
    shape, dtype = value["shape"], DTYPES.get(value["dtype"])
    if (dtype is None or not isinstance(shape, list) or not 1 <= len(shape) <= 3
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("Invalid encoded tensor dtype/shape")
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if size > MAX_FIXTURE_BYTES or not isinstance(value["data_base64"], str):
        raise ValueError("Encoded tensor exceeds bounded fixture size")
    try:
        raw = base64.b64decode(value["data_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid base64 tensor bytes") from error
    if len(raw) != size:
        raise ValueError("Encoded tensor byte length differs from shape")
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def relevant_weights(model):
    if model.predictor is None or not model.enabled:
        raise ValueError("Cotangent fixture requires active NextLat predictor")
    return tree_digests({"readout": model.backbone.readout_weight,
                         "predictor": model.predictor.state_dict()})


def write_fixture(path, model, records, provenance):
    """Export bounded raw CPU bytes, retaining activation dtypes and source pins.

    ``records`` contains key, batch (plain NextLatBatch tensor fields),
    token_embeddings and pass_hidden_states. The caller supplies the unchanged
    ``source_checkpoint`` and ``recipe_sha256`` in provenance. Exporting does
    not execute another backbone forward or keep a live autograd connection.
    """
    encoded = []
    for record in records:
        batch = record["batch"]
        encoded.append({"key": record["key"],
            "batch": {key: None if batch[key] is None else _encode_tensor(batch[key]) for key in BATCH_FIELDS},
            "token_embeddings": _encode_tensor(record["token_embeddings"]),
            "pass_hidden_states": [_encode_tensor(value) for value in record["pass_hidden_states"]]})
    payload = {"schema": SCHEMA, "config": model.config.to_dict(), "provenance": provenance,
               "sources": source_hashes(), "weights": relevant_weights(model), "records": encoded}
    # Validate before publication, including detached-target objective counts.
    decode_fixture(payload)
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(content) > MAX_FIXTURE_BYTES:
        raise ValueError("Fixture exceeds the bounded JSON file size")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".cotangents-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; never overwrite evidence.
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": len(content),
            "records": len(encoded), "schema": SCHEMA}


def decode_fixture(payload):
    if (not isinstance(payload, dict) or set(payload) !=
            {"schema", "config", "provenance", "sources", "weights", "records"}
            or payload["schema"] != SCHEMA):
        raise ValueError("Cotangent fixture schema/fields differ")
    config = NextLatConfig.from_dict(payload["config"])
    if config.dropout or config.lambda_latent <= 0 or config.lambda_kl <= 0:
        raise ValueError("Fixture requires zero dropout and both positive auxiliary weights")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or not {"source_checkpoint", "recipe_sha256"} <= provenance.keys():
        raise ValueError("Fixture provenance requires source checkpoint and recipe pins")
    if not isinstance(payload["records"], list) or not 1 <= len(payload["records"]) <= 8:
        raise ValueError("Fixture requires one to eight bounded records")
    records, keys, counts, pass_count = [], set(), dict.fromkeys(TERMS, 0), None
    for row in payload["records"]:
        if not isinstance(row, dict) or set(row) != {"key", "batch", "token_embeddings", "pass_hidden_states"}:
            raise ValueError("Invalid fixture record fields")
        key = row["key"]
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Fixture record keys must be distinct nonempty strings")
        keys.add(key)
        if set(row["batch"]) != set(BATCH_FIELDS):
            raise ValueError("Fixture batch fields differ")
        batch = _cpu_batch(NextLatBatch(**{k: None if v is None else _decode_tensor(v)
            for k, v in row["batch"].items()}), document_policy=config.document_policy)
        if not 1 <= batch.input_ids.shape[0] <= 2 or not 3 <= batch.input_ids.shape[1] <= 16:
            raise ValueError("Fixture is bounded to B1/B2 and T3..16")
        states = tuple(_decode_tensor(value) for value in row["pass_hidden_states"])
        if len(states) not in (1, 4) or (pass_count is not None and len(states) != pass_count):
            raise ValueError("Fixture records require the same one or four passes")
        pass_count = len(states)
        embeds = _decode_tensor(row["token_embeddings"])
        for tensor in (*states, embeds):
            if (tensor.shape != (*batch.input_ids.shape, config.model_dim)
                    or tensor.dtype not in (torch.float32, torch.bfloat16)
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Fixture activations must be finite FP32/BF16 [B,T,D]")
        masks = build_nextlat_masks(batch, document_policy=config.document_policy)
        for term in TERMS:
            counts[term] += int(masks[term].sum())
        records.append({"key": key, "batch": batch, "token_embeddings": embeds, "pass_hidden_states": states})
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Fixture requires positive global CE, latent and KL counts")
    return {"config": config, "records": records, "counts": counts, "passes": pass_count,
            "provenance": provenance, "sources": payload["sources"], "weights": payload["weights"]}


def load_fixture(path, expected_sha256):
    path = Path(path)
    if path.stat().st_size > MAX_FIXTURE_BYTES or sha256_file(path) != expected_sha256:
        raise ValueError("Fixture bytes differ from the supplied bounded SHA256 pin")
    payload = json.loads(path.read_text())
    if sha256_file(path) != expected_sha256:
        raise ValueError("Fixture changed while reading")
    return decode_fixture(payload)


def verify_reconstructed_model(model, recipe, checkpoint, fixture):
    if (model.config != fixture["config"] or checkpoint != fixture["provenance"]["source_checkpoint"]
            or recipe.sha256 != fixture["provenance"]["recipe_sha256"]):
        raise ValueError("Reconstructed configuration/checkpoint/recipe differs")
    if relevant_weights(model) != fixture["weights"]:
        raise ValueError("Reconstructed readout/predictor bytes differ")


def component_cotangents(fixture, readout, predictor, *, precision, layout, objective):
    """One accumulated loss-only backward case on immutable activation values.

    FP32 promotes any BF16 anchor to FP32 without changing its represented
    values. BF16 keeps anchor dtypes and enables autocast. Predictor masters
    remain FP32. Embeddings are separate leaves, so their cotangent is measured
    without an embedding lookup/backbone graph. Readout is a detached constant.
    """
    if precision not in ("fp32", "bf16_mixed") or layout not in ("sparse", "prepared") or objective not in OBJECTIVES:
        raise ValueError("Unsupported precision/layout/auxiliary objective")
    config, counts = fixture["config"], fixture["counts"]
    if any(p.dtype != torch.float32 for p in predictor.parameters()) or readout.dtype != torch.float32:
        raise ValueError("Loss weights must remain FP32 masters")
    device = readout.device
    readout = readout.detach()
    predictor.zero_grad(set_to_none=True)
    result, totals, selected_total = {}, dict.fromkeys(TERMS, 0.), 0.
    weights = {"ce": 1., "latent": config.lambda_latent, "kl": config.lambda_kl}
    for record in fixture["records"]:
        batch = record["batch"].to(device)
        leaf = lambda tensor: tensor.to(device=device, dtype=torch.float32 if precision == "fp32" else tensor.dtype).detach().clone().requires_grad_(True)
        hidden = tuple(leaf(value) for value in record["pass_hidden_states"])
        embeddings = leaf(record["token_embeddings"])
        prepared = DynamicNextLatLayout.from_batch(record["batch"], config, device=device) if layout == "prepared" else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16_mixed", cache_enabled=False):
            passes = []
            for state in hidden:
                if prepared is None:
                    sums = compute_nextlat_loss_sums(state, embeddings, readout, batch, predictor, config).sums
                else:
                    sums = compute_dynamic_nextlat_loss_sums(state, embeddings, readout, batch.input_ids,
                                                             predictor, config, prepared)
                passes.append(sums)
            sums = pass_sums(passes)
            normalized = {term: sums[term] * weights[term] / counts[term] for term in TERMS}
            selected = sum(normalized[term] * float(term == objective) for term in TERMS)
        selected.backward()
        selected_total += float(selected.detach())
        for term in TERMS:
            totals[term] += float(sums[term].detach())
        for index, value in enumerate(hidden):
            result[f"hidden/{record['key']}/pass_{index}"] = (torch.zeros_like(value) if value.grad is None else value.grad).detach().float().cpu()
        result[f"embedding/{record['key']}"] = (torch.zeros_like(embeddings) if embeddings.grad is None else embeddings.grad).detach().float().cpu()
    missing = []
    for name, parameter in predictor.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        result["predictor/"+name] = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).detach().float().cpu()
    metadata = {"precision": precision, "layout": layout, "objective_component": objective,
                "counts": counts, "loss_sums": totals, "objective": selected_total,
                "loss_backward_calls": len(fixture["records"]),
                "normalized_loss_means": {term: totals[term]/counts[term] for term in TERMS},
                "missing_predictor_gradients_zero_materialized": missing,
                "finite": all(bool(torch.isfinite(value).all()) for value in result.values())
                    and all(math.isfinite(value) for value in (*totals.values(), selected_total))}
    return metadata, result


@torch.no_grad()
def cotangent_comparison(actual, reference):
    if actual.keys() != reference.keys():
        raise ValueError("Cotangent names differ")
    accumulators = {name: dict(reference_squared=0., actual_squared=0., error_squared=0., dot=0., max_abs=0., tensors=0)
                    for name in ("all", "hidden", "embedding", "predictor")}
    for name, value in actual.items():
        expected = reference[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Cotangent shape/dtype differs")
        # Stream one tensor at a time; do not flatten a full predictor copy.
        a, b = value.double(), expected.double()
        error = a-b
        numbers = {"reference_squared": float(b.square().sum()), "actual_squared": float(a.square().sum()),
                   "error_squared": float(error.square().sum()), "dot": float((a*b).sum())}
        for group in ("all", name.split("/", 1)[0]):
            row = accumulators[group]
            for key, number in numbers.items():
                row[key] += number
            row["max_abs"] = max(row["max_abs"], float(error.abs().max()))
            row["tensors"] += 1
    result = {}
    for group, row in accumulators.items():
        ref, norm, err = (math.sqrt(row[key]) for key in ("reference_squared", "actual_squared", "error_squared"))
        cosine = max(-1., min(1., row["dot"]/(ref*norm))) if ref and norm else None
        result[group] = {"reference_norm": ref, "actual_norm": norm, "difference_norm": err,
            "relative_l2": err/max(ref, 1e-30), "norm_ratio": norm/ref if ref else None,
            "cosine": cosine, "max_abs": row["max_abs"], "tensors": row["tensors"],
            "reference_is_zero": ref == 0., "actual_is_zero": norm == 0.,
            "bitwise_equal": all(torch.equal(actual[k], reference[k]) for k in actual
                                 if group == "all" or k.startswith(group+"/"))}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.fixture_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.fixture_sha256):
        parser.error("Fixture SHA256 must be 64 lowercase hexadecimal characters")
    return args


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Cotangent diagnostic requires one process without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-campaign-aux-cotangents-v1", "status": "running", "passed": False,
        "sources": source_hashes(), "runtime": runtime, "determinism": determinism, "rows": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": args.fixture_sha256, "scope": __doc__,
        "acceptance": "Operational finite/state/source gates only; all numerical differences descriptive",
        "execution": "Identical detached activation values; FP32 promotes BF16 anchors exactly; BF16 preserves anchor dtype plus autocast; TF32 off; readout fixed; predictor FP32 masters",
        "optimizer_updates": 0, "backbone_backwards": 0}
    for relative in report["sources"]:
        destination = args.output_dir / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-campaign-aux-cotangents", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir / "report.json", report)

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        persist(stage)
        tracker.log(scalar_metrics(row, "diagnostic/"+stage), step=len(report["rows"]))
        print({"stage": stage, "passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
        if not row["passed"]:
            raise AssertionError("Fixed-state cotangent operational gate failed")

    try:
        tracker.start({key: report[key] for key in ("scope", "fixture_sha256", "determinism", "acceptance", "execution")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("load_fixture_and_reconstruct_weights_on_cpu")
        fixture = load_fixture(args.fixture, args.fixture_sha256)
        if fixture["sources"] != report["sources"]:
            raise ValueError("Fixture source pins differ from this execution")
        model, recipe, checkpoint, _, _ = construct(SimpleNamespace(scale="pretrained", length=16,
            artifacts=args.artifacts, document_policy=fixture["config"].document_policy), "NFR", torch.device("cpu"))
        verify_reconstructed_model(model, recipe, checkpoint, fixture)
        predictor, readout = model.predictor, model.backbone.readout_weight.detach()
        del model
        gc.collect()
        readout, predictor = readout.to("cuda"), predictor.to("cuda").train()
        weights_before = tree_digests({"readout": readout, "predictor": predictor.state_dict()})
        report.update(provenance=fixture["provenance"], weights=fixture["weights"], config=fixture["config"].to_dict(),
            counts=fixture["counts"], passes=fixture["passes"],
            fixture_records=[{"key": r["key"], "batch": tree_digests(vars(r["batch"])),
                "token_embeddings": tree_digests(r["token_embeddings"]),
                "pass_hidden_states": tree_digests(r["pass_hidden_states"])} for r in fixture["records"]])
        before = rng_snapshot()
        for objective in OBJECTIVES:
            references = {}
            for precision, layout in (("fp32", "sparse"), ("fp32", "prepared"),
                                      ("bf16_mixed", "sparse"), ("bf16_mixed", "prepared")):
                stage = objective+"/"+precision+"_"+layout
                persist(stage+"/backward")
                metrics, gradients = component_cotangents(fixture, readout, predictor,
                    precision=precision, layout=layout, objective=objective)
                row = {"metrics": metrics, "cotangent_digests": tree_digests(gradients),
                       "norms": {key: float(value.double().norm()) for key, value in gradients.items()},
                       "rng_unchanged": rng_unchanged(before), "memory": memory(),
                       "passed": metrics["finite"] and not metrics["missing_predictor_gradients_zero_materialized"] and rng_unchanged(before)}
                if references:
                    row["versus_fp32_sparse"] = cotangent_comparison(gradients, references["fp32"])
                if layout == "prepared":
                    row["versus_same_precision_sparse"] = cotangent_comparison(gradients, references[precision])
                    row["hidden_passes_vs_same_precision_sparse"] = {
                        key: cotangent_comparison({key: value}, {key: references[precision][key]})["hidden"]
                        for key, value in gradients.items() if key.startswith("hidden/")}
                else:
                    references[precision] = gradients
                publish(stage, row)
            del references, gradients
            predictor.zero_grad(set_to_none=True)
            gc.collect()
        weights_equal = weights_before == tree_digests({"readout": readout, "predictor": predictor.state_dict()})
        sources_equal = report["sources"] == source_hashes()
        fixture_equal = args.fixture_sha256 == sha256_file(args.fixture)
        publish("fixed_state_source_fixture_integrity", {"weights_unchanged": weights_equal,
            "sources_unchanged": sources_equal, "fixture_unchanged": fixture_equal,
            "rng_unchanged": rng_unchanged(before),
            "passed": weights_equal and sources_equal and fixture_equal and rng_unchanged(before)})
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
