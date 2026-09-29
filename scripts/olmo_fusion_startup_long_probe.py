#!/usr/bin/env python3
"""Supplementary held-out B2/T128 NF precision probe before/after fusion startup.

Prepare exports four additional development-document prefixes independently of
GPU execution. Each state runs two precision cases/four physical backwards, no
optimizer. Full gradients and per-position observations remain descriptive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_data import TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe, feedback_noise_for_rows
from cdrm.pretrained.document_shards import iter_documents
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_adapted_precision import REFERENCE_SHA
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata
from scripts.olmo_campaign_fusion_precision import load_reference
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_data import (
    StartupData, DEFAULT_ROOT, DEFAULT_MANIFEST_SHA256, _digest, _json,
    _file_signatures, _recipe_noise_contract, _tensor_pin, _noise_pins, load_fresh_fixture,
)
from scripts.olmo_fusion_startup_probe import measure_pair, source_hashes as probe_sources
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint, source_hashes as train_sources
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-fusion-startup-long-fixture-v1"
TRAINING_MANIFEST_SHA = "2316978559b8db357c8d4adf706e41c94f809922171e8fb0ba50dc17ad66cbc0"
SHORT_FIXTURE_SHA = "5e55ee7bab67bcffb9fb01de48b9a971bae6ecbee59b8b7ebdf189c52600918f"
COUNTS = {"ce": 508, "latent": 508, "kl": 504}
FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}


def source_hashes():
    sources = train_sources()
    for name, digest in probe_sources().items():
        if name in sources and sources[name] != digest:
            raise ValueError("Frozen training/probe source inventories disagree")
        sources[name] = digest
    for name in ("scripts/olmo_fusion_startup_long_probe.py", "tests/test_fusion_startup_long_probe.py",
                 "docs/reports/olmo-fusion-startup/long-probe-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def document_key(document):
    return _digest({"source": asdict(document.source), "document_id": document.document_id,
        "text_sha256": document.text_sha256, "content_sha256": _digest(document.tokens[:-1]), "split": document.split})


def build_long_fixture(documents, training_manifest, short_metadata, *, short_sha256, recipe, width):
    """Pure CPU construction; identities use the pinned startup document manifest."""
    if (training_manifest.get("schema") != "olmo-fusion-startup-data-v1"
            or short_metadata.get("training_manifest_sha256") != _digest(training_manifest)
            or short_metadata.get("prepared_manifest_sha256") != training_manifest["provenance"]["manifest_sha256"]):
        raise ValueError("Training and short-fixture manifest authorities disagree")
    exclude = {row["key"] for row in short_metadata["documents"]}
    if len(exclude) != 4 or exclude != {row["key"] for row in training_manifest["fresh_fixture"]["documents"]}:
        raise ValueError("Require exactly the original four short-fixture dev exclusions")
    inventory = {row["key"]: (index, row) for index, row in enumerate(training_manifest["documents"])}
    if len(inventory) != len(training_manifest["documents"]):
        raise ValueError("Training manifest has repeated document keys")
    by_source, observed = {}, set()
    for document in documents:
        if not isinstance(document, TokenizedDocument):
            raise TypeError("Long fixture requires verified tokenized documents")
        key = document_key(document)
        if key not in inventory or key in observed:
            raise ValueError("Prepared document identity differs from the training manifest")
        observed.add(key)
        index, expected = inventory[key]
        actual = {"key": key, "source": document.source.name, "split": document.split,
                  "tokens": len(document.tokens), "content_sha256": _digest(document.tokens[:-1])}
        if actual != expected or document.tokens[-1] != training_manifest["eos_id"]:
            raise ValueError("Prepared document metadata/content/EOS differs")
        if document.split == "dev" and len(document.tokens) >= 128 and key not in exclude:
            by_source.setdefault(document.source.name, []).append((key, index, document))
    if observed != set(inventory):
        raise ValueError("Long fixture preparation did not inspect the complete pinned corpus")
    for candidates in by_source.values():
        candidates.sort(key=lambda row: row[0])
    source_order = sorted(by_source)
    selected, offset = [], 0
    while len(selected) < 4:
        available = [by_source[name][offset] for name in source_order if len(by_source[name]) > offset]
        if not available:
            raise ValueError("Need four additional dev documents with at least128 actual tokens")
        selected.extend(available[:4-len(selected)])
        offset += 1
    docs, records = [], []
    for key, index, document in selected:
        docs.append({"key": key, "document_index": index, "source": asdict(document.source),
            "document_id": document.document_id, "split": document.split, "text_sha256": document.text_sha256,
            "full_token_count": len(document.tokens), "full_tokens_sha256": _digest(document.tokens),
            "slice_start": 0, "slice_length": 128, "slice_tokens_sha256": _digest(document.tokens[:128])})
    for start in (0, 2):
        pair = selected[start:start+2]
        ids = torch.tensor([row[2].tokens[:128] for row in pair], dtype=torch.long)
        valid = torch.ones_like(ids, dtype=torch.bool)
        doc_ids = torch.tensor([[row[1]]*128 for row in pair], dtype=torch.long)
        target = valid.clone(); target[:, 0] = False
        batch = NextLatBatch(ids, valid, doc_ids, target, target.clone(), target.clone())
        keys = [_digest(["startup-long-prefix-v1", row[0], 0, 128]) for row in pair]
        noise = feedback_noise_for_rows(recipe, keys, logical_update=0,
            sequence_length=128, width=width, physical_batch_size=2, device="cpu")
        records.append({"batch": {name: value.tolist() for name, value in vars(batch).items()},
            "batch_pins": {name: _tensor_pin(value) for name, value in vars(batch).items()},
            "noise_keys": keys, "noise_pins": _noise_pins(noise)})
    training_keys = sorted(key for key, (_, row) in inventory.items() if row["split"] == "train")
    return {"schema": SCHEMA, "training_manifest_sha256": _digest(training_manifest),
        "prepared_manifest_sha256": training_manifest["provenance"]["manifest_sha256"],
        "short_fixture_sha256": short_sha256, "excluded_short_document_keys": sorted(exclude),
        "training_document_keys_sha256": _digest(training_keys), "training_document_count": len(training_keys),
        "selection": {"policy": "round-robin alphabetically sorted source names, ascending document keys within source",
            "eligible_source_counts": {name: len(by_source[name]) for name in source_order},
            "eligible_split": "dev", "minimum_document_tokens": 128, "start_offset": 0},
        "document_policy": "isolated-v1", "length": 128, "physical_batch_size": 2,
        "counts": COUNTS, "input_tokens": 512, "microbatches": 2, "padding_tokens": 0,
        "pad_id": training_manifest["pad_id"], "vocab_size": training_manifest["vocab_size"],
        "documents": docs, "noise_contract": _recipe_noise_contract(recipe, width), "records": records,
        "disjointness": "All chosen keys are verified dev, outside both training and original short fixture; no confirmation data"}


def _publish_fixture(path, metadata):
    path = Path(path)
    raw = (_json(metadata)+"\n").encode()
    if len(raw) > 128*1024:
        raise ValueError("Long-fixture export exceeds its bounded JSON budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": len(raw)}


def load_long_fixture(path, *, expected_sha256, recipe, width,
                      expected_training_manifest_sha256=TRAINING_MANIFEST_SHA,
                      expected_prepared_manifest_sha256=DEFAULT_MANIFEST_SHA256,
                      expected_short_fixture_sha256=SHORT_FIXTURE_SHA):
    """Authenticate CPU tensor/noise reconstruction without opening training data."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024
            or sha256_file(path) != expected_sha256):
        raise ValueError("Long fixture differs from its bounded independent SHA256")
    raw = path.read_bytes()
    if sha256_file(path) != expected_sha256:
        raise ValueError("Long fixture changed while loading")
    metadata = json.loads(raw)
    if (metadata.get("schema") != SCHEMA or metadata.get("noise_contract") != _recipe_noise_contract(recipe, width)
            or metadata.get("training_manifest_sha256") != expected_training_manifest_sha256
            or metadata.get("prepared_manifest_sha256") != expected_prepared_manifest_sha256
            or metadata.get("short_fixture_sha256") != expected_short_fixture_sha256
            or metadata.get("document_policy") != "isolated-v1" or metadata.get("length") != 128
            or metadata.get("physical_batch_size") != 2 or metadata.get("counts") != COUNTS
            or metadata.get("input_tokens") != 512 or metadata.get("microbatches") != 2 or metadata.get("padding_tokens") != 0
            or len(metadata.get("documents", [])) != 4 or len(metadata.get("records", [])) != 2):
        raise ValueError("Long fixture contract differs")
    docs = metadata["documents"]
    exclusions = metadata.get("excluded_short_document_keys", [])
    if (len(set(exclusions)) != 4 or len({d["key"] for d in docs}) != 4
            or len({d["document_index"] for d in docs}) != 4 or {d["key"] for d in docs} & set(exclusions)
            or any(d["split"] != "dev" or d["full_token_count"] < 128 or d["slice_start"] != 0
                   or d["slice_length"] != 128 or type(d["document_index"]) is not int or d["document_index"] < 0 for d in docs)):
        raise ValueError("Long fixture document split/identity/exclusions differ")
    fixtures, counts = [], dict.fromkeys(COUNTS, 0)
    for index, record in enumerate(metadata["records"]):
        if set(record["batch"]) != FIELDS or set(record["batch_pins"]) != FIELDS:
            raise ValueError("Long batch tensor inventory differs")
        tensors = {}
        for name in FIELDS:
            dtype = torch.long if name in ("input_ids", "document_ids") else torch.bool
            values = record["batch"][name]
            scalar = int if dtype == torch.long else bool
            if (not isinstance(values, list) or len(values) != 2
                    or any(not isinstance(row, list) or len(row) != 128 or any(type(v) is not scalar for v in row) for row in values)):
                raise ValueError("Long batch tensor shape/scalar type differs")
            value = torch.tensor(values, dtype=dtype)
            if _tensor_pin(value) != record["batch_pins"][name]:
                raise ValueError("Long batch tensor hash differs")
            tensors[name] = value
        batch = NextLatBatch(**tensors)
        keys = []
        for row, doc in enumerate(docs[index*2:index*2+2]):
            tokens = record["batch"]["input_ids"][row]
            if (_digest(tokens) != doc["slice_tokens_sha256"] or any(not 0 <= t < metadata["vocab_size"] for t in tokens)
                    or record["batch"]["valid_mask"][row] != [True]*128
                    or record["batch"]["document_ids"][row] != [doc["document_index"]]*128
                    or any(record["batch"][name][row] != [False]+[True]*127 for name in ("ce_mask", "latent_mask", "kl_mask"))):
                raise ValueError("Long actual-token slice/mask/document alignment differs")
            keys.append(_digest(["startup-long-prefix-v1", doc["key"], 0, 128]))
        if record["noise_keys"] != keys:
            raise ValueError("Long keyed noise identities differ")
        noise = feedback_noise_for_rows(recipe, keys, logical_update=0,
            sequence_length=128, width=width, physical_batch_size=2, device="cpu")
        if _noise_pins(noise) != record["noise_pins"]:
            raise ValueError("Long keyed noise bytes differ")
        for name, mask in build_nextlat_masks(batch, document_policy="isolated-v1").items():
            counts[name] += int(mask.sum())
        fixtures.append(((batch,), (noise,)))
    if counts != COUNTS:
        raise ValueError("Long reconstructed objective counts differ")
    return fixtures, metadata


def load_cold_report(path, expected_sha256, sources, *, fixture_sha256):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 64*1024*1024 or sha256_file(path) != expected_sha256:
        raise ValueError("Cold long-probe report differs from its immutable pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha256:
        raise ValueError("Cold long-probe report changed while reading")
    if (report.get("schema") != "olmo-fusion-startup-long-probe-v1" or report.get("state") != "cold"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True
            or report.get("fixture_sha256") != fixture_sha256 or report.get("optimizer_updates") != 0
            or report.get("aggregate_backwards") != 2 or report.get("physical_backwards") != 4
            or report.get("determinism", {}).get("deterministic_algorithms") is not True
            or not report.get("integrity") or not all(report["integrity"].values())
            or not report.get("pair_integrity") or not all(report["pair_integrity"].values())):
        raise ValueError("Cold reference lacks completed matched long-fixture integrity")
    if not report.get("sources") or any(sources.get(name) != pin for name, pin in report["sources"].items()):
        raise ValueError("Cold long-probe sources differ")
    endpoints = {}
    for path_name in (FP32, BF16):
        rows = [row for row in report["rows"] if row.get("path") == path_name and row.get("objective") == "ce"]
        if (len(rows) != 1 or not rows[0].get("passed") or not rows[0].get("health")
                or not all(rows[0]["health"].values())):
            raise ValueError("Cold reference precision endpoint differs")
        endpoints[path_name] = rows[0]
    return report, endpoints


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only immutable development-fixture export")
    prepare.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    prepare.add_argument("--prepared-manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    prepare.add_argument("--training-manifest-sha256", default=TRAINING_MANIFEST_SHA)
    prepare.add_argument("--short-fixture", type=Path, default=ROOT/".runtime/olmo-fusion-startup/data-01/fresh_fixture.json")
    prepare.add_argument("--short-sha256", default=SHORT_FIXTURE_SHA)
    prepare.add_argument("--output-dir", type=Path, required=True)
    probe = commands.add_parser("probe", help="GPU-only cold or update128 matched precision pair")
    probe.add_argument("--state", choices=("cold", "startup"), required=True)
    probe.add_argument("--fixture", type=Path, required=True)
    probe.add_argument("--fixture-sha256", required=True)
    probe.add_argument("--checkpoint", type=Path)
    probe.add_argument("--checkpoint-sha256")
    probe.add_argument("--cold-report", type=Path)
    probe.add_argument("--cold-report-sha256")
    probe.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    probe.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        supplied = (args.checkpoint, args.checkpoint_sha256, args.cold_report, args.cold_report_sha256)
        if (args.state == "startup" and not all(supplied)) or (args.state == "cold" and any(supplied)):
            parser.error("Only startup requires complete checkpoint and cold-reference path/SHA pairs")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Evidence must remain in persistent project storage")
    return args


def snapshot_sources(output, sources):
    for name in sources:
        target = output/"source-snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)


def prepare_main(args):
    if not Path("/.dockerenv").exists() or Path.cwd() != ROOT:
        raise RuntimeError("Prepare inside the CPU-only project container")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(); snapshot_sources(args.output_dir, sources)
    started = time.monotonic()
    signatures = _file_signatures(args.data_root)
    data = StartupData.from_prepared(args.data_root, expected_manifest_sha256=args.prepared_manifest_sha256)
    if data.manifest_sha256 != args.training_manifest_sha256:
        raise ValueError("Long fixture must use the frozen warmup data manifest")
    recipe, width = CampaignRecipe("NF"), 2048
    _, short = load_fresh_fixture(args.short_fixture, expected_sha256=args.short_sha256, recipe=recipe, width=width)
    documents = iter_documents(args.data_root, verify=False)
    metadata = build_long_fixture(documents, data.manifest, short, short_sha256=args.short_sha256, recipe=recipe, width=width)
    if signatures != _file_signatures(args.data_root):
        raise ValueError("Prepared data changed during long-fixture export")
    artifact = _publish_fixture(args.output_dir/"long_fixture.json", metadata)
    fixtures, loaded = load_long_fixture(artifact["path"], expected_sha256=artifact["sha256"], recipe=recipe, width=width,
        expected_training_manifest_sha256=args.training_manifest_sha256,
        expected_prepared_manifest_sha256=args.prepared_manifest_sha256,
        expected_short_fixture_sha256=args.short_sha256)
    checks = {"export_reload_exact": loaded == metadata,
        "sources_unchanged": source_hashes() == sources, "two_records": len(fixtures) == 2,
        "original_short_fixture_unchanged": sha256_file(args.short_fixture) == args.short_sha256}
    report = {"schema": "olmo-fusion-startup-long-prepare-v1", "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()), "sources": sources, "checks": checks, "fixture": artifact,
        "metadata": metadata, "elapsed_seconds": time.monotonic()-started, "device": "cpu"}
    write_json(args.output_dir/"report.json", report)
    if not report["passed"]:
        raise AssertionError("Long-fixture CPU preparation failed")
    print({"fixture": artifact, "counts": COUNTS, "sources": [d["source"]["name"] for d in metadata["documents"]]}, flush=True)


def probe_main(args):
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Long precision probe is one GPU, without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(); snapshot_sources(args.output_dir, sources)
    report = {"schema": "olmo-fusion-startup-long-probe-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "runtime": runtime,
        "determinism": deterministic, "state": args.state, "fixture": str(args.fixture), "fixture_sha256": args.fixture_sha256,
        "rows": [], "aggregate_backwards": 2, "physical_backwards": 4, "optimizer_updates": 0,
        "qualification": "Four additional dev prefixes improve fixture coverage; no BF16 budget, quality or optimizer-update clearance",
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "cold_report": None if args.cold_report is None else str(args.cold_report), "cold_report_sha256": args.cold_report_sha256,
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-fusion-startup", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("state", "fixture_sha256", "checkpoint_sha256", "qualification", "determinism")})
        persist("construct_original_nf")
        model, recipe, checkpoint, _, _ = construct(SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
        cold = state_pins(model)
        matrix_path = ROOT/".runtime/olmo-recurrence-precision/matrix-01/report.json"
        matrix = load_reference(matrix_path, REFERENCE_SHA, sources)
        if cold != matrix["arms"]["NF"]["initial_state"]:
            raise AssertionError("Long probe cold construction differs from retained NF origin")
        fixtures, metadata = load_long_fixture(args.fixture, expected_sha256=args.fixture_sha256, recipe=recipe, width=model.config.model_dim)
        report.update(source_checkpoint=checkpoint, cold_state=cold, fixture_provenance=metadata,
            fixture_pins=fixture_pins(fixtures), fixture_metadata=global_fixture_metadata(model, fixtures), contract=arm_contract(model, recipe))
        diagonals = None
        if args.state == "startup":
            reference, diagonals = load_cold_report(args.cold_report, args.cold_report_sha256, sources, fixture_sha256=args.fixture_sha256)
            checks = {"cold_state_exact": cold == reference["cold_state"],
                "source_checkpoint_exact": checkpoint == reference["source_checkpoint"],
                "fixture_exact": report["fixture_pins"] == reference["fixture_pins"],
                "contract_exact": report["contract"] == reference["contract"]}
            for name in ("runtime", "determinism", "math_sdpa_reduced_precision_reduction", "bf16_matmul_reduced_precision_reduction"):
                checks[name+"_exact"] = report[name] == reference[name]
            report["cold_reference_checks"] = checks
            if not all(checks.values()):
                raise AssertionError("Long cold/endpoint comparison contract differs")
            report["import"] = load_fusion_checkpoint(model, args.checkpoint, checkpoint, expected_sha256=args.checkpoint_sha256)
            if report["import"]["counters"]["optimizer_updates"] != 128:
                raise ValueError("Supplementary endpoint must be the declared update128")
            if report["import"]["configuration"]["data_manifest_sha256"] != metadata["training_manifest_sha256"]:
                raise ValueError("Startup endpoint trained against a different data manifest")
            current = state_pins(model)
            if any(current[group] != cold[group] for group in ("backbone", "predictor")):
                raise AssertionError("Startup checkpoint changed frozen original state")
        report["initial_state"] = state_pins(model)
        rng = rng_snapshot()

        def publish(row):
            row["memory"] = memory()
            report["rows"].append(row)
            persist(row["path"])
            tracker.log(scalar_metrics(row, "long_probe/"+row["path"]), step=len(report["rows"]))
            print({"path": row["path"], "passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)

        pair = measure_pair(model, recipe, fixtures, original_flags=original, publish=publish, diagonal_by_path=diagonals)
        report["pair_integrity"] = pair["integrity"]
        report["integrity"] = {"state_unchanged": state_pins(model) == report["initial_state"],
            "sources_unchanged": source_hashes() == sources, "rng_unchanged": rng_unchanged(rng),
            "fixture_file_unchanged": sha256_file(args.fixture) == args.fixture_sha256,
            "fixture_tensors_unchanged": fixture_pins(fixtures) == report["fixture_pins"],
            "gradients_cleared": all(p.grad is None for p in model.parameters()), "two_cases_completed": len(report["rows"]) == 2,
            "production_flags_restored": all(getattr(model.backbone.backbone, name) == value for name, value in original.items()),
            "original_matrix_unchanged": sha256_file(matrix_path) == REFERENCE_SHA}
        if args.state == "startup":
            report["integrity"].update(checkpoint_unchanged=sha256_file(args.checkpoint) == args.checkpoint_sha256,
                cold_report_unchanged=sha256_file(args.cold_report) == args.cold_report_sha256)
        if not all(report["integrity"].values()):
            raise AssertionError("Long precision probe final integrity failed")
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False, error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
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


def main(argv=None):
    args = parse_args(argv)
    return prepare_main(args) if args.command == "prepare" else probe_main(args)


if __name__ == "__main__":
    main()
