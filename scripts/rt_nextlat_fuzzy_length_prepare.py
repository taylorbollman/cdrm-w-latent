#!/usr/bin/env python3
"""Freeze evaluation-only MAD Fuzzy length probes shared by all four mixed arms.

Only sequence length changes from the existing V16/T400 experiment. The two
development sets are independently seeded native draws, not new training or
confirmation data. Run with CDRM_DOCKER_GPUS=none in the project container.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.mad_data import (
    FUZZY_TASK, GENERATOR_SHA256, IGNORE_INDEX, REVISION, VENDOR_ROOT,
    file_sha256, generate_dataset, load_dataset, overlap_audit, save_dataset,
    task_spec,
)
from cdrm.rt_nextlat_fuzzy_metrics import evaluation_metadata
from scripts.rt_nextlat_fuzzy_prepare import source_hashes as original_source_hashes
from scripts.rt_nextlat_fuzzy_prepare import structural_audit


SCHEMA = "rt-nextlat-fuzzy-length-development-data-v1"
DEFAULT_REFERENCE = ROOT / ".runtime/rt-nextlat-fuzzy-a5/20260916T154000Z-d128-t400/data"
DEFAULT_LENGTHS = (512, 1024)
DEFAULT_SEEDS = (2026091851, 2026091852)


def source_hashes():
    hashes = original_source_hashes()
    for name in ("scripts/rt_nextlat_fuzzy_length_prepare.py",
                 "tests/test_rt_nextlat_fuzzy_length_prepare.py"):
        hashes[name] = file_sha256(ROOT / name)
    return hashes


def _write_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "manifest.json")


def _summary(values):
    values = np.asarray(values)
    return {"count": int(values.size), "min": int(values.min()) if values.size else None,
            "max": int(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None}


def _reference_records(reference_data_dir):
    result = {}
    for split in ("train", "dev"):
        path = Path(reference_data_dir).resolve() / FUZZY_TASK / f"{split}.manifest.json"
        record = json.loads(path.read_text())
        if record["task"] != FUZZY_TASK or record["split"] != split:
            raise ValueError("Reference manifest has the wrong task or split")
        if record["generator_sha256"] != GENERATOR_SHA256 or record["generator_revision"] != REVISION:
            raise ValueError("Reference generator differs from pinned native MAD")
        expected = task_spec(FUZZY_TASK, {"seq_len": record["actual_sequence_length"]})["config"]
        if record["config"] != expected:
            raise ValueError("Reference task differs from the fixed V16/motifs3/noise0 recipe")
        result[split] = {"original_path": str(path), "sha256": file_sha256(path),
                         "snapshot": f"reference/{split}.manifest.json",
                         "seed": record["seed"], "examples": record["num_examples"],
                         "length": record["actual_sequence_length"], "config": record["config"]}
    if result["train"]["config"] != result["dev"]["config"]:
        raise ValueError("Reference train/dev task configurations differ")
    return result


def verify_manifest(data_dir, *, verify_sources=True):
    """Return ``(manifest, {int_length: MadDataset})`` after checksum validation.

    Evaluation callers must also record the top-level manifest SHA256. This
    verifies frozen native arrays/metadata, source snapshots, fixed task recipe,
    and absence of any generated train/final split.
    """
    output = Path(data_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Expected a completed frozen Fuzzy length-development manifest")
    if (manifest.get("final_split_generated") is not False
            or manifest.get("training_split_generated") is not False
            or manifest.get("models_used") is not False
            or manifest.get("only_task_config_change") != "seq_len"):
        raise ValueError("Length probes must remain evaluation-only and independent of models")
    expected_sources = manifest["source_sha256"]
    if verify_sources and source_hashes() != expected_sources:
        raise ValueError("Fuzzy length-preparation source changed")
    for name, digest in expected_sources.items():
        if file_sha256(output / "source" / name) != digest:
            raise ValueError("Frozen preparation source snapshot changed")
    for record in manifest["reference_splits"].values():
        if file_sha256(output / record["snapshot"]) != record["sha256"]:
            raise ValueError("Frozen reference manifest changed")
    datasets = {}
    for length_text, record in manifest["lengths"].items():
        length = int(length_text)
        if record["data_directory"] != f"length-{length}":
            raise ValueError("Unexpected length-probe relative directory")
        directory = output / record["data_directory"]
        files = sorted(path.name for path in (directory / FUZZY_TASK).iterdir())
        if files != ["dev.manifest.json", "dev.metadata.json", "dev.npz"]:
            raise ValueError("Only the development split may exist for a length probe")
        path = directory / FUZZY_TASK / "dev.manifest.json"
        if file_sha256(path) != record["manifest_sha256"]:
            raise ValueError("Frozen length-probe split manifest changed")
        dataset = load_dataset(directory, FUZZY_TASK, "dev")
        if (dataset.sha256 != record["dataset_sha256"]
                or dataset.input_ids.shape != (record["examples"], length)
                or dataset.manifest["seed"] != record["seed"]
                or dataset.manifest["config"] != task_spec(FUZZY_TASK, {"seq_len": length})["config"]
                or not np.array_equal(dataset.labels, dataset.answer_labels)
                or int((dataset.answer_labels != IGNORE_INDEX).sum()) != record["answer_scored_tokens"]):
            raise ValueError("Frozen length-probe identity or native semantics changed")
        datasets[length] = dataset
    if not datasets:
        raise ValueError("Missing frozen length probes")
    return manifest, datasets


def prepare(output_dir, *, lengths=DEFAULT_LENGTHS, seeds=DEFAULT_SEEDS,
            dev_examples=1280, reference_data_dir=DEFAULT_REFERENCE,
            representative_examples=32):
    lengths, seeds = tuple(lengths), tuple(seeds)
    if (not lengths or len(lengths) != len(seeds) or len(set(lengths)) != len(lengths)
            or any(type(length) is not int for length in lengths)
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Expected distinct integer lengths and equally many distinct uint32 seeds")
    if type(dev_examples) is not int or min(dev_examples, representative_examples) < 1:
        raise ValueError("Positive development and audit sizes are required")
    references = _reference_records(reference_data_dir)
    reference_config = references["train"]["config"]
    if set(seeds) & {record["seed"] for record in references.values()}:
        raise ValueError("Length probes need seeds independent of existing train/development data")
    for length in lengths:
        config = task_spec(FUZZY_TASK, {"seq_len": length})["config"]
        differences = [name for name in config if config[name] != reference_config[name]]
        if differences != ["seq_len"] or length <= reference_config["seq_len"]:
            raise ValueError("Only sequence length may change, and it must exceed the training length")
    provenance = json.loads((VENDOR_ROOT / "CDRM_PROVENANCE.json").read_text())
    for name, record in provenance["files"].items():
        if file_sha256(VENDOR_ROOT / name) != record["sha256"]:
            raise ValueError(f"Pinned MAD source changed: {name}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen development data {output}")
    frozen_sources = source_hashes()
    output.mkdir(parents=True)
    for name in frozen_sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for record in references.values():
        destination = output / record["snapshot"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["original_path"], destination)
    manifest = {"schema": SCHEMA, "status": "preparing", "task": FUZZY_TASK,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": frozen_sources, "reference_splits": references,
                "generator_revision": REVISION, "generator_sha256": GENERATOR_SHA256,
                "only_task_config_change": "seq_len", "training_split_generated": False,
                "final_split_generated": False, "models_used": False,
                "scope": "Shared development-only length generalization for base/input/value/head arms; no model selection or training occurs in preparation.",
                "native_arrays_unchanged": True, "native_input_token_offset": 0,
                "model_input_mapping": "Evaluator adds 60 to inputs and restricts output logits to IDs 60:76; saved IDs remain native 0:16.",
                "labels_already_aligned_with_logits": True,
                "padding_attention_mask_added": False, "resampling_or_rejection": False,
                "lengths": {}}
    _write_manifest(output, manifest)
    datasets = {}
    try:
        for length, seed in zip(lengths, seeds):
            print(f"Generating shared development-only native Fuzzy T{length}: {dev_examples} rows", flush=True)
            dataset = generate_dataset(FUZZY_TASK, "dev", seed, dev_examples, {"seq_len": length})
            directory = output / f"length-{length}"
            saved = save_dataset(directory, dataset)
            loaded = load_dataset(directory, FUZZY_TASK, "dev")
            if dataset.sha256 != loaded.sha256:
                raise AssertionError("Saved native arrays differ from generated rows")
            metadata = evaluation_metadata(loaded)
            known_distances = metadata["distance"][metadata["masks"]["known_history"]]
            padding = (loaded.input_ids == 15).sum(axis=1)
            manifest["lengths"][str(length)] = {
                "data_directory": directory.name, "examples": len(loaded), "seed": seed,
                "length": length, "task_spec": task_spec(FUZZY_TASK, {"seq_len": length}),
                "dataset_sha256": loaded.sha256, "manifest_sha256": saved["sha256"],
                "native_scored_tokens": loaded.manifest["native_scored_tokens"],
                "answer_scored_tokens": loaded.manifest["answer_scored_tokens"],
                "oracle_coverage": loaded.manifest["oracle_coverage"],
                "evaluation_regions": {name: int(mask.sum()) for name, mask in metadata["masks"].items()},
                "answer_motifs": metadata["motifs"], "distance_definition": metadata["distance_definition"],
                "known_history_distance_tokens": _summary(known_distances),
                "left_padding_tokens_per_example": _summary(padding),
                "total_native_padding_tokens": int(padding.sum()),
                "total_input_tokens": int(loaded.input_ids.size),
                "structural_audit": structural_audit(loaded, representative_examples=representative_examples)}
            datasets[f"dev_length_{length}"] = loaded
            _write_manifest(output, manifest)
        manifest["overlap_audit"] = overlap_audit(datasets)
        manifest["reference_overlap_audit"] = {
            "method": "Length-aware exact full-input comparison: unequal complete sequence lengths cannot be identical. No prefix/subsequence independence is claimed.",
            "reference_lengths": {name: record["length"] for name, record in references.items()},
            "new_lengths": list(lengths), "no_exact_full_input_reference_overlap": True,
            "reference_arrays_loaded_or_modified": False}
        if source_hashes() != frozen_sources:
            raise RuntimeError("Preparation sources changed during execution")
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        saved_native_arrays_verified=True, source_hashes_unchanged=True)
        _write_manifest(output, manifest)
        verify_manifest(output)
    except Exception as error:
        manifest.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _write_manifest(output, manifest)
    print(json.dumps({"status": "complete", "manifest": str(output / "manifest.json"),
                      "manifest_sha256": file_sha256(output / "manifest.json")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dev-examples", type=int, default=1280)
    parser.add_argument("--reference-data-dir", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd().resolve() != ROOT:
        parser.error("Run CPU preparation inside the project container working directory")
    prepare(**vars(args))


if __name__ == "__main__":
    main()
