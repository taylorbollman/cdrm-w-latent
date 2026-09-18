#!/usr/bin/env python3
"""Prepare frozen native MAD fuzzy train/dev data for the new RT+NextLat pilot.

Run with CDRM_DOCKER_GPUS=none in the project container. No model is imported;
the confirmation split is declared prospectively but is not generated here.
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
    answer_oracle, baseline_audit, file_sha256, fuzzy_answer_annotation,
    fuzzy_prefix_prediction, generate_dataset, load_dataset, overlap_audit,
    save_dataset, task_spec,
)
from cdrm.rt_nextlat_fuzzy_metrics import evaluation_metadata


SCHEMA = "rt-nextlat-fuzzy-preparation-v1"


def path_record(path):
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def source_hashes():
    names = ("cdrm/mad_data.py", "cdrm/rt_nextlat_fuzzy_metrics.py",
             "scripts/rt_nextlat_fuzzy_prepare.py", "tests/test_rt_nextlat_fuzzy_data.py",
             "vendors/mad-lab/CDRM_PROVENANCE.json", "vendors/mad-lab/mad/data/instances.py",
             "vendors/mad-lab/mad/configs.py", "vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml")
    return {name: file_sha256(ROOT / name) for name in names}


def structural_audit(dataset, *, representative_examples=32, prefix_examples=8, prefixes_per_example=8):
    """Vector checks cover all arrays; expensive causal checks use bounded rows.

    generate_dataset already independently annotates every draw and verifies its
    native labels. This second check deliberately avoids all-prefix O(N*T^2).
    """
    if dataset.manifest["task"] != FUZZY_TASK:
        raise ValueError("Expected native fuzzy data")
    if representative_examples < 1 or prefix_examples < 1 or prefixes_per_example < 1:
        raise ValueError("Audit sample sizes must be positive")
    inputs, native, answers = dataset.input_ids, dataset.labels, dataset.answer_labels
    if dataset.manifest["split"] == "train":
        if (np.any(native == IGNORE_INDEX) or not np.array_equal(native[:, :-1], inputs[:, 1:])
                or not np.array_equal(native[:, -1], answers[:, -1])):
            raise AssertionError("Native dense training labels or existing shift changed")
    elif not np.array_equal(native, answers):
        raise AssertionError("Native held-out answer mask changed")
    valid = answers != IGNORE_INDEX
    if np.any((answers[valid] < 7) | (answers[valid] > 14)):
        raise AssertionError("Retrieval labels must use the native value alphabet")
    representative = np.unique(np.linspace(0, len(dataset) - 1,
                                           min(representative_examples, len(dataset)), dtype=int))
    available = dataset.oracle_available_mask
    prefix_checked = unavailable_checked = 0
    for audit_row, index in enumerate(representative):
        overrides = dataset.manifest.get("task_overrides")
        annotations, info = fuzzy_answer_annotation(inputs[index], overrides, terminal_target=int(native[index, -1]))
        oracle = answer_oracle(FUZZY_TASK, inputs[index], overrides)
        if (not np.array_equal(annotations, answers[index])
                or not np.array_equal(oracle != IGNORE_INDEX, available[index])
                or not np.array_equal(oracle[available[index]], answers[index, available[index]])):
            raise AssertionError("Fuzzy annotations or causal lookup disagree")
        if dataset.manifest["split"] != "train" and any(len(pair["key"]) != 3 for pair in info["pairs"]):
            raise AssertionError("Native held-out key motifs must have length three")
        if audit_row < prefix_examples:
            scored = np.flatnonzero(valid[index])
            positions = scored[np.unique(np.linspace(0, len(scored) - 1,
                                                     min(prefixes_per_example, len(scored)), dtype=int))]
            for position in positions:
                predicted = fuzzy_prefix_prediction(inputs[index, :position + 1], overrides)
                expected = answers[index, position] if available[index, position] else IGNORE_INDEX
                if predicted != expected:
                    raise AssertionError("Native answer retrieval required future tokens")
                prefix_checked += 1
                unavailable_checked += int(expected == IGNORE_INDEX)
    return {"array_rows_checked": len(dataset), "native_alignment_and_masks_exact": True,
            "representative_rows_reparsed": len(representative),
            "representative_indices": representative.tolist(),
            "causal_prefix_predictions_checked": prefix_checked,
            "unavailable_prefixes_checked": unavailable_checked,
            "no_all_prefix_quadratic_audit": True,
            "generation_already_checks_native_annotation_for_every_example": True}


def _write_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "manifest.json")


def prepare(output_dir, *, length=400, train_examples=12800, dev_examples=1280,
            train_seed=2026091601, dev_seed=2026091602, confirmation_seed=2026091603,
            shuffle_seed=2026091604, representative_examples=32):
    spec = task_spec(FUZZY_TASK, {"seq_len": length})
    seeds = (train_seed, dev_seed, confirmation_seed, shuffle_seed)
    if (any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Train/dev/confirmation/shuffle seeds must be distinct uint32 integers")
    if min(train_examples, dev_examples, representative_examples) < 1:
        raise ValueError("Dataset and audit sizes must be positive")
    provenance = json.loads((VENDOR_ROOT / "CDRM_PROVENANCE.json").read_text())
    for name, record in provenance["files"].items():
        if file_sha256(VENDOR_ROOT / name) != record["sha256"]:
            raise ValueError(f"Pinned MAD source changed: {name}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite data directory {output}")
    frozen_sources = source_hashes()
    output.mkdir(parents=True)
    for name in frozen_sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    calendar = {"train": {"seed": train_seed, "examples": train_examples},
                "dev": {"seed": dev_seed, "examples": dev_examples},
                "confirmation": {"seed": confirmation_seed, "examples": dev_examples, "generated": False},
                "shuffle_seed": shuffle_seed}
    manifest = {"schema": SCHEMA, "status": "preparing",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task": FUZZY_TASK, "length": length, "task_spec": spec,
                "seed_calendar": calendar, "shuffle_seed": shuffle_seed,
                "generator_revision": REVISION, "generator_sha256": GENERATOR_SHA256,
                "source_sha256": frozen_sources, "splits": {},
                "native_arrays_unchanged": True, "native_input_token_offset": 0,
                "model_input_mapping": "Trainer adds 60; serialized input and target arrays remain native MAD IDs.",
                "native_training_labels_dense_including_padding": True,
                "native_evaluation_labels_masked": True,
                "labels_already_aligned_with_logits": True,
                "padding_attention_mask_added": False, "resampling_or_rejection": False,
                "final_split_generated": False, "training_or_model_execution": False,
                "epoch_order_algorithm": "numpy.random.default_rng(SeedSequence([shuffle_seed, zero_based_epoch])).permutation(N)"}
    _write_manifest(output, manifest)
    datasets = {}
    try:
        for split in ("train", "dev"):
            record = calendar[split]
            print(f"Generating native fuzzy T{length} {split}: {record['examples']} examples", flush=True)
            dataset = generate_dataset(FUZZY_TASK, split, record["seed"], record["examples"], {"seq_len": length})
            saved = save_dataset(output, dataset)
            loaded = load_dataset(output, FUZZY_TASK, split)
            if loaded.sha256 != dataset.sha256:
                raise AssertionError("Native array readback differs from generated arrays")
            print(f"Auditing saved {split} arrays and bounded representative causal prefixes", flush=True)
            split_record = {"examples": len(loaded), "seed": record["seed"], "length": length,
                            "dataset_sha256": loaded.sha256, "manifest_sha256": saved["sha256"],
                            "manifest": path_record(saved["path"]),
                            "native_scored_tokens": loaded.manifest["native_scored_tokens"],
                            "answer_scored_tokens": loaded.manifest["answer_scored_tokens"],
                            "oracle_coverage": loaded.manifest["oracle_coverage"],
                            "structural_audit": structural_audit(loaded, representative_examples=representative_examples)}
            if split == "dev":
                print("Auditing development retrieval coverage and query-ignoring shortcuts", flush=True)
                split_record["baselines"] = baseline_audit(loaded)
                metadata = evaluation_metadata(loaded)
                split_record["evaluation_regions"] = {name: int(mask.sum()) for name, mask in metadata["masks"].items()}
                split_record["answer_motifs"] = metadata["motifs"]
                split_record["distance_definition"] = metadata["distance_definition"]
            manifest["splits"][split] = split_record
            datasets[split] = loaded
            _write_manifest(output, manifest)
        manifest["overlap_audit"] = overlap_audit(datasets)
        if source_hashes() != frozen_sources:
            raise RuntimeError("Preparation sources changed during execution")
        if any(file_sha256(output / "source" / name) != digest for name, digest in frozen_sources.items()):
            raise RuntimeError("Frozen source snapshot differs from execution source")
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        saved_native_arrays_verified=True, source_snapshots_verified=True,
                        source_hashes_unchanged=True)
    except Exception as error:
        manifest.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _write_manifest(output, manifest)
    print(json.dumps({"status": "complete", "manifest": path_record(output / "manifest.json")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--train-examples", type=int, default=12800)
    parser.add_argument("--dev-examples", type=int, default=1280)
    parser.add_argument("--train-seed", type=int, default=2026091601)
    parser.add_argument("--dev-seed", type=int, default=2026091602)
    parser.add_argument("--confirmation-seed", type=int, default=2026091603)
    parser.add_argument("--shuffle-seed", type=int, default=2026091604)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd().resolve() != ROOT:
        parser.error("Run CPU preparation inside the project container working directory")
    prepare(**vars(args))


if __name__ == "__main__":
    main()
