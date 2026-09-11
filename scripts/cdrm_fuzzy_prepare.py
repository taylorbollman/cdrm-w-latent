#!/usr/bin/env python3
"""CPU-only native fuzzy-recall calibration/numerical preparation; no final split."""
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
    answer_oracle, baseline_audit, epoch_indices, file_sha256,
    fuzzy_answer_annotation, fuzzy_prefix_prediction, generate_dataset,
    load_dataset, overlap_audit, save_dataset, task_spec,
)

NUMERICAL_LENGTHS = (17, 32, 128, 256, 300)
SCHEMA = "cdrm-fuzzy-preparation-v1"


def seed_calendar(role, length):
    if role == "calibration" and length == 256:
        return {"length": 256, "train": {"seed": 961001, "examples": 12800},
                "dev": {"seed": 961002, "examples": 1280}, "shuffle_seed": 961003}
    if role == "numerical" and length in NUMERICAL_LENGTHS:
        return {"length": length, "train": {"seed": 962000 + 2 * length, "examples": 128},
                "dev": {"seed": 962001 + 2 * length, "examples": 128}, "shuffle_seed": 963000 + length}
    raise ValueError("Require calibration T256 or a declared numerical length; final data is unsupported")


def path_record(path):
    path = Path(path)
    return {"path": project_path(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def project_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sources():
    names = ["cdrm/mad_data.py", "scripts/cdrm_fuzzy_prepare.py", "tests/test_mad_data.py",
             "tests/test_cdrm_fuzzy_prepare.py", "vendors/mad-lab/CDRM_PROVENANCE.json",
             "vendors/mad-lab/mad/data/instances.py", "vendors/mad-lab/mad/configs.py",
             "vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml"]
    return {name: file_sha256(ROOT / name) for name in names}


def preflight(protocol, protocol_sha256, role, length):
    protocol = Path(protocol)
    if file_sha256(protocol) != protocol_sha256:
        raise ValueError("Prospective protocol checksum mismatch")
    document = json.loads(protocol.read_text())
    if document.get("schema") != "cdrm-fuzzy-prospective-protocol-v1":
        raise ValueError("Require the prospective fuzzy protocol schema")
    data = document["data"]
    expected = {"task": FUZZY_TASK, "vocab_size": 16, "multi_query": True,
                "k_motif_size": 3, "v_motif_size": 3,
                "native_labels_unchanged": True, "no_rejection_resampling": True}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("Protocol changed native fuzzy semantics")
    calendar = seed_calendar(role, length)
    declared = data[role] if role == "calibration" else data[role][str(length)]
    if declared != calendar or document["schedule"]["epochs"] != 50:
        raise ValueError("Protocol differs from the prospective seed/size/shuffle calendar")
    provenance = json.loads((VENDOR_ROOT / "CDRM_PROVENANCE.json").read_text())
    for name, record in provenance["files"].items():
        if file_sha256(VENDOR_ROOT / name) != record["sha256"]:
            raise ValueError(f"Pinned native MAD source changed: {name}")
    return calendar, sources()


def structural_audit(dataset, prefix_limit=64):
    """Keep annotation/mask checks separate from prediction using causal prefixes."""
    if dataset.manifest["task"] != FUZZY_TASK:
        raise ValueError("Fuzzy audit only accepts native fuzzy-recall data")
    overrides = dataset.manifest.get("task_overrides")
    available = dataset.oracle_available_mask
    counts = {"prefix_predictions_checked": 0, "unavailable_prefixes_checked": 0,
              "counterfactual_examples_checked": 0}
    traces = []
    key_lengths, value_lengths = {}, {}
    for index, (tokens, native, answers) in enumerate(zip(dataset.input_ids, dataset.labels, dataset.answer_labels)):
        annotated, info = fuzzy_answer_annotation(tokens, overrides, terminal_target=int(native[-1]))
        oracle = answer_oracle(FUZZY_TASK, tokens, overrides)
        if (not np.array_equal(annotated, answers) or
                not np.array_equal(oracle != IGNORE_INDEX, available[index]) or
                not np.array_equal(oracle[available[index]], answers[available[index]])):
            raise AssertionError("Fuzzy annotation/retrieval availability mismatch")
        if dataset.manifest["split"] == "train":
            if np.any(native == IGNORE_INDEX) or not np.array_equal(native[:-1], tokens[1:]) or native[-1] != answers[-1]:
                raise AssertionError("Native dense fuzzy targets or shift changed")
        elif not np.array_equal(native, answers):
            raise AssertionError("Native held-out fuzzy masks changed")
        if np.any(answers == 15) or np.any(answers[:max(info["left_padding"] - 1, 0)] != IGNORE_INDEX):
            raise AssertionError("Padding was scored as a retrieval answer")
        for pair in info["pairs"]:
            k, v = len(pair["key"]), len(pair["value"])
            key_lengths[str(k)] = key_lengths.get(str(k), 0) + 1
            value_lengths[str(v)] = value_lengths.get(str(v), 0) + 1
            if dataset.manifest["split"] != "train" and k != 3:
                raise AssertionError("Held-out fuzzy keys must have maximum motif length3")
        if index < prefix_limit:
            for position in np.flatnonzero(answers != IGNORE_INDEX):
                prediction = fuzzy_prefix_prediction(tokens[:position + 1], overrides)
                if available[index, position]:
                    if prediction != answers[position]:
                        raise AssertionError("Scored fuzzy retrieval required future tokens")
                    counts["prefix_predictions_checked"] += 1
                else:
                    if prediction != IGNORE_INDEX:
                        raise AssertionError("Unseen terminal mapping unexpectedly became retrievable")
                    counts["unavailable_prefixes_checked"] += 1
            altered = tokens.copy()
            value_positions = (altered >= 7) & (altered < 15)
            altered[value_positions] = 7 + (altered[value_positions] - 7 + 1) % 8
            changed = answer_oracle(FUZZY_TASK, altered, overrides)
            expected = oracle.copy()
            expected[available[index]] = 7 + (expected[available[index]] - 7 + 1) % 8
            if not np.array_equal(changed, expected):
                raise AssertionError("Consistent value relabeling did not change retrieved answers")
            counts["counterfactual_examples_checked"] += 1
        terminal = info["pairs"][-1]
        if not terminal["repeated"] and len(traces) < 3:
            traces.append({"example_index": index, "input_ids": tokens.tolist(),
                           "native_terminal_target": int(native[-1]), "terminal_pair": terminal,
                           "earlier_matching_keys": [],
                           "unavailable_scored_positions": np.flatnonzero(
                               (answers != IGNORE_INDEX) & ~available[index]).tolist()})
    return {"examples_checked": len(dataset), "native_alignment_and_masks_exact": True,
            "retrieval_available_mask_exact": True, **counts,
            "key_motif_length_counts": key_lengths, "value_motif_length_counts": value_lengths,
            "oracle_coverage": dataset.manifest["oracle_coverage"],
            "unseen_terminal_examples": traces,
            "native_edge_case_source": {"path": "vendors/mad-lab/mad/data/instances.py",
                "sha256": GENERATOR_SHA256, "probe_index_line": 258,
                "loop_termination_line": 267, "insertion_lines": [274, 281], "terminal_append_lines": [323, 332]},
            "oracle_claim": "Predictions use causal prefixes conditional on native answer positions. Run-based mask annotation may inspect input boundaries; unknown mappings abstain. No hard accuracy ceiling is inferred."}


def prepare(output_dir, *, protocol, protocol_sha256, role, length, prefix_limit=64):
    calendar, frozen_sources = preflight(protocol, protocol_sha256, role, length)
    output = Path(output_dir).resolve()
    parts = output.parts
    if ".runtime" in parts and parts[parts.index(".runtime") + 1] != "cdrm-150m-fuzzy-recall":
        raise ValueError("Refusing to write inside another retained runtime lineage")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data directory {output}")
    output.mkdir(parents=True)
    source_root = output / "source"
    for name in frozen_sources:
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(protocol, source_root / "protocol.json")
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "preparing", "role": role, "task": FUZZY_TASK,
                "data_root": project_path(output), "length": length,
                "actual_sequence_length": length, "configured_sequence_length": length,
                "task_spec": task_spec(FUZZY_TASK, {"seq_len": length}),
                "protocol": path_record(protocol), "seed_calendar": calendar,
                "shuffle_seed": calendar["shuffle_seed"], "source_sha256": frozen_sources,
                "generator_revision": REVISION, "generator_sha256": GENERATOR_SHA256,
                "splits": {}, "training_or_model_execution": False,
                "final_split_generated": False, "resampling_or_rejection": False}
    datasets = {}
    try:
        for split in ("train", "dev"):
            declaration = calendar[split]
            print(f"Generating native fuzzy {role} T{length} {split}: {declaration['examples']} examples", flush=True)
            dataset = generate_dataset(FUZZY_TASK, split, declaration["seed"], declaration["examples"], {"seq_len": length})
            native = save_dataset(output, dataset)
            reloaded = load_dataset(output, FUZZY_TASK, split)
            if reloaded.sha256 != dataset.sha256:
                raise AssertionError("Prepared native arrays did not read back exactly")
            print(f"Auditing native masks, causal retrieval and shortcuts for {split}", flush=True)
            manifest["splits"][split] = {
                "dataset_sha256": reloaded.sha256, "manifest_sha256": native["sha256"],
                "manifest_path": project_path(native["path"]), "examples": len(dataset),
                "length": length, "seed": declaration["seed"],
                "native_scored_tokens": dataset.manifest["native_scored_tokens"],
                "answer_scored_tokens": dataset.manifest["answer_scored_tokens"],
                "integrity": structural_audit(reloaded, prefix_limit), "baselines": baseline_audit(reloaded)}
            datasets[split] = reloaded
        order = np.stack([epoch_indices(len(datasets["train"]), epoch, calendar["shuffle_seed"]) for epoch in range(50)])
        order_path = output / FUZZY_TASK / "epoch-indices.npy"
        np.save(order_path, order, allow_pickle=False)
        manifest["epoch_indices"] = {**path_record(order_path), "shape": list(order.shape),
                                     "seed": calendar["shuffle_seed"], "epochs": 50,
                                     "algorithm": "default_rng(SeedSequence([seed, zero_based_epoch])).permutation(N)"}
        manifest["overlap_audit"] = overlap_audit(datasets)
        if sources() != frozen_sources or file_sha256(protocol) != protocol_sha256:
            raise RuntimeError("Preparation source/protocol changed during execution")
        for name, digest in frozen_sources.items():
            if file_sha256(source_root / name) != digest:
                raise RuntimeError("Preparation source snapshot differs from frozen source")
        manifest.update(status="complete", source_hashes_unchanged=True,
                        source_snapshots_verified=True, saved_native_arrays_verified=True)
    except Exception as error:
        manifest.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "complete", "manifest": path_record(output / "manifest.json")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("calibration", "numerical"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd().resolve() != ROOT:
        parser.error("Run CPU preparation inside the project container working directory")
    prepare(args.output_dir, protocol=args.protocol, protocol_sha256=args.protocol_sha256,
            role=args.role, length=args.length)


if __name__ == "__main__":
    main()
