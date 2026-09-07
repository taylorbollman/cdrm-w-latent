#!/usr/bin/env python3
"""Archive the audited paired stream and check input/mapping split separation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cdrm.synthetic.common import load_batch, save_batch
from cdrm.synthetic.experiment import generate_training_batch, task_spec


def input_hashes(batch):
    """Hash model-visible inputs without relying on their target masks."""
    return [hashlib.sha256(row.astype("<i8", copy=False).tobytes()).hexdigest()
            for row in batch.input_ids]


def mapping_hashes(batch, task, config):
    """Reconstruct full per-example recall maps and verify retained metadata IDs.

    State tracking uses fixed operation permutations, deliberately shared across
    splits, so it has no retrieval-map holdout audit.
    """
    if task == "state_tracking":
        return []
    spec = task_spec(task, config)
    stop = 2 * spec["num_kv_pairs"] if task == "mqar" else batch.input_ids.shape[1] - 2
    hashes = []
    for ids, metadata in zip(batch.input_ids, batch.metadata):
        mapping = {}
        for pos in range(0, stop, 2):
            key, value = map(int, ids[pos:pos + 2])
            if spec["key_token_start"] <= key < spec["key_token_end"]:
                if key in mapping and mapping[key] != value:
                    raise AssertionError("A retrieval key has conflicting observed values")
                mapping[key] = value
        digest = hashlib.sha256(json.dumps(sorted(mapping.items())).encode()).hexdigest()
        if digest != metadata["mapping_id"]:
            raise AssertionError("Retained mapping metadata disagrees with visible input records")
        hashes.append(digest)
    return hashes


def check_cross_split(task, input_sets, mapping_sets):
    """Reject complete input/map overlap; allow controlled reuse within a split."""
    report = {}
    names = list(input_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            input_overlap = len(input_sets[left] & input_sets[right])
            mapping_overlap = (len(mapping_sets[left] & mapping_sets[right])
                               if task != "state_tracking" else None)
            report[f"{left}:{right}"] = {"exact_input_overlap": input_overlap,
                                        "complete_retrieval_mapping_overlap": mapping_overlap}
            if input_overlap:
                raise AssertionError(f"{task} input-only overlap between {left} and {right}: {input_overlap}")
            if mapping_overlap:
                raise AssertionError(f"{task} complete retrieval-map overlap between {left} and {right}: {mapping_overlap}")
    return report


def augment_state_chance(manifest):
    """Add comparable theoretical chance metrics without changing generators."""
    for condition in manifest["conditions"].values():
        for split in ("dev", "test"):
            baseline = condition[split]["baselines"]
            baseline.update(chance_task_class_accuracy=1 / 6,
                            chance_task_class_cross_entropy=float(np.log(6)),
                            chance_full_vocab_accuracy=1 / manifest["vocab_size"],
                            chance_full_vocab_cross_entropy=float(np.log(manifest["vocab_size"])),
                            chance_definition="Uniform over six legal state tokens; full-vocabulary chance is separate")


def source_audit(project_root):
    sources = ["cdrm/synthetic/common.py", "cdrm/synthetic/experiment.py",
               "cdrm/synthetic/retrieval.py", "cdrm/synthetic/state_tracking.py",
               "scripts/stage_b_archive_training.py"]
    hashes = {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
              for name in sources}
    upstream = {}
    for name in ("zoology", "mad-lab"):
        root = project_root / "vendors" / name
        provenance = json.loads((root / "PROVENANCE.json").read_text())
        for filename, info in provenance["files"].items():
            actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
            if actual != info["sha256"]:
                raise AssertionError(f"Pinned upstream source changed: {name}/{filename}")
        upstream[name] = {"revision": provenance["revision"], "source_hashes_verified": True,
                          "provenance_sha256": hashlib.sha256((root / "PROVENANCE.json").read_bytes()).hexdigest()}
    return {"source_sha256_at_archive": hashes, "upstream": upstream,
            "claim": "Current sources regenerate the previously audited exact training-stream hash; source hashes were captured at archive time."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the explicit CPU project container")
    plan = json.loads(args.plan.read_text())
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    collection_path = args.fixtures_dir / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if collection["plan_sha256"] != plan_hash or collection["status"] != "passed":
        raise ValueError("Complete fixture preparation for this exact plan before archiving")
    # Check every task before writing any archive; this preserves existing files.
    for task in plan["tasks"]:
        if (args.fixtures_dir / task / "training.npz").exists():
            raise FileExistsError(args.fixtures_dir / task / "training.npz")
    sources = source_audit(Path.cwd())
    completed = {}
    for task, settings in plan["tasks"].items():
        manifest_path = args.fixtures_dir / task / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["plan_sha256"] != plan_hash or manifest["audit"]["status"] != "passed":
            raise ValueError("Freeze and audit this exact plan before archiving its training stream")
        if manifest["audit"]["training_batches"] != plan["training"]["updates"]:
            raise ValueError("Prepared training horizon differs from the plan")
        path = args.fixtures_dir / task / "training.npz"
        input_sets = {split: set() for split in ("train", "dev", "test")}
        mapping_sets = {split: set() for split in ("train", "dev", "test")}
        condition_audits = {}
        started = time.perf_counter()
        for name, condition in manifest["conditions"].items():
            condition_audits[name] = {}
            for split in ("dev", "test"):
                batch = load_batch(args.fixtures_dir / task / f"{name}-{split}.npz")
                if batch.sha256 != condition[split]["sha256"]:
                    raise AssertionError(f"Retained {task}/{name}/{split} arrays differ from their frozen manifest")
                fingerprints = input_hashes(batch)
                unique_inputs = set(fingerprints)
                if len(unique_inputs) != len(fingerprints):
                    raise AssertionError(f"Duplicate input-only example inside {task}/{name}/{split}")
                mappings = mapping_hashes(batch, task, condition["config"])
                input_sets[split].update(unique_inputs)
                mapping_sets[split].update(mappings)
                condition_audits[name][split] = {
                    "examples": len(fingerprints), "unique_inputs": len(unique_inputs),
                    "unique_complete_retrieval_mappings": len(set(mappings)) if mappings else None,
                    "array_manifest_hash_verified": True,
                    "mapping_metadata_verified_from_inputs": bool(mappings)}
        # Check dev/test before doing the more expensive stream regeneration.
        check_cross_split(task, input_sets, mapping_sets)
        inputs, labels, hashes = [], [], hashlib.sha256()
        conditions = settings["training_conditions"]
        batch_size = plan["training"]["global_batch"]
        per_condition = batch_size // len(conditions)
        if per_condition * len(conditions) != batch_size:
            raise ValueError("Training conditions do not divide the global batch")
        for index in range(plan["training"]["updates"]):
            batch = generate_training_batch(plan, task, index)
            fingerprints = input_hashes(batch)
            if len(set(fingerprints)) != len(fingerprints) or input_sets["train"].intersection(fingerprints):
                raise AssertionError(f"Duplicate input-only training example at {task}/{index}")
            input_sets["train"].update(fingerprints)
            for condition_index, config in enumerate(conditions):
                segment = batch.take(slice(condition_index * per_condition, (condition_index + 1) * per_condition))
                mapping_sets["train"].update(mapping_hashes(segment, task, config))
            # Fail at the first offending batch; never archive a leaking stream.
            check_cross_split(task, input_sets, mapping_sets)
            inputs.append(batch.input_ids)
            labels.append(batch.labels)
            hashes.update(batch.sha256.encode())
            if (index + 1) % 500 == 0:
                print(f"archive audit {task}: {index + 1}/{plan['training']['updates']}", flush=True)
        if hashes.hexdigest() != manifest["audit"]["training_stream_sha256"]:
            raise AssertionError("Regenerated training stream differs from the audited stream")
        split_audit = {"status": "passed", "input_hash_contract": "SHA-256 of little-endian int64 input IDs only",
            "retrieval_mapping_hash_contract": "SHA-256 of sorted complete observed key/value mapping; checked against metadata",
            "conditions": condition_audits, "between_splits": check_cross_split(task, input_sets, mapping_sets),
            "unique_inputs_by_split": {split: len(values) for split, values in input_sets.items()},
            "unique_complete_retrieval_mappings_by_split": (
                {split: len(values) for split, values in mapping_sets.items()} if task != "state_tracking" else None),
            "mapping_claim": ("All complete observed retrieval mappings are disjoint across train/dev/test. Individual pairs and symbols may be shared."
                              if task != "state_tracking" else "Not applicable: fixed operation permutations are shared; composition condition withholds ordered token patterns."),
            "within_split_note": "Controlled delay conditions intentionally reuse semantic mappings within each evaluation split; this is allowed and not counted as independent evidence."}
        ids, targets = np.concatenate(inputs), np.concatenate(labels)
        with path.with_suffix(".npz.partial").open("wb") as handle:
            np.savez_compressed(handle, input_ids=ids, labels=targets)
        path.with_suffix(".npz.partial").replace(path)
        manifest["training"] = {"path": str(path), "examples": len(ids),
                                "global_batch": batch_size,
                                "training_stream_sha256": hashes.hexdigest(),
                                "sha256_file": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "input_tokens": int(ids.size),
                                "target_tokens": int((targets != -100).sum()),
                                "row_order": "completed_update * global_batch + within-batch index; balanced condition chunks",
                                "metadata": "Regenerable from pinned generator, exact plan, split=train, update index; arrays retained directly."}
        manifest["audit"]["input_and_mapping_separation"] = split_audit
        manifest["archive_source_audit"] = sources
        if task == "state_tracking":
            augment_state_chance(manifest)
        calibration = generate_training_batch(plan, task, 0, split="calibration")
        manifest["calibration"] = save_batch(args.fixtures_dir / task / "calibration.npz", calibration)
        temporary_manifest = manifest_path.with_suffix(".json.partial")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)
        completed[task] = {"training_stream_sha256": hashes.hexdigest(),
                           "training_archive_sha256": manifest["training"]["sha256_file"],
                           "input_and_mapping_separation": "passed"}
        print(f"archived {task}: {len(ids)} examples; {path.stat().st_size / 2**20:.2f} MiB; "
              f"verified audited stream and split separation; {time.perf_counter() - started:.1f}s", flush=True)
    collection["training_archives"] = completed
    collection["archive_source_audit"] = sources
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
