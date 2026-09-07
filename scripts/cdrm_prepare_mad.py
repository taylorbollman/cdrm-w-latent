#!/usr/bin/env python3
"""Freeze official MAD recall/selective-copy corpora and independent integrity audits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.mad_data import (
    GENERATOR_SHA256, IGNORE_INDEX, REVISION, SHUFFLE_SEED, SPLIT_SEEDS, SPLIT_SIZES,
    SCREENING_SPLIT_SEEDS, SETTINGS, setting_spec,
    TASKS, VENDOR_ROOT, answer_oracle, baseline_audit, epoch_indices, file_sha256,
    generate_dataset, load_dataset, overlap_audit, recall_prefix_prediction, save_dataset,
)


def structural_audit(dataset, limit=64):
    """Deterministic held-out integrity checks; these do not run or select a model."""
    task = dataset.manifest["task"]
    overrides = dataset.manifest.get("task_overrides")
    vocab = int(dataset.manifest["vocab_size"])
    half = vocab // 2
    all_oracle = answer_oracle(task, dataset.input_ids, overrides)
    if not np.array_equal(all_oracle, dataset.answer_labels):
        raise AssertionError("Oracle mismatch")
    if dataset.manifest["split"] != "train" or task == "selective-copying":
        if not np.array_equal(dataset.labels, dataset.answer_labels):
            raise AssertionError("Native held-out masks differ from oracle")
    elif (not np.array_equal(dataset.labels[:, :-1], dataset.input_ids[:, 1:]) or
          not np.array_equal(dataset.labels[:, -1], dataset.answer_labels[:, -1]) or
          np.any(dataset.labels == IGNORE_INDEX)):
        raise AssertionError("Native dense autoregressive training alignment changed")
    prefixes, counterfactuals = 0, 0
    for tokens, labels in zip(dataset.input_ids[:limit], dataset.answer_labels[:limit]):
        if task == "in-context-recall":
            for position in np.flatnonzero(labels != IGNORE_INDEX):
                if recall_prefix_prediction(tokens[:position + 1], vocab) != labels[position]:
                    raise AssertionError("Oracle required future information")
                prefixes += 1
            # Change all values through a bijection; keys/query positions stay fixed.
            altered = tokens.copy()
            value_mask = altered >= half
            altered[value_mask] = half + (altered[value_mask] - half + 1) % half
            expected = labels.copy()
            valid = labels != IGNORE_INDEX
            expected[valid] = half + (labels[valid] - half + 1) % half
            if not np.array_equal(answer_oracle(task, altered, overrides), expected):
                raise AssertionError("Value counterfactual did not propagate")
            counterfactuals += 1
            # Terminal-query intervention using an earlier key with a different value.
            mapping = dict(zip(map(int, tokens[:-1:2]), map(int, tokens[1::2])))
            alternative = next((key for key, value in mapping.items() if value != labels[-1]), None)
            if alternative is not None:
                altered = tokens.copy()
                altered[-1] = alternative
                result = answer_oracle(task, altered, overrides)
                if result[-1] != mapping[alternative] or not np.array_equal(result[:-1], labels[:-1]):
                    raise AssertionError("Query counterfactual failed")
                counterfactuals += 1
        else:
            marker = int(np.flatnonzero(tokens == vocab - 1)[0])
            source_positions = np.flatnonzero(tokens[:marker] != vocab - 2)
            if not np.array_equal(tokens[source_positions], labels[marker + 1:]):
                raise AssertionError("Ordered copy source differs from targets")
            prefixes += len(source_positions)
            altered = tokens.copy()
            altered[source_positions[0]] = (altered[source_positions[0]] + 1) % (vocab - 2)
            result = answer_oracle(task, altered, overrides)
            expected = labels.copy()
            expected[marker + 1] = altered[source_positions[0]]
            if not np.array_equal(result, expected):
                raise AssertionError("Selective-copy counterfactual failed")
            counterfactuals += 1
    return {"examples_checked_against_oracle": len(dataset),
            "oracle_exact": True, "native_alignment_and_masks_exact": True,
            "prefix_only_predictions_checked": prefixes,
            "counterfactuals_checked": counterfactuals,
            "counterfactual_examples": min(limit, len(dataset)),
            "future_tokens_used_to_predict_answers": False,
            "native_dense_training_targets_are_not_claimed_causally_predictable": True}


def prepare(output_dir, tasks=TASKS, sizes=None, *, setting=None, task_overrides=None,
            splits=("train", "dev", "final"), split_seeds=None, shuffle_seed=SHUFFLE_SEED,
            append_splits=False):
    output = Path(output_dir)
    selected_setting = setting_spec(setting) if setting is not None else None
    if selected_setting:
        if task_overrides:
            raise ValueError("Use a named official setting or explicit overrides, not both")
        tasks = (selected_setting["task"],)
        task_overrides = {selected_setting["task"]: selected_setting["overrides"]}
    task_overrides = task_overrides or {}
    split_seeds = dict(split_seeds or (SCREENING_SPLIT_SEEDS if setting else SPLIT_SEEDS))
    if len(splits) != len(set(splits)) or not splits or any(split not in SPLIT_SIZES for split in splits):
        raise ValueError("Choose unique train/dev/final splits")
    if len(set(split_seeds.values())) != len(split_seeds) or any(not 0 <= v < 2**32 for v in split_seeds.values()):
        raise ValueError("Split seeds must be distinct uint32 values")
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not append_splits:
        raise FileExistsError(f"Refusing to overwrite data directory {output}; additions require --append-splits")
    parent_manifest = None
    if append_splits:
        parent_path = output / "manifest.json"
        if not parent_path.is_file():
            raise ValueError("Split additions require an existing manifest.json")
        parent_manifest = json.loads(parent_path.read_text())
        if (parent_manifest.get("setting") != selected_setting or
            parent_manifest.get("task_overrides", {}) != task_overrides or
            parent_manifest["split_seeds"] != split_seeds or
            parent_manifest["epoch_shuffle"]["seed"] != shuffle_seed or
            parent_manifest["adapter_sha256"] != file_sha256(ROOT / "cdrm/mad_data.py") or
            parent_manifest["preparation_script_sha256"] != file_sha256(Path(__file__))):
            raise ValueError("Split addition changed the frozen setting/seeds/source identity")
        if set(parent_manifest["tasks"]) != set(tasks):
            raise ValueError("Split addition must retain the same tasks")
        for task in tasks:
            for split in splits:
                if any((output / task / f"{split}{suffix}").exists() for suffix in (".npz", ".metadata.json", ".manifest.json")):
                    raise FileExistsError(f"Refusing to overwrite existing split {task}/{split}")
    sizes = SPLIT_SIZES if sizes is None else sizes
    provenance = json.loads((VENDOR_ROOT / "CDRM_PROVENANCE.json").read_text())
    for relative, record in provenance["files"].items():
        if file_sha256(VENDOR_ROOT / relative) != record["sha256"]:
            raise RuntimeError(f"Pinned source mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator_revision": REVISION, "generator_sha256": GENERATOR_SHA256,
                "numpy_version": np.__version__,
                "source_provenance": provenance,
                "adapter_sha256": file_sha256(ROOT / "cdrm/mad_data.py"),
                "preparation_script_sha256": file_sha256(Path(__file__)),
                "tests_sha256": file_sha256(ROOT / "tests/test_mad_data.py"),
                "split_sizes": sizes, "split_seeds": split_seeds,
                "epoch_shuffle": {"seed": shuffle_seed, "epoch_indexing": "zero-based",
                                  "algorithm": "default_rng(SeedSequence([seed, epoch])).permutation(size)"},
                "native_generator_executed_unmodified": True,
                "overlap_handling": "Audit and retain all native draws; never resample to manufacture novelty.",
                "final_split_access_policy": "Integrity oracles and fixed task baselines only until model endpoint is selected.",
                "tasks": {}}
    if selected_setting is not None:
        manifest["setting"] = selected_setting
    if task_overrides:
        manifest["task_overrides"] = task_overrides
    if parent_manifest is not None:
        manifest["parent_manifest"] = {"path": "manifest.json", "sha256": file_sha256(output / "manifest.json")}
        if parent_manifest["split_sizes"] != sizes:
            raise ValueError("Split addition changed frozen split sizes")
    for task in tasks:
        datasets, task_record = {}, {"splits": {}}
        if append_splits:
            for split in SPLIT_SIZES:
                if (output / task / f"{split}.manifest.json").is_file():
                    datasets[split] = load_dataset(output, task, split)
                    if split not in parent_manifest["tasks"][task]["splits"]:
                        raise ValueError("Found an existing split absent from parent manifest")
                    recorded = parent_manifest["tasks"][task]["splits"][split]
                    if file_sha256(output / recorded["path"]) != recorded["sha256"]:
                        raise ValueError("Existing split manifest changed from the frozen parent")
            task_record["splits"].update(parent_manifest["tasks"][task]["splits"])
        for split in splits:
            count = sizes[split]
            print(f"Generating {task} {split}: {count} examples", flush=True)
            dataset = generate_dataset(task, split, split_seeds[split], count, task_overrides.get(task))
            audit = structural_audit(dataset)
            baselines = baseline_audit(dataset)
            record = save_dataset(output, dataset)
            loaded = load_dataset(output, task, split)
            if dataset.sha256 != loaded.sha256:
                raise AssertionError("Saved data did not read back exactly")
            record["path"] = str(Path(record["path"]).relative_to(output))
            task_record["splits"][split] = {**record, "shape": list(dataset.input_ids.shape),
                                             "integrity": audit, "answer_baselines": baselines}
            datasets[split] = dataset
        task_record["overlap_audit"] = overlap_audit(datasets)
        if not task_record["overlap_audit"]["no_exact_input_cross_split_overlap"]:
            print(f"NOTICE: {task} exact cross-split input overlap retained and reported", flush=True)
        if append_splits and "epoch_indices" in parent_manifest["tasks"][task]:
            task_record["epoch_indices"] = parent_manifest["tasks"][task]["epoch_indices"]
            info = task_record["epoch_indices"]
            if file_sha256(output / info["path"]) != info["sha256"]:
                raise ValueError("Frozen epoch indices changed")
        elif "train" in datasets:
            train_size = len(datasets["train"])
            permutations = np.stack([epoch_indices(train_size, epoch, shuffle_seed) for epoch in range(200)])
            shuffle_path = output / task / "epoch-indices.npy"
            if shuffle_path.exists():
                raise FileExistsError(shuffle_path)
            np.save(shuffle_path, permutations, allow_pickle=False)
            task_record["epoch_indices"] = {"path": str(shuffle_path.relative_to(output)),
                                              "sha256": file_sha256(shuffle_path),
                                              "shape": list(permutations.shape)}
        manifest["tasks"][task] = task_record
    path = output / ("manifest.added-" + "-".join(splits) + ".json" if append_splits else "manifest.json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "sha256": file_sha256(path), "tasks": list(tasks)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--setting", choices=tuple(SETTINGS))
    parser.add_argument("--task-overrides", type=Path, help="JSON mapping canonical task names to dimension overrides")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_SIZES), default=list(SPLIT_SIZES))
    parser.add_argument("--append-splits", action="store_true", help="Add new splits and a supplemental manifest; never overwrite existing data")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--dev-seed", type=int)
    parser.add_argument("--final-seed", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run from the explicit CPU project container")
    seeds = dict(SCREENING_SPLIT_SEEDS if args.setting else SPLIT_SEEDS)
    for split in SPLIT_SEEDS:
        value = getattr(args, split + "_seed")
        if value is not None:
            seeds[split] = value
    overrides = json.loads(args.task_overrides.read_text()) if args.task_overrides else None
    prepare(args.output_dir, args.tasks, setting=args.setting, task_overrides=overrides,
            splits=args.splits, split_seeds=seeds, shuffle_seed=args.shuffle_seed,
            append_splits=args.append_splits)


if __name__ == "__main__":
    main()
