#!/usr/bin/env python3
"""Freeze synthetic fixtures and audit the exact planned paired training stream.

Run explicitly in the CPU project container. This performs no model training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cdrm.synthetic.common import save_batch
from cdrm.synthetic.experiment import generate, generate_training_batch, task_spec, baselines, oracle


def example_hashes(batch):
    return [hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
            for x, y in zip(batch.input_ids, batch.labels)]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the explicit CPU project container and project working directory")
    plan = json.loads(args.plan.read_text())
    plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve frozen fixtures")
    args.output_dir.mkdir(parents=True)
    for task, settings in plan["tasks"].items():
        started = time.perf_counter()
        vocab = task_spec(task, settings["training_conditions"][0])["vocab_size"]
        manifest = {"schema": "stage-b-fixtures-v1", "task": task, "plan_sha256": plan_sha,
                    "vocab_size": vocab, "conditions": {},
                    "primary_dev_conditions": settings["primary_dev_conditions"], "audit": {}}
        heldout_hashes = {"dev": set(), "test": set()}
        for name, config in settings["evaluation_conditions"].items():
            assert task_spec(task, config)["vocab_size"] == vocab
            condition = {"config": config}
            for split in ("dev", "test"):
                count = plan["fixtures"][split + "_examples"]
                batch = generate(task, config, split, plan["fixtures"]["seed"], count)
                np.testing.assert_array_equal(oracle(task, config, batch), batch.labels)
                hashes = example_hashes(batch)
                if len(set(hashes)) != len(hashes):
                    raise AssertionError(f"Duplicate {task}/{name}/{split} examples")
                heldout_hashes[split].update(hashes)
                path = args.output_dir / task / f"{name}-{split}.npz"
                condition[split] = save_batch(path, batch)
                condition[split]["baselines"] = baselines(task, config, batch)
                condition[split]["oracle_exact"] = True
                print(f"fixture {task}/{name}/{split}: {count}, hash={batch.sha256[:12]}", flush=True)
            manifest["conditions"][name] = condition
        if heldout_hashes["dev"] & heldout_hashes["test"]:
            raise AssertionError(f"{task} dev/test overlap")
        training_hashes = set()
        stream = hashlib.sha256()
        targets = 0
        audit_batches = plan["fixtures"]["training_audit_batches"]
        if audit_batches != plan["training"]["updates"]:
            raise ValueError("Audit must cover the exact planned training horizon")
        for index in range(audit_batches):
            batch = generate_training_batch(plan, task, index)
            hashes = example_hashes(batch)
            if len(set(hashes)) != len(hashes) or training_hashes.intersection(hashes):
                raise AssertionError(f"Duplicate training example at {task}/{index}")
            for split in ("dev", "test"):
                if heldout_hashes[split].intersection(hashes):
                    raise AssertionError(f"{task} train/{split} overlap")
            training_hashes.update(hashes)
            stream.update(batch.sha256.encode())
            targets += int((batch.labels != -100).sum())
            if (index + 1) % 500 == 0:
                print(f"training audit {task}: {index + 1}/{audit_batches}", flush=True)
        manifest["audit"] = {"status": "passed", "training_batches": audit_batches,
                             "unique_training_examples": len(training_hashes),
                             "training_stream_sha256": stream.hexdigest(),
                             "training_input_tokens": len(training_hashes) * 128,
                             "training_target_tokens": targets,
                             "exact_example_overlap_train_dev_test": 0,
                             "duplicates_within_each_split_condition": 0,
                             "condition_note": "Controlled length variants deliberately reuse semantic examples within a split; no exact input/label example crosses splits.",
                             "seconds": time.perf_counter() - started}
        write(args.output_dir / task / "manifest.json", manifest)
        print(f"prepared {task}: {manifest['audit']}", flush=True)
    write(args.output_dir / "manifest.json", {"schema": "stage-b-fixture-collection-v1",
           "plan_sha256": plan_sha, "tasks": list(plan["tasks"]), "status": "passed"})


if __name__ == "__main__":
    main()
