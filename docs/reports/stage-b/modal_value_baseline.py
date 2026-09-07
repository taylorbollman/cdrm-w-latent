#!/usr/bin/env python3
"""Posthoc query-ignoring modal-value baseline; no training or fixture mutations.

MAD samples its terminal query uniformly among DISTINCT observed keys. Given
only the preceding mapping and no query key, the Bayes-optimal constant answer
is therefore the value assigned to the largest number of distinct keys. Ties
are resolved by the smallest value token ID, fixed before this computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cdrm.synthetic.common import load_batch
from cdrm.synthetic.retrieval import MAD_REVISION, ZOOLOGY_REVISION, task_spec


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modal_prediction(ids, task, config):
    """Uses only complete context records; never reads query tokens or labels."""
    spec = task_spec(task, config)
    stop = 2 * spec["num_kv_pairs"] if task == "mqar" else len(ids) - 2
    mapping = {}
    for position in range(0, stop, 2):
        key, value = map(int, ids[position:position + 2])
        if spec["key_token_start"] <= key < spec["key_token_end"]:
            if value not in spec["answer_token_ids"]:
                raise AssertionError("A parsed record value is outside the task vocabulary")
            if key in mapping and mapping[key] != value:
                raise AssertionError("A visible key has contradictory records")
            mapping[key] = value
    if not mapping:
        raise AssertionError("The visible context must contain at least one association")
    frequencies = Counter(mapping.values())
    largest = max(frequencies.values())
    prediction = min(value for value, count in frequencies.items() if count == largest)
    return {"prediction": prediction, "expected_accuracy": largest / len(mapping),
            "num_distinct_keys": len(mapping), "num_distinct_values": len(frequencies),
            "modal_value_multiplicity": largest,
            "tied_modal_values": sum(count == largest for count in frequencies.values())}


def verify_information_restriction():
    # Repeated records do not count as distinct keys. Key1's frequent value32
    # loses to value33 assigned to key2 and key3. Terminal copy/query excluded.
    ids = np.array([1, 32, 1, 32, 1, 32, 2, 33, 3, 33, 80, 1], dtype=np.int64)
    predicted = modal_prediction(ids, "noisy_recall", {"sequence_length": 12})
    assert predicted["prediction"] == 33 and predicted["expected_accuracy"] == 2 / 3
    changed_query = ids.copy()
    changed_query[-2:] = [79, 31]
    assert modal_prediction(changed_query, "noisy_recall", {"sequence_length": 12}) == predicted
    tied = np.array([1, 33, 2, 32, 64, 70, 80, 2], dtype=np.int64)
    assert modal_prediction(tied, "noisy_recall", {"sequence_length": 8})["prediction"] == 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run from the project directory in the explicit CPU container")
    if args.output_json.exists() or args.output_markdown.exists():
        raise FileExistsError("Preserve existing supplemental results; use new output paths")
    verify_information_restriction()
    plan = json.loads(args.plan.read_text())
    report = {"schema": "stage-b-posthoc-modal-value-v1", "analysis_status": "posthoc",
        "selection_disclosure": "Requested after inspecting this pilot's noisy-recall model metrics; not a prespecified primary baseline. No model, generator, fixture or training changes and no baseline tuning.",
        "plan_sha256": file_hash(args.plan), "script_sha256": file_hash(__file__),
        "baseline": "Deduplicate complete visible key/value records; choose most frequent value over distinct keys, breaking ties by smallest value ID. Ignore terminal query and all scored-answer labels when predicting.",
        "conditional_expectation": "max_v count(distinct keys mapping to v) / count(distinct observed keys)",
        "source_justification": {
            "mad": {"revision": MAD_REVISION,
                    "url": f"https://github.com/athms/mad-lab/blob/{MAD_REVISION}/mad/data/instances.py",
                    "terminal_query": "rng.choice(list(keys_presented.keys())): uniform among distinct observed keys"},
            "mqar": {"revision": ZOOLOGY_REVISION,
                     "url": f"https://github.com/HazyResearch/zoology/blob/{ZOOLOGY_REVISION}/zoology/data/multiquery_ar.py",
                     "queries": "Every distinct stored key is queried once and stored values are distinct, so the constant prediction scores exactly 1/K over answers."}},
        "information_restriction_checks": {"no_query_or_label_argument": True,
            "repeated_records_deduplicated": True, "fixed_smallest_value_tie_break": True,
            "terminal_query_mutation_invariance": True, "metadata_not_used_to_predict": True},
        "conditions": {}}
    for task in ("noisy_recall", "mqar"):
        manifest_path = args.fixtures_dir / task / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["plan_sha256"] != report["plan_sha256"]:
            raise AssertionError("Fixture manifest and frozen plan disagree")
        task_report = {"fixture_manifest_sha256": file_hash(manifest_path), "conditions": {}}
        for name, condition in manifest["conditions"].items():
            config = condition["config"]
            splits = {}
            for split in ("dev", "test"):
                entry = condition[split]
                path = Path(entry["path"])
                batch = load_batch(path if path.is_absolute() else ROOT / path)
                if batch.sha256 != entry["sha256"]:
                    raise AssertionError("Frozen fixture array hash changed")
                predictions = [modal_prediction(ids, task, config) for ids in batch.input_ids]
                predicted_values = np.array([row["prediction"] for row in predictions])
                valid = batch.labels != -100
                correct = (batch.labels == predicted_values[:, None]) & valid
                per_example_targets = valid.sum(axis=1)
                expected = np.array([row["expected_accuracy"] for row in predictions])
                if task == "mqar":
                    assert np.all(per_example_targets == config["num_kv_pairs"])
                    assert np.all(correct.sum(axis=1) == 1)
                    assert np.all(expected == 1 / config["num_kv_pairs"])
                # An actual-fixture query mutation cannot affect the decision.
                for index in range(min(16, len(predictions))):
                    ids = batch.input_ids[index].copy()
                    if task == "noisy_recall":
                        ids[-1] = (int(ids[-1]) + 1) % task_spec(task, config)["key_token_end"]
                    else:
                        ids[2 * config["num_kv_pairs"]:] = 0
                    assert modal_prediction(ids, task, config) == predictions[index]
                splits[split] = {"fixture_sha256": batch.sha256, "examples": len(predictions),
                    "supervised_answers": int(valid.sum()), "correct_answers": int(correct.sum()),
                    "empirical_answer_accuracy": float(correct.sum() / valid.sum()),
                    "empirical_all_answers_correct_accuracy": float(np.all(correct | ~valid, axis=1).mean()),
                    "expected_answer_accuracy_given_visible_mapping": float(np.sum(expected * per_example_targets) / valid.sum()),
                    "mean_distinct_keys": float(np.mean([row["num_distinct_keys"] for row in predictions])),
                    "mean_distinct_values": float(np.mean([row["num_distinct_values"] for row in predictions])),
                    "mean_modal_value_multiplicity": float(np.mean([row["modal_value_multiplicity"] for row in predictions])),
                    "fraction_with_tied_modal_values": float(np.mean([row["tied_modal_values"] > 1 for row in predictions])),
                    "frozen_uniform_observed_values_expected_accuracy": entry["baselines"]["observed_values_uniform_expected_accuracy"],
                    "prediction_sha256": hashlib.sha256(predicted_values.astype("<i8").tobytes()).hexdigest()}
            task_report["conditions"][name] = {"config": config, "splits": splits}
            print(f"{task}/{name}: dev={splits['dev']['empirical_answer_accuracy']:.6f}, test={splits['test']['empirical_answer_accuracy']:.6f}, test conditional expectation={splits['test']['expected_answer_accuracy_given_visible_mapping']:.6f}", flush=True)
        report["conditions"][task] = task_report
    for task, task_report in report["conditions"].items():
        conditions = task_report["conditions"]
        groups = [("iid", "delay256", "delay512")] if task == "mqar" else [
            ("low", "low_delay256", "low_delay512"), ("moderate", "moderate_delay256", "moderate_delay512")]
        for group in groups:
            for split in ("dev", "test"):
                baseline = conditions[group[0]]["splits"][split]
                for name in group[1:]:
                    extended = conditions[name]["splits"][split]
                    for key in ("prediction_sha256", "empirical_answer_accuracy", "expected_answer_accuracy_given_visible_mapping"):
                        assert baseline[key] == extended[key]
    report["information_restriction_checks"]["delay_control_prediction_and_accuracy_invariance"] = True
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Posthoc modal-value shortcut baseline", "",
        "This supplemental analysis was requested **after inspecting the pilot's noisy-recall model metrics**. "
        "It is not a prespecified primary baseline. No model, generator, frozen fixture, training setting or original baseline manifest was changed; no tie-breaking rule was tuned.", "",
        "The pinned MAD generator samples its terminal query uniformly among distinct observed keys. "
        "Deduplicating repeated key/value records and returning the value associated with the most distinct keys is therefore the optimal constant answer conditional on the visible mapping while ignoring the query. "
        "Ties use the smallest value token ID. Repeated appearances of one key do not receive extra votes. "
        f"[Pinned MAD source]({report['source_justification']['mad']['url']}).", "",
        "Expected accuracy is the largest value multiplicity divided by the number of distinct observed keys, averaged over fixtures. "
        "Empirical accuracy scores that fixed prediction against actual held-out answers. These differ through finite sampling of the terminal query. "
        "The predictor reads only complete context records; it receives neither labels nor metadata and never reads the terminal query. "
        "Deduplication, fixed ties, query-token mutation and controlled-delay invariance were verified.", "",
        "| Task / condition | Dev empirical | Test empirical | Test conditional expected | Frozen uniform observed-value expected |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for task, task_report in report["conditions"].items():
        for name, condition in task_report["conditions"].items():
            dev, test = condition["splits"]["dev"], condition["splits"]["test"]
            lines.append(f"| {task} / {name} | {100*dev['empirical_answer_accuracy']:.3f}% | {100*test['empirical_answer_accuracy']:.3f}% | {100*test['expected_answer_accuracy_given_visible_mapping']:.3f}% | {100*test['frozen_uniform_observed_values_expected_accuracy']:.3f}% |")
    lines += ["", "MQAR stores distinct values and queries every association once, so this query-ignoring constant prediction has exactly 1/K answer accuracy and zero all-answers-correct sequence accuracy when K>1. "
        "The modal baseline is more informative for MAD because different keys may share a value. "
        "A learned score near this shortcut cannot by itself establish key-conditioned retrieval; this posthoc comparison does not prove which algorithm a model learned.", "",
        "Longer-delay conditions preserve the base mappings and labels, so this baseline produces exactly the same predictions and accuracy there. "
        "These repeated controls are not independent samples. No confidence interval or training-seed inference is claimed by this supplement.", "",
        f"[Complete supplemental metrics, source references and hashes]({args.output_json.name}) · [Standalone CPU analysis script]({Path(__file__).name})", "",
        "```bash", "CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \\",
        "  'test -f /.dockerenv && test \"$PWD\" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python docs/reports/stage-b/modal_value_baseline.py --plan configs/stage_b/pilot.json --fixtures-dir .runtime/stage-b/20260906T190223Z/fixtures --output-json docs/reports/stage-b/modal-value-baseline.json --output-markdown docs/reports/stage-b/modal-value-baseline.md'", "```", "",
        "The command refuses existing outputs; use new output paths for an independent rerun.", ""]
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
