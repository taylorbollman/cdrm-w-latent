"""Frozen official MAD data, with native labels and separate causal answer labels.

The pinned generator is executed verbatim, without importing MAD's trainer. Labels
are already aligned with model logits: callers MUST NOT shift them again. Native
recall training labels are dense next-token targets; held-out labels are masked.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendors/mad-lab"
REVISION = "0f49a452b84ca0d13f8eb9c1ffa649032376fb1b"
GENERATOR_SHA256 = "d3b8e9ad8377344b59c073ac3649b4b4c8a47280aa75696ee967c5f96198b6f8"
IGNORE_INDEX = -100
TASKS = ("in-context-recall", "selective-copying")
SPLIT_SEEDS = {"train": 12345, "dev": 23456, "final": 34567}
SPLIT_SIZES = {"train": 12800, "dev": 1280, "final": 1280}
SHUFFLE_SEED = 45678
SCREENING_SPLIT_SEEDS = {"train": 112345, "dev": 123456, "final": 134567}
SETTINGS = {
    "recall-v128-t128": {"task": "in-context-recall", "overrides": {"vocab_size": 128},
                          "official_change": "changes.vocab_size=128"},
    "copy-v16-t256-k96": {"task": "selective-copying", "overrides": {"num_tokens_to_copy": 96},
                          "official_change": "changes.num_tokens_to_copy=96"},
    "copy-v128-t256-k16": {"task": "selective-copying", "overrides": {"vocab_size": 128},
                           "official_change": "changes.vocab_size=128"},
}
_GLOBAL_RNG_LOCK = threading.Lock()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def official_generators():
    path = VENDOR_ROOT / "mad/data/instances.py"
    if file_sha256(path) != GENERATOR_SHA256:
        raise RuntimeError("Pinned MAD generator checksum mismatch")
    spec = importlib.util.spec_from_file_location("_cdrm_pinned_mad_instances", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_config(task, overrides=None):
    """Official baseline YAML plus relevant MADConfig defaults, explicitly resolved."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}")
    cfg = {
        "vocab_size": 16,
        "seq_len": 128 if task == "in-context-recall" else 256,
        "num_tokens_to_copy": 0 if task == "in-context-recall" else 16,
        "multi_query": True,
        "noise_vocab_size": 0,
        "frac_noise": 0.0,
        "k_motif_size": 1,
        "v_motif_size": 1,
        "target_ignore_idx": IGNORE_INDEX,
    }
    if overrides:
        allowed = {"vocab_size", "seq_len", "num_tokens_to_copy"}
        if set(overrides) - allowed:
            raise ValueError(f"Unsupported MAD overrides: {set(overrides) - allowed}")
        if any(type(value) is not int for value in overrides.values()):
            raise ValueError("MAD dimension overrides must be integers")
        cfg.update(overrides)
    if cfg["vocab_size"] < 4 or cfg["seq_len"] < 4:
        raise ValueError("MAD requires vocabulary and configured sequence length >=4")
    if task == "in-context-recall":
        if cfg["vocab_size"] % 2 or cfg["seq_len"] % 2 or cfg["num_tokens_to_copy"] != 0:
            raise ValueError("Recall requires even vocabulary/length and no copying override")
    elif cfg["num_tokens_to_copy"] < 1 or cfg["seq_len"] <= 2 * cfg["num_tokens_to_copy"] + 1:
        raise ValueError("Copying requires positive copy count and seq_len >2*copy_count+1")
    return cfg


def setting_spec(setting):
    if setting not in SETTINGS:
        raise ValueError(f"Unknown named MAD setting {setting}")
    record = SETTINGS[setting]
    config_path = VENDOR_ROOT / "configs/tasks" / f"{record['task']}.yml"
    return {"name": setting, **record, "revision": REVISION,
            "config_source_url": f"https://github.com/athms/mad-lab/blob/{REVISION}/configs/tasks/{record['task']}.yml",
            "config_source_sha256": file_sha256(config_path),
            "resolved_config": task_config(record["task"], record["overrides"])}


def task_spec(task, overrides=None):
    cfg = task_config(task, overrides)
    recall = task == "in-context-recall"
    vocab = cfg["vocab_size"]
    return {
        "task": task, "config": cfg, "vocab_size": vocab,
        "configured_sequence_length": cfg["seq_len"],
        "actual_sequence_length": cfg["seq_len"] - 1 if recall else cfg["seq_len"],
        "input_id_range": [0, vocab - 1],
        "answer_id_range": [vocab // 2, vocab - 1] if recall else [0, vocab - 3],
        "reserved_symbols": {} if recall else {"blank": vocab - 2, "copy_marker": vocab - 1},
        "labels_already_aligned_with_logits": True,
        "native_training_objective": "dense_next_token" if recall else "masked_copy_answers",
        "native_evaluation_objective": "masked_repeated_key_values" if recall else "masked_copy_answers",
        "teacher_forcing": (
            "Earlier key/value pairs, including earlier probed values, are visible ground-truth tokens; "
            "each scored value is absent from its own causal prefix, except prior presentations of its key."
            if recall else "The output region contains blank tokens only; no answer is fed back."
        ),
    }


@dataclass
class MadDataset:
    input_ids: np.ndarray
    labels: np.ndarray
    answer_labels: np.ndarray
    metadata: list[dict]
    manifest: dict

    def __post_init__(self):
        for name in ("input_ids", "labels", "answer_labels"):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            setattr(self, name, value)
        if self.input_ids.ndim != 2 or any(
            array.shape != self.input_ids.shape for array in (self.labels, self.answer_labels)
        ):
            raise ValueError("Expected equally shaped [N,T] arrays")
        if len(self.metadata) != len(self.input_ids):
            raise ValueError("Metadata count differs from example count")
        vocab = int(self.manifest.get("vocab_size", 16))
        if np.any((self.input_ids < 0) | (self.input_ids >= vocab)):
            raise ValueError("Input IDs outside the resolved MAD vocabulary")
        for array in (self.labels, self.answer_labels):
            if np.any(((array < 0) & (array != IGNORE_INDEX)) | (array >= vocab)):
                raise ValueError("Invalid target ID")
            if np.any((array != IGNORE_INDEX).sum(axis=1) == 0):
                raise ValueError("Every example must contain a scored target")

    def __len__(self):
        return len(self.input_ids)

    def take(self, indices):
        indices = np.atleast_1d(np.arange(len(self))[indices])
        return MadDataset(self.input_ids[indices], self.labels[indices], self.answer_labels[indices],
                          [self.metadata[int(i)] for i in indices], dict(self.manifest))

    @property
    def sha256(self):
        digest = hashlib.sha256()
        for name in ("input_ids", "labels", "answer_labels"):
            array = np.ascontiguousarray(getattr(self, name), dtype="<i8")
            digest.update(name.encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def recall_prefix_prediction(prefix, vocab_size=16):
    """Retrieve only from completed *earlier* pairs, using the current query key."""
    prefix = np.asarray(prefix)
    if len(prefix) % 2 != 1:
        return IGNORE_INDEX
    query = int(prefix[-1])
    half = vocab_size // 2
    if not 0 <= query < half:
        raise ValueError("Recall query is not a key")
    prior = {}
    for position in range(0, len(prefix) - 1, 2):
        key, value = map(int, prefix[position:position + 2])
        if not 0 <= key < half or not half <= value < vocab_size:
            raise ValueError("Malformed recall key/value pair")
        if key in prior and prior[key] != value:
            raise ValueError("Inconsistent within-example recall mapping")
        prior[key] = value
    return prior.get(query, IGNORE_INDEX)


def answer_oracle(task, input_ids, overrides=None):
    """Independent parser; no targets, generator internals or future tokens used for answers."""
    cfg = task_config(task, overrides)
    vocab, half = cfg["vocab_size"], cfg["vocab_size"] // 2
    tokens = np.asarray(input_ids, dtype=np.int64)
    if tokens.ndim == 2:
        return np.stack([answer_oracle(task, row, overrides) for row in tokens])
    labels = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int64)
    if task == "in-context-recall":
        if len(tokens) % 2 != 1:
            raise ValueError("Recall input ends with an unpaired terminal query")
        # The map is updated only after predicting at the preceding key position.
        prior = {}
        for position in range(0, len(tokens), 2):
            key = int(tokens[position])
            if not 0 <= key < half:
                raise ValueError("Malformed recall key")
            labels[position] = prior.get(key, IGNORE_INDEX)
            if position + 1 < len(tokens):
                value = int(tokens[position + 1])
                if not half <= value < vocab or (key in prior and prior[key] != value):
                    raise ValueError("Malformed recall value or inconsistent mapping")
                prior[key] = value
        if labels[-1] == IGNORE_INDEX:
            raise ValueError("Terminal recall key was never presented")
    elif task == "selective-copying":
        markers = np.flatnonzero(tokens == vocab - 1)
        if len(markers) != 1:
            raise ValueError("Selective copying requires one copy marker")
        marker = int(markers[0])
        source = tokens[:marker][tokens[:marker] != vocab - 2]
        if np.any((source < 0) | (source >= vocab - 2)):
            raise ValueError("Invalid copied symbol")
        if len(source) != len(tokens) - marker - 1 or np.any(tokens[marker + 1:] != vocab - 2):
            raise ValueError("Malformed selective-copy output region")
        labels[marker + 1:] = source
    else:
        raise ValueError(f"Unsupported task {task}")
    return labels


def generate_dataset(task, split, seed, num_examples, overrides=None):
    """Execute the pinned generator serially with separately seeded split streams.

    MAD's selective copier uses both Generator and legacy global RandomState.
    Seed both with the split seed, exactly as its train.py does, and restore the
    caller's global state afterwards. No duplicates are removed or resampled.
    """
    if split not in SPLIT_SEEDS or num_examples < 1 or not 0 <= seed < 2**32:
        raise ValueError("Expected train/dev/final, positive size and a uint32 seed")
    cfg = task_config(task, overrides)
    module = official_generators()
    function = getattr(module, "generate_" + task.replace("-", "_") + "_instance")
    rng = np.random.default_rng(seed)
    inputs, native_labels, answers, metadata = [], [], [], []
    # A second stream checks native evaluation masks on the identical recall inputs.
    mask_rng = np.random.default_rng(seed)
    with _GLOBAL_RNG_LOCK:
        global_state = np.random.get_state()
        np.random.seed(seed)
        try:
            for index in range(num_examples):
                tokens, labels = function(**cfg, rng=rng, is_training=(split == "train"))
                oracle = answer_oracle(task, tokens, overrides)
                if task == "in-context-recall":
                    check_tokens, check_labels = function(**cfg, rng=mask_rng, is_training=False)
                    if not np.array_equal(tokens, check_tokens) or not np.array_equal(oracle, check_labels):
                        raise AssertionError("Independent recall oracle disagrees with native evaluation mask")
                elif not np.array_equal(oracle, labels):
                    raise AssertionError("Independent copy oracle disagrees with native targets")
                if not np.array_equal(labels[oracle != IGNORE_INDEX], oracle[oracle != IGNORE_INDEX]):
                    raise AssertionError("Native training answers disagree with answer labels")
                inputs.append(tokens)
                native_labels.append(labels)
                answers.append(oracle)
                metadata.append({"example_index": index, "split": split,
                                 "answer_positions": np.flatnonzero(oracle != IGNORE_INDEX).tolist()})
        finally:
            np.random.set_state(global_state)
    manifest = {
        "schema_version": 1, **task_spec(task, overrides), "split": split, "seed": seed,
        "num_examples": num_examples, "is_training": split == "train",
        "generator_revision": REVISION, "generator_sha256": GENERATOR_SHA256,
        "rng": {"generator": "numpy.random.default_rng/PCG64", "seed": seed,
                "legacy_global_rng": "MT19937 seeded identically and restored" if task == "selective-copying" else "unused"},
        "adaptations": ["independent train/dev/final RNG streams", "additional answer_labels for reporting",
                        "fixed shared epoch permutations; no generation or augmentation during training"],
        "duplicate_policy": "Preserve native draws; audit exact collisions without rejection or resampling.",
    }
    if overrides:
        manifest["task_overrides"] = dict(overrides)
    dataset = MadDataset(np.stack(inputs), np.stack(native_labels), np.stack(answers), metadata, manifest)
    manifest.update({"array_sha256": dataset.sha256, "shape": list(dataset.input_ids.shape),
                     "native_scored_tokens": int((dataset.labels != IGNORE_INDEX).sum()),
                     "answer_scored_tokens": int((dataset.answer_labels != IGNORE_INDEX).sum()),
                     "observed_input_ids": np.unique(dataset.input_ids).tolist(),
                     "observed_native_target_ids": np.unique(dataset.labels[dataset.labels != IGNORE_INDEX]).tolist(),
                     "observed_answer_ids": np.unique(dataset.answer_labels[dataset.answer_labels != IGNORE_INDEX]).tolist()})
    if overrides:
        counts = (dataset.answer_labels != IGNORE_INDEX).sum(axis=1)
        manifest["answer_count_per_example"] = {"min": int(counts.min()), "max": int(counts.max()),
                                                "mean": float(counts.mean())}
    return dataset


def epoch_indices(size, epoch, seed=SHUFFLE_SEED):
    """Shared zero-based epoch permutation; independent of model/global RNG state."""
    if size < 1 or epoch < 0 or seed < 0:
        raise ValueError("Expected positive size and nonnegative epoch/seed")
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)


def score_predictions(predictions, labels):
    """Token accuracy and directly counted all-scored-positions sequence exact match."""
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape or labels.ndim != 2:
        raise ValueError("Predictions and labels must have the same [N,T] shape")
    valid = labels != IGNORE_INDEX
    if not len(labels) or np.any(valid.sum(axis=1) == 0):
        raise ValueError("Every example needs at least one scored position")
    correct = (predictions == labels) & valid
    exact = np.all(correct | ~valid, axis=1)
    return {"answer_accuracy": float(correct.sum() / valid.sum()),
            "sequence_exact_match": float(exact.mean()),
            "correct_answers": int(correct.sum()), "scored_answers": int(valid.sum()),
            "exact_sequences": int(exact.sum()), "examples": len(labels)}


def baseline_audit(dataset):
    task = dataset.manifest["task"]
    overrides = dataset.manifest.get("task_overrides")
    vocab = int(dataset.manifest["vocab_size"])
    labels = dataset.answer_labels
    predictions = np.full(labels.shape, IGNORE_INDEX, dtype=np.int64)
    expected_sum = 0.0
    for row_index, tokens in enumerate(dataset.input_ids):
        if task == "in-context-recall":
            prior = {}
            for position in range(0, len(tokens), 2):
                if labels[row_index, position] != IGNORE_INDEX:
                    # Each previously seen distinct key receives one vote. The
                    # current query key and future values do not affect this guess.
                    votes = Counter(prior.values())
                    modal = min(votes, key=lambda value: (-votes[value], value))
                    predictions[row_index, position] = modal
                    expected_sum += votes[modal] / len(prior)
                if position + 1 < len(tokens):
                    prior[int(tokens[position])] = int(tokens[position + 1])
        else:
            marker = int(np.flatnonzero(tokens == vocab - 1)[0])
            source = tokens[:marker][tokens[:marker] != vocab - 2]
            votes = Counter(map(int, source))
            modal = min(votes, key=lambda value: (-votes[value], value))
            predictions[row_index, marker + 1:] = modal
            expected_sum += votes[modal]
    score = score_predictions(predictions, labels)
    alphabet = vocab // 2 if task == "in-context-recall" else vocab - 2
    answer_counts = (labels != IGNORE_INDEX).sum(axis=1)
    return {
        "label_scope": "answer_labels only; dense native training objective is separate",
        "oracle": score_predictions(answer_oracle(task, dataset.input_ids, overrides), labels),
        "uniform_answer_vocabulary_token_chance": 1.0 / alphabet,
        "uniform_full_vocabulary_token_chance": 1.0 / vocab,
        "independent_uniform_answer_guess_sequence_chance": float(np.mean((1.0 / alphabet) ** answer_counts)),
        "query_ignoring_modal": {**score,
            "conditional_expected_token_accuracy": expected_sum / int((labels != IGNORE_INDEX).sum()),
            "method": ("Mode of values over distinct earlier visible keys; smallest-value tie break. "
                       "Repeated internal queries conditional on being scored are uniform over seen keys; "
                       "terminal query is explicitly uniform over distinct seen keys."
                       if task == "in-context-recall" else
                       "Repeat the mode of visible copied symbols at every output position; smallest-value tie break. "
                       "Ignores required output order; conditional expected equals its exact token fraction."),
            "information_restriction": "Only tokens at or before the scored position; no labels or future values select predictions."},
    }


def _row_hashes(dataset, with_labels=False):
    result = []
    for index, row in enumerate(dataset.input_ids):
        digest = hashlib.sha256(np.asarray(row, dtype="<i8").tobytes())
        if with_labels:
            digest.update(np.asarray(dataset.labels[index], dtype="<i8").tobytes())
        result.append(digest.hexdigest())
    return result


def overlap_audit(datasets):
    """Input-only checks remain valid when recall train/eval label masks differ."""
    result = {"policy": "No resampling or deduplication; report observed exact input collisions.",
              "within_split": {}, "cross_split": {}}
    hashes = {}
    for split, dataset in datasets.items():
        hashes[split] = {"input": _row_hashes(dataset), "input_and_native_labels": _row_hashes(dataset, True)}
        result["within_split"][split] = {}
        for kind, values in hashes[split].items():
            counts = Counter(values)
            result["within_split"][split][kind] = {"duplicate_rows": len(values) - len(counts),
                                                    "duplicated_unique_examples": sum(count > 1 for count in counts.values())}
    splits = list(datasets)
    for first_index, first in enumerate(splits):
        for second in splits[first_index + 1:]:
            pair = {}
            for kind in hashes[first]:
                common = set(hashes[first][kind]) & set(hashes[second][kind])
                pair[kind] = {"overlapping_unique_examples": len(common),
                              "first_split_rows": sum(value in common for value in hashes[first][kind]),
                              "second_split_rows": sum(value in common for value in hashes[second][kind]),
                              "example_hashes": sorted(common)[:10]}
            result["cross_split"][f"{first}__{second}"] = pair
    result["no_exact_input_cross_split_overlap"] = all(
        not pair["input"]["overlapping_unique_examples"] for pair in result["cross_split"].values())
    return result


def save_dataset(root, dataset):
    directory = Path(root) / dataset.manifest["task"]
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / dataset.manifest["split"]
    paths = {"arrays": base.with_suffix(".npz"), "metadata": base.with_suffix(".metadata.json"),
             "manifest": base.with_suffix(".manifest.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite fixed MAD split {base}")
    np.savez_compressed(paths["arrays"], input_ids=dataset.input_ids, labels=dataset.labels,
                        answer_labels=dataset.answer_labels)
    paths["metadata"].write_text(json.dumps(dataset.metadata, sort_keys=True) + "\n")
    manifest = {**dataset.manifest, "files": {
        name: {"name": paths[name].name, "sha256": file_sha256(paths[name])}
        for name in ("arrays", "metadata")}}
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    dataset.manifest = manifest
    return {"path": str(paths["manifest"]), "sha256": file_sha256(paths["manifest"]),
            "array_sha256": dataset.sha256}


def load_dataset(root, task, split, verify=True):
    """Load fixed arrays; manifest includes its own on-disk identity for checkpoints."""
    path = Path(root) / task / f"{split}.manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["task"] != task or manifest["split"] != split:
        raise ValueError("Dataset manifest task/split mismatch")
    if verify:
        for record in manifest["files"].values():
            if file_sha256(path.parent / record["name"]) != record["sha256"]:
                raise ValueError("Frozen dataset file checksum mismatch")
    with np.load(path.parent / manifest["files"]["arrays"]["name"], allow_pickle=False) as arrays:
        dataset = MadDataset(arrays["input_ids"], arrays["labels"], arrays["answer_labels"],
                             json.loads((path.parent / manifest["files"]["metadata"]["name"]).read_text()), manifest)
    if verify and dataset.sha256 != manifest["array_sha256"]:
        raise ValueError("Frozen dataset array checksum mismatch")
    dataset.manifest["manifest_sha256"] = file_sha256(path)
    return dataset
