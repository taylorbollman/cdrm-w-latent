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
# Preserve the original preparation defaults and their frozen task identities.
FUZZY_TASK = "fuzzy-in-context-recall"
SUPPORTED_TASKS = TASKS + (FUZZY_TASK,)
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
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {SUPPORTED_TASKS}")
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
    if task == FUZZY_TASK:
        cfg.update(seq_len=128, num_tokens_to_copy=0, k_motif_size=3, v_motif_size=3)
    if overrides:
        allowed = {"seq_len"} if task == FUZZY_TASK else {"vocab_size", "seq_len", "num_tokens_to_copy"}
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
    elif task == FUZZY_TASK:
        # Native probe placement needs room for two maximum-sized key/value pairs.
        if cfg["seq_len"] <= 2 * (cfg["k_motif_size"] + cfg["v_motif_size"]):
            raise ValueError("Fuzzy recall requires seq_len >12 with fixed V16/motifs3/3")
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
    if task == FUZZY_TASK:
        return {
            "task": task, "config": cfg, "vocab_size": vocab,
            "configured_sequence_length": cfg["seq_len"], "actual_sequence_length": cfg["seq_len"],
            "input_id_range": [0, 15], "answer_id_range": [7, 14],
            "reserved_symbols": {"left_padding": 15},
            "labels_already_aligned_with_logits": True,
            "native_training_objective": "dense_next_token_including_native_padding",
            "native_evaluation_objective": "masked_repeated_key_value_tokens",
            "key_motif_lengths": {"train": [1, 2, 3], "held_out": [3]},
            "value_motif_lengths": [1, 2, 3], "motifs_are_permutations_without_replacement": True,
            "teacher_forcing": "Previous value tokens, including earlier tokens of the current answer, are visible; no free-running generation is scored.",
            "answer_mask_scope": "Input-run annotation locates repeated-value positions. Prefix lookup predicts their values conditional on being scored; the mask itself is not claimed causally inferable.",
            "padding_attention_mask": "No extra padding attention mask; preserve native symbolic inputs.",
        }
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

    @property
    def oracle_available_mask(self):
        """Diagnostic retrieval coverage, never a replacement training/test mask."""
        if self.manifest["task"] != FUZZY_TASK:
            return self.answer_labels != IGNORE_INDEX
        mask = np.zeros_like(self.answer_labels, dtype=bool)
        for index, record in enumerate(self.metadata):
            positions = record.get("oracle_available_positions")
            if positions is None:
                raise ValueError("Fuzzy dataset is missing explicit retrieval coverage")
            mask[index, positions] = True
        if np.any(mask & (self.answer_labels == IGNORE_INDEX)):
            raise ValueError("Fuzzy oracle availability includes unscored positions")
        return mask

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


def _fuzzy_runs(input_ids, cfg):
    """Parse visible disjoint-alphabet runs; the last value may be incomplete.

    Run boundaries annotate positions using the supplied input. This function is
    not a causal prediction of whether a variable-length key has ended.
    """
    tokens = np.asarray(input_ids, dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens):
        raise ValueError("Fuzzy parser requires a nonempty one-dimensional input")
    pad, key_end = cfg["vocab_size"] - 1, (cfg["vocab_size"] - 1) // 2
    if np.any((tokens < 0) | (tokens > pad)):
        raise ValueError("Fuzzy input IDs outside the fixed vocabulary")
    position = 0
    while position < len(tokens) and tokens[position] == pad:
        position += 1
    left_padding = position
    if np.any(tokens[position:] == pad):
        raise ValueError("Fuzzy padding is only allowed on the left")
    pairs = []
    while position < len(tokens):
        start = position
        while position < len(tokens) and tokens[position] < key_end:
            position += 1
        key = tuple(map(int, tokens[start:position]))
        if not 1 <= len(key) <= cfg["k_motif_size"] or len(set(key)) != len(key):
            raise ValueError("Malformed fuzzy key motif")
        value_start = position
        while position < len(tokens) and key_end <= tokens[position] < pad:
            position += 1
        value = tuple(map(int, tokens[value_start:position]))
        if len(value) > cfg["v_motif_size"] or len(set(value)) != len(value):
            raise ValueError("Malformed fuzzy value motif")
        if not value and position != len(tokens):
            raise ValueError("An earlier fuzzy key is missing its value")
        pairs.append({"key": key, "visible_value": value, "key_start": start,
                      "value_start": value_start, "end": position})
    return pairs, left_padding


def _remember_fuzzy(prior, pair):
    key, value = pair["key"], pair["visible_value"]
    if not value or (key in prior and prior[key] != value):
        raise ValueError("Incomplete or inconsistent earlier fuzzy mapping")
    prior[key] = value


def fuzzy_prefix_prediction(prefix, overrides=None):
    """Retrieve the next value token using only this prefix and earlier pairs.

    A short known key can also prefix a longer key. The return value is a lookup
    prediction, not a claim that this position belongs to the native answer mask.
    """
    cfg = task_config(FUZZY_TASK, overrides)
    pairs, _ = _fuzzy_runs(prefix, cfg)
    if not pairs:
        return IGNORE_INDEX
    prior = {}
    for pair in pairs[:-1]:
        _remember_fuzzy(prior, pair)
    current = pairs[-1]
    value = prior.get(current["key"])
    visible = current["visible_value"]
    if value is None or len(visible) >= len(value):
        return IGNORE_INDEX
    if visible != value[:len(visible)]:
        raise ValueError("Visible fuzzy answer prefix disagrees with the earlier mapping")
    return value[len(visible)]


def fuzzy_answer_annotation(input_ids, overrides=None, *, terminal_target=None):
    """Annotate repeated-value positions; recover answers from earlier mappings.

    Native complete inputs omit exactly the final value token after shifting.
    Some native terminal queries have NO earlier presentation: the generator's
    loop can finish before inserting its probe. With no terminal_target this
    function returns only independently retrievable answers, leaving that whole
    terminal value ignored. Supplying the native terminal target annotates all
    native answer positions for reporting, never for oracle prediction.
    """
    cfg = task_config(FUZZY_TASK, overrides)
    tokens = np.asarray(input_ids, dtype=np.int64)
    pairs, padding = _fuzzy_runs(tokens, cfg)
    if len(pairs) < 2:
        raise ValueError("Fuzzy input must contain an earlier pair and terminal query")
    labels = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int64)
    prior, records = {}, []
    for index, pair in enumerate(pairs):
        final = index == len(pairs) - 1
        key, visible = pair["key"], pair["visible_value"]
        remembered = prior.get(key)
        if final:
            if remembered is not None:
                if visible != remembered[:-1]:
                    raise ValueError("Terminal fuzzy query must omit exactly one previously presented value token")
                if terminal_target is not None and terminal_target != remembered[-1]:
                    raise ValueError("Native terminal target disagrees with earlier fuzzy mapping")
                value = remembered
            else:
                value = visible + (terminal_target,)
                known = [token for token in value if token is not None]
                if (len(value) > cfg["v_motif_size"] or len(set(known)) != len(known) or
                        any(not 7 <= token < 15 for token in known)):
                    raise ValueError("Malformed unseen terminal fuzzy value")
        else:
            value = visible
            if not value or (remembered is not None and remembered != value):
                raise ValueError("Inconsistent fuzzy key/value mapping")
        if remembered is not None or (final and terminal_target is not None):
            start = pair["value_start"] - 1
            labels[start:start + len(value)] = value
        records.append({"key": list(key), "value": list(value),
                        "key_start": pair["key_start"], "value_start": pair["value_start"],
                        "visible_value_tokens": len(visible), "repeated": remembered is not None,
                        "final_query": final, "retrievable_from_earlier_mapping": remembered is not None})
        if not final:
            prior[key] = value
    return labels, {"left_padding": padding, "pairs": records}


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
    elif task == FUZZY_TASK:
        labels, _ = fuzzy_answer_annotation(tokens, overrides)
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
                if task == FUZZY_TASK:
                    # This is native answer annotation, not a prediction of an
                    # unseen terminal value. Independent retrieval is audited below.
                    oracle, fuzzy_metadata = fuzzy_answer_annotation(tokens, overrides,
                                                                       terminal_target=int(labels[-1]))
                    retrieval = answer_oracle(task, tokens, overrides)
                    available = retrieval != IGNORE_INDEX
                    if not np.array_equal(retrieval[available], oracle[available]):
                        raise AssertionError("Causal fuzzy retrieval disagrees with native answer annotation")
                    fuzzy_metadata["oracle_available_positions"] = np.flatnonzero(available).tolist()
                    fuzzy_metadata["unretrievable_answer_positions"] = np.flatnonzero(
                        (oracle != IGNORE_INDEX) & ~available).tolist()
                else:
                    oracle = answer_oracle(task, tokens, overrides)
                if task == "in-context-recall":
                    check_tokens, check_labels = function(**cfg, rng=mask_rng, is_training=False)
                    if not np.array_equal(tokens, check_tokens) or not np.array_equal(oracle, check_labels):
                        raise AssertionError("Independent recall oracle disagrees with native evaluation mask")
                elif task == FUZZY_TASK:
                    if split != "train" and not np.array_equal(oracle, labels):
                        raise AssertionError("Independent fuzzy annotation disagrees with native held-out targets")
                    if split == "train" and (np.any(labels == IGNORE_INDEX) or
                            not np.array_equal(labels[:-1], tokens[1:]) or labels[-1] != oracle[-1]):
                        raise AssertionError("Native fuzzy dense next-token alignment changed")
                elif not np.array_equal(oracle, labels):
                    raise AssertionError("Independent copy oracle disagrees with native targets")
                if not np.array_equal(labels[oracle != IGNORE_INDEX], oracle[oracle != IGNORE_INDEX]):
                    raise AssertionError("Native training answers disagree with answer labels")
                inputs.append(tokens)
                native_labels.append(labels)
                answers.append(oracle)
                metadata.append({"example_index": index, "split": split,
                                 "answer_positions": np.flatnonzero(oracle != IGNORE_INDEX).tolist()})
                if task == FUZZY_TASK:
                    metadata[-1].update(fuzzy_metadata)
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
    if task == FUZZY_TASK:
        mask = dataset.oracle_available_mask
        valid = dataset.answer_labels != IGNORE_INDEX
        unavailable = valid & ~mask
        unknown = [record for record in metadata if record["unretrievable_answer_positions"]]
        manifest["oracle_coverage"] = {
            "scored_tokens": int(valid.sum()), "available_tokens": int(mask.sum()),
            "unavailable_tokens": int(unavailable.sum()),
            "examples_with_unavailable_answers": len(unknown),
            "terminal_unavailable_tokens": int(unavailable.sum()), "context_unavailable_tokens": 0,
            "terminal_first_value_tokens_unavailable": len(unknown),
            "terminal_continuation_tokens_unavailable": int(unavailable.sum()) - len(unknown),
            "native_edge_case": "The probe placement loop can end before its sampled probe index, leaving a scored terminal key unseen. Preserve every native label and draw.",
            "interpretation": "Exact earlier-mapping retrieval coverage, not a theoretical task accuracy ceiling; no training or primary evaluation mask is changed.",
        }
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


def fuzzy_baseline_audit(dataset):
    """Causal shortcuts and retrieval coverage, with all native answers scored.

    Terminal values whose key was never presented remain in the denominator.
    Retrieval abstentions are counted as incorrect, not dropped or imputed.
    """
    overrides = dataset.manifest.get("task_overrides")
    labels = dataset.answer_labels
    modal = np.full(labels.shape, 7, dtype=np.int64)
    prefix_guess = modal.copy()
    retrieval = np.full(labels.shape, IGNORE_INDEX, dtype=np.int64)
    terminal_unseen = 0
    for row_index, tokens in enumerate(dataset.input_ids):
        annotations, info = fuzzy_answer_annotation(tokens, overrides,
                                                   terminal_target=int(dataset.labels[row_index, -1]))
        if not np.array_equal(annotations, labels[row_index]):
            raise ValueError("Stored fuzzy answer labels differ from native annotation")
        retrieval[row_index] = answer_oracle(FUZZY_TASK, tokens, overrides)
        terminal_unseen += int(not info["pairs"][-1]["repeated"])
        votes = Counter()
        for position, token in enumerate(tokens):
            if 7 <= token < 15:
                votes[int(token)] += 1
            if votes:
                modal[row_index, position] = min(votes, key=lambda value: (-votes[value], value))
        prior = {}
        for pair in info["pairs"]:
            start = pair["value_start"]
            for offset in range(len(pair["value"])):
                # The suffix visible at this prediction position excludes the
                # next token being guessed. Current query identity is unused.
                visible = tuple(map(int, tokens[start:start + offset]))
                options = Counter(value[offset] for value in prior.values()
                                  if len(value) > offset and value[:offset] == visible)
                prediction = (min(options, key=lambda value: (-options[value], value)) if options
                              else next(value for value in range(7, 15) if value not in visible))
                prefix_guess[row_index, start - 1 + offset] = prediction
            if not pair["final_query"]:
                prior[tuple(pair["key"])] = tuple(pair["value"])
    available = retrieval != IGNORE_INDEX
    valid = labels != IGNORE_INDEX
    if np.any(available & ~valid) or not np.array_equal(retrieval[available], labels[available]):
        raise AssertionError("Fuzzy prefix retrieval must agree wherever it has an earlier mapping")
    answers = valid.sum(axis=1)
    return {
        "label_scope": "All native fuzzy answer positions, including unseen terminal queries; dense training objective is separate.",
        "oracle": {**score_predictions(retrieval, labels),
                   "available_answers": int(available.sum()),
                   "coverage": float(available.sum() / valid.sum()),
                   "available_answer_accuracy": 1.0 if available.any() else None,
                   "unseen_terminal_query_examples": terminal_unseen,
                   "unretrievable_answers": int((valid & ~available).sum()),
                   "interpretation": "Prefix-only retrieval conditional on native scored positions; unavailable mappings abstain and count as errors. Coverage is not a statistical accuracy ceiling."},
        "uniform_answer_vocabulary_token_chance": 1 / 8,
        "uniform_full_vocabulary_token_chance": 1 / 16,
        "independent_uniform_answer_guess_sequence_chance": float(np.mean((1 / 8) ** answers)),
        "query_ignoring_modal": {**score_predictions(modal, labels),
            "method": "Mode of all visible value-token occurrences through this position; smallest-value tie break, default7. Current key is ignored.",
            "information_restriction": "Only input tokens at or before each prediction; targets never select predictions."},
        "query_ignoring_answer_prefix": {**score_predictions(prefix_guess, labels),
            "method": "Among values of distinct earlier completed keys matching the visible current value prefix, predict the modal next token. Ignore current query identity. Empty candidate set uses the smallest value symbol absent from the visible value prefix.",
            "information_restriction": "Earlier completed mappings and the teacher-forced current answer prefix only; input-run annotation selects reporting positions, not answers."},
    }


def baseline_audit(dataset):
    task = dataset.manifest["task"]
    if task == FUZZY_TASK:
        return fuzzy_baseline_audit(dataset)
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
