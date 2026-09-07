"""Pinned MQAR and MAD recall adapters with explicitly aligned answer labels.

MQAR is adapted from HazyResearch/zoology (Apache-2.0); MAD calls an
unmodified vendored generator (MIT). See docs/reports/stage-b/task-sources.md
and the licenses/provenance under vendors/{zoology,mad-lab}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cdrm.synthetic.common import SyntheticBatch

ZOOLOGY_REVISION = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
MAD_REVISION = "0f49a452b84ca0d13f8eb9c1ffa649032376fb1b"
ADAPTER_VERSION = "retrieval-v1"
_SPLITS = {"train", "dev", "test", "calibration"}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _rng(task: str, split: str, seed: int, index: int, purpose: str = "base") -> np.random.Generator:
    if split not in _SPLITS:
        raise ValueError(f"Unknown split {split!r}; use {sorted(_SPLITS)}.")
    seed = _integer(seed, "seed")
    data = json.dumps([ADAPTER_VERSION, task, split, seed, index, purpose]).encode()
    words = np.frombuffer(hashlib.sha256(data).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(words))


def task_spec(task: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate and resolve the complete symbolic vocabulary/task contract."""
    if task not in {"mqar", "noisy_recall"}:
        raise ValueError(f"Unknown retrieval task {task!r}.")
    allowed = ({"sequence_length", "vocab_size", "delay_tokens", "num_kv_pairs", "power_a", "random_non_queries"}
               if task == "mqar" else
               {"sequence_length", "vocab_size", "delay_tokens", "noise_vocab_size", "frac_noise"})
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown {task} configuration fields: {sorted(unknown)}")
    length = _integer(config.get("sequence_length", 128), "sequence_length", 8)
    delay = _integer(config.get("delay_tokens", 0), "delay_tokens")
    content_length = length - delay
    if content_length < 8 or content_length % 2 or delay % 2:
        raise ValueError("Base content length must be even and >= 8; delay_tokens must be even.")
    vocab = _integer(config.get("vocab_size", 1024 if task == "mqar" else 81), "vocab_size", 4)
    result = {"task": task, "sequence_length": length, "content_length": content_length,
              "delay_tokens": delay, "vocab_size": vocab, "label_alignment": "logit_position_no_shift",
              "adapter_version": ADAPTER_VERSION}
    if task == "mqar":
        pairs = _integer(config.get("num_kv_pairs", 8), "num_kv_pairs", 1)
        power = float(config.get("power_a", 0.01))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("power_a must be finite and positive.")
        random_fill = config.get("random_non_queries", True)
        if not isinstance(random_fill, bool):
            raise ValueError("random_non_queries must be boolean.")
        if vocab <= content_length or 4 * pairs > content_length or pairs > vocab // 2 - 1:
            raise ValueError("MQAR requires vocab_size > base content length and 4*num_kv_pairs <= base length.")
        result.update(num_kv_pairs=pairs, power_a=power, random_non_queries=random_fill,
                      key_token_start=1, key_token_end=vocab // 2,
                      answer_token_ids=list(range(vocab // 2, vocab)),
                      chance_accuracy=1 / (vocab - vocab // 2), upstream_revision=ZOOLOGY_REVISION,
                      num_answers=pairs, num_passes=1)
    else:
        noise = _integer(config.get("noise_vocab_size", 16), "noise_vocab_size", 1)
        fraction = float(config.get("frac_noise", 0.2))
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("frac_noise must be finite and in [0, 1).")
        content_vocab = vocab - noise - 1
        if content_vocab < 4:
            raise ValueError("MAD requires at least two keys and two values after reserving noise and copy tokens.")
        result.update(noise_vocab_size=noise, frac_noise=fraction,
                      key_token_start=0, key_token_end=content_vocab // 2,
                      answer_token_ids=list(range(content_vocab // 2, content_vocab)),
                      noise_token_start=content_vocab, noise_token_end=content_vocab + noise,
                      copy_token_id=vocab - 1, multi_query=False, num_answers=1,
                      chance_accuracy=1 / (content_vocab - content_vocab // 2), upstream_revision=MAD_REVISION)
    return result


def _identity(ids: np.ndarray, labels: np.ndarray) -> str:
    return hashlib.sha256(ids.astype("<i8", copy=False).tobytes() + labels.astype("<i8", copy=False).tobytes()).hexdigest()


def _metadata(task: str, split: str, seed: int, index: int, ids: np.ndarray,
              labels: np.ndarray, mapping: dict[int, int], positions: list[int],
              delays: list[int], spec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"task": task, "split": split, "seed": int(seed), "example_index": index,
            "example_id": _identity(ids, labels),
            "mapping_id": hashlib.sha256(json.dumps(sorted(mapping.items())).encode()).hexdigest(),
            "answer_positions": positions, "answer_delays": delays,
            "num_associations": len(mapping), "sequence_length": len(ids),
            "content_length": spec["content_length"], "delay_tokens": spec["delay_tokens"],
            "upstream_revision": spec["upstream_revision"], "adapter_version": ADAPTER_VERSION,
            **extra}


def generate_mqar(config: dict, split: str, seed: int, num_examples: int) -> SyntheticBatch:
    """Zoology single-pass MQAR; fillers and all other randomness are locally seeded.

    Labels score each inserted query key itself. They are already shifted as in
    upstream. Random filler is allowed to collide with keys/values, also upstream.
    An optional delay is inserted immediately after the fixed association prefix.
    """
    spec = task_spec("mqar", config)
    count = _integer(num_examples, "num_examples", 1)
    _rng("mqar", split, seed, 0)  # validate even if future callers permit an empty batch
    length, pairs = spec["sequence_length"], spec["num_kv_pairs"]
    context = 2 * pairs
    space = (spec["content_length"] - context) // 2
    power = spec["power_a"]
    probabilities = power * np.arange(1, space + 1, dtype=np.float64) ** (power - 1)
    probabilities /= probabilities.sum()
    inputs = np.empty((count, length), dtype=np.int64)
    labels = np.full_like(inputs, -100)
    metadata = []
    for index in range(count):
        rng = _rng("mqar", split, seed, index)
        keys = rng.choice(np.arange(1, spec["vocab_size"] // 2), size=pairs, replace=False)
        values = rng.choice(spec["answer_token_ids"], size=pairs, replace=False)
        gaps = rng.choice(space, size=pairs, replace=False, p=probabilities)
        base = np.zeros(spec["content_length"], dtype=np.int64)
        base[:context:2], base[1:context:2] = keys, values
        positions = context + 2 * gaps
        base[positions] = keys
        if spec["random_non_queries"]:
            fill = rng.integers(spec["vocab_size"], size=base.shape, dtype=np.int64)
            base[base == 0] = fill[base == 0]
        delay_rng = _rng("mqar", split, seed, index, "delay")
        delay = (delay_rng.integers(spec["vocab_size"], size=spec["delay_tokens"], dtype=np.int64)
                 if spec["random_non_queries"] else np.zeros(spec["delay_tokens"], dtype=np.int64))
        inputs[index] = np.concatenate([base[:context], delay, base[context:]])
        positions = positions + spec["delay_tokens"]
        labels[index, positions] = values
        mapping = dict(zip(map(int, keys), map(int, values)))
        order = np.argsort(positions)
        ordered_positions = positions[order].tolist()
        # Distance from the source value token, which must precede its query.
        delays = (positions - (2 * np.arange(pairs) + 1))[order].tolist()
        metadata.append(_metadata("mqar", split, seed, index, inputs[index], labels[index], mapping,
                                  ordered_positions, delays, spec, num_records=pairs,
                                  num_kv_pairs=pairs, random_non_queries=spec["random_non_queries"]))
    return SyntheticBatch(input_ids=inputs, labels=labels, metadata=metadata)


@lru_cache(maxsize=1)
def _mad_generator():
    path = Path(__file__).resolve().parents[2] / "vendors/mad-lab/mad/data/instances.py"
    module_spec = importlib.util.spec_from_file_location("_cdrm_pinned_mad_instances", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load the pinned MAD source at {path}.")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.generate_noisy_in_context_recall_instance


def generate_noisy_recall(config: dict, split: str, seed: int, num_examples: int) -> SyntheticBatch:
    """MAD single terminal query, using answer-only (evaluation) labels on all splits.

    Unlike the official training objective this does not predict random context.
    Optional dedicated-noise tokens extend the terminal retrieval delay while
    preserving every base association and record, rather than adding more records.
    """
    spec = task_spec("noisy_recall", config)
    count = _integer(num_examples, "num_examples", 1)
    _rng("noisy_recall", split, seed, 0)
    inputs = np.empty((count, spec["sequence_length"]), dtype=np.int64)
    labels = np.full_like(inputs, -100)
    metadata = []
    generator = _mad_generator()
    for index in range(count):
        rng = _rng("noisy_recall", split, seed, index)
        # Upstream's guaranteed-record index includes the omitted final slot.
        # A rare all-noise draw therefore errors; condition on a valid record.
        for attempt in range(100):
            try:
                base, base_labels = generator(vocab_size=spec["vocab_size"],
                    seq_len=spec["content_length"], noise_vocab_size=spec["noise_vocab_size"],
                    frac_noise=spec["frac_noise"], is_training=False, multi_query=False, rng=rng)
            except ValueError as error:
                if "cannot be empty" not in str(error):
                    raise
                continue
            break
        else:
            raise RuntimeError("MAD generated no valid association in 100 attempts; reduce frac_noise.")
        if len(base) != spec["content_length"] or np.count_nonzero(base_labels != -100) != 1:
            raise AssertionError("Pinned MAD single-query length/alignment contract changed.")
        delay_rng = _rng("noisy_recall", split, seed, index, "delay")
        delay = delay_rng.integers(spec["noise_token_start"], spec["noise_token_end"],
                                   size=spec["delay_tokens"], dtype=np.int64)
        inputs[index] = np.concatenate([base[:-2], delay, base[-2:]])
        labels[index, -1] = base_labels[-1]
        mapping, last_value_positions = {}, {}
        num_records = 0
        for pos in range(0, len(base) - 2, 2):
            key, value = map(int, base[pos:pos + 2])
            if key < spec["key_token_end"]:
                if key in mapping and mapping[key] != value:
                    raise AssertionError("MAD must keep each within-example mapping consistent.")
                mapping[key], last_value_positions[key] = value, pos + 1
                num_records += 1
        query = int(inputs[index, -1])
        position = spec["sequence_length"] - 1
        noise_count = (spec["content_length"] - 2 - 2 * num_records) + spec["delay_tokens"]
        metadata.append(_metadata("noisy_recall", split, seed, index, inputs[index], labels[index], mapping,
            [position], [position - last_value_positions[query]], spec, num_records=num_records,
            frac_noise=spec["frac_noise"], observed_noise_fraction=noise_count / (len(inputs[index]) - 2),
            valid_instance_retries=attempt))
    return SyntheticBatch(input_ids=inputs, labels=labels, metadata=metadata)


def retrieval_oracle(batch: SyntheticBatch, task: str, config: dict,
                     history_tokens: int | None = None) -> np.ndarray:
    """Reconstruct answers from visible records, never from labels or saved answers.

    A restricted window returns -100 when no complete relevant record is visible.
    MQAR's scored query positions are task annotations because random filler may
    contain accidental key matches. No future or current-position value is read.
    """
    spec = task_spec(task, config)
    if history_tokens is not None:
        _integer(history_tokens, "history_tokens", 1)
    answers = np.full_like(batch.input_ids, -100)
    for row, meta in enumerate(batch.metadata):
        ids = batch.input_ids[row]
        for query_position in meta["answer_positions"]:
            lower = 0 if history_tokens is None else max(0, query_position - history_tokens)
            query = int(ids[query_position])
            stop = 2 * spec["num_kv_pairs"] if task == "mqar" else query_position - 1
            mapping = {}
            for pos in range(0, min(stop, query_position), 2):
                if pos < lower or pos + 1 >= query_position:
                    continue
                key, value = map(int, ids[pos:pos + 2])
                if spec["key_token_start"] <= key < spec["key_token_end"]:
                    mapping[key] = value
            if query in mapping:
                answers[row, query_position] = mapping[query]
    return answers


def retrieval_baselines(batch: SyntheticBatch, task: str, config: dict,
                        history_tokens: int = 16) -> dict[str, Any]:
    """Exact, chance, last-value and local-retrieval baselines on answer positions.

    Missing restricted-history answers receive uniform-value expected accuracy;
    coverage and abstention-as-error accuracy are also reported separately.
    """
    spec = task_spec(task, config)
    mask = batch.labels != -100
    truth = batch.labels[mask]
    if not truth.size:
        raise ValueError("Baseline needs scored answers.")
    oracle = retrieval_oracle(batch, task, config)[mask]
    recent = retrieval_oracle(batch, task, config, history_tokens)[mask]
    covered = recent != -100
    chance = spec["chance_accuracy"]
    last_predictions, candidate_chances = [], []
    for row, meta in enumerate(batch.metadata):
        for query_pos in meta["answer_positions"]:
            stop = 2 * spec["num_kv_pairs"] if task == "mqar" else query_pos - 1
            values = [int(batch.input_ids[row, pos + 1]) for pos in range(0, min(stop, query_pos), 2)
                      if spec["key_token_start"] <= batch.input_ids[row, pos] < spec["key_token_end"]]
            last_predictions.append(values[-1] if values else -100)
            candidate_chances.append(1 / len(set(values)) if values else chance)
    recent_correct = recent == truth
    return {"answer_count": int(truth.size), "oracle_accuracy": float(np.mean(oracle == truth)),
            "chance_accuracy": chance, "chance_task_class_accuracy": chance,
            "chance_full_vocab_accuracy": 1 / spec["vocab_size"],
            "chance_full_vocab_cross_entropy": float(np.log(spec["vocab_size"])),
            "chance_cross_entropy": float(np.log(len(spec["answer_token_ids"]))),
            "chance_definition": "uniform over legal value tokens; full-vocabulary random is lower",
            "restricted_history_tokens": history_tokens,
            "restricted_history_coverage": float(np.mean(covered)),
            "restricted_history_accuracy_abstention_as_error": float(np.mean(recent_correct)),
            "restricted_history_expected_accuracy_uniform_fallback": float(np.mean(recent_correct) + np.mean(~covered) * chance),
            "last_record_value_accuracy": float(np.mean(np.asarray(last_predictions) == truth)),
            "observed_values_uniform_expected_accuracy": float(np.mean(candidate_chances)),
            "observed_values_uniform_definition": "uniform over distinct values in preceding complete records; ignores query key"}


def audit_retrieval_splits(batches: dict[str, SyntheticBatch]) -> dict[str, Any]:
    """Count exact sequence duplicates and mapping overlap; reject input leakage.

    Distinct split seeds are necessary but not a proof of separation. This audit
    inspects actual arrays. A repeated *whole* mapping is reported separately;
    individual key/value pairs and trained symbols may occur in every split.
    """
    input_sets, mapping_sets, report = {}, {}, {"within_split": {}, "between_splits": {}}
    for name, batch in batches.items():
        fingerprints = [hashlib.sha256(row.astype("<i8", copy=False).tobytes()).hexdigest()
                        for row in batch.input_ids]
        input_sets[name] = set(fingerprints)
        mapping_sets[name] = {meta["mapping_id"] for meta in batch.metadata}
        report["within_split"][name] = {"examples": len(fingerprints),
            "duplicate_inputs": len(fingerprints) - len(input_sets[name]),
            "duplicate_whole_mappings": len(fingerprints) - len(mapping_sets[name])}
    names = list(batches)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            report["between_splits"][f"{left}:{right}"] = {
                "input_overlap": len(input_sets[left] & input_sets[right]),
                "whole_mapping_overlap": len(mapping_sets[left] & mapping_sets[right])}
    if any(item["duplicate_inputs"] for item in report["within_split"].values()):
        raise ValueError(f"Duplicate retrieval inputs within a split: {report}")
    if any(item["input_overlap"] for item in report["between_splits"].values()):
        raise ValueError(f"Retrieval input leakage between splits: {report}")
    return report
