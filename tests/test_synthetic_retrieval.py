"""CPU NUM checks for pinned recall semantics, independent oracles and data hygiene."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cdrm.synthetic.common import SyntheticBatch
from cdrm.synthetic.retrieval import (
    _mad_generator, _rng, audit_retrieval_splits, generate_mqar,
    generate_noisy_recall, retrieval_baselines, retrieval_oracle, task_spec,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = {"mqar": generate_mqar, "noisy_recall": generate_noisy_recall}


@pytest.mark.parametrize("task", GENERATORS)
@pytest.mark.parametrize("length", [128, 256, 512])
def test_oracle_masks_vocab_and_shapes(task, length):
    config = {"sequence_length": length, "delay_tokens": length - 128}
    batch = GENERATORS[task](config, "train", 17, 24)
    spec = task_spec(task, config)
    assert batch.input_ids.shape == batch.labels.shape == (24, length)
    assert batch.input_ids.dtype == batch.labels.dtype == np.int64
    assert batch.input_ids.min() >= 0 and batch.input_ids.max() < spec["vocab_size"]
    assert np.all((batch.labels != -100).sum(axis=1) == spec["num_answers"])
    assert np.array_equal(retrieval_oracle(batch, task, config), batch.labels)
    for ids, labels, meta in zip(batch.input_ids, batch.labels, batch.metadata):
        positions = np.flatnonzero(labels != -100)
        assert positions.tolist() == meta["answer_positions"]
        assert np.all(np.isin(labels[positions], spec["answer_token_ids"]))
        assert np.all(ids[positions] < spec["key_token_end"])
        assert np.all(ids[positions] != labels[positions])
        assert all(delay >= 1 for delay in meta["answer_delays"])
    metrics = retrieval_baselines(batch, task, config)
    assert metrics["oracle_accuracy"] == 1
    assert metrics["chance_task_class_accuracy"] > metrics["chance_full_vocab_accuracy"]
    assert 0 <= metrics["restricted_history_coverage"] <= 1


@pytest.mark.parametrize("task", GENERATORS)
def test_rng_is_local_deterministic_prefix_stable_and_split_separated(task):
    generator = GENERATORS[task]
    np.random.seed(456)
    before = np.random.get_state()
    first = generator({}, "train", 937, 5)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert first.sha256 == generator({}, "train", 937, 5).sha256
    larger = generator({}, "train", 937, 8)
    assert np.array_equal(first.input_ids, larger.input_ids[:5])
    assert first.metadata == larger.metadata[:5]
    splits = {split: generator({}, split, 937, 128) for split in ["train", "dev", "test"]}
    report = audit_retrieval_splits(splits)
    assert all(row["input_overlap"] == 0 for row in report["between_splits"].values())
    assert all(row["whole_mapping_overlap"] == 0 for row in report["between_splits"].values())
    assert first.sha256 != generator({}, "train", 938, 5).sha256
    with pytest.raises(ValueError, match="leakage"):
        audit_retrieval_splits({"train": first, "test": first})
    duplicate = SyntheticBatch(np.repeat(first.input_ids[:1], 2, axis=0),
                               np.repeat(first.labels[:1], 2, axis=0), [first.metadata[0]] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        audit_retrieval_splits({"train": duplicate})


@pytest.mark.parametrize("task", GENERATORS)
def test_length_extension_preserves_records_mappings_and_adds_only_delay(task):
    base = GENERATORS[task]({}, "test", 12, 8)
    for length in [256, 512]:
        delta = length - 128
        config = {"sequence_length": length, "delay_tokens": delta}
        extended = GENERATORS[task](config, "test", 12, 8)
        split = 16 if task == "mqar" else 126
        assert np.array_equal(base.input_ids, np.concatenate(
            [extended.input_ids[:, :split], extended.input_ids[:, split + delta:]], axis=1))
        for original, changed in zip(base.metadata, extended.metadata):
            assert original["mapping_id"] == changed["mapping_id"]
            assert original["num_associations"] == changed["num_associations"]
            assert original["num_records"] == changed["num_records"]
            assert changed["answer_delays"] == [d + delta for d in original["answer_delays"]]
        metrics = retrieval_baselines(extended, task, config, history_tokens=16)
        assert metrics["restricted_history_coverage"] == 0
        assert metrics["restricted_history_expected_accuracy_uniform_fallback"] == metrics["chance_accuracy"]


def test_mad_calls_pinned_source_without_reimplementing_its_sampling():
    for fraction in [0.2, 0.6]:
        config = {"frac_noise": fraction}
        batch = generate_noisy_recall(config, "train", 91, 4)
        for index in range(4):
            ids, labels = _mad_generator()(vocab_size=81, seq_len=128, noise_vocab_size=16,
                frac_noise=fraction, is_training=False, multi_query=False,
                rng=_rng("noisy_recall", "train", 91, index))
            assert np.array_equal(ids, batch.input_ids[index])
            assert np.array_equal(labels, batch.labels[index])
            assert np.flatnonzero(labels != -100).tolist() == [127]
            # The official train mode predicts all context tokens. The adapter
            # deliberately uses the answer-only evaluation policy for every split.
            _, official_training_labels = _mad_generator()(vocab_size=81, seq_len=128,
                noise_vocab_size=16, frac_noise=fraction, is_training=True, multi_query=False,
                rng=_rng("noisy_recall", "train", 91, index))
            assert np.all(official_training_labels != -100)


def test_pinned_zoology_source_has_same_aligned_query_oracle_contract():
    import torch
    source = ROOT / "vendors/zoology/zoology/data/multiquery_ar.py"
    module = ast.parse(source.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "multiquery_ar")
    namespace = {"np": np, "torch": torch,
                 "DataSegment": lambda inputs, labels, **kwargs: SimpleNamespace(inputs=inputs, labels=labels, **kwargs)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    torch.manual_seed(17)
    upstream = namespace["multiquery_ar"](vocab_size=1024, num_examples=8, input_seq_len=128,
                                          seed=23, num_kv_pairs=8, random_non_queries=True)
    batch = SyntheticBatch(upstream.inputs.numpy(), upstream.labels.numpy(),
        [{"answer_positions": np.flatnonzero(row.numpy() != -100).tolist()} for row in upstream.labels])
    assert np.array_equal(retrieval_oracle(batch, "mqar", {}), batch.labels)
    assert np.all(batch.labels[:, :16] == -100)
    assert np.all((batch.labels != -100).sum(axis=1) == 8)


@pytest.mark.parametrize("task", GENERATORS)
def test_oracle_relevant_record_counterfactual_and_no_future_dependency(task):
    config = {}
    batch = GENERATORS[task](config, "test", 183, 1)
    pos = batch.metadata[0]["answer_positions"][0]
    old_answer = int(batch.labels[0, pos])
    spec = task_spec(task, config)
    replacement = next(x for x in spec["answer_token_ids"] if x != old_answer)
    modified = SyntheticBatch(batch.input_ids.copy(), batch.labels.copy(), batch.metadata)
    query_key = int(modified.input_ids[0, pos])
    stop = 16 if task == "mqar" else pos - 1
    for at in range(0, stop, 2):
        if modified.input_ids[0, at] == query_key:
            modified.input_ids[0, at + 1] = replacement
    answer = retrieval_oracle(modified, task, config)
    assert answer[0, pos] == replacement
    if pos + 1 < modified.input_ids.shape[1]:
        # The aligned query answer cannot depend on later tokens/answers.
        modified.input_ids[0, pos + 1:] = replacement
        assert retrieval_oracle(modified, task, config)[0, pos] == replacement
    # Stored labels are irrelevant to the oracle's reconstructed answer.
    modified.labels[:] = replacement
    assert retrieval_oracle(modified, task, config)[0, pos] == replacement


def test_noisy_distractor_axis_changes_realized_noise():
    low = generate_noisy_recall({"frac_noise": 0.2}, "dev", 912, 256)
    moderate = generate_noisy_recall({"frac_noise": 0.6}, "dev", 912, 256)
    low_noise = np.mean([m["observed_noise_fraction"] for m in low.metadata])
    moderate_noise = np.mean([m["observed_noise_fraction"] for m in moderate.metadata])
    assert 0.15 < low_noise < 0.25
    assert 0.5 < moderate_noise < 0.7
    assert moderate_noise > low_noise + 0.25


@pytest.mark.parametrize("task,config", [
    ("mqar", {"num_kv_pairs": 33}), ("mqar", {"num_kv_pairs": True}),
    ("mqar", {"vocab_size": 128}), ("mqar", {"power_a": float("nan")}),
    ("mqar", {"sequence_length": 127}), ("mqar", {"delay_tokens": 127}),
    ("noisy_recall", {"frac_noise": 1}), ("noisy_recall", {"noise_vocab_size": 0}),
    ("noisy_recall", {"vocab_size": 18}), ("noisy_recall", {"multi_query": True}),
])
def test_unsupported_configs_fail_before_generation(task, config):
    with pytest.raises(ValueError):
        GENERATORS[task](config, "dev", 17, 2)


def test_pinned_source_snapshot_hashes():
    for name in ["zoology", "mad-lab"]:
        root = ROOT / "vendors" / name
        manifest = json.loads((root / "PROVENANCE.json").read_text())
        assert len(manifest["revision"]) == 40
        for filename, file_data in manifest["files"].items():
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == file_data["sha256"]
