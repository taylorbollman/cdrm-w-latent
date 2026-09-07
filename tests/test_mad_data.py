"""CPU-only tests for pinned MAD semantics, oracles, identities and fixed splits."""
import copy
import json

import numpy as np
import pytest

from cdrm.mad_data import (
    IGNORE_INDEX, TASKS, answer_oracle, baseline_audit, epoch_indices, generate_dataset,
    load_dataset, official_generators, overlap_audit, recall_prefix_prediction,
    save_dataset, score_predictions, task_config, task_spec, setting_spec, SETTINGS,
)
from scripts.cdrm_prepare_mad import structural_audit, prepare


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("split", ["train", "dev", "final"])
def test_verbatim_official_generator_and_native_masks(task, split):
    actual = generate_dataset(task, split, seed=391, num_examples=12)
    generator = getattr(official_generators(), "generate_" + task.replace("-", "_") + "_instance")
    rng = np.random.default_rng(391)
    saved_state = np.random.get_state()
    try:
        np.random.seed(391)
        expected = [generator(**task_config(task), rng=rng, is_training=split == "train") for _ in range(12)]
    finally:
        np.random.set_state(saved_state)
    np.testing.assert_array_equal(actual.input_ids, np.stack([row[0] for row in expected]))
    np.testing.assert_array_equal(actual.labels, np.stack([row[1] for row in expected]))
    assert structural_audit(actual)["oracle_exact"]
    assert actual.input_ids.shape == (12, task_spec(task)["actual_sequence_length"])


def test_recall_native_dense_training_is_not_replaced_by_answer_mask():
    train = generate_dataset("in-context-recall", "train", 23, 8)
    dev = generate_dataset("in-context-recall", "dev", 23, 8)
    np.testing.assert_array_equal(train.input_ids, dev.input_ids)
    np.testing.assert_array_equal(train.answer_labels, dev.labels)
    assert np.all(train.labels >= 0)
    assert np.any(train.answer_labels == IGNORE_INDEX)
    np.testing.assert_array_equal(train.labels[:, :-1], train.input_ids[:, 1:])
    assert np.all(train.answer_labels[:, 0] == IGNORE_INDEX)
    assert np.all(train.answer_labels[:, -1] >= 8)
    assert np.all(train.answer_labels[:, 1::2] == IGNORE_INDEX)


@pytest.mark.parametrize("task", TASKS)
def test_generation_deterministic_and_global_numpy_rng_restored(task):
    before = copy.deepcopy(np.random.get_state())
    first = generate_dataset(task, "train", 199, 16)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    second = generate_dataset(task, "train", 199, 16)
    assert first.sha256 == second.sha256
    assert first.sha256 != generate_dataset(task, "train", 200, 16).sha256


def test_recall_prefix_query_and_value_interventions_are_causal():
    # An answer is produced at the query key, before its following value is visible.
    tokens = np.array([0, 8, 1, 9, 0, 8, 1])
    expected = np.array([-100, -100, -100, -100, 8, -100, 9])
    np.testing.assert_array_equal(answer_oracle("in-context-recall", tokens), expected)
    assert recall_prefix_prediction(tokens[:5]) == 8
    altered_future = tokens.copy()
    altered_future[5:] = [15, 7]
    assert recall_prefix_prediction(altered_future[:5]) == 8
    changed_query = tokens.copy()
    changed_query[-1] = 0
    assert answer_oracle("in-context-recall", changed_query)[-1] == 8
    changed_mapping = tokens.copy()
    changed_mapping[[1, 5]] = 10
    assert answer_oracle("in-context-recall", changed_mapping)[4] == 10
    assert recall_prefix_prediction(np.array([2])) == IGNORE_INDEX


def test_copy_symbols_alignment_and_order_counterfactual():
    dataset = generate_dataset("selective-copying", "train", 111, 4)
    assert np.all(dataset.input_ids[:, 239] == 15)
    assert np.all(dataset.input_ids[:, 240:] == 14)
    assert np.all(dataset.labels[:, :240] == IGNORE_INDEX)
    assert np.all((dataset.labels[:, 240:] >= 0) & (dataset.labels[:, 240:] < 14))
    # Official np.insert never adds blanks after the final copied input symbol.
    assert np.all(dataset.input_ids[:, 238] < 14)
    tokens = dataset.input_ids[0].copy()
    positions = np.flatnonzero(tokens[:239] != 14)
    tokens[positions] = tokens[positions][::-1]
    np.testing.assert_array_equal(answer_oracle("selective-copying", tokens)[240:], dataset.labels[0, 240:][::-1])


def test_exact_sequence_accuracy_is_counted_not_inferred_from_token_accuracy():
    labels = np.array([[8, 9], [10, 11]])
    predictions = np.array([[8, 9], [10, 12]])
    score = score_predictions(predictions, labels)
    assert score["answer_accuracy"] == 0.75
    assert score["sequence_exact_match"] == 0.5
    assert score["sequence_exact_match"] != score["answer_accuracy"] ** 2


def test_query_ignoring_modal_baseline_uses_distinct_keys_not_occurrences():
    dataset = generate_dataset("in-context-recall", "dev", 9, 1)
    # key0 repeats, but two different keys map to9: key-deduplicated mode is9.
    tokens = np.array([[0, 8, 0, 8, 0, 8, 1, 9, 2, 9, 1]])
    dataset.input_ids = tokens
    dataset.labels = dataset.answer_labels = answer_oracle("in-context-recall", tokens)
    baseline = baseline_audit(dataset)
    assert baseline["query_ignoring_modal"]["correct_answers"] == 3
    assert baseline["uniform_answer_vocabulary_token_chance"] == 1 / 8
    assert baseline["uniform_full_vocabulary_token_chance"] == 1 / 16


def test_input_overlap_detected_when_train_dev_native_masks_differ():
    train = generate_dataset("in-context-recall", "train", 80, 5)
    dev = generate_dataset("in-context-recall", "dev", 80, 5)
    result = overlap_audit({"train": train, "dev": dev})
    assert not result["no_exact_input_cross_split_overlap"]
    assert result["cross_split"]["train__dev"]["input"]["overlapping_unique_examples"] == 5
    assert result["cross_split"]["train__dev"]["input_and_native_labels"]["overlapping_unique_examples"] == 0
    assert overlap_audit({"train": train.take([0, 0, 1])})["within_split"]["train"]["input"]["duplicate_rows"] == 1


def test_fixed_epoch_shuffles_reproduce_and_cover_every_example():
    first = epoch_indices(100, 0)
    np.testing.assert_array_equal(np.sort(first), np.arange(100))
    np.testing.assert_array_equal(first, epoch_indices(100, 0))
    assert not np.array_equal(first, epoch_indices(100, 1))
    assert not np.array_equal(first, epoch_indices(100, 0, seed=12))


@pytest.mark.parametrize("task", TASKS)
def test_save_load_identity_refuses_overwrite_and_detects_tampering(tmp_path, task):
    dataset = generate_dataset(task, "dev", 70, 3)
    saved = save_dataset(tmp_path, dataset)
    loaded = load_dataset(tmp_path, task, "dev")
    assert loaded.sha256 == dataset.sha256
    assert loaded.manifest["manifest_sha256"] == saved["sha256"]
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path, dataset)
    path = tmp_path / task / "dev.metadata.json"
    path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path, task, "dev")


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("split", ["train", "dev"])
def test_official_hard_settings_retain_exact_generator_outputs(setting, split):
    spec = setting_spec(setting)
    task, overrides = spec["task"], spec["overrides"]
    actual = generate_dataset(task, split, 112345, 10, overrides)
    generator = getattr(official_generators(), "generate_" + task.replace("-", "_") + "_instance")
    saved_state = np.random.get_state()
    try:
        np.random.seed(112345)
        rng = np.random.default_rng(112345)
        expected = [generator(**task_config(task, overrides), rng=rng, is_training=split == "train") for _ in range(10)]
    finally:
        np.random.set_state(saved_state)
    np.testing.assert_array_equal(actual.input_ids, np.stack([x for x, _ in expected]))
    np.testing.assert_array_equal(actual.labels, np.stack([y for _, y in expected]))
    assert structural_audit(actual)["oracle_exact"]
    baseline = baseline_audit(actual)
    assert baseline["oracle"]["sequence_exact_match"] == 1.0
    assert actual.manifest["vocab_size"] == spec["resolved_config"]["vocab_size"]
    assert actual.manifest["config"] == spec["resolved_config"]
    if task == "in-context-recall":
        assert actual.input_ids.shape[1] == 127
        assert np.all(actual.answer_labels[actual.answer_labels != IGNORE_INDEX] >= 64)
        for tokens, answers in zip(actual.input_ids, actual.answer_labels):
            assert (answers != IGNORE_INDEX).sum() == 64 - len(np.unique(tokens[:-1:2]))
        assert baseline["uniform_answer_vocabulary_token_chance"] == 1 / 64
    elif setting == "copy-v16-t256-k96":
        assert np.all(actual.input_ids[:, 159] == 15)
        assert np.all(actual.answer_labels[:, :160] == IGNORE_INDEX)
        assert np.all((actual.answer_labels != IGNORE_INDEX).sum(axis=1) == 96)
    else:
        assert np.all(actual.input_ids[:, 239] == 127)
        assert np.all(actual.input_ids[:, 240:] == 126)
        assert baseline["uniform_answer_vocabulary_token_chance"] == 1 / 126


@pytest.mark.parametrize("task", TASKS)
def test_default_generation_still_matches_retained_original_corpus_when_present(task):
    from pathlib import Path
    root = Path(".runtime/cdrm-naive/20260907T123830Z/data")
    if not (root / task / "train.manifest.json").exists():
        pytest.skip("Original retained corpus is not present in this checkout")
    expected = load_dataset(root, task, "train").take(slice(0, 16))
    actual = generate_dataset(task, "train", 12345, 16)
    assert actual.sha256 == expected.sha256
    np.testing.assert_array_equal(actual.labels, expected.labels)


def test_append_final_is_nonoverwriting_and_preserves_parent_and_epoch_order(tmp_path):
    root = tmp_path / "screen"
    sizes = {"train": 12, "dev": 5, "final": 5}
    prepare(root, sizes=sizes, setting="recall-v128-t128", splits=("train", "dev"))
    old_files = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not (root / "in-context-recall/final.npz").exists()
    supplement = prepare(root, sizes=sizes, setting="recall-v128-t128", splits=("final",), append_splits=True)
    assert all((root / path).read_bytes() == contents for path, contents in old_files.items())
    assert (root / "manifest.added-final.json").exists()
    assert supplement["tasks"]["in-context-recall"]["overlap_audit"]["no_exact_input_cross_split_overlap"]
    final = load_dataset(root, "in-context-recall", "final")
    assert final.manifest["seed"] == 134567
    with pytest.raises(FileExistsError):
        prepare(root, sizes=sizes, setting="recall-v128-t128", splits=("final",), append_splits=True)


@pytest.mark.parametrize("overrides", [{"noise_vocab_size": 1}, {"vocab_size": 15}, {"seq_len": 129}])
def test_unsupported_or_inconsistent_recall_settings_fail_explicitly(overrides):
    with pytest.raises(ValueError):
        task_config("in-context-recall", overrides)
