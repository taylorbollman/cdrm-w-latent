"""CPU-only tests for pinned MAD semantics, oracles, identities and fixed splits."""
import copy
import json

import numpy as np
import pytest

from cdrm.mad_data import (
    IGNORE_INDEX, TASKS, answer_oracle, baseline_audit, epoch_indices, generate_dataset,
    load_dataset, official_generators, overlap_audit, recall_prefix_prediction,
    save_dataset, score_predictions, task_config, task_spec, setting_spec, SETTINGS,
    FUZZY_TASK, SUPPORTED_TASKS, fuzzy_answer_annotation, fuzzy_prefix_prediction,
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


@pytest.mark.parametrize("length", [17, 32, 64, 128, 256, 300])
@pytest.mark.parametrize("split", ["train", "dev"])
def test_fuzzy_preserves_native_arrays_masks_lengths_and_distribution(length, split):
    from scripts.cdrm_fuzzy_prepare import structural_audit as fuzzy_audit
    seed = 962000 + 2 * length + (split == "dev")
    overrides = {"seq_len": length}
    actual = generate_dataset(FUZZY_TASK, split, seed, 12, overrides)
    rng = np.random.default_rng(seed)
    native = [official_generators().generate_fuzzy_in_context_recall_instance(
        **task_config(FUZZY_TASK, overrides), rng=rng, is_training=split == "train") for _ in range(12)]
    np.testing.assert_array_equal(actual.input_ids, np.stack([x for x, _ in native]))
    np.testing.assert_array_equal(actual.labels, np.stack([y for _, y in native]))
    assert actual.input_ids.shape == (12, length)
    assert actual.manifest["actual_sequence_length"] == length
    assert not np.any(actual.answer_labels == 15)
    assert fuzzy_audit(actual, prefix_limit=12)["native_alignment_and_masks_exact"]
    baseline = baseline_audit(actual)
    assert baseline["uniform_answer_vocabulary_token_chance"] == 1 / 8
    assert baseline["oracle"]["available_answer_accuracy"] in (None, 1.0)
    assert baseline["oracle"]["unretrievable_answers"] == actual.manifest["oracle_coverage"]["unavailable_tokens"]
    if split == "train":
        assert not np.any(actual.labels == IGNORE_INDEX)
        np.testing.assert_array_equal(actual.labels[:, :-1], actual.input_ids[:, 1:])
    else:
        np.testing.assert_array_equal(actual.labels, actual.answer_labels)
        assert all(len(pair["key"]) == 3 for row in actual.metadata for pair in row["pairs"])


def test_fuzzy_variable_key_training_cannot_replay_eval_to_recover_same_inputs():
    train = generate_dataset(FUZZY_TASK, "train", 87, 16)
    dev = generate_dataset(FUZZY_TASK, "dev", 87, 16)
    assert not np.array_equal(train.input_ids, dev.input_ids)
    assert {len(pair["key"]) for row in train.metadata for pair in row["pairs"]} == {1, 2, 3}
    assert {len(pair["key"]) for row in dev.metadata for pair in row["pairs"]} == {3}


def test_fuzzy_short_key_prediction_is_not_a_causal_supervision_mask():
    # At position3, known key(0,) also prefixes key(0,1): next token is random
    # key continuation1, not the remembered value7. Only position7 is scored.
    tokens = np.array([15, 0, 7, 0, 1, 8, 0, 1])
    labels = answer_oracle(FUZZY_TASK, tokens)
    assert labels.tolist() == [-100] * 7 + [8]
    assert fuzzy_prefix_prediction(tokens[:4]) == 7
    assert labels[3] == IGNORE_INDEX
    assert fuzzy_prefix_prediction(tokens) == 8
    # Future tokens do not alter a prediction on the unchanged prefix.
    changed_future = tokens.copy()
    changed_future[4:] = [2, 9, 0, 2]
    assert fuzzy_prefix_prediction(changed_future[:4]) == 7


def test_fuzzy_partial_final_value_is_recovered_from_an_earlier_complete_mapping():
    tokens = np.array([15, 15, 0, 1, 2, 7, 9, 11, 3, 4, 5, 8, 0, 1, 2, 7, 9])
    labels, metadata = fuzzy_answer_annotation(tokens)
    assert labels[-3:].tolist() == [7, 9, 11]
    assert np.all(labels[:-3] == IGNORE_INDEX)
    assert metadata["left_padding"] == 2
    for index in range(len(tokens) - 3, len(tokens)):
        assert fuzzy_prefix_prediction(tokens[:index + 1]) == labels[index]
    corrupted = tokens.copy()
    corrupted[-1] = 10
    with pytest.raises(ValueError, match="omit exactly one"):
        fuzzy_answer_annotation(corrupted)


@pytest.mark.parametrize("length,split,seed", [(17, "dev", 962035), (256, "train", 962512)])
def test_native_unseen_terminal_probe_is_preserved_and_qualified(length, split, seed):
    dataset = generate_dataset(FUZZY_TASK, split, seed, 1, {"seq_len": length})
    record = dataset.metadata[0]
    assert not record["pairs"][-1]["repeated"]
    assert record["unretrievable_answer_positions"]
    assert dataset.labels[0, -1] in range(7, 15)
    assert dataset.answer_labels[0, -1] == dataset.labels[0, -1]
    assert not dataset.oracle_available_mask[0, -1]
    assert answer_oracle(FUZZY_TASK, dataset.input_ids[0])[-1] == IGNORE_INDEX
    audit = baseline_audit(dataset)
    assert audit["oracle"]["coverage"] < 1
    assert audit["oracle"]["scored_answers"] == (dataset.answer_labels != IGNORE_INDEX).sum()
    assert dataset.manifest["oracle_coverage"]["context_unavailable_tokens"] == 0


def test_fuzzy_unseen_terminal_annotation_never_supplies_oracle_answers():
    tokens = np.array([15, 0, 1, 2, 7, 8, 3, 4, 5, 9])
    oracle, _ = fuzzy_answer_annotation(tokens)
    assert np.all(oracle == IGNORE_INDEX)
    annotated, _ = fuzzy_answer_annotation(tokens, terminal_target=10)
    assert annotated[-2:].tolist() == [9, 10]
    assert fuzzy_prefix_prediction(tokens[:-1]) == IGNORE_INDEX
    assert fuzzy_prefix_prediction(tokens) == IGNORE_INDEX


def test_fuzzy_save_load_coverage_identity_and_legacy_task_defaults(tmp_path):
    assert TASKS == ("in-context-recall", "selective-copying")
    assert FUZZY_TASK in SUPPORTED_TASKS
    dataset = generate_dataset(FUZZY_TASK, "dev", 962035, 12, {"seq_len": 17})
    save_dataset(tmp_path, dataset)
    restored = load_dataset(tmp_path, FUZZY_TASK, "dev")
    assert dataset.sha256 == restored.sha256
    np.testing.assert_array_equal(dataset.oracle_available_mask, restored.oracle_available_mask)
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path, restored)


@pytest.mark.parametrize("overrides", [{"seq_len": 12}, {"vocab_size": 32}, {"k_motif_size": 2}, {"multi_query": False}])
def test_fuzzy_initial_scope_guards_are_explicit(overrides):
    with pytest.raises(ValueError):
        task_config(FUZZY_TASK, overrides)
