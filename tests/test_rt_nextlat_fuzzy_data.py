"""Native MAD alignment and meaningful aggregation tests for the new pilot."""
import json

import numpy as np
import pytest

from cdrm.mad_data import (
    FUZZY_TASK, IGNORE_INDEX, MadDataset, answer_oracle, fuzzy_answer_annotation,
    generate_dataset, load_dataset, task_spec,
)
from cdrm.rt_nextlat_fuzzy_metrics import FuzzyMetrics, evaluation_metadata
from scripts.rt_nextlat_fuzzy_prepare import prepare, structural_audit


def fixture_dataset():
    # First query repeats an earlier key; the other terminal key is unseen.
    rows = np.array([[15, 15, 15, 0, 1, 2, 7, 8, 3, 4, 5, 9, 0, 1, 2, 7],
                     [15, 15, 15, 0, 1, 2, 7, 8, 3, 4, 5, 9, 2, 3, 4, 7]])
    labels, metadata = [], []
    for row in rows:
        answer, record = fuzzy_answer_annotation(row, {"seq_len": 16}, terminal_target=8)
        record["oracle_available_positions"] = np.flatnonzero(answer_oracle(FUZZY_TASK, row, {"seq_len": 16}) != IGNORE_INDEX).tolist()
        labels.append(answer)
        metadata.append(record)
    labels = np.stack(labels)
    return MadDataset(rows, labels, labels, metadata, {**task_spec(FUZZY_TASK, {"seq_len": 16}), "split": "dev"})


def test_metrics_keep_unseen_terminal_answers_and_separate_first_token():
    data = fixture_dataset()
    metadata = evaluation_metadata(data)
    np.testing.assert_array_equal(metadata["distance"][:, -2:], [[9, 9], [-1, -1]])
    assert metadata["masks"]["answer"].sum() == 4
    assert metadata["masks"]["known_history"].sum() == 2
    assert metadata["masks"]["first_value_token"].sum() == 2
    predictions = np.where(data.answer_labels == IGNORE_INDEX, 0, data.answer_labels)
    predictions[1, -2] = 9
    ce = np.full(data.labels.shape, np.nan)
    ce[data.answer_labels != IGNORE_INDEX] = [1., 2., 3., 4.]
    meter = FuzzyMetrics(data, metadata)
    meter.update(predictions, ce)
    result = meter.compute()
    assert result["answer_accuracy"] == .75
    assert result["answer_ce"] == 2.5
    assert result["sequence_exact_match"] == .5
    assert result["answer_motif_exact_match"] == .5
    assert result["first_value_token_accuracy"] == .5
    assert result["continuation_value_token_accuracy"] == 1.
    assert result["known_history_accuracy"] == 1.
    assert result["unavailable_history_accuracy"] == .5
    assert result["oracle_coverage"] == .5
    assert result["distance_17_64_accuracy"] is None
    assert result["terminal_probe_tokens"] == 4


def test_streaming_metrics_equal_whole_corpus_for_unequal_microbatches():
    data = generate_dataset(FUZZY_TASK, "dev", 731, 9, {"seq_len": 64})
    predictions = np.random.default_rng(800).integers(0, 16, size=data.labels.shape)
    ce = np.random.default_rng(801).random(data.labels.shape)
    whole, streamed = FuzzyMetrics(data), FuzzyMetrics(data)
    whole.update(predictions, ce)
    for indices in (np.arange(1), np.arange(1, 4), np.arange(4, 9)):
        streamed.update(predictions[indices], ce[indices], indices=indices)
    expected, actual = whole.compute(), streamed.compute()
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(expected[key], abs=1e-14)
    streamed.reset()
    assert streamed.compute()["examples"] == 0


def test_native_padding_and_label_shift_survive_preparation(tmp_path):
    output = tmp_path / "fuzzy"
    report = prepare(output, length=32, train_examples=11, dev_examples=7,
                     train_seed=31, dev_seed=32, confirmation_seed=33, shuffle_seed=34,
                     representative_examples=3)
    assert report["status"] == "complete"
    assert not report["final_split_generated"]
    assert report["seed_calendar"]["confirmation"] == {"seed": 33, "examples": 7, "generated": False}
    assert report["native_input_token_offset"] == 0
    assert report["overlap_audit"]["no_exact_input_cross_split_overlap"]
    assert not list(output.rglob("final*"))
    train = load_dataset(output, FUZZY_TASK, "train")
    dev = load_dataset(output, FUZZY_TASK, "dev")
    np.testing.assert_array_equal(train.labels[:, :-1], train.input_ids[:, 1:])
    assert (train.labels == IGNORE_INDEX).sum() == 0
    assert (dev.labels == IGNORE_INDEX).sum() > 0
    assert np.any(train.input_ids == 15)
    assert train.input_ids.max() < 16
    np.testing.assert_array_equal(dev.labels, dev.answer_labels)
    assert structural_audit(train)["native_alignment_and_masks_exact"]
    assert report["splits"]["dev"]["baselines"]["query_ignoring_answer_prefix"]["scored_answers"] == report["splits"]["dev"]["answer_scored_tokens"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(output)
    assert (output / "manifest.json").read_bytes() == before
    assert json.loads(before)["source_snapshots_verified"]


def test_seed_collision_fails_before_creation(tmp_path):
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="distinct"):
        prepare(output, train_seed=4, dev_seed=4)
    assert not output.exists()


def test_metrics_reject_remapped_ids_and_nonfinite_scored_ce():
    data = fixture_dataset()
    meter = FuzzyMetrics(data)
    predictions = np.where(data.labels == IGNORE_INDEX, 0, data.labels)
    with pytest.raises(ValueError, match="native MAD IDs"):
        meter.update(predictions + 60)
    with pytest.raises(ValueError, match="finite"):
        meter.update(predictions, np.full(data.labels.shape, np.nan))
    assert meter.compute()["examples"] == 0
