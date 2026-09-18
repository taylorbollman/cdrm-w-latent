"""Frozen held-out lengths must preserve native masks and reject corruption."""
import json

import numpy as np
import pytest

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, generate_dataset, save_dataset
from cdrm.rt_nextlat_fuzzy_metrics import evaluation_metadata
from scripts.rt_nextlat_fuzzy_length_prepare import prepare, verify_manifest


def reference_data(tmp_path):
    directory = tmp_path / "reference"
    for split, seed in (("train", 10), ("dev", 11)):
        save_dataset(directory, generate_dataset(FUZZY_TASK, split, seed, 3, {"seq_len": 32}))
    return directory


def test_shared_probes_preserve_masks_denominators_and_only_generate_dev(tmp_path):
    output = tmp_path / "probes"
    reference = reference_data(tmp_path)
    manifest = prepare(output, lengths=(48, 64), seeds=(20, 21), dev_examples=7,
                       reference_data_dir=reference, representative_examples=3)
    checked, datasets = verify_manifest(output)
    assert checked == manifest
    assert set(datasets) == {48, 64}
    assert not manifest["training_split_generated"]
    assert not manifest["final_split_generated"]
    assert not manifest["models_used"]
    assert manifest["reference_overlap_audit"]["no_exact_full_input_reference_overlap"]
    for length, dataset in datasets.items():
        record = manifest["lengths"][str(length)]
        expected = generate_dataset(FUZZY_TASK, "dev", record["seed"], 7, {"seq_len": length})
        assert dataset.sha256 == expected.sha256
        np.testing.assert_array_equal(dataset.labels, dataset.answer_labels)
        assert dataset.input_ids.max() <= 15
        assert record["answer_scored_tokens"] == (dataset.labels != IGNORE_INDEX).sum()
        assert record["total_native_padding_tokens"] == (dataset.input_ids == 15).sum()
        assert record["evaluation_regions"] == {
            name: int(mask.sum()) for name, mask in evaluation_metadata(dataset)["masks"].items()}
        assert sorted(path.name for path in (output / f"length-{length}" / FUZZY_TASK).iterdir()) == [
            "dev.manifest.json", "dev.metadata.json", "dev.npz"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(output, lengths=(48, 64), seeds=(20, 21), reference_data_dir=reference)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("lengths,seeds", [((32,), (20,)), ((24,), (20,)), ((48, 48), (20, 21)),
                                         ((48, 64), (20, 20)), ((48,), (10,)),
                                         ((48,), (2**32,))])
def test_invalid_scope_or_seed_fails_without_creating_output(tmp_path, lengths, seeds):
    reference = reference_data(tmp_path)
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        prepare(output, lengths=lengths, seeds=seeds, reference_data_dir=reference)
    assert not output.exists()


def test_verifier_rejects_changed_arrays_and_nondev_split(tmp_path):
    output = tmp_path / "probes"
    prepare(output, lengths=(48,), seeds=(20,), dev_examples=3,
            reference_data_dir=reference_data(tmp_path), representative_examples=2)
    directory = output / "length-48" / FUZZY_TASK
    unexpected = directory / "train.manifest.json"
    unexpected.write_text("{}")
    with pytest.raises(ValueError, match="Only the development"):
        verify_manifest(output)
    unexpected.unlink()
    arrays = directory / "dev.npz"
    original = arrays.read_bytes()
    arrays.write_bytes(original + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(output)


def test_verifier_rejects_top_level_claiming_training_data(tmp_path):
    output = tmp_path / "probes"
    prepare(output, lengths=(48,), seeds=(20,), dev_examples=3,
            reference_data_dir=reference_data(tmp_path), representative_examples=2)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training_split_generated"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation-only"):
        verify_manifest(output)
