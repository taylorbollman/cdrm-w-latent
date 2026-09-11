"""CPU mathematical, data-separation, and immutable-artifact checks for A5."""
import json

import numpy as np
import pytest

from scripts import rt_a5_data as data


def _small_dataset(root):
    return data.prepare_dataset(root, pool_size=100, confirmation_size=17,
                                ood_eval_size=13, short_length=2, long_length=4)


def test_alphabet_and_all_pair_products_against_permutation_matrices():
    alphabet = data.alphabet()
    assert alphabet.shape == (60, 5)
    assert [tuple(row) for row in alphabet] == sorted(tuple(row) for row in alphabet)
    assert tuple(alphabet[0]) == (1, 2, 3, 4, 5)
    ids = {tuple(row): index for index, row in enumerate(alphabet)}
    matrices = []
    for row in alphabet:
        assert sorted(row) == [1, 2, 3, 4, 5]
        matrix = np.zeros((5, 5), dtype=np.int64)
        matrix[row - 1, np.arange(5)] = 1
        assert round(np.linalg.det(matrix)) == 1
        matrices.append(matrix)
    table = data.multiplication_table()
    # Matrix multiplication is an independent representation of the action.
    for left in range(60):
        for right in range(60):
            images = np.argmax(matrices[left] @ matrices[right], axis=0) + 1
            assert table[left, right] == ids[tuple(images)]
    np.testing.assert_array_equal(table[0], np.arange(60))
    np.testing.assert_array_equal(table[:, 0], np.arange(60))
    for left in range(60):
        inverse = np.flatnonzero(table[left] == 0)
        assert len(inverse) == 1 and table[inverse[0], left] == 0


def test_associativity_and_noncommuting_prefix_labels():
    table = data.multiplication_table()
    a, b, c = np.indices((60, 60, 60))
    np.testing.assert_array_equal(table[table[a, b], c], table[a, table[b, c]])
    ids = {tuple(row): index for index, row in enumerate(data.alphabet())}
    p = ids[(2, 3, 1, 4, 5)]
    q = ids[(1, 2, 4, 5, 3)]
    pq = ids[(2, 3, 4, 5, 1)]
    qp = ids[(2, 4, 1, 5, 3)]
    assert pq != qp
    inverse_q = int(np.flatnonzero(table[q] == 0)[0])
    words = np.array([[p, q, inverse_q], [q, p, 0], [0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(data.prefix_labels(words),
                                  [[p, pq, p], [q, qp, qp], [0, 0, 0]])
    np.testing.assert_array_equal(data.prefix_labels(words)[:, 0], words[:, 0])
    assert data.prefix_labels(np.empty((0, 12), dtype=np.uint8)).shape == (0, 12)


@pytest.mark.parametrize("words", [np.array([1, 2]), np.array([[60]]),
                                    np.array([[-1]]), np.array([[1.5]]),
                                    np.empty((2, 0), dtype=np.uint8)])
def test_invalid_words_fail_before_silent_cast(words):
    with pytest.raises(ValueError):
        data.prefix_labels(words)


def test_sampling_preserves_draw_order_and_rejects_duplicate_words():
    sampled = data.generate_unique_words(100, 12, 444)
    expected = np.random.Generator(np.random.PCG64(444)).integers(
        0, 60, size=(100, 12), dtype=np.uint8)
    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(sampled, data.generate_unique_words(100, 12, 444))
    assert not np.array_equal(sampled, data.generate_unique_words(100, 12, 445))
    dense, statistics = data.generate_unique_words(60, 1, 42, return_stats=True)
    np.testing.assert_array_equal(np.sort(dense[:, 0]), np.arange(60))
    assert statistics["duplicate_words_rejected"] > 0
    assert statistics["drawn"] == len(dense) + statistics["duplicate_words_rejected"]


def test_prefix_rejection_and_exhausted_word_space():
    forbidden = np.arange(59, dtype=np.uint8).reshape(-1, 1)
    words, statistics = data.generate_unique_words(20, 2, 3,
        forbidden_prefixes=forbidden, return_stats=True)
    assert np.all(words[:, 0] == 59)
    assert statistics["forbidden_prefixes_rejected"] > 0
    assert len(np.unique(words, axis=0)) == 20
    with pytest.raises(ValueError, match="more unique words"):
        data.generate_unique_words(61, 1, 0)
    with pytest.raises(ValueError, match="Too few words"):
        data.generate_unique_words(61, 2, 0, forbidden_prefixes=forbidden)


def test_preparation_freezes_order_hashes_splits_and_fresh_confirmation(tmp_path):
    root = tmp_path / "dataset"
    manifest = _small_dataset(root)
    assert manifest["confirmation_evaluated"] is False
    assert manifest["sampling"]["seed12"] == 444
    assert manifest["sampling"]["seed36"] == 445
    assert manifest["sampling"]["confirmation_seed"] == 446
    assert manifest["roles"]["train36"]["allowed_use"] == "reserved_train_at_36_control"
    expected_rows = {"train": 80, "dev": 20, "train36": 80, "ood_dev": 13, "confirmation": 17}
    for role, rows in expected_rows.items():
        words, labels = data.load_split(root, role)
        assert isinstance(words, np.memmap) and isinstance(labels, np.memmap)
        assert not words.flags.writeable and not labels.flags.writeable
        assert words.dtype == labels.dtype == np.uint8 and len(words) == rows
        np.testing.assert_array_equal(labels, data.prefix_labels(words))
    assert manifest["roles"]["ood_dev"]["stored_rows"] == 20
    shuffled = np.random.RandomState(42).permutation(100)
    np.testing.assert_array_equal(np.load(root / "train.indices.npy"), shuffled[20:])
    np.testing.assert_array_equal(np.load(root / "dev.indices.npy"), shuffled[:20])
    words = data.generate_unique_words(100, 2, 444)
    np.testing.assert_array_equal(data.load_split(root, "train")[0], words[shuffled[20:]])
    for pair, count in manifest["leakage_audit"]["prefix_intersections"].items():
        if pair != "train36__ood_dev":
            assert count == 0
    assert not any(manifest["leakage_audit"]["complete_word_intersections"].values())
    assert not any(manifest["leakage_audit"]["duplicate_words"].values())
    second = _small_dataset(tmp_path / "replica")
    assert manifest == second
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        _small_dataset(root)
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_hash_validation_is_cached_and_rechecked_after_file_mutation(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    manifest = _small_dataset(root)
    original = data.file_sha256
    calls = []

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(data, "file_sha256", counted)
    data._VALIDATED.clear()
    data.load_split(root, "train")
    assert len(calls) == len(manifest["files"])
    data.load_split(root, "dev")
    assert len(calls) == len(manifest["files"])
    path = root / "train.inputs.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="hash/size differs"):
        data.load_split(root, "train")


@pytest.mark.parametrize("mutation", ["traversal", "rows", "shape", "role"])
def test_invalid_manifests_are_rejected(tmp_path, mutation):
    root = tmp_path / "dataset"
    manifest = _small_dataset(root)
    if mutation == "traversal":
        manifest["files"]["../outside.npy"] = manifest["files"].pop("train.inputs.npy")
    elif mutation == "rows":
        manifest["roles"]["train"]["rows"] = 81
    elif mutation == "shape":
        manifest["files"]["train.inputs.npy"]["shape"] = [80, 3]
    else:
        manifest["roles"].pop("confirmation")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.load_split(root, "train")


def test_ineligible_preparation_fails_without_creating_directory(tmp_path):
    for kwargs in ({"seed12": 445}, {"ood_eval_size": 21}, {"long_length": 2}):
        root = tmp_path / next(iter(kwargs))
        options = {"pool_size": 100, "confirmation_size": 17, "ood_eval_size": 13,
                   "short_length": 2, "long_length": 4, **kwargs}
        with pytest.raises(ValueError):
            data.prepare_dataset(root, **options)
        assert not root.exists()
    with pytest.raises(ValueError, match="Unknown A5 split"):
        data.load_split(tmp_path, "test")
