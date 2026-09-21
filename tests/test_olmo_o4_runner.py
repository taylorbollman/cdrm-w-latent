"""Bounded O4 runner checks with no GPU, cloud requests, or training runs."""

import base64
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import olmo_o4_common as common
from scripts import olmo_o4_train as runner
from cdrm.pretrained.recurrent import RTMode


class FakeBlob:
    def __init__(self, name):
        self.name = name
        self.present = False
        self.generation = "314"
        self.metadata = {}
        self.upload_calls = []
        self.reload_calls = 0
        self.corrupt_after_upload = None

    def exists(self):
        return self.present

    def upload_from_filename(self, filename, *, if_generation_match, timeout, checksum):
        assert not self.present
        assert if_generation_match == 0
        assert checksum == "md5"
        assert timeout > 0
        self.upload_calls.append((filename, if_generation_match, checksum))
        self.data = Path(filename).read_bytes()
        self.size = len(self.data)
        self.md5_hash = base64.b64encode(hashlib.md5(self.data).digest()).decode()
        self.present = True
        if self.corrupt_after_upload == "md5":
            self.md5_hash = "incorrect-server-checksum"

    def reload(self):
        self.reload_calls += 1


@pytest.fixture
def cloud(monkeypatch):
    from google.cloud import storage
    objects, bucket_names, client_calls = {}, [], []

    def bucket(name):
        bucket_names.append(name)
        return SimpleNamespace(blob=lambda key: objects.setdefault(key, FakeBlob(key)))

    def client():
        client_calls.append(True)
        return SimpleNamespace(bucket=bucket)

    monkeypatch.setattr(storage, "Client", client)
    return SimpleNamespace(objects=objects, bucket_names=bucket_names, client_calls=client_calls)


def test_retention_uses_create_only_upload_and_reuses_verified_identical_object(tmp_path, cloud):
    path = tmp_path / "update-000100.pt"
    path.write_bytes(b"test checkpoint content\x00\xff")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    uri = "gs://fast-chunks/o4/rt-nextlat/update-000100.pt"
    first = common.retain_file(path, uri, expected_sha256=digest)
    second = common.retain_file(path, uri, expected_sha256=digest)
    assert first == second
    assert first["uri"] == uri and first["generation"] == "314"
    assert first["sha256"] == digest and first["size_bytes"] == path.stat().st_size
    blob = cloud.objects["o4/rt-nextlat/update-000100.pt"]
    assert len(blob.upload_calls) == 1
    assert blob.reload_calls == 2
    assert blob.metadata["sha256"] == digest
    assert blob.md5_hash == first["md5_base64"]
    assert cloud.bucket_names == ["fast-chunks", "fast-chunks"]


@pytest.mark.parametrize("mismatch", ["size", "md5", "sha256"])
def test_retention_refuses_to_overwrite_conflicting_existing_object(tmp_path, cloud, mismatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"selected checkpoint")
    uri = "gs://fast-chunks/o4/checkpoint.pt"
    common.retain_file(path, uri)
    blob = cloud.objects["o4/checkpoint.pt"]
    if mismatch == "size":
        blob.size += 1
    elif mismatch == "md5":
        blob.md5_hash = "wrong-md5"
    else:
        blob.metadata["sha256"] = "0" * 64
    snapshot = (blob.size, blob.md5_hash, dict(blob.metadata))
    with pytest.raises(ValueError, match="remote object differs"):
        common.retain_file(path, uri)
    assert len(blob.upload_calls) == 1
    assert (blob.size, blob.md5_hash, blob.metadata) == snapshot


def test_retention_rejects_changed_local_checkpoint_before_creating_cloud_client(tmp_path, cloud):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"new bytes")
    with pytest.raises(ValueError, match="changed before retention"):
        common.retain_file(path, "gs://fast-chunks/o4/checkpoint.pt", expected_sha256="0" * 64)
    assert not cloud.client_calls


def test_retention_verifies_server_checksum_after_upload(tmp_path, cloud):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"bytes")
    blob = cloud.objects.setdefault("o4/checkpoint.pt", FakeBlob("o4/checkpoint.pt"))
    blob.corrupt_after_upload = "md5"
    with pytest.raises(ValueError, match="remote object differs"):
        common.retain_file(path, "gs://fast-chunks/o4/checkpoint.pt")
    assert len(blob.upload_calls) == 1


@pytest.mark.parametrize("invalid", ["symlink", "missing", "http", "traversal"])
def test_retention_rejects_invalid_sources_and_keys_before_cloud_calls(tmp_path, cloud, invalid):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"bytes")
    uri = "gs://fast-chunks/o4/checkpoint.pt"
    if invalid == "symlink":
        link = tmp_path / "link.pt"
        link.symlink_to(path)
        path = link
    elif invalid == "missing":
        path = tmp_path / "missing.pt"
    elif invalid == "http":
        uri = "https://example.org/checkpoint.pt"
    else:
        uri = "gs://fast-chunks/o4/../checkpoint.pt"
    with pytest.raises(ValueError):
        common.retain_file(path, uri)
    assert not cloud.client_calls


@pytest.mark.parametrize("code,retention,expected", [
    (4.0, 5.0, True),
    (4.0, 3.0, False),
    (2.0, 5.0, False),
    (3.5, 4.5, False),
    (3.5001, 4.5001, True),
    (1.0, 2.0, False),
])
def test_catastrophic_gate_requires_both_domains_strictly_exceed_margin(code, retention, expected):
    initial = {"dev": {"mean_nll": 2.0}, "retention_dev": {"mean_nll": 3.0}}
    current = {"dev": {"mean_nll": code}, "retention_dev": {"mean_nll": retention}}
    assert runner.catastrophic(current, initial, 1.5) is expected


@pytest.mark.parametrize("raises", [False, True])
def test_preserve_rng_restores_python_numpy_torch_and_mock_cuda_on_exit(monkeypatch, raises):
    random.seed(414)
    np.random.seed(415)
    torch.manual_seed(416)
    # CUDA methods are mocked so this remains an intentional CPU-only check.
    cuda_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [value.clone() for value in cuda_state])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: restored.append(state))
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        with common.preserve_rng():
            random.random(); np.random.rand(8); torch.rand(8)
            if raises:
                raise RuntimeError("simulated SDK failure")
    except RuntimeError:
        assert raises
    assert random.getstate() == py
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy[0]
    assert np.array_equal(actual_numpy[1], numpy[1])
    assert actual_numpy[2:] == numpy[2:]
    assert torch.equal(torch.get_rng_state(), cpu)
    assert len(restored) == 1 and torch.equal(restored[0][0], cuda_state[0])


def test_source_inventory_covers_model_objective_data_schedule_tracking_and_runner(tmp_path, monkeypatch):
    required = {
        "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_tiled.py",
        "cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py",
        "cdrm/pretrained/lm_data.py", "cdrm/pretrained/lm_schedule.py",
        "cdrm/pretrained/lm_evaluation.py", "scripts/experiment_tracking.py",
        "scripts/olmo_lm_common.py", "scripts/olmo_o4_common.py",
        "scripts/olmo_o4_train.py", "scripts/olmo_o4_preflight.py",
    }
    assert required <= set(common.SOURCE_FILES)
    assert len(common.SOURCE_FILES) == len(set(common.SOURCE_FILES))
    for name in common.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source bytes for " + name)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    initial = common.source_hashes()
    for name, digest in initial.items():
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    changed = "cdrm/pretrained/lm_data.py"
    (tmp_path / changed).write_text("changed preprocessing")
    after = common.source_hashes()
    assert {name for name in initial if initial[name] != after[name]} == {changed}


def test_canonical_identity_changes_with_data_or_source_hash_but_not_dictionary_order():
    first = {"data_manifest_sha256": "a" * 64, "code": {"b.py": "b" * 64, "a.py": "c" * 64}}
    reordered = {"code": {"a.py": "c" * 64, "b.py": "b" * 64}, "data_manifest_sha256": "a" * 64}
    assert common.canonical_digest(first) == common.canonical_digest(reordered)
    assert common.canonical_digest(first) != common.canonical_digest({**first, "data_manifest_sha256": "d" * 64})
    assert common.canonical_digest(first) != common.canonical_digest({**first, "code": {"b.py": "e" * 64, "a.py": "c" * 64}})


def test_evaluation_uses_only_fixed_development_prefixes_and_never_reserved_test(monkeypatch):
    requests = []

    def batch(split, indices, device):
        indices = list(indices)
        requests.append((split, indices, device))
        return (split, indices)

    corpus = SimpleNamespace(split_sizes={"dev": 5, "retention_dev": 2, "test": 100}, batch=batch)
    mode = RTMode((0,), 0.37)
    model = object()

    def evaluate(actual_model, batches, **kwargs):
        assert actual_model is model
        assert kwargs == {"mode": mode, "precision": "bf16_mixed", "include_document_records": True}
        rows = list(batches)
        return {"ce_count": sum(len(indices) for _, indices in rows), "mean_nll": 1.0}

    monkeypatch.setattr(runner, "evaluate_batches", evaluate)
    result = runner.evaluation(model, corpus, {"eval_batch_size": 2, "precision": "bf16_mixed"}, mode, 3,
                               documents=True)
    assert set(result) == {"dev", "retention_dev"}
    assert requests == [("dev", [0, 1], "cuda"), ("dev", [2], "cuda"), ("retention_dev", [0, 1], "cuda")]
    assert result["dev"]["ce_count"] == 3 and result["retention_dev"]["ce_count"] == 2


def test_event_records_reject_nonfinite_values_without_writing_invalid_json(tmp_path):
    common.write_event(tmp_path, {"update": 1, "metric": 2.0})
    with pytest.raises(ValueError):
        common.write_event(tmp_path, {"update": 2, "metric": float("nan")})
    rows = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(row) for row in rows] == [{"update": 1, "metric": 2.0}]
