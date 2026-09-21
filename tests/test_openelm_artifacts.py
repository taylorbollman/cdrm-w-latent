"""Artifact integrity/resume and native-checkpoint rejection tests; no GPU needed."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
import torch

from cdrm.pretrained import artifacts


@pytest.fixture
def download_server():
    state = {"payload": b"an immutable model artifact\n" * 37, "etag": '"version-one"', "ignore_range": False, "bad_range": False, "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["payload"])))
            self.send_header("ETag", state["etag"])
            self.end_headers()

        def do_GET(self):
            range_value = self.headers.get("Range")
            state["ranges"].append(range_value)
            offset = int(range_value.split("=")[1].split("-")[0]) if range_value and not state["ignore_range"] else 0
            self.send_response(206 if offset else 200)
            self.send_header("ETag", state["etag"])
            self.send_header("Content-Length", str(len(state["payload"]) - offset))
            if offset:
                start = offset + 1 if state["bad_range"] else offset
                self.send_header("Content-Range", f"bytes {start}-{len(state['payload'])-1}/{len(state['payload'])}")
            self.end_headers()
            self.wfile.write(state["payload"][offset:])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/checkpoint"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def spec_for(server):
    return artifacts.ArtifactSpec(server["url"], len(server["payload"]), hashlib.sha256(server["payload"]).hexdigest())


def seed_partial(path, server, offset=17):
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(server["payload"][:offset])
    artifacts.write_json(partial.with_name(partial.name + ".json"), {
        "url": server["url"], "size_bytes": len(server["payload"]),
        "etag": server["etag"], "last_modified": None,
    })


def test_complete_download_reuse_and_tamper_detection(tmp_path, download_server):
    path = tmp_path / "checkpoint.pt"
    spec = spec_for(download_server)
    record = artifacts.download_artifact(spec, path)
    assert path.read_bytes() == download_server["payload"]
    assert record["sha256"] == spec.sha256
    assert artifacts.download_artifact(spec, path) == record
    assert len(download_server["ranges"]) == 1
    path.write_bytes(b"X" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.download_artifact(spec, path)


@pytest.mark.parametrize("ignore_range", [False, True])
def test_resume_or_restart_when_server_ignores_range(tmp_path, download_server, ignore_range):
    path = tmp_path / "checkpoint.pt"
    seed_partial(path, download_server)
    download_server["ignore_range"] = ignore_range
    artifacts.download_artifact(spec_for(download_server), path)
    assert path.read_bytes() == download_server["payload"]
    assert download_server["ranges"] == ["bytes=17-"]
    assert not path.with_name(path.name + ".part").exists()


def test_resume_rejects_changed_object_and_bad_range(tmp_path, download_server):
    path = tmp_path / "checkpoint.pt"
    seed_partial(path, download_server)
    download_server["etag"] = '"different-version"'
    with pytest.raises(ValueError, match="changed remote identity"):
        artifacts.download_artifact(spec_for(download_server), path)
    assert not path.exists()
    seed_partial(path, download_server)
    download_server["bad_range"] = True
    with pytest.raises(ValueError, match="Content-Range"):
        artifacts.download_artifact(spec_for(download_server), path)
    assert not path.exists()


def test_unmanifested_existing_file_and_wrong_size_are_not_adopted(tmp_path, download_server):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(download_server["payload"])
    with pytest.raises(ValueError, match="no integrity record"):
        artifacts.download_artifact(spec_for(download_server), path)
    path.unlink()
    with pytest.raises(ValueError, match="pinned artifact"):
        artifacts.download_artifact(artifacts.ArtifactSpec(download_server["url"], 1), path)
    assert not path.exists()


def test_digest_failure_does_not_publish(tmp_path, download_server):
    path = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="pinned digest"):
        artifacts.download_artifact(artifacts.ArtifactSpec(download_server["url"], sha256="0" * 64), path)
    assert not path.exists()
    assert path.with_name(path.name + ".part").exists()


def small_state():
    return {"token_embeddings.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4), "norm.weight": torch.ones(4)}


def test_weights_only_native_load_retains_every_row_and_tensor(tmp_path):
    path = tmp_path / "native.pt"
    source = small_state()
    torch.save(source, path)
    shapes = {key: tuple(value.shape) for key, value in source.items()}
    actual = artifacts.load_native_state_dict(path, expected_vocab_size=7, expected_model_dim=4, expected_shapes=shapes)
    assert set(actual) == set(source)
    assert all(torch.equal(source[key], actual[key]) and actual[key].device.type == "cpu" for key in source)
    assert artifacts.state_dict_manifest(actual)["parameter_count"] == 32


@pytest.mark.parametrize("mutation", ["wrapped", "prefix", "cropped", "extra", "missing", "shape", "nonfinite"])
def test_invalid_native_checkpoint_is_rejected(tmp_path, mutation):
    path = tmp_path / "invalid.pt"
    source = small_state()
    shapes = {key: tuple(value.shape) for key, value in source.items()}
    if mutation == "wrapped":
        source = {"model_state_dict": source, "iterations": 299999}
    elif mutation == "prefix":
        source = {"module." + key: value for key, value in source.items()}
    elif mutation == "cropped":
        source["token_embeddings.weight"] = source["token_embeddings.weight"][:-1]
    elif mutation == "extra":
        source["classifier.weight"] = source["token_embeddings.weight"].clone()
    elif mutation == "missing":
        del source["norm.weight"]
    elif mutation == "shape":
        source["norm.weight"] = torch.ones(5)
    else:
        source["norm.weight"][0] = float("nan")
    torch.save(source, path)
    with pytest.raises(ValueError):
        artifacts.load_native_state_dict(path, expected_vocab_size=7, expected_model_dim=4, expected_shapes=shapes)


def test_manifest_distinguishes_incomplete_preparation(tmp_path):
    artifacts.write_json(tmp_path / "source_manifest.json", {"schema_version": 1, "artifacts": {}})
    manifest = artifacts.refresh_manifest(tmp_path)
    assert manifest["preparation_complete"] is False
    assert manifest["native_vocab_size"] == 32128
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="incomplete"):
        artifacts.validate_prepared_manifest(tmp_path)
    manifest["native_vocab_size"] = 32000
    artifacts.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Stage A pins"):
        artifacts.validate_prepared_manifest(tmp_path)


def test_tokenizer_substitution_is_rejected_before_copy(tmp_path):
    unrelated = tmp_path / "different.model"
    unrelated.write_bytes(b"not the official tokenizer")
    with pytest.raises(ValueError, match="refusing substitution"):
        artifacts.prepare_tokenizer(tmp_path / "artifacts", unrelated)
    assert not (tmp_path / "artifacts/tokenizer/tokenizer.model").exists()


def test_real_tokenizer_contract_when_prepared():
    path = Path(__file__).resolve().parents[1] / ".runtime/openelm-import/artifacts/tokenizer/tokenizer.model"
    if not path.exists():
        pytest.skip("Official tokenizer has not been prepared")
    pytest.importorskip("ftfy")
    tokenizer = artifacts.OpenELMTokenizer(path)
    ids = tokenizer.encode("A compact world model.")
    assert ids[0] == 1 and ids[-1] == 2
    assert all(0 <= token < 32000 for token in ids)
    assert tokenizer.encode("caf\u00e9") == tokenizer.encode("cafe\u0301")
    assert tokenizer.encode("caf\u00c3\u00a9") == tokenizer.encode("caf\u00e9")
    assert tokenizer.decode(ids[1:-1]) == "A compact world model."
    with pytest.raises(ValueError, match="padding"):
        tokenizer.decode([32000])
