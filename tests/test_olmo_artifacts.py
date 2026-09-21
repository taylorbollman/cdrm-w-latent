"""Bounded integrity and native-format checks; no network or GPU required."""
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from cdrm.pretrained import olmo_artifacts as artifacts
from cdrm.pretrained.artifacts import sha256_file


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.mkdir()
    vocabulary = {f"tok_{i}": i for i in range(50280)}
    for index, token in [(0, "<unk>"), (1, "<|padding|>"), (50279, "<|endoftext|>")]:
        vocabulary.pop(f"tok_{index}")
        vocabulary[token] = index
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(native / "tokenizer.json"))
    for filename in ["config.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (native / filename).write_text("{}\n")
    save_file({"model.transformer.wte.weight": torch.arange(12).reshape(4, 3).float()},
              str(native / "model.safetensors"))
    specs = {name: ((native / name).stat().st_size, sha256_file(native / name))
             for name in artifacts.FILE_SPECS}
    monkeypatch.setattr(artifacts, "FILE_SPECS", specs)
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected network call"))
    manifest = artifacts.prepare_artifacts(tmp_path)
    return tmp_path, manifest


def test_native_fixed_geometry_counts():
    shapes = artifacts.native_tensor_shapes()
    assert len(shapes) == 65
    assert sum(torch.Size(s).numel() for s in shapes.values()) == artifacts.PARAMETER_COUNT
    assert shapes["transformer.wte.weight"] == (50304, 2048)
    assert shapes["transformer.blocks.0.ff_proj.weight"] == (16384, 2048)


def test_prepare_reuses_immutable_manifest_and_rechecks(prepared):
    root, manifest = prepared
    path = root / artifacts.MANIFEST_FILENAME
    before = path.read_bytes()
    assert artifacts.prepare_artifacts(root) == manifest
    assert path.read_bytes() == before
    assert manifest["checkpoint"]["path"] == "native/model.safetensors"
    assert artifacts.validate_prepared_manifest(root) == manifest


def test_actual_safetensors_strict_load(prepared):
    root, _ = prepared
    state = artifacts.load_native_state_dict(root, expected_shapes={"transformer.wte.weight": (4, 3)})
    torch.testing.assert_close(state["transformer.wte.weight"], torch.arange(12).reshape(4, 3).float())
    assert state["transformer.wte.weight"].device.type == "cpu"
    with pytest.raises(ValueError, match="key mismatch"):
        artifacts.load_native_state_dict(root)


def test_same_size_tampering_rejected(prepared):
    root, _ = prepared
    path = root / "native/model.safetensors"
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.prepare_artifacts(root)


@pytest.mark.parametrize("field,value,match", [
    ("revision", "another", "provenance"),
    ("raw_native_config", {"weight_tying": False}, "raw config"),
    ("tokenizer", {}, "tokenizer manifest"),
    ("checkpoint", {}, "checkpoint identity"),
    ("artifacts", {}, "artifact records"),
    ("load_contract", {}, "load contract"),
])
def test_manifest_tampering_rejected(prepared, field, value, match):
    root, manifest = prepared
    manifest[field] = value
    (root / artifacts.MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        artifacts.validate_prepared_manifest(root)


def test_tokenizer_adds_only_explicit_eos(prepared):
    root, _ = prepared
    tokenizer = artifacts.load_native_tokenizer(root)
    tokens = tokenizer.encode("tok_2 tok_3")
    assert tokens == [2, 3]
    assert tokenizer.encode("tok_2 tok_3", add_eos=True) == [2, 3, 50279]
    assert tokenizer.encode("") == []
    assert tokenizer.decode(tokens) == "tok_2 tok_3"
    assert tokenizer.encode("cafe\u0301") == tokenizer.encode("café")


def test_tokenizer_hash_tampering_rejected(prepared):
    root, _ = prepared
    path = root / "native/tokenizer_config.json"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="tokenizer file differs"):
        artifacts.load_native_tokenizer(root)


@pytest.mark.parametrize("case", ["wrong_prefix", "extra", "missing", "shape", "bf16", "nan", "not_tensor"])
def test_strict_tensor_rejections(case):
    raw = {"model.transformer.wte.weight": torch.ones(3, 2)}
    expected = {"transformer.wte.weight": (3, 2)}
    if case == "wrong_prefix": raw = {"transformer.wte.weight": torch.ones(3, 2)}
    elif case == "extra": raw["model.lm_head.weight"] = torch.ones(3, 2)
    elif case == "missing": expected["missing"] = (3, 2)
    elif case == "shape": raw["model.transformer.wte.weight"] = torch.ones(2, 3)
    elif case == "bf16": raw["model.transformer.wte.weight"] = torch.ones(3, 2, dtype=torch.bfloat16)
    elif case == "nan": raw["model.transformer.wte.weight"][0, 0] = float("nan")
    elif case == "not_tensor": raw["model.transformer.wte.weight"] = [[1.0]]
    with pytest.raises(ValueError):
        artifacts.validate_native_tensors(raw, expected_shapes=expected)


def test_native_tensor_values_not_reinitialized_or_copied():
    tensor = torch.randn(3, 2)
    state = artifacts.validate_native_tensors({"model.transformer.wte.weight": tensor},
                                             expected_shapes={"transformer.wte.weight": (3, 2)})
    assert state["transformer.wte.weight"] is tensor
