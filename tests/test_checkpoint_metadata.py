"""Checkpoint metadata must never require custom globals during safe loading."""
from dataclasses import dataclass
from enum import Enum
import io
import json
from pathlib import Path

import pytest
import torch
from torch.torch_version import TorchVersion

from cdrm.pretrained import distributed_checkpoint
from cdrm.pretrained.lm_training import _plain, TrainingCounters, build_adamw


class StringValue(str):
    pass


class IntegerValue(int):
    pass


class FloatValue(float):
    pass


class Choice(Enum):
    first = StringValue("first")


@dataclass
class Runtime:
    version: str
    steps: int


def assert_builtin_tree(value):
    if isinstance(value, dict):
        assert type(value) is dict
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_builtin_tree(item)
    elif isinstance(value, list):
        assert type(value) is list
        for item in value:
            assert_builtin_tree(item)
    else:
        assert type(value) in (str, bool, int, float, type(None))


def metadata():
    return {StringValue("runtime"): Runtime(TorchVersion("2.13.0a0+fixture"), IntegerValue(3)),
            StringValue("values"): (FloatValue(0.25), IntegerValue(7), StringValue("x"),
                                    True, False, None, Choice.first, Path("fixture"), torch.float32)}


def test_plain_strips_scalar_and_key_subclasses_and_keeps_values():
    result = _plain(metadata())
    assert_builtin_tree(result)
    assert result == {"runtime": {"version": "2.13.0a0+fixture", "steps": 3},
                      "values": [0.25, 7, "x", True, False, None, "first", "fixture", "torch.float32"]}
    assert json.loads(json.dumps(result)) == result
    stream = io.BytesIO()
    torch.save(result, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert_builtin_tree(restored)
    assert restored == result


@pytest.mark.parametrize("value", [FloatValue(float("nan")), FloatValue(float("inf")),
                                  FloatValue(float("-inf")), object(), torch.tensor(1.)])
def test_plain_still_rejects_nonfinite_or_nonmetadata_values(value):
    with pytest.raises(ValueError, match="Unsupported configuration"):
        _plain({"nested": [value]})


def test_plain_still_rejects_nonstring_mapping_keys():
    with pytest.raises(ValueError, match="keys must be strings"):
        _plain({IntegerValue(1): "value"})


def test_distributed_checkpoint_safely_loads_version_and_scalar_subclass_metadata(tmp_path, monkeypatch):
    # Exercise the real atomic saver/validator/weights_only loader; only the
    # rank coordination is local here. Actual NCCL restart has its own probe.
    monkeypatch.setattr(distributed_checkpoint, "_context", lambda group: (0, 1))
    monkeypatch.setattr(distributed_checkpoint, "_gather", lambda value, group: [value])
    model = torch.nn.Linear(3, 2)
    optimizer = build_adamw(model, lr=0.003, fused=False)
    configuration = metadata()
    source = {StringValue("checkpoint_sha256"): StringValue("a" * 64), "torch": torch.__version__}
    cursor = {StringValue("offset"): IntegerValue(4), "fraction": FloatValue(0.5)}
    path = tmp_path / "checkpoint"
    receipt = distributed_checkpoint.save_distributed_checkpoint(path, model, optimizer,
        counters=TrainingCounters(), data_cursor=cursor, configuration=configuration,
        source_fingerprint=source, device="cpu")
    payload = torch.load(path / distributed_checkpoint.STATE_FILENAME, weights_only=True)
    for key in ("configuration", "source_fingerprint"):
        assert_builtin_tree(payload["metadata"][key])
    assert_builtin_tree(payload["rank_states"][0]["data_cursor"])
    other = torch.nn.Linear(3, 2)
    other_optimizer = build_adamw(other, lr=0.003, fused=False)
    restored = distributed_checkpoint.load_distributed_checkpoint(path, other, other_optimizer,
        configuration=configuration, source_fingerprint=source, device="cpu",
        expected_manifest_sha256=receipt["manifest_sha256"])
    assert restored["configuration"] == _plain(configuration)
    assert restored["data_cursor"] == {"offset": 4, "fraction": 0.5}
    assert_builtin_tree(restored["data_cursor"])
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(other.state_dict()[name], parameter, rtol=0, atol=0)
