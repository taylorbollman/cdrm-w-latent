"""CPU contract and independent frozen-operand reference for the CUDA helper.

GPU numerical evidence belongs to the opt-in actual-runtime harness; these tests
never pretend to execute the fused kernel on a CPU.
"""

import builtins
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.olmo_rt_kernels import _validate_tile_metadata, add_tile
from cdrm.pretrained.olmo_tiled import _Invocation, _add_tile


def tile_fixture(target=3, source=5, dim=16, strided=False):
    generator = torch.Generator().manual_seed(4971)
    def sample(shape, dtype):
        if strided:
            full = torch.randn(tuple(2 * size for size in shape), generator=generator, dtype=dtype)
            return full[tuple(slice(None, None, 2) for _ in shape)]
        return torch.randn(shape, generator=generator, dtype=dtype)
    q = sample((2, 2, target, dim), torch.bfloat16)
    k = sample((2, 2, source, dim), torch.bfloat16)
    v = sample(k.shape, torch.bfloat16)
    n = sample(q.shape, torch.float32)
    m = sample(q.shape[:-1], torch.float32)
    d = sample(m.shape, torch.float32).abs() + 1.0
    valid = torch.ones(2, source, dtype=torch.bool)
    valid[0, ::2] = False
    return q, k, v, valid, n, m, d


def frozen_operand_reference(q, k, v, valid, numerator, maximum, denominator):
    """Independent CPU reference with explicit eager mixed cast boundaries."""
    with torch.autocast(q.device.type, enabled=False):
        scores = (q.bfloat16() @ k.bfloat16().transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
        scores = torch.where(valid[:, None, None, :], scores, -torch.inf)
        new_max = torch.maximum(maximum, scores.amax(-1))
        origin = torch.where(torch.isfinite(new_max), new_max, 0.0)
        old_scale = torch.exp(maximum - origin)
        weights = torch.exp(scores - origin[..., None])
        addition = (weights.bfloat16() @ v.bfloat16()).float()
        return (numerator * old_scale[..., None] + addition,
                new_max, denominator * old_scale + weights.sum(-1))


@pytest.mark.parametrize("dim", [16, 32, 64, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_supported_metadata_and_independent_reference_match_eager(dim, strided):
    tensors = tile_fixture(dim=dim, strided=strided)
    assert _validate_tile_metadata(*tensors) == (2, 2, 3, 5, dim)
    spec = _Invocation(SimpleNamespace(head_dim=dim), 1.0, "mixed", False, torch.bfloat16)
    actual = _add_tile(*tensors, spec, torch.bfloat16)
    expected = frozen_operand_reference(*tensors)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("source,target", [(1, 1), (1, 256), (256, 1), (256, 256), (17, 33)])
def test_metadata_accepts_complete_supported_length_range(source, target):
    assert _validate_tile_metadata(*tile_fixture(source=source, target=target)) == (2, 2, target, source, 16)


def test_broadcast_strides_are_read_only_supported_inputs():
    tensors = list(tile_fixture())
    tensors[0] = tensors[0][:1, :1].expand(2, 2, 3, 16)
    tensors[3] = tensors[3][:1].expand(2, 5)
    assert _validate_tile_metadata(*tensors) == (2, 2, 3, 5, 16)


@pytest.mark.parametrize("empty_state", [False, True])
def test_all_masked_history_is_zero_contribution(empty_state):
    tensors = list(tile_fixture())
    tensors[3].zero_()
    if empty_state:
        tensors[4].zero_(); tensors[5].fill_(-torch.inf); tensors[6].zero_()
    expected = tuple(tensors[4:])
    observed = frozen_operand_reference(*tensors)
    for got, want in zip(observed, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert not torch.isnan(got).any()


@pytest.mark.parametrize("index,dtype", [(0, torch.float32), (1, torch.float32),
    (2, torch.float16), (3, torch.float32), (4, torch.bfloat16),
    (5, torch.bfloat16), (6, torch.float64)])
def test_wrong_dtypes_rejected(index, dtype):
    tensors = list(tile_fixture()); tensors[index] = tensors[index].to(dtype)
    with pytest.raises(ValueError):
        _validate_tile_metadata(*tensors)


@pytest.mark.parametrize("index", range(7))
def test_wrong_shapes_rejected(index):
    tensors = list(tile_fixture()); tensors[index] = tensors[index].unsqueeze(0)
    with pytest.raises(ValueError):
        _validate_tile_metadata(*tensors)


@pytest.mark.parametrize("target,source,dim", [(0, 4, 16), (3, 0, 16),
    (257, 4, 16), (3, 257, 16), (3, 4, 8), (3, 4, 256)])
def test_unsupported_extents_rejected(target, source, dim):
    with pytest.raises(ValueError):
        _validate_tile_metadata(*tile_fixture(target, source, dim))


def test_cpu_rejected_before_loading_triton(monkeypatch):
    def unexpected():
        raise AssertionError("CPU call attempted to import/compile Triton")
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_kernels._get_tile_kernel", unexpected)
    with pytest.raises(ValueError, match="requires CUDA"):
        add_tile(*tile_fixture())


def test_module_import_does_not_import_triton(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise AssertionError("Triton imported at CPU module-import time")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    path = Path(__file__).parents[1] / "cdrm/pretrained/olmo_rt_kernels.py"
    spec = importlib.util.spec_from_file_location("isolated_rt_kernel_import", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_forward_only_contract_rejects_silent_missing_autograd():
    tensors = list(tile_fixture()); tensors[0].requires_grad_(True)
    with pytest.raises(ValueError, match="forward-only"):
        _validate_tile_metadata(*tensors)
    with torch.no_grad():
        assert _validate_tile_metadata(*tensors) == (2, 2, 3, 5, 16)


def test_mixed_device_metadata_and_non_tensor_rejected():
    tensors = list(tile_fixture()); tensors[3] = tensors[3].to("meta")
    with pytest.raises(ValueError, match="share one device"):
        _validate_tile_metadata(*tensors)
    tensors = list(tile_fixture()); tensors[3] = None
    with pytest.raises(TypeError, match="must all be tensors"):
        _validate_tile_metadata(*tensors)
