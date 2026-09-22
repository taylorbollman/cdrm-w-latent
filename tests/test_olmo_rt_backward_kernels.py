"""CPU metadata contracts and frozen arithmetic reference for the RT dK/dV tile."""

import builtins
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.olmo_rt_backward_kernels import _validate_backward_tile_metadata, backward_tile
from cdrm.pretrained.olmo_tiled import _Invocation, _mm


def backward_tile_fixture(rows=3, columns=5, dim=16, strided=False):
    generator = torch.Generator().manual_seed(34197)
    def sample(shape, dtype):
        if strided:
            storage = torch.randn(tuple(2 * extent for extent in shape), generator=generator, dtype=dtype)
            return storage[tuple(slice(None, None, 2) for _ in shape)]
        return torch.randn(shape, generator=generator, dtype=dtype)
    p = sample((2, 2, rows, columns), torch.float32)
    # Deliberately non-normalized fractions exercise exactly the supplied tile,
    # which usually contains only part of each row's attention probability mass.
    p.abs_().mul_(.15)
    p[..., ::2] = 0
    g = sample((2, 2, rows, dim), torch.float32)
    v = sample((2, 2, columns, dim), torch.bfloat16)
    q = sample((2, 2, rows, dim), torch.bfloat16)
    dot = sample((2, 2, rows), torch.float32)
    return p, g, v, q, dot


def frozen_backward_reference(p, g, v, q, dot):
    """Independent eager mixed algebra, with no fused or RT helper calls."""
    with torch.autocast(p.device.type, enabled=False):
        gv = (g.bfloat16() @ v.bfloat16().transpose(-1, -2)).float()
        error = p.float() * (gv - dot.float().unsqueeze(-1))
        dkey = (error.bfloat16().transpose(-1, -2) @ q.bfloat16()).float() / math.sqrt(q.shape[-1])
        dvalue = (p.bfloat16().transpose(-1, -2) @ g.bfloat16()).float()
    return dkey, dvalue


@pytest.mark.parametrize("dim", [16, 32, 64, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_independent_reference_matches_original_reverse_loop_boundaries(dim, strided):
    p, g, v, q, dot = backward_tile_fixture(dim=dim, strided=strided)
    assert _validate_backward_tile_metadata(p, g, v, q, dot) == (2, 2, 3, 5, dim)
    spec = _Invocation(SimpleNamespace(head_dim=dim), 1.0, "mixed", False, torch.bfloat16)
    dvalue = _mm(p.transpose(-1, -2), g, spec, torch.bfloat16)
    error = p * (_mm(g, v.transpose(-1, -2), spec, torch.bfloat16) - dot.unsqueeze(-1))
    dkey = _mm(error.transpose(-1, -2), q, spec, torch.bfloat16) / math.sqrt(dim)
    for actual, expected in zip((dkey, dvalue), frozen_backward_reference(p, g, v, q, dot)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_probability_columns_receive_no_historical_key_or_value_credit():
    tensors = list(backward_tile_fixture())
    expected = frozen_backward_reference(*tensors)
    for result in expected:
        assert torch.count_nonzero(result[..., ::2, :]) == 0
    tensors[0].zero_()
    for result in frozen_backward_reference(*tensors):
        assert torch.count_nonzero(result) == 0
        assert torch.isfinite(result).all()


def test_single_active_row_has_independent_exact_closed_form():
    p = torch.ones(1, 1, 1, 1)
    g = torch.full((1, 1, 1, 16), .5)
    v = torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16)
    q = torch.full_like(v, 2.)
    dot = torch.full((1, 1, 1), .25)
    dkey, dvalue = frozen_backward_reference(p, g, v, q, dot)
    torch.testing.assert_close(dkey, torch.full_like(dkey, -.125), rtol=0, atol=0)
    torch.testing.assert_close(dvalue, torch.full_like(dvalue, .5), rtol=0, atol=0)


@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 256), (256, 1), (256, 256), (17, 33)])
def test_supported_tile_extents_are_metadata_valid(rows, columns):
    assert _validate_backward_tile_metadata(*backward_tile_fixture(rows, columns)) == (2, 2, rows, columns, 16)


def test_broadcast_input_strides_are_supported_read_only_metadata():
    values = list(backward_tile_fixture())
    values[0] = values[0][:1, :1].expand(2, 2, 3, 5)
    values[4] = values[4][:1, :1].expand(2, 2, 3)
    assert _validate_backward_tile_metadata(*values) == (2, 2, 3, 5, 16)


@pytest.mark.parametrize("index,dtype", [(0, torch.bfloat16), (1, torch.bfloat16),
    (2, torch.float32), (3, torch.float16), (4, torch.float64)])
def test_wrong_dtypes_rejected(index, dtype):
    values = list(backward_tile_fixture()); values[index] = values[index].to(dtype)
    with pytest.raises(ValueError):
        _validate_backward_tile_metadata(*values)


@pytest.mark.parametrize("index", range(5))
def test_wrong_ranks_rejected(index):
    values = list(backward_tile_fixture()); values[index] = values[index].unsqueeze(0)
    with pytest.raises(ValueError):
        _validate_backward_tile_metadata(*values)


@pytest.mark.parametrize("index", range(5))
def test_mismatched_batch_or_attention_extents_rejected(index):
    values = list(backward_tile_fixture()); values[index] = values[index][:1]
    with pytest.raises(ValueError):
        _validate_backward_tile_metadata(*values)


@pytest.mark.parametrize("rows,columns,dim", [(0, 3, 16), (3, 0, 16),
    (257, 3, 16), (3, 257, 16), (3, 5, 8), (3, 5, 256)])
def test_unsupported_extents_rejected(rows, columns, dim):
    with pytest.raises(ValueError):
        _validate_backward_tile_metadata(*backward_tile_fixture(rows, columns, dim))


def test_cpu_rejected_before_importing_triton(monkeypatch):
    def unexpected():
        raise AssertionError("CPU input attempted Triton import or compile")
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_backward_kernels._get_backward_tile_kernel", unexpected)
    with pytest.raises(ValueError, match="requires CUDA"):
        backward_tile(*backward_tile_fixture())


def test_module_import_does_not_import_triton(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise AssertionError("CPU module import unexpectedly loaded Triton")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    path = Path(__file__).parents[1] / "cdrm/pretrained/olmo_rt_backward_kernels.py"
    spec = importlib.util.spec_from_file_location("isolated_backward_kernel_import", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize("index", range(5))
def test_autograd_is_not_silently_dropped(index):
    values = list(backward_tile_fixture()); values[index].requires_grad_(True)
    with pytest.raises(ValueError, match="no autograd"):
        _validate_backward_tile_metadata(*values)
    with torch.no_grad():
        assert _validate_backward_tile_metadata(*values) == (2, 2, 3, 5, 16)


def test_device_mismatch_and_non_tensor_input_rejected():
    values = list(backward_tile_fixture()); values[4] = values[4].to("meta")
    with pytest.raises(ValueError, match="share one device"):
        _validate_backward_tile_metadata(*values)
    values = list(backward_tile_fixture()); values[0] = None
    with pytest.raises(TypeError, match="must all be tensors"):
        _validate_backward_tile_metadata(*values)
