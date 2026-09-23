"""CPU contracts and independent BF16 algebra for probability recomputation."""

import builtins
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.olmo_rt_recompute_kernels import (
    _validate_recomputed_tile_metadata, backward_recomputed_tile,
)
from cdrm.pretrained.olmo_tiled import _Invocation, _mm


def recomputed_tile_fixture(rows=3, columns=5, dim=16, strided=False):
    generator = torch.Generator().manual_seed(49321)

    def sample(shape, dtype):
        if strided:
            storage = torch.randn(tuple(2 * extent for extent in shape), generator=generator, dtype=dtype)
            return storage[tuple(slice(None, None, 2) for _ in shape)]
        return torch.randn(shape, generator=generator, dtype=dtype)

    q = sample((2, 2, rows, dim), torch.bfloat16)
    k = sample((2, 2, columns, dim), torch.bfloat16)
    v = sample((2, 2, columns, dim), torch.bfloat16)
    g = sample((2, 2, rows, dim), torch.float32)
    dot = sample((2, 2, rows), torch.float32)
    maximum = sample((2, 2, rows), torch.float32).abs_().add_(5)
    denominator = sample((2, 2, rows), torch.float32).abs_().add_(2)
    validity = torch.ones((2, 2 * columns if strided else columns), dtype=torch.bool)
    valid = validity[:, ::2] if strided else validity
    valid[:, ::2] = False
    return q, k, v, g, dot, maximum, denominator, valid


def frozen_recomputed_reference(q, k, v, g, dot, maximum, denominator, valid):
    """Independent eager algebra with full reductions and explicit boundaries."""
    with torch.autocast(q.device.type, enabled=False):
        scores = (q.bfloat16() @ k.bfloat16().transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
        safe_denominator = torch.where(denominator > 0, denominator, 1)
        p = (scores - maximum.unsqueeze(-1)).exp() / safe_denominator.unsqueeze(-1)
        p = torch.where(valid[:, None, None, :] & (denominator > 0).unsqueeze(-1), p, 0)
        gv = (g.bfloat16() @ v.bfloat16().transpose(-1, -2)).float()
        error = p * (gv - dot.unsqueeze(-1))
        dk = (error.bfloat16().transpose(-1, -2) @ q.bfloat16()).float() / math.sqrt(q.shape[-1])
        dv = (p.bfloat16().transpose(-1, -2) @ g.bfloat16()).float()
    return dk, dv, p


@pytest.mark.parametrize("dim", [16, 32, 64, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_reference_matches_materialized_product_boundaries(dim, strided):
    values = recomputed_tile_fixture(dim=dim, strided=strided)
    assert _validate_recomputed_tile_metadata(*values) == (2, 2, 3, 5, dim)
    q, k, v, g, dot, maximum, denominator, valid = values
    expected_key, expected_value, p = frozen_recomputed_reference(*values)
    spec = _Invocation(SimpleNamespace(head_dim=dim), 1.0, "mixed", False, torch.bfloat16)
    dvalue = _mm(p.transpose(-1, -2), g, spec, torch.bfloat16)
    error = p * (_mm(g, v.transpose(-1, -2), spec, torch.bfloat16) - dot.unsqueeze(-1))
    dkey = _mm(error.transpose(-1, -2), q, spec, torch.bfloat16) / math.sqrt(dim)
    torch.testing.assert_close(dkey, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(dvalue, expected_value, rtol=0, atol=0)


def test_invalid_keys_and_empty_rows_have_zero_adjoint_with_nonfinite_empty_maximum():
    values = list(recomputed_tile_fixture())
    values[5].fill_(-float("inf"))
    values[6].zero_()
    for result in frozen_recomputed_reference(*values):
        assert torch.isfinite(result).all()
        assert torch.count_nonzero(result) == 0
    values = list(recomputed_tile_fixture())
    dk, dv, p = frozen_recomputed_reference(*values)
    for result in (dk, dv):
        assert torch.count_nonzero(result[..., ::2, :]) == 0
    assert torch.count_nonzero(p[..., ::2]) == 0


def test_single_active_row_has_closed_form_with_complete_softmax_denominator():
    q = torch.full((1, 1, 1, 16), 2., dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.full_like(q, .5, dtype=torch.float32)
    dot = torch.full((1, 1, 1), .25)
    maximum = torch.zeros_like(dot)
    denominator = torch.full_like(dot, 2.)  # Other attention columns retain half the mass.
    valid = torch.ones((1, 1), dtype=torch.bool)
    dk, dv, p = frozen_recomputed_reference(q, k, v, g, dot, maximum, denominator, valid)
    torch.testing.assert_close(p, torch.full_like(p, .5), rtol=0, atol=0)
    torch.testing.assert_close(dk, torch.full_like(dk, -.0625), rtol=0, atol=0)
    torch.testing.assert_close(dv, torch.full_like(dv, .25), rtol=0, atol=0)


def test_rounding_query_chunks_before_accumulation_changes_required_product():
    # An adversarial exact BF16 reduction: 256 + 1 + 1 = 258. Rounding
    # each partial two-row product first discards the first +1 at a tie.
    a = torch.ones((1, 1, 1, 3), dtype=torch.bfloat16)
    b = torch.tensor([256., 1., 1.], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
    full = (a @ b).float()
    rounded_chunks = (a[..., :2] @ b[..., :2, :]).float() + (a[..., 2:] @ b[..., 2:, :]).float()
    assert full.item() == 258.
    assert rounded_chunks.item() == 257.


@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 2048), (2048, 1), (2048, 2048), (33, 17), (65, 129)])
def test_supported_tile_extents_are_metadata_valid(rows, columns):
    assert _validate_recomputed_tile_metadata(*recomputed_tile_fixture(rows, columns)) == (2, 2, rows, columns, 16)


def test_broadcast_inputs_are_supported_read_only_views():
    values = list(recomputed_tile_fixture())
    for index in range(7):
        values[index] = values[index][:1, :1].expand_as(values[index])
    values[7] = values[7][:1].expand_as(values[7])
    assert _validate_recomputed_tile_metadata(*values) == (2, 2, 3, 5, 16)


@pytest.mark.parametrize("index,dtype", [(0, torch.float32), (1, torch.float16),
    (2, torch.float32), (3, torch.bfloat16), (4, torch.float64),
    (5, torch.bfloat16), (6, torch.bfloat16), (7, torch.float32)])
def test_wrong_dtypes_rejected(index, dtype):
    values = list(recomputed_tile_fixture()); values[index] = values[index].to(dtype)
    with pytest.raises(ValueError):
        _validate_recomputed_tile_metadata(*values)


@pytest.mark.parametrize("index", range(8))
def test_wrong_ranks_rejected(index):
    values = list(recomputed_tile_fixture()); values[index] = values[index].unsqueeze(0)
    with pytest.raises(ValueError):
        _validate_recomputed_tile_metadata(*values)


@pytest.mark.parametrize("index", range(8))
def test_mismatched_batch_extents_rejected(index):
    values = list(recomputed_tile_fixture()); values[index] = values[index][:1]
    with pytest.raises(ValueError):
        _validate_recomputed_tile_metadata(*values)


@pytest.mark.parametrize("rows,columns,dim", [(0, 3, 16), (3, 0, 16),
    (2049, 3, 16), (3, 2049, 16), (3, 5, 8), (3, 5, 256)])
def test_unsupported_extents_rejected(rows, columns, dim):
    with pytest.raises(ValueError):
        _validate_recomputed_tile_metadata(*recomputed_tile_fixture(rows, columns, dim))


def test_empty_batch_or_head_rejected():
    for batch_empty in (True, False):
        values = list(recomputed_tile_fixture())
        for index in range(7):
            values[index] = values[index][:0] if batch_empty else values[index][:, :0]
        if batch_empty:
            values[7] = values[7][:0]
        with pytest.raises(ValueError, match="must be positive"):
            _validate_recomputed_tile_metadata(*values)


def test_cpu_rejected_before_importing_triton(monkeypatch):
    def unexpected():
        raise AssertionError("CPU input attempted Triton import or compile")
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_recompute_kernels._get_recomputed_backward_tile_kernel", unexpected)
    with pytest.raises(ValueError, match="requires CUDA"):
        backward_recomputed_tile(*recomputed_tile_fixture())


def test_module_import_does_not_import_triton(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise AssertionError("CPU module import unexpectedly loaded Triton")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    path = Path(__file__).parents[1] / "cdrm/pretrained/olmo_rt_recompute_kernels.py"
    spec = importlib.util.spec_from_file_location("isolated_recomputed_kernel_import", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize("index", range(7))
def test_autograd_is_not_silently_dropped(index):
    values = list(recomputed_tile_fixture()); values[index].requires_grad_(True)
    with pytest.raises(ValueError, match="no autograd"):
        _validate_recomputed_tile_metadata(*values)
    with torch.no_grad():
        assert _validate_recomputed_tile_metadata(*values) == (2, 2, 3, 5, 16)


def test_device_mismatch_non_tensor_and_non_strided_input_rejected():
    values = list(recomputed_tile_fixture()); values[4] = values[4].to("meta")
    with pytest.raises(ValueError, match="share one device"):
        _validate_recomputed_tile_metadata(*values)
    values = list(recomputed_tile_fixture()); values[0] = None
    with pytest.raises(TypeError, match="must all be tensors"):
        _validate_recomputed_tile_metadata(*values)
    values = list(recomputed_tile_fixture()); values[0] = values[0].to_sparse()
    with pytest.raises(ValueError, match="strided layout"):
        _validate_recomputed_tile_metadata(*values)
