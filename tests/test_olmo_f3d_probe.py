"""CPU checks for F3d fixture/reference and allocation-observer semantics."""
import json

import pytest
import torch

from cdrm.pretrained.olmo_tiled import _attention_from_completed
from scripts.olmo_f3d_probe import (
    AttentionShapeObserver, block_cases, bounded_block_fixture, fp64_row_oracle,
    json_safe, long_history_fixture, materialized_history_fixture, output_comparison,
    row_fixture, row_spec,
)


@pytest.mark.parametrize("length,prefix", [(1, 0), (1, 3), (9, 0), (17, 3), (33, 3), (65, 0), (129, 3)])
def test_cpu_materialized_reconstruction_and_independent_boundary_oracle(length, prefix):
    torch.set_num_threads(1)
    for variant in range(4):
        fixture = row_fixture(length, prefix, variant)
        probability, attention = _attention_from_completed(
            *fixture["operands"], prefix, row_spec(fixture), torch.bfloat16)
        oracle = fp64_row_oracle(fixture)
        assert output_comparison(attention, oracle[3])["passed"], (length, prefix, variant)
        assert output_comparison(probability, oracle[4])["passed"]
        if variant == 1:
            assert torch.count_nonzero(probability) == torch.count_nonzero(attention) == 0
            assert torch.isneginf(oracle[0]).all()
            assert torch.count_nonzero(oracle[1]) == 0


def test_shape_observer_detects_materialized_attention_and_flattened_or_transposed_forms():
    with AttentionShapeObserver(17, 20) as observer:
        full = torch.empty(2, 4, 17, 20)
        full.flatten(0, 1)
        full.transpose(-1, -2)
        full.clone()
    result = observer.record()
    assert not result["no_full_attention_shape"]
    assert len(result["full_attention_outputs"]) == 4
    assert result["largest_tensor_numel"] == 2*4*17*20
    assert result["largest_tensor_shape"] == [2, 4, 17, 20]


def test_shape_observer_accepts_bounded_rows_and_keeps_no_tensor_references():
    with AttentionShapeObserver(129, 132) as observer:
        q = torch.empty(2, 4, 129, 16)
        row = torch.empty(2, 4, 32, 132)
        state = torch.zeros(2, 4, 129)
    result = observer.record()
    assert result["no_full_attention_shape"]
    assert result["tensor_outputs"] == 3
    assert not any(isinstance(value, torch.Tensor) for value in observer.__dict__.values())
    assert json.loads(json.dumps(result))["largest_tensor_shape"] == [2, 4, 32, 132]


def test_shape_observer_positive_control_catches_old_backward_reconstruction():
    fixture = row_fixture(65, 3, 2)
    with AttentionShapeObserver(65, 68) as observer:
        _attention_from_completed(*fixture["operands"], 3, row_spec(fixture), torch.bfloat16)
    assert not observer.record()["no_full_attention_shape"]
    assert any("bmm" in row["operation"] for row in observer.full_attention_outputs)


def test_probe_cases_cover_alphas_prefixes_masks_and_lengths_beyond_chunk():
    cases = block_cases()
    assert len(cases) == 14
    assert {(length, prefix, alpha) for length in (9, 17) for prefix in (0, 3)
            for alpha in (0., .37, 1.)}.issubset(cases)
    assert (65, 3, 1.) in cases and (129, 3, 1.) in cases
    fixture = bounded_block_fixture(129, 3, 1., 42)
    assert fixture["config"]["max_context_length"] >= 132
    assert fixture["cotangents"][1].shape[-2] == 132
    assert not fixture["valid"].all()
    assert torch.equal(fixture["query_positions"], fixture["key_positions"][:, 3:])


def test_strided_rows_and_randomness_are_reproducible_without_global_rng_change():
    before = torch.random.get_rng_state().clone()
    first, second = row_fixture(33, 3, 2), row_fixture(33, 3, 2)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first["operands"], second["operands"]))
    assert all(value.stride()[-1] == 2 for value in first["operands"])
    assert not torch.cuda.is_initialized()


def test_output_screen_rejects_nonfinite_and_even_subnormal_nonzero_against_zero():
    zero = torch.zeros(3)
    assert output_comparison(zero, zero)["passed"]
    assert not output_comparison(torch.full((3,), 1e-40), zero)["passed"]
    bad = output_comparison(torch.full((3,), float("nan")), zero)
    assert not bad["passed"]
    assert json.loads(json.dumps(json_safe(bad), allow_nan=False))["relative_l2"] is None


@pytest.mark.parametrize("rows,columns", [(257, 255), (513, 511), (1024, 1024), (2048, 3)])
def test_long_history_fixtures_cover_reduction_loop_and_keep_masked_probability_zero(rows, columns):
    torch.set_num_threads(1)
    fixture = long_history_fixture(rows, columns)
    operands, probability, spec = materialized_history_fixture(fixture)
    assert probability.shape == (1, 2, rows, columns)
    assert all(torch.isfinite(x).all() for x in operands)
    assert torch.count_nonzero(probability[..., ::7]) == 0
    assert (probability.sum(-1) <= 1+1e-6).all()
    assert (operands[-2] > 0).all()
    assert spec.config.head_dim == 128
