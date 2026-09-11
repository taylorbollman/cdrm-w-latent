import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cdrm_precision_attention_attribution import decomposition


def test_storage_error_and_arithmetic_cancellation_are_separate():
    exact = torch.tensor([1.002, -0.009878, 2.315], dtype=torch.float64)
    local = exact.float()+torch.tensor([2e-5, -1e-7, 3e-6])
    row = decomposition(exact, local, local.bfloat16())
    assert row['native_equals_cast_local_fp32']
    assert row['components']['native_extra']['energy'] == 0.
    assert row['components']['storage_cast']['energy'] > 0.
    assert row['components']['fp32_arithmetic']['energy'] > 0.
    assert abs(row['energy_identity_residual']) < 1e-15


def test_native_deviation_is_retained_even_when_it_cancels_storage_error():
    exact = torch.tensor([1.004], dtype=torch.float64)
    local = exact.float()
    native = torch.tensor([1.0], dtype=torch.bfloat16)
    row = decomposition(exact, local, native)
    assert not row['native_equals_cast_local_fp32']
    assert row['native_cast_local_fp32_mismatches'] == 1
    assert row['components']['native_extra']['energy'] > 0
    assert row['twice_cross_inner_products']['native_extra__storage_cast'] < 0
