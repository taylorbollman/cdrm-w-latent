import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cdrm_precision_block_attribution import vector_components


def test_opposing_error_components_do_not_become_additive_percentages():
    reference = {'w': torch.tensor([2., 0.], dtype=torch.float64)}
    counter = {'w': torch.tensor([3., 0.], dtype=torch.float64)}
    actual = {'w': torch.tensor([2.25, 0.], dtype=torch.float64)}
    r = vector_components(reference, actual, counter)['global']
    assert r['local_precision_energy'] == 1.
    assert r['incoming_cotangent_energy'] == .75**2
    assert r['twice_cross_inner_product'] == -1.5
    assert r['total_error_energy'] == .25**2
    assert r['energy_identity_residual'] == 0.


def test_gradient_coverage_is_not_silently_dropped():
    r = {'a': torch.ones(2), 'b': torch.ones(2)}
    with pytest.raises(ValueError, match='coverage'):
        vector_components(r, {'a': torch.ones(2)}, r)
