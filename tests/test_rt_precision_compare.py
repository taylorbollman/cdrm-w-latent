"""CPU checks for reference normalization, review arithmetic and state identity."""
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_batch_profile import head_chunked_backward
from rt_precision_compare import (
    GRADIENT_LIMITS, OPTIMIZER_OPTIONS, adam_deltas, compare_arm_packets,
    compare_tensors, load_ids, near_zero_update_analysis,
    reference_microbatch_backward, validate_starting_state,
)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('CPU tests must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)


def test_review_uses_no_absolute_floor_even_for_tiny_gradients():
    reference = {'tiny': torch.tensor([1e-12, -1e-12])}
    result = compare_tensors(reference, {'tiny': reference['tiny'] * 1.1}, limits=GRADIENT_LIMITS)
    assert result['requires_review']
    assert result['relative_l2'] == pytest.approx(.1, rel=1e-5)
    assert result['tensors']['tiny']['reference_rms'] == pytest.approx(1e-12)
    assert result['absolute_acceptance_floor'] == 0
    assert result['tensors']['tiny']['reference_rms_below_2e-6']


def test_zero_reference_coordinates_and_tensors_are_not_dropped():
    left = {'zero': torch.zeros(3), 'live': torch.ones(3)}
    right = {'zero': torch.tensor([1e-12, 0., 0.]), 'live': torch.ones(3)}
    result = compare_tensors(left, right, limits=GRADIENT_LIMITS)
    assert result['numel'] == 6
    assert result['tensors']['zero']['relative_l2'] is None
    assert result['tensors']['zero']['reference_zero_coordinates'] == 3
    assert result['tensor_review_flags']['zero']
    assert not result['global_review_flags']
    assert compare_tensors({'x': torch.zeros(2)}, {'x': torch.zeros(2)}, limits=GRADIENT_LIMITS)['relative_l2'] == 0


def test_per_tensor_maximum_detects_local_error_hidden_by_global_norm():
    left = {'large': torch.ones(1000), 'small': torch.tensor([1., 1.])}
    right = copy.deepcopy(left)
    right['small'][1] = 1.1
    result = compare_tensors(left, right, limits=GRADIENT_LIMITS)
    assert not result['global_review_flags']
    assert 'tensor_max_relative_to_reference_max' in result['tensor_review_flags']['small']
    assert result['tensors']['small']['top_error_coordinates'][0]['flat_index'] == 1


@pytest.mark.parametrize('actual', [{}, {'x': torch.ones(3)}, {'x': torch.ones(2, dtype=torch.float64)}])
def test_missing_or_mismatched_gradient_contract_is_hard_error(actual):
    with pytest.raises(ValueError):
        compare_tensors({'x': torch.ones(2)}, actual, limits=GRADIENT_LIMITS)


def test_nonfinite_gradient_is_hard_error_not_review_flag():
    with pytest.raises(FloatingPointError):
        compare_tensors({'x': torch.ones(2)}, {'x': torch.tensor([1., float('nan')])})


def test_adam_delta_subtracts_promoted_saved_weights():
    before, after = {'x': torch.tensor([1e-8])}, {'x': torch.tensor([1.])}
    result = adam_deltas(before, after)['x']
    assert result.dtype == torch.float64
    assert torch.equal(result, after['x'].double() - before['x'].double())
    assert not torch.equal(result, (after['x'] - before['x']).double())


def test_near_zero_bucket_retains_sign_flips_outside_bucket():
    g, h = {'w': torch.tensor([1e-8, 1.])}, {'w': torch.tensor([-1e-8, -1.])}
    d, e = {'w': torch.tensor([-1., -1.], dtype=torch.float64)}, {'w': torch.tensor([1., 0.], dtype=torch.float64)}
    result = near_zero_update_analysis(g, h, d, e)
    assert result['inside']['coordinates'] == result['outside']['coordinates'] == 1
    assert result['inside']['sign_flips'] == result['outside']['sign_flips'] == 1
    assert result['inside_error_energy_fraction'] == pytest.approx(.8)


class CausalTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scale_logits=True, d_model=7)
        self.transformer = torch.nn.ModuleDict({'wte': torch.nn.Embedding(13, 7),
                                                'layer': torch.nn.Linear(7, 7),
                                                'ff_out': torch.nn.Linear(7, 13, bias=False)})
        self.forward_batches = []

    def forward(self, ids, *, return_pre_logits, return_logits):
        self.forward_batches.append(len(ids))
        value = torch.tanh(self.transformer.layer(self.transformer.wte(ids))).cumsum(dim=1)
        return SimpleNamespace(pre_logits=value)


@pytest.mark.parametrize('microbatch', [1, 2, 3, 5])
def test_reference_microbatch_preserves_global_normalization_and_parameter_credit(microbatch):
    torch.manual_seed(724)
    reference = CausalTinyModel()
    candidate = copy.deepcopy(reference)
    ids = torch.randint(0, 13, (5, 7))
    expected = head_chunked_backward(reference, ids, bf16=False)
    actual = reference_microbatch_backward(candidate, ids, microbatch=microbatch)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert candidate.forward_batches == [min(microbatch, 5 - start) for start in range(0, 5, microbatch)]
    for name, parameter in reference.named_parameters():
        other = dict(candidate.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(other.grad, parameter.grad, rtol=3e-6, atol=3e-8, msg=name)
    # Clip and Adam happen once after the complete global gradient in both arms.
    for model in (reference, candidate):
        torch.nn.utils.clip_grad_norm_(model.parameters(), .2, foreach=False)
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
        optimizer.step()
        assert {float(state['step']) for state in optimizer.state.values()} == {1.}
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(dict(candidate.named_parameters())[name], parameter, rtol=1e-6, atol=1e-7)


def test_reference_invalid_partition_does_not_execute_model():
    model = CausalTinyModel()
    with pytest.raises(ValueError):
        reference_microbatch_backward(model, torch.ones((2, 3), dtype=torch.long), microbatch=3)
    assert not model.forward_batches


def test_ids_prefix_is_explicit_and_integer_conversion_preserves_values(tmp_path):
    path = tmp_path / 'tokens.npy'
    values = np.arange(24, dtype=np.uint16).reshape(4, 6)
    np.save(path, values)
    actual = load_ids(path, batch=2, length=6, vocab=24, seed=5)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, values[:2])
    with pytest.raises(ValueError, match='at least batch'):
        load_ids(path, batch=5, length=6, vocab=24, seed=5)
    with pytest.raises(ValueError, match='valid vocabulary'):
        load_ids(path, batch=4, length=6, vocab=23, seed=5)


@dataclass
class StateConfig:
    d_model: int = 2
    init_device: str = 'cpu'
    recurrent_precision_policy: str = 'bf16_fp32_state'
    reference_eager: bool = False
    precision: object = None


def state_fixture():
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
    return {'schema': 'rt-precision-state-v1', 'model_config': asdict(StateConfig()), 'model': copy.deepcopy(model.state_dict()),
            'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': ['weight']}


def test_starting_state_accepts_precision_override_but_rejects_semantic_or_owner_changes():
    packet = state_fixture()
    packet['model_config']['recurrent_precision_policy'] = 'legacy'
    assert not validate_starting_state(packet, StateConfig(), ['weight'])['trained_moments']
    altered = copy.deepcopy(packet)
    altered['model_config']['d_model'] = 3
    with pytest.raises(ValueError, match='architecture'):
        validate_starting_state(altered, StateConfig(), ['weight'])
    with pytest.raises(ValueError, match='parameter order'):
        validate_starting_state(packet, StateConfig(), ['renamed'])


def test_partial_or_negative_adam_state_is_rejected():
    packet = state_fixture()
    packet['optimizer']['state'][0] = {'step': torch.tensor(1.), 'exp_avg': torch.zeros(2, 2),
                                       'exp_avg_sq': torch.full((2, 2), -1.)}
    with pytest.raises(ValueError, match='nonnegative'):
        validate_starting_state(packet, StateConfig(), ['weight'])
    packet['optimizer']['state'][0]['exp_avg_sq'].zero_()
    assert validate_starting_state(packet, StateConfig(), ['weight'])['trained_moments']
    packet['optimizer']['state'][0]['step'].zero_()
    packet['optimizer']['state'][0]['exp_avg'].fill_(1.)
    with pytest.raises(ValueError, match='zero moments'):
        validate_starting_state(packet, StateConfig(), ['weight'])


def test_review_flag_does_not_prevent_complete_comparison():
    starting = state_fixture()
    reference = copy.deepcopy(starting)
    reference.update(arm='C', loss=torch.tensor(1.), raw_gradients={'weight': torch.ones(2, 2)},
                     clipped_gradients={'weight': torch.ones(2, 2)}, clip_coefficient=torch.tensor(1.),
                     layer_output_samples={})
    reference['model']['weight'] -= .01
    reference['optimizer']['state'][0] = {'step': torch.tensor(1.), 'exp_avg': torch.ones(2, 2),
                                          'exp_avg_sq': torch.ones(2, 2)}
    actual = copy.deepcopy(reference)
    actual['arm'] = 'B'
    actual['raw_gradients']['weight'] *= 1.2
    result = compare_arm_packets(starting, reference, actual, trained=False)
    assert result['requires_review']
    assert result['raw_gradients']['relative_l2'] == pytest.approx(.2, rel=1e-5)
    assert result['adam_delta']['cosine'] == pytest.approx(1.)
    assert result['adam_delta']['requires_review'] is False
    reference['arm'] = 'A'
    pairwise = compare_arm_packets(starting, reference, actual, trained=False)
    assert pairwise['reference'] == 'A' and pairwise['candidate'] == 'B'
    assert 'not an FP32 oracle' in pairwise['comparison_role']
    assert pairwise['near_zero']['inside']['coordinates'] == result['near_zero']['inside']['coordinates']
