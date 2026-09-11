"""Independent optimizer math, expected rounding, and corruption detection."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import cdrm_precision_adam_reference as reference


def native_packet(gradient, *, step=19., weight=None):
    assert not torch.cuda.is_available(), "Run in the GPU-disabled CPU container"
    weight = torch.tensor([1., -2., .01, 32., -1e-4], dtype=torch.float32) if weight is None else weight.clone()
    p = torch.nn.Parameter(weight.clone())
    optimizer = torch.optim.AdamW([p], lr=.0004992308747664155, betas=(.9, .98),
                                  eps=1e-8, weight_decay=0., foreach=False, fused=False)
    m = torch.tensor([.01, -.02, 0., 1e-8, -.001], dtype=torch.float32)
    v = torch.tensor([.0001, .001, 1e-9, 1e-12, .0003], dtype=torch.float32)
    prior = {"w": {"step": torch.tensor(step), "exp_avg": m, "exp_avg_sq": v}}
    optimizer.state[p] = copy.deepcopy(prior["w"])
    p.grad = gradient.clone()
    norm = torch.nn.utils.clip_grad_norm_([p], 1., error_if_nonfinite=True)
    coefficient = torch.clamp(1./(norm+1e-6), max=1.).item()
    clipped = p.grad.clone()
    optimizer.step()
    native = {"clip_norm": norm.item(), "clip_coefficient": coefficient,
              "clipped_gradients": {"w": clipped}, "weights": {"w": p.detach().clone()},
              "deltas": {"w": p.detach().double()-weight.double()},
              "state": {"w": copy.deepcopy(optimizer.state[p])}}
    return {"w": weight}, {"w": gradient}, prior, native, optimizer.param_groups[0]


@pytest.mark.parametrize("gradient", [torch.tensor([4., -3., 0., 1e-9, .1]), torch.tensor([.1, -.09, 0., 1e-20, .01])])
def test_clipped_and_unclipped_native_fp32_satisfy_independent_reference(gradient):
    args = native_packet(gradient)
    result, _ = reference.local_reference(*args)
    assert result["pass"]
    assert result["rows"]["w"]["delta_packet_exact"]
    assert result["native_coefficient_exact"]
    assert result["fp64_coefficient"] < 1 if gradient.norm() > 1 else result["fp64_coefficient"] == 1


def test_closed_form_uses_nonempty_moments_bias_correction_and_actual_step():
    w = {"w": torch.tensor([2.], dtype=torch.float32)}
    g = {"w": torch.tensor([.4], dtype=torch.float64)}
    state = {"w": {"step": torch.tensor(2.), "exp_avg": torch.tensor([.125]), "exp_avg_sq": torch.tensor([.25])}}
    hp = {"lr": .007, "betas": (.8, .9), "eps": 1e-4, "weight_decay": 0.}
    out = reference.adam64(w, g, state, hp)
    m, v = .8*.125+.2*.4, .9*.25+.1*.4**2
    expected = -.007*(m/(1-.8**3))/((v/(1-.9**3))**.5+1e-4)
    assert out["increments"]["w"].item() == pytest.approx(expected, rel=1e-14)
    assert out["moments"]["w"]["step"] == 3
    with pytest.raises(ValueError, match="weight decay"):
        reference.adam64(w, g, state, {**hp, "weight_decay": .01})


def test_fp32_weight_write_can_erase_an_ideal_increment_without_a_bug():
    args = native_packet(torch.zeros(5), weight=torch.full((5,), 2**20, dtype=torch.float32))
    result, oracle = reference.local_reference(*args)
    assert result["pass"]
    assert bool((oracle["increments"]["w"] != 0).any())
    assert torch.equal(oracle["realized_deltas"]["w"], torch.zeros(5, dtype=torch.float64))
    assert result["native_delta_vs_ideal_increment"]["global"]["relative_l2"] == pytest.approx(1.)
    assert result["native_delta_vs_fp32_write_realized"]["global"]["error_energy"] == 0


def test_moment_cancellation_and_corrupted_native_packets_are_distinguished():
    args = native_packet(torch.tensor([-.09, .18, 0., -9e-8, .009]))
    assert reference.local_reference(*args)[0]["pass"]
    corrupt = copy.deepcopy(args)
    corrupt[3]["state"]["w"]["exp_avg"][0] += .01
    assert not reference.local_reference(*corrupt)[0]["rows"]["w"]["moment_first"]["pass"]
    corrupt = copy.deepcopy(args)
    corrupt[3]["deltas"]["w"][0] += 1e-6
    assert not reference.local_reference(*corrupt)[0]["rows"]["w"]["delta_packet_exact"]
    corrupt = copy.deepcopy(args)
    corrupt[3]["clipped_gradients"]["w"][0] += .001
    assert not reference.local_reference(*corrupt)[0]["rows"]["w"]["clipped_gradient_single_multiply_exact"]


def test_gradient_decomposition_retains_cross_term_and_fixed_near_zero_mask():
    r = {"w": torch.tensor([3., 4., 0.])}
    a = {"w": torch.tensor([4., 3., 1.])}
    value = reference.norm_direction(r, a)
    assert value["energy_identity_residual"] == pytest.approx(0., abs=1e-13)
    assert value["twice_cross_inner_product"] != 0
    stats = reference.compare_maps(r, a, {"w": torch.tensor([False, False, True])})
    assert stats["global"]["near_zero"]["error_energy_fraction"] == pytest.approx(1/3)
    assert stats["global"]["outside_near_zero"]["count"] == 2
