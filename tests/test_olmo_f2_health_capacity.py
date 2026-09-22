"""Attribution must preserve attached passes and distinct target denominators."""
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.nextlat import NextLatBatch
from scripts.olmo_f1_common import build_model
from scripts.olmo_f2_health_capacity import (case_for, component_objectives,
    gradient_attribution, gradient_summary)


@pytest.mark.parametrize("name", ["ordinary", "rt", "fbt", "nextlat", "combined", "combined-k3"])
def test_attribution_closes_on_tiny_fp32_attached_graph(name):
    torch.set_num_threads(1); torch.manual_seed(212)
    config = replace(OLMoConfig.tiny(), model_dim=64, mlp_intermediate_size=128)
    source = OLMoForCausalLM(config, attention_backend="math")
    case = case_for(name, batch=1, length=8)
    model = build_model(source.state_dict(), case, device="cpu", model_config=config,
                        chunk_size=3, backend="math")
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 60]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    ce = valid.clone(); ce[:, :4] = False
    kl = valid.clone(); kl[:, :6] = False
    batch = NextLatBatch(ids, valid, torch.zeros_like(ids), ce, valid, kl)
    result = model.loss_sums(batch, backbone_kwargs={"mode": case.mode()})
    terms = component_objectives(result)
    torch.testing.assert_close(sum(value for _, _, value in terms), result.total)
    if name == "combined-k3":
        assert result.pass_coefficients == (1, .5, .5)
        assert result.counts == {"ce": 4, "latent": 7, "kl": 2}
        # A last-pass loss must differentiate into fusion through attached states.
        last_ce = next(v for p, t, v in terms if p == 2 and t == "ce")
        gradient = torch.autograd.grad(last_ce, model.backbone.fusion.parameters(), retain_graph=True)
        assert sum(float(g.square().sum()) for g in gradient) > 0
    row = gradient_attribution(result, [(n, p) for n, p in model.named_parameters() if p.requires_grad])
    assert row["passed"]
    assert row["component_sum_relative_l2"] < 2e-6
    assert all(p.grad is None for p in model.parameters())


def test_gradient_summary_reports_scale_missing_groups_and_direction():
    names = ["backbone.backbone.a", "backbone.fusion.b", "predictor.c"]
    values = [torch.tensor([3., 4.]), None, torch.tensor([12.])]
    reference = [torch.tensor([-3., -4.]), torch.tensor([6.]), torch.tensor([12.])]
    report = gradient_summary(names, values, reference)
    assert report["all"]["norm"] == 13
    assert report["native"]["cosine_on_participating_tensors"] == -1
    assert report["fusion"]["participating_tensors"] == 0
    assert report["all"]["dot_with_total"] == 119


def test_invalid_case_rejected():
    with pytest.raises(ValueError):
        case_for("typo")
