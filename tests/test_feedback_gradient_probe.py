"""Small CPU checks of existing-objective decomposition and gradient geometry."""
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.fbt_training import aggregate_pass_losses
from scripts.olmo_feedback_gradient_probe import (
    CONTRIBUTIONS, GRADIENTS, feedback_gradient_probe, gradient_accounting, weighted_feedback_terms,
)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatLosses
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(173)


def test_weighted_terms_use_independent_denominators_and_existing_k4_coefficients():
    counts = {"ce": 11, "latent": 7, "kl": 5}
    weights = {"ce": 1., "latent": .3, "kl": .7}
    leaves = torch.arange(1., 13., dtype=torch.float64).reshape(4, 3).requires_grad_()
    losses = aggregate_pass_losses([
        NextLatLosses(dict(zip(counts, row.square())), counts, weights) for row in leaves
    ], pass_loss_policy="campaign_v1")
    coefficients = {term: torch.tensor(weights[term]/counts[term], dtype=torch.float64) for term in counts}
    pieces = weighted_feedback_terms(losses, coefficients)
    literal = {
        "ce_first": .5*leaves[0, 0].square()/11,
        "ce_later": leaves[1:, 0].square().sum()/66,
        "latent": .3*leaves[:, 1].square().mean()/7,
        "kl": .7*leaves[:, 2].square().mean()/5,
    }
    for name in CONTRIBUTIONS:
        torch.testing.assert_close(pieces[name], literal[name], rtol=1e-14, atol=1e-14)
        actual = torch.autograd.grad(pieces[name], leaves, retain_graph=True)[0]
        expected = torch.autograd.grad(literal[name], leaves, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(sum(pieces.values()), losses.total)


def gradients():
    # CE and auxiliaries intentionally oppose on the backbone. Structural zeros
    # are represented by both absent tensors and measured zero tensors.
    names = ("backbone.backbone.weight", "backbone.fusion.weight", "predictor.weight")
    vectors = {
        "ce_first": ([1., 0., 2.], None, None),
        "ce_later": ([0., 1., 1.], [1., 2., 0.], [0., 0., 0.]),
        "latent": ([-2., -1., -3.], [1., 0., 1.], [1., 2., 3.]),
        "kl": ([0., 0., 0.], [-1., 0., 0.], [0., 1., 1.]),
    }
    result = {term: {name: None if vector is None else torch.tensor(vector)
                     for name, vector in zip(names, values)} for term, values in vectors.items()}
    result["joint"] = {name: sum(result[t][name] if result[t][name] is not None else torch.zeros(3)
                                 for t in CONTRIBUTIONS) for name in names}
    return result, dict.fromkeys(names, 3)


def test_geometry_handles_zero_cosines_conflict_and_chunked_joint_reconstruction():
    values, sizes = gradients()
    report = gradient_accounting(values, sizes, chunk_elements=2)
    assert report["passed"] and all(report["checks"].values())
    groups = report["components"]
    assert groups["predictor"]["ce_vs_aux"]["cosine"] is None
    assert groups["fusion"]["cosines"]["ce_first"]["joint"] is None
    assert groups["backbone"]["ce_vs_aux"]["dot"] == -12
    assert groups["backbone"]["ce_vs_aux"]["opposes"] is True
    assert groups["all"]["reconstruction"]["difference_norm"] == 0
    assert groups["all"]["parameter_elements"] == 9
    # Independent literal flattened vectors include structural zeros exactly.
    flat = {t: torch.cat([v if v is not None else torch.zeros(3) for v in values[t].values()]).double()
            for t in GRADIENTS}
    for a in GRADIENTS:
        for b in GRADIENTS:
            assert groups["all"]["dot_products"][a][b] == float(flat[a].dot(flat[b]))


def test_bad_joint_reconstruction_and_nonfinite_gradients_do_not_pass():
    values, sizes = gradients()
    values["joint"]["backbone.backbone.weight"][0] += .01
    report = gradient_accounting(values, sizes, chunk_elements=2)
    assert not report["passed"] and not report["checks"]["global_reconstruction_within_limit"]
    values["kl"]["predictor.weight"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        gradient_accounting(values, sizes)


def make_adapter(arm="NF", **mode_changes):
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0, 1), feedback_jitter=0,
                            document_policy="continuous-stream-v1")
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True)
    model = build_campaign_model(base, recipe).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 8, 9, 2, 0, 0]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.tensor([[1, 1, 1, 2, 2, 2], [3, 3, 3, 3, -1, -1]])
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 1] = False
    latent[1, 1] = False
    batch = NextLatBatch(ids, valid, docs, ce, latent, kl)
    mode = replace(recipe.mode(), **mode_changes)
    return CampaignObjective(model, batch, mode=mode, global_counts=model.counts(batch))


@pytest.mark.parametrize("arm", ["NF", "NFR"])
def test_actual_campaign_decomposition_preserves_state_and_reconstructs_joint(arm):
    adapter = make_adapter(arm)
    model = adapter.model
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    published = []
    result = feedback_gradient_probe(adapter, publish=published.append, chunk_elements=113)
    assert result["passed"] and all(result["checks"].values())
    assert result["forward_backward_pairs"] == 5 and result["optimizer_updates"] == 0
    assert [p["contribution"] for p in published] == list(GRADIENTS)
    assert result["counts"] == adapter.counts
    assert result["components"]["all"]["parameter_elements"] == sum(p.numel() for p in model.parameters())
    assert result["components"]["predictor"]["norms"]["joint"] > 0
    assert result["components"]["fusion"]["norms"]["ce_later"] > 0
    assert abs(result["objective_decomposition_difference"]) < 2e-6
    assert torch.equal(rng, torch.get_rng_state())
    for n, p in model.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, initial[n], rtol=0, atol=0)


def test_probe_rejects_control_beta_and_preexisting_gradients_without_mutating_them():
    with pytest.raises(ValueError, match="beta=1"):
        feedback_gradient_probe(make_adapter(beta=0.))
    adapter = make_adapter()
    p = next(adapter.model.parameters())
    p.grad = torch.ones_like(p)
    original = p.grad
    with pytest.raises(ValueError, match="grad=None"):
        feedback_gradient_probe(adapter)
    assert p.grad is original and bool(torch.all(original == 1))
