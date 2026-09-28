"""CPU checks of the bounded GPU probe's independent oracle and scope guards."""
import pytest
import torch

from cdrm.pretrained.fbt_training import aggregate_pass_losses
from cdrm.pretrained.nextlat import NextLatLosses
from scripts.olmo_campaign_probe import explicit_k4_objective, gradient_record, parse_args


def test_literal_campaign_probe_objective_preserves_term_specific_pass_derivatives():
    counts = {"ce": 7, "latent": 5, "kl": 3}
    weights = {term: 1.0 for term in counts}
    leaves = [torch.tensor([2.0 + index, 4.0 + index, 6.0 + index], requires_grad=True)
              for index in range(4)]
    passes = tuple(NextLatLosses(dict(zip(counts, leaf.unbind())), counts, weights) for leaf in leaves)
    result = aggregate_pass_losses(passes, pass_loss_policy="campaign_v1")
    actual = explicit_k4_objective(result)
    torch.testing.assert_close(actual, result.total, atol=0, rtol=0)
    actual.backward()
    for index, leaf in enumerate(leaves):
        want = torch.tensor([(.5 if index == 0 else 1 / 6) / 7, .25 / 5, .25 / 3])
        torch.testing.assert_close(leaf.grad, want, atol=0, rtol=1e-7)


def test_gradient_probe_reports_missing_nonfinite_and_reference_mismatch():
    model = torch.nn.Linear(2, 2, bias=False)
    missing, _ = gradient_record(model)
    assert not missing["finite"] and missing["missing_active_gradients"] == ["weight"]
    model.weight.grad = torch.ones_like(model.weight)
    record, reference = gradient_record(model, save_cpu=True)
    assert record["finite"]
    same, _ = gradient_record(model, reference)
    assert same["comparison"]["all_parameters_close"]
    model.weight.grad.mul_(2)
    changed, _ = gradient_record(model, reference)
    assert not changed["comparison"]["all_parameters_close"]
    model.weight.grad[0, 0] = float("nan")
    bad, _ = gradient_record(model)
    assert not bad["finite"]


@pytest.mark.parametrize("extra", [["--length", "1024"], ["--updates", "100"], ["--tiny"]])
def test_probe_parser_cannot_expand_into_training_or_substitute_tiny_weights(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/campaign-probe", *extra])
