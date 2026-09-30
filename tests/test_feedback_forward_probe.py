"""Small canonical NF/NFR forward observations, beta controls and exact masks."""
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from scripts.olmo_feedback_forward_probe import feedback_forward_probe
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_evaluation import evaluation_runtime


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(384)


def setup(arm="NF"):
    recipe = CampaignRecipe(arm, sequence_length=8, rt_layers=(0, 1), document_policy="continuous-stream-v1")
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="sdpa", attention_precision="mixed")
    model = build_campaign_model(base, recipe).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]])
    valid = torch.tensor([[True]*7+[False]])
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 1, -1]])
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False
    latent[0, 4] = False
    kl[0, 5] = False
    return model, recipe, NextLatBatch(ids, valid, docs, ce, latent, kl)


@pytest.mark.parametrize("arm", ["NF", "NFR"])
def test_actual_forward_beta_controls_first_pass_invariance_and_state_preservation(arm):
    model, recipe, batch = setup(arm)
    initial = {n: value.detach().clone() for n, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with evaluation_runtime(model) as preservation:
        records = [feedback_forward_probe(model, recipe, batch, beta) for beta in (0., .5, 1.)]
    assert preservation["integrity_passed"] and preservation["restored"]
    assert records[0]["beta_zero_exact_pass_and_input_control"] is True
    assert len({r["passes"][0]["hidden_sha256"] for r in records}) == 1
    assert [r["fusion_forward_calls"] for r in records] == [0, 3, 3]
    for r in records:
        assert r["native_stack_forward_calls"] == 4 and r["model_forward_calls"] == 1
        assert r["counts"] == {"ce": 5, "latent": 4, "kl": 2}
        for p in r["passes"]:
            assert all(math.isfinite(v) for v in p["means"].values())
            assert p["hidden_geometry"]["positions"] == 7
            assert p["teacher_entropy_on_kl_positions"]["count"] == 2
            assert p["student_entropy_on_kl_positions"]["count"] == 2
            for key in ("teacher_entropy_on_kl_positions", "student_entropy_on_kl_positions"):
                entropy = p[key]
                assert entropy["bounds_checked_per_position"] is True
                assert -1e-5 <= entropy["mean_nats"] <= math.log(entropy["vocabulary_rows"])+1e-5
        for f in r["feedback"]:
            # Continuous feedback crosses the document boundary; KL does not.
            assert f["eligible_positions"] == 6
            assert (f["raw_fusion_vs_token_input"] is None) == (r["mode"]["beta"] == 0.)
            if r["mode"]["beta"] == 0.:
                assert f["actual_blended_vs_token_input"]["difference_rms"] == 0.
            if r["mode"]["beta"] == 1.:
                assert f["actual_blended_vs_token_input"] == f["raw_fusion_vs_token_input"]
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_entropy_uses_literal_kl_coordinates_and_full_vocabulary():
    model, recipe, batch = setup()
    captured = []
    handle = model.backbone.register_forward_hook(lambda _, args, kwargs, output:
        captured.append((kwargs["inputs_embeds"], output.pass_hidden_states)), with_kwargs=True)
    with evaluation_runtime(model):
        record = feedback_forward_probe(model, recipe, batch, .5)
        handle.remove()
        embeddings, states = captured[0]
        # Eligible triples end at2 and6: the other possible triple ends at5,
        # which is masked. Triples crossing docs/padding are ineligible.
        for p, hidden in zip(record["passes"], states):
            teacher_values, student_values = [], []
            for target in (2, 6):
                teacher = hidden[0, target-1:target]
                predicted = model.predictor(hidden[0, target-2:target-1], embeddings[0, target-1:target])
                for tensor, destination in ((teacher, teacher_values), (predicted, student_values)):
                    logp = F.log_softmax(F.linear(tensor, model.backbone.readout_weight), -1)
                    destination.append(float(-(logp.exp()*logp).sum()))
            # Literal B1 projections and the batched observer have different
            # FP32 GEMM grouping; compare at their rounding scale.
            assert p["teacher_entropy_on_kl_positions"]["sum_nats"] == pytest.approx(sum(teacher_values), rel=5e-7, abs=2e-6)
            assert p["student_entropy_on_kl_positions"]["sum_nats"] == pytest.approx(sum(student_values), rel=5e-7, abs=2e-6)
            assert p["teacher_entropy_on_kl_positions"]["vocabulary_rows"] == model.backbone.readout_weight.shape[0]


def test_zero_kl_targets_have_null_entropy_and_hooks_removed_after_error(monkeypatch):
    model, recipe, batch = setup()
    batch = replace(batch, kl_mask=torch.zeros_like(batch.kl_mask))
    modules = (model.backbone, model.backbone.fusion, model.backbone.backbone)
    hook_counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    with evaluation_runtime(model):
        result = feedback_forward_probe(model, recipe, batch, 1.)
        for p in result["passes"]:
            assert p["means"]["kl"] is None
            assert p["teacher_entropy_on_kl_positions"]["mean_nats"] is None
            assert p["student_entropy_on_kl_positions"]["count"] == 0
        def fail(*args, **kwargs):
            raise RuntimeError("sentinel")
        monkeypatch.setattr(model, "loss_sums", fail)
        with pytest.raises(RuntimeError, match="sentinel"):
            feedback_forward_probe(model, recipe, batch, .5)
    assert [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules] == hook_counts


def test_broken_beta_zero_pass_control_is_rejected():
    model, recipe, batch = setup()
    def change_one_pass(module, args, output):
        changed = list(output.pass_hidden_states)
        changed[1] = changed[1]+.01
        return replace(output, pass_hidden_states=tuple(changed))
    handle = model.backbone.register_forward_hook(change_one_pass)
    try:
        with evaluation_runtime(model):
            with pytest.raises(ValueError, match="Beta-zero control"):
                feedback_forward_probe(model, recipe, batch, 0.)
    finally:
        handle.remove()
