"""Continuous-stream boundaries against an independent single-stream oracle.

These CPU checks qualify policy/math wiring, not Flash/NCCL or BF16 arithmetic.
"""
import copy
from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.campaign_losses import DynamicNextLatLayout, compute_dynamic_nextlat_loss_sums
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe, build_campaign_model, feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.document_policy import CONTINUOUS_STREAM, ISOLATED_DOCUMENTS
from cdrm.pretrained.distributed_training import validate_microbatch_inputs
from cdrm.pretrained.nextlat import (NextLatBatch, NextLatConfig, NextLatPredictor,
    build_nextlat_masks, compute_nextlat_loss_sums)
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.static_nextlat import PreparedNextLatLayout, compute_static_nextlat_loss_sums
from cdrm.pretrained.static_training import StaticFBTTraining


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(927)


def packed_batch():
    # EOS60 at position3 is literal content, not a boundary. EOS at1 and5 are
    # actual document ends. Last row is an all-empty distributed filler.
    ids = torch.tensor([[2, 60, 3, 60, 4, 60, 5], [60, 6, 7, 8, 60, 1, 1], [1]*7])
    valid = torch.tensor([[1]*7, [1]*5+[0]*2, [0]*7], dtype=torch.bool)
    docs = torch.tensor([[10, 10, 11, 11, 11, 11, 12], [20]*5+[-1]*2, [-1]*7])
    return NextLatBatch(ids, valid, docs)


def oracle_masks(batch):
    masks = {"ce": torch.zeros(batch.input_ids.shape[0], batch.input_ids.shape[1]-1, dtype=torch.bool),
             "latent": torch.zeros(batch.input_ids.shape[0], batch.input_ids.shape[1]-1, dtype=torch.bool),
             "kl": torch.zeros(batch.input_ids.shape[0], batch.input_ids.shape[1]-2, dtype=torch.bool)}
    for row in range(batch.input_ids.shape[0]):
        for start in range(batch.input_ids.shape[1]-1):
            pair = bool(batch.valid_mask[row, start:start+2].all())
            same = pair and batch.document_ids[row, start] == batch.document_ids[row, start+1]
            masks["ce"][row, start] = pair and (batch.ce_mask is None or bool(batch.ce_mask[row, start+1]))
            masks["latent"][row, start] = same and (batch.latent_mask is None or bool(batch.latent_mask[row, start+1]))
            if start+2 < batch.input_ids.shape[1]:
                triple = same and bool(batch.valid_mask[row, start+2]) and batch.document_ids[row, start+1] == batch.document_ids[row, start+2]
                masks["kl"][row, start] = triple and (batch.kl_mask is None or bool(batch.kl_mask[row, start+2]))
    return masks


def oracle_batch(batch):
    # A single-document reference yields identical continuous attention/RT/FBT
    # without enabling the new policy. Explicit independent auxiliary masks
    # restore the real document-boundary loss exclusions.
    masks = oracle_masks(batch)
    target = lambda term, prefix: F.pad(masks[term], (prefix, 0))
    return replace(batch, document_ids=torch.zeros_like(batch.document_ids).masked_fill(~batch.valid_mask, -1),
                   ce_mask=target("ce", 1), latent_mask=target("latent", 1), kl_mask=target("kl", 2))


def setup(arm="NFR", *, stream=True, jitter=.02):
    recipe = CampaignRecipe(arm, sequence_length=7, rt_layers=(0, 1), feedback_jitter=jitter,
                            document_policy=CONTINUOUS_STREAM if stream else ISOLATED_DOCUMENTS)
    model = build_campaign_model(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True), recipe)
    return recipe, model


def noise_for(recipe, model):
    return feedback_noise_for_rows(recipe, ["row0", "row1"], logical_update=0,
                                  sequence_length=7, width=model.config.model_dim, physical_batch_size=3)


def assert_grads(model, reference):
    for (name, value), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert (value.grad is None) == (other.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=3e-5, rtol=5e-4, msg=name)


def test_masks_separate_stream_ce_from_auxiliary_boundaries_and_literal_eos():
    batch = packed_batch()
    actual = build_nextlat_masks(batch, document_policy=CONTINUOUS_STREAM)
    expected = oracle_masks(batch)
    assert {k: int(v.sum()) for k, v in actual.items()} == {"ce": 10, "latent": 8, "kl": 5}
    for term in actual:
        torch.testing.assert_close(actual[term], expected[term], rtol=0, atol=0)
    assert actual["ce"][0, 1] and not actual["latent"][0, 1]  # EOS->new document
    assert actual["latent"][0, 3] and actual["kl"][0, 2]  # Literal EOS remains within doc11
    assert actual["ce"].shape[1] == 6  # Final logit never predicts into another row.
    masked = replace(batch, ce_mask=batch.valid_mask.clone(), latent_mask=batch.valid_mask.clone(),
                     kl_mask=batch.valid_mask.clone())
    masked.ce_mask[:, 2] = False
    masked.latent_mask[:, 3] = False
    masked.kl_mask[:, 4] = False
    for term, value in build_nextlat_masks(masked, document_policy=CONTINUOUS_STREAM).items():
        torch.testing.assert_close(value, oracle_masks(masked)[term], rtol=0, atol=0)
    # Old raw-loss behavior remains same-document CE when no policy is supplied.
    assert not build_nextlat_masks(batch)["ce"][0, 1]


def test_eos_at_chunk_start_is_a_ce_source_without_an_external_previous_target():
    batch = NextLatBatch(torch.tensor([[60, 2, 3, 60]]), torch.ones(1, 4, dtype=torch.bool),
                         torch.tensor([[10, 11, 11, 11]]))
    masks = build_nextlat_masks(batch, document_policy=CONTINUOUS_STREAM)
    assert masks["ce"].tolist() == [[True, True, True]]
    assert masks["latent"].tolist() == [[False, True, True]]
    assert masks["kl"].tolist() == [[False, True]]


@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_eager_and_prepared_raw_gradients_match_independent_stream_oracle(arm):
    recipe, model = setup(arm)
    _, reference = setup(arm, stream=False)
    reference.load_state_dict(model.state_dict())
    batch, noise = packed_batch(), noise_for(recipe, model)
    expected = reference.loss_sums(oracle_batch(batch), backbone_kwargs={"mode": replace(recipe.mode(),
        document_policy=ISOLATED_DOCUMENTS), "feedback_noise": noise, "right_padded_causal": True})
    expected.total.backward()
    got = model.loss_sums(batch, backbone_kwargs={"mode": recipe.mode(),
        "feedback_noise": noise, "right_padded_causal": True})
    assert got.counts == expected.counts == model.counts(batch)
    for term in got.sums:
        torch.testing.assert_close(got.sums[term], expected.sums[term], rtol=0, atol=0)
    got.total.backward()
    assert_grads(model, reference)
    model.zero_grad(set_to_none=True)
    adapter = CampaignObjective(model, batch, mode=recipe.mode(), global_counts=expected.counts,
                                feedback_noise=noise)
    actual = CampaignGraphTraining(adapter).backward([batch], feedback_noises=[noise])
    for term in got.sums:
        assert actual["loss_sums"][term] == pytest.approx(float(expected.sums[term].detach()), rel=3e-6, abs=3e-6)
    assert_grads(model, reference)
    torch.testing.assert_close(adapter.batch.document_ids, batch.document_ids, rtol=0, atol=0)
    assert adapter.forward_layout.feedback_eligible[0, 1]  # Cross-EOS feedback stays eligible.
    assert not adapter.forward_layout.feedback_eligible[-1].any()


@pytest.mark.parametrize("arm", ["B", "N", "R", "NFR"])
def test_legacy_wrappers_and_layouts_still_reject_packed_rows(arm):
    recipe, model = setup(arm, stream=False)
    batch = packed_batch()
    with pytest.raises(ValueError, match="Packed"):
        model.counts(batch)
    with pytest.raises(ValueError, match="Packed"):
        model.loss_sums(batch, backbone_kwargs={"mode": recipe.mode()})
    with pytest.raises(ValueError, match="Packed"):
        PreparedFBTLayout(model.backbone, batch, right_padded_causal=True)
    with pytest.raises(ValueError, match="Packed"):
        DynamicNextLatLayout.from_batch(batch, model.config)
    with pytest.raises(ValueError, match="Packed"):
        PreparedNextLatLayout.from_batch(batch, model.config)


def test_loss_config_mode_and_layout_policy_must_agree_even_without_feedback():
    recipe, model = setup("B")
    batch = oracle_batch(packed_batch())
    bad = replace(recipe.mode(), document_policy=ISOLATED_DOCUMENTS)
    with pytest.raises(ValueError, match="document_policy"):
        model.loss_sums(batch, backbone_kwargs={"mode": bad})
    with pytest.raises(ValueError, match="document_policy"):
        CampaignObjective(model, batch, mode=bad, global_counts=model.counts(batch))
    with pytest.raises(ValueError, match="document_policy"):
        validate_microbatch_inputs(model, batch, backbone_kwargs={"mode": bad})
    layout = PreparedFBTLayout(model.backbone, batch, document_policy=CONTINUOUS_STREAM)
    with pytest.raises(ValueError, match="document_policy"):
        layout.validate_execution(bad)
    with pytest.raises(ValueError, match="document_policy"):
        layout.forward(batch.input_ids, bad)
    layout.document_policy = ISOLATED_DOCUMENTS
    with pytest.raises(ValueError, match="ownership/configuration"):
        layout.validate_execution(bad)


def test_policy_serialization_preserves_isolated_defaults_and_records_stream():
    config = NextLatConfig(64)
    assert "document_policy" not in config.to_dict()
    assert NextLatConfig.from_dict(config.to_dict()) == config
    assert "document_policy" not in CampaignRecipe("NFR").to_dict()
    stream = replace(config, document_policy=CONTINUOUS_STREAM)
    assert stream.to_dict()["document_policy"] == CONTINUOUS_STREAM
    assert NextLatConfig.from_dict(stream.to_dict()) == stream
    recipe = CampaignRecipe("NFR", document_policy=CONTINUOUS_STREAM)
    assert recipe.to_dict()["document_policy"] == CONTINUOUS_STREAM
    assert recipe.mode().document_policy == CONTINUOUS_STREAM
    assert recipe.sha256 != CampaignRecipe("NFR").sha256
    for cls, args in ((NextLatConfig, (64,)), (CampaignRecipe, ("B",)), (FBTMode, ())):
        with pytest.raises(ValueError, match="document_policy"):
            cls(*args, document_policy="guess-from-eos")
    assert set(vars(packed_batch())) == {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}


def test_refilling_true_boundaries_changes_aux_counts_without_changing_forward_or_storage():
    recipe, model = setup()
    batch, noise = packed_batch(), noise_for(recipe, model)
    adapter = CampaignObjective(model, batch, mode=recipe.mode(), global_counts=model.counts(batch), feedback_noise=noise)
    pointers = tuple(t.data_ptr() for t in adapter.owned_inputs())
    signature = adapter.forward_layout.validate_execution(recipe.mode(), feedback_noise=adapter.feedback_noise)
    before = adapter.forward_layout.forward(adapter.batch.input_ids, recipe.mode(), feedback_noise=adapter.feedback_noise)
    before = tuple(t.detach().clone() for t in before.pass_hidden_states)
    # Remove only a document boundary; tokens/positions and continuous forward
    # must stay identical, while NextLat receives newly eligible pairs/triples.
    changed = replace(batch, document_ids=batch.document_ids.clone())
    changed.document_ids[0, :6] = 10
    counts = model.counts(changed)
    assert counts["latent"] > adapter.counts["latent"] and counts["ce"] == adapter.counts["ce"]
    adapter.load_batch(changed, feedback_noise=noise, global_counts=counts)
    after = adapter.forward_layout.forward(adapter.batch.input_ids, recipe.mode(), feedback_noise=adapter.feedback_noise)
    for left, right in zip(before, after.pass_hidden_states):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    adapter.forward_layout.validate_execution(recipe.mode(), expected_signature=signature, feedback_noise=adapter.feedback_noise)
    assert pointers == tuple(t.data_ptr() for t in adapter.owned_inputs())
    assert adapter.counts == counts
    # A policy change requires a new preparation/graph, even if tokens coincide.
    model.config = replace(model.config, document_policy=ISOLATED_DOCUMENTS)
    with pytest.raises(ValueError, match="objective settings"):
        adapter.validate_execution()


def test_static_selected_position_loss_and_training_paths_propagate_stream_policy():
    recipe, model = setup(jitter=0)
    reference = copy.deepcopy(model)
    batch = packed_batch()
    expected = reference.loss_sums(batch, backbone_kwargs={"mode": recipe.mode()})
    expected.total.backward()
    runtime = StaticFBTTraining(model, batch, mode=recipe.mode())
    actual = runtime.loss_sums()
    assert actual.counts == expected.counts
    for term in expected.sums:
        torch.testing.assert_close(actual.sums[term], expected.sums[term], rtol=0, atol=0)
    actual.total.backward()
    assert_grads(model, reference)


@pytest.mark.parametrize("term", ["latent", "kl"])
def test_packed_auxiliary_targets_remain_detached(term):
    batch = packed_batch()
    config = NextLatConfig(64, vocab_chunk_size=8, document_policy=CONTINUOUS_STREAM)
    hidden = torch.randn(3, 7, 64, requires_grad=True)
    embeds = torch.randn(3, 7, 64, requires_grad=True)
    readout = torch.randn(61, 64, requires_grad=True)
    predictor = NextLatPredictor(config)
    layout = DynamicNextLatLayout.from_batch(batch, config)
    losses = compute_dynamic_nextlat_loss_sums(hidden, embeds, readout, batch.input_ids, predictor, config, layout)
    hgrad, egrad, wgrad = torch.autograd.grad(losses[term], (hidden, embeds, readout), allow_unused=True)
    assert wgrad is None  # Auxiliary readout is fixed; lookup gradient tested above.
    assert torch.count_nonzero(hgrad[:, -1]) == 0  # Final hidden is only a detached target.
    assert torch.count_nonzero(egrad[:, 0]) == 0  # There is no previous-row conditioning.
    assert torch.count_nonzero(hgrad[-1]) == torch.count_nonzero(egrad[-1]) == 0


def test_later_tokens_and_other_rows_do_not_leak_backwards_through_stream_feedback():
    recipe, model = setup()
    batch, noise = packed_batch(), noise_for(recipe, model)
    core = model.backbone
    inputs = core.token_embeddings(batch.input_ids).detach().requires_grad_()
    changed = inputs.detach().clone()
    changed[0, 4:] += 11
    changed[1] += 17
    kwargs = {"mode": recipe.mode(), "attention_mask": batch.valid_mask, "document_ids": batch.document_ids,
              "feedback_noise": noise, "right_padded_causal": True, "return_logits": False}
    got = core(inputs_embeds=inputs, **kwargs)
    other = core(inputs_embeds=changed, **kwargs)
    for left, right in zip(got.pass_hidden_states, other.pass_hidden_states):
        torch.testing.assert_close(left[0, :4], right[0, :4], rtol=0, atol=0)
    got.last_hidden_state[0, :4].square().sum().backward()
    assert torch.count_nonzero(inputs.grad[0, 4:]) == 0
    assert torch.count_nonzero(inputs.grad[1:]) == 0
    with pytest.raises(ValueError, match="caches"):
        core(inputs_embeds=inputs, use_cache=True, **kwargs)
