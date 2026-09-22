"""Prepared finite forwarding matches the public, independently validated path."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_tiny_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(6931)


def model(checkpoint=False):
    return OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        ordinary_activation_checkpointing=checkpoint))


def batch(padded=False):
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid = torch.tensor([[False, True, True, False, True], [True, True, True, False, False]])
    docs = torch.tensor([[3]*5, [8]*5]).masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


MODES = [FBTMode(enabled=False, rt_mode=RTMode(())),
         FBTMode(enabled=False, rt_mode=RTMode((0,), .37)),
         FBTMode(num_passes=1, rt_mode=RTMode((0,))),
         FBTMode(num_passes=3, beta=0, rt_mode=RTMode(())),
         FBTMode(num_passes=2, beta=.37, rt_mode=RTMode((0,), .37)),
         FBTMode(num_passes=3, beta=1, rt_mode=RTMode((0, 1), 1))]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("padded,checkpoint", [(False, False), (True, False), (False, True)])
def test_pass_states_and_all_gradients_match_canonical(mode, padded, checkpoint):
    actual = model(checkpoint)
    expected = copy.deepcopy(actual)
    tokens = batch(padded)
    # Nonconsecutive RoPE positions prove that attention causality still follows
    # token order, independently of the supplied positional coordinates.
    positions = torch.tensor([[0, 3, 4, 8, 11], [2, 3, 9, 10, 15]])
    prepared = PreparedFBTLayout(actual, tokens, positions)
    prepared.validate_execution(mode)
    observed_lookup = []
    handle = expected.token_embeddings.register_forward_hook(
        lambda module, args, output: observed_lookup.append(output))
    try:
        want = expected(tokens.input_ids, attention_mask=tokens.valid_mask,
            document_ids=tokens.document_ids, position_ids=positions,
            mode=mode, return_logits=False)
    finally:
        handle.remove()
    got = prepared.forward(tokens.input_ids, mode)
    assert len(observed_lookup) == 1
    got.embeddings.retain_grad()
    observed_lookup[0].retain_grad()
    assert len(got.pass_hidden_states) == len(want.pass_hidden_states)
    assert got.logits is None
    assert got.last_hidden_state is got.pass_hidden_states[-1]
    got_loss, want_loss = 0, 0
    for hidden, reference in zip(got.pass_hidden_states, want.pass_hidden_states):
        torch.testing.assert_close(hidden, reference, atol=3e-6, rtol=2e-5)
        probe = torch.randn_like(hidden)
        got_loss = got_loss + (hidden * probe).sum()
        want_loss = want_loss + (reference * probe).sum()
    # Also check that consumers can share precisely this embedding lookup.
    embed_probe = torch.randn_like(got.embeddings)
    got_loss = got_loss + (got.embeddings * embed_probe).sum()
    want_loss = want_loss + (observed_lookup[0] * embed_probe).sum()
    got_loss.backward()
    want_loss.backward()
    torch.testing.assert_close(got.embeddings.grad, observed_lookup[0].grad, atol=3e-5, rtol=3e-4)
    for (name, p), (other, q) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)
    assert prepared.metadata["all_valid_causal_lowering_proved"] is (not padded)
    assert (prepared.attention_mask is None) is (not padded)
    assert prepared.is_causal is (not padded)


def test_fresh_tokens_and_in_place_updates_are_allowed_and_used():
    core = model()
    tokens = batch()
    mode = FBTMode(beta=.5, rt_mode=RTMode((0,)))
    layout = PreparedFBTLayout(core, tokens)
    signature = layout.validate_execution(mode)
    before = layout.forward(tokens.input_ids, mode).last_hidden_state.detach().clone()
    fresh = replace(tokens, input_ids=tokens.input_ids + 5, document_ids=tokens.document_ids + 100)
    # CPU-staged batches also remain valid when the execution device is CUDA;
    # no CUDA tensor or GPU work is required for this validation-only contract.
    layout._device = torch.device("cuda")
    layout.validate_batch(fresh)
    layout._device = torch.device("cpu")
    with torch.no_grad():
        core.fusion.state_proj.weight.add_(.03)
        core.readout_weight.add_(.01)
    layout.validate_execution(mode, expected_signature=signature)
    after = layout.forward(fresh.input_ids, mode).last_hidden_state
    expected = core(fresh.input_ids, attention_mask=fresh.valid_mask,
        document_ids=fresh.document_ids, mode=mode, return_logits=False).last_hidden_state
    assert not torch.equal(after, before)
    torch.testing.assert_close(after, expected, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("mutation", ["validity", "packed", "negative_doc", "token_range", "positions", "shape"])
def test_fresh_layout_changes_fail_before_replay(mutation):
    tokens = batch(True)
    layout = PreparedFBTLayout(model(), tokens)
    positions = None
    if mutation == "validity":
        value = tokens.valid_mask.clone()
        value[0, 2] = False
        tokens = replace(tokens, valid_mask=value)
    elif mutation == "packed":
        value = tokens.document_ids.clone()
        value[0, 2] += 1
        tokens = replace(tokens, document_ids=value)
    elif mutation == "negative_doc":
        value = tokens.document_ids.clone()
        value[0, 2] = -1
        tokens = replace(tokens, document_ids=value)
    elif mutation == "token_range":
        # Padding tokens still pass through the shared lookup.
        value = tokens.input_ids.clone()
        value[0, 0] = 999999
        tokens = replace(tokens, input_ids=value)
    elif mutation == "positions":
        positions = layout.position_ids + 1
    elif mutation == "shape":
        tokens = NextLatBatch(tokens.input_ids[:, :-1], tokens.valid_mask[:, :-1], tokens.document_ids[:, :-1])
    with pytest.raises(ValueError):
        layout.validate_batch(tokens, positions)


@pytest.mark.parametrize("mutation", ["training", "backend", "checkpoint", "parameter", "dtype_roundtrip",
    "buffer", "owned_mask", "owned_positions", "owned_flag", "freeze"])
def test_execution_changes_require_new_preparation(mutation):
    core = model()
    layout = PreparedFBTLayout(core, batch(True))
    if mutation == "training":
        core.backbone.layers[1].eval()
    elif mutation == "backend":
        core.backbone.attention_backend = "sdpa"
    elif mutation == "checkpoint":
        core.backbone.ordinary_activation_checkpointing = True
    elif mutation == "parameter":
        core.fusion.state_proj.weight = torch.nn.Parameter(core.fusion.state_proj.weight.detach().clone())
    elif mutation == "dtype_roundtrip":
        core.fusion.to(torch.float64).to(torch.float32)
    elif mutation == "buffer":
        core.fusion.output_scale.add_(.1)
    elif mutation == "owned_mask":
        layout.valid_mask[0, 0] = True
    elif mutation == "owned_positions":
        layout.position_ids = layout.position_ids.clone()
    elif mutation == "owned_flag":
        layout.is_causal = True
    elif mutation == "freeze":
        core.fusion.state_proj.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="Prepared"):
        layout.validate_execution()


def test_external_mode_and_runtime_signature_are_checked_separately():
    layout = PreparedFBTLayout(model(), batch())
    signature = layout.validate_execution()
    with pytest.raises(ValueError, match="outside"):
        layout.validate_execution(FBTMode(rt_mode=RTMode((2,))))
    with pytest.raises(ValueError, match="context"):
        layout.validate_execution(FBTMode(beta=.37), expected_signature=signature)
    # Preparation does not impose its initial disabled autocast on later runs.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert layout.validate_execution()["autocast_enabled"] is True
        with pytest.raises(ValueError, match="context"):
            layout.validate_execution(expected_signature=signature)


def test_prepared_body_avoids_public_host_value_validation(monkeypatch):
    core = model()
    tokens = batch(True)
    layout = PreparedFBTLayout(core, tokens)
    mode = FBTMode(num_passes=3, beta=.5, rt_mode=RTMode((0,), .37))
    layout.validate_execution(mode)
    def forbidden(*args, **kwargs):
        raise AssertionError("Host-value validation entered tensor-only forward")
    monkeypatch.setattr(core.backbone, "_prepare_inputs", forbidden)
    monkeypatch.setattr(core, "_documents", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    output = layout.forward(tokens.input_ids, mode)
    output.last_hidden_state.sum().backward()


def test_preparation_owns_caller_metadata_without_consuming_rng():
    tokens = batch(True)
    core = model()
    rng = torch.get_rng_state().clone()
    positions = torch.arange(5).expand(2, -1).clone()
    layout = PreparedFBTLayout(core, tokens, positions)
    assert torch.equal(rng, torch.get_rng_state())
    valid_before = layout.valid_mask.clone()
    positions_before = layout.position_ids.clone()
    tokens.valid_mask.fill_(True)
    positions.add_(100)
    torch.testing.assert_close(layout.valid_mask, valid_before, atol=0, rtol=0)
    torch.testing.assert_close(layout.position_ids, positions_before, atol=0, rtol=0)
    layout.validate_execution()
