"""Right-prefix mask elision, invalid gradients and mutable graph-layout guards."""
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
def seeded_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(12041)


def model(checkpoint=False):
    return OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        reuse_rope=True, kv_only_writes=True, ordinary_activation_checkpointing=checkpoint))


def batch(lengths=(5, 3, 0)):
    ids = torch.arange(2, 17).view(3, 5)
    valid = torch.arange(5)[None] < torch.tensor(lengths)[:, None]
    docs = torch.arange(3)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def noise(mode):
    return tuple(torch.rand(3, 4, 32) * 2 - 1 for _ in range(3)) if mode.enabled else None


def compare_gradients(actual, expected):
    for (name, parameter), (other, reference) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference.grad, atol=4e-5, rtol=4e-4, msg=name)


@pytest.mark.parametrize("fbt,rt", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("lengths", [(5, 3, 0), (0, 0, 0), (1, 2, 4)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_eager_right_padding_matches_explicit_mask_valid_states_and_all_gradients(fbt, rt, lengths, checkpoint):
    actual, tokens = model(checkpoint), batch(lengths)
    expected = copy.deepcopy(actual)
    mode = FBTMode(enabled=fbt, num_passes=4, rt_mode=RTMode((0,) if rt else ()),
        first_pass_policy="configured-rt-v1", feedback_jitter=.02 if fbt else 0)
    jitter = noise(mode)
    x = actual.token_embeddings(tokens.input_ids).detach().requires_grad_()
    y = x.detach().clone().requires_grad_()
    kwargs = dict(attention_mask=tokens.valid_mask, document_ids=tokens.document_ids,
        mode=mode, feedback_noise=jitter, return_logits=False)
    got = actual(inputs_embeds=x, right_padded_causal=True, **kwargs)
    want = expected(inputs_embeds=y, **kwargs)
    loss, ref_loss = 0, 0
    for hidden, reference in zip(got.pass_hidden_states, want.pass_hidden_states):
        assert torch.count_nonzero(hidden[~tokens.valid_mask]) == 0
        assert torch.isfinite(hidden).all()
        torch.testing.assert_close(hidden[tokens.valid_mask], reference[tokens.valid_mask], atol=4e-6, rtol=3e-5)
        probe = torch.randn_like(hidden)
        # The new path must suppress invalid-query gradients even when the
        # consumer accidentally supplies a nonzero gradient at padding.
        loss = loss + (hidden * probe).sum()
        ref_loss = ref_loss + (reference * probe * tokens.valid_mask.unsqueeze(-1)).sum()
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(x.grad, y.grad, atol=4e-5, rtol=4e-4)
    assert torch.count_nonzero(x.grad[~tokens.valid_mask]) == 0
    compare_gradients(actual, expected)


@pytest.mark.parametrize("fbt,rt", [(False, False), (False, True), (True, False), (True, True)])
def test_dynamic_layout_refills_preserve_ownership_signature_and_eager_gradients(fbt, rt):
    core = model(checkpoint=True)
    expected = copy.deepcopy(core)
    mode = FBTMode(enabled=fbt, num_passes=4, rt_mode=RTMode((0, 1) if rt else ()),
        first_pass_policy="configured-rt-v1", feedback_jitter=.02 if fbt else 0)
    jitter = noise(mode)
    initial = batch((5, 5, 5))
    layout = PreparedFBTLayout(core, initial, right_padded_causal=True)
    signature = layout.validate_execution(mode, feedback_noise=jitter)
    addresses = [(id(value), value.data_ptr()) for value in layout._owned_tensors() if value is not None]
    rope_versions = (layout.rope_tables.cos._version, layout.rope_tables.sin._version)
    for lengths in ((5, 3, 0), (1, 0, 4), (0, 0, 0), (5, 5, 5)):
        tokens = batch(lengths)
        core.zero_grad(set_to_none=True)
        expected.zero_grad(set_to_none=True)
        layout.load_batch(tokens)
        layout.validate_batch(tokens)
        layout.validate_execution(mode, expected_signature=signature, feedback_noise=jitter)
        assert addresses == [(id(value), value.data_ptr()) for value in layout._owned_tensors() if value is not None]
        assert rope_versions == (layout.rope_tables.cos._version, layout.rope_tables.sin._version)
        assert layout.metadata["valid_tokens"] == sum(lengths)
        assert layout.metadata["dynamic_valid_prefixes"]
        got = layout.forward(tokens.input_ids, mode, feedback_noise=jitter)
        want = expected(tokens.input_ids, attention_mask=tokens.valid_mask,
            document_ids=tokens.document_ids, mode=mode, feedback_noise=jitter, return_logits=False)
        loss, ref_loss = 0, 0
        for hidden, reference in zip(got.pass_hidden_states, want.pass_hidden_states):
            torch.testing.assert_close(hidden[tokens.valid_mask], reference[tokens.valid_mask], atol=4e-6, rtol=3e-5)
            assert torch.count_nonzero(hidden[~tokens.valid_mask]) == 0
            probe = torch.randn_like(hidden)
            loss += (hidden * probe).sum()
            ref_loss += (reference * probe * tokens.valid_mask.unsqueeze(-1)).sum()
        loss.backward()
        ref_loss.backward()
        compare_gradients(core, expected)


def test_padding_tokens_cannot_affect_valid_outputs_or_input_gradients():
    core, tokens = model(), batch()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0, 1)), first_pass_policy="configured-rt-v1")
    x = torch.randn(3, 5, 32, requires_grad=True)
    changed = x.detach().clone()
    changed[~tokens.valid_mask] = torch.randn_like(changed[~tokens.valid_mask]) * 50
    kwargs = dict(attention_mask=tokens.valid_mask, document_ids=tokens.document_ids,
        mode=mode, right_padded_causal=True)
    got = core(inputs_embeds=x, **kwargs)
    other = core(inputs_embeds=changed, **kwargs)
    for actual, expected in zip(got.pass_hidden_states, other.pass_hidden_states):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    got.last_hidden_state.square().sum().backward()
    assert torch.count_nonzero(x.grad[~tokens.valid_mask]) == 0


def test_prepared_dynamic_forward_performs_no_host_value_checks(monkeypatch):
    core, tokens = model(), batch()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0,)), first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    jitter = noise(mode)
    layout = PreparedFBTLayout(core, tokens, right_padded_causal=True)
    layout.validate_execution(mode, feedback_noise=jitter)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host value access in prepared forward")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    layout.forward(tokens.input_ids, mode, feedback_noise=jitter).last_hidden_state.sum().backward()


@pytest.mark.parametrize("target", ["valid_mask", "feedback_eligible", "position_ids", "rope"])
def test_external_inplace_mutation_is_not_laundered_by_load_batch(target):
    core, tokens = model(), batch()
    layout = PreparedFBTLayout(core, tokens, right_padded_causal=True)
    value = layout.rope_tables.cos if target == "rope" else getattr(layout, target)
    value.copy_(value.clone())
    with pytest.raises(ValueError, match="Prepared layout tensors changed"):
        layout.load_batch(tokens)
    with pytest.raises(ValueError, match="Prepared layout tensors changed"):
        layout.validate_execution()


@pytest.mark.parametrize("valid", [
    [[False, True, True, False, False]] * 3,
    [[True, False, True, False, False]] * 3,
])
def test_nonprefix_padding_is_rejected_everywhere(valid):
    core, tokens = model(), batch()
    layout = PreparedFBTLayout(core, tokens, right_padded_causal=True)
    valid = torch.tensor(valid)
    docs = torch.zeros_like(tokens.document_ids).masked_fill(~valid, -1)
    invalid = replace(tokens, valid_mask=valid, document_ids=docs)
    with pytest.raises(ValueError, match="valid prefixes"):
        core(tokens.input_ids, attention_mask=valid, document_ids=docs, right_padded_causal=True)
    with pytest.raises(ValueError, match="valid prefixes"):
        PreparedFBTLayout(core, invalid, right_padded_causal=True)
    with pytest.raises(ValueError, match="valid prefixes"):
        layout.load_batch(invalid)


def test_dynamic_load_rejects_packed_documents_and_positions_and_legacy_unchanged():
    core, tokens = model(), batch()
    with pytest.raises(ValueError, match="canonical arange"):
        PreparedFBTLayout(core, tokens, torch.arange(5).expand(3, -1), right_padded_causal=True)
    layout = PreparedFBTLayout(core, tokens, right_padded_causal=True)
    docs = tokens.document_ids.clone()
    docs[0, 2:] = 17
    packed = replace(tokens, document_ids=docs)
    with pytest.raises(ValueError, match="document"):
        layout.load_batch(packed)
    legacy = PreparedFBTLayout(core, tokens)
    assert not legacy.is_causal and legacy.attention_mask is not None
    with pytest.raises(ValueError, match="explicit dynamic"):
        legacy.load_batch(tokens)
    with pytest.raises(ValueError, match="adjacency differs"):
        legacy.validate_batch(batch((5, 4, 0)))
    with pytest.raises(ValueError, match="adjacency differs"):
        layout.validate_batch(batch((5, 4, 0)))


def test_eager_padding_fast_path_rejects_cache_and_conflicting_options():
    core, tokens = model(), batch()
    with pytest.raises(ValueError, match="cache-free"):
        core.backbone(tokens.input_ids, right_padded_causal=True, use_cache=True)
    with pytest.raises(ValueError, match="either"):
        core(tokens.input_ids, full_valid_causal=True, right_padded_causal=True)
    with pytest.raises(TypeError, match="boolean"):
        core(tokens.input_ids, right_padded_causal=1)


def test_candidate_preflight_never_mutates_loaded_layout_or_authorizes_buffer_writes():
    core, tokens = model(), batch()
    layout = PreparedFBTLayout(core, tokens, right_padded_causal=True)
    before = layout._owned_signature()
    metadata = layout.metadata
    candidate = batch((1, 5, 4))
    valid, eligible = layout.validate_replacement_batch(candidate)
    assert layout._owned_signature() == before and layout.metadata == metadata
    assert torch.equal(valid, candidate.valid_mask)
    assert torch.equal(eligible, valid[:, :-1] & valid[:, 1:])
    valid.zero_()
    eligible.zero_()
    assert layout._owned_signature() == before and layout.metadata == metadata
    layout.validate_batch(tokens)
    with pytest.raises(ValueError, match="adjacency differs"):
        layout.validate_batch(candidate)
    layout.load_batch(candidate)
    layout.validate_batch(candidate)
