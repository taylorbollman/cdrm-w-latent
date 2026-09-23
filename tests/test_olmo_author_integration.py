"""Bounded native/author RT integration, ownership and unsupported-scope guards.

CPU FP32 tests exercise the real uncompiled author backward. Flash routing is
structural here; actual Flash/CUDA graphs require the separate GPU protocol.
"""
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained import olmo_author
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_training import StaticFBTTraining


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(920241)
    yield
    torch.set_num_threads(previous)


def batch(*, padded=False):
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid[1, -1] = False
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False
    latent[1, 1] = False
    kl[0, 3] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def base(implementation="native", **kwargs):
    config = replace(OLMoConfig.tiny(), model_dim=64, mlp_intermediate_size=128, num_layers=3)
    options = dict(attention_backend="math", attention_precision="fp32",
        ordinary_activation_checkpointing=True, reuse_rope=True, kv_only_writes=True,
        backward_memory="recompute", rt_implementation=implementation,
        author_compiled_helpers=False)
    options.update(kwargs)
    return OLMoTiledRTForCausalLM(config, **options).train()


def core_pair():
    reference, candidate = OLMoFBT(base()), OLMoFBT(base("author"))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


def assert_gradients(candidate, reference):
    actual, expected = dict(candidate.named_parameters()), dict(reference.named_parameters())
    assert actual.keys() == expected.keys()
    for name in actual:
        a, b = actual[name].grad, expected[name].grad
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.isfinite(a).all(), name
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4, msg=name)


MODES = [
    pytest.param(FBTMode(enabled=False, rt_mode=RTMode(())), id="ordinary"),
    pytest.param(FBTMode(enabled=False, rt_mode=RTMode((0, 2))), id="rt"),
    pytest.param(FBTMode(num_passes=2, beta=.37, rt_mode=RTMode((0, 2))), id="k2"),
    pytest.param(FBTMode(num_passes=3, beta=.37, rt_mode=RTMode((0, 2))), id="k3-shared"),
]


def hidden_forward(core, tokens, mode, *, prepared):
    positions = torch.tensor([[3, 4, 7, 12, 31], [0, 2, 5, 13, 17]])
    if prepared:
        layout = PreparedFBTLayout(core, tokens, positions)
        layout.validate_execution(mode)
        output = layout.forward(tokens.input_ids, mode)
        return output, output.embeddings
    lookups = []
    hook = core.token_embeddings.register_forward_hook(lambda module, args, output: lookups.append(output))
    try:
        output = core(tokens.input_ids, mode=mode, position_ids=positions,
            attention_mask=tokens.valid_mask, document_ids=tokens.document_ids, return_logits=False)
    finally:
        hook.remove()
    assert len(lookups) == 1
    return output, lookups[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prepared", [False, True])
def test_public_and_prepared_pass_states_raw_gradients_and_state_ownership(mode, prepared):
    reference, candidate = core_pair()
    tokens = batch()
    before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    ownership = {name: id(value) for name, value in candidate.named_parameters()}
    expected, expected_embeddings = hidden_forward(reference, tokens, mode, prepared=prepared)
    actual, actual_embeddings = hidden_forward(candidate, tokens, mode, prepared=prepared)
    expected_embeddings.retain_grad()
    actual_embeddings.retain_grad()
    assert len(actual.pass_hidden_states) == len(expected.pass_hidden_states)
    probes = [torch.randn_like(value) for value in expected.pass_hidden_states]
    actual_loss = expected_loss = 0
    for a, b, probe in zip(actual.pass_hidden_states, expected.pass_hidden_states, probes):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        actual_loss = actual_loss + (a * probe).sum()
        expected_loss = expected_loss + (b * probe).sum()
    # A readout cotangent also exercises the same tied parameter through both
    # lookup and output projection, rather than only through token inputs.
    readout_probe = torch.randn(*tokens.input_ids.shape, candidate.config.vocab_size)
    actual_loss = actual_loss + (candidate.project_logits(actual.last_hidden_state) * readout_probe).sum()
    expected_loss = expected_loss + (reference.project_logits(expected.last_hidden_state) * readout_probe).sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_embeddings.grad, expected_embeddings.grad, atol=3e-5, rtol=3e-4)
    assert_gradients(candidate, reference)
    assert candidate.readout_weight is candidate.token_embeddings.weight
    assert ownership == {name: id(value) for name, value in candidate.named_parameters()}
    assert before.keys() == candidate.state_dict().keys()
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("passes", [2, 3])
@pytest.mark.parametrize("prepared", [False, True])
def test_combined_nextlat_losses_and_every_parameter_gradient_match_native(passes, prepared):
    reference_core, candidate_core = core_pair()
    config = NextLatConfig(model_dim=64, proj_factor=2, lambda_latent=.3, lambda_kl=.7,
                           vocab_chunk_size=3, ce_chunk_size=4)
    reference = FBTNextLatLM(reference_core, config, gamma=.6)
    candidate = FBTNextLatLM(candidate_core, config, gamma=.6)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    tokens = batch()
    mode = FBTMode(num_passes=passes, beta=.37, rt_mode=RTMode((0, 2)))
    expected = reference.loss_sums(tokens, backbone_kwargs={"mode": mode})
    actual = (StaticFBTTraining(candidate, tokens, mode=mode).loss_sums() if prepared else
              candidate.loss_sums(tokens, backbone_kwargs={"mode": mode}))
    assert actual.counts == expected.counts
    assert actual.weights == expected.weights
    assert actual.pass_coefficients == expected.pass_coefficients
    for a, b in zip(actual.pass_losses, expected.pass_losses):
        for term in ("ce", "latent", "kl"):
            torch.testing.assert_close(a.sums[term], b.sums[term], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual.total, expected.total, atol=3e-6, rtol=3e-5)
    actual.total.backward()
    expected.total.backward()
    assert_gradients(candidate, reference)
    assert all(parameter.grad is not None for parameter in candidate.parameters())


def test_k3_autograd_grad_accumulates_shared_calls_without_real_leaf_grad_side_effects():
    reference, candidate = core_pair()
    mode = FBTMode(num_passes=3, beta=.5, rt_mode=RTMode((0, 2)))
    expected, _ = hidden_forward(reference, batch(), mode, prepared=True)
    actual, _ = hidden_forward(candidate, batch(), mode, prepared=True)
    for model in (reference, candidate):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
    probes = [torch.randn_like(value) for value in expected.pass_hidden_states]
    want = torch.autograd.grad(expected.pass_hidden_states, tuple(reference.parameters()), probes)
    got = torch.autograd.grad(actual.pass_hidden_states, tuple(candidate.parameters()), probes)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    for parameter in candidate.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, .125))


def test_prepared_author_forward_and_backward_have_no_host_tensor_reads(monkeypatch):
    core = OLMoFBT(base("author"))
    tokens = batch()
    mode = FBTMode(num_passes=2, rt_mode=RTMode((0, 2)))
    layout = PreparedFBTLayout(core, tokens)
    layout.validate_execution(mode)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared author execution read a device tensor on the host")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    output = layout.forward(tokens.input_ids, mode)
    output.last_hidden_state.backward(torch.randn_like(output.last_hidden_state))


def test_ordinary_blocks_keep_sdpa_routing_for_flash_and_fbt_first_pass_is_ordinary(monkeypatch):
    # CUDA callers select Flash through the outer SDPA backend context. The
    # native block configuration itself remains "sdpa", not a "flash" option.
    core = OLMoFBT(base("author", attention_backend="sdpa"))
    indices = {id(layer): index for index, layer in enumerate(core.backbone.layers)}
    ordinary, recurrent = [], []
    original_ordinary, original_author = OLMoBlock.forward, olmo_author.author_tiled_recurrent_layer

    def ordinary_cpu(self, *args, **kwargs):
        ordinary.append((indices[id(self)], kwargs["attention_backend"]))
        # Observe the requested backend, then explicitly use CPU math for this
        # routing-only test. This is not CUDA Flash execution evidence.
        return original_ordinary(self, *args, **(kwargs | {"attention_backend": "math"}))

    def observed_author(layer, *args, **kwargs):
        recurrent.append(indices[id(layer)])
        return original_author(layer, *args, **kwargs)

    monkeypatch.setattr(OLMoBlock, "forward", ordinary_cpu)
    monkeypatch.setattr(olmo_author, "author_tiled_recurrent_layer", observed_author)
    mode = FBTMode(num_passes=2, rt_mode=RTMode((0, 2)))
    layout = PreparedFBTLayout(core, batch())
    layout.validate_execution(mode)
    layout.forward(batch().input_ids, mode)
    assert ordinary == [(0, "sdpa"), (1, "sdpa"), (2, "sdpa"), (1, "sdpa")]
    assert recurrent == [0, 2]


@pytest.mark.parametrize("mode", [FBTMode(enabled=False, rt_mode=RTMode(())),
    FBTMode(num_passes=1, rt_mode=RTMode((0, 2), .37))])
def test_inactive_author_rt_keeps_padded_ordinary_and_k1_bypass(mode, monkeypatch):
    reference, candidate = core_pair()

    def forbidden(*args, **kwargs):
        raise AssertionError("An ordinary-only invocation entered author RT")

    monkeypatch.setattr(olmo_author, "author_tiled_recurrent_layer", forbidden)
    tokens = batch(padded=True)
    expected, _ = hidden_forward(reference, tokens, mode, prepared=True)
    actual, _ = hidden_forward(candidate, tokens, mode, prepared=True)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0)


@pytest.mark.parametrize("unsupported", ["fractional", "zero", "padding", "export_cache", "prefix_cache"])
def test_public_author_rejects_unsupported_active_rt_scope(unsupported):
    model = base("author", ordinary_activation_checkpointing=False)
    tokens = batch(padded=unsupported == "padding")
    kwargs = {"mode": RTMode((0, 2)), "attention_mask": tokens.valid_mask}
    if unsupported in ("fractional", "zero"):
        kwargs["mode"] = RTMode((0, 2), .37 if unsupported == "fractional" else 0)
    elif unsupported == "export_cache":
        kwargs["use_cache"] = True
    elif unsupported == "prefix_cache":
        cache = model(tokens.input_ids[:, :1], mode=RTMode(()), use_cache=True).past_key_values
        kwargs.pop("attention_mask")
        kwargs["past_key_values"] = cache
    with pytest.raises(ValueError):
        model(tokens.input_ids, **kwargs)


@pytest.mark.parametrize("padded", [False, True])
def test_static_author_scope_is_rejected_before_execution(padded):
    core = OLMoFBT(base("author"))
    layout = PreparedFBTLayout(core, batch(padded=padded))
    mode = FBTMode(rt_mode=RTMode((0, 2), 1 if padded else .37))
    with pytest.raises(ValueError):
        layout.validate_execution(mode)


OPTIONS = [("rt_implementation", "native"), ("author_precision", "fp32_state"),
    ("author_compiled_helpers", True), ("author_bwd_mlp_chunks", 1), ("author_autocast_cache", False)]


@pytest.mark.parametrize("name,value", OPTIONS)
def test_author_options_are_recorded_and_changing_them_invalidates_prepared_execution(name, value):
    core = OLMoFBT(base("author"))
    layout = PreparedFBTLayout(core, batch())
    mode = FBTMode(rt_mode=RTMode((0, 2)))
    signature = layout.validate_execution(mode)
    assert signature[name] == getattr(core.backbone, name)
    assert layout.metadata[name] == getattr(core.backbone, name)
    setattr(core.backbone, name, value)
    with pytest.raises(ValueError, match="Prepared"):
        layout.validate_execution(mode)


@pytest.mark.parametrize("name,value", OPTIONS)
def test_ordinary_cache_records_author_options_and_rejects_changed_context(name, value):
    model = base("author", ordinary_activation_checkpointing=False)
    cache = model(batch().input_ids[:, :1], mode=RTMode(()), use_cache=True).past_key_values
    assert getattr(cache, name) == getattr(model, name)
    setattr(model, name, value)
    with pytest.raises(ValueError, match="Cached"):
        model(batch().input_ids[:, 1:2], mode=RTMode(()), past_key_values=cache)


def test_author_choice_preserves_checkpoint_keys_tied_parameters_and_native_default():
    candidate = base("author")
    ordinary = OLMoTiledRTForCausalLM(candidate.config, attention_backend="math")
    assert ordinary.rt_implementation == "native"
    assert candidate.state_dict().keys() == ordinary.state_dict().keys()
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in ordinary.parameters())
    ordinary.load_state_dict(candidate.state_dict(), strict=True)
    assert candidate.readout_weight is candidate.token_embeddings.weight
    assert ordinary.readout_weight is ordinary.token_embeddings.weight
    assert len(list(candidate.parameters())) == len({id(p) for p in candidate.parameters()})


@pytest.mark.parametrize("options", [
    {"rt_implementation": "unknown"}, {"author_precision": "unknown"},
    {"author_compiled_helpers": 1}, {"author_bwd_mlp_chunks": 0},
    {"author_bwd_mlp_chunks": True}, {"author_autocast_cache": 1},
    {"reuse_rope": False},
])
def test_invalid_author_configuration_is_not_silently_ignored(options):
    with pytest.raises((TypeError, ValueError)):
        base("author", **options)
