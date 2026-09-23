"""Native RoPE table reuse preserves arithmetic and prepared-layout ownership."""
import copy
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig, _apply_rope
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_rope import RopeTables, apply_rope_tables, build_rope_tables
from cdrm.pretrained.olmo_recurrent import recurrent_layer_reference
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained import olmo_tiled
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(9242026)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("base", [10000.0, 500000.0])
@pytest.mark.parametrize("interval", [(None, None), (0, 1), (2, 5), (3, 3)])
def test_native_rotation_and_input_gradient_are_exact(dtype, base, interval):
    # Offsets, repeated and nonconsecutive coordinates, distinct across rows.
    positions = torch.tensor([[7, 9, 9, 14, 8001], [0, 3, 11, 29, 65535]])
    tables = build_rope_tables(positions, 8, base).slice(*interval)
    selected = positions[:, slice(*interval)]
    left = torch.randn(2, 3, selected.shape[1], 8, dtype=dtype, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = apply_rope_tables(left, tables)
        expected = _apply_rope(right, selected, base)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_precomputation_has_no_rng_scalar_read_or_autocast_dependence(monkeypatch):
    positions = torch.tensor([[0, 7, 10000], [4, 12, 99999]])
    rng = torch.get_rng_state().clone()
    expected = build_rope_tables(positions, 128, 10000.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("RoPE tables must not inspect device tensor values")

    with monkeypatch.context() as isolated:
        isolated.setattr(torch.Tensor, "item", forbidden)
        isolated.setattr(torch.Tensor, "__bool__", forbidden)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = build_rope_tables(positions, 128, 10000.0)
            apply_rope_tables(torch.randn(2, 2, 3, 128, dtype=torch.bfloat16), actual)
    # Table construction itself never consumes RNG; application uses no RNG.
    torch.set_rng_state(rng)
    build_rope_tables(positions, 128, 10000.0)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(actual.cos, expected.cos, atol=0, rtol=0)
    torch.testing.assert_close(actual.sin, expected.sin, atol=0, rtol=0)
    assert not actual.cos.requires_grad and not actual.sin.requires_grad
    with pytest.raises(FrozenInstanceError):
        actual.cos = expected.cos


@pytest.mark.parametrize("mutation", ["positions_rank", "positions_dtype", "head_odd", "head_bool", "base_nan"])
def test_builder_metadata_guards(mutation):
    positions, head, base = torch.zeros(2, 3, dtype=torch.long), 8, 10000.0
    if mutation == "positions_rank": positions = positions[0]
    if mutation == "positions_dtype": positions = positions.float()
    if mutation == "head_odd": head = 7
    if mutation == "head_bool": head = True
    if mutation == "base_nan": base = float("nan")
    with pytest.raises(ValueError, match="RoPE"):
        build_rope_tables(positions, head, base)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ordinary_block_reuses_query_and_prefix_key_tables_without_math_change(dtype):
    config = OLMoConfig.tiny()
    baseline = OLMoBlock(config)
    candidate = copy.deepcopy(baseline)
    x = torch.randn(2, 3, config.model_dim, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    past = tuple(torch.randn(2, config.num_heads, 2, config.head_dim, requires_grad=True) for _ in range(2))
    other_past = tuple(t.detach().clone().requires_grad_() for t in past)
    positions = torch.tensor([[11, 12, 20], [4, 9, 13]])
    keys = torch.cat((torch.tensor([[3, 7], [0, 2]]), positions), dim=1)
    key_tables = build_rope_tables(keys, config.head_dim, config.rope_freq_constant)
    mask = (torch.arange(5)[None] <= torch.arange(2, 5)[:, None])[None, None]
    kwargs = dict(query_positions=positions, key_positions=keys, mask=mask,
                  is_causal=False, attention_backend="math")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        want, want_cache = baseline(x, past=past, **kwargs)
        got, got_cache = candidate(y, past=other_past, query_rope=key_tables.slice(2, None),
                                   key_rope=key_tables, **kwargs)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    probes = tuple(torch.randn_like(t) for t in (want, *want_cache))
    torch.autograd.backward((want, *want_cache), probes)
    torch.autograd.backward((got, *got_cache), probes)
    for a, b in zip((y.grad, *(t.grad for t in other_past), *(p.grad for p in candidate.parameters())),
                    (x.grad, *(t.grad for t in past), *(p.grad for p in baseline.parameters()))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def _core(*, enabled=True, checkpoint=True):
    return OLMoFBT(OLMoTiledRTForCausalLM(replace(OLMoConfig.tiny(), num_layers=3),
        attention_backend="math", ordinary_activation_checkpointing=checkpoint, reuse_rope=enabled))


def _batch():
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    valid = torch.tensor([[False, True, True, True, True], [True, True, True, False, False]])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


@pytest.mark.parametrize("mutation", ["cos_value", "sin_value", "replacement", "storage", "shape",
                                     "dtype", "grad", "flag", "positions"])
def test_static_layout_guards_tables_and_execution(mutation):
    core = _core()
    layout = PreparedFBTLayout(core, _batch())
    layout.validate_execution()
    if mutation == "cos_value": layout.rope_tables.cos.add_(1)
    if mutation == "sin_value": layout.rope_tables.sin.zero_()
    if mutation == "replacement": layout.rope_tables = RopeTables(layout.rope_tables.cos, layout.rope_tables.sin)
    if mutation == "storage": layout.rope_tables.cos.data = layout.rope_tables.cos.data.clone()
    if mutation == "shape": layout.rope_tables.cos.data = layout.rope_tables.cos.data[:, :, :-1]
    if mutation == "dtype": layout.rope_tables.sin.data = layout.rope_tables.sin.data.to(torch.bfloat16)
    if mutation == "grad": layout.rope_tables.cos.requires_grad_(True)
    if mutation == "flag": core.backbone.reuse_rope = False
    if mutation == "positions": layout.position_ids.add_(3)
    with pytest.raises(ValueError, match="Prepared"):
        layout.validate_execution()


def test_layouts_own_distinct_tables_and_native_state_is_unchanged():
    core = _core()
    tokens = _batch()
    positions = torch.tensor([[0, 3, 7, 11, 29], [1, 2, 9, 11, 13]])
    before = {name: value.detach().clone() for name, value in core.state_dict().items()}
    first = PreparedFBTLayout(core, tokens, positions)
    second = PreparedFBTLayout(core, tokens, positions + 19)
    assert first.rope_tables.cos.data_ptr() != second.rope_tables.cos.data_ptr()
    assert not torch.equal(first.rope_tables.cos, second.rope_tables.cos)
    positions.add_(7)
    first.validate_execution(); second.validate_execution()
    assert before.keys() == core.state_dict().keys()
    for name, value in core.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert first.metadata["rope_tables_owned_here"]
    assert first.metadata["rope_table_bytes"] == 2 * 2 * 5 * core.config.head_dim * 4
    core.backbone.reuse_rope = False
    disabled = PreparedFBTLayout(core, tokens)
    assert disabled.rope_tables is None and disabled.metadata["rope_table_bytes"] == 0


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", [FBTMode(enabled=False, rt_mode=RTMode(())),
    FBTMode(num_passes=3, beta=.37, rt_mode=RTMode((0, 2), .37))])
def test_static_shared_pass_and_recomputation_match_old_rope(mode, dtype, checkpoint):
    baseline = _core(enabled=False, checkpoint=checkpoint)
    candidate = copy.deepcopy(baseline)
    candidate.backbone.reuse_rope = True
    tokens = _batch()
    positions = torch.tensor([[0, 3, 7, 11, 29], [1, 2, 9, 11, 13]])
    old = PreparedFBTLayout(baseline, tokens, positions)
    new = PreparedFBTLayout(candidate, tokens, positions)
    new.validate_execution(mode)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        want = old.forward(tokens.input_ids, mode)
        got = new.forward(tokens.input_ids, mode)
    probes = tuple(torch.randn_like(x) for x in want.pass_hidden_states)
    for actual, expected in zip(got.pass_hidden_states, want.pass_hidden_states):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.autograd.backward(want.pass_hidden_states, probes)
    torch.autograd.backward(got.pass_hidden_states, probes)
    for (name, actual), (_, expected) in zip(candidate.named_parameters(), baseline.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0, msg=name)


def test_prepared_forward_and_backward_never_rebuild_trig_tables(monkeypatch):
    core = _core(checkpoint=True)
    tokens = _batch()
    mode = FBTMode(num_passes=3, rt_mode=RTMode((0, 2), .37))
    layout = PreparedFBTLayout(core, tokens)
    layout.validate_execution(mode)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared forward or recomputation rebuilt RoPE trigonometry")

    monkeypatch.setattr(torch.Tensor, "cos", forbidden)
    monkeypatch.setattr(torch.Tensor, "sin", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    output = layout.forward(tokens.input_ids, mode)
    output.last_hidden_state.backward(torch.randn_like(output.last_hidden_state))


def test_dynamic_stack_builds_once_and_shares_tables_with_recomputation(monkeypatch):
    core = _core(checkpoint=True)
    baseline = copy.deepcopy(core.backbone)
    baseline.reuse_rope = False
    tokens = _batch()
    positions = torch.tensor([[0, 3, 7, 11, 29], [1, 2, 9, 11, 13]])
    calls = []
    original = olmo_tiled.build_rope_tables

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(olmo_tiled, "build_rope_tables", observed)
    kwargs = dict(mode=RTMode((0, 2), .37), attention_mask=tokens.valid_mask,
                  position_ids=positions, return_logits=False)
    expected = baseline(tokens.input_ids, **kwargs).last_hidden_state
    actual = core.backbone(tokens.input_ids, **kwargs).last_hidden_state
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    assert len(calls) == 1
    for (name, value), (_, reference) in zip(core.backbone.named_parameters(), baseline.named_parameters()):
        torch.testing.assert_close(value.grad, reference.grad, atol=0, rtol=0, msg=name)


def test_combined_rope_and_kv_only_prefix_cache_cotangents_match_independent_scan():
    """Both switches preserve attached history and explicit cache-output credit."""
    config = OLMoConfig.tiny()
    reference = OLMoBlock(config)
    candidate = copy.deepcopy(reference)
    source = torch.randn(2, 5, config.model_dim, requires_grad=True)
    expected_source = source.detach().clone().requires_grad_()
    prefix = tuple(torch.randn(2, config.num_heads, 3, config.head_dim, requires_grad=True) for _ in range(2))
    expected_prefix = tuple(t.detach().clone().requires_grad_() for t in prefix)
    positions = torch.tensor([[19, 22, 26, 31, 37], [13, 15, 17, 21, 22]])
    keys = torch.cat((torch.tensor([[7, 10, 14], [2, 4, 8]]), positions), dim=1)
    valid = torch.tensor([[True, False, True, True, True, False, True, True],
                          [False, False, False, False, False, True, True, True]])
    shared = dict(alpha=.37, query_positions=positions, key_positions=keys, key_valid=valid)
    actual, actual_cache = olmo_tiled.tiled_recurrent_layer(candidate, source, past=prefix,
        attention_precision="fp32", backward_memory="recompute", reuse_rope=True, kv_only_writes=True,
        **shared)
    expected, expected_cache = recurrent_layer_reference(reference, expected_source, past=expected_prefix,
        attention_backend="math", **shared)
    probes = tuple(torch.randn_like(t) for t in (actual, *actual_cache))
    torch.autograd.backward((actual, *actual_cache), probes)
    torch.autograd.backward((expected, *expected_cache), probes)
    for value, target in zip((actual, *actual_cache), (expected, *expected_cache)):
        torch.testing.assert_close(value, target, atol=4e-6, rtol=4e-6)
    for value, target in zip((source.grad, *(t.grad for t in prefix)),
                             (expected_source.grad, *(t.grad for t in expected_prefix))):
        torch.testing.assert_close(value, target, atol=8e-6, rtol=1e-4)
    for (name, value), (_, target) in zip(candidate.named_parameters(), reference.named_parameters()):
        assert value.grad is not None and target.grad is not None, name
        torch.testing.assert_close(value.grad, target.grad, atol=8e-6, rtol=1e-4, msg=name)
