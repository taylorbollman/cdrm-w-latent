"""Native phase preservation, ordinary-only dispatch and fixed-layout ownership."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained import olmo, olmo_ordinary, olmo_static, olmo_tiled
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_rope import DaoRopeTables, RopeTables, apply_rope_tables, build_dao_rope_tables, build_rope_tables
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(35202)


def model(**options):
    return OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", reuse_rope=True, **options)


def inputs(padded=False):
    ids = torch.tensor([[2, 3, 5, 7, 11], [13, 17, 19, 23, 29]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid[0, 0] = False
        valid[1, -1] = False
    docs = torch.zeros_like(ids).masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def positions():
    return torch.tensor([[0, 1, 7, 12, 31], [3, 5, 5, 19, 47]])


@pytest.fixture
def mock_dao(monkeypatch):
    """Exercise production casts/layout, replacing only the CUDA kernel/guard."""
    calls = []

    def fake(x, cos, sin, *, interleaved, inplace, seqlen_offsets):
        assert x.dtype == cos.dtype == sin.dtype == torch.float32
        assert not interleaved and not inplace
        assert cos.is_contiguous() and sin.is_contiguous() and seqlen_offsets.is_contiguous()
        assert not torch.is_autocast_enabled(x.device.type)
        index = torch.arange(x.shape[1])[None] + seqlen_offsets[:, None]
        c = torch.cat((cos[index], cos[index]), -1).unsqueeze(2)
        s = torch.cat((sin[index], sin[index]), -1).unsqueeze(2)
        first, second = x.chunk(2, -1)
        calls.append((x.shape, cos.data_ptr(), sin.data_ptr(), seqlen_offsets.data_ptr()))
        return x * c + torch.cat((-second, first), -1) * s

    monkeypatch.setattr(olmo_ordinary, "_load_dao_rope", lambda: fake)
    monkeypatch.setattr(olmo_ordinary, "_validate_dao_rope_inputs", lambda *args: None)
    return calls


def test_compaction_uses_exact_native_batch_specific_nonconsecutive_phases():
    native = build_rope_tables(positions(), 8, 10000.)
    compact = build_dao_rope_tables(native)
    assert compact.cos.shape == compact.sin.shape == (10, 4)
    assert compact.cos.is_contiguous() and compact.sin.is_contiguous()
    torch.testing.assert_close(compact.offsets, torch.tensor([0, 5]), rtol=0, atol=0)
    for batch in range(2):
        for token in range(5):
            row = compact.offsets[batch] + token
            torch.testing.assert_close(compact.cos[row], native.cos[batch, 0, token, :4], rtol=0, atol=0)
            torch.testing.assert_close(compact.sin[row], native.sin[batch, 0, token, :4], rtol=0, atol=0)


def test_noncanonical_native_tables_cannot_silently_drop_second_half():
    native = build_rope_tables(positions(), 8, 10000.)
    changed = native.cos.clone()
    changed[..., -1] += .01
    with pytest.raises(ValueError, match="identical native split-half"):
        build_dao_rope_tables(RopeTables(changed, native.sin))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_adapter_preserves_fp32_cast_boundary_values_raw_vjp_and_input_storage(dtype, mock_dao):
    native = build_rope_tables(positions(), 8, 10000.)
    compact = build_dao_rope_tables(native)
    x = torch.randn(2, 5, 4, 8).transpose(1, 2).to(dtype).requires_grad_()
    before = x.detach().clone()
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = olmo_ordinary.apply_dao_rope(x, compact)
        expected = apply_rope_tables(reference_x, native)
    assert actual.dtype == dtype and actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    a = torch.autograd.grad(actual, x, probe)[0]
    b = torch.autograd.grad(expected, reference_x, probe)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(mock_dao) == 1 and mock_dao[0][0] == (2, 5, 4, 8)


def test_cpu_fails_before_dao_import_and_has_no_native_fallback(monkeypatch):
    def unexpected():
        pytest.fail("Dao imported on CPU")
    monkeypatch.setattr(olmo_ordinary, "_load_dao_rope", unexpected)
    tables = build_dao_rope_tables(build_rope_tables(positions(), 8, 10000.))
    with pytest.raises(ValueError, match="requires CUDA"):
        olmo_ordinary.apply_dao_rope(torch.randn(2, 4, 5, 8), tables)


def test_dao_import_and_kernel_errors_propagate(monkeypatch):
    monkeypatch.setattr(olmo_ordinary, "_validate_dao_rope_inputs", lambda *args: None)
    def broken(*args, **kwargs):
        raise RuntimeError("unavailable Dao kernel")
    monkeypatch.setattr(olmo_ordinary, "_load_dao_rope", lambda: broken)
    compact = build_dao_rope_tables(build_rope_tables(positions(), 8, 10000.))
    with pytest.raises(RuntimeError, match="unavailable Dao kernel"):
        olmo_ordinary.apply_dao_rope(torch.randn(2, 4, 5, 8), compact)


@pytest.mark.parametrize("value", ["fa4", "compiled", None, False])
def test_unknown_rope_backend_rejected(value):
    with pytest.raises(ValueError, match="ordinary_rope_backend"):
        model(ordinary_rope_backend=value)


def test_dao_requires_reused_tables():
    with pytest.raises(ValueError, match="reuse_rope=True"):
        OLMoTiledRTForCausalLM(OLMoConfig.tiny(), ordinary_rope_backend="dao")


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_prepared_dao_matches_public_native_all_parameter_gradients_and_reuses_tables(padded, checkpoint, mock_dao, monkeypatch):
    expected = model(ordinary_activation_checkpointing=checkpoint)
    actual = copy.deepcopy(expected)
    actual.ordinary_rope_backend = "dao"
    tokens = inputs(padded)
    core = OLMoFBT(actual)
    layout = PreparedFBTLayout(core, tokens, positions())
    mode = FBTMode(enabled=False)
    signature = layout.validate_execution(mode)
    assert signature["ordinary_rope_backend"] == "dao"
    assert layout.metadata["ordinary_rope_tables_owned_here"]
    expected_bytes = (2 * 5 * 8 * 4) + 2 * 8
    assert layout.metadata["ordinary_rope_table_bytes"] == expected_bytes
    def unexpected(*args, **kwargs):
        pytest.fail("compact tables rebuilt inside prepared execution or checkpoint replay")
    monkeypatch.setattr(olmo_static, "build_dao_rope_tables", unexpected)
    monkeypatch.setattr(olmo_tiled, "build_dao_rope_tables", unexpected)
    want = expected(tokens.input_ids, attention_mask=tokens.valid_mask,
        position_ids=positions(), mode=RTMode(())).last_hidden_state
    got = layout.forward(tokens.input_ids, mode).last_hidden_state
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    probe = torch.randn_like(want)
    want_grads = torch.autograd.grad(want, tuple(expected.parameters()), probe)
    got_grads = torch.autograd.grad(got, tuple(actual.parameters()), probe)
    for a, b in zip(want_grads, got_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(mock_dao) == 8 if checkpoint else len(mock_dao) == 4
    assert len({tuple(call[1:]) for call in mock_dao}) == 1
    assert tuple(actual.state_dict()) == tuple(expected.state_dict())
    assert actual.readout_weight is actual.token_embeddings.weight and not tuple(actual.buffers())


def test_public_dao_packs_once_then_runs_all_ordinary_layers(mock_dao, monkeypatch):
    actual = model(ordinary_rope_backend="dao")
    expected = copy.deepcopy(actual)
    expected.ordinary_rope_backend = "native"
    packed = []
    real = olmo_tiled.build_dao_rope_tables
    def counted(tables):
        packed.append(tables)
        return real(tables)
    monkeypatch.setattr(olmo_tiled, "build_dao_rope_tables", counted)
    tokens = inputs(True)
    got = actual(tokens.input_ids, attention_mask=tokens.valid_mask,
        position_ids=positions(), mode=RTMode(())).logits
    want = expected(tokens.input_ids, attention_mask=tokens.valid_mask,
        position_ids=positions(), mode=RTMode(())).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert len(packed) == 1 and len(mock_dao) == 4


@pytest.mark.parametrize("mutation", ["backend", "owner", "cos", "sin", "offsets", "storage"])
def test_static_layout_rejects_compact_table_and_backend_changes(mutation):
    core = OLMoFBT(model(ordinary_rope_backend="dao"))
    layout = PreparedFBTLayout(core, inputs(), positions())
    mode = FBTMode(enabled=False)
    signature = layout.validate_execution(mode)
    if mutation == "backend":
        core.backbone.ordinary_rope_backend = "native"
    elif mutation == "owner":
        layout.ordinary_rope_tables = replace(layout.ordinary_rope_tables)
    elif mutation == "storage":
        layout.ordinary_rope_tables.cos.data = layout.ordinary_rope_tables.cos.clone()
    else:
        getattr(layout.ordinary_rope_tables, mutation).add_(1)
    with pytest.raises(ValueError, match="changed"):
        layout.validate_execution(mode, expected_signature=signature)


def test_native_cache_pins_rope_backend_and_dao_export_is_rejected():
    base = model().eval()
    with torch.no_grad():
        cached = base(inputs().input_ids, mode=RTMode(()), use_cache=True).past_key_values
        assert cached.ordinary_rope_backend == "native"
        base.ordinary_rope_backend = "dao"
        with pytest.raises(ValueError, match="Cached ordinary execution differs"):
            base(torch.tensor([[31], [37]]), mode=RTMode(()), past_key_values=cached)
        with pytest.raises(ValueError, match="exported caches"):
            base(inputs().input_ids, mode=RTMode(()), use_cache=True)


def test_all_rt_execution_never_calls_dao_or_compacts_tables(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Ordinary Dao RoPE was used by tiled RT")
    monkeypatch.setattr(olmo, "apply_dao_rope", unexpected)
    monkeypatch.setattr(olmo_tiled, "build_dao_rope_tables", unexpected)
    expected = model()
    actual = copy.deepcopy(expected)
    actual.ordinary_rope_backend = "dao"
    mode = RTMode((0, 1))
    got, want = actual(inputs().input_ids, mode=mode).logits, expected(inputs().input_ids, mode=mode).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fbt_ordinary_bootstrap_and_partial_rt_only_rotate_ordinary_blocks(mock_dao):
    core = OLMoFBT(model(ordinary_rope_backend="dao"))
    layout = PreparedFBTLayout(core, inputs(), positions())
    mode = FBTMode(num_passes=2, rt_mode=RTMode((0,)))
    layout.validate_execution(mode)
    layout.forward(inputs().input_ids, mode).last_hidden_state.square().mean().backward()
    # Bootstrap: two ordinary blocks; second pass: only block one ordinary.
    assert len(mock_dao) == 6
