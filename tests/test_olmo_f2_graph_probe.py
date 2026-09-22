"""CPU contracts for the proposed graph body; no CPU substitute for capture."""
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained import olmo_tiled as tiled_module
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_f2_graph_probe as probe


@pytest.fixture
def tiny():
    torch.manual_seed(8392)
    torch.set_num_threads(1)
    return OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math").eval()


@pytest.mark.parametrize("layers", [(), (0,), (0, 1)])
def test_graph_body_uses_public_stack_and_fixed_cotangent_with_persistent_gradients(tiny, layers, monkeypatch):
    ids, cotangent = probe.make_static_inputs(tiny.config, 2, 5, device="cpu")
    addresses = probe.allocate_gradients(tiny)
    monkeypatch.setattr(tiny, "project_logits", lambda _: (_ for _ in ()).throw(AssertionError("No vocabulary projection")))
    mode = RTMode(layers, .37)
    hidden = probe.stack_step(tiny, ids, cotangent, mode, "fp32").clone()
    first = {name: parameter.grad.clone() for name, parameter in tiny.named_parameters()}
    again = probe.stack_step(tiny, ids, cotangent, mode, "fp32")
    torch.testing.assert_close(again, hidden, rtol=0, atol=0)
    for name, parameter in tiny.named_parameters():
        torch.testing.assert_close(parameter.grad, first[name], rtol=0, atol=0)
    assert probe.gradient_addresses(tiny) == addresses
    # A separate explicit scalar objective gives the same gradient as the VJP.
    for parameter in tiny.parameters(): parameter.grad.zero_()
    public_hidden = tiny(ids, mode=mode, return_logits=False).last_hidden_state
    (public_hidden*cotangent).sum().backward()
    for name, parameter in tiny.named_parameters():
        torch.testing.assert_close(parameter.grad, first[name], rtol=0, atol=0)


def test_changed_tokens_and_external_sgd_are_effective_without_new_buffer_addresses(tiny):
    ids, cotangent = probe.make_static_inputs(tiny.config, 1, 5, device="cpu")
    addresses = probe.allocate_gradients(tiny)
    token_address = ids.data_ptr()
    mode = RTMode((0,), 1)
    first = probe.stack_step(tiny, ids, cotangent, mode, "fp32").clone()
    ids.copy_((ids+17) % tiny.config.tokenizer_vocab_size)
    changed_tokens = probe.stack_step(tiny, ids, cotangent, mode, "fp32").clone()
    assert not torch.equal(first, changed_tokens)
    optimizer = torch.optim.SGD(tiny.parameters(), lr=1e-3, foreach=False)
    optimizer.step()
    changed_weights = probe.stack_step(tiny, ids, cotangent, mode, "fp32")
    assert not torch.equal(changed_weights, changed_tokens)
    assert ids.data_ptr() == token_address and probe.gradient_addresses(tiny) == addresses


def test_tensor_equivalence_records_exactness_and_rejects_meaningful_error():
    x = torch.arange(100, dtype=torch.float32)/100
    assert probe.tensor_comparison(x, x)["bitwise_equal"]
    assert probe.tensor_comparison(x+1e-7, x)["passed"]
    assert not probe.tensor_comparison(x+.01, x)["passed"]


@pytest.mark.parametrize("value", ["0,0", "-1", "a", "0,"])
def test_layer_scope_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError, __import__("argparse").ArgumentTypeError)):
        probe.parse_layers(value)


def test_options_are_bounded_to_native_static_shape():
    args = SimpleNamespace(batch_size=1, length=5, warmup=2, repeats=3,
        precision="fp32", rt_layers=probe.parse_layers("none"), alpha=1)
    probe.validate_options(args, OLMoConfig.tiny())
    args.length = 513
    with pytest.raises(ValueError): probe.validate_options(args, OLMoConfig.tiny())


@pytest.mark.parametrize("prefix", [0, 1, 7])
def test_reconstructed_attention_keeps_prefix_values_and_uses_temporary_self(prefix):
    """A rectangular history must zero the offset self diagonal, not the main diagonal."""
    config = OLMoConfig.tiny()
    batch, length = 2, 4
    shape = (batch, config.num_heads, length, config.head_dim)
    query = torch.zeros(shape)
    temporary_value = -torch.arange(1, length+1).float()[None, None, :, None].expand(shape)
    memory_shape = (batch, config.num_heads, prefix+length, config.head_dim)
    keys = torch.zeros(memory_shape)
    values = (100+torch.arange(prefix+length).float())[None, None, :, None].expand(memory_shape)
    valid = torch.ones(batch, prefix+length, dtype=torch.bool)
    if prefix:
        valid[1, 0] = False
    spec = tiled_module._Invocation(config, .37, "fp32", False, torch.bfloat16)
    probabilities, actual = tiled_module._attention_from_completed(
        query, query, temporary_value, keys, values, valid, prefix, spec, torch.float32)
    for row in range(batch):
        for index in range(length):
            historical = values[row, :, :prefix+index][:, valid[row, :prefix+index]]
            selected = torch.cat((historical, temporary_value[row, :, index:index+1]), dim=1)
            expected = selected.mean(dim=1)
            torch.testing.assert_close(actual[row, :, index], expected, rtol=1e-6, atol=1e-5)
            torch.testing.assert_close(probabilities[row, :, index].sum(-1), torch.ones(config.num_heads))
