"""Independent history and gradient semantics for original OLMo recurrence."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, recurrent_layer_reference
from cdrm.pretrained.olmo_recurrent_oracle import olmo_recurrent_block_oracle, olmo_recurrent_model_oracle
from cdrm.pretrained.olmo_reference import build_olmo_reference, olmo_reference_forward
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(2411)


def _models():
    config = OLMoConfig.tiny()
    native = build_olmo_reference(config)
    model = OLMoRTForCausalLM(config, attention_backend="math")
    model.load_state_dict(native.state_dict(), strict=True)
    return native, model


def _compare_gradients(actual, expected, *, atol=5e-6, rtol=6e-5):
    parameters = dict(actual.named_parameters())
    for name, parameter in expected.named_parameters():
        assert parameter.grad is not None, name
        assert parameters[name].grad is not None, name
        torch.testing.assert_close(parameters[name].grad, parameter.grad, atol=atol, rtol=rtol, msg=name)


@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_alpha_zero_oracle_recovers_native_ordinary_outputs_and_all_gradients(selected):
    oracle, _ = _models()
    ordinary = copy.deepcopy(oracle)
    ids = torch.tensor([[1, 2, 3, 5, 8], [9, 7, 6, 4, 2]])
    with sdpa_kernel(SDPBackend.MATH):
        expected = olmo_reference_forward(ordinary, ids)
        actual = olmo_recurrent_model_oracle(oracle, ids, selected_layers=selected, alpha=0.0)
        F.cross_entropy(expected.logits.flatten(0, 1), ids.flip(-1).flatten()).backward()
        F.cross_entropy(actual.logits.flatten(0, 1), ids.flip(-1).flatten()).backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=3e-6)
    _compare_gradients(oracle, ordinary)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_scan_matches_oracle_outputs_all_parameter_and_input_gradients(alpha, selected):
    native, model = _models()
    x = torch.randn(2, 5, model.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    targets = torch.randint(0, model.config.vocab_size, (2, 5))
    with sdpa_kernel(SDPBackend.MATH):
        actual = model(inputs_embeds=x, mode=RTMode(selected, alpha))
        expected = olmo_recurrent_model_oracle(native, inputs_embeds=expected_x, selected_layers=selected, alpha=alpha)
        F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten()).backward()
        F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten()).backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=4e-6, rtol=6e-5)
    _compare_gradients(model, native)


@pytest.mark.parametrize("alpha", [0.37, 1.0])
def test_oracle_matches_masked_queries_and_explicit_rotary_positions(alpha):
    native, model = _models()
    ids = torch.tensor([[0, 0, 2, 3, 5], [6, 7, 8, 0, 0]])
    valid = torch.tensor([[False, False, True, True, True], [True, True, True, False, False]])
    positions = torch.tensor([[0, 0, 7, 9, 12], [2, 3, 5, 0, 0]])
    with sdpa_kernel(SDPBackend.MATH):
        actual = model(ids, mode=RTMode((0, 1), alpha), attention_mask=valid, position_ids=positions)
        expected = olmo_recurrent_model_oracle(native, ids, selected_layers=(0, 1), alpha=alpha, attention_mask=valid, position_ids=positions)
        actual.logits[valid].square().mean().backward()
        expected.logits[valid].square().mean().backward()
    assert torch.isfinite(actual.logits).all() and torch.isfinite(expected.logits).all()
    torch.testing.assert_close(actual.logits, expected.logits, atol=4e-6, rtol=4e-6)
    _compare_gradients(model, native)


def _production_block(block, x, alpha):
    positions = torch.arange(x.shape[1]).expand(x.shape[0], -1)
    return recurrent_layer_reference(
        block, x, alpha=alpha, past=None,
        query_positions=positions, key_positions=positions,
        key_valid=torch.ones(x.shape[:2], dtype=torch.bool), attention_backend="math",
    )[0]


def test_first_token_uses_temporary_self_and_later_tokens_read_completed_output():
    native, model = _models()
    x = torch.randn(1, 2, model.config.model_dim)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ordinary_first = native.transformer.blocks[0](x[:, :1])[0]
        zero = _production_block(model.transformer.blocks[0], x, 0.0)
        recurrent = _production_block(model.transformer.blocks[0], x, 1.0)
        expected = olmo_recurrent_block_oracle(native.transformer.blocks[0], x, alpha=1.0)
    torch.testing.assert_close(recurrent[:, :1], ordinary_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(zero[:, :1], ordinary_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recurrent, expected, atol=2e-6, rtol=2e-6)
    assert (recurrent[:, 1] - zero[:, 1]).norm() > 0.01


def test_fractional_memory_returns_input_and_completed_output_adjoints():
    _, model = _models()
    block = model.transformer.blocks[0]
    x = torch.randn(1, 2, model.config.model_dim, requires_grad=True)
    normalized_inputs, completed_outputs = [], []
    input_hook = block.attn_norm.register_forward_pre_hook(lambda module, args: normalized_inputs.append(args[0]))
    output_hook = block.register_forward_hook(lambda module, args, output: completed_outputs.append(output[0]))
    try:
        _production_block(block, x, 0.37)
    finally:
        input_hook.remove()
        output_hook.remove()
    memory, completed = normalized_inputs[1], completed_outputs[0]
    probe = torch.randn_like(memory)
    to_x, to_z = torch.autograd.grad((memory * probe).sum(), (x, completed), retain_graph=True)
    block_vjp = torch.autograd.grad((completed * probe).sum(), x)[0]
    direct = torch.zeros_like(x)
    direct[:, :1] = 0.63 * probe
    torch.testing.assert_close(to_z, 0.37 * probe, atol=0, rtol=0)
    torch.testing.assert_close(to_x, direct + 0.37 * block_vjp, atol=2e-6, rtol=2e-6)


def test_fractional_history_gradient_matches_finite_difference():
    _, model = _models()
    block = model.transformer.blocks[0]
    x = torch.randn(1, 4, model.config.model_dim, requires_grad=True)
    direction = torch.zeros_like(x)
    direction[:, 0] = torch.randn_like(x[:, 0])
    direction /= direction.norm()
    probe = torch.randn_like(x[:, -1])

    def objective(value):
        return (_production_block(block, value, 0.37)[:, -1] * probe).sum()

    analytic = (torch.autograd.grad(objective(x), x)[0] * direction).sum()
    with torch.no_grad():
        epsilon = 0.004
        numerical = (objective(x + epsilon * direction) - objective(x - epsilon * direction)) / (2 * epsilon)
    assert analytic.abs() > 0.01
    torch.testing.assert_close(analytic, numerical, atol=7e-4, rtol=0.01)


def test_future_tokens_do_not_change_prefix_or_its_input_gradients():
    native, model = _models()
    x = torch.randn(1, 5, model.config.model_dim, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 3:] += 100 * torch.randn_like(changed[:, 3:])
    for evaluate in (
        lambda value: _production_block(model.transformer.blocks[0], value, 0.37),
        lambda value: olmo_recurrent_block_oracle(native.transformer.blocks[0], value, alpha=0.37),
    ):
        output = evaluate(x)
        torch.testing.assert_close(output[:, :3], evaluate(changed)[:, :3], atol=0, rtol=0)
        derivative = torch.autograd.grad(output[:, :3].square().sum(), x)[0]
        assert torch.count_nonzero(derivative[:, 3:]) == 0


def test_two_composed_modes_share_attached_gradients_like_native_oracle():
    native, model = _models()
    x = torch.randn(1, 4, model.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    targets = torch.tensor([[2, 3, 5, 7]])

    def compose(evaluate, value):
        first = evaluate(value, (0,), 0.37)
        shifted = torch.cat((torch.zeros_like(first.last_hidden_state[:, :1]), first.last_hidden_state[:, :-1]), dim=1)
        second = evaluate(value + 0.1 * shifted, (0, 1), 0.83)
        # Attached two-call gradient probe; this is not an FBT implementation.
        loss = 0.3 * first.logits.square().mean() + F.cross_entropy(second.logits.flatten(0, 1), targets.flatten())
        return second, loss

    with sdpa_kernel(SDPBackend.MATH):
        actual, loss = compose(lambda value, layers, alpha: model(inputs_embeds=value, mode=RTMode(layers, alpha)), x)
        expected, expected_loss = compose(lambda value, layers, alpha: olmo_recurrent_model_oracle(native, inputs_embeds=value, selected_layers=layers, alpha=alpha), expected_x)
        loss.backward()
        expected_loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=4e-6, rtol=6e-5)
    _compare_gradients(model, native)


def test_oracle_rejects_mutable_alpha_and_duplicate_selected_layers():
    native, _ = _models()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(TypeError, match="immutable"):
        olmo_recurrent_model_oracle(native, ids, alpha=torch.tensor(0.37))
    with pytest.raises(ValueError, match="unique"):
        olmo_recurrent_model_oracle(native, ids, selected_layers=(0, 0))
