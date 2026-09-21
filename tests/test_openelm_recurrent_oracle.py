"""Independent semantic checks for the native OpenELM RT reference milestone."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.openelm import OpenELMConfig
from cdrm.pretrained.recurrent import OpenELMRecurrentModel, RTMode, recurrent_layer_reference
from cdrm.pretrained.recurrent_oracle import recurrent_block_oracle, recurrent_model_oracle
from cdrm.pretrained.reference import build_corenet_reference, reference_forward


def _models():
    torch.manual_seed(2301)
    # Actual 4:1 grouping and nonuniform attention/FFN widths. Bottom attention
    # width is smaller than residual width, as in the selected 1.1B checkpoint.
    config = OpenELMConfig(
        vocab_size=43, padding_idx=40, model_dim=64, head_dim=8,
        num_query_heads=(4, 8), num_kv_heads=(1, 2),
        ffn_intermediate_sizes=(48, 112), max_context_length=32,
    )
    native = build_corenet_reference(config)
    with torch.no_grad():
        for block in native.layers:
            # Unit gains would hide some normalization/RoPE order mistakes.
            block.attn.q_norm.weight.copy_(torch.linspace(0.5, 1.5, config.head_dim))
            block.attn.k_norm.weight.copy_(torch.linspace(1.8, 0.4, config.head_dim))
    model = OpenELMRecurrentModel(config, attention_backend="math")
    model.load_state_dict(native.state_dict(), strict=True)
    return native, model


def _compare_gradients(actual, expected, *, atol=5e-6, rtol=5e-5):
    actual_parameters = dict(actual.named_parameters())
    for name, parameter in expected.named_parameters():
        assert parameter.grad is not None, name
        assert actual_parameters[name].grad is not None, name
        torch.testing.assert_close(actual_parameters[name].grad, parameter.grad, atol=atol, rtol=rtol, msg=name)


@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_alpha_zero_oracle_recovers_native_ordinary_outputs_and_all_gradients(selected):
    oracle, _ = _models()
    ordinary = copy.deepcopy(oracle)
    ids = torch.tensor([[1, 2, 3, 5, 8], [9, 7, 6, 4, 2]])
    targets = ids.flip(-1)
    with sdpa_kernel(SDPBackend.MATH):
        expected = reference_forward(ordinary, ids)
        actual = recurrent_model_oracle(oracle, ids, selected_layers=selected, alpha=0.0)
        F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten()).backward()
        F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten()).backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=3e-6)
    _compare_gradients(oracle, ordinary)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_scan_matches_independent_oracle_outputs_parameter_and_input_gradients(alpha, selected):
    native, model = _models()
    torch.manual_seed(2302)
    x = torch.randn(2, 5, model.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    targets = torch.randint(0, model.config.vocab_size, (2, 5))
    with sdpa_kernel(SDPBackend.MATH):
        actual = model(inputs_embeds=x, mode=RTMode(selected, alpha))
        expected = recurrent_model_oracle(native, inputs_embeds=expected_x, selected_layers=selected, alpha=alpha)
        F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten()).backward()
        F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten()).backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=5e-6, rtol=5e-5)
    _compare_gradients(model, native)


@pytest.mark.parametrize("alpha", [0.37, 1.0])
def test_oracle_matches_scan_for_masked_queries_and_distinct_absolute_positions(alpha):
    native, model = _models()
    ids = torch.tensor([[0, 0, 2, 3, 5], [6, 7, 8, 0, 0]])
    valid = torch.tensor([[False, False, True, True, True], [True, True, True, False, False]])
    positions = torch.tensor([[0, 0, 7, 9, 12], [2, 3, 5, 0, 0]])
    with sdpa_kernel(SDPBackend.MATH):
        actual = model(ids, mode=RTMode((0, 1), alpha), attention_mask=valid, position_ids=positions)
        expected = recurrent_model_oracle(native, ids, selected_layers=(0, 1), alpha=alpha, attention_mask=valid, position_ids=positions)
        # Padding query outputs are not supervised. The zero-key rows still
        # exercise the finite fully-masked attention convention.
        actual.logits[valid].square().mean().backward()
        expected.logits[valid].square().mean().backward()
    assert torch.isfinite(actual.logits).all()
    assert torch.isfinite(expected.logits).all()
    torch.testing.assert_close(actual.logits, expected.logits, atol=4e-6, rtol=4e-6)
    _compare_gradients(model, native, atol=8e-6)


def _production_block(block, x, alpha):
    positions = torch.arange(x.shape[1]).expand(x.shape[0], -1)
    return recurrent_layer_reference(
        block, x, alpha=alpha, past=None,
        query_positions=positions, key_positions=positions,
        key_valid=torch.ones(x.shape[:2], dtype=torch.bool), attention_backend="math",
    )[0]


def test_first_token_uses_temporary_self_then_publishes_complete_block_memory():
    native, model = _models()
    x = torch.randn(1, 2, model.config.model_dim)
    block, expected_block = model.layers[0], native.layers[0]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ordinary_first = expected_block(x[:, :1])[0]
        zero = _production_block(block, x, 0.0)
        recurrent = _production_block(block, x, 1.0)
        expected = recurrent_block_oracle(expected_block, x, alpha=1.0)
    torch.testing.assert_close(recurrent[:, :1], ordinary_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(zero[:, :1], ordinary_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recurrent, expected, atol=2e-6, rtol=2e-6)
    # At token two the completed first block output changes historical memory.
    assert (recurrent[:, 1] - zero[:, 1]).norm() > 0.01


def test_fractional_recurrence_gradient_agrees_with_directional_finite_difference():
    _, model = _models()
    block = model.layers[0]
    torch.manual_seed(2303)
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
    # The final-token objective sees this direction only through historical
    # memory. A detached memory-source or completed-output branch changes this
    # derivative while leaving its forward values unchanged.
    assert analytic.abs() > 0.01
    torch.testing.assert_close(analytic, numerical, atol=7e-4, rtol=0.01)


def test_fractional_memory_source_returns_both_input_and_completed_output_adjoints():
    _, model = _models()
    block = model.layers[0]
    x = torch.randn(1, 2, model.config.model_dim, requires_grad=True)
    normalized_inputs, completed_outputs = [], []
    input_hook = block.attn_norm.register_forward_pre_hook(
        lambda module, inputs: normalized_inputs.append(inputs[0])
    )
    output_hook = block.register_forward_hook(
        lambda module, inputs, output: completed_outputs.append(output[0])
    )
    try:
        _production_block(block, x, 0.37)
    finally:
        input_hook.remove()
        output_hook.remove()
    # In a two-token scan the first norm reads the temporary input and the
    # second reads the first permanent memory source. Observing those values
    # checks the source *before* native normalization, not an inferred cache.
    memory_source, completed = normalized_inputs[1], completed_outputs[0]
    probe = torch.randn_like(memory_source)
    source_to_x, source_to_z = torch.autograd.grad(
        (memory_source * probe).sum(), (x, completed), retain_graph=True,
    )
    output_to_x = torch.autograd.grad((completed * probe).sum(), x)[0]
    direct_input = torch.zeros_like(x)
    direct_input[:, :1] = 0.63 * probe
    torch.testing.assert_close(source_to_z, 0.37 * probe, atol=0, rtol=0)
    torch.testing.assert_close(source_to_x, direct_input + 0.37 * output_to_x, atol=2e-6, rtol=2e-6)


def test_future_tokens_are_isolated_in_oracle_and_scan_outputs_and_input_gradients():
    native, model = _models()
    x = torch.randn(1, 5, model.config.model_dim, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 3:] += 100.0 * torch.randn_like(changed[:, 3:])
    for evaluate in (
        lambda value: _production_block(model.layers[0], value, 0.37),
        lambda value: recurrent_block_oracle(native.layers[0], value, alpha=0.37),
    ):
        original_output = evaluate(x)
        changed_output = evaluate(changed)
        torch.testing.assert_close(original_output[:, :3], changed_output[:, :3], atol=0, rtol=0)
        gradient = torch.autograd.grad(original_output[:, :3].square().sum(), x)[0]
        assert torch.count_nonzero(gradient[:, 3:]) == 0


def test_oracle_rejects_mutable_alpha_and_duplicate_layers():
    native, _ = _models()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(TypeError, match="immutable"):
        recurrent_model_oracle(native, ids, alpha=torch.tensor(0.37))
    with pytest.raises(ValueError, match="unique"):
        recurrent_model_oracle(native, ids, selected_layers=(0, 0))


def test_composed_forward_modes_share_attached_gradients_like_independent_oracle():
    native, model = _models()
    torch.manual_seed(2304)
    x = torch.randn(1, 4, model.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    targets = torch.tensor([[2, 3, 5, 7]])

    def compose(evaluate, value):
        first = evaluate(value, (0,), 0.37)
        # A small attached causal state path exercises shared weights and
        # saved per-call modes before backward. This is a gradient-ownership
        # probe, not an implementation or validation of the planned FBT gate.
        shifted = torch.cat((torch.zeros_like(first.last_hidden_state[:, :1]), first.last_hidden_state[:, :-1]), dim=1)
        second = evaluate(value + 0.1 * shifted, (0, 1), 0.83)
        loss = 0.3 * first.logits.square().mean() + F.cross_entropy(second.logits.flatten(0, 1), targets.flatten())
        return first, second, loss

    with sdpa_kernel(SDPBackend.MATH):
        first, second, loss = compose(
            lambda value, selected, alpha: model(inputs_embeds=value, mode=RTMode(selected, alpha)), x,
        )
        expected_first, expected_second, expected_loss = compose(
            lambda value, selected, alpha: recurrent_model_oracle(native, inputs_embeds=value, selected_layers=selected, alpha=alpha), expected_x,
        )
        loss.backward()
        expected_loss.backward()
    torch.testing.assert_close(first.logits, expected_first.logits, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(second.logits, expected_second.logits, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=5e-6, rtol=5e-5)
    _compare_gradients(model, native, atol=8e-6, rtol=8e-5)
