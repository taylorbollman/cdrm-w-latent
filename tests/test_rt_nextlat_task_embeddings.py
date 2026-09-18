"""CPU semantic/gradient checks for the paired mixed-task embedding routes."""
import copy

import pytest
import torch

from cdrm import rt_nextlat_task_embeddings as routes
from cdrm.rt_nextlat_tasks import (
    build_model as build_base_model, encode_inputs, task_logits, task_loss,
)
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_embedding_head import _PermanentEmbeddingHeadProjection
from scripts.rt_a5_embedding_injection import make_projection
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock
from scripts.rt_nextlat_a5_fuzzy_train import train_step


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)
    configure_fp32_runtime()


def config(variant, **overrides):
    result = routes.read_configuration(routes.ROOT / f"configs/rt_nextlat_task_embeddings/{variant}_d128.json")
    result["backbone"]["max_sequence_length"] = 16
    result["embedding_injection"].update(overrides)
    return result


def small(variant, **overrides):
    return routes.build_model(config(variant, **overrides), backend="naive")


def fixture(task="a5"):
    local = torch.tensor([[2, 4, 8, 1, 5], [3, 7, 2, 6, 9]])
    return encode_inputs(local, task), local.flip(1)


@pytest.mark.parametrize("variant", routes.VARIANTS)
def test_exact_shared_initialization_rng_ownership_and_contract(variant):
    cfg = config(variant)
    original_cfg = copy.deepcopy(cfg)
    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop("embedding_injection")
    baseline = build_base_model(base_cfg, backend="naive")
    rng = torch.get_rng_state().clone()
    model = routes.build_model(cfg, backend="naive")
    assert torch.equal(rng, torch.get_rng_state())
    assert cfg == original_cfg
    actual, original = dict(model.named_parameters()), dict(baseline.named_parameters())
    expected_added = set() if variant == "head" else {"backbone.embedding_projection.weight"}
    assert set(actual) - set(original) == expected_added
    assert all(torch.equal(value, actual[name]) for name, value in original.items())
    assert len(list(model.named_parameters(remove_duplicate=False))) == len(actual)
    assert model.initialization["baseline_initialization"] == baseline.initialization
    assert model.initialization["shared_model_parameter_sha256"] == _parameter_sha256(baseline)
    assert model.initialization["predictor_sha256"] == baseline.initialization["predictor_sha256"]
    assert model.initialization["baseline_parameter_tensors_changed"] == []
    assert model.initialization["parameter_count"] == (479616 if variant == "head" else 496000)
    assert model.initialization["added_parameter_count"] == (0 if variant == "head" else 16384)
    assert model.initialization["model_parameter_sha256"] == _parameter_sha256(model)
    assert model.task_config == model.initialization["task_config"]
    assert model.task_config["embedding_injection"] == cfg["embedding_injection"]
    assert model.nextlat_config == baseline.nextlat_config
    blocks = model.backbone.transformer.blocks
    assert len(blocks) == 2 and type(blocks[0]) is WindowTwoRecurrentAutogradBlock
    assert blocks[0].attention_window == 2 and not hasattr(blocks[1], "attention_window")
    for block in blocks:
        assert block.pre_attention_block.kv_proj is block.kv_proj
        assert block.pre_attention_block.q_proj is block.q_proj
    if variant == "head":
        assert blocks[1].embedding_head_index == 15
    else:
        assert torch.equal(model.backbone.embedding_projection.weight, make_projection(128, 1237).weight)
        assert model.backbone.embedding_projection.bias is None
    assert "wpe" not in model.backbone.transformer


def test_input_and_value_share_isolated_projection_and_baseline_fallback():
    first, second = small("input"), small("value")
    assert _parameter_sha256(first) == _parameter_sha256(second)
    changed = small("input", projection_seed=1238)
    assert _parameter_sha256(first.backbone.transformer) == _parameter_sha256(changed.backbone.transformer)
    assert _parameter_sha256(first.predictor) == _parameter_sha256(changed.predictor)
    assert not torch.equal(first.backbone.embedding_projection.weight, changed.backbone.embedding_projection.weight)
    original = build_base_model(backend="naive")
    fallback = routes.build_model(backend="naive")
    assert fallback.initialization == original.initialization
    assert type(fallback.backbone) is type(original.backbone)


@pytest.mark.parametrize("variant", routes.VARIANTS)
@pytest.mark.parametrize("task", ["a5", "fuzzy"])
def test_disabled_route_is_exact_baseline_loss_and_shared_gradients(variant, task):
    cfg = config(variant, **({"enabled": False} if variant == "head" else {"coefficient": 0}))
    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop("embedding_injection")
    baseline = build_base_model(base_cfg, backend="naive")
    model = routes.build_model(cfg, backend="naive")
    x, y = fixture(task)
    with fp32_context("cpu"):
        reference, actual = task_loss(baseline, x, y, task), task_loss(model, x, y, task)
        for name in ("loss", "ce", "latent", "logits"):
            assert torch.equal(reference[name], actual[name]), name
        reference["loss"].backward()
        actual["loss"].backward()
    values = dict(model.named_parameters())
    for name, parameter in baseline.named_parameters():
        assert parameter.grad is not None and torch.equal(parameter.grad, values[name].grad), name
    if variant != "head":
        assert model.backbone.embedding_projection.weight.grad is None


@pytest.mark.parametrize("variant", routes.VARIANTS)
def test_attached_gradient_causality_examples_and_forward_isolation(variant):
    model = small(variant)
    x, y = fixture("fuzzy")
    captured = {}
    def observe(_block, _args, kwargs, _output):
        for key in ("value_bypass", "embedding_values"):
            if key in kwargs:
                captured["routed"] = kwargs[key]
                kwargs[key].retain_grad()
    handle = model.backbone.transformer.blocks[1].register_forward_hook(observe, with_kwargs=True)
    with fp32_context("cpu"):
        result = task_loss(model, x, y, "fuzzy")
        handle.remove()
        result["loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.count_nonzero(model.backbone.transformer.wte.weight.grad[60:]) > 0
        if variant != "input":
            # Permanent values enter only later positions, never their own
            # provisional self term. The terminal write has no consumer.
            routed_gradient = captured["routed"].grad
            assert routed_gradient is not None
            assert torch.count_nonzero(routed_gradient[:, :-1]) > 0
            assert torch.count_nonzero(routed_gradient[:, -1]) == 0
        else:
            assert torch.count_nonzero(model.backbone.embedding_projection.weight.grad) > 0
        changed = x.clone()
        changed[:, 3:] = 74
        other_example = x.clone()
        other_example[1] = 73
        with torch.no_grad():
            original = task_logits(model, x, "fuzzy")
            future_changed = task_logits(model, changed, "fuzzy")
            other_changed = task_logits(model, other_example, "fuzzy")
            repeated = task_logits(model, x, "fuzzy")
        assert torch.equal(original[:, :3], future_changed[:, :3])
        assert torch.equal(original, repeated)
        # Hold physical shape fixed so GEMM batch-shape roundoff is unrelated
        # to this semantic test for cross-example dependence.
        assert torch.equal(original[:1], other_changed[:1])
    assert not model.backbone._embedding_route_in_progress
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks


def test_input_addition_is_before_unchanged_upper_normalization():
    model = small("input")
    cfg = copy.deepcopy(model.task_config)
    cfg.pop("embedding_injection")
    baseline = build_base_model(cfg, backend="naive")
    x, _ = fixture()
    raw = baseline.backbone.transformer.wte(x)
    bypass = .01 * torch.nn.functional.linear(raw, model.backbone.embedding_projection.weight)
    hook = baseline.backbone.transformer.blocks[1].register_forward_pre_hook(
        lambda _m, args: (args[0] + bypass, *args[1:]))
    with fp32_context("cpu"):
        expected = baseline.backbone(x, return_pre_logits=True)
        actual = model.backbone(x, return_pre_logits=True)
    hook.remove()
    assert torch.equal(actual.logits, expected.logits)
    assert torch.equal(actual.pre_logits, expected.pre_logits)


@pytest.mark.parametrize("variant", ["value", "head"])
def test_first_self_position_is_unchanged_but_historical_outputs_change(variant):
    model = small(variant)
    cfg = copy.deepcopy(model.task_config)
    cfg.pop("embedding_injection")
    baseline = build_base_model(cfg, backend="naive")
    x, _ = fixture()
    with torch.no_grad(), fp32_context("cpu"):
        actual, expected = model.backbone(x).logits, baseline.backbone(x).logits
    assert torch.equal(actual[:, :1], expected[:, :1])
    assert not torch.equal(actual[:, 1:], expected[:, 1:])


def test_last_head_replaces_only_its_permanent_values_and_retains_contextual_keys():
    model = small("head")
    block = model.backbone.transformer.blocks[1]
    x, _ = fixture()
    raw = model.backbone.transformer.wte(x)
    values = torch.nn.functional.linear(raw, block.kv_proj.weight[248:256])
    projection = _PermanentEmbeddingHeadProjection(block, values)
    generator = torch.Generator().manual_seed(10)
    contextual = torch.randn(2, 5, 128, generator=generator)
    before = block.pre_attention_block(contextual)
    for position in range(5):
        state = contextual[:, position:position + 1]
        ordinary = block.kv_proj(state)
        actual = projection(state)
        assert torch.equal(actual[..., :128], ordinary[..., :128])
        assert torch.equal(actual[..., 128:248], ordinary[..., 128:248])
        assert torch.equal(actual[..., 248:256], values[:, position:position + 1])
    projection.finish()
    assert all(torch.equal(a, b) for a, b in zip(before, block.pre_attention_block(contextual)))


@pytest.mark.parametrize("variant", routes.VARIANTS)
def test_real_mixed_step_optimizer_coverage_and_reloaded_next_update(variant):
    model, reloaded = small(variant), small(variant)
    optimizer, second_optimizer = make_optimizer(model), make_optimizer(reloaded)
    batches = {task: fixture(task) for task in ("a5", "fuzzy")}
    train_step(model, optimizer, batches, mode="mixed", microbatch=2)
    first_state = copy.deepcopy(model.state_dict())
    first_optimizer = copy.deepcopy(optimizer.state_dict())
    assert len(optimizer.state) == len(list(model.parameters()))
    if variant != "head":
        projection = model.backbone.embedding_projection.weight
        assert optimizer.state[projection]["step"].item() == 1
        assert torch.count_nonzero(projection.grad) > 0
        assert any(group["weight_decay"] == .01 and any(p is projection for p in group["params"])
                   for group in optimizer.param_groups)
    train_step(model, optimizer, batches, mode="mixed", microbatch=2)
    reloaded.load_state_dict(first_state, strict=True)
    second_optimizer.load_state_dict(first_optimizer)
    train_step(reloaded, second_optimizer, batches, mode="mixed", microbatch=2)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), name
    for left, right in zip(optimizer.state.values(), second_optimizer.state.values()):
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize("change", [
    {"layer_index": 0}, {"coefficient": float("nan")}, {"coefficient": True},
    {"projection_seed": -1}, {"head_index": 0}, {"schedule": "linear"},
])
def test_invalid_or_unintended_routes_fail_closed(change):
    with pytest.raises(ValueError):
        routes.read_configuration(config("input", **change))


def test_source_closure_includes_every_new_adapter_and_config():
    manifest = routes.source_manifest()
    assert set(manifest) == set(routes.SOURCE_PATHS)
    assert all(len(digest) == 64 for digest in manifest.values())
    assert "scripts/rt_a5_value_bypass.py" in manifest
    assert "scripts/rt_a5_embedding_head.py" in manifest


def test_temporary_hook_is_removed_after_failure_and_model_reusable():
    model = small("input")
    x, _ = fixture()
    def fail(_module, _args):
        raise RuntimeError("intentional diagnostic interruption")
    handle = model.backbone.transformer.blocks[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="intentional"):
        model.backbone(x)
    handle.remove()
    assert not model.backbone._embedding_route_in_progress
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks
    with torch.no_grad(), fp32_context("cpu"):
        assert torch.isfinite(model.backbone(x).logits).all()
