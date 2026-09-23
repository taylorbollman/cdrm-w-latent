"""Bounded block-stack experiment fixtures, ownership, screens and static guards."""
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from scripts import olmo_rt_author_compare as harness


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(stage="verify", *, backend="author", batch=1, length=32):
    return ["--stage", stage, "--backend", backend, "--batch-size", str(batch),
            "--length", str(length), "--output-dir", "/tmp/unused-author-comparison"]


def test_cli_primary_policy_and_shared_event_boundary_are_explicit():
    args = harness.parse_args(arguments("capacity", batch=128, length=512))
    assert args.native_arm == "both" and args.native_backward == "recompute"
    assert args.author_precision == "author_legacy" and args.bwd_mlp_chunks == 4
    assert args.precision == "bf16_mixed" and args.region_events
    assert not args.profile


@pytest.mark.parametrize("values", [
    arguments(batch=32), arguments(batch=1, length=512),
    arguments(batch=8, length=512) + ["--precision", "fp32"],
    arguments() + ["--profile"], arguments() + ["--bwd-mlp-chunks", "3"],
    arguments("capacity", batch=8, length=512), arguments("capacity", batch=32, length=32),
    arguments("capacity", batch=32, length=512) + ["--precision", "fp32"],
    arguments("capacity", batch=128, length=512) + ["--profile"],
    arguments("capacity", batch=32, length=512) + ["--layers", "6", "--profile"],
])
def test_unplanned_expensive_or_misleading_modes_are_rejected(values):
    with pytest.raises(SystemExit):
        harness.parse_args(values)


def test_fixtures_and_raw_cotangents_are_reproducible_without_global_rng_changes():
    before = torch.random.get_rng_state().clone()
    first = harness.make_fixture(2, 7, 32, seed=123)
    repeat = harness.make_fixture(2, 7, 32, seed=123)
    changed = harness.make_fixture(2, 7, 32, seed=124)
    probe = harness.make_cotangent(first.inputs.shape, seed=125)
    generator = torch.Generator().manual_seed(125)
    expected = torch.randn(first.inputs.shape, generator=generator) / first.inputs.numel() ** .5
    assert torch.equal(first.inputs, repeat.inputs) and torch.equal(first.targets, repeat.targets)
    assert not torch.equal(first.inputs, first.targets)
    assert not torch.equal(first.inputs, changed.inputs)
    assert torch.equal(probe, expected)
    assert torch.equal(before, torch.random.get_rng_state())


def tiny_stack(layers=2):
    config = replace(OLMoConfig.tiny(), num_layers=layers)
    return harness.BlockStack(config).train()


def execution(backend="native"):
    return harness.Execution(backend=backend, precision="fp32", compiled_helpers=False, native_tiles="eager")


def test_checkpoint_subset_preserves_packed_native_weights_without_embedding_or_readout():
    original = OLMoForCausalLM(replace(OLMoConfig.tiny(), num_layers=6))
    state = original.state_dict()
    assert "transformer.blocks.0.att_proj.weight" in state
    assert "transformer.wte.weight" in state
    candidate = harness.build_stack(state, replace(original.config, num_layers=2), device="cpu")
    expected_names = {"layers." + name.removeprefix("transformer.blocks.") for name in state
                      if name.startswith(("transformer.blocks.0.", "transformer.blocks.1."))}
    assert set(candidate.state_dict()) == expected_names
    assert len(list(candidate.parameters())) == 8
    for name, parameter in candidate.named_parameters():
        native_name = "transformer.blocks." + name.removeprefix("layers.")
        assert torch.equal(parameter, state[native_name])
        assert parameter.data_ptr() != state[native_name].data_ptr()
    with torch.no_grad():
        candidate.layers[0].att_proj.weight.add_(1)
    assert not torch.equal(candidate.layers[0].att_proj.weight, state["transformer.blocks.0.att_proj.weight"])


@pytest.mark.parametrize("backend", ["native", "author", "author_scan"])
def test_common_stack_raw_output_input_and_packed_gradients_match_native_scan(backend):
    model = tiny_stack()
    fixture = harness.make_fixture(1, 5, model.config.model_dim, seed=13)
    probe = harness.make_cotangent(fixture.inputs.shape, seed=14)
    reference = harness.raw_snapshot(model, fixture, probe, execution("native_scan"))
    candidate = harness.raw_snapshot(model, fixture, probe, execution(backend))
    result = harness.compare_snapshots(candidate, reference, name="test", policy="fp32")
    assert result["passed"], result
    assert result["ownership_matches"]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("backend", ["native", "author"])
def test_prepared_eager_training_preserves_grad_storage_and_overwrites_changed_input(backend):
    model = tiny_stack(1)
    first = harness.make_fixture(1, 3, model.config.model_dim, seed=71)
    second = harness.make_fixture(1, 3, model.config.model_dim, seed=72)
    plan = harness.PreparedBlockTraining(model, first, execution(backend))
    original = harness.plan_snapshot(plan, plan.backward())
    addresses = dict(plan.gradient_addresses)
    plan.load_fixture(second)
    changed = harness.plan_snapshot(plan, plan.backward())
    repeat = harness.plan_snapshot(plan, plan.backward())
    assert addresses == plan.grad_addresses()
    assert not torch.equal(original["output"], changed["output"])
    assert harness.compare_snapshots(changed, repeat, name="repeat", policy="exact")["passed"]


@pytest.mark.parametrize("damage", ["rope_mutation", "rope_replacement", "positions", "parameters", "gradients", "mode"])
def test_static_guard_rejects_changed_capture_contract(damage):
    model = tiny_stack(1)
    fixture = harness.make_fixture(1, 3, model.config.model_dim)
    plan = harness.PreparedBlockTraining(model, fixture, execution())
    plan.backward()
    if damage == "rope_mutation":
        plan.tables.cos.add_(1)
    elif damage == "rope_replacement":
        plan.tables = type(plan.tables)(plan.tables.cos.clone(), plan.tables.sin)
    elif damage == "positions":
        plan.positions.add_(1)
    elif damage == "parameters":
        model.layers[0].att_proj.weight = torch.nn.Parameter(model.layers[0].att_proj.weight.detach().clone())
    elif damage == "gradients":
        model.layers[0].att_proj.weight.grad = model.layers[0].att_proj.weight.grad.clone()
    elif damage == "mode":
        model.layers[0].eval()
    with pytest.raises(ValueError):
        plan.validate()


def test_bf16_budget_does_not_silently_relax_known_coordinate_miss():
    reference = {"output": torch.ones(4), "input_gradient": torch.ones(4), "mse": torch.tensor(1.0),
        "parameters": {"weight": torch.ones(4096)}, "expected_parameters": ["weight"], "execution": {}}
    candidate = {**reference, "parameters": {"weight": reference["parameters"]["weight"].clone()}}
    candidate["parameters"]["weight"][0] += .06349206349
    result = harness.compare_snapshots(candidate, reference, name="coordinate", policy="bf16")
    assert result["global_parameter_relative_l2"] < 1 / 64
    assert not result["passed"]
    diagnostic = harness.compare_snapshots(candidate, reference, name="context", policy="diagnostic", gate=False)
    assert diagnostic["passed"] and not diagnostic["gate"]


def test_incomplete_parameter_ownership_is_not_a_passing_diagnostic():
    reference = {"output": torch.ones(4), "input_gradient": torch.ones(4), "mse": torch.tensor(1.0),
        "parameters": {"weight": torch.ones(4)}, "expected_parameters": ["weight"], "execution": {}}
    candidate = {**reference, "parameters": {}}
    assert not harness.compare_snapshots(candidate, reference, name="ownership", policy="diagnostic")["passed"]


def test_source_snapshot_includes_new_backend_and_event_preflight():
    assert {"cdrm/pretrained/olmo_author.py", "scripts/olmo_rt_author_event_probe.py",
            "scripts/olmo_rt_author_compare.py"} <= set(harness.SOURCES)
