"""CPU tests for multi-layer dispatch, bounded scope and exact resource metadata."""
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import olmo_tiled, olmo_rt_kernels, olmo_rt_backward_kernels, olmo_rt_recompute_kernels
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_f3e_validate as validation


@pytest.mark.parametrize("layout,layers,role", [
    ("single", (0,), "single_layer_reference"),
    ("adjacent2", (0, 1), "primary_multi_layer_integration"),
    ("spread2", (0, 15), "primary_multi_layer_integration"),
    ("spread4", (0, 5, 10, 15), "primary_multi_layer_integration"),
    ("all16", tuple(range(16)), "optional_all_layer_stress"),
])
def test_layouts_are_explicit_and_do_not_assume_all_layers_for_main_experiment(layout, layers, role):
    case = validation.selected_case("combined", layout, batch=4, length=128)
    assert case.rt_layers == layers
    assert case.mode().rt_mode.selected_layers == layers
    assert case.batch_size == 4 and case.length == 128
    assert case.fbt and case.nextlat and case.mode().num_passes == 2
    assert validation.layout_role(layout) == role


@pytest.mark.parametrize("case_name,passes,nextlat,fbt", [
    ("rt", 1, False, False), ("combined", 2, True, True), ("combined-k3", 3, True, True),
])
def test_case_layout_override_preserves_architecture_objectives_and_passes(case_name, passes, nextlat, fbt):
    case = validation.selected_case(case_name, "spread2", batch=1, length=2048)
    assert case.mode().num_passes == passes
    assert case.nextlat == nextlat and case.fbt == fbt
    assert case.alpha == 1.0 and case.beta == 1.0


@pytest.mark.parametrize("case,layout", [("ordinary", "spread2"), ("rt", "unknown")])
def test_unknown_case_or_layout_rejected(case, layout):
    with pytest.raises(ValueError):
        validation.selected_case(case, layout, batch=1, length=32)


@pytest.mark.parametrize("length,passes,layers,variant,tiles,fused", [
    (32, 1, (0, 15), "recompute", 62, 0),
    (32, 3, (0, 15), "recompute", 124, 0),
    (512, 2, (0, 1), "reference", 1022, 1022),
    (1024, 2, (0, 15), "reference", 2046, 2044),
    (2048, 2, (0, 15), "reference", 4094, 4088),
    (2048, 2, (0, 15), "recompute", 4094, 0),
])
def test_expected_dispatch_covers_passes_layers_and_old_large_rectangle_fallback(
        length, passes, layers, variant, tiles, fused):
    mode = FBTMode(enabled=passes > 1, num_passes=passes, rt_mode=RTMode(layers))
    observed = validation.expected_dispatch(length, mode, variant)
    key = "recompute_tiles" if variant == "recompute" else "materialized_tiles"
    assert observed["totals"][key] == tiles
    assert observed["totals"]["materialized_fused_tiles"] == fused
    assert observed["totals"]["forward_blocks"] == len(layers) * (passes - 1 if passes > 1 else 1)
    assert observed["totals"]["forward_blocks"] == observed["totals"]["backward_blocks"]
    assert set(observed["by_layer"]) == {str(index) for index in layers}
    for event in validation.EVENTS:
        assert sum(row[event] for row in observed["by_layer"].values()) == observed["totals"][event]


def fake_model():
    weights = [object() for _ in range(16)]
    blocks = [SimpleNamespace(att_proj=SimpleNamespace(weight=weight)) for weight in weights]
    return SimpleNamespace(backbone=SimpleNamespace(backbone=SimpleNamespace(layers=blocks))), weights


def install_cpu_dispatch(monkeypatch):
    def kernel(*args, **kwargs):
        return args
    monkeypatch.setattr(olmo_rt_recompute_kernels, "backward_recomputed_tile", kernel)
    monkeypatch.setattr(olmo_rt_backward_kernels, "backward_tile", kernel)
    monkeypatch.setattr(olmo_rt_kernels, "add_tile", kernel)

    def forward_tile(*args, **kwargs):
        return olmo_rt_kernels.add_tile(*args, **kwargs)
    monkeypatch.setattr(olmo_tiled, "_add_tile", forward_tile)

    def historical(*args, **kwargs):
        return olmo_rt_backward_kernels.backward_tile(*args, **kwargs)
    monkeypatch.setattr(olmo_tiled, "_historical_backward_tile", historical)

    def forward(ctx, x, weight, memory):
        ctx.memory = memory
        olmo_tiled._add_tile(x)
        return x

    def backward(ctx, grad):
        if ctx.memory == "recompute":
            olmo_rt_recompute_kernels.backward_recomputed_tile(grad)
        else:
            olmo_tiled._historical_backward_tile(grad)
        return grad
    monkeypatch.setattr(olmo_tiled._TiledRecurrence, "forward", staticmethod(forward))
    monkeypatch.setattr(olmo_tiled._TiledRecurrence, "backward", staticmethod(backward))
    return forward, backward, historical, kernel


def test_dispatch_tracks_context_identity_across_multiple_passes_and_reverse_order(monkeypatch):
    model, weights = fake_model()
    originals = install_cpu_dispatch(monkeypatch)
    contexts = []
    with validation.count_layer_dispatch(model, (0, 15)) as observed:
        for _ in range(2):
            for index in (0, 15):
                ctx = SimpleNamespace()
                assert olmo_tiled._TiledRecurrence.forward(ctx, "x", weights[index], "recompute") == "x"
                contexts.append(ctx)
        for ctx in reversed(contexts):
            assert olmo_tiled._TiledRecurrence.backward(ctx, "g") == "g"
    assert observed == validation.expected_dispatch(2, FBTMode(enabled=True, num_passes=3,
                                               rt_mode=RTMode((0, 15))), "recompute")
    assert olmo_tiled._TiledRecurrence.forward is originals[0]
    assert olmo_tiled._TiledRecurrence.backward is originals[1]
    assert olmo_tiled._historical_backward_tile is originals[2]
    assert olmo_rt_backward_kernels.backward_tile is originals[3]
    assert olmo_rt_recompute_kernels.backward_recomputed_tile is originals[3]


def test_dispatch_tracks_materialized_helper_and_actual_fused_kernel_separately(monkeypatch):
    model, weights = fake_model()
    install_cpu_dispatch(monkeypatch)
    ctx = SimpleNamespace()
    with validation.count_layer_dispatch(model, (15,)) as observed:
        olmo_tiled._TiledRecurrence.forward(ctx, "x", weights[15], "materialized")
        olmo_tiled._TiledRecurrence.backward(ctx, "g")
    assert observed == validation.expected_dispatch(2, FBTMode(rt_mode=RTMode((15,))), "reference")


def test_dispatch_rejects_unselected_layers_and_restores_after_failure(monkeypatch):
    model, weights = fake_model()
    originals = install_cpu_dispatch(monkeypatch)
    with pytest.raises(AssertionError, match="Unexpected RT layer"):
        with validation.count_layer_dispatch(model, (0,)):
            olmo_tiled._TiledRecurrence.forward(SimpleNamespace(), "x", weights[15], "recompute")
    assert olmo_tiled._TiledRecurrence.forward is originals[0]
    assert olmo_tiled._TiledRecurrence.backward is originals[1]
    assert olmo_rt_recompute_kernels.backward_recomputed_tile is originals[3]


def test_dispatch_rejects_kernel_without_observed_backward(monkeypatch):
    model, _ = fake_model()
    install_cpu_dispatch(monkeypatch)
    with pytest.raises(AssertionError, match="outside an observed RT block"):
        with validation.count_layer_dispatch(model, (0,)):
            olmo_rt_recompute_kernels.backward_recomputed_tile("g")


def test_actual_cpu_autograd_context_preserves_dispatch_identity_and_exact_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(771)
    base = olmo_tiled.OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", backward_memory="materialized")
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=base))
    ids = torch.tensor([[2, 3, 5, 7]])
    mode = RTMode((0, 1))
    expected = base(ids, mode=mode).last_hidden_state
    expected.square().sum().backward()
    gradients = {name: value.grad.clone() for name, value in base.named_parameters() if value.grad is not None}
    base.zero_grad(set_to_none=True)
    with validation.count_layer_dispatch(model, mode.selected_layers) as observed:
        actual = base(ids, mode=mode).last_hidden_state
        actual.square().sum().backward()
    assert torch.equal(actual, expected)
    assert all(torch.equal(parameter.grad, gradients[name]) for name, parameter in base.named_parameters()
               if parameter.grad is not None)
    for index in ("0", "1"):
        assert observed["by_layer"][index] == {"forward_blocks": 1, "forward_tiles": 3,
            "forward_fused_tiles": 0, "forward_eager_tiles": 3, "backward_blocks": 1,
            "recompute_tiles": 0, "materialized_tiles": 3, "materialized_fused_tiles": 0}


@pytest.mark.parametrize("batch", [0, -1, 129])
def test_parser_rejects_unbounded_batches(batch):
    with pytest.raises(SystemExit):
        validation.parse_args(["--batch-size", str(batch), "--output-dir", "unused"])


def test_parser_prevents_expanding_k3_into_capacity_sweep():
    with pytest.raises(SystemExit):
        validation.parse_args(["--case", "combined-k3", "--stage", "capacity", "--output-dir", "unused"])


def test_parser_defaults_to_primary_two_layer_candidate():
    args = validation.parse_args(["--output-dir", "unused"])
    assert args.layout == "adjacent2" and args.variant == "recompute"
    assert args.stage == "correctness" and args.case == "combined"


@pytest.mark.parametrize("reference,candidate,expected", [
    ([0.0, 0.0], [0.0, 0.0], 0.0),
    ([0.0, 0.0], [1.0, 0.0], float("inf")),
    ([1.0, 0.0], [1.25, 0.0], .25),
])
def test_coordinate_budget_has_no_zero_reference_escape(reference, candidate, expected):
    result = validation.comparison_metrics(torch.tensor(candidate), torch.tensor(reference))
    assert result["relative_l2"] == expected
    assert result["max_relative"] == expected


def test_resource_card_uses_actual_mask_union_and_explicit_execution_ledger(monkeypatch):
    case = validation.selected_case("combined-k3", "spread2", batch=2, length=32)
    native = OLMoConfig.native_1b()
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=SimpleNamespace(
        config=native, backward_memory="recompute")), config=NextLatConfig(model_dim=2048),
        named_parameters=lambda: [("declared_execution", torch.empty(3, 4))])
    plan = SimpleNamespace(model=model, mode=case.mode(), counts={"ce": 30, "latent": 50, "kl": 20},
        loss_layout=SimpleNamespace(needed_source_indices=torch.arange(55)),
        weights={"ce": 1.0, "latent": .1, "kl": .1}, active_names={"observed_gradient"})
    monkeypatch.setattr(validation, "active_names", lambda model, mode: {"declared_execution"})
    monkeypatch.setattr(validation, "inference_names", lambda model, case: {"declared_inference"})
    inventory = []
    def record_inventory(model, **kwargs):
        inventory.append(kwargs)
        return {"registered_unique": 123}
    monkeypatch.setattr(validation, "parameter_inventory", record_inventory)
    result = validation.resource_card(plan, case, optimizer="optimizer")
    assert result["loss_work"] == {"ce_targets": 30, "latent_pairs": 50,
                                   "kl_triples": 20, "predictor_positions": 55}
    assert result["analytic_matrix_work"]["backward_memory"] == "recompute"
    assert result["analytic_matrix_work"]["rt_block_calls_per_microbatch"] == 4
    assert result["analytic_matrix_work"]["objective_positions_across_passes"]["ce"] == 90
    assert inventory == [{"optimizer": "optimizer", "executed_names": {"declared_execution"},
                          "inference_names": {"declared_inference"}}]


def test_sources_cover_transitive_runtime_and_validator_dependencies():
    expected = {"scripts/olmo_f3e_validate.py", "scripts/olmo_f3d_validate.py",
                "scripts/olmo_f3_graph_training.py", "cdrm/pretrained/resource_estimates.py",
                "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_rt_memory.py"}
    assert expected <= set(validation.SOURCES)
    assert all((validation.ROOT / source).is_file() for source in validation.SOURCES)
