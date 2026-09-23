"""Independent feature ownership and actual tiny update checks for F4 cards."""

from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts import olmo_f4_resources as validation
from scripts.olmo_f3e_validate import count_layer_dispatch, expected_dispatch


# Independent expected behavior: FBT and NextLat must never be inferred from
# one another merely because a case is not named "rt".
CASES = [
    ("ordinary", False, False, False),
    ("rt", True, False, False),
    ("fbt", False, True, False),
    ("nextlat", False, False, True),
    ("rt-fbt", True, True, False),
    ("rt-nextlat", True, False, True),
    ("fbt-nextlat", False, True, True),
    ("combined", True, True, True),
]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(9431)


def _batch():
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19], [23, 29, 31, 37, 41, 43, 47, 53]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(2)[:, None].expand_as(ids).clone()
    response = torch.zeros_like(valid)
    response[:, 4:] = True
    return NextLatBatch(ids, valid, docs, response, valid.clone(), response.clone())


@pytest.mark.parametrize("name,rt,fbt,nextlat", CASES)
def test_feature_flags_have_independent_pass_and_layer_semantics(name, rt, fbt, nextlat):
    case = validation.selected_case(name, batch=64, length=512)
    mode = case.mode()
    assert case.fbt is fbt and case.nextlat is nextlat
    assert case.rt_layers == ((0, 15) if rt else ())
    assert mode.enabled is fbt
    assert mode.num_passes == (2 if fbt else 1)
    assert mode.rt_mode.alpha == 1.0
    assert mode.beta == 1.0
    dispatch = expected_dispatch(512, mode, "recompute")
    assert dispatch["totals"]["forward_blocks"] == (2 if rt else 0)
    assert dispatch["totals"]["backward_blocks"] == (2 if rt else 0)
    assert dispatch["totals"]["recompute_tiles"] == (1022 if rt else 0)
    if not rt:
        assert dispatch["by_layer"] == {}
        assert set(dispatch["totals"].values()) == {0}


@pytest.mark.parametrize("name,rt,fbt,nextlat", CASES)
def test_actual_tiny_update_resource_card_preserves_independent_ownership(name, rt, fbt, nextlat):
    # Retain 16 tiny layers so the exact spread2 indices are executed, rather
    # than replacing the native selection with a synthetic adjacent shortcut.
    config = replace(OLMoConfig.tiny(), num_layers=16)
    source = OLMoTiledRTForCausalLM(config, attention_backend="math")
    case = validation.selected_case(name, batch=2, length=8)
    model = validation.build_model(source.state_dict(), case, device="cpu",
        model_config=config, backend="math", chunk_size=4)
    base = model.backbone.backbone
    base.ordinary_activation_checkpointing = True
    base.backward_memory = "recompute"
    base.cast_weights_once = True
    batch = _batch()
    plan = StaticFBTTraining(model, batch, mode=case.mode(), config=LMTrainingConfig(precision="fp32"))
    plan.initialize_gradients()
    with count_layer_dispatch(model, case.rt_layers) as observed:
        result = plan.backward(replay=False)
    assert len(result.pass_losses) == (2 if fbt else 1)
    assert result.pass_coefficients == ((1.0, 1.0) if fbt else (1.0,))
    assert result.counts == {"ce": 8, "latent": 14 if nextlat else 0, "kl": 8 if nextlat else 0}
    assert result.weights == {"ce": 1., "latent": float(nextlat), "kl": float(nextlat)}
    assert observed["totals"]["forward_blocks"] == observed["totals"]["backward_blocks"] == (2 if rt else 0)
    assert set(observed["by_layer"]) == ({"0", "15"} if rt else set())
    for row in observed["by_layer"].values():
        assert row["forward_blocks"] == row["backward_blocks"] == 1
        assert row["forward_tiles"] == row["forward_eager_tiles"] == 7
        assert row["forward_fused_tiles"] == row["recompute_tiles"] == 0  # Explicit CPU semantic scope.

    optimizer, scheduler = validation.build_optimizer(model)
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    metrics = plan.optimizer_step(optimizer, batch, scheduler=scheduler, counters=TrainingCounters())
    assert metrics["update_completed"] and metrics["counters"]["optimizer_updates"] == 1
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert (model.predictor is not None) is nextlat
    for parameter in model.backbone.fusion.parameters():
        assert parameter.requires_grad is fbt
        assert (parameter.grad is not None) is fbt
    for name, p in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(p, frozen_before[name])
            assert p not in optimizer.state

    card = validation.resource_card(plan, case, optimizer=optimizer)
    inventory = card["observed_parameters"]
    named = dict(model.named_parameters())
    unique = {id(p): p for p in named.values()}
    assert len(unique) == len(named)
    native_count = config.vocab_size * config.model_dim + 16 * (
        4 * config.model_dim**2 + 3 * config.model_dim * config.mlp_intermediate_size)
    fusion_count = 2 * config.model_dim**2
    predictor_count = sum(p.numel() for p in model.predictor.parameters()) if nextlat else 0
    active = native_count + (fusion_count if fbt else 0) + predictor_count
    resident = native_count + fusion_count + predictor_count
    assert inventory == {
        "registered_unique": resident, "resident_parameter_bytes": resident * 4,
        "trainable": active, "gradient_participating": active, "optimizer_owned": active,
        "executed_declared": active, "deployable_inference_declared": native_count + (fusion_count if fbt else 0),
    }
    expected_executed = {n for n in named if fbt or not n.startswith("backbone.fusion.")}
    expected_inference = {n for n in expected_executed if not n.startswith("predictor.")}
    assert set(card["declared_execution_names"]) == expected_executed == set(plan.active_names)
    assert set(card["declared_inference_names"]) == expected_inference
    assert card["named_parameter_shapes"] == {n: list(p.shape) for n, p in named.items()}
    estimate = card["analytic_matrix_work"]
    assert estimate["parameter_counts"]["training_architecture"] == active
    assert estimate["parameter_counts"]["deployable_inference"] == inventory["deployable_inference_declared"]
    assert estimate["input_tokens_per_update"] == 16
    assert estimate["pass_token_work_per_update"] == (32 if fbt else 16)
    assert estimate["rt_block_calls_per_microbatch"] == (2 if rt else 0)
    assert estimate["ordinary_block_calls_per_microbatch"] == (32 if fbt else 16) - (2 if rt else 0)
    assert card["loss_work"] == {"ce_targets": 8, "latent_pairs": 14 if nextlat else 0,
        "kl_triples": 8 if nextlat else 0, "predictor_positions": 14 if nextlat else 0}
    without_optimizer = validation.resource_card(plan, case)
    assert without_optimizer["observed_parameters"]["optimizer_owned"] is None


def _args(*extra):
    return ["--case", "rt-nextlat", "--stage", "correctness", "--output-dir", "unused", *extra]


@pytest.mark.parametrize("batch", [64, 96])
def test_capacity_accepts_only_planned_common_or_bounded_batch(batch):
    args = validation.parse_args(_args("--stage", "capacity", "--batch-size", str(batch), "--length", "512"))
    assert args.batch_size == batch and args.length == 512 and not args.operator_trace


@pytest.mark.parametrize("extra", [
    ["--batch-size", "0"], ["--batch-size", "97"], ["--length", "2048"],
    ["--variant", "reference"], ["--stage", "capacity"],
    ["--stage", "capacity", "--batch-size", "32", "--length", "512"],
    ["--stage", "capacity", "--batch-size", "64", "--length", "32"],
    ["--operator-trace", "--batch-size", "8"],
    ["--operator-trace", "--length", "512"],
    ["--operator-trace", "--stage", "capacity", "--batch-size", "64", "--length", "512"],
])
def test_parser_rejects_outside_bounded_scope(extra):
    with pytest.raises(SystemExit):
        validation.parse_args(_args(*extra))


def test_operator_trace_has_small_correctness_scope_separate_from_capacity():
    args = validation.parse_args(_args("--operator-trace"))
    assert args.operator_trace and args.stage == "correctness"
    assert (args.batch_size, args.length) == (1, 32)


def test_unknown_feature_combination_rejected():
    with pytest.raises(ValueError, match="Unknown F4"):
        validation.selected_case("combined-k3", batch=1, length=32)


def test_source_inventory_covers_shared_legacy_runtime_helpers():
    required = {"scripts/olmo_f4_resources.py", "scripts/olmo_f3e_validate.py",
        "scripts/olmo_f3d_validate.py", "scripts/olmo_f3_graph_training.py",
        "scripts/olmo_f1_common.py", "cdrm/pretrained/static_training.py",
        "cdrm/pretrained/static_nextlat.py", "cdrm/pretrained/resource_estimates.py"}
    assert required <= set(validation.SOURCES)
    assert all((validation.ROOT / name).is_file() for name in validation.SOURCES)
