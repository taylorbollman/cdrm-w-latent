"""F1 harness contracts; model/loss mathematical oracles remain in their suites."""
from dataclasses import asdict, replace
from itertools import product
import json

import pytest
import torch

from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TrainingCounters, load_training_checkpoint,
    optimizer_ownership, optimizer_step, save_training_checkpoint,
)
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from scripts.olmo_f1_common import (
    GradientObserver, IntegrationCase, active_names, boundary_digests,
    build_model, build_optimizer, changed_fixture, default_cases,
    inference_names, rng_probe, split_rows, state_change, state_health,
)
from scripts.olmo_lm_common import state_digests
from scripts.olmo_f1_integrate import checkpoint_configuration, load_configuration
from scripts import olmo_f1_integrate as integration


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(49217)


@pytest.fixture
def source():
    # Native predictor rounding at factor 1.6 is nonzero at this width.
    config = replace(OLMoConfig.tiny(), model_dim=64, mlp_intermediate_size=128)
    model = OLMoForCausalLM(config, attention_backend="math")
    return config, model.state_dict()


def make_model(source, case):
    config, state = source
    return build_model(state, case, device="cpu", model_config=config,
                       chunk_size=3, backend="math")


def small_batch(generator=None):
    ids = torch.tensor([[2, 4, 7, 11, 18, 29, 41, 60],
                        [3, 8, 13, 23, 60, 1, 1, 1]])
    valid = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    if generator is not None:
        sampled = torch.randint(2, 59, ids.shape, generator=generator)
        ids = torch.where(valid, sampled, ids)
        ids[0, 7] = ids[1, 4] = 60
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, 2] = False
    latent[1, 1] = False
    kl[0, 3] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def observed_update(model, optimizer, case, *, update=0, scheduler=None,
                    counters=None, value=None):
    mode = case.mode(update)
    observer = GradientObserver(model)
    try:
        row = optimizer_step(model, optimizer, split_rows(small_batch() if value is None else value),
                             config=LMTrainingConfig(), scheduler=scheduler,
                             counters=counters, backbone_kwargs={"mode": mode})
        gradients = observer.report(active_names(model, mode))
    finally:
        observer.close()
    return row, gradients


def test_default_matrix_covers_independent_switches_and_declared_extensions():
    cases = default_cases()
    assert len({case.name for case in cases}) == len(cases)
    primary = cases[:8]
    assert {(case.fbt, bool(case.rt_layers), case.nextlat) for case in primary} == set(product((False, True), repeat=3))
    assert all((case.batch_size, case.length, case.updates) == (2, 32, 3) for case in primary)
    assert {case.name for case in primary if case.resume} == {"ordinary", "rt-fbt-nextlat"}
    for case in primary:
        mode = case.mode()
        assert mode.enabled == case.fbt
        assert mode.num_passes == (2 if case.fbt else 1)
        assert mode.rt_mode.selected_layers == case.rt_layers
    indexed = {case.name: case for case in cases}
    assert indexed["all-three-k3"].mode().num_passes == 3
    assert indexed["all-three-two-rt-layers"].rt_layers == (0, 15)
    assert indexed["all-three-fractional"].mode().beta == .35
    assert indexed["all-three-fractional"].mode().rt_mode.alpha == .37
    assert {(case.fbt, bool(case.rt_layers), case.nextlat) for case in cases if case.profile} == {
        (False, False, False), (False, True, False),
        (True, False, False), (True, True, True),
    }
    assert all(case.length == 512 and case.batch_size == 1 for case in cases if case.profile)


@pytest.mark.parametrize("kwargs", [
    {"fbt": True, "passes": 1},
    {"resume": True, "updates": 2},
    {"transition": True, "fbt": False, "rt_layers": (0,)},
    {"transition": True, "fbt": True, "rt_layers": ()},
    {"transition": True, "fbt": True, "rt_layers": (0,), "profile": True},
    {"batch_size": True},
    {"length": 2049},
])
def test_case_rejects_misleading_runtime_contracts(kwargs):
    with pytest.raises((ValueError, TypeError)):
        IntegrationCase("invalid", **kwargs)


@pytest.mark.parametrize("update", [-1, True, .5])
def test_mode_rejects_invalid_update_indices(update):
    case = IntegrationCase("transition", fbt=True, nextlat=True,
                           rt_layers=(0,), transition=True)
    with pytest.raises((ValueError, TypeError)):
        case.mode(update)


@pytest.mark.parametrize("fbt,rt,nextlat", list(product((False, True), repeat=3)))
def test_cpu_build_preserves_source_tying_and_inactive_ownership(source, fbt, rt, nextlat):
    config, state = source
    before = {name: tensor.clone() for name, tensor in state.items()}
    case = IntegrationCase("ownership", fbt=fbt, nextlat=nextlat,
                           rt_layers=(0,) if rt else (), length=8)
    model = make_model(source, case)
    optimizer, _ = build_optimizer(model)
    tied = model.backbone.token_embeddings.weight
    assert model.backbone.readout_weight is tied
    assert sum(p is tied for group in optimizer.param_groups for p in group["params"]) == 1
    owned = {name for group in optimizer_ownership(model, optimizer) for name in group}
    assert owned == {name for name, p in model.named_parameters() if p.requires_grad}
    fusion = {name for name, _ in model.named_parameters() if name.startswith("backbone.fusion.")}
    assert len(fusion) == 2
    assert fusion <= owned if fbt else fusion.isdisjoint(owned)
    predictor = {name for name, _ in model.named_parameters() if name.startswith("predictor.")}
    assert len(predictor) == (4 if nextlat else 0)
    assert inference_names(model, case).isdisjoint(predictor)
    assert fusion <= inference_names(model, case) if fbt else fusion.isdisjoint(inference_names(model, case))
    assert model.config.model_dim == config.model_dim
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # A CPU training harness must not alias its source even before an update.
    with torch.no_grad():
        tied.add_(.25)
    for name, tensor in state.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_observer_records_canonical_backward_without_changing_update(source):
    case = IntegrationCase("all-three", fbt=True, nextlat=True, rt_layers=(0,),
                           alpha=.37, beta=.35, length=8)
    model, reference = make_model(source, case), make_model(source, case)
    optimizer, scheduler = build_optimizer(model)
    other_optimizer, other_scheduler = build_optimizer(reference)
    counters, other_counters = TrainingCounters(), TrainingCounters()
    row, gradients = observed_update(model, optimizer, case, scheduler=scheduler, counters=counters)
    other = optimizer_step(reference, other_optimizer, split_rows(small_batch()),
                           config=LMTrainingConfig(), scheduler=other_scheduler,
                           counters=other_counters, backbone_kwargs={"mode": case.mode()})
    assert gradients["passed"]
    assert all(item["backward_contributions"] == 2 for item in gradients["tensors"].values())
    assert all(p.grad is None for p in model.parameters())
    assert row == other
    assert boundary_digests(model, optimizer, scheduler, counters) == boundary_digests(
        reference, other_optimizer, other_scheduler, other_counters)
    assert state_health(model, optimizer)["passed"]


def test_observer_missing_unexpected_and_nonfinite_gradients_fail(source):
    case = IntegrationCase("ordinary", length=8)
    model = make_model(source, case)
    name, parameter = next((name, p) for name, p in model.named_parameters() if p.requires_grad)
    observer = GradientObserver(model)
    try:
        assert not observer.report({name})["passed"]
        (parameter.sum() * float("nan")).backward()
        report = observer.report({"nonexistent"})
        assert report["missing"] == ["nonexistent"]
        assert report["unexpected"] == [name]
        assert not report["tensors"][name]["finite"]
        assert not report["passed"]
    finally:
        observer.close()
    previous = len(observer.values[name])
    model.zero_grad(set_to_none=True)
    parameter.sum().backward()
    assert not observer.handles
    assert len(observer.values[name]) == previous


def test_transition_changes_participation_without_fabricating_fusion_moments(source):
    case = IntegrationCase("transition", fbt=True, nextlat=True,
                           rt_layers=(0,), transition=True, length=8)
    model = make_model(source, case)
    optimizer, scheduler = build_optimizer(model)
    counters = TrainingCounters()
    fusion = {name: p for name, p in model.named_parameters() if name.startswith("backbone.fusion.")}
    for update, (alpha, beta) in enumerate(((0., 0.), (.37, .35), (1., 1.))):
        mode = case.mode(update)
        assert (mode.rt_mode.alpha, mode.beta) == (alpha, beta)
        before = state_digests(model)
        row, gradients = observed_update(model, optimizer, case, update=update,
                                         scheduler=scheduler, counters=counters)
        assert gradients["passed"]
        assert row["counters"]["optimizer_updates"] == update + 1
        assert state_change(before, state_digests(model), active_names(model, mode))["passed"]
        if update == 0:
            assert set(fusion).isdisjoint(gradients["tensors"])
            assert all(p not in optimizer.state for p in fusion.values())
        else:
            assert set(fusion) <= gradients["tensors"].keys()
            assert all(p in optimizer.state for p in fusion.values())


def test_state_change_rejects_missing_active_updates_and_unexpected_state_mutation():
    before = {"weight": "a", "frozen": "b", "buffer": "c"}
    changed = dict(before, weight="d")
    assert state_change(before, changed, {"weight"})["passed"]
    unchanged = state_change(before, before, {"weight"})
    assert not unchanged["passed"]
    assert unchanged["unchanged_active"] == ["weight"]
    contaminated = state_change(before, dict(changed, buffer="bad"), {"weight"})
    assert not contaminated["passed"]
    assert contaminated["unexpected_changes"] == ["buffer"]
    with pytest.raises(AssertionError, match="keys"):
        state_change(before, {"weight": "d"}, {"weight"})


class FixtureTokenizer:
    def encode(self, text):
        # Long enough for all fixture sizes, with non-period-three token order.
        return [2 + (index % 53) for index in range(4096)]


def test_changed_fixture_preserves_masks_eos_and_rng_while_changing_tokens():
    case = IntegrationCase("fixture", length=32)
    rng = torch.get_rng_state().clone()
    first = changed_fixture(FixtureTokenizer(), case, 0, device="cpu")
    again = changed_fixture(FixtureTokenizer(), case, 0, device="cpu")
    later = changed_fixture(FixtureTokenizer(), case, 1, device="cpu")
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(first.input_ids, again.input_ids, rtol=0, atol=0)
    assert not torch.equal(first.input_ids, later.input_ids)
    for name in ("valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"):
        torch.testing.assert_close(getattr(first, name), getattr(later, name), rtol=0, atol=0)
    for row in range(case.batch_size):
        valid = first.valid_mask[row]
        count = int(valid.sum())
        assert int(first.input_ids[row, count - 1]) == 50279
        assert int(later.input_ids[row, count - 1]) == 50279
        assert first.document_ids[row, valid].unique().numel() == 1
        assert bool((later.input_ids[row, ~valid] == 1).all())


def test_split_rows_keeps_distinct_global_objective_counts(source):
    case = IntegrationCase("nextlat", nextlat=True, length=8)
    model = make_model(source, case)
    value = small_batch()
    pieces = split_rows(value)
    assert len(pieces) == 2
    counts = [model.counts(piece) for piece in pieces]
    assert counts == [{"ce": 6, "latent": 7, "kl": 5},
                      {"ce": 4, "latent": 3, "kl": 3}]
    assert model.counts(value) == {key: sum(row[key] for row in counts) for key in counts[0]}
    for name, tensor in vars(value).items():
        torch.testing.assert_close(torch.cat([getattr(piece, name) for piece in pieces]), tensor)


@pytest.mark.parametrize("all_three", [False, True])
def test_harness_checkpoint_replays_changed_input_update_and_rng(source, tmp_path, all_three):
    case = IntegrationCase("resume", fbt=all_three, nextlat=all_three,
                           rt_layers=(0,) if all_three else (), resume=True, length=8)
    model = make_model(source, case)
    optimizer, scheduler = build_optimizer(model)
    counters = TrainingCounters()
    generator = torch.Generator().manual_seed(608)
    configuration = {"case": asdict(case), "training": asdict(LMTrainingConfig())}
    fingerprint = {"checkpoint_sha256": "a" * 64, "scope": "tiny F1 CPU harness"}
    for update in range(2):
        observed_update(model, optimizer, case, update=update, scheduler=scheduler,
                        counters=counters, value=small_batch(generator))
    checkpoint = save_training_checkpoint(tmp_path / "update2.pt", model, optimizer,
        scheduler=scheduler, counters=counters, data_cursor={"next_update": 2},
        configuration=configuration, source_fingerprint=fingerprint, generators={"data": generator})
    third = small_batch(generator)
    expected_row, expected_gradients = observed_update(model, optimizer, case, update=2,
        scheduler=scheduler, counters=counters, value=third)
    expected = boundary_digests(model, optimizer, scheduler, counters)
    expected_rng = rng_probe("cpu")

    resumed = make_model(source, case)
    restored_optimizer, restored_scheduler = build_optimizer(resumed)
    restored_generator = torch.Generator().manual_seed(999)
    restored = load_training_checkpoint(checkpoint["path"], resumed, restored_optimizer,
        scheduler=restored_scheduler, configuration=configuration, source_fingerprint=fingerprint,
        generators={"data": restored_generator}, expected_sha256=checkpoint["sha256"])
    assert restored["data_cursor"] == {"next_update": 2}
    assert restored["counters"].optimizer_updates == 2
    replay = small_batch(restored_generator)
    torch.testing.assert_close(replay.input_ids, third.input_ids, rtol=0, atol=0)
    actual_row, actual_gradients = observed_update(resumed, restored_optimizer, case, update=2,
        scheduler=restored_scheduler, counters=restored["counters"], value=replay)
    assert actual_row == expected_row
    assert actual_gradients == expected_gradients
    assert boundary_digests(resumed, restored_optimizer, restored_scheduler, restored["counters"]) == expected
    assert rng_probe("cpu") == expected_rng


def runtime_config():
    return {"schema": "olmo-f1-configuration-v1", "seed": 419,
            "learning_rate": 1e-5, "vocab_chunk_size": 3,
            "profile_warmup": 1, "profile_repeats": 2,
            "cases": [asdict(IntegrationCase("ordinary", length=8)),
                      asdict(IntegrationCase("all-three", fbt=True, nextlat=True,
                                             rt_layers=(0,), length=8))]}


def test_runtime_configuration_roundtrips_modes_and_chunk_policy(tmp_path):
    path = tmp_path / "config.json"
    config = runtime_config()
    path.write_text(json.dumps(config))
    loaded, cases = load_configuration(path)
    assert loaded == json.loads(path.read_text())
    assert loaded["vocab_chunk_size"] == 3
    assert [case.name for case in cases] == ["ordinary", "all-three"]
    assert cases[0].mode().num_passes == 1
    assert cases[1].mode().num_passes == 2
    assert cases[1].rt_layers == (0,)


@pytest.mark.parametrize("change", [
    {"schema": "unrelated"},
    {"learning_rate": True},
    {"learning_rate": float("nan")},
    {"learning_rate": 1e-2},
    {"profile_repeats": 0},
    {"vocab_chunk_size": True},
    {"cases": []},
    {"unrecognized": "silently ignored"},
])
def test_runtime_configuration_rejects_ambiguous_or_unsafe_scope(tmp_path, change):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(runtime_config() | change))
    with pytest.raises((ValueError, TypeError)):
        load_configuration(path)


def test_runtime_configuration_rejects_duplicate_case_names(tmp_path):
    config = runtime_config()
    config["cases"].append(dict(config["cases"][0]))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unique"):
        load_configuration(path)


def test_checkpoint_configuration_records_executed_model_and_objective(source):
    case = IntegrationCase("all-three", fbt=True, nextlat=True, rt_layers=(0,),
                           alpha=.37, beta=.35, length=8)
    model = make_model(source, case)
    config = checkpoint_configuration(model, case, runtime_config())
    assert config["case"] == asdict(case)
    assert config["gamma"] == 1.0
    assert config["nextlat"] == model.config.to_dict()
    assert config["nextlat_enabled"] is True
    assert config["fusion"] == model.backbone.fusion_config.to_dict()
    assert config["native"] == source[0].to_dict()
    assert config["attention_backend"] == model.backbone.backbone.attention_backend
    assert config["attention_precision"] == model.backbone.backbone.attention_precision


@pytest.mark.parametrize("zero_everywhere", [False, True])
def test_runner_observation_returns_failure_evidence_for_persistence(source, monkeypatch, zero_everywhere):
    case = IntegrationCase("ordinary", length=8)
    model = make_model(source, case)
    optimizer, scheduler = build_optimizer(model)

    def incomplete_step(model, optimizer, batches, **kwargs):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if zero_everywhere:
            sum(p.sum() * 0 for p in parameters).backward()
            norm = 0.
        else:
            # An attached but incomplete graph must preserve evidence of the
            # missing leaves rather than let the runner call the case healthy.
            parameters[0].sum().backward()
            norm = float(parameters[0].grad.norm())
        optimizer.zero_grad(set_to_none=True)
        return {"gradient_norm_before_clip": norm}

    monkeypatch.setattr(integration, "optimizer_step", incomplete_step)
    metrics, gradients = integration.observed_step(
        model, optimizer, scheduler, TrainingCounters(), [small_batch()], case.mode())
    assert "gradient_norm_before_clip" in metrics
    assert not gradients["passed"]
    if zero_everywhere:
        assert not gradients["missing"]
        assert not gradients["nonzero_accumulated_gradient"]
    else:
        assert gradients["missing"]
        assert gradients["nonzero_accumulated_gradient"]
