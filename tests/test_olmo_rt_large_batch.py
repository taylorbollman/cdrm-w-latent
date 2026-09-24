"""CPU safeguards for native RT integration and physical-batch measurement.

Tiny CPU models and fake replay/memory observers validate harness contracts;
these tests do not claim CUDA, installed-kernel, or large-batch clearance.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.lm_training import LMTrainingConfig
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts import olmo_rt_large_batch as harness
from scripts.olmo_ordinary_throughput import make_batch


def command(stage="correctness", case="rt", arm="optimized", batch=8, length=512):
    return ["--stage", stage, "--case", case, "--arm", arm,
        "--batch-size", str(batch), "--length", str(length),
        "--output-dir", "/tmp/unused-native-rt-large-batch"]


@pytest.mark.parametrize("args", [
    command(batch=64), command(arm="other"), command(case="ordinary"),
    command() + ["--profile"], command(stage="capacity", batch=8),
    command(stage="capacity", batch=1024), command(stage="capacity", batch=65),
    command(stage="capacity", batch=64, length=32),
    command(stage="capacity", batch=64) + ["--reference-arm", "optimized"],
    command(stage="capacity", batch=64) + ["--continue-after-compatibility-miss"],
])
def test_cli_rejects_unbounded_or_misleading_runs(args):
    with pytest.raises(SystemExit):
        harness.parse_args(args)


@pytest.mark.parametrize("case", ["rt", "combined"])
def test_large_batch_scope_keeps_spread_rt_selection_and_k2_bootstrap(case):
    args = harness.parse_args(command(stage="capacity", case=case, batch=512))
    assert not args.release_transient_cache
    selected = harness.selected_case(case, batch=args.batch_size, length=args.length)
    mode = selected.mode()
    assert selected.rt_layers == (0, 15) and mode.rt_mode.alpha == 1
    assert selected.fbt is (case == "combined")
    assert selected.nextlat is (case == "combined")
    assert mode.num_passes == (2 if case == "combined" else 1)
    assert mode.enabled is (case == "combined") and mode.beta == 1


@pytest.mark.parametrize("arm", tuple(harness.ARMS))
def test_arm_and_optimizer_preserve_rt_math_and_unique_tied_ownership(arm):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 4)
            self.readout = torch.nn.Linear(4, 7, bias=False)
            self.readout.weight = self.embedding.weight
            self.vector = torch.nn.Parameter(torch.ones(4))
            self.backbone = SimpleNamespace(backbone=SimpleNamespace(
                rt_implementation="native", attention_backend="sdpa",
                attention_precision="mixed", tile_backend="triton",
                backward_tile_backend="triton", backward_memory="recompute"))
            self.config = NextLatConfig(model_dim=32, vocab_chunk_size=128)

    model = Model()
    old_config, base = model.config, model.backbone.backbone
    rt_options = vars(base).copy()
    harness.set_arm(model, arm)
    assert {key: getattr(base, key) for key in rt_options} == rt_options
    assert base.reuse_rope and base.kv_only_writes
    assert base.ordinary_checkpoint_layers is None
    assert base.ordinary_attention_backend == harness.ARMS[arm]["attention"]
    assert base.ordinary_rope_backend == harness.ARMS[arm]["rope"]
    assert base.ordinary_pointwise_backend == harness.ARMS[arm]["pointwise"]
    assert model.config == replace(old_config, ce_chunk_size=2048)
    optimizer, scheduler = harness.optimizer_for(model, arm)
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len({id(p) for p in owned}) == 2
    assert sum(p is model.embedding.weight for p in owned) == 1
    assert optimizer.defaults["fused"] is (None if arm == "control" else True)
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["betas"] == (.9, .95)
    assert optimizer.defaults["eps"] == 1e-8
    assert all(group["lr"] == 5e-6 for group in optimizer.param_groups)
    assert scheduler._cdrm_warmup_updates == 2
    assert all(p.dtype == torch.float32 for p in owned)
    assert not optimizer.state  # Construction only, no implicit warmup update.


def test_full_ce_is_kept_for_every_preparation_parity_timed_and_profile_batch(monkeypatch):
    original = make_batch(2, 32, vocab_size=67, supervision="half")
    original_ce = original.ce_mask.clone()
    seen = []

    def changed(tokenizer, case, update):
        seen.append(update)
        return original

    monkeypatch.setattr(harness, "changed_batch", changed)
    for step in (*range(8), 9):
        batch = harness.batch_for(None, None, step)
        assert torch.equal(batch.ce_mask, batch.valid_mask)
        assert batch.ce_mask.data_ptr() != batch.valid_mask.data_ptr()
        for field in ("input_ids", "valid_mask", "document_ids", "latent_mask", "kl_mask"):
            assert getattr(batch, field) is getattr(original, field)
    assert torch.equal(original.ce_mask, original_ce)
    assert seen == [*range(8), 9]


class ReplayFixture:
    def __init__(self, mutation=None):
        self.model = torch.nn.Linear(2, 2)
        self.mode = object()
        self.active_names = {"weight", "bias"}
        self.calls = []
        self.mutation = mutation
        self.loss = torch.tensor(3.)
        for parameter in self.model.parameters():
            parameter.grad = torch.empty_like(parameter)

    def backward(self, *, replay):
        self.calls.append(replay)
        for parameter in self.model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            if replay and self.mutation == "accumulate":
                parameter.grad.add_(1.)
            else:
                parameter.grad.fill_(1.)
        self.loss.fill_(3.)
        losses = {"pass0/ce": self.loss, "pass0/latent": torch.tensor(0.)}
        if replay:
            if self.mutation == "gradient":
                self.model.weight.grad[0, 0] = 1.25
            elif self.mutation == "missing_gradient":
                self.model.bias.grad = None
            elif self.mutation == "loss":
                self.loss.fill_(3.25)
            elif self.mutation == "missing_loss":
                del losses["pass0/latent"]
            elif self.mutation == "unexpected_loss":
                losses["pass1/ce"] = torch.tensor(3.)
        return losses


def install_replay_fixture(monkeypatch, mutation=None):
    plan = ReplayFixture(mutation)
    monkeypatch.setattr(harness, "loss_snapshot", lambda result: result)
    monkeypatch.setattr(harness, "active_names", lambda model, mode: {"weight", "bias"})
    return plan


def test_cpu_reference_copies_support_repeated_persistent_buffer_overwrite(monkeypatch):
    plan = install_replay_fixture(monkeypatch)
    pointers = [p.grad.data_ptr() for p in plan.model.parameters()]
    check = harness.compare_graph_cpu(plan, "overwrite", replays=2)
    assert check["passed"] and check["all_bitwise_equal"] and check["ownership_matches"]
    assert plan.calls == [False, True, True]
    assert pointers == [p.grad.data_ptr() for p in plan.model.parameters()]
    assert set(check["gradients"]) == {"weight", "bias"}


@pytest.mark.parametrize("mutation", ["gradient", "loss", "accumulate", "missing_gradient",
    "missing_loss", "unexpected_loss"])
def test_cpu_reference_cannot_accept_stale_accumulated_or_missing_results(monkeypatch, mutation):
    plan = install_replay_fixture(monkeypatch, mutation)
    check = harness.compare_graph_cpu(plan, mutation, replays=2)
    assert not check["passed"]


def test_cpu_reference_requires_declared_and_actual_gradient_ownership(monkeypatch):
    plan = install_replay_fixture(monkeypatch)
    plan.active_names = {"weight"}
    check = harness.compare_graph_cpu(plan, "wrong_owner")
    assert check["all_bitwise_equal"] and not check["ownership_matches"] and not check["passed"]


@pytest.mark.parametrize("replays", [0, -1, True, 1.5])
def test_cpu_reference_rejects_invalid_replay_count_before_execution(monkeypatch, replays):
    plan = install_replay_fixture(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        harness.compare_graph_cpu(plan, "invalid", replays=replays)
    assert not plan.calls


@pytest.mark.parametrize("case_name", ["rt", "combined"])
def test_resource_card_keeps_rt_nextlat_and_bootstrap_arithmetic(case_name):
    torch.set_num_threads(1)
    config = replace(OLMoConfig.tiny(), num_layers=16)
    source = OLMoTiledRTForCausalLM(config, attention_backend="math")
    case = harness.selected_case(case_name, batch=2, length=8)
    model = harness.build_model(source.state_dict(), case, device="cpu",
        model_config=config, backend="math", chunk_size=128)
    harness.set_arm(model, "control")
    model.backbone.backbone.ordinary_activation_checkpointing = True
    model.backbone.backbone.backward_memory = "recompute"
    batch = make_batch(2, 8, vocab_size=config.vocab_size, supervision="full")
    plan = StaticFBTTraining(model, batch, mode=case.mode(), config=LMTrainingConfig(precision="fp32"))
    plan.initialize_gradients()
    card = harness.resource_card(plan, case)
    work = card["analytic_matrix_work"]
    assert work["rt_block_calls_per_microbatch"] == 2
    assert work["ordinary_block_calls_per_microbatch"] == (30 if case.nextlat else 14)
    assert work["input_tokens_per_update"] == 16
    assert work["pass_token_work_per_update"] == (32 if case.nextlat else 16)
    assert work["kv_only_writes"] and work["backward_memory"] == "recompute"
    assert work["parameter_counts"]["training_architecture"] == card["observed_parameters"]["trainable"]
    assert (work["parameter_counts"]["nextlat_training_only"] > 0) is case.nextlat
    assert card["loss_work"]["ce_targets"] == 14
    assert (card["loss_work"]["latent_pairs"] > 0) is case.nextlat
    assert (card["loss_work"]["kl_triples"] > 0) is case.nextlat


def memory_fixture(monkeypatch):
    state = {"allocated_gib": 2., "reserved_gib": 3., "peak_allocated_gib": 2.,
             "peak_reserved_gib": 3., "device_used_gib": 4., "device_total_gib": 80.,
             "device_free_gib": 76.}
    resets = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def reset():
        resets.append(True)
        state["peak_allocated_gib"] = state["allocated_gib"]
        state["peak_reserved_gib"] = state["reserved_gib"]

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(harness, "detailed_memory_snapshot", lambda: dict(state))
    report, persisted = {}, []
    phases = harness.MemoryPhases(report, lambda: persisted.append(deepcopy(report)))
    return state, resets, report, persisted, phases


def test_phase_reset_retains_prior_absolute_peaks_and_separate_current_memory(monkeypatch):
    state, resets, report, persisted, phases = memory_fixture(monkeypatch)
    with phases.phase("load"):
        state.update(peak_allocated_gib=9., peak_reserved_gib=13.)
    with phases.phase("validation"):
        state.update(peak_allocated_gib=5., peak_reserved_gib=7.)
    total = phases.setup_summary()
    assert total["peak_allocated_gib"] == 9. and total["peak_reserved_gib"] == 13.
    assert total["allocated_gib"] == 2. and total["reserved_gib"] == 3.
    assert total["device_used_gib"] == 4.  # Sampled boundary usage, not a synthetic peak.
    assert len(resets) == 2 and len(persisted) == 4
    assert report["memory_phases"]["validation"]["start"]["peak_allocated_gib"] == 2.


def test_phase_failure_preserves_observed_peak_and_original_exception(monkeypatch):
    state, resets, report, persisted, phases = memory_fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="intentional failure"):
        with phases.phase("preparation"):
            state.update(peak_allocated_gib=17., peak_reserved_gib=21.)
            raise RuntimeError("intentional failure")
    row = report["memory_phases"]["preparation"]
    assert row["error"] is True and row["peak_allocated_gib"] == 17.
    assert persisted[-1]["memory_phases"]["preparation"]["end"]["peak_reserved_gib"] == 21.
    assert phases.setup_summary()["peak_reserved_gib"] == 21.


def test_capture_subphases_and_incomplete_phase_survive_setup_aggregation(monkeypatch):
    state, _, report, _, phases = memory_fixture(monkeypatch)
    phases.capture_observer("warmup", "begin")
    state.update(peak_allocated_gib=11., peak_reserved_gib=15.)
    phases.capture_observer("warmup", "end")
    phases.capture_observer("capture", "begin")
    assert "end" not in report["memory_phases"]["capture_capture"]
    assert phases.setup_summary()["peak_allocated_gib"] == 11.


def test_numerical_miss_can_be_retained_without_promoting_or_deferring_operation_failures():
    miss = {"name": "same_state_candidate_vs_reference", "passed": False,
        "ownership_matches": True, "finite": True, "counts_equal": True,
        "losses": {"ce": {}}, "outputs": {"pass0": {}}, "gradients": {"weight": {}}}
    assert harness.can_continue_after_failure(miss, enabled=True)
    assert not harness.can_continue_after_failure(miss, enabled=False)
    operational = {**miss, "name": "candidate_changed_tokens_overwrite"}
    assert not harness.can_continue_after_failure(operational, enabled=True)
    summary = harness.final_check_summary([miss, {"name": "candidate_changed_weights", "passed": True}])
    assert summary == {"status": "failed", "numerical_compatibility_passed": False,
                       "operational_checks_passed": True}
