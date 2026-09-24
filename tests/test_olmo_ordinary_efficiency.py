"""CPU safeguards for the prospective ordinary OLMo execution comparison."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.static_nextlat import PreparedNextLatLayout
from scripts import olmo_ordinary_efficiency as harness
from scripts import olmo_rt_efficiency as parity
from scripts.olmo_ordinary_throughput import make_batch


def command(stage="correctness", arm="fa4", batch=8, length=512):
    return ["--stage", stage, "--arm", arm, "--batch-size", str(batch),
            "--length", str(length), "--output-dir", "/tmp/unused-ordinary-efficiency"]


@pytest.mark.parametrize("batch,length", [(16, 512), (32, 512), (64, 512), (8, 2048), (16, 2048)])
def test_capacity_is_bounded_by_physical_batch_and_context(batch, length):
    args = harness.parse_args(command("capacity", batch=batch, length=length))
    assert args.reference_arm == "control"


@pytest.mark.parametrize("arguments", [
    command(batch=64), command() + ["--profile"],
    command("capacity", batch=8), command("capacity", batch=64, length=2048),
    command("capacity", batch=16, length=32),
    command("capacity", batch=64) + ["--reference-arm", "fa4"],
    command("capacity", batch=64) + ["--continue-after-compatibility-miss"],
])
def test_cli_rejects_unplanned_work(arguments):
    with pytest.raises(SystemExit):
        harness.parse_args(arguments)


def test_long_context_correctness_and_stacked_candidate_are_explicit():
    args = harness.parse_args(command(arm="fa4-compiled-checkpoint-none", batch=2, length=2048)
                              + ["--reference-arm", "fa4-compiled"])
    assert args.batch_size == 2 and args.length == 2048
    assert harness.exact_comparison(args.reference_arm, args.arm)


@pytest.mark.parametrize("arm,options", list(harness.ARMS.items()))
def test_arm_switches_are_execution_only_with_fixed_ce(arm, options):
    base = SimpleNamespace(config=SimpleNamespace(num_layers=16), reuse_rope=False)
    config = NextLatConfig(model_dim=32, vocab_chunk_size=128)
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=base), config=config)
    harness.set_arm(model, arm)
    assert base.ordinary_attention_backend == options[0]
    assert base.ordinary_pointwise_backend == options[2]
    assert base.ordinary_checkpoint_layers == harness.checkpoint_layers(options[1], 16)
    assert base.reuse_rope is True
    assert model.config == replace(config, ce_chunk_size=2048)


def test_full_ce_factory_preserves_tokens_and_non_ce_selection(monkeypatch):
    original = make_batch(2, 32, vocab_size=67, supervision="half")
    monkeypatch.setattr(harness, "changed_batch", lambda *args: original)
    result = harness.batch_for(None, None, 7)
    assert result is not original
    for name in ("input_ids", "valid_mask", "document_ids", "latent_mask", "kl_mask"):
        assert torch.equal(getattr(original, name), getattr(result, name))
    assert result.ce_mask.all()
    layout = PreparedNextLatLayout.from_batch(result, NextLatConfig(model_dim=32), enabled=False)
    assert layout.counts == {"ce": 62, "latent": 0, "kl": 0}


def test_every_update_in_both_parity_arms_uses_full_ce(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff_out = torch.nn.Linear(2, 1, bias=False)

    model = Model()
    calls = []
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    monkeypatch.setattr(parity, "build_optimizer", lambda _: (optimizer, scheduler))
    monkeypatch.setattr(parity, "boundary_digests", lambda model, *args: {
        name: value.detach().tolist() for name, value in model.state_dict().items()})
    monkeypatch.setattr(parity, "state_health", lambda *args: {"passed": True})
    original = make_batch(2, 32, vocab_size=67, supervision="half")
    monkeypatch.setattr(harness, "changed_batch", lambda *args: deepcopy(original))

    def step(optimizer, batch, *, replay, scheduler, counters):
        assert bool(batch.ce_mask.all())
        calls.append(replay)
        optimizer.zero_grad(set_to_none=True)
        loss = model.ff_out(torch.ones(2)).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return {"objective": float(loss.detach()), "ce_targets": 62}

    plan = SimpleNamespace(model=model, optimizer_step=step)
    report = {"physical_optimizer_updates": 0}
    result = harness.complete_update_parity(plan, None, None, report=report,
                                           persist=lambda: None, batch_factory=harness.batch_for)
    assert result["passed"] and report["physical_optimizer_updates"] == 6
    assert calls == [False] * 3 + [True] * 3


def test_checkpoint_only_change_requires_exact_comparison():
    assert harness.exact_comparison("control", "checkpoint-none")
    assert harness.exact_comparison("control", "checkpoint-alternating")
    assert not harness.exact_comparison("control", "fa4")
    assert not harness.exact_comparison("control", "compiled")


def test_selected_checkpoint_accounting_preserves_non_recompute_work(monkeypatch):
    config = OLMoConfig(model_dim=8, num_layers=4, num_heads=2,
                        mlp_intermediate_size=16, vocab_size=64, tokenizer_vocab_size=64,
                        eos_token_id=63, max_context_length=32)
    base = SimpleNamespace(config=config, ordinary_checkpoint_layers=None)
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=base))
    plan = SimpleNamespace(model=model, mode=FBTMode(enabled=False, num_passes=1),
        counts={"ce": 14, "latent": 0, "kl": 0},
        loss_layout=SimpleNamespace(needed_source_indices=torch.empty(0, dtype=torch.long)))
    case = SimpleNamespace(batch_size=2, length=8)
    monkeypatch.setattr(harness, "parameter_inventory", lambda *args, **kwargs: {"registered_unique": 123})
    monkeypatch.setattr(harness, "active_names", lambda *args: set())
    monkeypatch.setattr(harness, "inference_names", lambda *args: set())
    records = {}
    for name, selected in (("all", None), ("alternating", (0, 2)), ("none", ())):
        base.ordinary_checkpoint_layers = selected
        records[name] = harness.resource_card(plan, case)
    for boundary in ("matrix_flops_minimum", "matrix_flops_maximum"):
        values = {name: card["analytic_matrix_work"][boundary] for name, card in records.items()}
        assert values["all"] > values["alternating"] > values["none"]
        assert values["all"] - values["alternating"] == values["alternating"] - values["none"]
    names = lambda row: {part["name"]: part for part in row["analytic_matrix_work"]["components"]}
    for name, row in names(records["none"]).items():
        assert row == names(records["all"])[name]
    assert [records[name]["checkpointed_ordinary_layer_count"] for name in ("all", "alternating", "none")] == [4, 2, 0]


def test_dependency_hash_guard_detects_change(tmp_path):
    path = tmp_path / "interface.py"
    path.write_text("original")
    record = {"fa4_sources": {"interface.py": {"source": str(path), "sha256": harness.sha256_file(path)}}}
    assert harness.check_dependencies(record)
    path.write_text("changed")
    assert not harness.check_dependencies(record)


def test_source_set_includes_imported_helper_and_runtime_math():
    for name in ("scripts/olmo_rt_efficiency.py", "scripts/olmo_ordinary_efficiency.py",
                 "cdrm/pretrained/olmo_ordinary.py", "cdrm/pretrained/olmo_static.py",
                 "cdrm/pretrained/olmo_rope.py", "scripts/docker_shell.sh"):
        assert name in harness.SOURCES


def compatibility_miss():
    return {"name": "same_state_candidate_vs_reference", "passed": False,
        "ownership_matches": True, "finite": True, "counts_equal": True,
        "losses": {"ce": {"relative_l2": 0.0000271257}},
        "gradients": {"weight": {}}, "outputs": {"pass0": {}}}


def test_numeric_compatibility_continuation_is_opt_in_and_preserves_failure():
    check = compatibility_miss()
    original = deepcopy(check)
    assert not harness.can_continue_after_failure(check, enabled=False)
    assert harness.can_continue_after_failure(check, enabled=True)
    assert check == original and check["passed"] is False
    assert not harness.parse_args(command()).continue_after_compatibility_miss
    assert harness.parse_args(command() + ["--continue-after-compatibility-miss"]).continue_after_compatibility_miss


@pytest.mark.parametrize("field", ["ownership_matches", "finite", "counts_equal", "losses", "outputs", "gradients"])
def test_structural_or_incomplete_checks_cannot_continue(field):
    check = compatibility_miss()
    check[field] = False
    assert not harness.can_continue_after_failure(check, enabled=True)


@pytest.mark.parametrize("name", ["ordinary_dispatch_no_fallback", "candidate_initial_graph",
    "candidate_changed_tokens_overwrite", "complete_adamw_update_parity", "candidate_changed_weights"])
def test_operational_failure_is_never_deferrable(name):
    check = compatibility_miss()
    check["name"] = name
    assert not harness.can_continue_after_failure(check, enabled=True)


def test_healthy_operational_checks_do_not_promote_numeric_failure_to_pass():
    checks = [compatibility_miss(), {"name": "candidate_changed_weights", "passed": True}]
    summary = harness.final_check_summary(checks)
    assert summary == {"status": "failed", "numerical_compatibility_passed": False,
                       "operational_checks_passed": True}
    checks[0]["passed"] = True
    assert harness.final_check_summary(checks)["status"] == "passed"
    assert harness.final_check_summary([])["status"] == "failed"
