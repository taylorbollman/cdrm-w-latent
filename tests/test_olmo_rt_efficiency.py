"""CPU checks for RT efficiency experiment isolation and prospective gates."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.static_nextlat import PreparedNextLatLayout
from scripts import olmo_rt_efficiency as harness
from scripts.olmo_ordinary_throughput import make_batch


def command(stage, case="rt", batch=None, *extra):
    return ["--stage", stage, "--case", case, "--batch-size",
            str(batch or (8 if stage == "correctness" else 64)),
            "--output-dir", "/tmp/unused-rt-efficiency", *extra]


def test_stage_defaults_keep_correctness_masks_and_full_capacity_separate():
    correctness = harness.parse_args(command("correctness") + ["--comparison", "control-rope"])
    capacity = harness.parse_args(command("capacity") + ["--arm", "control"])
    assert correctness.supervision == "half"
    assert capacity.supervision == "full"
    assert correctness.arm is None and capacity.comparison is None


@pytest.mark.parametrize("arguments", [
    command("correctness", batch=64) + ["--comparison", "control-rope"],
    command("correctness") + ["--comparison", "control-rope", "--supervision", "full"],
    command("correctness", case="ordinary") + ["--comparison", "rope-both"],
    command("correctness") + ["--comparison", "control-rope", "--profile"],
    command("correctness") + ["--arm", "rope"],
    command("capacity", batch=8) + ["--arm", "rope"],
    command("capacity", case="ordinary", batch=128) + ["--arm", "rope"],
    command("capacity", case="ordinary") + ["--arm", "both"],
    command("capacity", case="combined") + ["--arm", "control", "--profile"],
    command("capacity") + ["--arm", "rope", "--profile"],
    command("capacity", batch=128) + ["--arm", "both", "--profile"],
])
def test_cli_rejects_unplanned_or_misleading_comparisons(arguments):
    with pytest.raises(SystemExit):
        harness.parse_args(arguments)


@pytest.mark.parametrize("arm,expected", list(harness.ARMS.items()))
def test_arm_changes_only_execution_flags_and_ce_override(arm, expected):
    base = SimpleNamespace(reuse_rope=False, kv_only_writes=False)
    original = NextLatConfig(model_dim=32, vocab_chunk_size=128)
    model = SimpleNamespace(backbone=SimpleNamespace(backbone=base), config=original)
    harness.set_arm(model, arm)
    assert (base.reuse_rope, base.kv_only_writes) == expected
    assert model.config.effective_ce_chunk_size == 2048
    expected_config = original.to_dict()
    expected_config["ce_chunk_size"] = 2048
    assert model.config.to_dict() == expected_config


def test_full_capacity_changes_ce_selection_only_and_preserves_data(monkeypatch):
    original = make_batch(2, 512, vocab_size=67, supervision="half")
    monkeypatch.setattr(harness, "changed_batch", lambda *args: original)
    half = harness.batch_for(None, None, 0, "half")
    full = harness.batch_for(None, None, 0, "full")
    assert half is original and full is not original
    for name in ("input_ids", "valid_mask", "document_ids", "latent_mask", "kl_mask"):
        assert torch.equal(getattr(half, name), getattr(full, name))
    assert torch.equal(half.ce_mask[:, :256], torch.zeros(2, 256, dtype=torch.bool))
    config = NextLatConfig(model_dim=32)
    counts = [PreparedNextLatLayout.from_batch(batch, config, enabled=True).counts for batch in (half, full)]
    assert [item["ce"] for item in counts] == [512, 1022]
    assert counts[0]["latent"] == counts[1]["latent"]
    assert counts[0]["kl"] == counts[1]["kl"]


def screen_arguments():
    reference = torch.ones(4)
    exact = harness.metric(reference, reference)
    return {"exact_required": False, "ownership": True, "finite": True, "counts_equal": True,
        "losses": {"ce": deepcopy(exact)}, "gradients": {"weight": deepcopy(exact)},
        "outputs": {"pass0": deepcopy(exact)}}


def test_rope_reuse_requires_bitwise_not_merely_budget_agreement():
    args = screen_arguments()
    args["outputs"]["pass0"] = harness.metric(torch.tensor([1.000001]), torch.tensor([1.0]))
    assert harness.comparison_passes(**args)["passed"]
    args["exact_required"] = True
    assert not harness.comparison_passes(**args)["passed"]


@pytest.mark.parametrize("key", ["ownership", "finite", "counts_equal"])
def test_valid_numeric_values_do_not_excuse_structural_failures(key):
    args = screen_arguments()
    args[key] = False
    assert not harness.comparison_passes(**args)["passed"]


def test_max_error_gate_retains_previous_bf16_boundary_qualification():
    args = screen_arguments()
    args["gradients"]["weight"]["max_relative"] = 0.06349206349
    # Global L2 and other values still pass: one tensor max crossing must fail.
    assert harness.comparison_passes(**args)["global_gradient_relative_l2"] == 0
    assert not harness.comparison_passes(**args)["passed"]
    args["gradients"]["weight"]["max_relative"] = 1 / 16
    assert harness.comparison_passes(**args)["passed"]


def test_nonzero_candidate_against_zero_gradient_reference_fails():
    args = screen_arguments()
    args["gradients"]["weight"] = harness.metric(torch.tensor([0.0001]), torch.tensor([0.0]))
    assert not harness.comparison_passes(**args)["passed"]


def test_completed_optimizer_steps_remain_counted_before_scheduler_failure():
    weight = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    report = {"physical_optimizer_updates": 0}
    hook = harness.track_optimizer_steps(optimizer, report)
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    assert report["physical_optimizer_updates"] == 1
    # The count is independent of TrainingCounters, returned metrics or a
    # subsequent scheduler exception. Removing the hook restores normal use.
    hook.remove()
    optimizer.step()
    assert report["physical_optimizer_updates"] == 1


def test_runtime_snapshot_includes_both_new_entry_and_rope_math():
    assert "scripts/olmo_rt_efficiency.py" in harness.SOURCES
    assert "cdrm/pretrained/olmo_rope.py" in harness.SOURCES
    assert "cdrm/pretrained/olmo_tiled.py" in harness.SOURCES
