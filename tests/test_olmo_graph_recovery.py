"""CPU orchestration/strict-checkpoint contracts; never CUDA-performance evidence."""
from contextlib import nullcontext
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters, build_adamw,
    build_warmup_scheduler, load_training_checkpoint, save_training_checkpoint)
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts.olmo_f1_common import IntegrationCase, build_model
from scripts import olmo_graph_recovery as recovery


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_tracking_rng_guard_restores_python_numpy_and_torch_on_error():
    before = recovery.rng_digests()
    with pytest.raises(RuntimeError), recovery.preserve_rng():
        random.random(); np.random.random(); torch.rand(5)
        raise RuntimeError("network failure")
    assert recovery.rng_digests() == before


@pytest.mark.parametrize("update", [-1, True, .5])
def test_cursor_rejects_invalid_update(update):
    with pytest.raises(ValueError):
        recovery.cursor_for(IntegrationCase("rt", rt_layers=(0, 1)), update)


@pytest.mark.parametrize("field,value", [("next_update", 3), ("length", 12),
    ("batch_size", 8), ("fixture", "different")])
def test_cursor_rejects_wrong_boundary_or_source(field, value):
    case = IntegrationCase("rt", rt_layers=(0, 1), batch_size=2, length=8)
    cursor = recovery.cursor_for(case, 2)
    recovery.validate_cursor(cursor, case, TrainingCounters(optimizer_updates=2))
    cursor[field] = value
    with pytest.raises(ValueError, match="cursor"):
        recovery.validate_cursor(cursor, case, TrainingCounters(optimizer_updates=2))


def checkpoint_fixture(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = build_adamw(model, lr=1e-3, fused=False)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
    model(torch.ones(2, 3)).square().sum().backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "boundary.pt"
    config = {"ordinary_rope_backend": "native", "ordinary_pointwise_backend": "compiled",
        "tile_backend": "triton", "selected_layers": [0, 15], "fused_adam": False}
    fingerprint = {"checkpoint_sha256": "a" * 64}
    receipt = save_training_checkpoint(path, model, optimizer, scheduler=scheduler,
        counters=TrainingCounters(optimizer_updates=1), data_cursor={"next_update": 1},
        configuration=config, source_fingerprint=fingerprint)
    return model, optimizer, scheduler, path, config, fingerprint, receipt


@pytest.mark.parametrize("field,value", [("ordinary_rope_backend", "dao"),
    ("ordinary_pointwise_backend", "eager"), ("tile_backend", "eager"),
    ("selected_layers", [0]), ("fused_adam", True)])
def test_runtime_flag_mismatch_rejected_before_state_mutation(tmp_path, field, value):
    model, optimizer, scheduler, path, config, fp, _ = checkpoint_fixture(tmp_path)
    before = recovery.boundary_record(model, optimizer, scheduler, TrainingCounters(), {})
    config[field] = value
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, model, optimizer, scheduler=scheduler,
            configuration=config, source_fingerprint=fp)
    assert recovery.boundary_record(model, optimizer, scheduler, TrainingCounters(), {}) == before


def test_fused_optimizer_identity_rejected_before_state_mutation(tmp_path):
    model, _, _, path, config, fp, _ = checkpoint_fixture(tmp_path)
    optimizer = build_adamw(model, lr=1e-3, fused=True)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
    before = recovery.boundary_record(model, optimizer, scheduler, TrainingCounters(), {})
    with pytest.raises(ValueError, match="optimizer_descriptor"):
        load_training_checkpoint(path, model, optimizer, scheduler=scheduler,
            configuration=config, source_fingerprint=fp)
    assert recovery.boundary_record(model, optimizer, scheduler, TrainingCounters(), {}) == before


@pytest.mark.parametrize("mutation", ["checks", "empty", "count", "bytes", "path"])
def test_checkpoint_disposal_refuses_partial_failed_or_changed_evidence(tmp_path, mutation):
    *_, path, _, _, receipt = checkpoint_fixture(tmp_path)
    checks, updates = [{"passed": True}], 6
    if mutation == "checks": checks.append({"passed": False})
    if mutation == "empty": checks = []
    if mutation == "count": updates = 5
    if mutation == "bytes":
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    if mutation == "path": receipt["path"] += ".other"
    with pytest.raises(ValueError):
        recovery.verified_discard_checkpoint(path, receipt, checks, expected_updates=6, actual_updates=updates)
    assert path.exists()


def test_checkpoint_disposal_keeps_receipt_and_removes_only_verified_file(tmp_path):
    *_, path, _, _, receipt = checkpoint_fixture(tmp_path)
    other = tmp_path / "keep.pt"; other.write_bytes(b"do not remove")
    result = recovery.verified_discard_checkpoint(path, receipt, [{"passed": True}],
        expected_updates=6, actual_updates=6)
    assert not path.exists() and other.exists()
    assert result["sha256"] == receipt["sha256"] and result["deleted_after_success"]


class CPUReplayPlan(StaticFBTTraining):
    """Real tiny autograd with replay orchestration substituted for CUDA capture."""
    def capture(self, *, warmup, release_transient_cache):
        assert warmup == 10 and release_transient_cache
        self.initialize_gradients()
        assert self.graph is None
        self.graph = SimpleNamespace(reset=lambda: None)

    def backward(self, *, replay=False):
        if replay:
            assert self.graph is not None
            self.replay_calls += 1
        return super().backward(replay=False)


class NoNetworkTracker:
    def __init__(self, **kwargs):
        self.record = {"run_url": "CPU orchestration test", "status": "test"}
    def start(self, config): pass
    def log(self, *args, **kwargs): pass
    def finish(self, **kwargs): pass


def cpu_harness(monkeypatch, tmp_path):
    original_root = recovery.ROOT
    source = "scripts/olmo_graph_recovery.py"
    (tmp_path / "scripts").mkdir()
    (tmp_path / source).write_bytes((original_root / source).read_bytes())
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "SOURCES", (source,))
    monkeypatch.setattr(recovery, "require_container_gpu", lambda: {"scope": "CPU test only"})
    monkeypatch.setattr(recovery, "configure_determinism", lambda _: {})
    monkeypatch.setattr(recovery, "compiler_configuration", lambda: {})
    monkeypatch.setattr(recovery, "compiler_observations", lambda: {})
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: "test-commit")
    monkeypatch.setattr(recovery.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(recovery.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(recovery, "OnlineTracker", NoNetworkTracker)
    monkeypatch.setattr(recovery, "backend_context", lambda _: nullcontext())
    monkeypatch.setattr(recovery, "dependency_record", lambda *a, **k: {})
    monkeypatch.setattr(recovery, "check_dependencies", lambda _: True)
    monkeypatch.setattr(recovery, "validate_prepared_manifest", lambda _: {"checkpoint": {"sha256": "a" * 64}})
    monkeypatch.setattr(recovery, "load_native_tokenizer", lambda _: None)
    tiny = replace(OLMoConfig.tiny(), model_dim=80, num_heads=4, mlp_intermediate_size=128)
    monkeypatch.setattr(recovery, "load_native_state_dict", lambda _: OLMoForCausalLM(tiny).state_dict())
    monkeypatch.setattr(recovery, "selected_case", lambda name, **_: IntegrationCase(name,
        rt_layers=(0, 1), fbt=name == "combined", nextlat=name == "combined", batch_size=2, length=8))
    monkeypatch.setattr(recovery, "build_model", lambda state, case: build_model(state, case,
        device="cpu", model_config=tiny, chunk_size=128, backend="math"))
    monkeypatch.setattr(recovery, "set_arm", lambda *a: None)
    def optimizer(model, arm):
        opt = build_adamw(model, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.1, fused=False)
        return opt, build_warmup_scheduler(opt, warmup_updates=2)
    monkeypatch.setattr(recovery, "optimizer_for", optimizer)
    monkeypatch.setattr(recovery, "new_plan", lambda model, batch, mode, _: CPUReplayPlan(
        model, batch, mode=mode, config=LMTrainingConfig(precision="fp32")))
    def fixture(tokenizer, case, update):
        ids = (torch.arange(16).reshape(2, 8) + 3 * update) % 50 + 2
        valid = torch.ones_like(ids, dtype=torch.bool)
        docs = torch.arange(2)[:, None].expand_as(ids).clone()
        return NextLatBatch(ids, valid, docs)
    monkeypatch.setattr(recovery, "batch_for", fixture)
    return tmp_path / "run"


@pytest.mark.parametrize("case", ["rt", "combined"])
def test_real_cpu_checkpoint_roundtrip_through_both_rebuilt_branches(monkeypatch, tmp_path, case):
    output = cpu_harness(monkeypatch, tmp_path)
    report = recovery.main(["--case", case, "--output-dir", str(output)])
    assert report["status"] == "passed"
    assert report["physical_optimizer_updates"] == 6
    assert report["branch_physical_optimizer_updates"] == {"preparation": 2, "reference": 2, "restored": 2}
    assert report["reference_final"] == report["restored_final"]
    assert report["reference_records"] == report["restored_records"]
    assert report["boundary"]["cursor"]["next_update"] == 2
    assert report["restored_final"]["cursor"]["next_update"] == 4
    assert report["checkpoint_disposal"]["deleted_after_success"]
    assert not (output / "diagnostic-boundary.pt").exists()
    assert all(row["passed"] for row in report["checks"])


def test_corrupt_restored_model_fails_before_continuation_and_retains_checkpoint(monkeypatch, tmp_path):
    output = cpu_harness(monkeypatch, tmp_path)
    original = recovery.load_training_checkpoint
    def corrupt(path, model, optimizer, **kwargs):
        restored = original(path, model, optimizer, **kwargs)
        with torch.no_grad(): next(model.parameters()).flatten()[0].add_(1)
        return restored
    monkeypatch.setattr(recovery, "load_training_checkpoint", corrupt)
    with pytest.raises(AssertionError, match="restored_boundary"):
        recovery.main(["--case", "combined", "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["physical_optimizer_updates"] == 4
    assert (output / "diagnostic-boundary.pt").exists()
    assert "checkpoint_disposal" not in report


def test_output_must_be_persistent_project_local(tmp_path):
    with pytest.raises(SystemExit):
        recovery.parse_args(["--case", "rt", "--output-dir", str(tmp_path / "outside")])


def test_physical_update_is_persisted_when_scheduler_fails(monkeypatch, tmp_path):
    output = cpu_harness(monkeypatch, tmp_path)
    original = recovery.optimizer_for
    def factory(model, arm):
        optimizer, scheduler = original(model, arm)
        def fail():
            raise RuntimeError("injected post-Adam scheduler failure")
        scheduler.step = fail
        return optimizer, scheduler
    monkeypatch.setattr(recovery, "optimizer_for", factory)
    with pytest.raises(RuntimeError, match="scheduler failure"):
        recovery.main(["--case", "rt", "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["physical_optimizer_updates"] == 1
    assert report["branch_physical_optimizer_updates"]["preparation"] == 1
    assert report["preparation_records"] == []
    assert report["status"] == "failed"


def test_tracking_finish_failure_cannot_publish_passed_status(monkeypatch, tmp_path):
    output = cpu_harness(monkeypatch, tmp_path)
    class FailingTracker(NoNetworkTracker):
        def finish(self, **kwargs):
            raise RuntimeError("injected tracking failure")
    monkeypatch.setattr(recovery, "OnlineTracker", FailingTracker)
    with pytest.raises(RuntimeError, match="tracking failure"):
        recovery.main(["--case", "rt", "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["physical_optimizer_updates"] == 6
    assert report["tracking_error_type"] == "RuntimeError"
