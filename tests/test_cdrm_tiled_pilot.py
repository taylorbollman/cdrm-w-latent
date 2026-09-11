"""CPU tests of pilot ancestry, pairing, evaluation isolation and epoch recovery."""
import copy
import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cdrm_tiled_pilot as pilot


@pytest.fixture(scope="module")
def initial():
    common = pilot.common
    model, initialization = common.cpu_initial_model(7501)
    optimizer = common.optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    cfg = dataclasses.asdict(model.config)
    identity = {"model_config": cfg, "source_sha256": common.sources(),
                "preset_sha256": common.file_digest(common.PRESET), "optimizer": pilot.OPTIMIZER,
                "schedule": pilot.SCHEDULE, "seeds": {"shuffle": pilot.SEEDS["shuffle"]},
                "initialization": initialization}
    return {"format": common.INIT_FORMAT, "identity": identity, "identity_sha256": common.json_digest(identity),
            "model_config": cfg, "model": common.cpu_tree(model.state_dict()), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "initialization": initialization,
            "completed_updates": 0, "completed_epochs": 0, "batch_in_epoch": 0}


def test_three_arms_keep_seed7501_backbone_and_only_two_adapters_differ(initial):
    pilot.validate_initial(initial, pilot.common.PRESET)
    models = {arm: pilot.build_arm_model(initial, arm, device="cpu")[0] for arm in pilot.ARMS}
    seq = models["seq-fp32"]
    assert not seq.config.cdrm_enabled and getattr(seq, "cdrm", None) is None
    assert pilot.common.state_digest(seq.state_dict()) == initial["initialization"]["backbone_initialization_sha256"]
    for arm, model in models.items():
        for name, weight in seq.state_dict().items():
            assert torch.equal(weight, model.state_dict()[name])
        optimizer, scheduler = pilot.optimizer_and_scheduler(model, initial, arm)
        assert not optimizer.state and scheduler.state_dict() == initial["scheduler"]
        assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))
        assert len({id(p) for p in optimizer.param_groups[0]["params"]}) == len(list(model.parameters()))
    assert set(models["cdrm-fp32"].state_dict()) - set(seq.state_dict()) == pilot.ADAPTERS
    assert models["cdrm-bf16"].config.cdrm_precision_policy == "bf16_fp32_state"


def test_initial_weights_and_configuration_are_checked(initial):
    bad = copy.deepcopy(initial)
    bad["model"][next(iter(bad["model"]))].flatten()[0] += 1
    with pytest.raises(ValueError, match="initialization weights"):
        pilot.validate_initial(bad, pilot.common.PRESET)
    bad = copy.deepcopy(initial)
    bad["model_config"]["n_layers"] = 6
    bad["identity_sha256"] = pilot.common.json_digest(bad["identity"])
    with pytest.raises(ValueError, match="architecture"):
        pilot.validate_initial(bad, pilot.common.PRESET)


def identity_payload(source=None):
    cfg = {"cdrm_enabled": True}
    identity = {"source_sha256": source or {}, "model_config": cfg, "precision": "fp32", "arm": "cdrm-fp32",
                "execution_contract": {"inductor_cache_directory": "/retained/cache"}, "schedule": pilot.SCHEDULE,
                "origin": {"sha256": "original-parent"}}
    return {"format": pilot.FORMAT, "pilot_schema": pilot.PILOT_SCHEMA, "identity": identity,
            "identity_sha256": pilot.common.json_digest(identity), "model_config": cfg, "precision": "fp32",
            "completed_updates": 1000, "run_target_updates": 1000}


def test_resume_allows_only_target_extension_and_explicit_short_recovery():
    payload = identity_payload()
    pilot.validate_resume_target(payload, payload["identity"], 2500)
    payload["completed_updates"] = 190
    with pytest.raises(ValueError, match="shorten"):
        pilot.validate_resume_target(payload, payload["identity"], 210)
    pilot.validate_resume_target(payload, payload["identity"], 210, recovery=True)
    changed = copy.deepcopy(payload["identity"])
    changed["execution_contract"]["inductor_cache_directory"] = "/cold/cache"
    with pytest.raises(ValueError, match="identity changed"):
        pilot.validate_resume_target(payload, changed, 1000)


def test_resume_checks_source_mutation_and_legacy_scope(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("constant = 1\n")
    payload = identity_payload({str(source): pilot.common.file_digest(source)})
    pilot.validate_resume_target(payload, payload["identity"], 2500)
    source.write_text("constant = 2\n")
    with pytest.raises(RuntimeError, match="source identity changed"):
        pilot.validate_resume_target(payload, payload["identity"], 2500)
    old = identity_payload()
    old.pop("pilot_schema")
    old["completed_updates"] = 100
    pilot.validate_legacy_parent(old, old["identity"], "cdrm-fp32")
    with pytest.raises(ValueError, match="original CDRM"):
        pilot.validate_legacy_parent(old, old["identity"], "seq-fp32")
    changed = copy.deepcopy(old["identity"])
    changed["precision"] = "bf16_fp32_state"
    with pytest.raises(ValueError, match="original CDRM"):
        pilot.validate_legacy_parent(old, changed, "cdrm-bf16")


class SyntheticIndexDataset:
    """Test-only stand-in retaining the real 12800-example permutation contract."""
    def __len__(self):
        return 12800

    def take(self, indices):
        return SimpleNamespace(sha256=pilot.common.state_digest(np.asarray(indices)))


def toy_system():
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9, .98), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    return model, optimizer, scheduler


def toy_training(model, optimizer, scheduler, target, restored=None):
    common = pilot.common
    data = SyntheticIndexDataset()
    if restored:
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        common.restore_rng(restored["rng"])
    completed = restored["completed_updates"] if restored else 0
    epochs, position = pilot.cursor_at(completed)
    history = copy.deepcopy(restored["history"]) if restored else []
    development = copy.deepcopy(restored["development"]) if restored else {}
    records = {}
    while completed < target:
        indices = pilot.batch_indices(completed, len(data), pilot.SEEDS["shuffle"])
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.rand(2, 3)).square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({"update": completed + 1, "epoch": epochs + 1, "batch_in_epoch": position,
                        "learning_rate": optimizer.param_groups[0]["lr"], "native_loss": loss.item(),
                        "indices_sha256": common.state_digest(indices), "batch_sha256": data.take(indices).sha256,
                        "seconds": .001})
        completed += 1
        epochs, position = pilot.advance_cursor(epochs, position, scheduler)
        if completed % 200 == 0:
            development[str(completed)] = {"native": {"ce": loss.item()}, "answer": {"ce": loss.item()}}
        if completed in (190, 210, target):
            records[completed] = common.cpu_tree({"format": pilot.FORMAT, "identity": {"same": "child"},
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": common.rng_state(), "completed_updates": completed, "completed_epochs": epochs,
                "batch_in_epoch": position, "history": history, "development": development,
                "next_batch_sha256": data.take(pilot.batch_indices(completed, len(data), pilot.SEEDS["shuffle"])).sha256,
                "ancestry": {"original": "retained"}, "model_config": {"same": True}, "precision": "fp32",
                "initialization": {"seed": 7501}, "ablation_development": {}})
    return records


def test_exact_recovery_crosses_epoch_boundary_and_preserves_next_shuffle():
    torch.manual_seed(901)
    uninterrupted = toy_training(*toy_system(), 210)
    restored = toy_training(*toy_system(), 210, restored=uninterrupted[190])[210]
    result = pilot.compare_pilot_recovery(uninterrupted[210], restored)
    assert result["bitwise_state_and_nontiming_metrics_equal"] and not result["differences"]
    pilot.validate_position(restored, SyntheticIndexDataset(), pilot.SEEDS["shuffle"])
    assert restored["completed_epochs"] == 1 and restored["batch_in_epoch"] == 10
    assert restored["history"][199]["learning_rate"] == 5e-4
    assert restored["history"][200]["learning_rate"] < 5e-4
    assert restored["history"][199]["indices_sha256"] != restored["history"][200]["indices_sha256"]
    bad = copy.deepcopy(restored)
    bad["optimizer"]["state"][0]["exp_avg"].flatten()[0] += .1
    assert not pilot.compare_pilot_recovery(uninterrupted[210], bad)["bitwise_state_and_nontiming_metrics_equal"]


def test_epoch_cursor_validation_rejects_lr_and_data_position_drift():
    torch.manual_seed(902)
    payload = toy_training(*toy_system(), 210)[210]
    for mutate, message in ((lambda p: p.update(batch_in_epoch=11), "cursor"),
                            (lambda p: p.update(next_batch_sha256="wrong"), "next batch"),
                            (lambda p: p["history"][200].update(learning_rate=5e-4), "learning-rate"),
                            (lambda p: p["history"][0].update(indices_sha256="wrong"), "data order")):
        bad = copy.deepcopy(payload)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            pilot.validate_position(bad, SyntheticIndexDataset(), pilot.SEEDS["shuffle"])


def test_lambda_zero_eval_restores_weights_optimizer_gradients_modes_and_rng(initial, monkeypatch):
    model, _ = pilot.build_arm_model(initial, "cdrm-fp32", device="cpu")
    optimizer, _ = pilot.optimizer_and_scheduler(model, initial, "cdrm-fp32")
    calls = []
    def evaluate(candidate, data, precision):
        candidate.eval()
        calls.append(candidate.config.cdrm_lambda)
        torch.rand(1)
        return {"native": {"ce": 2.}, "answer": {"ce": 2.}, "seconds": .1}
    monkeypatch.setattr(pilot.common, "evaluate", evaluate)
    before = pilot.common.state_digest(pilot.common.rng_state())
    result = pilot.evaluate_preserving_state(model, optimizer, None, "fp32", ablate=True)
    assert calls == [.01, 0.] and model.config.cdrm_lambda == .01 and model.training
    assert result["training_state_unchanged"] and set(result) == {"active", "lambda_zero", "training_state_unchanged"}
    assert pilot.common.state_digest(pilot.common.rng_state()) == before


def test_cli_enforces_bounded_targets_fixed_eval_and_required_recovery_input():
    base = ["--arm", "seq-fp32", "--initial-checkpoint", "init.pt", "--data-root", "data", "--output-dir", "new"]
    for extra in (["--stop-updates", "2501"], ["--stop-updates", "1000", "--eval-every", "250"],
                  ["--stop-updates", "210", "--reference-final", "u0210.pt"]):
        with pytest.raises(SystemExit):
            pilot.parse_args(base + extra)
    args = pilot.parse_args(base + ["--stop-updates", "1000"])
    assert args.save_updates == [190, 210, 1000, 2500]
