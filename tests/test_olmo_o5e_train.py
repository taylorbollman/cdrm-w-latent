"""Launch and bookkeeping contracts for the ordinary continuation control."""
import copy
import json

import pytest

from cdrm.pretrained.artifacts import sha256_file
from scripts.olmo_o5e_train import validate_configuration, validate_preflight, health_failure


def configuration():
    return {
        "fbt_enabled": False, "num_passes": 1, "beta": 0., "gamma": 1.,
        "rt_layers": [], "nextlat_enabled": False, "native_backbone_frozen": False,
        "data_plan_arm": "mixed", "objective": "single_ordinary_ce", "lr": 1e-5,
        "warmup_updates": 50, "precision": "bf16_mixed", "physical_batch_size": 16,
        "eval_batch_size": 8, "compile": False, "cuda_graphs": False, "distributed": False,
        "schedule": {"total_updates": 512, "ce_per_update": 8192, "total_ce": 4194304,
                     "arms": {"mixed": {}}},
        "source_hashes": {"example": "a" * 64}, "runtime": {"gpu": "fixture"},
    }


@pytest.mark.parametrize("key,value", [
    ("fbt_enabled", True), ("num_passes", 2), ("beta", 1.), ("rt_layers", [0]),
    ("nextlat_enabled", True), ("native_backbone_frozen", True),
    ("objective", "two_ce_losses"), ("lr", 1e-4), ("warmup_updates", 100),
    ("data_plan_arm", "code"), ("precision", "fp32"), ("physical_batch_size", 32),
])
def test_rejects_changed_experiment_before_launch(key, value):
    config = configuration()
    validate_configuration(config)
    config[key] = value
    with pytest.raises(ValueError):
        validate_configuration(config)


def test_rejects_budget_or_second_arm():
    original = configuration()
    for key, value in (("total_updates", 513), ("ce_per_update", 4096),
                       ("total_ce", 100), ("arms", {"mixed": {}, "code": {}})):
        config = copy.deepcopy(original)
        config["schedule"][key] = value
        with pytest.raises(ValueError):
            validate_configuration(config)


def test_launch_requires_passed_preflight_for_exact_config_bytes(tmp_path):
    config = configuration()
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(config))
    report_path = tmp_path / "report.json"
    report = {"status": "passed", "schema": "olmo-o5e-preflight-v1",
              "configuration_sha256": sha256_file(path), "source_hashes": config["source_hashes"],
              "runtime": config["runtime"]}
    report_path.write_text(json.dumps(report))
    validate_preflight(config, path, config["runtime"])
    for key, value in (("status", "running"), ("source_hashes", {}), ("runtime", {}),
                       ("configuration_sha256", "b" * 64)):
        report_path.write_text(json.dumps({**report, key: value}))
        with pytest.raises(ValueError):
            validate_preflight(config, path, config["runtime"])
    report_path.write_text(json.dumps(report))
    path.write_text(json.dumps(config, indent=2))
    with pytest.raises(ValueError):
        validate_preflight(config, path, config["runtime"])


def test_ordinary_health_gate_checks_both_single_pass_domains():
    def values(code, wiki):
        return {"dev": {"passes": [{"mean_nll": code}]},
                "retention_dev": {"passes": [{"mean_nll": wiki}]}}
    initial = values(2., 3.)
    assert not health_failure(values(3.5, 4.5), initial, 1.5)
    assert health_failure(values(3.51, 3.), initial, 1.5)
    assert health_failure(values(2., 4.51), initial, 1.5)
