"""CPU-only adversarial identity checks; no fresh scientific data/init generation."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import cdrm_precision_confirm as confirm
from cdrm.mad_data import MadDataset


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(), "Use the explicitly GPU-disabled CPU container"
    torch.set_num_threads(1)


def roles():
    return json.loads((confirm.LINEAGE/"decisions/future-confirmation-roles-v1.json").read_text())


def decision():
    arm = confirm.ARMS["bf16_attention_fp32"]
    return {"schema": "cdrm-precision-candidate-freeze-v1", "status": "frozen_for_confirmation",
            "candidate": {"arm": arm.name, "specification": confirm.json_value(asdict(arm))},
            "original_criteria": {"sha256": confirm.OLD_CONTRACT}, "reference_contract": {"sha256": confirm.LOCAL_CONTRACT}}


def fixtures():
    return {"schema": "cdrm-precision-fresh-fixtures-v1", "status": "generated",
            "candidate_freeze_sha256": "candidate", "roles_sha256": "roles",
            "corpus": {"seed": 925903, "examples": 192, "split": "dev"},
            "roles": [{"name": name, "example_offset": spec[0], "example_stop": spec[0]+64,
                       "checkpoint": {"sha256": spec[1] or "synthetic-initialization"}}
                      for name,spec in confirm.ROLE_SPECS.items()]}


def synthetic_checkpoint(role):
    """Shape-valid adversarial fixture of constants, not the seed7502 experiment."""
    _, _, seed, updates, precision = confirm.ROLE_SPECS[role]
    model = {name: torch.full(shape, .01) for name,shape in confirm.expected_parameter_shapes().items()}
    cfg = asdict(confirm.config("tiled", "fp32"))
    identity = {"source_sha256": {}, "model_config": cfg, "test_only": True}
    state = {pid: {"step": torch.tensor(float(updates)), "exp_avg": torch.zeros_like(value), "exp_avg_sq": torch.ones_like(value)}
             for pid,value in enumerate(model.values())} if updates else {}
    return {"format": confirm.FORMAT if updates else confirm.INIT_FORMAT, "model": model,
            "model_config": cfg, "completed_updates": updates, "initialization": {"seed": seed},
            "precision": precision, "identity": identity, "identity_sha256": confirm.json_digest(identity),
            "optimizer": {"state": state, "param_groups": [{"params": list(range(45)), "lr": .0005,
                "betas": (.9,.98), "eps": 1e-8, "weight_decay": 0., "foreach": False, "fused": False}]}}


def test_candidate_requires_exact_active_policy_and_unchanged_criteria():
    good = decision()
    assert confirm.validate_decision(good, roles()).name == "bf16_attention_fp32"
    bad = copy.deepcopy(good); bad["candidate"]["specification"]["fp32_head"] = True
    with pytest.raises(ValueError, match="specification"):
        confirm.validate_decision(bad, roles())
    bad = copy.deepcopy(good); bad["candidate"]["arm"] = "bf16_bypass"
    with pytest.raises(ValueError, match="complete active"):
        confirm.validate_decision(bad, roles())
    bad = copy.deepcopy(good); bad["original_criteria"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="criteria"):
        confirm.validate_decision(bad, roles())


def test_role_and_fixture_binding_rejects_reassignment_and_prefreeze_generation():
    confirm.validate_roles(roles())
    changed = roles(); changed["roles"][2]["initialization_seed"] = 7501
    with pytest.raises(ValueError, match="initialization seed"):
        confirm.validate_roles(changed)
    value = fixtures()
    assert set(confirm.validate_fixture_roles(value,"candidate","roles")) == set(confirm.ROLE_SPECS)
    with pytest.raises(ValueError, match="prior candidate"):
        confirm.validate_fixture_roles(value,"another-candidate","roles")
    changed = copy.deepcopy(value); changed["roles"][1]["example_offset"] = 0
    with pytest.raises(ValueError, match="offset"):
        confirm.validate_fixture_roles(changed,"candidate","roles")
    changed = copy.deepcopy(value); changed["roles"][0]["checkpoint"]["sha256"] = value["roles"][1]["checkpoint"]["sha256"]
    with pytest.raises(ValueError, match="trained checkpoint"):
        confirm.validate_fixture_roles(changed,"candidate","roles")


def test_json_anchor_and_actual_array_hashes_are_checked(tmp_path):
    path = tmp_path/"freeze.json"; path.write_text('{"frozen": true}\n')
    assert confirm.checked_json(path,confirm.digest(path))["frozen"]
    with pytest.raises(ValueError, match="SHA"):
        confirm.checked_json(path,"wrong")
    ids = np.zeros((192,256), dtype=np.int64)
    labels = np.full_like(ids,-100); labels[:,-96:] = 0
    manifest = {"seed":925903,"task":"selective-copying","vocab_size":16,
                "task_overrides":{"num_tokens_to_copy":96},"manifest_sha256":"manifest"}
    data = MadDataset(ids,labels,labels.copy(),[{} for _ in range(192)],manifest)
    corpus = {"dataset_sha256":data.sha256,"manifest_sha256":"manifest"}
    rows = confirm.validate_fixture_roles(fixtures(),"candidate","roles")
    for row in rows.values():
        row["batch_sha256"] = data.take(slice(row["example_offset"],row["example_stop"])).sha256
    confirm.validate_dataset(data,corpus,rows)
    data.input_ids[0,0] = 1
    with pytest.raises(ValueError, match="arrays/manifest"):
        confirm.validate_dataset(data,corpus,rows)


def test_initial_checkpoint_rejects_carried_moments_wrong_seed_and_zero_adapters():
    value = synthetic_checkpoint("distinct_initialization")
    assert confirm.validate_checkpoint(value,"distinct_initialization")["initialization"]
    bad = copy.deepcopy(value); bad["initialization"]["seed"] = 7500
    with pytest.raises(ValueError, match="seed"):
        confirm.validate_checkpoint(bad,"distinct_initialization")
    bad = copy.deepcopy(value); bad["optimizer"]["state"][0] = {}
    with pytest.raises(ValueError, match="empty Adam"):
        confirm.validate_checkpoint(bad,"distinct_initialization")
    bad = copy.deepcopy(value); bad["model"]["cdrm.deep_adapter.weight"].zero_()
    with pytest.raises(ValueError, match="nonzero adapters"):
        confirm.validate_checkpoint(bad,"distinct_initialization")


def test_trained_checkpoint_rejects_optimizer_or_architecture_drift():
    value = synthetic_checkpoint("bf16_trained_u1000")
    assert confirm.validate_checkpoint(value,"bf16_trained_u1000")["completed_updates"] == 1000
    bad = copy.deepcopy(value); bad["model_config"]["cdrm_lambda"] = 0.
    bad["identity_sha256"] = confirm.json_digest(bad["identity"])
    with pytest.raises(ValueError, match="architecture"):
        confirm.validate_checkpoint(bad,"bf16_trained_u1000")
    bad = copy.deepcopy(value); bad["optimizer"]["state"][0]["step"] = torch.tensor(999.)
    with pytest.raises(ValueError, match="step"):
        confirm.validate_checkpoint(bad,"bf16_trained_u1000")
    bad = copy.deepcopy(value); bad["optimizer"]["param_groups"][0]["params"][1] = 0
    with pytest.raises(ValueError, match="ownership"):
        confirm.validate_checkpoint(bad,"bf16_trained_u1000")


@pytest.mark.parametrize("role",["fp32_trained_u1000","bf16_trained_u1000"])
def test_actual_retained_trained_checkpoint_passes_preflight_shape_and_moment_checks(role):
    trajectory = "fp32" if role.startswith("fp32") else "bf16"
    path = Path(".runtime/cdrm-tiled-pilot/20260907T212606Z/seed7500")/f"cdrm-{trajectory}-u1000/u1000.pt"
    assert confirm.digest(path) == confirm.ROLE_SPECS[role][1]
    value = torch.load(path,map_location="cpu",weights_only=False)
    assert confirm.validate_checkpoint(value,role)["canonical_parameter_tensors"] == 45


def test_comparison_matches_archived_failed_packet_without_relaxing_any_flag():
    path = Path(".runtime/cdrm-tiled-pilot/20260907T212606Z/num/seed7500-u1000-fp32")
    report = json.loads((path/"report.json").read_text())
    payload = torch.load(path/"tensors.pt",map_location="cpu",weights_only=False)
    checkpoint = torch.load(report["checkpoint"]["path"],map_location="cpu",weights_only=False)
    r,a = "tiled_fp32","tiled_bf16"
    row = confirm.compare_pair(payload["packets"][r],payload["packets"][a],payload["side_packets"][r],payload["side_packets"][a],
                               payload["steps"][r],payload["steps"][a],checkpoint,fp32=False)
    assert row == report["comparisons"]["tiled_bf16_vs_tiled_fp32"]
    assert row["machine_screens_pass"] is False
    assert row["adam"]["guardrail_pass"] is False
