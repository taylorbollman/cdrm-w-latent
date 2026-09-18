import copy

import pytest
import torch

from scripts import rt_a5_bf16_report as report


@pytest.fixture
def contracts(monkeypatch):
    monkeypatch.setattr(report, "validate_original_contract", lambda value, variant: None)
    a = {"schema": "original", "precision": "fp32", "source_sha256": "old",
         "training_step": "original_step", "evaluation": "original_eval", "one_step_diagnostics": "original_diag",
         "model_config": {"recurrent_precision_policy": "legacy", "d_model": 512, "n_layers": 2},
         "batch_size": 1024, "seed": 1234, "optimizer": "unchanged"}
    b = copy.deepcopy(a)
    b.update(schema="rt-a5-bf16-training-v1", precision="bf16_mixed", source_sha256="new",
             precision_contract=copy.deepcopy(report.PRECISION_CONTRACT),
             training_step="scripts.rt_a5_bf16.train_step",
             evaluation="scripts.rt_a5_bf16.evaluate_arrays; unchanged A5Metrics on FP32 logits",
             one_step_diagnostics="scripts.rt_a5_bf16.evaluate_diagnostics")
    b["model_config"]["recurrent_precision_policy"] = "bf16_fp32_state"
    return a, b


def test_contract_permits_only_explicit_precision_changes(contracts):
    assert report.check_contracts(*contracts)["passed"]


@pytest.mark.parametrize("change", ["batch", "width", "optimizer", "seed", "policy", "precision_detail"])
def test_contract_rejects_unapproved_changes(contracts, change):
    a, b = contracts
    if change == "batch": b["batch_size"] = 2048
    if change == "width": b["model_config"]["d_model"] = 128
    if change == "optimizer": b["optimizer"] = "different"
    if change == "seed": b["seed"] = 42
    if change == "policy": b["model_config"]["recurrent_precision_policy"] = "legacy"
    if change == "precision_detail": b["precision_contract"]["parameters"] = "bfloat16"
    with pytest.raises(ValueError): report.check_contracts(a, b)


def test_cli_pairing_allows_output_labels_not_budget():
    a = {"output_dir": "old", "updates": 80000, "resume": None, "batch_size": 1024}
    b = {**a, "output_dir": "new", "stop_file": "STOP", "wandb_group": "new"}
    report.check_configuration(a, b)
    b["updates"] = 40000
    with pytest.raises(ValueError): report.check_configuration(a, b)


@pytest.mark.parametrize("changed", [False, True])
def test_initial_tensor_pairing_from_cpu_checkpoints(tmp_path, changed):
    packets = []
    initial = {"seed": 1234}
    state = {"model": {"weight": torch.ones(3)}, "initialization": initial,
             "optimizer": {"state": {}, "param_groups": [{"lr": 0.0001, "params": [0]}]},
             "optimizer_parameter_names": ["weight"], "order_chain": "same"}
    for name in ["base", "mixed"]:
        path = tmp_path / f"{name}.pt"
        value = copy.deepcopy(state)
        if changed and name == "mixed": value["model"]["weight"][0] = 2
        torch.save(value, path)
        packets.append({"checkpoints": {0: {"local_path": str(path)}}, "report": {"initialization": initial}})
    if changed:
        with pytest.raises(ValueError): report.initial_pairing(*packets)
    else:
        assert report.initial_pairing(*packets)["exact_model_tensors"] == 1
