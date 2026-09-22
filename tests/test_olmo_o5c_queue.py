"""Completed queues require the full frozen experiment and retained boundary."""
import copy
from types import SimpleNamespace

import pytest

from scripts.olmo_o5c_queue import validate_completed_arm
from scripts import olmo_o5c_train as train


@pytest.fixture
def completed():
    plan = {"total_updates": 2, "batch_row_prefix": [0, 3, 8],
        "batch_ce_prefix": [0, 10, 20], "batch_token_prefix": [0, 13, 28],
        "domain_prefixes": {"code": {"ce_positions": [0, 10, 20],
            "input_tokens": [0, 13, 28], "documents": [0, 3, 8]},
            "general": {"ce_positions": [0, 0, 0], "input_tokens": [0, 0, 0], "documents": [0, 0, 0]}}}
    config = {"source_checkpoint_sha256": "a"*64, "source_hashes": {"train.py": "b"*64},
        "data_manifest_sha256": "c"*64, "physical_batch_size": 2,
        "schedule": {"total_updates": 2, "arms": {"code": plan}},
        "trainable_layout": [{"name": "fusion", "numel": 4, "shape": [2, 2]}],
        "frozen_state_initial": {"native": "d"*64}}
    prefix = "gs://test/experiment"
    runtime = {"torch": "test"}
    record = {"optimizer_updates": 2, "path": "/removed/local/update-000002.pt",
        "sha256": "e"*64, "size_bytes": 100, "input_tokens": 28, "ce_positions": 20,
        "storage": {"uri": prefix+"/code/update-000002.pt", "sha256": "e"*64,
            "size_bytes": 100, "generation": "123", "md5_base64": "abc=",
            "verification": "GCS generation, size, server MD5 and SHA256 metadata verified"}}
    report = {"schema": "olmo-o5c-arm-v1", "status": "completed", "arm": "code", "finished_utc": "now",
        "configuration": copy.deepcopy(config), "source_fingerprint": {
            "checkpoint_sha256": config["source_checkpoint_sha256"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "code": copy.deepcopy(config["source_hashes"]),
            "data_manifest_sha256": config["data_manifest_sha256"], "runtime": runtime.copy()},
        "storage_prefix": prefix, "counters": {"optimizer_updates": 2, "microbatches": 5,
            "input_tokens": 28, "ce_positions": 20, "documents": 8, "latent_pairs": 0, "kl_triples": 0},
        "data_cursor": 2, "domain_counts": {domain: {key: values[-1] for key, values in fields.items()}
            for domain, fields in plan["domain_prefixes"].items()},
        "trainable_layout": copy.deepcopy(config["trainable_layout"]),
        "frozen_state_initial": config["frozen_state_initial"].copy(),
        "frozen_state_final": config["frozen_state_initial"].copy(), "checkpoints": [record]}
    return report, config, prefix, runtime


def test_completed_arm_accepts_retained_deleted_local_with_exact_partial_microbatch_count(completed):
    report, config, prefix, runtime = completed
    before = copy.deepcopy(report)
    validate_completed_arm(report, config, "code", prefix+"/", runtime)
    assert report == before


@pytest.mark.parametrize("case", ["source_code", "source_endpoint", "data", "runtime", "config",
    "cursor", "microbatches", "count_bool", "domains", "unfrozen", "trainable", "missing_final",
    "duplicate_final", "nonlatest_final", "receipt_absent", "receipt_sha", "receipt_uri",
    "receipt_size", "receipt_generation", "receipt_md5", "receipt_verification", "short_sha",
    "boolean_size", "wrong_name", "checkpoint_ce"])
def test_completed_arm_rejects_incomplete_or_wrong_lineage(completed, case):
    report, config, prefix, runtime = completed
    record = report["checkpoints"][0]
    if case == "source_code": report["source_fingerprint"]["code"]["train.py"] = "wrong"
    elif case == "source_endpoint": report["source_fingerprint"]["checkpoint_sha256"] = "wrong"
    elif case == "data": report["source_fingerprint"]["data_manifest_sha256"] = "wrong"
    elif case == "runtime": report["source_fingerprint"]["runtime"] = {}
    elif case == "config": report["configuration"]["physical_batch_size"] = 3
    elif case == "cursor": report["data_cursor"] = 1
    elif case == "microbatches": report["counters"]["microbatches"] = 4
    elif case == "count_bool": report["counters"]["kl_triples"] = False
    elif case == "domains": report["domain_counts"]["general"]["ce_positions"] = 10
    elif case == "unfrozen": report["frozen_state_final"]["native"] = "changed"
    elif case == "trainable": report["trainable_layout"].append({"name": "native"})
    elif case == "missing_final": report["checkpoints"] = []
    elif case == "duplicate_final": report["checkpoints"].append(copy.deepcopy(record))
    elif case == "nonlatest_final": report["checkpoints"].append({"optimizer_updates": 1})
    elif case == "receipt_absent": del record["storage"]
    elif case == "receipt_sha": record["storage"]["sha256"] = "f"*64
    elif case == "receipt_uri": record["storage"]["uri"] = prefix+"/mixed/update-000002.pt"
    elif case == "receipt_size": record["storage"]["size_bytes"] = 99
    elif case == "receipt_generation": record["storage"]["generation"] = ""
    elif case == "receipt_md5": record["storage"]["md5_base64"] = ""
    elif case == "receipt_verification": record["storage"]["verification"] = "upload attempted"
    elif case == "short_sha": record["sha256"] = record["storage"]["sha256"] = "a"
    elif case == "boolean_size": record["size_bytes"] = record["storage"]["size_bytes"] = True
    elif case == "wrong_name": record["path"] = "/other/update-000001.pt"
    elif case == "checkpoint_ce": record["ce_positions"] -= 1
    with pytest.raises(ValueError): validate_completed_arm(report, config, "code", prefix, runtime)


def test_model_configuration_guard_checks_native_config_and_exact_layout(monkeypatch):
    model = SimpleNamespace(backbone=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"width": 32})))
    layout = [{"name": "fusion", "numel": 4, "shape": [2, 2]}]
    monkeypatch.setattr(train, "trainable_layout", lambda model: copy.deepcopy(layout))
    config = {"model_config": {"width": 32}, "trainable_layout": copy.deepcopy(layout)}
    train.validate_model_configuration(model, config)
    config["model_config"]["width"] = 64
    with pytest.raises(ValueError): train.validate_model_configuration(model, config)
    config["model_config"]["width"] = 32
    config["trainable_layout"][0]["shape"] = [4, 1]
    with pytest.raises(ValueError): train.validate_model_configuration(model, config)
