"""Host-side restart protocol checks; CUDA/NCCL acceptance uses two processes."""
import copy
import json

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_campaign_restart import (
    SCHEMA, checkpoint_disk_preflight, compare_continuation, cursor_for,
    parse_args, validate_cursor, validate_reference,
)


def arguments(phase="write", **overrides):
    values = {"phase": phase, "scale": "tiny", "output-dir": "/tmp/restart-report",
              "checkpoint-dir": "/tmp/restart-checkpoint", **overrides}
    if phase == "resume":
        values.update({"reference-report": "/tmp/write-report.json", "reference-sha256": "a" * 64,
                       "expected-manifest-sha256": "b" * 64})
    return [entry for key, value in values.items() for entry in ("--" + key, str(value))]


def test_parser_requires_separate_write_and_pinned_resume_phases():
    assert parse_args(arguments()).length == 8
    assert parse_args(arguments(scale="pretrained")).length == 16
    assert parse_args(arguments("resume")).phase == "resume"
    with pytest.raises(SystemExit):
        parse_args(arguments(**{"reference-report": "/tmp/foreign.json"}))
    with pytest.raises(SystemExit):
        parse_args(arguments(length=1024))
    with pytest.raises(SystemExit):
        parse_args(arguments(arm="NR"))
    with pytest.raises(SystemExit):
        parse_args(arguments("resume")[:-2])
    with pytest.raises(SystemExit):
        parse_args(arguments("resume") + ["--reference-sha256", "bad"])


def test_restart_cursor_pins_changing_accumulation_plan_and_recipe():
    for rank in (0, 1):
        for index, count in enumerate((2, 3, None)):
            cursor = cursor_for("a" * 64, rank, index, length=16)
            assert cursor["microbatches_per_rank"] == count
            assert cursor["fixture_update"] == index + 1
            validate_cursor(cursor, "a" * 64, rank, TrainingCounters(optimizer_updates=index), length=16)
            for key, value in (("rank", 1-rank), ("next_update", index+1), ("recipe_sha256", "b" * 64),
                               ("length", 32), ("microbatches_per_rank", 7)):
                changed = {**cursor, key: value}
                with pytest.raises(ValueError, match="cursor differs"):
                    validate_cursor(changed, "a" * 64, rank, TrainingCounters(optimizer_updates=index), length=16)
    for rank, index in ((2, 0), (True, 0), (0, 3), (0, True)):
        with pytest.raises(ValueError):
            cursor_for("a" * 64, rank, index, length=16)


def test_bitwise_restart_gates_include_gradients_adam_rng_data_and_metrics():
    expected = {"input": {"noise": "1", "data": "2"}, "raw_gradients": {"w": "3"},
                "metrics": {"gradient_norm_before_clip": 1., "counts": {"ce": 4}},
                "boundary": {"state": {"optimizer": {"step": 2, "exp_avg": "4"},
                                       "model": {"weight": "5"}}, "rng": "6", "cursor": {"next_update": 2}},
                "rng_draws": {"torch_local": "7", "explicit_data": "8"}}
    assert compare_continuation(expected, copy.deepcopy(expected))["passed"]
    for key in expected:
        changed = copy.deepcopy(expected)
        changed[key] = {}
        check = compare_continuation(changed, expected)
        assert not check["passed"] and not check["bitwise_checks"][key]
        del changed[key]
        assert not compare_continuation(changed, expected)["passed"]
    changed = copy.deepcopy(expected)
    changed["metrics"]["gradient_norm_before_clip"] += 1e-12
    assert not compare_continuation(changed, expected)["passed"]
    changed = copy.deepcopy(expected)
    changed["boundary"]["state"]["optimizer"]["step"] += 1
    assert not compare_continuation(changed, expected)["passed"]


def test_reference_validation_rejects_incomplete_foreign_or_modified_evidence(tmp_path):
    path = tmp_path / "report.json"
    reference = {"schema": SCHEMA, "phase": "write", "status": "passed", "passed": True,
                 "scale": "tiny", "arm": "NFR", "length": 8, "sources": {"runtime.py": "x"},
                 "checkpoint": {"manifest_sha256": "b" * 64},
                 "continuation": [{"rank": 0}, {"rank": 1}], "saved_boundaries": [{}, {}]}
    kwargs = dict(scale="tiny", arm="NFR", length=8, expected_manifest_sha256="b" * 64,
                  sources={"runtime.py": "x"})
    path.write_text(json.dumps(reference))
    digest = sha256_file(path)
    assert validate_reference(path, digest, **kwargs) == reference
    path.write_text(json.dumps(reference) + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        validate_reference(path, digest, **kwargs)
    for key, value in (("schema", "unknown"), ("phase", "resume"), ("status", "running"),
                       ("passed", False), ("scale", "pretrained"), ("arm", "B"), ("length", 16),
                       ("sources", {"runtime.py": "y"}), ("checkpoint", {"manifest_sha256": "c" * 64}),
                       ("continuation", [{}]), ("saved_boundaries", [{}])):
        path.write_text(json.dumps({**reference, key: value}))
        with pytest.raises(ValueError):
            validate_reference(path, sha256_file(path), **kwargs)


def test_checkpoint_preflight_does_not_overwrite_and_includes_adam_space(tmp_path):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    destination = tmp_path / "new" / "checkpoint"
    record = checkpoint_disk_preflight(destination, model)
    assert record["estimated_required_bytes"] == 2**30 + 6 * 4 * 3 + 2 * 4
    assert not destination.exists()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        checkpoint_disk_preflight(destination, model)
