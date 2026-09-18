"""Saved-evidence checks; no model inference or GPU use."""
import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_report as reporter


CORE = ("cdrm/rt_nextlat_tasks.py", "scripts/rt_a5_window.py", "scripts/rt_a5_nextlat.py",
        "scripts/rt_a5_nextlat_variant.py", "recurrent-transformer/olmo/model.py")


def test_module_cli_help_executes_entry_point_before_any_input_access():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.rt_nextlat_a5_fuzzy_report", "--help"],
        cwd=reporter.ROOT, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    for option in ("--train", "--a5-control", "--fuzzy-control", "--fuzzy-data", "--output", "--wandb"):
        assert option in completed.stdout
    assert "Traceback" not in completed.stderr


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def a5_metric(update, *, role="ood_dev", rows=102400, correct=100):
    length = {"dev": 12, "ood_dev": 36}[role]
    accuracy = correct / rows
    return {"task": "a5", "role": role, "update": update, "rows": rows,
            "evaluated_rows": rows, "length": length, "tokens": rows*length,
            "ce": 1.0, "latent": .1, "token_accuracy": accuracy,
            "final_state_accuracy": accuracy, "whole_word_exact_match": accuracy,
            "whole_word_correct": correct,
            "isolated_state_accuracy": [accuracy]*length,
            "cumulative_prefix_exactness": [accuracy]*length,
            "per_position_ce": [1.0]*length,
            "scope": "full" if rows == 102400 else "subset"}


def fuzzy_metric(update):
    return {"task": "fuzzy", "role": "dev", "update": update, "examples": 1280,
            "evaluated_rows": 1280, "answer_tokens": 100, "answer_correct": 99,
            "answer_accuracy": .99, "answer_motif_exact_match": .98,
            "sequence_exact_match": .8, "first_value_token_accuracy": .99,
            "terminal_probe_accuracy": .97, "scope": "full"}


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "fuzzy-data"
    baseline = {"query_ignoring_answer_prefix": {"answer_accuracy": .4063673095108462}}
    write_json(data / "manifest.json", {"splits": {"dev": {"baselines": baseline}}})
    return data


def fixture(directory, data, *, mode="mixed", endpoint=3):
    old = mode == "fuzzy"
    tasks = ["fuzzy"] if old else (["a5"] if mode == "a5-only" else ["a5", "fuzzy"])
    config = {"backbone": {"d_model": 128, "n_heads": 16, "mlp_hidden_size": 512}}
    initialization = {"model_parameter_sha256": "1"*64, "predictor_sha256": "2"*64}
    fuzzy = {"train": {"manifest_sha256": "3"*64}, "dev": {"manifest_sha256": "4"*64}}
    data_sha = reporter.sha(data / "manifest.json")
    identity = fuzzy if old else {"a5": {"manifest_sha256": "5"*64}}
    if mode == "mixed":
        identity["fuzzy"] = {**fuzzy, "preparation_manifest_sha256": data_sha}
    schema = "rt-nextlat-fuzzy-training-v1" if old else "rt-nextlat-a5-fuzzy-training-v1"
    sources = {}
    for name in CORE:
        path = directory / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen model fixture\n")
        sources[name] = reporter.sha(path)
    write_json(directory / "source-manifest.json", sources)
    write_json(directory / "data-identity.json", identity)
    write_json(directory / "model-config.json", config)
    contract = {"schema": schema, "model_config": config, "initialization": initialization,
                "configuration_file_sha256": reporter.sha(directory / "model-config.json"),
                "data_sha256": reporter.json_sha(identity), "source_sha256": reporter.json_sha(sources),
                "optimizer": {"lr": 1e-4}, "runtime": {"precision": "fp32"}}
    streams = {task: {"train_rows": 12800 if task == "fuzzy" else 800000,
                       "length": 400 if task == "fuzzy" else 12, "order_seed": 42} for task in tasks}
    if old:
        contract.update(streams["fuzzy"], batch_size=128, preparation_manifest_sha256=data_sha)
    else:
        contract.update(mode=mode, streams=streams, batch_per_task=128)
    checkpoints, evaluations, history = [], [], []
    for update in range(endpoint+1):
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"saved weights {mode} {update}".encode())
        checkpoint = {"completed_updates": update, "path": str(path),
                      "sha256": reporter.sha(path), "bytes": path.stat().st_size}
        checkpoints.append(checkpoint)
        if not update:
            continue
        metrics = []
        if "a5" in tasks:
            metrics += [a5_metric(update, role=role) for role in ("dev", "ood_dev")]
        if "fuzzy" in tasks:
            metrics.append(fuzzy_metric(update))
        for metric in metrics:
            if old:
                metric.pop("task")
                metric.pop("role")
            else:
                metric["checkpoint"] = {key: checkpoint[key] for key in ("path", "sha256", "completed_updates")}
        evaluations += metrics
        chains = {task: hashlib.sha256(f"{task} {update}".encode()).hexdigest() for task in tasks}
        row = {"update": update, "loss": 1.1, "tasks": {task: {"ce": 1.0, "latent": .1} for task in tasks}}
        row.update({"order_chain": chains["fuzzy"]} if old else {"order_chains": chains})
        history.append(row)
    report = {"schema": schema, "status": "stopped" if old else "complete", "contract": contract,
              "completed_updates": endpoint, "requested_endpoint": 10000 if old else endpoint,
              "start_update": 0, "parent_checkpoint": None, "initialization": initialization,
              "parameter_count": 479616, "checkpoints": checkpoints, "evaluations": evaluations,
              "confirmation_evaluated": False, "latent_rollout_evaluated": False,
              "wandb": {"run_url": "https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/fixture"}}
    write_json(directory / "report.json", report)
    (directory / "history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
    return report


def test_gate_uses_integer_count_full_pool_and_reports_early_timing():
    small = a5_metric(500, rows=4096, correct=1)
    zero = a5_metric(1000, correct=0)
    positive = a5_metric(3000, correct=1)
    late = a5_metric(5000, correct=100)
    gate = reporter.a5_gate([late, small, positive, zero])
    assert gate["passed"] and gate["first_positive_update"] == 3000
    assert gate["first_positive_whole_word_correct"] == 1
    assert gate["positive_within_first_few_thousand"] is True
    assert reporter.a5_gate([small, zero])["passed"] is False
    assert reporter.a5_gate([a5_metric(10500)])["passed"] is False
    broken = copy.deepcopy(positive)
    broken["whole_word_correct"] = 2
    with pytest.raises(ValueError, match="whole_word_correct"):
        reporter.a5_gate([broken])


def test_endpoint_requires_every_task_at_same_retained_checkpoint(tmp_path, inputs):
    path = tmp_path / "mixed"
    report = fixture(path, inputs)
    report["evaluations"][-1]["update"] = 2
    write_json(path / "report.json", report)
    with pytest.raises(ValueError, match="Duplicate|Endpoint"):
        reporter.load_run(path)
    report = fixture(path, inputs)
    report["evaluations"][-1]["checkpoint"]["sha256"] = "9"*64
    write_json(path / "report.json", report)
    with pytest.raises(ValueError, match="different checkpoint"):
        reporter.load_run(path)


def test_checkpoint_tampering_and_running_status_are_rejected(tmp_path, inputs):
    path = tmp_path / "a5"
    report = fixture(path, inputs, mode="a5-only")
    (path / "checkpoints/step-000003.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        reporter.load_run(path)
    report["status"] = "running"
    write_json(path / "report.json", report)
    with pytest.raises(ValueError, match="completed or explicitly stopped"):
        reporter.load_run(path)


def test_subset_confirmation_prefers_full_curve_but_retains_both_rows(tmp_path, inputs):
    path = tmp_path / "a5"
    report = fixture(path, inputs, mode="a5-only")
    subset = a5_metric(1, rows=4096, correct=1)
    subset["checkpoint"] = report["evaluations"][1]["checkpoint"]
    report["evaluations"].insert(0, subset)
    write_json(path / "report.json", report)
    run = reporter.load_run(path)
    assert len(run["evaluations"]) == 7
    selected = reporter.selected_evaluations(run["evaluations"])
    assert len(selected) == 6
    assert all(row["rows"] == 102400 for row in selected)


def test_control_matching_stops_at_real_endpoint_and_checks_order(tmp_path, inputs):
    primary, control = tmp_path / "mixed", tmp_path / "fuzzy"
    fixture(primary, inputs, endpoint=3)
    fixture(control, inputs, mode="fuzzy", endpoint=2)
    a, b = reporter.load_run(primary), reporter.load_run(control)
    match = reporter.compare_control(a, b, "fuzzy")
    assert match["control_endpoint"] == 2
    assert [row["update"] for row in match["comparisons"]] == [1, 2]
    assert match["matched_order_updates"] == 2
    b["history"][1]["order_chain"] = "0"*64
    with pytest.raises(ValueError, match="data order differs"):
        reporter.compare_control(a, b, "fuzzy")


def test_control_rejects_configuration_data_and_initialization_changes(tmp_path, inputs):
    primary, control = tmp_path / "mixed", tmp_path / "a5"
    fixture(primary, inputs)
    fixture(control, inputs, mode="a5-only")
    a, b = reporter.load_run(primary), reporter.load_run(control)
    assert reporter.compare_control(a, b, "a5")["matched"]
    for section in ("model_config", "initialization"):
        changed = copy.deepcopy(b)
        changed["report"]["contract"][section]["different"] = True
        with pytest.raises(ValueError, match=section):
            reporter.compare_control(a, changed, "a5")
    b["data_identity"]["a5"]["manifest_sha256"] = "9"*64
    with pytest.raises(ValueError, match="task data"):
        reporter.compare_control(a, b, "a5")


def test_mixed_report_renders_bound_common_budgets_without_extending_fuzzy(tmp_path, inputs):
    primary, a5, fuzzy = [tmp_path / name for name in ("mixed", "a5", "fuzzy")]
    fixture(primary, inputs, endpoint=3)
    fixture(a5, inputs, mode="a5-only", endpoint=3)
    fixture(fuzzy, inputs, mode="fuzzy", endpoint=2)
    output = tmp_path / "report"
    args = SimpleNamespace(train=primary, a5_control=a5, fuzzy_control=fuzzy,
                           fuzzy_data=inputs, output=output)
    summary = reporter.build_report(args)
    assert summary["endpoint"] == 3
    assert {row["update"] for row in summary["final"].values()} == {3}
    assert summary["controls"]["fuzzy"]["endpoint"] == 2
    assert max(row["update"] for row in summary["matched_controls"]["fuzzy"]["comparisons"]) == 2
    assert len(summary["figures"]) == 5
    assert all((output / f"{name}.png").stat().st_size > 1000 for name in summary["figures"])
    assert "FUZZY control actual endpoint: **2 updates**" in (output / "report.md").read_text()
    assert reporter.read_json(output / "evidence.json")["checkpoint"]["sha256"] == summary["checkpoint"]["sha256"]


def test_a5_only_accepts_prior_fuzzy_as_reference_not_matched_task(tmp_path, inputs):
    primary, fuzzy = tmp_path / "a5", tmp_path / "fuzzy"
    fixture(primary, inputs, mode="a5-only")
    fixture(fuzzy, inputs, mode="fuzzy", endpoint=2)
    summary = reporter.build_report(SimpleNamespace(train=primary, a5_control=None,
        fuzzy_control=fuzzy, fuzzy_data=inputs, output=tmp_path / "report"))
    assert summary["matched_controls"] == {}
    assert summary["controls"]["fuzzy"]["role"].startswith("prior single-task reference")


def continuation_fixture(tmp_path, inputs, *, different_boundary_bytes=False):
    parent, child = tmp_path / "interrupted", tmp_path / "resumed"
    prior = fixture(parent, inputs, endpoint=4)
    prior["status"] = "running"
    prior["requested_endpoint"] = 5
    for index, metric in enumerate(prior["evaluations"]):
        if metric["update"] == 3 and metric["task"] == "a5":
            subset = a5_metric(3, role=metric["role"], rows=4096)
            subset["checkpoint"] = metric["checkpoint"]
            prior["evaluations"][index] = subset
        if metric["update"] == 4:
            # Valid JSON beyond the restored boundary must never enter metrics.
            metric["ce"] = 777
    write_json(parent / "report.json", prior)
    history = [json.loads(line) for line in (parent / "history.jsonl").read_text().splitlines()]
    history[-1]["loss"] = 999
    (parent / "history.jsonl").write_bytes(
        "".join(json.dumps(row)+"\n" for row in history).encode()+b"\x00\x00partial interrupted tail")
    final = fixture(child, inputs, endpoint=5)
    final.update(start_update=3, parent_checkpoint={key: prior["checkpoints"][3][key] for key in ("path", "sha256")})
    final["checkpoints"] = [record for record in final["checkpoints"] if record["completed_updates"] >= 3]
    final["evaluations"] = [metric for metric in final["evaluations"] if metric["update"] > 3]
    if different_boundary_bytes:
        cp = child / "checkpoints/step-000003.pt"
        cp.write_bytes(cp.read_bytes()+b"different serialization metadata")
        final["checkpoints"][0].update(sha256=reporter.sha(cp), bytes=cp.stat().st_size)
    write_json(child / "report.json", final)
    history = [json.loads(line) for line in (child / "history.jsonl").read_text().splitlines()]
    (child / "history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history if row["update"] > 3))
    return parent, child, prior, final


def boundary_proof_fixture(parent, child, prior, final):
    fields = ("model", "optimizer", "rng", "contract", "initialization", "order_chains", "next_cursors",
              "completed_updates", "examples_seen", "optimizer_parameter_names", "schema", "finite_state")
    parent_history = json.loads((parent / "history.jsonl").read_bytes().splitlines()[2])
    proof = {"schema": "rt-nextlat-a5-fuzzy-recovery-boundary-state-audit-v1",
             "status": "passed", "passed": True, "completed_updates": 3,
             "entire_packet_exact": True, "differences": [], "field_equality": dict.fromkeys(fields, True),
             "checkpoints": {"parent": {k: prior["checkpoints"][3][k] for k in ("path", "sha256", "bytes")},
                             "child": {k: final["checkpoints"][0][k] for k in ("path", "sha256", "bytes")}},
             "matching_examples_seen": {"a5": 384, "fuzzy": 384},
             "matching_next_cursors": {task: {"absolute_example_offset": 384, "epoch": 0, "position": 384}
                                       for task in ("a5", "fuzzy")},
             "matching_order_chains": parent_history["order_chains"]}
    path = child / "recovery-boundary-state-audit.json"
    write_json(path, proof)
    return path, proof


def test_exact_resume_stitches_only_committed_parent_prefix_and_preserves_corrupt_tail(tmp_path, inputs):
    parent, child, _, _ = continuation_fixture(tmp_path, inputs)
    before = (parent / "history.jsonl").read_bytes()
    run = reporter.load_run(child)
    assert [row["update"] for row in run["history"]] == [1, 2, 3, 4, 5]
    assert run["history"][3]["loss"] == 1.1
    assert all(row.get("ce") != 777 for row in run["evaluations"])
    assert (parent / "history.jsonl").read_bytes() == before
    assert len(run["lineage"]) == 2
    audit = run["lineage"][0]["history"]
    assert audit["used_through_update"] == 3 and audit["discarded_valid_update_range"] == [4, 4]
    assert audit["discarded_valid_records_before_first_invalid"] == 1
    assert audit["first_invalid_tail_byte_offset"] > audit["used_prefix_bytes"]
    assert run["lineage"][0]["discarded_post_boundary_evaluations"] == 3
    assert run["lineage"][1]["boundary_checkpoint_copy"]["byte_identical"] is True
    assert run["checkpoints"][3]["verified_local_path"] == str(parent / "checkpoints/step-000003.pt")
    with pytest.raises(ValueError, match="completed or explicitly stopped"):
        reporter.load_run(parent)


def test_resume_rejects_corruption_before_boundary_or_missing_child_update(tmp_path, inputs):
    parent, child, _, _ = continuation_fixture(tmp_path, inputs)
    original = (parent / "history.jsonl").read_bytes()
    lines = original.splitlines(keepends=True)
    lines[1] = b"\x00corrupt committed update\n"
    (parent / "history.jsonl").write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="Invalid committed history"):
        reporter.load_run(child)
    (parent / "history.jsonl").write_bytes(original)
    (child / "history.jsonl").write_text((child / "history.jsonl").read_text().splitlines()[1]+"\n")
    with pytest.raises(ValueError, match="exactly the completed optimizer updates"):
        reporter.load_run(child)


def test_resume_rejects_wrong_parent_hash_and_changed_contract(tmp_path, inputs):
    _, child, _, final = continuation_fixture(tmp_path, inputs)
    correct_hash = final["parent_checkpoint"]["sha256"]
    final["parent_checkpoint"]["sha256"] = "0"*64
    write_json(child / "report.json", final)
    with pytest.raises(ValueError, match="Parent checkpoint hash mismatch"):
        reporter.load_run(child)
    final["parent_checkpoint"]["sha256"] = correct_hash
    final["contract"]["streams"]["a5"]["order_seed"] += 1
    write_json(child / "report.json", final)
    with pytest.raises(ValueError, match="Exact continuation contract"):
        reporter.load_run(child)


def test_changed_boundary_serialization_requires_hash_bound_full_state_proof(tmp_path, inputs):
    parent, child, prior, final = continuation_fixture(tmp_path, inputs, different_boundary_bytes=True)
    with pytest.raises(ValueError, match="exact full-state audit receipt is required"):
        reporter.load_run(child)
    proof_path, proof = boundary_proof_fixture(parent, child, prior, final)
    run = reporter.load_run(child)
    boundary = run["lineage"][-1]
    assert boundary["boundary_checkpoint_copy"]["byte_identical"] is False
    assert boundary["boundary_checkpoint_copy"]["exact_state_verified"] is True
    assert boundary["boundary_state_audit"]["sha256"] == reporter.sha(proof_path)
    proof["checkpoints"]["child"]["sha256"] = "0"*64
    write_json(proof_path, proof)
    with pytest.raises(ValueError, match="different checkpoint bytes"):
        reporter.load_run(child)


def test_stitched_control_comparison_and_report_keep_parent_budget_points(tmp_path, inputs):
    _, child, _, _ = continuation_fixture(tmp_path, inputs)
    control = tmp_path / "fuzzy-control"
    fixture(control, inputs, mode="fuzzy", endpoint=2)
    run, old = reporter.load_run(child), reporter.load_run(control)
    compared = reporter.compare_control(run, old, "fuzzy")
    assert [row["update"] for row in compared["comparisons"]] == [1, 2]
    summary = reporter.build_report(SimpleNamespace(train=child, a5_control=None,
        fuzzy_control=control, fuzzy_data=inputs, output=tmp_path / "report"))
    assert len(summary["lineage"]) == 2
    assert summary["endpoint"] == 5
    assert {row["update"] for row in summary["final"].values()} == {5}
    assert "Exact checkpoint recovery" in (tmp_path / "report/report.md").read_text()
