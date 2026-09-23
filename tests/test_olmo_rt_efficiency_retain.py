"""Exact trace retention and per-run frozen source provenance, without cloud access."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import olmo_rt_efficiency_retain as retain


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data))
    elif isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data)
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Retainer test",
        "-c", "user.email=retainer-test@example.invalid", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def trace_report(directory, *, status="passed"):
    path = write(directory, "operator-trace.json.gz", gzip.compress(b'{"traceEvents":[]}'))
    digest = retain.file_digest(path)
    return {"status": status, "configuration": {"profile": True}, "profile": {
        "trace_file": path.name, "trace_bytes": digest["size_bytes"], "trace_sha256": digest["sha256"]}}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    for name in set(retain.PROJECT_FILES) | retain.ESSENTIAL_SOURCES:
        write(root, name, "first version " + name)
    write(root, retain.PROTOCOL_RELATIVE, "first frozen protocol")
    write(root, retain.DOCS_RELATIVE + "/test-results.txt", "tests passed")
    write(root, retain.RUNTIME_RELATIVE + "/final-gpu.log", "GPU idle")
    checkpoint = {"sha256": "a" * 64, "uri": "gs://fast-chunks/native", "generation": "42"}
    write(root, retain.ARTIFACT_RELATIVE, {"checkpoint": checkpoint})
    write(root, retain.RECEIPT_RELATIVE, {"checkpoint": checkpoint})
    monkeypatch.setattr(retain, "checkpoint_reference", lambda _receipt, recorded: recorded)
    git(root, "add", ".")
    git(root, "commit", "-m", "first frozen runtime")
    first = git(root, "rev-parse", "HEAD")
    write(root, "cdrm/pretrained/olmo_tiled.py", "second version")
    write(root, retain.PROTOCOL_RELATIVE, "second frozen protocol")
    git(root, "add", ".")
    git(root, "commit", "-m", "second frozen runtime")
    second = git(root, "rev-parse", "HEAD")
    return SimpleNamespace(root=root, first=first, second=second, checkpoint=checkpoint, rows=[])


def add_run(evidence, name, revision, *, status="passed", profile=False):
    directory = evidence.root / retain.RUNTIME_RELATIVE / name
    sources = {}
    for source in sorted(retain.ESSENTIAL_SOURCES):
        path = write(directory, "source-snapshot/" + source,
            retain.git_bytes(evidence.root, revision, source))
        sources[source] = retain.file_digest(path)["sha256"]
    protocol = write(directory, "protocol.md", retain.git_bytes(evidence.root, revision, retain.PROTOCOL_RELATIVE))
    report = {"schema": retain.REPORT_SCHEMA, "status": status, "finished_utc": "2026-09-23T00:00:00Z",
        "checks": [{"name": "test", "passed": status == "passed"}], "checkpoint": evidence.checkpoint,
        "physical_optimizer_updates": 8 if status == "passed" else 2,
        "source_hashes": sources, "protocol_sha256": retain.file_digest(protocol)["sha256"],
        "configuration": {"profile": profile}}
    if status != "passed":
        report["error"] = {"type": "AssertionError", "message": "diagnostic failure"}
    if profile:
        report["profile"] = trace_report(directory, status=status)["profile"]
    path = write(directory, "report.json", report)
    write(evidence.root, retain.RUNTIME_RELATIVE + "/" + name + ".log", "selected run log")
    row = {"name": name, "runtime_commit": revision, "status": status,
        "report_path": path.relative_to(evidence.root).as_posix(), "report_sha256": retain.file_digest(path)["sha256"]}
    if profile:
        row["profile"] = deepcopy(report["profile"])
    evidence.rows.append(row)
    return path, report, row


def summary(evidence):
    reports = [json.loads((evidence.root / row["report_path"]).read_text()) for row in evidence.rows]
    return write(evidence.root, retain.DOCS_RELATIVE + "/summary.json", {
        "status": "completed", "runtime_commit": evidence.second, "runs": evidence.rows,
        "physical_optimizer_updates": sum(report["physical_optimizer_updates"] for report in reports),
        "source_pairs_checked": sum(len(report["source_hashes"]) for report in reports)})


def test_explicit_failures_and_per_run_revisions_retain_historical_sources_and_profile(evidence):
    add_run(evidence, "old-failed", evidence.first, status="failed")
    add_run(evidence, "new-passed-profile", evidence.second, profile=True)
    summary(evidence)
    _, _, members, validated = retain.collect_evidence(evidence.root)
    assert validated["statuses"] == {"failed": 1, "passed": 1}
    assert validated["physical_optimizer_updates"] == 10
    assert validated["source_pairs_checked"] == 2 * len(retain.ESSENTIAL_SOURCES)
    assert "cdrm/pretrained/olmo_tiled.py" in validated["current_source_differences"]["old-failed"]
    assert validated["current_source_differences"]["new-passed-profile"] == {}
    assert validated["profile_count"] == 1
    assert "runtime/new-passed-profile/operator-trace.json.gz" in members
    assert "runtime/old-failed/report.json" in members


def test_matching_report_and_snapshot_cannot_bypass_frozen_git_commit(evidence):
    path, report, row = add_run(evidence, "modified", evidence.first)
    source = "cdrm/pretrained/olmo_tiled.py"
    snapshot = write(path.parent, "source-snapshot/" + source, "changed snapshot")
    report["source_hashes"][source] = retain.file_digest(snapshot)["sha256"]
    write(path.parent, path.name, report)
    row["report_sha256"] = retain.file_digest(path)["sha256"]
    summary(evidence)
    with pytest.raises(ValueError, match="frozen commit"):
        retain.collect_evidence(evidence.root)


def test_summary_profile_must_match_selected_report(evidence):
    _, _, row = add_run(evidence, "profile", evidence.second, profile=True)
    row["profile"]["trace_bytes"] += 1
    summary(evidence)
    with pytest.raises(ValueError, match="summary profile"):
        retain.collect_evidence(evidence.root)


def test_unselected_trace_and_runtime_files_never_enter_archive(evidence):
    path, _, _ = add_run(evidence, "plain", evidence.second)
    trace_report(path.parent)
    write(path.parent, "weights.safetensors", "unselected")
    write(path.parent, ".env", "unselected")
    summary(evidence)
    _, _, members, validated = retain.collect_evidence(evidence.root)
    assert validated["profile_count"] == 0
    assert not any(name.endswith((".gz", ".safetensors", ".env")) for name in members)


@pytest.mark.parametrize("damage", ["missing", "bytes", "sha", "boolean_bytes", "unsafe_name",
    "symlink", "undeclared", "not_gzip"])
def test_trace_verification_rejects_invalid_declared_artifact(tmp_path, damage):
    report = trace_report(tmp_path)
    record = report["profile"]
    path = tmp_path / record["trace_file"]
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        record["trace_bytes"] += 1
    elif damage == "sha":
        record["trace_sha256"] = "a" * 64
    elif damage == "boolean_bytes":
        record["trace_bytes"] = True
    elif damage == "unsafe_name":
        record["trace_file"] = "../operator-trace.json.gz"
    elif damage == "symlink":
        target = tmp_path / "original.gz"
        path.rename(target)
        path.symlink_to(target)
    elif damage == "undeclared":
        report["configuration"]["profile"] = False
    elif damage == "not_gzip":
        path.write_bytes(b"plain text")
        digest = retain.file_digest(path)
        record.update(trace_bytes=digest["size_bytes"], trace_sha256=digest["sha256"])
    with pytest.raises(ValueError):
        retain.verify_operator_trace(report, tmp_path)


def test_requested_but_incomplete_profile_is_retained_only_as_failed_run(tmp_path):
    report = {"configuration": {"profile": True}, "status": "passed"}
    with pytest.raises(ValueError, match="missing"):
        retain.verify_operator_trace(report, tmp_path)
    report["status"] = "failed"
    assert retain.verify_operator_trace(report, tmp_path) is None


def test_valid_profile_verification_preserves_exact_compressed_bytes(tmp_path):
    report = trace_report(tmp_path)
    before = (tmp_path / report["profile"]["trace_file"]).read_bytes()
    assert retain.verify_operator_trace(report, tmp_path) == report["profile"]
    assert (tmp_path / report["profile"]["trace_file"]).read_bytes() == before
