"""Queue completion cannot masquerade as a verified final comparison."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import olmo_o5b_finish as finish


def args(tmp_path):
    result = SimpleNamespace(**{name: tmp_path / name for name in
        ("data", "preflight", "runs", "retention_output", "report_dir")}, prefix="gs://test/pilot", prior_data_manifest=None)
    for name in ("runs", "retention_output", "report_dir"):
        getattr(result, name).mkdir()
    return result


def queue(a, status):
    (a.runs / "queue.json").write_text(json.dumps({"status": status}))


def receipt(a, status="verified"):
    (a.retention_output / "upload-result.json").write_text(json.dumps({
        "schema": "olmo-o5b-evidence-receipt-v1", "status": status, "phase": "final", "receipt_object": {"uri": "gs://receipt"}}))


def test_wait_then_strict_report_then_retention_and_copy(tmp_path):
    a = args(tmp_path); queue(a, "running")
    calls, waits = [], []
    def sleep(seconds):
        waits.append(seconds); queue(a, "completed")
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        if command[1].endswith("_retain.py"): receipt(a)
    result = finish.finish(a, sleep=sleep, run=run)
    assert waits == [20] and result["status"] == "completed"
    assert [call[1] for call in calls] == ["scripts/olmo_o5b_report.py", "scripts/olmo_o5b_retain.py"]
    assert (a.report_dir / "final-storage-receipt.json").read_bytes() == (a.retention_output / "upload-result.json").read_bytes()


def test_missing_queue_is_waited_for_without_starting_report(tmp_path):
    a = args(tmp_path)
    def sleep(seconds):
        assert seconds == 20; queue(a, "stopped")
    result = finish.finish(a, sleep=sleep, run=lambda *a, **kw: pytest.fail("Must not run report"))
    assert result["status"] == "queue_stopped"


@pytest.mark.parametrize("status", ["failed", "stopped", "unknown"])
def test_noncompleted_queue_never_builds_or_retains(tmp_path, status):
    a = args(tmp_path); queue(a, status)
    result = finish.finish(a, run=lambda *a, **kw: pytest.fail("Unexpected report"))
    assert result["status"] == "queue_stopped" and result["queue_status"] == status
    assert not (a.report_dir / "final-storage-receipt.json").exists()


def test_strict_report_failure_stops_before_retention(tmp_path):
    a = args(tmp_path); queue(a, "completed")
    calls = []
    def run(command, **kwargs):
        calls.append(command); raise subprocess.CalledProcessError(1, command)
    with pytest.raises(subprocess.CalledProcessError): finish.finish(a, run=run)
    assert len(calls) == 1
    assert json.loads((a.runs / "finish-status.json").read_text())["status"] == "failed"


def test_unverified_retention_cannot_mark_finished(tmp_path):
    a = args(tmp_path); queue(a, "completed"); receipt(a, "failed")
    with pytest.raises(ValueError, match="verified"): finish.finish(a, run=lambda *a, **kw: None)
    assert not (a.report_dir / "final-storage-receipt.json").exists()


def test_upload_retry_preserves_validated_report_bytes(tmp_path, monkeypatch):
    from scripts import olmo_o5b_retain as retain
    a = args(tmp_path); queue(a, "completed"); receipt(a)
    (a.retention_output / "evidence.tar.gz").write_bytes(b"existing archive")
    (a.report_dir / "final-comparison.json").write_text("existing report")
    checked, calls = [], []
    monkeypatch.setattr(retain, "validate_final_comparison", lambda *a, **kw: checked.append(True))
    result = finish.finish(a, run=lambda command, **kw: calls.append(command))
    assert result["status"] == "completed" and checked == [True]
    assert [call[1] for call in calls] == ["scripts/olmo_o5b_retain.py"]
    assert (a.report_dir / "final-comparison.json").read_text() == "existing report"
