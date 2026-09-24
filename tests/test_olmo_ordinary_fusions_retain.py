"""Selected evidence retention and generation-pinned verification without cloud access."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import olmo_ordinary_fusions_retain as retain
from test_olmo_ordinary_fusions_report import add_run, evidence, write


@pytest.fixture
def ready(evidence, monkeypatch):
    root = evidence.root
    for name in retain.PROJECT_FILES:
        if not (root / name).exists():
            write(root, name, "project support " + name)
    for name in retain.REQUIRED_DOCS:
        if name != "summary.json" and not (root / retain.report.DOCS / name).exists():
            write(root, retain.report.DOCS + "/" + name, "bounded report evidence")
    write(root, retain.report.RUNTIME + "/final-gpu.log", "GPU idle")
    evidence.checkpoint = {"sha256": "a" * 64, "uri": "gs://fast-chunks/fixture"}
    write(root, retain.ARTIFACT, {"checkpoint": evidence.checkpoint})
    write(root, retain.RECEIPT, {"checkpoint": evidence.checkpoint})
    monkeypatch.setattr(retain, "checkpoint_reference", lambda _receipt, checkpoint: checkpoint)
    evidence.selected = []
    return evidence


def run(ready, name, **kwargs):
    path, raw = add_run(ready, name, **kwargs)
    write(ready.root, retain.report.RUNTIME + "/" + name + ".log", "retained log")
    ready.selected.append((name, raw["runtime_commit"]))
    return path, raw


def summary(ready):
    selected = retain.report.summarize([name for name, _ in ready.selected], ready.second,
        overrides=dict(ready.selected), root=ready.root)
    write(ready.root, retain.report.DOCS + "/summary.json", selected)
    return selected


def qualify(path, raw):
    check = next(item for item in raw["checks"] if item["name"] == "same_state_candidate_vs_reference")
    metric = {"x": {"relative_l2": .001, "max_relative": .01, "bitwise_equal": False}}
    check.update(passed=False, finite=True, ownership_matches=True, counts_equal=True,
        losses=deepcopy(metric), outputs=deepcopy(metric), gradients=deepcopy(metric))
    raw["configuration"]["continue_after_compatibility_miss"] = True
    raw.update(status="failed", compatibility_miss_continued=True,
        numerical_compatibility_passed=False, operational_checks_passed=True,
        error={"type": "AssertionError", "message": "Numerical compatibility remains failed"})
    write(path.parent, "report.json", raw)


def test_mixed_revisions_failures_dependencies_and_exact_traces_are_retained(ready):
    run(ready, "early-failure", revision=ready.first, status="failed")
    run(ready, "dao-profile", arm="dao-rope", profile=True)
    summary(ready)
    _, checkpoint, members, checked = retain.collect_evidence(ready.root)
    assert checkpoint == ready.checkpoint
    assert checked["statuses"] == {"failed": 1, "passed": 1}
    assert checked["physical_optimizer_updates"] == 9
    assert checked["run_revisions"]["early-failure"] == ready.first
    assert "cdrm/pretrained/olmo_ordinary.py" in checked["current_source_differences"]["early-failure"]
    assert checked["dependency_files_checked"]["dao-profile"] == 2
    assert checked["profile_count"] == 2
    assert "runtime/dao-profile/operator-trace.json.gz" in members
    assert "runtime/dao-profile/full-step-trace.json.gz" in members
    assert "runtime/dao-profile/dependency-snapshot/flash_attn/layers/rotary.py" in members
    assert "runtime/early-failure/report.json" in members
    assert "project/scripts/olmo_ordinary_fusions_report.py" in members


def test_completed_operational_diagnostic_retains_failed_numerics_and_six_updates(ready):
    path, raw = run(ready, "qualified", stage="correctness", arm="dao-rope")
    qualify(path, raw)
    selected = summary(ready)
    _, _, _, checked = retain.collect_evidence(ready.root)
    assert checked["statuses"] == {"failed": 1}
    assert checked["physical_optimizer_updates"] == 6
    assert selected["runs"][0]["gate_groups"]["operational"]["complete_and_passed"]
    assert not selected["runs"][0]["gate_groups"]["compatibility"]["complete_and_passed"]


@pytest.mark.parametrize("damage", ["summary_updates", "summary_gates", "report", "source", "dependency", "trace", "log", "checkpoint"])
def test_changed_evidence_cannot_be_retained(ready, damage):
    path, raw = run(ready, "dao-profile", arm="dao-rope", profile=True)
    selected = summary(ready)
    if damage == "summary_updates":
        selected["physical_optimizer_updates"] += 1
    elif damage == "summary_gates":
        selected["runs"][0]["gate_groups"]["operational"]["complete_and_passed"] = False
    elif damage == "report":
        raw["wandb"]["run_url"] = "changed"
        write(path.parent, "report.json", raw)
    elif damage == "source":
        write(path.parent, "source-snapshot/cdrm/pretrained/olmo.py", "different source")
    elif damage == "dependency":
        write(path.parent, "dependency-snapshot/flash_attn/layers/rotary.py", "different dependency")
    elif damage == "trace":
        write(path.parent, "operator-trace.json.gz", b"different trace")
    elif damage == "log":
        (path.parent.parent / "dao-profile.log").unlink()
    elif damage == "checkpoint":
        write(ready.root, retain.ARTIFACT, {"checkpoint": {"sha256": "b" * 64}})
    write(ready.root, retain.report.DOCS + "/summary.json", selected)
    with pytest.raises(ValueError):
        retain.collect_evidence(ready.root)


def test_unselected_weights_secrets_logs_and_traces_are_excluded(ready):
    path, _ = run(ready, "plain")
    write(path.parent, "weights.safetensors", "unselected model")
    write(path.parent, ".env", "unselected secret")
    write(path.parent, "operator-trace.json.gz", b"unrequested trace")
    write(ready.root, retain.report.RUNTIME + "/other.log", "unselected log")
    write(ready.root, retain.report.RUNTIME + "/run_ordinary_queue.py", "explicit queue source")
    write(ready.root, retain.report.DOCS + "/throughput.pdf", b"report plot")
    summary(ready)
    _, _, members, checked = retain.collect_evidence(ready.root)
    assert checked["profile_count"] == 0
    assert not any(name.endswith(("weights.safetensors", ".env", "operator-trace.json.gz", "other.log")) for name in members)
    assert "queue/run_ordinary_queue.py" in members
    assert "report/throughput.pdf" in members


def test_queue_symlinks_and_symlinked_docs_are_rejected(ready):
    run(ready, "plain")
    summary(ready)
    runtime = ready.root / retain.report.RUNTIME
    (runtime / "run_ordinary_queue.py").symlink_to(runtime / "plain.log")
    with pytest.raises(ValueError, match="Queue artifact"):
        retain.collect_evidence(ready.root)
    (runtime / "run_ordinary_queue.py").unlink()
    docs = ready.root / retain.report.DOCS
    (docs / "extra.txt").symlink_to(docs / "results.md")
    with pytest.raises(ValueError, match="regular file"):
        retain.collect_evidence(ready.root)


def test_evidence_budget_is_enforced_before_archive_or_upload(ready, monkeypatch):
    run(ready, "plain")
    summary(ready)
    monkeypatch.setattr(retain, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="64 MiB"):
        retain.collect_evidence(ready.root)


@pytest.mark.parametrize("corruption", [None, "download", "server_md5", "server_sha"])
def test_upload_verifies_create_only_generation_and_downloaded_sha(tmp_path, corruption):
    path = write(tmp_path, "evidence.tar.gz", b"small retained archive")
    expected = retain.file_digest(path)

    class Blob:
        name = "evidence/evidence.tar.gz"
        generation = 42
        size = expected["size_bytes"]
        md5_hash = expected["md5_base64"]
        metadata = None

        def upload_from_filename(self, filename, *, if_generation_match, checksum):
            assert Path(filename) == path and if_generation_match == 0 and checksum == "md5"

        def reload(self):
            if corruption == "server_md5":
                self.md5_hash = "corrupted"
            if corruption == "server_sha":
                self.metadata["sha256"] = "b" * 64

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return b"corrupted download" if corruption == "download" else path.read_bytes()

    class Bucket:
        def blob(self, name):
            assert name == Blob.name
            return Blob()

    if corruption is not None:
        with pytest.raises(ValueError):
            retain.upload_verified(Bucket(), path, "evidence")
    else:
        result = retain.upload_verified(Bucket(), path, "evidence")
        assert result["sha256"] == expected["sha256"] and result["generation"] == "42"
