"""Explicit revisions, gate qualification and safe evidence selection; no GPU/cloud."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import olmo_rt_author_report as report
from scripts import olmo_rt_author_retain as retain


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data)
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Evidence test",
        "-c", "user.email=evidence@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=root, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    for path in set(retain.PROJECT_FILES) | report.ESSENTIAL_SOURCES:
        write(root, path, "initial " + path)
    write(root, report.PROTOCOL, "frozen protocol")
    write(root, report.AUDIT, "frozen audit")
    write(root, report.DOCS + "/test-results.txt", "CPU tests passed")
    write(root, report.RUNTIME + "/final-gpu.log", "GPU idle")
    checkpoint = {"sha256": "a" * 64, "uri": "gs://fast-chunks/native", "generation": "42"}
    write(root, retain.ARTIFACT, {"checkpoint": checkpoint})
    write(root, retain.RECEIPT, {"checkpoint": checkpoint})
    monkeypatch.setattr(retain, "checkpoint_reference", lambda receipt, checkpoint: checkpoint)
    git(root, "add", "."); git(root, "commit", "-m", "initial")
    first = git(root, "rev-parse", "HEAD")
    write(root, "cdrm/pretrained/olmo_author.py", "repaired author implementation")
    write(root, report.AUDIT, "repaired frozen audit")
    git(root, "add", "."); git(root, "commit", "-m", "repaired")
    return SimpleNamespace(root=root, first=first, second=git(root, "rev-parse", "HEAD"), checkpoint=checkpoint)


def add_run(evidence, name, revision, *, status="passed", stage="verify", compatibility=True,
            preload=False, events=False):
    root = evidence.root
    directory = root / report.RUNTIME / name
    sources = {}
    for source in sorted(report.ESSENTIAL_SOURCES):
        snapshot = write(directory, "source-snapshot/" + source, git(root, "show", revision + ":" + source))
        # git() strips whitespace; fixture sources intentionally have no newline.
        sources[source] = report.digest(snapshot)
    protocol = write(directory, "protocol.md", git(root, "show", revision + ":" + report.PROTOCOL))
    audit = write(directory, "author-port-audit.md", git(root, "show", revision + ":" + report.AUDIT))
    checks = [{"name": name, "passed": True} for name in sorted(report.OPERATIONAL[stage])]
    if stage == "verify":
        checks.extend({"name": name + "_fp32_vs_native_scan", "passed": True}
                      for name in ("author_scan", "author", "native"))
        checks.append({"name": "author_vs_native_bf16_mixed", "gate": True, "passed": compatibility})
        checks.extend({"name": name + "_bf16_vs_fp32_diagnostic", "gate": False, "passed": False}
                      for name in ("author", "native"))
    config = {"stage": stage, "backend": "author", "layers": 1, "batch_size": 32 if stage == "capacity" else 1,
        "length": 512 if stage == "capacity" else 32, "precision": "bf16_mixed", "profile": False,
        "region_events": events, "native_arm": "both", "native_backward": "recompute",
        "author_precision": "author_legacy", "bwd_mlp_chunks": 4}
    raw = {"schema": report.SCHEMA, "status": status, "stage": "load" if preload else "complete",
        "finished_utc": "2026-09-23T00:00:00Z", "runtime_commit": revision, "configuration": config,
        "checks": [] if preload else checks, "physical_optimizer_updates": 0 if preload else 6 if stage == "verify" else 8,
        "source_hashes": sources, "protocol_sha256": report.digest(protocol), "audit_sha256": report.digest(audit)}
    if not preload:
        raw["checkpoint"] = evidence.checkpoint
    if status != "passed":
        raw["error"] = {"type": "RuntimeError", "message": "explicit retained failure"}
    if stage == "capacity" and not preload:
        timing = lambda count: {"wall_seconds": [2.] * count, "cuda_seconds": [1.9] * count,
            "median_wall_seconds": 2., "median_cuda_seconds": 1.9}
        raw.update(full_update=timing(5), forward_loss_backward=timing(3), preparation_records=[{}] * 3,
            timed_records=[{}] * 5, input_tokens_per_second=8192.,
            fixtures=[{"seed": 12, "shape": [32, 512, 32], "input_l2": 1.},
                      {"seed": 13, "shape": [32, 512, 32], "input_l2": 2.}], fixture_order="alternating two fixtures",
            region_timings=[{"forward_objective_seconds": .5, "backward_seconds": 1.3}] * 5 if events else [])
    path = write(directory, "report.json", raw)
    write(root, report.RUNTIME + "/" + name + ".log", "selected run")
    return path, raw


def select(evidence, names, **kwargs):
    summary = report.summarize(names, evidence.second, root=evidence.root, **kwargs)
    write(evidence.root, report.DOCS + "/summary.json", summary)
    return summary


def preflight(evidence):
    write(evidence.root, report.RUNTIME + "/event-preflight.json", {
        "schema": "olmo-author-event-preflight-v1", "status": "passed", "checks": [{"passed": True}]})
    write(evidence.root, report.RUNTIME + "/event-preflight.log", "standalone event test")


def test_early_failure_override_and_compatibility_failure_preserve_operational_success(evidence):
    add_run(evidence, "load-failed", evidence.first, status="failed", preload=True)
    add_run(evidence, "compatibility-failed", evidence.second, status="failed", compatibility=False)
    add_run(evidence, "all-gates-passed", evidence.second)
    summary = select(evidence, ["load-failed", "compatibility-failed", "all-gates-passed"],
        overrides={"load-failed": evidence.first})
    assert summary["statuses"] == {"failed": 2, "passed": 1}
    assert summary["physical_optimizer_updates"] == 12
    first, qualified, passed = summary["runs"]
    assert first["gate_groups"]["operational"]["complete_and_passed"] is False
    assert qualified["gate_groups"]["operational"]["complete_and_passed"] is True
    assert qualified["gate_groups"]["compatibility"]["complete_and_passed"] is False
    assert passed["gate_groups"]["diagnostic"]["all_observed_passed"] is False
    assert passed["status"] == "passed"  # A non-gating diagnostic is not a gate.
    _, _, members, checked = retain.collect_evidence(evidence.root)
    assert checked["physical_optimizer_updates"] == 12
    assert checked["source_pairs_checked"] == 3 * len(report.ESSENTIAL_SOURCES)
    assert checked["run_revisions"]["load-failed"] == evidence.first
    assert "runtime/load-failed/report.json" in members
    assert "cdrm/pretrained/olmo_author.py" in checked["current_source_differences"]["load-failed"]


@pytest.mark.parametrize("damage", ["source", "audit", "protocol", "revision", "unfinished", "passed_failed_gate"])
def test_report_validation_rejects_inconsistent_provenance_or_status(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second)
    if damage == "source":
        source = "cdrm/pretrained/olmo_author.py"
        snapshot = write(path.parent, "source-snapshot/" + source, "tampered")
        raw["source_hashes"][source] = report.digest(snapshot)
    elif damage in ("audit", "protocol"):
        filename = "author-port-audit.md" if damage == "audit" else "protocol.md"
        changed = write(path.parent, filename, "tampered")
        raw[damage + "_sha256"] = report.digest(changed)
    elif damage == "revision":
        raw["runtime_commit"] = evidence.first
    elif damage == "unfinished":
        raw["status"] = "running"
    else:
        raw["checks"][0]["passed"] = False
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["candidate"])


def test_successful_event_capacity_requires_preflight_and_records_region_medians(evidence):
    add_run(evidence, "capacity", evidence.second, stage="capacity", events=True)
    with pytest.raises(ValueError, match="preflight"):
        select(evidence, ["capacity"])
    preflight(evidence)
    summary = select(evidence, ["capacity"])
    assert summary["capacity"][0]["event_region_medians"] == {
        "forward_objective_seconds": .5, "backward_seconds": 1.3}
    _, _, members, _ = retain.collect_evidence(evidence.root)
    assert "runtime/event-preflight.json" in members
    assert "logs/event-preflight.log" in members


@pytest.mark.parametrize("damage", ["report", "summary_count", "summary_gate", "preflight"])
def test_retainer_rejects_changed_selected_evidence(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second)
    preflight(evidence)
    summary = select(evidence, ["candidate"])
    if damage == "report":
        raw["extra"] = "changed after selection"
        write(path.parent, "report.json", raw)
    elif damage == "summary_count":
        summary["physical_optimizer_updates"] += 1
    elif damage == "summary_gate":
        summary["runs"][0]["gate_groups"]["operational"]["complete_and_passed"] = False
    else:
        write(evidence.root, report.RUNTIME + "/event-preflight.json", {
            "schema": "olmo-author-event-preflight-v1", "status": "failed", "checks": [{"passed": False}]})
    write(evidence.root, report.DOCS + "/summary.json", summary)
    with pytest.raises(ValueError):
        retain.collect_evidence(evidence.root)


def test_explicit_allowlist_excludes_weights_credentials_and_unselected_runs(evidence):
    path, _ = add_run(evidence, "selected", evidence.second)
    add_run(evidence, "unselected", evidence.second)
    write(path.parent, "weights.safetensors", "excluded")
    write(path.parent, ".env", "excluded")
    write(path.parent, "wandb/private.json", "excluded")
    write(evidence.root, report.DOCS + "/unapproved.json", "excluded")
    select(evidence, ["selected"])
    _, _, members, _ = retain.collect_evidence(evidence.root)
    assert not any(any(part in name for part in ("unselected", "safetensors", ".env", "wandb", "unapproved"))
                   for name in members)


def test_oom_before_updates_has_no_capacity_claim(evidence):
    add_run(evidence, "oom", evidence.second, status="oom", stage="capacity", preload=True)
    summary = select(evidence, ["oom"])
    assert summary["physical_optimizer_updates"] == 0
    assert summary["capacity"][0]["input_tokens_per_second"] is None
    assert report.plot_capacity(summary, evidence.root / "unused-plot-dir") == []


def test_same_shape_comparison_rejects_different_recorded_fixture(evidence):
    add_run(evidence, "first", evidence.second, stage="capacity")
    path, raw = add_run(evidence, "second", evidence.second, stage="capacity")
    raw["fixtures"][0]["seed"] += 1
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError, match="comparison fixtures"):
        select(evidence, ["first", "second"])


def test_retainer_recomputes_derived_capacity_numbers(evidence):
    add_run(evidence, "capacity", evidence.second, stage="capacity")
    summary = select(evidence, ["capacity"])
    summary["capacity"][0]["input_tokens_per_second"] *= 2
    write(evidence.root, report.DOCS + "/summary.json", summary)
    with pytest.raises(ValueError, match="Derived summary"):
        retain.collect_evidence(evidence.root)


def test_failed_preflight_status_cannot_be_overridden_by_passing_check_flags(evidence):
    add_run(evidence, "capacity", evidence.second, stage="capacity", events=True)
    write(evidence.root, report.RUNTIME + "/event-preflight.json", {
        "schema": "olmo-author-event-preflight-v1", "status": "failed", "checks": [{"passed": True}]})
    with pytest.raises(ValueError, match="passing preflight"):
        select(evidence, ["capacity"])


def test_passing_verify_requires_raw_comparisons_as_well_as_graph_health(evidence):
    path, raw = add_run(evidence, "verify", evidence.second)
    raw["checks"] = [check for check in raw["checks"] if check["name"] in report.OPERATIONAL["verify"]]
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError, match="planned raw comparisons"):
        select(evidence, ["verify"])


def test_queue_selection_is_top_level_safe_and_does_not_duplicate_selected_logs(evidence):
    add_run(evidence, "capacity-selected", evidence.second)
    runtime = evidence.root / report.RUNTIME
    for name in ("run_capacity_queue.py", "capacity-first.json", "capacity-first.log", "capacity-later-r2.json"):
        write(runtime, name, "queue evidence")
    for name in ("unrelated.json", ".env", "capacity-secret.env.json", "capacity-first.pt"):
        write(runtime, name, "unselected")
    write(runtime, "capacity-subdir.json/nested.json", "never recurse")
    select(evidence, ["capacity-selected"])
    _, _, members, checked = retain.collect_evidence(evidence.root)
    assert set(checked["queue_artifacts"]) == {"run_capacity_queue.py", "capacity-first.json", "capacity-first.log",
        "capacity-later-r2.json", "capacity-selected.log"}
    assert "queue/run_capacity_queue.py" in members
    assert "queue/capacity-first.json" in members
    assert "logs/capacity-selected.log" in members
    assert "queue/capacity-selected.log" not in members
    assert not any("nested" in name or "unrelated" in name or ".env" in name or name.endswith(".pt") for name in members)


def test_queue_selection_rejects_matching_symlinks(tmp_path):
    target = write(tmp_path, "original.json", "data")
    (tmp_path / "capacity-first.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        list(retain.queue_artifacts(tmp_path))
