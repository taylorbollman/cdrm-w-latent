"""Small multi-run archives preserve failures without promoting their source."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import olmo_f2_retain as retain
from scripts.olmo_tiled_retain import (O1_CHECKPOINT_URI, O1_CHECKPOINT_GENERATION,
    O1_CHECKPOINT_MD5, CHECKPOINT_SIZE, CHECKPOINT_SHA256)


def write(root, name, value="evidence"):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


@pytest.fixture
def evidence(tmp_path):
    project, docs, runtime = (tmp_path/name for name in ("project", "docs", "runs"))
    core_files = {"cdrm/pretrained/olmo_tiled.py", "scripts/olmo_f2_health_capacity.py", "scripts/olmo_f2_observe.py"}
    graph_files = {"cdrm/pretrained/olmo_tiled.py", "scripts/olmo_f2_graph_probe.py"}
    for name in retain.EXTRA_PROJECT_FILES | core_files | graph_files:
        write(project, name)
    references = {}
    for name, files in retain.SNAPSHOT_FILES.items():
        references[name] = {"files": [{"file": filename} for filename in sorted(files)]}
        for filename in files | {"README.md", "manifest.json"}:
            write(project, f"cdrm/pretrained/{name}/{filename}")
    for name in ("protocol.md", "results.md", "assessment.md", "test-results.txt"):
        write(docs, name)
    reference = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
                 "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "md5_base64": O1_CHECKPOINT_MD5}
    receipt = write(tmp_path, "o1-storage-receipt.json", {"schema": "olmo-o1-storage-receipt-v1",
                    "status": "verified", "objects": [reference]})
    checkpoint = {"path": "native/model.safetensors", "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}
    common = {"status": "passed", "finished_utc": "now", "checkpoint": checkpoint,
        "wandb": {"status": "synced", "run_url": "https://wandb.ai/taylorbollman/test/runs/id"},
        "protocol_sha256": retain.file_digest(docs/"protocol.md")["sha256"]}
    reports, directories = [], []
    for stage in ("health", "capacity", "checkpoint"):
        report = copy.deepcopy(common) | {"schema": retain.CORE_SCHEMA, "config": {"stage": stage},
            "source_hashes": {name: retain.file_digest(project/name)["sha256"] for name in core_files},
            "rows": [{"passed": True, "status": "measured" if stage == "capacity" else "passed"}]}
        directory = runtime/f"f2-{stage}-01"
        write(directory, "report.json", report); write(directory, "config.json", report["config"])
        reports.append(report); directories.append(directory)
    graph = copy.deepcopy(common) | {"schema": retain.GRAPH_SCHEMA, "configuration": {"length": 32},
        "stage": "complete", "capture_succeeded": True, "comparisons": [{"passed": True}],
        "source_hashes": {name: retain.file_digest(project/name)["sha256"] for name in graph_files}}
    graph_directory = runtime/"f2-graph-01"; write(graph_directory, "report.json", graph)
    return SimpleNamespace(project=project, docs=docs, runtime=runtime, receipt=receipt, reference=reference,
        references=references, reports=reports, directories=directories, graph=graph, graph_directory=graph_directory)


def validate(e, *, graph=False, **kwargs):
    return retain.validate_runs(e.directories+([e.graph_directory] if graph else []), project_root=e.project,
        checkpoint_receipt=e.receipt, report_dir=e.docs, **kwargs)


def collect(e, *, graph=False, **kwargs):
    runs, _, sources = validate(e, graph=graph, **kwargs)
    return retain.collect_evidence(runs, sources, e.references, project_root=e.project,
        checkpoint_receipt=e.receipt, report_dir=e.docs)


def test_all_core_stages_and_optional_successful_graph(evidence):
    runs, reference, sources = validate(evidence, graph=True)
    assert len(runs) == 4 and reference == evidence.reference
    assert all(run["provenance"]["counts_as_success"] for run in runs)
    assert "scripts/olmo_f2_graph_probe.py" in sources
    assert {run["provenance"]["stage"] for run in runs} == {"health", "capacity", "checkpoint", "graph"}


def test_capture_blocked_retains_exact_old_snapshot_and_labels_failure(evidence):
    graph = evidence.graph
    graph.update(status="capture_blocked", stage="capture", capture_succeeded=False, error_type="RuntimeError", comparisons=[])
    graph["wandb"]["status"] = "synced_failed_experiment"
    name = "cdrm/pretrained/olmo_tiled.py"
    old = write(evidence.graph_directory, "source-snapshot/"+name, "old native implementation")
    graph["source_hashes"][name] = retain.file_digest(old)["sha256"]
    write(evidence.graph_directory, "report.json", graph)
    runs, _, sources = validate(evidence, graph=True)
    historical = runs[-1]["provenance"]
    assert historical["historical_failure_only"] and not historical["counts_as_success"]
    assert historical["historical_runtime_sources_complete"]
    assert not historical["current_source_hashes_match"]
    assert historical["historical_source_mismatches"][name]["exact_historical_snapshot"] == "source-snapshot/"+name
    assert sources[name] != graph["source_hashes"][name]
    assert "scripts/olmo_f2_graph_probe.py" not in sources
    members = collect(evidence, graph=True)
    assert (old, "runtime/f2-graph-01/source-snapshot/"+name) in members


def test_failed_graph_requires_explicit_diagnostic_flag_and_is_never_promoted(evidence):
    evidence.graph.update(status="failed", stage="equivalence", error_type="AssertionError", comparisons=[{"passed": False}])
    evidence.graph["wandb"]["status"] = "synced_failed_experiment"
    write(evidence.graph_directory, "report.json", evidence.graph)
    with pytest.raises(ValueError, match="allow-failed-graph"):
        validate(evidence, graph=True)
    runs, _, sources = validate(evidence, graph=True, allow_failed_graph=True)
    assert runs[-1]["provenance"]["diagnostic_failure_only"]
    assert not runs[-1]["provenance"]["counts_as_success"]
    assert runs[-1]["provenance"]["current_source_hashes_match"]
    assert "scripts/olmo_f2_graph_probe.py" not in sources
    assert runs[-1]["report"]["comparisons"] == [{"passed": False}]


def test_completed_localization_preserves_disagreements_without_claiming_success(evidence):
    source = write(evidence.project, "scripts/olmo_f2_graph_localize.py")
    evidence.graph.update(schema=retain.LOCALIZATION_SCHEMA, status="completed",
        comparisons=[{"passed": False, "all_bitwise_equal": False}],
        all_comparisons_within_original_budget=False, all_comparisons_bitwise_equal=False,
        structural_invariants_passed=True)
    evidence.graph["source_hashes"]["scripts/olmo_f2_graph_localize.py"] = retain.file_digest(source)["sha256"]
    write(evidence.graph_directory, "report.json", evidence.graph)
    runs, _, sources = validate(evidence, graph=True)
    provenance = runs[-1]["provenance"]
    assert provenance["diagnostic_only"] and provenance["execution_completed"]
    assert not provenance["counts_as_success"] and not provenance["historical_failure_only"]
    assert provenance["localization_outcomes"]["all_comparisons_within_original_budget"] is False
    assert "scripts/olmo_f2_graph_localize.py" not in sources
    assert (source, "project/scripts/olmo_f2_graph_localize.py") in collect(evidence, graph=True)
    source.write_text("changed after completed diagnostic")
    with pytest.raises(ValueError, match="source is absent or changed"):
        validate(evidence, graph=True)


def test_historical_source_absence_is_explicit_and_wrong_snapshot_rejected(evidence):
    evidence.graph.update(status="capture_blocked", stage="capture", capture_succeeded=False, error_type="RuntimeError")
    evidence.graph["wandb"]["status"] = "synced_failed_experiment"
    evidence.graph["source_hashes"]["scripts/removed.py"] = "a"*64
    write(evidence.graph_directory, "report.json", evidence.graph)
    runs, _, _ = validate(evidence, graph=True)
    assert not runs[-1]["provenance"]["historical_runtime_sources_complete"]
    assert runs[-1]["provenance"]["historical_source_mismatches"]["scripts/removed.py"]["current_sha256"] is None
    write(evidence.graph_directory, "source-snapshot/scripts/removed.py", "wrong")
    with pytest.raises(ValueError, match="snapshot differs"):
        validate(evidence, graph=True)


@pytest.mark.parametrize("change", ["running", "failed", "rows", "nested", "source", "missing_source", "config", "protocol", "receipt"])
def test_core_reports_require_completed_matching_passing_evidence(evidence, change):
    source = evidence.reports[0]
    if change in ("running", "failed"): source["status"] = change
    elif change == "rows": source["rows"] = []
    elif change == "nested": source["rows"][0]["health"] = {"finite": False}
    elif change == "source": write(evidence.project, "scripts/olmo_f2_health_capacity.py", "changed")
    elif change == "missing_source": (evidence.project/"scripts/olmo_f2_health_capacity.py").unlink()
    elif change == "config": write(evidence.directories[0], "config.json", {"stage": "other"})
    elif change == "protocol": write(evidence.docs, "protocol.md", "changed")
    elif change == "receipt":
        receipt = json.loads(evidence.receipt.read_text()); receipt["objects"][0]["generation"] = "wrong"
        evidence.receipt.write_text(json.dumps(receipt))
    write(evidence.directories[0], "report.json", source)
    with pytest.raises(ValueError): validate(evidence)


def test_missing_core_stage_duplicate_directories_and_graph_inventory_fail(evidence):
    directory = evidence.directories.pop()
    with pytest.raises(ValueError, match="requires passing"): validate(evidence)
    evidence.directories.append(directory); evidence.directories.append(directory)
    with pytest.raises(ValueError, match="unique"): validate(evidence)
    evidence.directories.pop()
    del evidence.graph["source_hashes"]["scripts/olmo_f2_graph_probe.py"]
    write(evidence.graph_directory, "report.json", evidence.graph)
    with pytest.raises(ValueError, match="essential"): validate(evidence, graph=True)


def test_positive_archive_inventory_excludes_secrets_and_weights(evidence, tmp_path):
    for root in [evidence.project, evidence.docs, *evidence.directories]:
        for name in ("checkpoint.pt", "model.safetensors", ".env", "credentials.json", "wandb/config.json", "unlisted.json"):
            write(root, name, "NEVER RETAIN")
    selected_log = write(evidence.runtime, evidence.directories[0].name+".log", "scoped log")
    write(evidence.runtime, "unselected-run.log", "NEVER RETAIN")
    write(evidence.docs, "capacity.png", "small plot")
    members = collect(evidence)
    assert (selected_log, "runtime-logs/"+selected_log.name) in members
    assert any(name == "report/capacity.png" for _, name in members)
    assert not any("NEVER RETAIN" in path.read_text() for path, _ in members)
    archive = tmp_path/"evidence.tar.gz"
    retain.build_evidence_archive(archive, members, "restore")
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == {name for _, name in members} | {"RESTORE.md", "evidence-members.json"}


@pytest.mark.parametrize("filename", [".env", "credentials.json", "model.safetensors", "../outside.py"])
def test_dangerous_source_entries_are_rejected_even_for_historical_graph(evidence, filename):
    evidence.graph.update(status="capture_blocked", stage="capture", capture_succeeded=False, error_type="RuntimeError")
    evidence.graph["wandb"]["status"] = "synced_failed_experiment"
    evidence.graph["source_hashes"][filename] = "a"*64
    write(evidence.graph_directory, "report.json", evidence.graph)
    with pytest.raises(ValueError): validate(evidence, graph=True)


def test_symlinked_source_and_missing_final_assessment_fail(evidence):
    target = write(evidence.project, "target.py")
    source = evidence.project/"scripts/olmo_f2_health_capacity.py"
    source.unlink(); source.symlink_to(target)
    with pytest.raises(ValueError): validate(evidence)
    source.unlink(); write(evidence.project, "scripts/olmo_f2_health_capacity.py")
    (evidence.docs/"assessment.md").unlink()
    with pytest.raises(ValueError, match="assessment"): collect(evidence)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata, self.uploads = bucket, name, "123", {}, 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data); self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.bucket.objects[self.name] = self; self.uploads += 1
    def reload(self): pass


class Bucket:
    name = "fast-chunks"
    def __init__(self): self.objects = {}
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


def test_upload_is_scoped_create_only_and_f2_labeled(tmp_path):
    path = write(tmp_path, "evidence.tar.gz"); bucket = Bucket()
    _, prefix = retain.parse_prefix(retain.PREFIX_ROOT+"20260922T010000Z")
    key = prefix+"/evidence.tar.gz"; digest = retain.file_digest(path)
    first = retain.upload_verified(bucket, key, path, digest)
    assert retain.upload_verified(bucket, key, path, digest) == first
    blob = bucket.objects[key]
    assert blob.uploads == 1 and blob.metadata["artifact_schema"] == retain.SCHEMA
    blob.md5_hash = "wrong"
    with pytest.raises(ValueError): retain.upload_verified(bucket, key, path, digest)
    with pytest.raises(ValueError): retain.upload_verified(bucket, "other/evidence.tar.gz", path, digest)


def test_original_checkpoint_verification_never_uploads(evidence):
    bucket = Bucket(); key = O1_CHECKPOINT_URI.split("/", 3)[3]
    blob = Blob(bucket, key); bucket.objects[key] = blob
    blob.size, blob.md5_hash, blob.generation = CHECKPOINT_SIZE, O1_CHECKPOINT_MD5, O1_CHECKPOINT_GENERATION
    blob.metadata = {"sha256": CHECKPOINT_SHA256}
    assert retain.verify_checkpoint_reference(bucket, evidence.reference)["reused_without_upload"]
    assert blob.uploads == 0
    blob.generation = "changed"
    with pytest.raises(ValueError): retain.verify_checkpoint_reference(bucket, evidence.reference)


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT+"../bad",
    retain.PREFIX_ROOT+"20260922T010000Z/extra", "gs://other/20260922T010000Z"])
def test_prefix_requires_exact_f2_timestamp(prefix):
    with pytest.raises(ValueError): retain.parse_prefix(prefix)
