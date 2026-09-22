"""O5b archives reuse data/model objects and contain bounded evidence only."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from scripts import olmo_o5b_retain as retain


def write(root, name, content="evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    e = SimpleNamespace(**{name: tmp_path / name for name in ("project", "preflight", "runs", "data", "docs", "artifacts")})
    sources = {"cdrm/pretrained/olmo_fbt.py", "scripts/olmo_o5b_train.py"}
    for name in sources:
        write(e.project, name)
    monkeypatch.setattr(retain, "learning_source_files", lambda: sources)
    write(e.data, "manifest.json")
    e.corpus = SimpleNamespace(root=e.data, manifest_sha256=retain.file_digest(e.data / "manifest.json")["sha256"], manifest={"files": {}})
    e.config = {"schema": "olmo-o5b-pilot-config-v1", "source_hashes": {name: retain.file_digest(e.project / name)["sha256"] for name in sources},
                "data_manifest_sha256": e.corpus.manifest_sha256, "schedule": {"total_updates": 3}}
    e.report = {"schema": "olmo-o5b-preflight-v1", "status": "passed", "finished_utc": "now",
                "source_hashes": e.config["source_hashes"], "data_manifest_sha256": e.corpus.manifest_sha256,
                "schedule": e.config["schedule"]}
    write(e.preflight, "configuration.json", json.dumps(e.config))
    write(e.preflight, "report.json", json.dumps(e.report))
    write(e.runs, "queue.json", json.dumps({"schema": "olmo-o5b-queue-v1", "status": "completed"}))
    e.references = {}
    for name, files in retain.SNAPSHOT_FILES.items():
        e.references[name] = {"files": [{"file": file} for file in files]}
        for file in files | {"README.md", "manifest.json"}:
            write(e.project, f"cdrm/pretrained/{name}/{file}")
    for name in (retain.MANIFEST_FILENAME, "checkpoint-inspection.json"):
        write(e.artifacts, name)
    for name in retain.FILE_SPECS:
        write(e.artifacts, "native/" + name)
    for name in retain.REPORT_FILES:
        write(e.docs, name)
    e.o1_receipt = write(tmp_path, "o1.json")
    e.o4_receipt = write(tmp_path, "o4.json")
    e.o4_manifest = write(tmp_path, "o4-manifest.json")
    for arm in ("ordinary", "fbt"):
        for name in ("report.json", "events.jsonl", "update-000003.receipt.json"):
            write(e.runs, f"{arm}/{name}")
    return e


def validate(e, phase="initial"):
    return retain.validate_learning(e.preflight, e.runs, phase, e.corpus, project_root=e.project)


def collect(e, phase="initial"):
    return retain.collect_evidence(e.preflight, e.runs, e.config, e.references, phase=phase,
        data=e.data, report_dir=e.docs, project_root=e.project, artifacts=e.artifacts,
        o1_receipt=e.o1_receipt, o4_receipt=e.o4_receipt, o4_manifest=e.o4_manifest, arms=("ordinary", "fbt"))


def test_passing_configuration_and_completed_queue_validate(evidence):
    assert validate(evidence) == (evidence.config, evidence.report)
    assert validate(evidence, "final") == (evidence.config, evidence.report)


@pytest.mark.parametrize("field,value", [("status", "running"), ("schema", "other"), ("finished_utc", None),
    ("source_hashes", {}), ("data_manifest_sha256", "wrong"), ("schedule", {})])
def test_preflight_drift_or_incompletion_rejected(evidence, field, value):
    evidence.report[field] = value
    write(evidence.preflight, "report.json", json.dumps(evidence.report))
    with pytest.raises(ValueError): validate(evidence)


def test_changed_source_and_extra_source_rejected(evidence):
    write(evidence.project, "cdrm/pretrained/olmo_fbt.py", "changed")
    with pytest.raises(ValueError, match="changed"): validate(evidence)
    evidence.config["source_hashes"]["secret.json"] = "bad"
    write(evidence.preflight, "configuration.json", json.dumps(evidence.config))
    with pytest.raises(ValueError, match="inventory"): validate(evidence)


def test_final_retention_rejects_unfinished_queue(evidence):
    write(evidence.runs, "queue.json", json.dumps({"schema": "olmo-o5b-queue-v1", "status": "running"}))
    with pytest.raises(ValueError, match="unfinished"): validate(evidence, "final")


@pytest.mark.parametrize("phase", ["initial", "final"])
def test_evidence_whitelist_excludes_dataset_models_secrets_and_logs(evidence, tmp_path, phase):
    e = evidence
    for root in (e.project, e.artifacts, e.preflight, e.runs / "fbt", e.docs, e.data):
        for name in ("checkpoint.pt", "model.safetensors", "train.tokens.bin", ".env", "credentials.json", "private.log"):
            write(root, name, "exclude")
    members = collect(e, phase)
    names = [name for _, name in members]
    assert "prepared-data/manifest.json" in names
    assert "provenance/o4-initial-manifest.json" in names
    assert not any(name.endswith((".pt", ".bin", ".safetensors", ".env", "credentials.json", ".log")) for name in names)
    assert ("runs/fbt/events.jsonl" in names) == (phase == "final")
    archive = tmp_path / "evidence.tar.gz"
    retain.build_evidence_archive(archive, members, "restore")
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == set(names) | {"RESTORE.md", "evidence-members.json"}


def test_missing_final_graph_and_snapshot_injection_rejected(evidence):
    (evidence.docs / "online-comparison.pdf").unlink()
    with pytest.raises(ValueError, match="artifacts"): collect(evidence, "final")
    evidence.references["_fbt_reference"]["files"].append({"file": "credentials.json"})
    with pytest.raises(ValueError, match="whitelist"): collect(evidence)


def test_symlinked_source_cannot_escape_inventory(evidence, tmp_path):
    path = evidence.project / "scripts/olmo_o5b_train.py"
    path.unlink(); path.symlink_to(write(tmp_path, "outside.py"))
    with pytest.raises(ValueError, match="regular"): collect(evidence)


def test_prior_prepared_data_pin_checks_archive_manifest_and_every_member(evidence, monkeypatch):
    e = evidence
    token_path = write(e.data, "train.tokens.bin", "tokens")
    e.corpus.manifest["files"]["train.tokens.bin"] = {k: v for k, v in retain.file_digest(token_path).items() if k != "md5_base64"}
    members = {"prepared-data/manifest.json": {k: v for k, v in retain.file_digest(e.data / "manifest.json").items() if k != "md5_base64"},
               "prepared-data/train.tokens.bin": e.corpus.manifest["files"]["train.tokens.bin"]}
    prior = {"schema": "olmo-o4-evidence-v1", "phase": "initial", "data_manifest_sha256": e.corpus.manifest_sha256, "members": members}
    write(e.o4_manifest.parent, e.o4_manifest.name, json.dumps(prior))
    pins = {"manifest.json": {"generation": "2", **retain.file_digest(e.o4_manifest)},
            "evidence.tar.gz": {"generation": "1", "sha256": "archive", "size_bytes": 42, "md5_base64": "md5"}}
    monkeypatch.setattr(retain, "O4_OBJECTS", pins)
    receipt = {"schema": "olmo-o4-evidence-receipt-v1", "phase": "initial", "status": "verified",
               "objects": [{"uri": retain.O4_PREFIX + name, **record} for name, record in pins.items()]}
    assert len(retain.prepared_data_reference(receipt, e.o4_manifest, e.corpus)) == 2
    e.corpus.manifest["files"]["train.tokens.bin"] = {"sha256": "changed", "size_bytes": 6}
    with pytest.raises(ValueError, match="member"): retain.prepared_data_reference(receipt, e.o4_manifest, e.corpus)
    receipt["objects"][0]["generation"] = "wrong"
    with pytest.raises(ValueError, match="pin"): retain.prepared_data_reference(receipt, e.o4_manifest, e.corpus)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata = bucket, name, "123", {}
        self.uploads = 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        payload = Path(filename).read_bytes()
        self.size = len(payload); self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode()
        self.bucket.objects[self.name] = self; self.uploads += 1
    def reload(self): pass


class Bucket:
    name = "fast-chunks"
    def __init__(self): self.objects = {}
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


def test_upload_is_create_only_and_conflicting_remote_data_rejected(tmp_path):
    path = write(tmp_path, "evidence.tar.gz")
    bucket = Bucket(); digest = retain.file_digest(path)
    result = retain.upload_verified(bucket, "p/initial/evidence.tar.gz", path, digest)
    assert retain.upload_verified(bucket, "p/initial/evidence.tar.gz", path, digest) == result
    blob = bucket.get_blob("p/initial/evidence.tar.gz")
    assert blob.uploads == 1 and blob.metadata["artifact_schema"] == retain.SCHEMA
    blob.metadata["sha256"] = "changed"
    with pytest.raises(ValueError): retain.upload_verified(bucket, "p/initial/evidence.tar.gz", path, digest)


def test_reused_reference_requires_exact_generation_and_never_uploads(tmp_path):
    path = write(tmp_path, "manifest.json")
    bucket = Bucket(); digest = retain.file_digest(path)
    record = retain.upload_verified(bucket, "old/manifest.json", path, digest)
    blob = bucket.get_blob("old/manifest.json"); blob.uploads = 0
    assert retain.verify_reference(bucket, record)["reused_without_upload"]
    assert blob.uploads == 0
    blob.generation = "changed"
    with pytest.raises(ValueError, match="generation"): retain.verify_reference(bucket, record)


@pytest.mark.parametrize("name", ["model.safetensors", "checkpoint.pt", "train.tokens.bin", "credentials.json"])
def test_only_named_evidence_objects_can_upload(tmp_path, name):
    path = write(tmp_path, name)
    with pytest.raises(ValueError, match="bounded"): retain.upload_verified(Bucket(), "p/" + name, path, retain.file_digest(path))


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT + "../escape",
    retain.PREFIX_ROOT + "20260922T010000Z/extra", "gs://other/20260922T010000Z"])
def test_prefix_must_be_single_timestamp_under_o5b(prefix):
    with pytest.raises(ValueError): retain.parse_prefix(prefix)


def test_training_receipts_allow_only_explicit_local_cleanup_annotation(tmp_path, monkeypatch):
    prefix = retain.PREFIX_ROOT + "20260922T010000Z"
    storage = {"uri": prefix + "/fbt/update-000003.pt", "sha256": "x", "size_bytes": 12}
    record = {"path": "fbt/update-000003.pt", "sha256": "x", "size_bytes": 12, "storage": storage}
    write(tmp_path, "fbt/update-000003.receipt.json", json.dumps(record))
    record["local_file_removed_after_verified_successor"] = True
    write(tmp_path, "fbt/report.json", json.dumps({"checkpoints": [record]}))
    monkeypatch.setattr(retain, "verify_reference", lambda b, r: r)
    assert retain.verify_training_checkpoints(None, tmp_path, ("fbt",), prefix) == {"fbt": [storage]}
    record["reason"] = "altered"
    write(tmp_path, "fbt/report.json", json.dumps({"checkpoints": [record]}))
    with pytest.raises(ValueError, match="receipt differ"): retain.verify_training_checkpoints(None, tmp_path, ("fbt",), prefix)


@pytest.mark.parametrize("mutation", [None, "summary", "figure", "helper", "input"])
def test_final_comparison_rechecked_against_runs_and_rendering_sources(evidence, monkeypatch, mutation):
    from scripts import olmo_o5b_report as reporter
    e = evidence
    expected = {"status": "completed", "configuration": e.config}
    monkeypatch.setattr(reporter, "validate_runs", lambda *_: expected)
    for name in ("scripts/olmo_o5b_report.py", "scripts/olmo_o4_report.py", "cdrm/pretrained/lm_schedule.py"):
        write(e.project, name)
    comparison = {**expected, "report_source_sha256": retain.file_digest(e.project / "scripts/olmo_o5b_report.py")["sha256"],
        "input_sha256": {name: retain.file_digest(path)["sha256"] for name, path in {
            "preflight": e.preflight / "report.json", "configuration": e.preflight / "configuration.json",
            "ordinary": e.runs / "ordinary/report.json", "fbt": e.runs / "fbt/report.json"}.items()},
        "helper_source_sha256": {name: retain.file_digest(e.project / name)["sha256"] for name in
                                  ("scripts/olmo_o4_report.py", "cdrm/pretrained/lm_schedule.py")},
        "figure_sha256": {name: retain.file_digest(e.docs / name)["sha256"] for name in
                          ("learning-curves.pdf", "learning-curves.png", "online-comparison.pdf", "online-comparison.png")}}
    if mutation == "summary": comparison["status"] = "different"
    write(e.docs, "final-comparison.json", json.dumps(comparison))
    if mutation == "figure": write(e.docs, "online-comparison.pdf", "changed")
    if mutation == "helper": write(e.project, "scripts/olmo_o4_report.py", "changed")
    if mutation == "input": write(e.runs, "fbt/report.json", "changed")
    if mutation is None:
        assert retain.validate_final_comparison(e.preflight, e.runs, e.docs, project_root=e.project) == comparison
    else:
        with pytest.raises(ValueError): retain.validate_final_comparison(e.preflight, e.runs, e.docs, project_root=e.project)
