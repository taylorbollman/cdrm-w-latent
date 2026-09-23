"""F4 archival lineage, explicit diagnostic failures and create-only transport."""
import base64
import copy
import hashlib
import gzip
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import olmo_f4_retain as retain
from test_olmo_f4_report import make_report, TRACE_BYTES
from scripts.olmo_tiled_retain import O1_CHECKPOINT_URI, O1_CHECKPOINT_GENERATION, O1_CHECKPOINT_MD5


def write(root, name, value="fixture"):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


@pytest.fixture
def evidence(tmp_path):
    project, docs, directory = (tmp_path/name for name in ("project", "docs", "f4-native-01"))
    for name in set.union(*retain.ESSENTIAL_SOURCES.values()) | retain.EXTRA_PROJECT_FILES:
        write(project, name, "current "+name)
    references = {}
    for name, files in retain.SNAPSHOT_FILES.items():
        references[name] = {"files": [{"file": filename} for filename in sorted(files)]}
        for filename in files | {"README.md", "manifest.json"}:
            write(project, f"cdrm/pretrained/{name}/{filename}")
    for name in ("protocol.md", "results.md", "assessment.md", "test-results.txt"):
        write(docs, name)
    reference = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
        "size_bytes": retain.CHECKPOINT_SIZE, "sha256": retain.CHECKPOINT_SHA256, "md5_base64": O1_CHECKPOINT_MD5}
    receipt = write(tmp_path, "o1-receipt.json", {"schema": "olmo-o1-storage-receipt-v1", "status": "verified", "objects": [reference]})
    report = make_report(stage="capacity", case="combined")
    report["checks"][0]["strict_comparison_diagnostic"] = {"passed":False}
    report.update(protocol_sha256=retain.file_digest(docs/"protocol.md")["sha256"],
        source_hashes={name: retain.file_digest(project/name)["sha256"] for name in retain.ESSENTIAL_SOURCES[retain.NATIVE]},
        checkpoint={"path":"native/model.safetensors", "sha256":retain.CHECKPOINT_SHA256, "size_bytes":retain.CHECKPOINT_SIZE})
    for name in report["source_hashes"]:
        write(directory, "source-snapshot/"+name, (project/name).read_text())
    write(directory, "protocol.md", (docs/"protocol.md").read_text())
    write(directory, "report.json", report)
    return SimpleNamespace(project=project, docs=docs, directory=directory, directories=[directory],
        report=report, receipt=receipt, reference=reference, references=references)


def validate(e, **kwargs):
    return retain.validate_runs(e.directories, project_root=e.project, report_dir=e.docs,
        checkpoint_receipt=e.receipt, **kwargs)


def collect(e, **kwargs):
    runs, _, sources = validate(e, **kwargs)
    return retain.collect_evidence(runs, sources, e.references, project_root=e.project,
        report_dir=e.docs, checkpoint_receipt=e.receipt)


def test_success_keeps_failed_strict_diagnostics_and_exact_source_bytes(evidence):
    runs, reference, current = validate(evidence)
    assert reference == evidence.reference
    assert current == evidence.report["source_hashes"]
    assert runs[0]["provenance"]["counts_as_success"]
    assert not runs[0]["report"]["checks"][0]["strict_comparison_diagnostic"]["passed"]
    assert len(runs[0]["source_snapshots"]) == len(current)


def test_retention_invokes_strict_resource_and_update_validation(evidence):
    evidence.report["resources"]["analytic_matrix_work"]["matrix_flops_minimum"] += 1
    write(evidence.directory, "report.json", evidence.report)
    with pytest.raises(ValueError):
        validate(evidence)


def test_successful_earlier_source_and_protocol_are_valid_historical_lineage(evidence):
    name = "cdrm/pretrained/olmo_tiled.py"
    write(evidence.project, name, "current later implementation")
    write(evidence.docs, "protocol.md", "later explicitly revised protocol")
    runs, _, current = validate(evidence)
    p = runs[0]["provenance"]
    assert p["counts_as_success"] and p["exact_runtime_sources_complete"]
    assert not p["current_source_hashes_match"] and not p["current_protocol_hash_matches"]
    assert current[name] != p["reported_source_hashes"][name]
    other = evidence.directory.with_name("f4-native-02")
    report = copy.deepcopy(evidence.report)
    for source in report["source_hashes"]:
        path = write(other, "source-snapshot/"+source, (evidence.project/source).read_text())
        report["source_hashes"][source] = retain.file_digest(path)["sha256"]
    report["protocol_sha256"] = retain.file_digest(write(other, "protocol.md", (evidence.docs/"protocol.md").read_text()))["sha256"]
    write(other, "report.json", report)
    evidence.directories.append(other)
    runs, _, _ = validate(evidence)
    assert len(runs) == 2 and runs[1]["provenance"]["current_source_hashes_match"]


@pytest.mark.parametrize("damage", ["missing", "corrupt", "symlink"])
def test_exact_runtime_snapshot_required_even_when_current_source_matches(evidence, damage):
    source = evidence.directory/"source-snapshot/cdrm/pretrained/olmo_tiled.py"
    source.unlink()
    if damage == "corrupt":
        source.write_text("not original")
    elif damage == "symlink":
        source.symlink_to(evidence.project/"cdrm/pretrained/olmo_tiled.py")
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(evidence)


def test_failures_require_opt_in_and_never_count_as_success(evidence):
    report = evidence.report
    report.update(status="failed", error_type="AssertionError", error_message="screen failed")
    report["wandb"]["status"] = "synced_failed_experiment"
    report["checks"][0]["passed"] = False
    write(evidence.directory, "report.json", report)
    with pytest.raises(ValueError, match="allow-failed-diagnostics"):
        validate(evidence)
    runs, _, _ = validate(evidence, allow_failed_diagnostics=True)
    assert not runs[0]["provenance"]["counts_as_success"]
    assert runs[0]["provenance"]["diagnostic_failure_only"]


@pytest.mark.parametrize("mutation", ["running", "failed_aggregate", "missing_sources", "checkpoint", "protocol",
    "configuration", "wandb", "unsafe_source", "binary_source", "missing_error"])
def test_inconsistent_or_unsafe_report_rejected(evidence, mutation):
    r = evidence.report
    if mutation == "running": r["status"] = "running"
    elif mutation == "failed_aggregate": r["checks"][0]["passed"] = False
    elif mutation == "missing_sources": r["source_hashes"].pop("cdrm/pretrained/olmo_tiled.py")
    elif mutation == "checkpoint": r["checkpoint"]["sha256"] = "a"*64
    elif mutation == "protocol": write(evidence.directory, "protocol.md", "wrong")
    elif mutation == "configuration": write(evidence.directory, "configuration.json", {"wrong": True})
    elif mutation == "wandb": r["wandb"]["status"] = "running"
    elif mutation == "unsafe_source": r["source_hashes"]["scripts/../../secret.py"] = "a"*64
    elif mutation == "binary_source": r["source_hashes"]["scripts/model.pt"] = "a"*64
    elif mutation == "missing_error":
        r["status"] = "failed"; r["wandb"]["status"] = "synced_failed_experiment"
    write(evidence.directory, "report.json", r)
    with pytest.raises(ValueError):
        validate(evidence, allow_failed_diagnostics=True)










def test_historical_f3b_report_is_reference_only_not_accepted_as_f4_runtime(evidence):
    evidence.report["schema"] = "olmo-f3b-native-v1"
    write(evidence.directory, "report.json", evidence.report)
    with pytest.raises(ValueError, match="supported F4"):
        validate(evidence)


def test_archive_whitelist_excludes_models_fixtures_secrets_and_unrelated_logs(evidence):
    for root in (evidence.project, evidence.docs, evidence.directory):
        for name in (".env", "model.safetensors", "fixtures.pt", "checkpoint.pt.tmp", "credentials.json", "wandb/run.json", "unrelated.json"):
            write(root, name, "NEVER RETAIN")
    write(evidence.directory.parent, "unrelated.log", "NEVER RETAIN")
    log = write(evidence.directory.parent, evidence.directory.name+".log", "selected log")
    members = collect(evidence)
    assert any(path == log for path, _ in members)
    assert all("NEVER RETAIN" not in path.read_text() for path, _ in members)
    assert any(name.startswith("runtime/f4-native-01/source-snapshot/") for _, name in members)
    archive = evidence.directory.parent/"evidence.tar.gz"
    retain.build_evidence_archive(archive, members, "restore")
    with tarfile.open(archive) as tar:
        assert "evidence-members.json" in tar.getnames()
        assert all(not name.endswith((".pt", ".tmp", ".safetensors")) for name in tar.getnames())
        assert {"project/scripts/olmo_f3b_report.py", "project/scripts/olmo_f3_report.py", "project/scripts/olmo_f4_report.py", "project/scripts/olmo_f3e_report.py", "project/tests/test_olmo_f3e_report.py", "project/tests/test_olmo_multilayer_execution.py"} <= set(tar.getnames())


def test_dry_run_builds_local_archive_without_remote_verification(evidence, monkeypatch):
    monkeypatch.setattr(retain, "verify_snapshots", lambda root: evidence.references)
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run attempted cloud access")
    monkeypatch.setattr(retain, "verify_checkpoint_reference", forbidden)
    monkeypatch.setattr(retain, "upload_verified", forbidden)
    args = SimpleNamespace(runtime_dir=evidence.directories, report_dir=evidence.docs,
        checkpoint_receipt=evidence.receipt, output_dir=evidence.directory.parent/"retention",
        storage_prefix=retain.PREFIX_ROOT+"20260922T010000Z", allow_failed_diagnostics=False, dry_run=True)
    result = retain.retain(args, project_root=evidence.project)
    assert result["status"] == "dry_run" and not result["uploaded"]
    assert not result["remote_checkpoint_verified"]
    assert not (evidence.docs/"storage-receipt.json").exists()


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata, self.uploads = bucket, name, "123", {}, 0

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.bucket.objects[self.name] = self
        self.uploads += 1

    def reload(self):
        pass


class Bucket:
    name = "fast-chunks"

    def __init__(self):
        self.objects = {}

    def get_blob(self, name):
        return self.objects.get(name)

    def blob(self, name):
        return Blob(self, name)


def test_create_only_transport_checks_scope_and_checksum_without_uploading_weights(tmp_path):
    path = write(tmp_path, "evidence.tar.gz")
    bucket = Bucket()
    _, prefix = retain.parse_prefix(retain.PREFIX_ROOT+"20260922T010000Z")
    key, digest = prefix+"/evidence.tar.gz", retain.file_digest(path)
    first = retain.upload_verified(bucket, key, path, digest)
    assert retain.upload_verified(bucket, key, path, digest) == first
    assert bucket.objects[key].uploads == 1
    assert bucket.objects[key].metadata["artifact_schema"] == retain.SCHEMA
    bucket.objects[key].metadata["sha256"] = "a"*64
    with pytest.raises(ValueError): retain.upload_verified(bucket, key, path, digest)
    with pytest.raises(ValueError): retain.upload_verified(bucket, "other/evidence.tar.gz", path, digest)
    with pytest.raises(ValueError): retain.upload_verified(bucket, prefix+"/model.safetensors", path, digest)


@pytest.mark.parametrize("prefix", ["gs://other/abc", retain.PREFIX_ROOT+"../bad", retain.PREFIX_ROOT+"20260922T010000Z/child"])
def test_storage_prefix_is_narrow(prefix):
    with pytest.raises(ValueError): retain.parse_prefix(prefix)



def test_all_frozen_runtime_sources_required_not_only_kernel_files(evidence):
    evidence.report["source_hashes"].pop("cdrm/pretrained/static_training.py")
    write(evidence.directory,"report.json",evidence.report)
    with pytest.raises(ValueError,match="essential runtime source"):
        validate(evidence)


def trace_fixture(directory):
    path=directory/"operator-trace.json.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(b'{"traceEvents":[]}'))
    digest=retain.file_digest(path)
    return {"file":path.name,"sha256":digest["sha256"],"bytes":digest["size_bytes"]}


def test_operator_trace_exact_bytes_and_provenance(evidence):
    # Diagnostic failure permits retaining a completed trace without claiming
    # that the surrounding numerical/resource run passed.
    original=evidence.report
    report=make_report(stage="correctness",case="combined",length=32,batch=1)
    report.update({k:original[k] for k in ("source_hashes","protocol_sha256","checkpoint")})
    report.update(status="failed",error_type="AssertionError",error_message="later diagnostic failed")
    report["configuration"]["operator_trace"]=True
    report["wandb"]["status"]="synced_failed_experiment"
    report["operator_trace"]=trace_fixture(evidence.directory)
    write(evidence.directory,"report.json",report)
    runs,_,_=validate(evidence,allow_failed_diagnostics=True)
    trace=runs[0]["provenance"]["operator_trace"]
    assert trace["sha256"]==report["operator_trace"]["sha256"]
    assert trace["size_bytes"]==report["operator_trace"]["bytes"]
    assert not runs[0]["provenance"]["counts_as_success"]
    members=collect(evidence,allow_failed_diagnostics=True)
    assert (evidence.directory/trace["file"],f"runtime/{evidence.directory.name}/operator-trace.json.gz") in members


@pytest.mark.parametrize("damage",["missing","hash","size","boolean_size","unsafe_name","symlink","undeclared_flag"])
def test_operator_trace_rejects_corruption_and_unsafe_artifacts(tmp_path,damage):
    trace=trace_fixture(tmp_path)
    report={"configuration":{"operator_trace":True},"operator_trace":trace}
    if damage=="missing":(tmp_path/trace["file"]).unlink()
    if damage=="hash":trace["sha256"]="a"*64
    if damage=="size":trace["bytes"]+=1
    if damage=="boolean_size":trace["bytes"]=True
    if damage=="unsafe_name":trace["file"]="../operator-trace.json.gz"
    if damage=="symlink":
        original=tmp_path/trace["file"]; target=tmp_path/"original.gz";original.rename(target);original.symlink_to(target)
    if damage=="undeclared_flag":report["configuration"]["operator_trace"]=False
    with pytest.raises((ValueError,FileNotFoundError)):
        retain.verify_operator_trace(report,tmp_path,failed=True)


def test_missing_requested_trace_only_allowed_for_failed_run(tmp_path):
    report={"configuration":{"operator_trace":True}}
    with pytest.raises(ValueError,match="missing"):
        retain.verify_operator_trace(report,tmp_path,failed=False)
    assert retain.verify_operator_trace(report,tmp_path,failed=True) is None


def test_unreferenced_trace_is_not_added_to_archive(evidence):
    trace_fixture(evidence.directory)
    assert all(not member.endswith("operator-trace.json.gz") for _,member in collect(evidence))


def test_passed_run_retains_requested_trace_after_strict_report_validation(evidence):
    report=make_report(stage="correctness",case="combined",length=32,batch=1,trace=True)
    report.update({key:evidence.report[key] for key in ("source_hashes","protocol_sha256","checkpoint")})
    (evidence.directory/"operator-trace.json.gz").write_bytes(TRACE_BYTES)
    write(evidence.directory,"report.json",report)
    runs,_,_=validate(evidence)
    assert runs[0]["provenance"]["counts_as_success"]
    assert runs[0]["provenance"]["operator_trace"]["sha256"]==report["operator_trace"]["sha256"]
    members=collect(evidence)
    assert (evidence.directory/"operator-trace.json.gz",f"runtime/{evidence.directory.name}/operator-trace.json.gz") in members


@pytest.fixture
def roundoff_evidence(evidence):
    from test_olmo_f4_report import make_roundoff_report
    directory = evidence.directory.with_name('f4-roundoff-01')
    report = make_roundoff_report()
    report['wandb'] = copy.deepcopy(evidence.report['wandb'])
    report['checkpoint'] = copy.deepcopy(evidence.report['checkpoint'])
    write(evidence.docs, 'roundoff-protocol.md', 'bounded roundoff protocol')
    write(directory, 'protocol.md', (evidence.docs/'roundoff-protocol.md').read_text())
    report['protocol_sha256'] = retain.file_digest(directory/'protocol.md')['sha256']
    report['source_hashes'] = {}
    for name in retain.ESSENTIAL_SOURCES[retain.ROUNDOFF]:
        source = write(directory, 'source-snapshot/'+name, (evidence.project/name).read_text())
        report['source_hashes'][name] = retain.file_digest(source)['sha256']
    write(directory, 'report.json', report)
    evidence.roundoff = directory
    evidence.roundoff_report = report
    return evidence


def validate_roundoff(e, **kwargs):
    return retain.validate_roundoff_runs([e.roundoff], project_root=e.project,
        report_dir=e.docs, checkpoint_receipt=e.receipt, **kwargs)


def test_roundoff_separate_completed_scope_and_exact_41_sources(roundoff_evidence):
    e = roundoff_evidence
    runs, reference, sources = validate_roundoff(e)
    p = runs[0]['provenance']
    assert reference == e.reference and len(sources) == 41
    assert not p['counts_as_success'] and not p['diagnostic_failure_only']
    assert p['roundoff_diagnostic']['physical_optimizer_updates'] == 6
    assert p['roundoff_diagnostic']['original_screen_cleared'] is False
    assert runs[0]['operator_trace'] is None
    with pytest.raises(ValueError, match='supported F4'):
        retain.validate_runs([e.roundoff], project_root=e.project, report_dir=e.docs,
            checkpoint_receipt=e.receipt)


def test_roundoff_failure_explicit_and_historical_source_preserved(roundoff_evidence):
    e = roundoff_evidence
    report = e.roundoff_report
    report.update(status='failed', arms={}, comparisons={}, graph_checks=[],
        error_type='ValueError', error_message='Prepared mode/runtime context differs from capture')
    report['wandb']['status'] = 'synced_failed_experiment'
    write(e.roundoff, 'report.json', report)
    write(e.project, 'scripts/olmo_f4_roundoff.py', 'later harness fix')
    write(e.docs, 'roundoff-protocol.md', 'later protocol note')
    with pytest.raises(ValueError, match='allow-failed-diagnostics'):
        validate_roundoff(e)
    runs, _, _ = validate_roundoff(e, allow_failed_diagnostics=True)
    p = runs[0]['provenance']
    assert p['diagnostic_failure_only'] and not p['counts_as_success']
    assert p['roundoff_diagnostic']['physical_optimizer_updates'] == 0
    assert not p['current_source_hashes_match'] and not p['current_protocol_hash_matches']


@pytest.mark.parametrize('damage', ['wrong_case','cleared','capacity','trace','missing_source',
    'corrupt_source','corrupt_protocol','wrong_checkpoint','wandb','missing_arm','bad_graph'])
def test_roundoff_malformed_diagnostic_cannot_be_retained(roundoff_evidence, damage):
    e = roundoff_evidence
    r = e.roundoff_report
    if damage == 'wrong_case': r['case']['batch_size'] = 64
    elif damage == 'cleared': r['original_screen_cleared'] = True
    elif damage == 'capacity': r['capacity'] = {}
    elif damage == 'trace': r['operator_trace'] = {'file':'operator-trace.json.gz'}
    elif damage == 'missing_source': r['source_hashes'].pop('scripts/olmo_f4_roundoff.py')
    elif damage == 'corrupt_source': write(e.roundoff, 'source-snapshot/scripts/olmo_f4_roundoff.py', 'wrong')
    elif damage == 'corrupt_protocol': write(e.roundoff, 'protocol.md', 'wrong')
    elif damage == 'wrong_checkpoint': r['checkpoint']['sha256'] = 'a'*64
    elif damage == 'wandb': r['wandb']['status'] = 'running'
    elif damage == 'missing_arm': r['arms'].pop('fp32-recompute')
    elif damage == 'bad_graph': r['graph_checks'][0]['passed'] = False
    write(e.roundoff, 'report.json', r)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_roundoff(e)


def test_roundoff_manifest_archive_and_update_counts_stay_separate(roundoff_evidence, monkeypatch):
    e = roundoff_evidence
    monkeypatch.setattr(retain, 'verify_snapshots', lambda root: e.references)
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run attempted cloud access')
    monkeypatch.setattr(retain, 'upload_verified', forbidden)
    monkeypatch.setattr(retain, 'verify_checkpoint_reference', forbidden)
    write(e.docs, 'roundoff-summary.json', {'separate_updates':6})
    write(e.roundoff, 'operator-trace.json.gz', 'undeclared stray trace')
    args = SimpleNamespace(runtime_dir=e.directories, roundoff_dir=[e.roundoff], report_dir=e.docs,
        checkpoint_receipt=e.receipt, output_dir=e.directory.parent/'retention',
        storage_prefix=retain.PREFIX_ROOT+'20260922T010000Z', allow_failed_diagnostics=False, dry_run=True)
    result = retain.retain(args, project_root=e.project)
    manifest = json.loads(Path(result['manifest']).read_text())
    assert len(manifest['runs']) == 1 and len(manifest['roundoff_diagnostics']) == 1
    assert manifest['roundoff_completed_physical_optimizer_updates'] == 6
    assert manifest['roundoff_diagnostics_clear_original_screen'] is False
    assert not manifest['roundoff_diagnostics'][0]['counts_as_success']
    with tarfile.open(result['archive']) as archive:
        names = set(archive.getnames())
        assert 'roundoff/f4-roundoff-01/report.json' in names
        assert 'roundoff/f4-roundoff-01/source-snapshot/scripts/olmo_f4_roundoff.py' in names
        assert 'report/roundoff-protocol.md' in names and 'report/roundoff-summary.json' in names
        assert 'roundoff/f4-roundoff-01/operator-trace.json.gz' not in names


def test_roundoff_same_name_as_native_rejected(roundoff_evidence):
    e = roundoff_evidence
    args = SimpleNamespace(runtime_dir=e.directories, roundoff_dir=e.directories,
        report_dir=e.docs, output_dir=e.directory.parent/'out', checkpoint_receipt=e.receipt,
        storage_prefix=retain.PREFIX_ROOT+'20260922T010000Z', allow_failed_diagnostics=False, dry_run=True)
    with pytest.raises(ValueError, match='unique'):
        retain.retain(args, project_root=e.project)
