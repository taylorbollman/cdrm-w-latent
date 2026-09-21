"""O4 evidence whitelist/provenance checks; cloud uploads are local fakes."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import olmo_o4_retain as retain


def _write(root, name, content="evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    project, preflight, data, runs, output = (tmp_path / name for name in
                                            ("project", "preflight", "data", "runs", "retention"))
    source_name = "cdrm/pretrained/nextlat.py"
    source_file = _write(project, source_name, "# frozen native NextLat source\n")
    source = {source_name: hashlib.sha256(source_file.read_bytes()).hexdigest()}
    _write(project, "cdrm/pretrained/__init__.py", "")
    for reference in ("_olmo_reference", "_nextlat_reference"):
        _write(project, f"cdrm/pretrained/{reference}/manifest.json",
               json.dumps({"files": [{"file": "model.py"}, {"file": "LICENSE"}]}))
        for name in ("model.py", "LICENSE", "README.md"):
            _write(project, f"cdrm/pretrained/{reference}/{name}")
    _write(project, "docs/reports/olmo1b-o1/storage-receipt.json", '{"status":"verified","objects":[]}')
    _write(project, "docs/reports/olmo1b-o4/protocol.md")
    _write(project, "scripts/olmo_o4_retain.py")
    _write(project, "scripts/olmo_o4_queue.py")
    _write(project, "tests/test_olmo_o4_retention.py")
    _write(project, "AGENTS.md")
    _write(preflight, "report.json", '{"status":"passed"}')
    _write(preflight, "configuration.pre-retention-retry.json", '{"old_source":"preserved"}')
    _write(preflight, "olmo_o4_train_before_retention_retry.py", "# original preflight training driver\n")
    config = {"source_hashes": source, "data_manifest_sha256": "a" * 64}
    _write(preflight, "configuration.json", json.dumps(config))
    _write(data, "train.tokens.bin", "prepared token bytes")
    _write(data, "train.windows.npy", "prepared window bytes")
    _write(data, "train.documents.jsonl", '{"document_id":1}\n')
    files = {path.name: {"size_bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in data.iterdir()}
    manifest = {"files": files}
    _write(data, "manifest.json", json.dumps(manifest))
    for repo in ("code_search_net", "wikitext"):
        _write(data.parent, f"raw/{repo}/README.md", "pinned dataset card")
    _write(runs, "queue.json", '{"status":"completed"}')
    for arm in ("ordinary", "ordinary-nextlat", "rt", "rt-nextlat"):
        _write(runs, f"{arm}/report.json", '{"status":"completed"}')
        _write(runs, f"{arm}/events.jsonl", '{"update":1}\n')
        _write(runs, f"{arm}/update-000001.receipt.json", '{"storage":{"uri":"gs://checkpoint"}}')
        for name in ("update-000001.pt", "model.safetensors", ".env", "credentials.json", "private.log"):
            _write(runs, f"{arm}/{name}", "excluded run artifact")
    corpus = SimpleNamespace(manifest_sha256="a" * 64, manifest=manifest)
    uploads = []

    def fake_upload(path, uri, **kwargs):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {"uri": uri, "generation": "123", "sha256": digest, "size_bytes": path.stat().st_size}
        uploads.append(row)
        return row

    monkeypatch.setattr(retain, "ROOT", project)
    monkeypatch.setattr(retain, "source_hashes", lambda: dict(source))
    monkeypatch.setattr(retain, "load_lm_data", lambda _: corpus)
    monkeypatch.setattr(retain, "retain_file", fake_upload)
    return SimpleNamespace(project=project, preflight=preflight, data=data, runs=runs, output=output,
                           source=source, config=config, corpus=corpus, uploads=uploads)


def _run(monkeypatch, evidence, phase="initial"):
    monkeypatch.setattr(sys, "argv", ["olmo_o4_retain.py", "--data", str(evidence.data),
        "--preflight", str(evidence.preflight), "--runs", str(evidence.runs),
        "--output-dir", str(evidence.output), "--prefix", "gs://fast-chunks/o4/test", "--phase", phase])
    retain.main()


def _archive(evidence):
    with tarfile.open(evidence.output / "evidence.tar.gz") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def test_initial_archive_uses_exact_prepared_inventory_and_excludes_unrelated_secrets_or_models(evidence, monkeypatch):
    for name in ("credentials.json", "model.safetensors", "checkpoint.pt", ".env", "notes.txt"):
        _write(evidence.data, name, "not in prepared-data manifest")
        _write(evidence.project, f"cdrm/pretrained/{name}", "not in selected source inventory")
        _write(evidence.project, f"cdrm/pretrained/_nextlat_reference/{name}", "not in source snapshot manifest")
    _run(monkeypatch, evidence)
    members = _archive(evidence)
    assert {name.removeprefix("prepared-data/") for name in members if name.startswith("prepared-data/")} == {
        "manifest.json", *evidence.corpus.manifest["files"]}
    assert {"dataset-cards/code_search_net.md", "dataset-cards/wikitext.md",
            "preflight/configuration.json", "preflight/configuration.pre-retention-retry.json",
            "preflight/olmo_o4_train_before_retention_retry.py"} <= set(members)
    assert not any(name.startswith("runs/") for name in members)
    assert not any(Path(name).name in {"credentials.json", "model.safetensors", "checkpoint.pt", ".env", "notes.txt"}
                   for name in members)
    manifest = json.loads((evidence.output / "manifest.json").read_text())
    assert set(manifest["members"]) == set(members)
    for name, data in members.items():
        assert manifest["members"][name] == {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert manifest["model_checkpoints_in_archive"] is False
    assert len(evidence.uploads) == 3
    assert all(row["uri"].startswith("gs://fast-chunks/o4/test/initial/") for row in evidence.uploads)


def test_final_archive_keeps_reports_receipts_and_events_without_weights_or_data_duplicates(evidence, monkeypatch):
    _run(monkeypatch, evidence, "final")
    members = _archive(evidence)
    expected = {"runs/queue.json"}
    for arm in ("ordinary", "ordinary-nextlat", "rt", "rt-nextlat"):
        expected.update(f"runs/{arm}/{name}" for name in
                        ("report.json", "events.jsonl", "update-000001.receipt.json"))
    assert {name for name in members if name.startswith("runs/")} == expected
    assert not any(name.startswith("prepared-data/") for name in members)
    assert not any(name.endswith((".pt", ".safetensors")) or Path(name).name in (".env", "credentials.json")
                   for name in members)
    assert len(evidence.uploads) == 3


@pytest.mark.parametrize("mismatch", ["source", "data", "preflight"])
def test_source_data_and_preflight_freeze_must_match_before_archival(evidence, monkeypatch, mismatch):
    if mismatch == "source":
        evidence.source["cdrm/pretrained/nextlat.py"] = "b" * 64
    elif mismatch == "data":
        evidence.corpus.manifest_sha256 = "b" * 64
    else:
        _json(evidence.preflight / "report.json", {"status": "failed"})
    with pytest.raises(ValueError):
        _run(monkeypatch, evidence)
    assert not evidence.uploads
    assert not evidence.output.exists()


@pytest.mark.parametrize("status", ["running", "paused", "stopped"])
def test_final_retention_rejects_unfinished_queue(evidence, monkeypatch, status):
    _json(evidence.runs / "queue.json", {"status": status})
    with pytest.raises(ValueError, match="unfinished queue"):
        _run(monkeypatch, evidence, "final")
    assert not evidence.uploads


@pytest.mark.parametrize("member", ["source", "data", "dataset_card", "report"])
def test_selected_symlink_members_are_rejected(evidence, monkeypatch, member):
    paths = {"source": evidence.project / "cdrm/pretrained/nextlat.py",
             "data": evidence.data / "train.tokens.bin",
             "dataset_card": evidence.data.parent / "raw/code_search_net/README.md",
             "report": evidence.project / "docs/reports/olmo1b-o4/protocol.md"}
    path = paths[member]
    original = path.read_bytes()
    path.unlink()
    outside = _write(evidence.project.parent, "outside.txt", "unapproved linked bytes")
    outside.write_bytes(original)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="regular explicit files"):
        _run(monkeypatch, evidence)
    assert not evidence.uploads
