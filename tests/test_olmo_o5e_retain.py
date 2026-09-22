"""Evidence retention rejects wrong lineages, remote replacements and stale reports."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_o5e_retain as retain


@pytest.mark.parametrize("suffix", ("", "../20260922T000000Z", "20260922T000000Z/extra", "date", "20260922T000000Z//"))
def test_prefix_rejects_noncanonical_lineage(suffix):
    if suffix.endswith("//"):
        assert retain.checked_prefix(retain.PREFIX_ROOT+suffix) == retain.PREFIX_ROOT+"20260922T000000Z/"
    else:
        with pytest.raises(ValueError): retain.checked_prefix(retain.PREFIX_ROOT+suffix)


def test_prefix_requires_designated_bucket_and_experiment():
    assert retain.checked_prefix(retain.PREFIX_ROOT+"20260922T120000Z") == retain.PREFIX_ROOT+"20260922T120000Z/"
    with pytest.raises(ValueError): retain.checked_prefix("gs://other/20260922T120000Z")


@pytest.mark.parametrize("failure", (None, "missing", "generation", "size", "md5", "sha", "lineage"))
def test_remote_reference_checks_immutable_generation_and_all_checksums(failure):
    record = {"uri": "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/fixture.pt", "generation": "42",
              "size_bytes": 17, "md5_base64": "md5", "sha256": "a"*64}
    blob = SimpleNamespace(generation=42, size=17, md5_hash="md5", metadata={"sha256": "a"*64})
    if failure == "generation": blob.generation = 43
    elif failure == "size": blob.size = 18
    elif failure == "md5": blob.md5_hash = "different"
    elif failure == "sha": blob.metadata["sha256"] = "b"*64
    elif failure == "lineage": record["uri"] = "gs://other/fixture.pt"
    client = SimpleNamespace(bucket=lambda _: SimpleNamespace(get_blob=lambda _: None if failure == "missing" else blob))
    if failure:
        with pytest.raises(ValueError): retain.verify_reference(client, record)
    else:
        assert retain.verify_reference(client, record) == {**record, "reused_without_upload": True}


@pytest.mark.parametrize("failure", (None, "comparison", "input", "helper", "figure", "markdown"))
def test_final_retention_rebuilds_result_and_checks_rendered_evidence(tmp_path, monkeypatch, failure):
    from scripts import olmo_o5e_report as reporter
    paths = {}
    for name in ("preflight", "configuration", "ordinary", "online_reference", "fusion_preflight",
                 "fusion_config", "fusion_code", "fusion_mixed"):
        paths[name] = tmp_path/(name+".json")
        paths[name].write_text(json.dumps({"name": name}))
    monkeypatch.setattr(retain, "comparison_inputs", lambda *_: paths)
    monkeypatch.setattr(retain, "ROOT", tmp_path)
    monkeypatch.setattr(reporter.fusion_report, "build_comparison", lambda *_: {"fusion": "validated"})
    def rebuild(*args):
        assert args[0] == {"name": "preflight"} and args[2] == {"name": "ordinary"}
        assert args[3] == {"fusion": "validated"}
        return {"status": "completed", "paired": {"estimate": .01}}
    monkeypatch.setattr(reporter, "build_comparison", rebuild)
    monkeypatch.setattr(reporter, "markdown", lambda result: "validated report\n")
    helper = tmp_path/"helper.py"; helper.write_text("fixture")
    for name in reporter.FIGURES: (tmp_path/name).write_bytes(b"figure")
    rendered = {"status": "completed", "paired": {"estimate": .01},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "report_source_sha256": sha256_file(reporter.__file__), "helper_source_sha256": {"helper.py": sha256_file(helper)},
        "figure_sha256": {name: sha256_file(tmp_path/name) for name in reporter.FIGURES}}
    (tmp_path/"results.md").write_text("validated report\n")
    if failure == "comparison": rendered["paired"]["estimate"] = .02
    elif failure == "input": paths["ordinary"].write_text("{\"different\":true}")
    elif failure == "helper": helper.write_text("different")
    elif failure == "figure": (tmp_path/reporter.FIGURES[0]).write_bytes(b"different")
    elif failure == "markdown": (tmp_path/"results.md").write_text("stale")
    (tmp_path/"final-comparison.json").write_text(json.dumps(rendered))
    if failure:
        with pytest.raises((ValueError, AssertionError)):
            retain.validate_final(tmp_path, tmp_path, tmp_path)
    else:
        result, actual_paths = retain.validate_final(tmp_path, tmp_path, tmp_path)
        assert result == {"name": "ordinary"} and actual_paths == paths
