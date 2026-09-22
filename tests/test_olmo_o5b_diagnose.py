"""Bounded post-hoc evaluation contract, without GPU or real artifact loads."""

import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA, parameter_layout
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTOnlineMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_o5b_common import ObservedFBTLM
from scripts import olmo_o5b_diagnose as diagnostic


def test_fixed_grid_has_only_prespecified_settings_and_no_rt():
    cases = diagnostic.diagnostic_cases()
    assert len(cases) == 13 and len({case.label for case in cases}) == 13
    assert [(c.beta, c.passes, c.rows, c.max_length) for c in cases[:5]] == [
        (b, 2, 128, 512) for b in (0, .25, .5, .75, 1)]
    assert [(c.beta, c.passes, c.rows, c.max_length) for c in cases[5:]] == [
        (b, k, 32, 64) for b in (.5, 1) for k in (2, 3, 4, None)]
    assert all(case.mode().rt_mode.selected_layers == () for case in cases)
    assert sum(isinstance(case.mode(), FBTOnlineMode) for case in cases) == 2


class Corpus:
    split_sizes = {"dev": 128, "retention_dev": 128}
    def batch(self, split, rows, device):
        rows = list(rows)
        ids = torch.tensor([[2+(row % 20)]*80 for row in rows], device=device)
        valid = torch.ones_like(ids, dtype=torch.bool)
        docs = torch.tensor(rows, device=device)[:, None].expand_as(ids).clone()
        return NextLatBatch(ids, valid, docs, ce_mask=valid.clone(), kl_mask=valid.clone())


def test_grid_slices_identical_first_rows_and_prefixes_for_all_finite_online_settings():
    corpus = Corpus()
    reference = None
    for case in diagnostic.diagnostic_cases()[5:]:
        batches = list(diagnostic.selected_batches(corpus, "dev", case, batch_size=8, device="cpu"))
        assert len(batches) == 4 and all(batch.input_ids.shape == (8, 64) for batch in batches)
        values = {name: None if getattr(batches[0], name) is None else torch.cat([getattr(batch, name) for batch in batches])
                  for name in batches[0].__dict__}
        if reference is None:
            reference = values
        for name, value in values.items():
            assert value is None if reference[name] is None else torch.equal(value, reference[name])
        assert torch.equal(values["document_ids"][:, 0], torch.arange(32))
    with pytest.raises(ValueError, match="development"):
        list(diagnostic.selected_batches(corpus, "test", diagnostic.diagnostic_cases()[0], batch_size=8, device="cpu"))
    corpus.split_sizes = {"dev": 31}
    with pytest.raises(ValueError, match="fewer rows"):
        list(diagnostic.selected_batches(corpus, "dev", diagnostic.diagnostic_cases()[5], batch_size=8, device="cpu"))


def test_case_evaluation_preserves_full_pass_metrics_and_document_record_request(monkeypatch):
    calls = []
    def evaluate(model, batches, **kwargs):
        consumed = list(batches)
        calls.append((consumed, kwargs))
        return {"passes": [{"mean_nll": 3}, {"mean_nll": 2}, {"mean_nll": 1}]}
    monkeypatch.setattr(diagnostic, "evaluate_fbt_batches", evaluate)
    model = SimpleNamespace(backbone=SimpleNamespace(readout_weight=torch.ones(1)))
    case = diagnostic.diagnostic_cases()[6]  # beta .5, K3.
    result = diagnostic.evaluate_case(model, Corpus(), {"eval_batch_size": 8, "precision": "fp32"}, case)
    assert set(result["metrics"]) == {"dev", "retention_dev"}
    assert all(len(value["passes"]) == 3 for value in result["metrics"].values())
    assert len(calls) == 2
    assert all(options["include_document_records"] and options["mode"].num_passes == 3 for _, options in calls)


def _results():
    rows = []
    for case in diagnostic.diagnostic_cases():
        metrics = {}
        for split in diagnostic.SPLITS:
            passes = []
            for index in range(case.passes or 1):
                nll = 2+(case.beta-.5)**2 if case.section == "full_beta" else 3-(.7 if case.passes is None else .1*case.passes)
                records = [{"batch_index": 0, "row_index": 0, "document_id": 19, "ce_count": case.max_length-1}]
                passes.append({"mean_nll": nll, "next_token_accuracy": .25, "document_records": records})
            metrics[split] = {"passes": passes}
        rows.append({"case": asdict(case), "label": case.label, "metrics": metrics})
    return rows


def test_report_math_keeps_two_contexts_separate_and_labels_posthoc_selection():
    rows = _results()
    summary = diagnostic.summarize_cases(rows)
    for split in diagnostic.SPLITS:
        assert summary["full_beta"][split]["lowest_observed_beta"] == .5
        assert summary["full_beta"][split]["difference_from_beta0"] == -.25
        assert summary["full_beta"][split]["difference_from_beta1"] == -.25
        assert summary["short_passes"][split][0]["differences_from_finite_K2"] == pytest.approx({"K2": 0, "K3": -.1, "K4": -.2, "online": -.5})
    text = diagnostic.markdown({"cases": rows, "checkpoint": {"sha256": "1"*64}, "weights_unchanged": True})
    assert "post-hoc" in text and "not a confirmatory result" in text
    assert "128 windows, max512" in text and "32 prefixes, max64" in text
    assert "not the separately trained" in text
    altered = copy.deepcopy(rows)
    altered[-1]["metrics"]["dev"]["passes"][0]["document_records"][0]["document_id"] = 20
    with pytest.raises(ValueError, match="identical"):
        diagnostic.summarize_cases(altered)
    with pytest.raises(ValueError, match="fixed ordered grid"):
        diagnostic.summarize_cases(rows[:-1])


def test_tracking_uses_shared_series_with_explicit_beta_and_execution_axes():
    rows = _results()
    full = diagnostic.tracking_metrics(rows[2], 2)
    assert full["diagnostic/beta"] == .5
    assert full["diagnostic/full_beta/dev/final_nll"] == 2
    short = diagnostic.tracking_metrics(rows[-1], 12)
    assert short["diagnostic/short_execution_index"] == 3
    assert short["diagnostic/short_passes/beta1/dev/final_nll"] == pytest.approx(2.3)


def _payload():
    torch.set_num_threads(1)
    model = ObservedFBTLM(OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny())), NextLatConfig(32), enabled=False)
    config = {"tiny_fixture": True}
    report = {"storage_prefix": "gs://fixture", "source_fingerprint": {"checkpoint_sha256": "1"*64},
              "counters": {"optimizer_updates": 4}, "data_cursor": 8}
    payload = {"schema": CHECKPOINT_SCHEMA, "model_type": type(model).__module__+"."+type(model).__qualname__,
               "configuration": {**config, "arm": "fbt", "storage_prefix": "gs://fixture"},
               "source_fingerprint": report["source_fingerprint"], "counters": report["counters"],
               "data_cursor": {"next_window": 8}, "parameter_layout": parameter_layout(model),
               "model": copy.deepcopy(model.state_dict()), "module_training": {n: m.training for n, m in model.named_modules()}}
    return payload, model, config, report


def test_valid_endpoint_payload_returns_exact_strict_model_state_without_loading_optimizer():
    payload, model, config, report = _payload()
    assert diagnostic.validate_checkpoint_payload(payload, model, config, report) is payload["model"]
    assert "optimizer" not in payload  # Evaluation validation never builds/restores one.


def test_weights_only_loader_handles_native_torch_version_metadata_with_scoped_allowlist(tmp_path):
    from torch.torch_version import TorchVersion
    path = tmp_path/"version_metadata.pt"
    torch.save({"version": TorchVersion(str(torch.__version__)), "weight": torch.arange(3)}, path)
    before = list(torch.serialization.get_safe_globals())
    loaded = diagnostic.load_endpoint_payload(path)
    assert loaded["version"] == torch.__version__
    assert torch.equal(loaded["weight"], torch.arange(3))
    assert torch.serialization.get_safe_globals() == before


@pytest.mark.parametrize("invalid", ["class", "configuration", "source", "cursor", "keys", "dtype", "nonfinite", "modules"])
def test_endpoint_payload_rejects_mismatches_before_model_mutation(invalid):
    payload, model, config, report = _payload()
    before = copy.deepcopy(model.state_dict())
    first = next(iter(payload["model"]))
    if invalid == "class": payload["model_type"] = "wrong.Model"
    elif invalid == "configuration": payload["configuration"]["arm"] = "ordinary"
    elif invalid == "source": payload["source_fingerprint"] = {"checkpoint_sha256": "9"*64}
    elif invalid == "cursor": payload["data_cursor"]["next_window"] = 7
    elif invalid == "keys": del payload["model"][first]
    elif invalid == "dtype": payload["model"][first] = payload["model"][first].double()
    elif invalid == "nonfinite": payload["model"][first].flatten()[0] = float("nan")
    elif invalid == "modules": payload["module_training"] = {}
    with pytest.raises(ValueError): diagnostic.validate_checkpoint_payload(payload, model, config, report)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_completed_inputs_require_reporter_validation_source_identity_and_checkpoint_hash(tmp_path, monkeypatch):
    preflight, runs = tmp_path/"preflight", tmp_path/"runs"
    preflight.mkdir(); (runs/"fbt").mkdir(parents=True); (runs/"ordinary").mkdir()
    checkpoint = runs/"fbt"/"update-000007.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = {"source_hashes": {"model.py": "a"*64}, "schedule": {"total_updates": 7}}
    report = {"checkpoints": [{"optimizer_updates": 7, "path": str(checkpoint), "sha256": sha256_file(checkpoint), "size_bytes": checkpoint.stat().st_size}]}
    for path, value in ((preflight/"report.json", {}), (preflight/"configuration.json", config),
                        (runs/"fbt"/"report.json", report), (runs/"ordinary"/"report.json", {})):
        path.write_text(json.dumps(value))
    calls = []
    def validate(a, b):
        calls.append((a, b)); return {"configuration": config}
    monkeypatch.setattr(diagnostic, "validate_runs", validate)
    monkeypatch.setattr(diagnostic, "source_hashes", lambda: config["source_hashes"])
    result = diagnostic.read_completed_inputs(preflight, runs)
    assert result[2] == checkpoint and calls == [(preflight, runs)]
    checkpoint.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="bytes"):
        diagnostic.read_completed_inputs(preflight, runs)
    monkeypatch.setattr(diagnostic, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="frozen O5b"):
        diagnostic.read_completed_inputs(preflight, runs)
    def incomplete(*args): raise ValueError("Require both completed arms")
    monkeypatch.setattr(diagnostic, "validate_runs", incomplete)
    with pytest.raises(ValueError, match="both completed"):
        diagnostic.read_completed_inputs(preflight, runs)


def test_cli_requires_new_output_directory_and_explicit_artifacts(tmp_path):
    arguments = []
    for name in ("preflight", "runs", "data", "artifacts"):
        directory = tmp_path/name
        directory.mkdir()
        arguments += ["--"+name, str(directory)]
    arguments += ["--output-dir", str(tmp_path/"new-diagnostic")]
    args = diagnostic.parse_args(arguments)
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    with pytest.raises(SystemExit): diagnostic.parse_args(arguments)
