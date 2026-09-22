"""Fixed selections and safe recovery for the evaluation-only driver."""
import copy
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from scripts import olmo_o5d_diagnose as diagnostic


def test_grid_preserves_training_precision_context_and_only_development_modes():
    cases = diagnostic.diagnostic_cases()
    assert [(c.section, c.rows, c.max_length, c.passes) for c in cases] == [
        (section, rows, length, passes)
        for section, rows, length in (("short_prefix", 32, 64), ("full_context", 512, 512))
        for passes in (2, 3, 4, None)]
    assert all(c.beta == 1 and c.mode().rt_mode.selected_layers == () for c in cases)
    assert len({c.label for c in cases}) == 8


class Corpus:
    split_sizes = {"dev": 512, "retention_dev": 512}
    def batch(self, split, rows, device):
        rows = list(rows)
        ids = torch.arange(512, device=device).expand(len(rows), -1).clone()
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[-1, 40:] = False
        docs = torch.tensor(rows, device=device)[:, None].expand_as(ids).clone()
        docs[~valid] = -1
        return NextLatBatch(ids, valid, docs, ce_mask=valid.clone())


@pytest.mark.parametrize("section", ("short_prefix", "full_context"))
def test_every_execution_uses_exact_same_inputs_masks_documents_and_target_boundaries(section):
    expected = None
    for case in (c for c in diagnostic.diagnostic_cases() if c.section == section):
        batches = list(diagnostic.selected_batches(Corpus(), "dev", case, batch_size=8, device="cpu"))
        actual = {key: None if getattr(batches[0], key) is None else torch.cat([getattr(b, key) for b in batches])
                  for key in batches[0].__dict__}
        assert actual["input_ids"].shape == (case.rows, case.max_length)
        assert actual["valid_mask"][-1].sum() == 40
        if expected is None:
            expected = actual
        for key, value in actual.items():
            assert value is None if expected[key] is None else torch.equal(value, expected[key])


def test_reserved_splits_and_insufficient_data_are_rejected():
    case = diagnostic.diagnostic_cases()[0]
    with pytest.raises(ValueError, match="development"):
        list(diagnostic.selected_batches(Corpus(), "test", case, device="cpu"))
    corpus = Corpus(); corpus.split_sizes = {"dev": 31}
    with pytest.raises(ValueError, match="Fewer"):
        list(diagnostic.selected_batches(corpus, "dev", case, device="cpu"))


def fixture_resume():
    case = diagnostic.diagnostic_cases()[0]
    identity = {"schema": "test", "configuration": {"batch_size": 8}, "source_hashes": {"source": "sha"}}
    report = {**copy.deepcopy(identity), "status": "paused", "cases": [
        {"endpoint": "source", "case": asdict(case), "label": "source-"+case.label,
         "weights_unchanged": True, "metrics": {"dev": {}, "retention_dev": {}}}]}
    return report, identity


def test_resume_accepts_only_completed_prefix_with_identical_authority():
    report, identity = fixture_resume()
    diagnostic.validate_resume(report, identity)
    report["in_progress_case"] = {"unfinished": True}
    diagnostic.validate_resume(report, identity)  # discarded/recomputed, never treated as complete


@pytest.mark.parametrize("mutation", (
    lambda r: r.update(status="passed"),
    lambda r: r["configuration"].update(batch_size=16),
    lambda r: r["source_hashes"].update(source="other"),
    lambda r: r["cases"][0].update(endpoint="mixed"),
    lambda r: r["cases"][0].update(weights_unchanged=False),
    lambda r: r["cases"][0]["metrics"].pop("retention_dev"),
    lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
))
def test_resume_rejects_changed_lineage_modes_or_incomplete_cases(mutation):
    report, identity = fixture_resume(); mutation(report)
    with pytest.raises(ValueError):
        diagnostic.validate_resume(report, identity)
