"""CPU preparation guards and native readback, with no model or final data."""
import json

import numpy as np
import pytest

from cdrm.mad_data import FUZZY_TASK, file_sha256, load_dataset
from scripts import cdrm_fuzzy_prepare as preparation


def protocol(tmp_path):
    document = {"schema": "cdrm-fuzzy-prospective-protocol-v1", "schedule": {"epochs": 50},
        "data": {"task": FUZZY_TASK, "vocab_size": 16, "multi_query": True,
                 "k_motif_size": 3, "v_motif_size": 3, "native_labels_unchanged": True,
                 "no_rejection_resampling": True,
                 "calibration": preparation.seed_calendar("calibration", 256),
                 "numerical": {str(length): preparation.seed_calendar("numerical", length)
                               for length in preparation.NUMERICAL_LENGTHS}}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(document))
    return path


def test_numerical_preparation_retains_native_masks_sources_and_epoch_order(tmp_path):
    plan = protocol(tmp_path)
    output = tmp_path / "numerical17"
    report = preparation.prepare(output, protocol=plan, protocol_sha256=file_sha256(plan),
                                 role="numerical", length=17, prefix_limit=3)
    assert report["status"] == "complete"
    assert report["final_split_generated"] is False
    assert report["source_snapshots_verified"]
    assert not list(output.glob("**/final*"))
    for split in ("train", "dev"):
        data = load_dataset(output, FUZZY_TASK, split)
        assert len(data) == 128
        assert data.sha256 == report["splits"][split]["dataset_sha256"]
        assert data.manifest["manifest_sha256"] == report["splits"][split]["manifest_sha256"]
    order = np.load(report["epoch_indices"]["path"], allow_pickle=False)
    assert order.shape == (50, 128)
    np.testing.assert_array_equal(np.sort(order, axis=1), np.tile(np.arange(128), (50, 1)))
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        preparation.prepare(output, protocol=plan, protocol_sha256=file_sha256(plan), role="numerical", length=17)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("mutation", ["hash", "seed", "mask", "final"])
def test_invalid_protocol_or_final_role_fails_before_any_draw(tmp_path, monkeypatch, mutation):
    path = protocol(tmp_path)
    document = json.loads(path.read_text())
    if mutation == "seed":
        document["data"]["numerical"]["17"]["dev"]["seed"] += 1
    if mutation == "mask":
        document["data"]["native_labels_unchanged"] = False
    path.write_text(json.dumps(document))
    monkeypatch.setattr(preparation, "generate_dataset", lambda *a, **k: pytest.fail("Unexpected data draw"))
    with pytest.raises(ValueError):
        preparation.prepare(tmp_path / "output", protocol=path,
            protocol_sha256="0" * 64 if mutation == "hash" else file_sha256(path),
            role="final" if mutation == "final" else "numerical", length=17)
    assert not (tmp_path / "output").exists()


def test_seed_calendar_does_not_overlap_roles_or_lengths():
    calendars = [preparation.seed_calendar("calibration", 256)] + [
        preparation.seed_calendar("numerical", length) for length in preparation.NUMERICAL_LENGTHS]
    seeds = [row[split]["seed"] for row in calendars for split in ("train", "dev")]
    seeds.extend(row["shuffle_seed"] for row in calendars)
    assert len(seeds) == len(set(seeds))
    with pytest.raises(ValueError):
        preparation.seed_calendar("calibration", 128)
