"""CPU artifact tests: pairing guards, exact plot inputs and retained outputs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import rt_a5_report as report


def digest(value):
    return hashlib.sha256(value).hexdigest()


def metric(length, offset=0):
    accuracy = [0.9 - 0.03 * index + offset for index in range(length)]
    exact, product = [], 1.0
    for value in accuracy:
        product *= value
        exact.append(product)
    return {"rows": 102400, "tokens": 102400 * length, "length": length,
            "ce": 2.0 - offset, "token_accuracy": sum(accuracy) / length,
            "whole_word_exact_match": exact[-1], "final_state_accuracy": accuracy[-1],
            "isolated_state_accuracy": accuracy, "cumulative_prefix_exactness": exact,
            "per_position_ce": [2.0 - offset] * length}


def make_pair(root, endpoint=205):
    dirs = {}
    for arm in ("seq", "rt"):
        directory = root / arm
        directory.mkdir()
        dirs[arm] = directory
        source = directory / "source/scripts/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fixture source only\n")
        sources = {"scripts/fixture.py": digest(source.read_bytes())}
        contract = {"architecture": arm, "width": 64, "seed": 9, "data_order_seed": 8,
                    "batch_size": 1024, "length": 3, "train_rows": 800000,
                    "data_manifest_sha256": "a" * 64,
                    "source_sha256": digest(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
                    "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
                    "model_config": {"block_type": "sequential" if arm == "seq" else "recurrent",
                                     "n_layers": 2, "activation_type": "gelu", "alibi": True,
                                     "cdrm_enabled": False, "recurrent_layers": None}}
        checkpoint = directory / "endpoint.pt"
        checkpoint.write_bytes(f"opaque retained {arm} checkpoint bytes".encode())
        history = [{"update": update, "examples_seen": update * 1024,
                    "order_chain": digest(str(update).encode()), "seconds": 1 if arm == "seq" else 2,
                    "loss": 4 - update / 1000, "token_accuracy": .1 + update / 1000,
                    "whole_word_exact": .01, "grad_norm": 1.0} for update in range(1, endpoint + 1)]
        (directory / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        packet = {"schema": report.TRAIN_SCHEMA, "status": "complete", "confirmation_evaluated": False,
                  "contract": contract, "endpoint": endpoint, "completed_updates": endpoint,
                  "start_update": 0, "order_chain": history[-1]["order_chain"], "source_files": sources,
                  "initialization": {"canonical_sha256": "b" * 64, "parameter_count": 106560},
                  "checkpoints": [{**report.hash_file(checkpoint), "completed_updates": endpoint}],
                  "evaluations": [{**metric(length, .01 if arm == "rt" else 0), "role": role, "update": endpoint}
                                  for role, length in (("dev", 3), ("ood_dev", 6))]}
        report.write_json(directory / "report.json", packet)
    return dirs


def edit_report(directory, mutate):
    path = directory / "report.json"
    value = json.loads(path.read_text())
    mutate(value)
    report.write_json(path, value)


def test_exact_pair_and_bins_preserve_all_observations(tmp_path):
    dirs = make_pair(tmp_path)
    summary = report.compare_runs(dirs["seq"], dirs["rt"])
    assert summary["completed_updates"] == 205 and summary["confirmation_evaluated"] is False
    assert summary["arms"]["seq"]["inputs"]["history"]["sha256"] == digest((dirs["seq"] / "history.jsonl").read_bytes())
    seq, rt = [summary["arms"][name]["training_curve"] for name in ("seq", "rt")]
    assert [row["updates_in_bin"] for row in seq] == [100, 100, 5]
    assert [row["update"] for row in seq] == [100, 200, 205]
    assert [row["training_seconds"] for row in seq] == [100, 200, 205]
    assert [row["training_seconds"] for row in rt] == [200, 400, 410]
    assert seq[0]["loss"] == pytest.approx(4 - 50.5 / 1000)
    assert seq[-1]["loss"] == pytest.approx(4 - 203 / 1000)
    markdown = report.markdown_report(summary)
    assert "one paired development seed" in markdown
    assert "400,000-update" in markdown and "Final confirmation remains unevaluated" in markdown


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.update(status="running"), "completed"),
    (lambda p: p.update(confirmation_evaluated=True), "confirmation"),
    (lambda p: p["contract"].update(seed=10), "contracts differ"),
    (lambda p: p["contract"].update(data_order_seed=10), "contracts differ"),
    (lambda p: p["contract"].update(data_manifest_sha256="x" * 64), "contracts differ"),
    (lambda p: p["initialization"].update(canonical_sha256="c" * 64), "initialization differs"),
    (lambda p: p.update(order_chain="d" * 64), "order chain differs"),
    (lambda p: p["contract"].update(precision="bf16"), "requires precision"),
    (lambda p: p["evaluations"][0].update(whole_word_exact_match=.5), "scalars disagree"),
])
def test_incompatible_or_unfinished_pairs_are_rejected(tmp_path, mutate, match):
    dirs = make_pair(tmp_path, endpoint=5)
    edit_report(dirs["rt"], mutate)
    with pytest.raises(ValueError, match=match):
        report.compare_runs(dirs["seq"], dirs["rt"])


def test_checkpoint_source_history_and_row_count_guards(tmp_path):
    dirs = make_pair(tmp_path, endpoint=5)
    checkpoint = dirs["rt"] / "endpoint.pt"
    saved = checkpoint.read_bytes()
    checkpoint.write_bytes(saved + b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        report.compare_runs(dirs["seq"], dirs["rt"])
    checkpoint.write_bytes(saved)
    source = dirs["rt"] / "source/scripts/fixture.py"
    saved = source.read_bytes()
    source.write_bytes(b"different implementation")
    with pytest.raises(ValueError, match="source snapshot"):
        report.compare_runs(dirs["seq"], dirs["rt"])
    source.write_bytes(saved)
    with pytest.raises(ValueError, match="Expected 4096"):
        report.compare_runs(dirs["seq"], dirs["rt"], expected_eval_rows=4096)
    history = dirs["rt"] / "history.jsonl"
    rows = history.read_text().splitlines()
    history.write_text("\n".join(rows[1:]) + "\n")
    with pytest.raises(ValueError, match="missing, duplicate"):
        report.compare_runs(dirs["seq"], dirs["rt"])


def test_report_exports_actual_plot_data_and_logs_figures(tmp_path, monkeypatch):
    dirs = make_pair(tmp_path)
    calls = []
    class FakeTracker:
        def __init__(self, **kwargs):
            self.record = {"run_url": "https://wandb.example/fixture", "status": "pending"}
        def start(self, config):
            calls.append(("start", config))
        def log(self, values):
            calls.append(("log", values))
        def summary(self, values):
            calls.append(("summary", values))
        def finish(self, succeeded):
            self.record["status"] = "synced" if succeeded else "failed"
    monkeypatch.setattr(report, "OnlineTracker", FakeTracker)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=lambda path: {"fixture_image": path}))
    output = tmp_path / "output"
    result = report.run(SimpleNamespace(seq_dir=dirs["seq"], rt_dir=dirs["rt"], output_dir=output,
                                       wandb_project="fixture", wandb_group="fixture", expected_eval_rows=102400))
    assert result["status"] == "complete"
    for name in ("length-generalization", "training-curves"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    saved = json.loads((output / "plot-data.json").read_text())
    assert saved["arms"]["rt"]["endpoint_metrics"] == result["arms"]["rt"]["endpoint_metrics"]
    assert any("report/length-generalization" in values for kind, values in calls if kind == "log")
    with pytest.raises(FileExistsError):
        report.run(SimpleNamespace(seq_dir=dirs["seq"], rt_dir=dirs["rt"], output_dir=output,
                                  wandb_project="fixture", wandb_group="fixture", expected_eval_rows=102400))
