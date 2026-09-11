"""CPU-only provenance, count and output checks for the bounded RT continuation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import rt_a5_budget_report as report


def metrics(step, role):
    n, length = report.ROWS, 12 if role == "dev" else 36
    exact = ([n] * 12 + [80000, 20000, 1000, 1] + [0] * 20)[:length]
    state = ([n] * 12 + [80000, 21000, 2000, 1000] + [1700] * 20)[:length]
    return {"update": step, "role": role, "rows": n, "evaluated_rows": n,
            "length": length, "tokens": n * length, "ce": 2., "per_position_ce": [2.] * length,
            "isolated_state_accuracy": [x / n for x in state],
            "cumulative_prefix_exactness": [x / n for x in exact],
            "token_accuracy": sum(state) / (n * length), "whole_word_exact_match": exact[-1] / n,
            "final_state_accuracy": state[-1] / n}


def make_runs(root, endpoint=30000):
    """Opaque checkpoint bytes; real historical reader checks report/source/history hashes."""
    directories = {name: root / name for name in ("pilot", "continuation")}
    source_text = b"fixture executed source\n"
    sources = {"scripts/fixture.py": hashlib.sha256(source_text).hexdigest()}
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract = {"architecture": "rt", "batch_size": 1024, "train_rows": 800000, "length": 12,
                "width": 512, "seed": 1234, "precision": "fp32", "tf32": False, "compile": False,
                "cuda_graphs": False, "source_sha256": source_sha, "data_manifest_sha256": "a" * 64,
                "model_config": {"block_type": "recurrent", "n_layers": 2, "activation_type": "gelu",
                                 "alibi": True, "cdrm_enabled": False, "recurrent_layers": None}}
    packets = {}
    for phase, first, last in (("pilot", 0, 10000), ("continuation", 10000, endpoint)):
        directory = directories[phase]
        (directory / "source/scripts").mkdir(parents=True)
        (directory / "source/scripts/fixture.py").write_bytes(source_text)
        selected = report.PILOT_STEPS if phase == "pilot" else report.checkpoint_steps(endpoint)[2:]
        checkpoints = []
        for step in selected:
            path = directory / f"step-{step}.pt"
            path.write_bytes(f"opaque retained checkpoint {step}".encode())
            checkpoints.append({**report.hash_file(path), "completed_updates": step})
        with (directory / "history.jsonl").open("w") as stream:
            for update in range(first + 1, last + 1):
                stream.write(json.dumps({"update": update, "examples_seen": update * 1024,
                             "seconds": .05, "loss": .5, "token_accuracy": .8, "whole_word_exact": .6,
                             "grad_norm": 1., "order_chain": f"{update:064x}"}) + "\n")
        parent = None if phase == "pilot" else {k: packets["pilot"]["checkpoints"][-1][k] for k in ("path", "sha256")}
        packet = {"schema": "rt-a5-training-v1", "status": "complete", "confirmation_evaluated": False,
                  "contract": contract, "source_files": sources, "start_update": first, "completed_updates": last,
                  "endpoint": last, "order_chain": f"{last:064x}", "parent_checkpoint": parent,
                  "initialization": {"canonical_sha256": "b" * 64, "parameter_count": 6357504},
                  "checkpoints": checkpoints,
                  "evaluations": [metrics(step, role) for step in selected for role in ("dev", "ood_dev")],
                  "train_seconds": (last - first) * .05, "elapsed_seconds": (last - first) * .05 + 20,
                  "wandb": {"run_url": f"https://example.test/{phase}"}}
        report.write_json(directory / "report.json", packet)
        packets[phase] = packet
    return directories, packets


def edit_packet(directory, mutation):
    path = directory / "report.json"
    packet = json.loads(path.read_text())
    mutation(packet)
    report.write_json(path, packet)


def test_default_budget_is_100k_total_and_25_percent_of_reference(tmp_path):
    dirs, _ = make_runs(tmp_path, endpoint=100000)
    summary = report.make_summary(dirs["pilot"], dirs["continuation"])
    assert summary["primary_update"] == 100000
    assert summary["checkpoint_updates"] == [5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    assert summary["budget"]["total_word_presentations"] == 102400000
    assert summary["budget"]["additional_updates"] == 90000
    assert summary["budget"]["nominal_training_passes"] == 128
    assert summary["budget"]["fraction_of_reference_budget"] == .25
    assert sum(b["updates_in_bin"] for b in summary["training_curve"]) == 100000
    assert summary["training_curve"][-1]["training_seconds"] == pytest.approx(5000.)
    assert summary["curves"]["100000"]["ood_dev"][15]["prefix_exact_count"] == 1
    assert summary["horizons"]["100000"]["first_observed_zero_length"] == 17
    text = report.markdown_report(summary)
    assert "25% of the 400,000-update reference budget" in text
    assert "Final confirmation remains unevaluated" in text
    assert "not replace the primary endpoint" in text


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(status="running"), "completed"),
    (lambda p: p.update(confirmation_evaluated=True), "confirmation"),
    (lambda p: p["parent_checkpoint"].update(sha256="wrong"), "parent SHA256"),
    (lambda p: p["contract"].update(seed=987), "contracts differ"),
    (lambda p: p["initialization"].update(canonical_sha256="wrong"), "lineage differs"),
    (lambda p: p["evaluations"].pop(0), "full dev evaluation at update 20000"),
    (lambda p: p["evaluations"][0].update(evaluated_rows=4096), "full 102400"),
    (lambda p: p.update(train_seconds=1.), "training time disagrees"),
])
def test_incomplete_or_incompatible_continuation_rejected(tmp_path, mutation, match):
    dirs, _ = make_runs(tmp_path)
    edit_packet(dirs["continuation"], mutation)
    with pytest.raises(ValueError, match=match):
        report.make_summary(dirs["pilot"], dirs["continuation"], endpoint=30000)


def test_missing_duplicate_history_or_changed_checkpoint_rejected(tmp_path):
    dirs, packets = make_runs(tmp_path)
    path = dirs["continuation"] / "history.jsonl"
    original = path.read_text()
    rows = original.splitlines()
    path.write_text("\n".join(rows[1:]) + "\n")
    with pytest.raises(ValueError, match="missing, duplicate"):
        report.make_summary(dirs["pilot"], dirs["continuation"], endpoint=30000)
    path.write_text(rows[0] + "\n" + original)
    with pytest.raises(ValueError, match="missing, duplicate"):
        report.make_summary(dirs["pilot"], dirs["continuation"], endpoint=30000)
    path.write_text(original)
    Path(packets["continuation"]["checkpoints"][0]["path"]).write_bytes(b"changed diagnostic checkpoint")
    with pytest.raises(ValueError, match="Checkpoint hash/size differs at update 20000"):
        report.make_summary(dirs["pilot"], dirs["continuation"], endpoint=30000)


def test_endpoint_is_explicit_and_must_match_completed_run(tmp_path):
    dirs, _ = make_runs(tmp_path)
    with pytest.raises(ValueError, match="10000->100000"):
        report.make_summary(dirs["pilot"], dirs["continuation"])
    with pytest.raises(ValueError, match="multiple of 10000"):
        report.make_summary(dirs["pilot"], dirs["continuation"], endpoint=35000)


def test_counts_validate_scalars_and_cumulative_semantics():
    metric = metrics(20000, "ood_dev")
    rows = report.evaluation_rows(metric)
    assert rows[-1]["M"] == pytest.approx(metric["token_accuracy"])
    assert rows[-1]["token_denominator"] == 102400 * 36
    assert rows[16]["prefix_exact_count"] == 0 and rows[16]["A"] > 0
    assert not any("M_low" in key or "M_high" in key for key in rows[0])
    metric["ce"] = 3
    with pytest.raises(ValueError, match="scalar ce disagrees"):
        report.evaluation_rows(metric)
    metric = metrics(20000, "ood_dev")
    metric["cumulative_prefix_exactness"][17] = 1 / 102400
    with pytest.raises(ValueError, match="cumulative correctness"):
        report.evaluation_rows(metric)
    metric = metrics(20000, "ood_dev")
    metric["isolated_state_accuracy"][17] = .123456
    with pytest.raises(ValueError, match="integer count"):
        report.evaluation_rows(metric)


def test_report_exports_matching_counts_curves_and_online_metrics(tmp_path, monkeypatch):
    dirs, _ = make_runs(tmp_path)
    calls = []

    class FakeTracker:
        def __init__(self, **kwargs):
            assert kwargs["project"] == "rt-a5-state-tracking" and kwargs["entity"] == "taylorbollman"
            self.record = {"run_url": "https://example.test/budget-report", "status": "pending"}
        def start(self, config):
            calls.append(("start", config))
        def log(self, values):
            calls.append(("log", values))
        def summary(self, values):
            calls.append(("summary", values))
        def finish(self, succeeded):
            self.record["status"] = "synced" if succeeded else "failed"

    monkeypatch.setattr(report, "OnlineTracker", FakeTracker)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=lambda p: {"image": p}, Table=lambda **kw: {"table": kw}))
    output = tmp_path / "output"
    args = SimpleNamespace(pilot_dir=dirs["pilot"], run_dir=dirs["continuation"], endpoint=30000,
                           output_dir=output, wandb_group="fixture")
    summary = report.run(args)
    assert summary["status"] == "complete"
    assert len((output / "length-curves.csv").read_text().splitlines()) == 1 + 5 * 48
    assert len((output / "training-curves.csv").read_text().splitlines()) == 1 + 300
    saved = json.loads((output / "plot-data.json").read_text())
    assert saved["curves"] == summary["curves"]
    for name in ("final-length-curves", "length-accuracy-vs-updates", "training-curves"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    logged = [row for kind, row in calls if kind == "log" and "update" in row]
    assert [row["update"] for row in logged] == list(range(100, 30001, 100))
    assert logged[-1]["dev/ood_prefix_36/M"] == summary["curves"]["30000"]["ood_dev"][-1]["M"]
    assert summary["artifacts"]["inputs/continuation-report.json"]["sha256"] == report.hash_file(dirs["continuation"] / "report.json")["sha256"]
    with pytest.raises(FileExistsError):
        report.run(args)
