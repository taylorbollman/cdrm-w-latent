"""CPU-only provenance, equal-budget and plotting checks for the RoPE report."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rt_a5_rope_report as reporter
from test_rt_a5_reference_report import fixture_runs, metric, replace_report


@pytest.fixture(scope="module")
def rope_runs(fixture_runs):
    directory = fixture_runs.seq_dir.parent / "seq_rope"
    directory.mkdir()
    seq = json.loads((fixture_runs.seq_pilot_dir / "report.json").read_text())
    report = copy.deepcopy(seq)
    report.update(schema=reporter.TRAIN_SCHEMA, completed_updates=50000, endpoint=50000,
                  order_chain=f"{50000:064x}", train_seconds=2500., elapsed_seconds=2550.)
    report["contract"].update(schema=reporter.TRAIN_SCHEMA, architecture="seq_rope",
                              training_step="scripts.rt_a5_train.train_step (same function object)",
                              evaluation="scripts.rt_a5_train.evaluate_arrays (same function object)",
                              objective="scripts.rt_a5_common.task_loss: unshifted CE mean over B*T")
    report["contract"]["model_config"].update(alibi=False, rope=True)
    initial = report["initialization"]
    report["initialization"] = {**initial, "schema": "rt-a5-seq-rope-initialization-v1",
        "architecture": "seq_rope", "paired_base_architecture": "seq",
        "paired_base_canonical_sha256": initial["canonical_sha256"],
        "paired_base_initialization": copy.deepcopy(initial),
        "config_delta": {"alibi": {"from": True, "to": False}, "rope": {"from": False, "to": True}}}
    source = report["source_files"]
    source.update({name: hashlib.sha256(name.encode()).hexdigest() for name in reporter.NEW_SOURCES})
    report["contract"]["source_sha256"] = reporter.dictionary_sha(source)
    for relative in source:
        path = directory / "source" / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode() if relative in reporter.NEW_SOURCES else b"fixture historical source\n")
    checkpoints = []
    for step in (0, *reporter.STEPS):
        path = directory / f"step-{step}.pt"; path.write_bytes(f"seq_rope/{step}".encode())
        checkpoints.append({**reporter.hash_file(path), "completed_updates": step})
    report["checkpoints"] = checkpoints
    report["evaluations"] = [metric(step, role) for step in reporter.STEPS for role in ("dev", "ood_dev")]
    reporter.write_json(directory / "report.json", report)
    (directory / "config.json").write_bytes((fixture_runs.seq_pilot_dir / "config.json").read_bytes())
    history = (fixture_runs.seq_pilot_dir / "history.jsonl").read_text().splitlines()
    history += (fixture_runs.seq_dir / "history.jsonl").read_text().splitlines()[:40000]
    (directory / "history.jsonl").write_text("\n".join(history)+"\n")
    return SimpleNamespace(**vars(fixture_runs), rope_dir=directory)


def test_primary_pair_uses_exact_50k_budget_and_initialization(rope_runs):
    result = reporter.make_summary(rope_runs)
    assert result["primary_update"] == 50000
    assert result["checkpoint_updates"] == [10000,25000,50000]
    assert result["primary_arms"] == ["seq", "seq_rope"]
    assert result["budget"]["nominal_training_passes"] == 64
    assert result["budget"]["training_token_presentations_per_arm"] == 614400000
    for arm, record in result["arms"].items():
        assert record["selected_checkpoint_update"] == 50000
        assert record["source_completed_updates"] == (50000 if arm == "seq_rope" else 100000)
        assert list(record["curves"]) == ["10000", "25000", "50000"]
        assert "100000" not in record["checkpoints"]
        assert len(record["training_curve"]) == 500
        assert record["training_curve"][-1]["update"] == 50000
        assert record["order_chain"] == f"{50000:064x}"
        assert record["timing"]["training_seconds_through_50k"] == pytest.approx(2500.)
    assert result["arms"]["seq"]["initialization"] != result["arms"]["seq_rope"]["initialization"]
    assert result["arms"]["seq"]["initialization"]["canonical_sha256"] == result["arms"]["seq_rope"]["initialization"]["canonical_sha256"]
    assert result["arms"]["seq"]["initialization"]["canonical_sha256"] != result["arms"]["reference_gpt"]["initialization"]["canonical_sha256"]
    text = reporter.markdown_report(result)
    assert "same Transformer with ALiBi replaced by RoPE, both at 50,000 updates" in text
    assert "no later metrics or training updates" in text
    assert "Stop after this 50k experiment" in text


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(status="running"), "completed"),
    (lambda p: p.update(endpoint=100000), "exact 0->50000"),
    (lambda p: p.update(parent_checkpoint={"sha256": "0"*64}), "paired initialization"),
    (lambda p: p["contract"]["model_config"].update(activation_type="swiglu"), "differ only"),
    (lambda p: p["contract"].update(evaluation="different evaluator"), "evaluation"),
    (lambda p: p["initialization"].update(canonical_sha256="0"*64), "paired initialization"),
    (lambda p: p["initialization"]["paired_base_initialization"].update(parameter_count=1), "original SEQ lineage"),
    (lambda p: p["evaluations"][0].update(rows=4096, evaluated_rows=4096), "102400"),
    (lambda p: p.update(confirmation_evaluated=True), "confirmation"),
])
def test_incompatible_rope_runs_are_rejected(rope_runs, mutation, match):
    path, original = replace_report(rope_runs.rope_dir, mutation)
    try:
        with pytest.raises(ValueError, match=match):
            reporter.make_summary(rope_runs)
    finally:
        path.write_bytes(original)


def test_primary_pre50k_order_change_is_rejected(rope_runs):
    path = rope_runs.rope_dir / "history.jsonl"; original = path.read_bytes()
    rows = original.decode().splitlines(); row = json.loads(rows[12345]); row["order_chain"] = "d" * 64
    rows[12345] = json.dumps(row); path.write_text("\n".join(rows)+"\n")
    try:
        with pytest.raises(ValueError, match="per-update training data order"):
            reporter.make_summary(rope_runs)
    finally:
        path.write_bytes(original)


def test_source_snapshot_changes_are_rejected(rope_runs):
    path = rope_runs.rope_dir / "source/scripts/fixture.py"; original = path.read_bytes()
    path.write_bytes(b"changed historical source\n")
    try:
        with pytest.raises(ValueError, match="source snapshot changed"):
            reporter.make_summary(rope_runs)
    finally:
        path.write_bytes(original)


def test_100k_outcomes_do_not_enter_50k_curves(rope_runs):
    def replace_late_metrics(report):
        for m in report["evaluations"]:
            if m["update"] == 100000:
                m.update(isolated_state_accuracy=[1.] * m["length"],
                         cumulative_prefix_exactness=[1.] * m["length"],
                         token_accuracy=1., whole_word_exact_match=1., final_state_accuracy=1.)
    path, original = replace_report(rope_runs.seq_dir, replace_late_metrics)
    try:
        result = reporter.make_summary(rope_runs)
        assert result["arms"]["seq"]["curves"]["50000"]["ood_dev"][-1]["E"] == 0.
        assert result["arms"]["seq"]["checkpoint_metrics"]["50000"]["ood_dev"]["whole_word_exact_match"] == 0.
    finally:
        path.write_bytes(original)


def test_primary_differences_have_same_position_denominators(rope_runs):
    def alter_one_position(report):
        m = next(m for m in report["evaluations"] if m["update"] == 50000 and m["role"] == "ood_dev")
        m["isolated_state_accuracy"][12] += 1000/102400
        m["cumulative_prefix_exactness"][12] += 1000/102400
        m["token_accuracy"] = sum(m["isolated_state_accuracy"])/36
    path, original = replace_report(rope_runs.rope_dir, alter_one_position)
    try:
        result = reporter.make_summary(rope_runs)
        row = next(r for r in result["primary_differences"] if r["update"] == 50000 and r["role"] == "ood_dev" and r["length"] == 13)
        assert row["E_delta_pp"] == pytest.approx(100*1000/102400)
        assert row["A_delta_pp"] == pytest.approx(100*1000/102400)
        assert row["M_delta_pp"] == pytest.approx(100*1000/(102400*13))
    finally:
        path.write_bytes(original)


def test_full_zoom_and_e_only_share_the_same_50k_data(rope_runs, tmp_path, monkeypatch):
    summary = reporter.make_summary(rope_runs)
    import matplotlib.axes
    original_plot = matplotlib.axes.Axes.plot
    calls = []
    def capture(axis, x, y, *args, **kwargs):
        calls.append((list(x), list(y)))
        return original_plot(axis, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
    figures = reporter.plot_results(summary, tmp_path)
    assert calls[:12] == calls[12:24]
    assert calls[24:28] == calls[:4]
    assert all(max(x) == 50 for x, _ in calls[28:])
    for name in figures:
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF")


def test_local_artifacts_include_context_labels_and_primary_differences(rope_runs, tmp_path, monkeypatch):
    args = SimpleNamespace(**vars(rope_runs), output_dir=tmp_path/"report", mirror_dir=tmp_path/"mirror",
                           no_wandb=True, wandb_group="fixture")
    def no_tracker(**unused):
        raise AssertionError("Local-only tests must not start W&B")
    monkeypatch.setattr(reporter, "OnlineTracker", no_tracker)
    monkeypatch.setattr(reporter, "plot_results", lambda *_: {})
    result = reporter.run(args)
    assert result["status"] == "complete" and not result["wandb"]["enabled"]
    assert len((args.output_dir/"metrics.csv").read_text().splitlines()) == 1+4*3*48
    assert len((args.output_dir/"primary-differences.csv").read_text().splitlines()) == 1+3*48
    assert (args.output_dir/"report.json").read_bytes() == (args.mirror_dir/"report.json").read_bytes()
    assert result["arms"]["seq_rope"]["comparison_role"] == "primary"
    assert result["arms"]["reference_gpt"]["comparison_role"] == "context"
    with pytest.raises(FileExistsError):
        reporter.run(args)
