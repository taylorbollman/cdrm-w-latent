"""CPU-only artifact checks for the fixed-budget architecture report."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rt_a5_reference_report as reporter


def metric(update, role):
    length, words = (12 if role == "dev" else 36), 102400
    exact = ([words] * 12 + [80000, 20000, 1000, 1] + [0] * 20)[:length]
    state = ([words] * 12 + [80000, 21000, 2000, 1000] + [1700] * 20)[:length]
    return {"update": update, "role": role, "length": length, "rows": words,
            "evaluated_rows": words, "tokens": words * length,
            "isolated_state_accuracy": [x / words for x in state],
            "cumulative_prefix_exactness": [x / words for x in exact],
            "per_position_ce": [2.] * length, "ce": 2.,
            "token_accuracy": sum(state) / (words * length),
            "whole_word_exact_match": exact[-1] / words,
            "final_state_accuracy": state[-1] / words}


@pytest.fixture(scope="module")
def fixture_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference_report")
    data = root / "data"
    data.mkdir(); (data / "manifest.json").write_text('{"fixture": true}\n')
    data_sha = reporter.hash_file(data / "manifest.json")["sha256"]
    source_text = b"fixture historical source\n"
    historical = {"scripts/fixture.py": hashlib.sha256(source_text).hexdigest()}
    reference_sources = {**historical, **{name: hashlib.sha256(name.encode()).hexdigest()
                                       for name in reporter.NEW_SOURCES}}
    histories = []
    for update in range(1, 100001):
        histories.append(json.dumps({"update": update, "examples_seen": update * 1024,
            "order_chain": f"{update:064x}", "seconds": .05, "loss": .5,
            "token_accuracy": .8, "whole_word_exact": .6, "grad_norm": 1.}))
    directories = {}
    for name, architecture, start, endpoint in (
            ("seq_pilot", "seq", 0, 10000), ("seq", "seq", 10000, 100000),
            ("rt_pilot", "rt", 0, 10000), ("rt", "rt", 10000, 100000),
            ("reference", "reference_gpt", 0, 100000)):
        directory = root / name; directory.mkdir(); directories[name] = directory
        source = reference_sources if architecture == "reference_gpt" else historical
        for path in source:
            target = directory / "source" / path; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_text if path in historical else path.encode())
        cfg = ({"n_layer": 2, "n_embd": 512, "n_head": 8, "mlp_hidden_width": 1408,
                "activation": "swiglu", "normalization": "rms_norm", "qk_normalization": False,
                "rope": True, "alibi": False, "weight_tying": False, "vocab_size": 60,
                "initialization": "normal", "initialization_std": .02,
                "recurrent": False, "nextlat": False} if architecture == "reference_gpt" else
               {"block_type": "recurrent" if architecture == "rt" else "sequential",
                "n_layers": 2, "d_model": 512, "n_heads": 8, "mlp_hidden_size": 2048,
                "activation_type": "gelu", "alibi": True, "rope": False,
                "attention_layer_norm": True, "cdrm_enabled": False, "recurrent_layers": None,
                "recurrent_write_rho": 1., "weight_tying": False, "vocab_size": 60,
                "reference_eager": True})
        schema = "rt-a5-reference-gpt-training-v1" if architecture == "reference_gpt" else "rt-a5-training-v1"
        contract = {"schema": schema, "architecture": architecture, "width": 512,
                    "seed": 1234, "data_order_seed": 1234, "batch_size": 1024,
                    "train_rows": 800000, "length": 12, "precision": "fp32", "tf32": False,
                    "compile": False, "cuda_graphs": False, "torch": "fixture", "cuda": "fixture",
                    "device_capability": [9, 0], "word_order": "same frozen order",
                    "optimizer": "AdamW-lr1e-4-betas0.9,0.95-eps1e-8-wd0.01-matrices-clip1",
                    "data_manifest_sha256": data_sha, "source_sha256": reporter.dictionary_sha(source),
                    "model_config": cfg}
        if architecture == "reference_gpt":
            contract.update(training_step="scripts.rt_a5_train.train_step (same function object)",
                            evaluation="scripts.rt_a5_train.evaluate_arrays (same function object)")
        steps = ([0, *reporter.STEPS] if architecture == "reference_gpt" else
                 [0, 10000] if start == 0 else [25000, 50000, 100000])
        checkpoints = []
        for step in steps:
            path = directory / f"step-{step}.pt"; path.write_bytes(f"{architecture}/{step}".encode())
            checkpoints.append({**reporter.hash_file(path), "completed_updates": step})
        parent = None
        if start:
            parent_path = directories[architecture + "_pilot"] / "step-10000.pt"
            parent = {"path": str(parent_path), "sha256": reporter.hash_file(parent_path)["sha256"]}
        report = {"schema": schema, "status": "complete", "confirmation_evaluated": False,
                  "start_update": start, "completed_updates": endpoint, "endpoint": endpoint,
                  "contract": contract, "source_files": source, "parent_checkpoint": parent,
                  "initialization": {"canonical_sha256": ("b" if architecture == "reference_gpt" else "a") * 64,
                                     "parameter_count": 6486528 if architecture == "reference_gpt" else 6357504},
                  "checkpoints": checkpoints, "order_chain": f"{endpoint:064x}",
                  "evaluations": [metric(step, role) for step in steps if step
                                  for role in ("dev", "ood_dev")],
                  "train_seconds": (endpoint - start) * .05,
                  "elapsed_seconds": (endpoint - start) * .05 + 20}
        reporter.write_json(directory / "report.json", report)
        reporter.write_json(directory / "config.json", {"data_dir": str(data)})
        (directory / "history.jsonl").write_text("\n".join(histories[start:endpoint]) + "\n")
    return SimpleNamespace(seq_pilot_dir=directories["seq_pilot"], rt_pilot_dir=directories["rt_pilot"],
                           seq_dir=directories["seq"], rt_dir=directories["rt"],
                           reference_dir=directories["reference"])


def replace_report(directory, mutation):
    path = Path(directory) / "report.json"; original = path.read_bytes()
    packet = json.loads(original); mutation(packet); reporter.write_json(path, packet)
    return path, original


def test_three_histories_are_matched_but_reference_initialization_need_not_match(fixture_runs):
    result = reporter.make_summary(fixture_runs)
    assert result["primary_update"] == 100000
    assert result["checkpoint_updates"] == [10000, 25000, 50000, 100000]
    assert result["budget"]["nominal_training_passes"] == 128
    assert result["budget"]["fraction_of_reference_budget"] == .25
    assert result["arms"]["seq"]["initialization"]["canonical_sha256"] != result["arms"]["reference_gpt"]["initialization"]["canonical_sha256"]
    assert len(result["arms"]["seq"]["training_curve"]) == 1000
    assert result["arms"]["seq"]["training_curve"][-1]["training_seconds"] == pytest.approx(5000.)
    assert result["arms"]["seq"]["curves"]["100000"]["ood_dev"][15]["prefix_exact_count"] == 1
    assert result["arms"]["seq"]["horizons"]["100000"]["first_observed_zero_length"] == 17
    text = reporter.markdown_report(result)
    assert "does not isolate an ALiBi or RoPE effect" in text
    assert "does not make its weights paired" in text
    assert "Final confirmation remains unevaluated" in text


@pytest.mark.parametrize("field,mutation,match", [
    ("seq_dir", lambda p: p.update(status="running"), "completed"),
    ("seq_dir", lambda p: p["parent_checkpoint"].update(sha256="0" * 64), "parent differs"),
    ("seq_dir", lambda p: p["evaluations"][0].update(rows=4096, evaluated_rows=4096), "102400"),
    ("reference_dir", lambda p: p["contract"].update(evaluation="another evaluator"), "existing training/evaluation"),
    ("reference_dir", lambda p: p["contract"]["model_config"].update(rope=False), "model architecture"),
    ("reference_dir", lambda p: p.update(confirmation_evaluated=True), "confirmation"),
])
def test_incompatible_runs_are_rejected(fixture_runs, field, mutation, match):
    path, original = replace_report(getattr(fixture_runs, field), mutation)
    try:
        with pytest.raises(ValueError, match=match):
            reporter.make_summary(fixture_runs)
    finally:
        path.write_bytes(original)


def test_changed_reference_order_is_not_hidden_by_equal_endpoint_hash(fixture_runs):
    path = fixture_runs.reference_dir / "history.jsonl"; original = path.read_bytes()
    lines = original.decode().splitlines(); row = json.loads(lines[0]); row["order_chain"] = "d" * 64
    lines[0] = json.dumps(row); path.write_text("\n".join(lines) + "\n")
    try:
        with pytest.raises(ValueError, match="per-update training data order"):
            reporter.make_summary(fixture_runs)
    finally:
        path.write_bytes(original)


def test_plot_full_zoom_and_e_only_use_the_same_frozen_metrics(fixture_runs, tmp_path, monkeypatch):
    summary = reporter.make_summary(fixture_runs)
    import matplotlib.axes
    original_plot = matplotlib.axes.Axes.plot
    calls = []
    def capture(axis, x, y, *args, **kwargs):
        calls.append((list(x), list(y)))
        return original_plot(axis, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
    figures = reporter.plot_results(summary, tmp_path)
    assert calls[:9] == calls[9:18]
    assert calls[18:21] == calls[:3]
    for name in figures:
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF")


def test_artifact_report_can_be_generated_without_online_side_effects(fixture_runs, tmp_path, monkeypatch):
    output = tmp_path / "report"; mirror = tmp_path / "mirror"
    args = SimpleNamespace(**vars(fixture_runs), output_dir=output, mirror_dir=mirror,
                           no_wandb=True, wandb_group="fixture")
    def no_tracker(**unused):
        raise AssertionError("Explicit local-only report must not create online runs")
    monkeypatch.setattr(reporter, "OnlineTracker", no_tracker)
    # Plot behavior is covered above; avoid rendering a second full set here.
    monkeypatch.setattr(reporter, "plot_results", lambda *_: {})
    result = reporter.run(args)
    assert result["status"] == "complete" and not result["wandb"]["enabled"]
    assert len((output / "metrics.csv").read_text().splitlines()) == 1 + 3 * 4 * 48
    assert (mirror / "report.json").read_bytes() == (output / "report.json").read_bytes()
    assert result["artifacts"]["inputs/reference_gpt-job0-report.json"]["sha256"] == reporter.hash_file(fixture_runs.reference_dir / "report.json")["sha256"]
    with pytest.raises(FileExistsError):
        reporter.run(args)
