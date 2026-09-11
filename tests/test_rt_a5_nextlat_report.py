"""Report guards and arithmetic on saved records; no model execution or W&B."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import rt_a5_nextlat_report as reporter


def metric(role="ood_dev", update=10000, rows=102400):
    length = 36 if role == "ood_dev" else 12
    state = [rows // 2] * length
    exact = [max(0, rows // 2 - t * (rows // 50)) for t in range(length)]
    return {"role": role, "update": update, "rows": rows, "evaluated_rows": rows,
            "length": length, "tokens": rows * length, "route": "backbone_only",
            "isolated_state_accuracy": [n / rows for n in state],
            "cumulative_prefix_exactness": [n / rows for n in exact],
            "per_position_ce": [2.0] * length, "ce": 2.0,
            "token_accuracy": sum(state) / (rows * length),
            "final_state_accuracy": state[-1] / rows, "whole_word_exact_match": exact[-1] / rows}


def contract(architecture="rt", augmented=False):
    value = {"architecture": architecture, "schema": "rt-a5-nextlat-training-v1" if augmented else "rt-a5-training-v1",
             "width": 512, "seed": 1234, "data_order_seed": 1234, "batch_size": 1024,
             "length": 12, "train_rows": 800000, "precision": "fp32", "tf32": False,
             "compile": False, "cuda_graphs": False, "optimizer": reporter.OPTIMIZER + ("-all-hybrid" if augmented else ""),
             "source_sha256": "c" * 64, "data_manifest_sha256": "d" * 64,
             "model_config": {"block_type": "recurrent" if architecture == "rt" else "sequential",
                               "n_layers": 2, "d_model": 512, "n_heads": 8, "mlp_hidden_size": 2048,
                               "activation_type": "gelu", "alibi": True, "recurrent_layers": None,
                               "cdrm_enabled": False, "reference_eager": True, "recurrent_write_rho": 1.0,
                               "vocab_size": 60, "weight_tying": False}}
    if augmented:
        value.update(predictor_seed=1235, evaluation_route="backbone_only", latent_rollout_evaluated=False,
                     objective={"horizon": 1, "latent_weight": 1.0, "target_detached": True,
                                "source_and_embedding_attached": True, "kl_weight": 0.0,
                                "predicted_state_ce_weight": 0.0},
                     nextlat_config={"latent": "post_final_layer_norm", "predictor": {
                         "hidden_width": 512, "input_width": 1024, "output_width": 512,
                         "normalization": "rms_norm", "normalization_epsilon": 1e-5,
                         "linear_layers": 3, "activation": "gelu", "bias": False,
                         "residual": True, "output_normalization": False}})
    return value


def initialization(architecture, augmented, predictor):
    original = {"canonical_sha256": "a" * 64, "parameter_count": 6357504,
                "architecture": architecture, "seed": 1234, "width": 512}
    if not augmented:
        return original
    return {"canonical_sha256": "a" * 64, "backbone": original,
            "backbone_parameter_count": 6357504, "predictor_parameter_count": 1049600,
            "parameter_count": 7407104, "predictor_sha256": "b" * 64,
            "predictor_config": predictor}


def memory_arms():
    result = {}
    for name in reporter.ARMS:
        augmented = name.endswith("_nextlat")
        architecture = name.split("_")[0]
        identity = contract(architecture, augmented)
        source = {"scripts/rt_a5_common.py": "f" * 64}
        if augmented:
            source.update({key: "e" * 64 for key in reporter.NEW_SOURCES})
        curves, metrics = {}, {}
        for step in reporter.STEPS:
            metrics[str(step)] = {role: metric(role, step, 4096 if step == 1000 and not augmented else 102400)
                                  for role in reporter.ROLES}
            curves[str(step)] = {role: reporter.metric_rows(value, name) for role, value in metrics[str(step)].items()}
        history = [{"update": 1, "seconds": 1.0, "order_chain": "9" * 64,
                    "loss": 3.0, "state_loss": 2.0, "latent_loss": 1.0,
                    "weighted_latent_loss": 1.0, "token_accuracy": .5}]
        result[name] = {"report": {"contract": identity, "source_files": source,
                                   "initialization": initialization(architecture, augmented,
                                        identity.get("nextlat_config", {}).get("predictor")),
                                   "order_chain": "9" * 64, "train_seconds": 1.0,
                                   "elapsed_seconds": 2.0, "one_step_diagnostics": []},
                        "history": history, "curves": curves, "metrics": metrics,
                        "checkpoints": {}, "input_files": {}}
    return result


def file_fixture(directory):
    directory.mkdir()
    data = directory / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"fixture": true}\n')
    source_path = directory / "source/scripts/fixture.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# saved source\n")
    source = {"scripts/fixture.py": reporter.hash_file(source_path)["sha256"]}
    identity = contract("rt", True)
    identity.update(source_sha256=reporter._digest_dict(source),
                    data_manifest_sha256=reporter.hash_file(data / "manifest.json")["sha256"])
    history = [{"update": step, "examples_seen": step * 1024, "seconds": 1.0,
                "order_chain": hashlib.sha256(str(step).encode()).hexdigest(),
                "loss": 3.0, "state_loss": 2.0, "latent_loss": 1.0, "weighted_latent_loss": 1.0,
                "token_accuracy": .5, "whole_word_exact": .1, "grad_norm": 1.0}
               for step in range(1, 10001)]
    checkpoints = []
    for step in (0, *reporter.STEPS):
        path = directory / f"step-{step}.pt"
        path.write_text(f"opaque checkpoint fixture {step}")
        checkpoints.append({**reporter.hash_file(path), "completed_updates": step})
    packet = {"schema": "rt-a5-nextlat-training-v1", "status": "complete", "start_update": 0,
              "completed_updates": 10000, "endpoint": 10000, "parent_checkpoint": None,
              "confirmation_evaluated": False, "latent_rollout_evaluated": False,
              "contract": identity, "source_files": source, "checkpoints": checkpoints,
              "initialization": initialization("rt", True, identity["nextlat_config"]["predictor"]),
              "order_chain": history[-1]["order_chain"], "train_seconds": 10000.0,
              "elapsed_seconds": 10001.0, "one_step_diagnostics": [],
              "evaluations": [metric(role, step) for step in reporter.STEPS for role in reporter.ROLES]}
    (directory / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    reporter.write_json(directory / "report.json", packet)
    reporter.write_json(directory / "config.json", {"data_dir": str(data)})
    return packet


def test_curve_recovery_distinguishes_all_three_metrics_and_keeps_denominators():
    rows = reporter.metric_rows(metric(), "rt_nextlat")
    assert len(rows) == 36
    assert rows[0]["E"] == rows[0]["A"] == rows[0]["M"] == .5
    assert rows[-1]["E"] == 0 and rows[-1]["A"] == rows[-1]["M"] == .5
    assert rows[-1]["token_denominator"] == 102400 * 36
    assert rows[-1]["E_high95"] > 0
    assert "M_low95" not in rows[-1]
    small = reporter.metric_rows(metric(update=1000, rows=4096), "rt")
    assert small[0]["words"] == 4096 and rows[0]["words"] == 102400


@pytest.mark.parametrize("mutate", [
    lambda m: m["cumulative_prefix_exactness"].__setitem__(0, .25),
    lambda m: m["cumulative_prefix_exactness"].__setitem__(10, 1.0),
    lambda m: m["isolated_state_accuracy"].__setitem__(2, .500001),
    lambda m: m.update(token_accuracy=.4),
])
def test_inconsistent_or_rounded_accuracy_records_rejected(mutate):
    value = metric()
    mutate(value)
    with pytest.raises(ValueError):
        reporter.metric_rows(value, "rt")


def test_comparison_allows_new_source_files_but_preserves_shared_identity(monkeypatch):
    arms = memory_arms()
    monkeypatch.setattr(reporter, "read_training", lambda _directory, arm: arms[arm])
    summary = reporter.compare_runs({name: name for name in reporter.ARMS})
    assert summary["evaluation_route"] == "backbone_only"
    assert summary["latent_rollout_evaluated"] is False
    assert summary["arms"]["rt_nextlat"]["training_curve"][0]["state_ce"] == 2
    assert summary["arms"]["rt"]["training_curve"][0]["state_ce"] == 3
    markdown = reporter.markdown_report(summary)
    assert "4,096 words" in markdown and "102,400" in markdown
    assert "Final confirmation remains unevaluated" in markdown
    assert "not" in markdown and "training-seed" in markdown
    assert "RT + NextLat minus Transformer + NextLat" in markdown


@pytest.mark.parametrize("mutate,match", [
    (lambda a: a["rt_nextlat"]["report"]["source_files"].update({"scripts/rt_a5_common.py": "0" * 64}), "historical execution"),
    (lambda a: a["rt_nextlat"]["report"]["source_files"].update({"unexpected.py": "0" * 64}), "additions"),
    (lambda a: a["rt_nextlat"]["report"]["contract"].update(data_order_seed=9), "contracts differ"),
    (lambda a: a["rt_nextlat"]["report"]["initialization"].update(canonical_sha256="c" * 64), "initialization differs"),
    (lambda a: a["seq_nextlat"]["report"]["initialization"].update(predictor_sha256="c" * 64), "Predictor initial"),
    (lambda a: a["rt_nextlat"]["history"][0].update(order_chain="8" * 64), "data order"),
])
def test_comparison_rejects_unpaired_experiments(monkeypatch, mutate, match):
    arms = memory_arms()
    mutate(arms)
    monkeypatch.setattr(reporter, "read_training", lambda _directory, arm: arms[arm])
    with pytest.raises(ValueError, match=match):
        reporter.compare_runs({name: name for name in reporter.ARMS})


def test_file_reader_checks_completion_source_manifest_checkpoint_and_inference_route(tmp_path):
    directory = tmp_path / "fixture"
    packet = file_fixture(directory)
    read = reporter.read_training(directory, "rt_nextlat")
    assert len(read["history"]) == 10000
    assert set(read["checkpoints"]) == {"0", "1000", "5000", "10000"}
    for mutate, match in (
        (lambda p: p.update(status="running"), "completed"),
        (lambda p: p.update(completed_updates=9999), "10000"),
        (lambda p: p.update(confirmation_evaluated=True), "confirmation"),
        (lambda p: p.update(latent_rollout_evaluated=True), "only the backbone"),
        (lambda p: p["evaluations"][0].update(route="latent_rollout"), "backbone predictions"),
        (lambda p: p["contract"].update(data_manifest_sha256="0" * 64), "data manifest"),
        (lambda p: p["checkpoints"][-1].update(sha256="0" * 64), "checkpoint differs"),
    ):
        changed = copy.deepcopy(packet)
        mutate(changed)
        reporter.write_json(directory / "report.json", changed)
        with pytest.raises(ValueError, match=match):
            reporter.read_training(directory, "rt_nextlat")
    reporter.write_json(directory / "report.json", packet)
    (directory / "source/scripts/fixture.py").write_text("changed\n")
    with pytest.raises(ValueError, match="Source snapshot differs"):
        reporter.read_training(directory, "rt_nextlat")


def test_plot_exports_are_standalone_and_use_task_ce(tmp_path, monkeypatch):
    arms = memory_arms()
    monkeypatch.setattr(reporter, "read_training", lambda _directory, arm: arms[arm])
    summary = reporter.compare_runs({name: name for name in reporter.ARMS})
    figures = reporter.plot_results(summary, tmp_path)
    assert set(figures) == {"length-full", "length-boundary", "task-ce-learning", "auxiliary-diagnostics"}
    for paths in figures.values():
        assert (tmp_path / paths["png"]).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / paths["pdf"]).read_bytes().startswith(b"%PDF")
