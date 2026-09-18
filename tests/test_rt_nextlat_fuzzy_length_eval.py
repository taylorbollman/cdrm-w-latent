"""CPU evidence/restoration tests; no CUDA, training or optimizer construction."""
import copy
import subprocess
import sys

import numpy as np
import pytest
import torch

from cdrm.mad_data import FUZZY_TASK, generate_dataset
from cdrm.rt_nextlat_fuzzy_metrics import FuzzyMetrics
from scripts import rt_nextlat_fuzzy_length_eval as evaluate


def metric(dataset):
    meter = FuzzyMetrics(dataset)
    predictions = np.where(dataset.answer_labels < 0, 0, dataset.answer_labels)
    meter.update(predictions, np.zeros_like(predictions, dtype=float))
    return {**meter.compute(), "evaluated_rows": len(dataset), "ce": 0., "latent": .1,
            "evaluation_seconds": .5, "route": "backbone logits; teacher-conditioned NextLat diagnostic; no rollout"}


@pytest.fixture
def datasets(monkeypatch):
    monkeypatch.setattr(evaluate, "EXPECTED_ROWS", 5)
    monkeypatch.setattr(evaluate, "LENGTHS", (48, 64))
    original = evaluate.check_metric
    monkeypatch.setattr(evaluate, "check_metric", lambda row, **kwargs: original(row, expected_rows=5))
    return {length: generate_dataset(FUZZY_TASK, "dev", seed, 5, {"seq_len": length})
            for length, seed in ((48, 71), (64, 72))}


def test_metrics_preserve_empty_distance_regions_and_reject_bad_denominators(datasets):
    result = metric(datasets[48])
    assert evaluate.check_metric(result) is result
    assert result["distance_513_plus_tokens"] == 0
    assert result["distance_513_plus_accuracy"] is None
    changed = copy.deepcopy(result)
    changed["answer_accuracy"] = .5
    with pytest.raises(ValueError, match="count mismatch"):
        evaluate.check_metric(changed)
    changed = copy.deepcopy(result)
    changed["distance_513_plus_accuracy"] = 0.
    with pytest.raises(ValueError, match="count mismatch"):
        evaluate.check_metric(changed)


def test_eval_dispatch_uses_shared_full_pools_and_preserves_weights(datasets):
    model = torch.nn.Linear(2, 3, bias=False).requires_grad_(False).eval()
    before = evaluate.model_digest(model)
    calls = []
    def fake(model_arg, data, **options):
        assert model_arg is model
        assert options == {"microbatch": 64, "device": "cpu", "latent_weight": 1.0, "limit": 5}
        calls.append(data)
        return metric(data)
    result = evaluate.evaluate_model(model, datasets, microbatch=64, device="cpu", evaluator=fake)
    assert calls == list(datasets.values())
    assert set(result) == {"48", "64"}
    assert all(row["training_update"] == 15000 and row["training_sequence_length"] == 400 for row in result.values())
    assert evaluate.model_digest(model) == before
    assert model.weight.grad is None


def test_eval_rejects_model_mutation(datasets):
    model = torch.nn.Linear(2, 3, bias=False).requires_grad_(False).eval()
    def fake(model_arg, data, **options):
        with torch.no_grad():
            model_arg.weight.add_(1)
        return metric(data)
    with pytest.raises(ValueError, match="changed model"):
        evaluate.evaluate_model(model, datasets, microbatch=64, device="cpu", evaluator=fake)


def test_restore_is_strict_cpu_load_without_optimizer(monkeypatch):
    seeds = {"seed": 1234, "predictor_seed": 1235, "fuzzy_seed": 1236}
    initialization = {**seeds, "conversion": {"source_keys": ["weight"], "split_sizes": [3, 2]}}
    packet = {"initialization": initialization, "model": {"weight": torch.ones((3, 2))},
              "optimizer": {"state": {0: {"step": torch.tensor(15000.), "exp_avg": torch.zeros((3, 2))}}},
              "finite_state": {"parameters_finite_fp32": True, "gradients_finite_fp32": True,
                               "adam_finite_fp32": True}}
    monkeypatch.setattr(evaluate.paired, "_packet", lambda current, update: packet)
    current = {"report": {"contract": {"model_config": {"fixture": True}}, "parameter_count": 6}}
    def factory(config, **options):
        assert config == {"fixture": True}
        assert options == {**seeds, "device": "cpu", "backend": "tiled"}
        model = torch.nn.Linear(2, 3, bias=False)
        model.initialization = copy.deepcopy(initialization)
        model.initialization["conversion"] = {"source_keys": ("weight",), "split_sizes": (3, 2)}
        return model
    model, digest = evaluate.restore_model(current, factory=factory)
    assert torch.equal(model.weight, packet["model"]["weight"])
    assert not model.training and not model.weight.requires_grad
    assert digest == evaluate.model_digest(model)
    packet["optimizer"]["state"][0]["exp_avg"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite saved tensor"):
        evaluate.restore_model(current, factory=factory)


def test_finite_saved_state_rejects_silent_precision_change():
    with pytest.raises(ValueError, match="Non-FP32"):
        evaluate.finite_tree({"weight": torch.ones(2, dtype=torch.bfloat16)})


def test_plots_and_report_retain_counts_qualifications_and_source_lengths(tmp_path, datasets):
    observations = {str(length): metric(data) for length, data in datasets.items()}
    observations["400"] = copy.deepcopy(observations["48"])
    arms = {}
    for label in ("baseline", "input", "value", "head"):
        arms[label] = {"metrics": copy.deepcopy(observations), "checkpoint": {"sha256": label*8},
                       "mechanism": {"description": label}, "parameters": 479616}
    evaluate.validate_shared_counts(arms)
    summary = {"arms": arms, "eval_microbatch": 64, "data_manifest_sha256": "datafixture"}
    summary["figures"] = evaluate.make_plots(summary, tmp_path)
    assert len(summary["figures"]) == 3
    assert all((tmp_path / f"{name}.pdf").stat().st_size > 1000 for name in summary["figures"])
    text = evaluate.markdown(summary)
    assert "Longer sequences are not necessarily harder" in text
    assert "T400 is the retained training-endpoint" in text
    assert "No automatic winner" in text
    arms["head"]["metrics"]["48"]["answer_tokens"] += 1
    with pytest.raises(ValueError, match="different native scoring populations"):
        evaluate.validate_shared_counts(arms)


def test_four_arm_endpoint_guard_rejects_live_or_wrong_budget(monkeypatch):
    fake = {"endpoint": 14999, "report": {"status": "complete"}}
    monkeypatch.setattr(evaluate.saved, "load_run", lambda path: fake)
    with pytest.raises(ValueError, match="completed 15k"):
        evaluate.load_inputs([(name, name) for name in ("baseline", "input", "value", "head")])
    with pytest.raises(ValueError, match="four uniquely"):
        evaluate.load_inputs([("baseline", "base")])


def test_current_source_hash_must_match_saved_implementation(tmp_path, monkeypatch):
    path = tmp_path / "frozen.py"
    path.write_text("original")
    frozen = evaluate.saved.sha(path)
    path.write_text("changed")
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Executed source differs"):
        evaluate.verify_live_sources({"baseline": {"sources": {"frozen.py": frozen}}})


def test_cli_help_does_not_enter_gpu_or_read_experiment_data():
    completed = subprocess.run([sys.executable, "-m", "scripts.rt_nextlat_fuzzy_length_eval", "--help"],
                               cwd=evaluate.ROOT, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    for option in ("--run", "--data", "--output", "--eval-microbatch", "--wandb"):
        assert option in completed.stdout
