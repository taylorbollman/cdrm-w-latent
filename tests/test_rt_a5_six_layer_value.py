"""Bounded CPU structure and attached-gradient checks for embedding routing."""
import copy
import hashlib
import json

import numpy as np

from scripts import rt_a5_nextlat_train, rt_a5_train
from scripts import rt_a5_six_layer_value_train as driver

import pytest
import torch

from olmo.model import OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import fp32_context, configure_fp32_runtime
from scripts.rt_a5_six_layer_value import build_model, configuration, make_projection, SixLayerValueBackbone, coefficient, set_update, SCHEDULES
from scripts.rt_a5_embedding_injection import EmbeddingInjectedBackbone
from scripts.rt_a5_l1r_depth import build_model as build_depth_model
from scripts.rt_a5_nextlat import _parameter_sha256, nextlat_objective
from scripts.rt_a5_value_bypass import ValueBypassRecurrentAutogradBlock, ValueBypassRecurrentBlockTiled
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
    configure_fp32_runtime()


def small(**kwargs):
    return build_model(width=64, backend="naive", predictor_hidden_width=256, **kwargs)


def inputs():
    return torch.tensor([[2, 4, 8, 1], [3, 7, 2, 6]], dtype=torch.long)


def test_actual_width_six_layer_count_and_exact_shared_initialization():
    baseline = build_depth_model(n_layers=6)
    rng = torch.get_rng_state().clone()
    model = build_model()
    assert torch.equal(rng, torch.get_rng_state())
    original, actual = dict(baseline.named_parameters()), dict(model.named_parameters())
    assert set(actual) - set(original) == {"backbone.embedding_projection.weight"}
    assert all(torch.equal(value, actual[name]) for name, value in original.items())
    assert sum(p.numel() for p in model.parameters()) == 20260352
    assert sum(p.numel() for p in model.backbone.parameters()) == 19210752
    assert len(actual) == 62 and len(list(model.backbone.parameters())) == 58
    assert model.nextlat_initialization["shared_six_layer_model_sha256"] == _parameter_sha256(baseline)
    assert model.nextlat_initialization["predictor_sha256"] == "3a1fccdd63a53e50328c010b03ecd81d7cf8782229b63cf1c843d1c658731e58"
    assert model.nextlat_initialization["baseline_parameter_tensors_changed"] == []
    assert model.nextlat_initialization["canonical_sha256"] != baseline.nextlat_initialization["canonical_sha256"]
    assert model.nextlat_config == baseline.nextlat_config
    linear = build_model(variant="linear")
    assert _parameter_sha256(model) == _parameter_sha256(linear)
    assert model.backbone.injection_coefficient == .01 and linear.backbone.injection_coefficient == 0
    assert model.backbone.injection_variant == linear.backbone.injection_variant == "value"
    assert SixLayerValueBackbone.forward is EmbeddingInjectedBackbone.forward
    assert torch.equal(model.backbone.embedding_projection.weight, make_projection(512, 1236).weight)
    assert "injection_coefficient" not in actual
    assert len(list(model.named_parameters(remove_duplicate=False))) == len(actual)


@pytest.mark.parametrize("backend", ["naive", "tiled"])
def test_six_blocks_positions_and_owned_views(backend):
    model = build_model(width=128, backend=backend)
    blocks = model.backbone.transformer.blocks
    assert len(blocks) == 6
    assert type(blocks[0]) is (WindowTwoRecurrentAutogradBlock if backend == "naive" else WindowTwoRecurrentBlockTiled)
    assert type(blocks[1]) is (ValueBypassRecurrentAutogradBlock if backend == "naive" else ValueBypassRecurrentBlockTiled)
    assert all(type(block) is (OLMoRecurrentAutogradBlock if backend == "naive" else OLMoRecurrentBlockTiled)
               for block in blocks[2:])
    assert [block.layer_id for block in blocks] == [0, 1, 2, 3, 4, 5]
    assert model.backbone.config.alibi and not model.backbone.config.rope
    assert "wpe" not in model.backbone.transformer and "emb_norm" not in model.backbone.transformer
    assert model.backbone.config.embedding_dropout == 0 and not model.backbone.config.scale_emb_init
    for block in blocks:
        assert block.pre_attention_block.kv_proj is block.kv_proj
        assert block.pre_attention_block.q_proj is block.q_proj
        assert block.post_attention_block.ff_proj is block.ff_proj


def test_coefficient_zero_is_exact_full_nextlat_baseline_outputs_and_gradients():
    baseline = build_depth_model(width=64, n_layers=6, backend="naive", predictor_hidden_width=256)
    model = small(variant="linear")
    x = inputs()
    with fp32_context("cpu"):
        expected = nextlat_objective(baseline, x, x)
        actual = nextlat_objective(model, x, x)
        for key in ("logits", "loss", "state_loss", "latent_loss"):
            assert torch.equal(actual[key], expected[key]), key
        expected["loss"].backward()
        actual["loss"].backward()
    actual_parameters = dict(model.named_parameters())
    for name, parameter in baseline.named_parameters():
        assert parameter.grad is not None and torch.equal(parameter.grad, actual_parameters[name].grad), name
    assert model.backbone.embedding_projection.weight.grad is None
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks


@pytest.mark.parametrize("variant", ["constant", "linear"])
def test_active_projection_and_original_embedding_gradients_final_latents_and_causality(variant):
    model = small(variant=variant)
    set_update(model, 1)
    x = inputs()
    seen = {}
    hook = model.backbone.transformer.ln_f.register_forward_hook(lambda _m, _args, out: seen.update(hidden=out))
    with fp32_context("cpu"):
        output = model.backbone(x, return_pre_logits=True)
        assert output.pre_logits is seen["hidden"]
        hook.remove()
        result = nextlat_objective(model, x, x)
        result["loss"].backward()
        grad = model.backbone.embedding_projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.count_nonzero(model.backbone.transformer.wte.weight.grad) > 0
        changed = x.clone()
        changed[:, -1] += 1
        with torch.no_grad():
            original_logits, changed_logits = model(x).logits, model(changed).logits
        assert torch.equal(original_logits[:, :-1], changed_logits[:, :-1])
        assert torch.equal(original_logits, output.logits)
    assert not model.backbone._injection_in_progress
    assert not model.backbone.transformer.blocks[1]._forward_pre_hooks


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("variant", ["constant", "linear"])
def test_first_step_keeps_projection_attached_and_resume_is_exact(tmp_path, variant):
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", "unused",
                                     "--width", "64", "--batch-size", "2", "--variant", variant])
    model, restored = small(variant=variant), small(variant=variant)
    optimizer, restored_optimizer = driver.make_optimizer(model), driver.make_optimizer(restored)
    contract = driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8, 4), dtype=np.uint8), hardware={"capability": [0, 0]})
    assert contract["injection_schedule"] == SCHEDULES[variant] and contract["n_layers"] == 6
    args.updates = 50000
    assert driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8, 4), dtype=np.uint8), hardware={"capability": [0, 0]}) == contract
    args.stop_file = "another-runtime-stop-path"
    assert driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8, 4), dtype=np.uint8), hardware={"capability": [0, 0]}) == contract
    assert driver.train_step is rt_a5_nextlat_train.train_step
    assert driver.evaluate_arrays is rt_a5_train.evaluate_arrays
    x = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    y = x.flip(1)
    set_update(model, 1)
    driver.train_step(model, optimizer, x, y)
    projection = model.backbone.embedding_projection.weight
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    assert torch.count_nonzero(projection.grad) > 0
    assert len(optimizer.state) == 62 and all(s["step"].item() == 1 for s in optimizer.state.values())
    path = tmp_path / "step.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
        completed=1, order_chain=hashlib.sha256(b"fixture-order").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    set_update(model, 2)
    driver.train_step(model, optimizer, y, x)
    expected_random = torch.rand(4)
    packet = driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    assert restored.schedule_update == 1 and restored.backbone.injection_coefficient == coefficient(1, variant)
    set_update(restored, 2)
    driver.train_step(restored, restored_optimizer, y, x)
    tree_equal(restored.state_dict(), model.state_dict())
    tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(4), expected_random)
    before = copy.deepcopy(restored.state_dict())
    packet["injection_coefficient"] = .02
    invalid = tmp_path / "invalid.pt"
    torch.save(packet, invalid)
    with pytest.raises(ValueError, match="schedule or current coefficient"):
        driver.load_checkpoint(invalid, model=restored, optimizer=restored_optimizer, contract=contract)
    tree_equal(before, restored.state_dict())
    assert restored.backbone.injection_coefficient == coefficient(2, variant)



@pytest.mark.parametrize("stop_at, checkpoints", [(1, [3]), (2, [2, 3]), (None, [3])])
def test_driver_stops_after_exact_completed_update_or_completes_endpoint(tmp_path, monkeypatch, stop_at, checkpoints):
    """Exercise the real CPU update/save/load loop with tiny data and local tracking."""
    x = np.arange(32, dtype=np.uint8).reshape(8, 4) % 60
    y = x[:, ::-1].copy()
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    stop_file = tmp_path / "USERSTOP"
    output = tmp_path / "run"
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", str(output),
        "--updates", "3", "--width", "64", "--batch-size", "2", "--predictor-hidden-width", "64",
        "--eval-every", "2", "--eval-rows", "2", "--full-eval-rows", "4",
        "--diagnostic-rows", "2", "--log-every", "3", "--stop-file", str(stop_file),
        "--checkpoint-steps", *map(str, checkpoints)])
    monkeypatch.setattr(driver, "require_cuda_container", lambda: {"capability": [0, 0]})
    monkeypatch.setattr(driver, "validate_manifest", lambda _root: None)
    monkeypatch.setattr(driver, "load_split", lambda _root, _role: (x, y))
    monkeypatch.setattr(driver, "source_manifest", lambda: {})
    monkeypatch.setattr(driver, "build_model", lambda **_kwargs: small())
    monkeypatch.setattr(driver, "batch_tensors", lambda tx, ty, indices, _device:
                        rt_a5_train.batch_tensors(tx, ty, indices, "cpu"))
    monkeypatch.setattr(driver, "evaluate_arrays", lambda model, dx, dy, **kw:
                        rt_a5_train.evaluate_arrays(model, dx, dy, device="cpu", **kw))
    monkeypatch.setattr(driver, "evaluate_diagnostics", lambda model, dx, dy, **kw:
                        rt_a5_nextlat_train.evaluate_diagnostics(model, dx, dy, device="cpu", **kw))
    calls = []
    def update(model, optimizer, tx, ty, **kwargs):
        step = len(calls) + 1
        assert model.schedule_update == step and model.backbone.injection_coefficient == coefficient(step, "constant")
        calls.append(step)
        values = rt_a5_nextlat_train.train_step(model, optimizer, tx, ty, **kwargs)
        if step == stop_at:
            stop_file.write_text("Requested stop for CPU fixture\n")
        return values
    monkeypatch.setattr(driver, "train_step", update)
    class LocalTracker:
        def __init__(self, **_kwargs):
            self.record = {"run_url": "local-test", "status": "running"}
        def start(self, _contract): pass
        def log(self, _values): pass
        def summary(self, _values): pass
        def finish(self, *, succeeded):
            assert succeeded
            self.record["status"] = "synced"
    monkeypatch.setattr(driver, "OnlineTracker", LocalTracker)
    report = driver.run(args)
    end = stop_at or 3
    assert calls == list(range(1, end + 1))
    assert report["endpoint"] == 3 and report["completed_updates"] == end
    assert report["status"] == ("stopped" if stop_at else "complete")
    assert report["requested_endpoint_reached"] is (stop_at is None)
    assert report["wandb"]["status"] == "synced"
    assert report["injection_coefficient"] == .01
    saved = [row["completed_updates"] for row in report["checkpoints"]]
    assert saved == [0, end] and len(saved) == len(set(saved))
    evaluations = [r for r in report["evaluations"] if r["update"] == end]
    assert len(evaluations) == 2 and all(r["evaluated_rows"] == 4 for r in evaluations)
    assert all(r["injection_coefficient"] == .01 for r in evaluations)
    if stop_at:
        assert report["stop_request"]["observed_after_update"] == end
        assert stop_file.is_file()
    packet = torch.load(output / f"checkpoints/step-{end:06d}.pt", map_location="cpu", weights_only=False)
    assert packet["schedule_update"] == end and packet["injection_coefficient"] == .01
    assert all(s["step"].item() == end for s in packet["optimizer"]["state"].values())
    rows = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(rows) == end and all(row["injection_coefficient"] == .01 for row in rows)


def test_schedule_anchors_keep_initial_metadata_immutable():
    for step, expected in [(-1,0.),(0,0.),(1,5e-7),(10000,.005),(20000,.01),(50000,.01),(60000,.01)]:
        assert coefficient(step, "linear") == pytest.approx(expected, rel=1e-15, abs=0)
        assert coefficient(step, "constant") == .01
    model = small(variant="linear")
    initial, config = copy.deepcopy(model.nextlat_initialization), copy.deepcopy(model.experiment_config)
    digest = _parameter_sha256(model)
    for step in (10000, 10001, 20000, 50000):
        assert set_update(model, step) == coefficient(step, "linear")
    assert model.nextlat_initialization == initial and model.experiment_config == config
    assert _parameter_sha256(model) == digest
    with pytest.raises(ValueError):
        coefficient(1, "unsupported")
    with pytest.raises(ValueError):
        set_update(model, -1)


def test_zero_initial_checkpoint_restores_only_same_variant(tmp_path):
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", "unused", "--width", "64", "--variant", "linear"])
    model = small(variant="linear")
    optimizer = driver.make_optimizer(model)
    contract = driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8,4), dtype=np.uint8), hardware={"capability":[0,0]})
    path = tmp_path / "initial.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
        completed=0, order_chain=hashlib.sha256(b"fixture").hexdigest(), initialization=rt_a5_train.json_value(model.nextlat_initialization))
    restored = small(variant="linear")
    driver.load_checkpoint(path, model=restored, optimizer=driver.make_optimizer(restored), contract=contract)
    assert restored.schedule_update == 0 and restored.backbone.injection_coefficient == 0
    wrong = small(variant="constant")
    with pytest.raises(ValueError, match="schedule or current coefficient"):
        driver.load_checkpoint(path, model=wrong, optimizer=driver.make_optimizer(wrong), contract=contract)
    with pytest.raises(ValueError, match="Requested variant differs"):
        driver.make_contract(args, model=wrong, sources={}, data_root=tmp_path,
            train_x=np.zeros((8,4), dtype=np.uint8), hardware={"capability":[0,0]})
