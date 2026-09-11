"""CPU checks of common CDRM weights, mixed CE semantics and retained identity."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
import cdrm_tiled_common as common
from cdrm.mad_data import generate_dataset


def test_three_execution_arms_share_exact_cpu_initialization():
    original, construction = common.cpu_initial_model(1907)
    import dataclasses
    packet = {"model_config": dataclasses.asdict(original.config), "model": common.cpu_tree(original.state_dict())}
    for backend, precision in (("naive", "fp32"), ("tiled", "fp32"), ("tiled", "bf16_fp32_state")):
        model, built = common.build_model(backend, precision, checkpoint=packet, device="cpu")
        assert common.state_digest(model.state_dict()) == construction["full_initialization_sha256"]
        assert model.config.n_layers == 5
        assert (model.config.cdrm_early_layer, model.config.cdrm_late_layer) == (1, 3)
        assert not model.config.recurrent_layers
        assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
        assert built["starting_weights_sha256"] == construction["full_initialization_sha256"]


def test_naive_bf16_arm_is_rejected_before_model_construction():
    with pytest.raises(ValueError, match="FP32"):
        common.build_model("naive", "bf16", device="cpu")


def test_bf16_logits_use_fp32_masked_ce_without_shift():
    logits = torch.tensor([[[0., 7., 1.], [8., 1., 0.], [0., 2., 9.]]], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[-100, 0, 2]])
    summed, count = common.loss_sum(logits, labels)
    reference = logits.detach().float().requires_grad_()
    expected = torch.nn.functional.cross_entropy(reference.reshape(-1, 3), labels.reshape(-1),
                                                ignore_index=-100, reduction="sum")
    assert count == 2 and summed.dtype == torch.float32 and torch.equal(summed, expected)
    summed.backward()
    expected.backward()
    assert torch.equal(logits.grad, reference.grad.bfloat16())
    assert torch.count_nonzero(logits.grad[:, 0]) == 0


def test_native_copying_shape_and_answer_objective_are_checked():
    data = generate_dataset("selective-copying", "dev", 1937, 3, {"num_tokens_to_copy": 96})
    common.validate_data(data)
    wrong = generate_dataset("selective-copying", "dev", 1937, 3)
    with pytest.raises(ValueError, match="K96"):
        common.validate_data(wrong)


def test_saved_source_identity_detects_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("input.py")
    source.write_text("value = 1\n")
    tracked = {str(source): common.file_digest(source)}
    common.snapshot_sources(Path("run"), tracked)
    common.verify_sources(tracked)
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source identity changed"):
        common.verify_sources(tracked)
    assert Path("run/source/input.py").read_text() == "value = 1\n"
