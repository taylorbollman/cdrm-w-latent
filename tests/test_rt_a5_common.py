"""CPU correctness tests for the new A5 configuration, pairing, loss and metrics."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from olmo.checkpoint_conversion import convert_model
from olmo.model import OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled, OLMoSequentialBlock
from rt_a5_common import (
    A5Metrics, build_model, canonical_parameter_sha256, configure_fp32_runtime,
    fp32_context, make_optimizer, model_config, parameter_count, task_loss,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("width,expected", [(64, 106560), (128, 409728),
                                            (256, 1605888), (512, 6357504)])
def test_constructed_models_have_exact_pairing_and_counts(width, expected):
    torch.random.default_generator.manual_seed(17)
    rng_before = torch.random.get_rng_state().clone()
    seq = build_model("seq", width=width, seed=91)
    rt = build_model("rt", width=width, seed=91)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert parameter_count(width) == expected
    assert sum(p.numel() for p in seq.parameters()) == expected
    assert sum(p.numel() for p in rt.parameters()) == expected
    assert seq.a5_initialization["canonical_sha256"] == rt.a5_initialization["canonical_sha256"]
    assert all(type(block) is OLMoSequentialBlock for block in seq.transformer.blocks)
    assert all(type(block) is OLMoRecurrentBlockTiled for block in rt.transformer.blocks)
    assert all(p.dtype == torch.float32 for p in rt.parameters())
    assert rt.a5_initialization["conversion"]["target_recurrent_layers"] == [0, 1]
    assert rt.a5_initialization["conversion"]["missing"] == []
    assert rt.a5_initialization["conversion"]["unexpected"] == []
    # Check every mapped parameter directly rather than relying only on hashes.
    recovered = build_model("seq", width=width, seed=98)
    convert_model(rt, recovered)
    for name, value in seq.state_dict().items():
        assert torch.equal(value, recovered.state_dict()[name]), name
    assert {p.data_ptr() for p in seq.parameters()}.isdisjoint(p.data_ptr() for p in rt.parameters())


def test_seed_changes_all_parameter_digest_and_naive_uses_autograd():
    rt = build_model("rt", width=64, seed=91, backend="naive")
    assert all(type(block) is OLMoRecurrentAutogradBlock for block in rt.transformer.blocks)
    assert all(block.config.reference_eager for block in rt.transformer.blocks)
    assert canonical_parameter_sha256(rt) == canonical_parameter_sha256(build_model("seq", 64, 91))
    assert canonical_parameter_sha256(rt) != canonical_parameter_sha256(build_model("seq", 64, 92))
    before = canonical_parameter_sha256(rt)
    with torch.no_grad():
        rt.transformer.ln_f.weight[0].add_(1)
    assert canonical_parameter_sha256(rt) != before


@pytest.mark.parametrize("width", [128, 256, 512])
@pytest.mark.parametrize("architecture", ["seq", "rt"])
def test_published_configs_agree_with_factory(width, architecture):
    raw = json.loads((ROOT / f"configs/rt_a5/{architecture}_d{width}.json").read_text())
    resolved = asdict(model_config(architecture, width))
    assert {key: resolved[key] for key in raw} == raw
    assert resolved["reference_eager"] and resolved["precision"] == "fp32"
    assert resolved["n_heads"] == resolved["n_kv_heads"] == width // 64
    assert resolved["activation_type"] == "gelu"
    assert resolved["max_sequence_length"] == 36
    assert resolved["embedding_size"] is None


def test_loss_is_unshifted_and_supervises_all_positions():
    logits = torch.zeros(2, 3, 60, requires_grad=True)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss = task_loss(logits, labels)
    expected = torch.stack([F.cross_entropy(logits[b, t].unsqueeze(0), labels[b, t].reshape(1))
                            for b in range(2) for t in range(3)]).mean()
    assert torch.equal(loss, expected)
    loss.backward()
    for b in range(2):
        for t in range(3):
            assert logits.grad[b, t, labels[b, t]] < 0
            assert int((logits.grad[b, t] < 0).sum()) == 1
    assert torch.allclose(logits.grad, (torch.full_like(logits, 1 / 60)
                                      - F.one_hot(labels, 60)) / 6)


def test_metrics_distinguish_state_accuracy_and_entire_prefix_correctness():
    labels = torch.zeros(3, 4, dtype=torch.long)
    predictions = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    logits = F.one_hot(predictions, 60).float() * 4
    metrics = A5Metrics()
    metrics.update(logits, labels)
    result = metrics.compute()
    assert result["rows"] == 3 and result["tokens"] == 12
    assert result["isolated_state_accuracy"] == pytest.approx([2 / 3, 2 / 3, 1, 1])
    assert result["cumulative_prefix_exactness"] == pytest.approx([2 / 3, 1 / 3, 1 / 3, 1 / 3])
    assert result["whole_word_exact_match"] == pytest.approx(1 / 3)
    assert result["final_state_accuracy"] == 1
    assert result["token_accuracy"] == pytest.approx(10 / 12)
    assert result["ce"] == pytest.approx(float(task_loss(logits, labels)))
    # Changing batch boundaries and row order preserves integer count metrics.
    other = A5Metrics()
    for index in [2, 0, 1]:
        other.update(logits[index:index+1], labels[index:index+1])
    assert other.compute() == result
    with pytest.raises(ValueError, match="separate"):
        metrics.update(logits[:, :3], labels[:, :3])
    with pytest.raises(ValueError, match="without examples"):
        A5Metrics().compute()


def test_loss_rejects_shifted_shape_or_mixed_precision_inputs():
    with pytest.raises(ValueError, match="Expected"):
        task_loss(torch.zeros(2, 3, 60), torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(TypeError, match="FP32"):
        task_loss(torch.zeros(2, 3, 60, dtype=torch.bfloat16), torch.zeros(2, 3, dtype=torch.long))


def test_fp32_context_overrides_enclosing_cpu_autocast():
    configure_fp32_runtime()
    model = build_model("seq", width=64, seed=4)
    tokens = torch.tensor([[0, 1, 2]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with fp32_context("cpu"):
            assert not torch.is_autocast_enabled("cpu")
            logits = model(tokens).logits
        assert torch.is_autocast_enabled("cpu")
    assert logits.dtype == torch.float32
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


@pytest.mark.parametrize("architecture", ["seq", "rt"])
def test_optimizer_decay_partition_is_exhaustive_and_paper_aligned(architecture):
    model = build_model(architecture, width=64, seed=5)
    optimizer = make_optimizer(model)
    seen = []
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-4 and group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8 and not group["foreach"] and not group["fused"]
        for parameter in group["params"]:
            assert group["weight_decay"] == (0.01 if parameter.ndim >= 2 else 0.0)
            seen.append(id(parameter))
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))
    assert set(seen) == {id(parameter) for parameter in model.parameters()}
