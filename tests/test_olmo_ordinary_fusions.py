"""Integration safeguards for the bounded RoPE/optimizer benchmark."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatConfig
from scripts import olmo_ordinary_fusions as harness
from scripts.olmo_ordinary_throughput import make_batch


def command(arm="fused-adam", stage="correctness", batch=8, length=512):
    return ["--stage", stage, "--arm", arm, "--batch-size", str(batch),
            "--length", str(length), "--output-dir", "/tmp/unused-fusions"]


@pytest.mark.parametrize("args", [command(arm="dao-rope"), command(batch=2),
    command(length=32), command(stage="capacity", batch=64)])
def test_fixed_gradient_probe_cannot_silently_expand_scope(args):
    with pytest.raises(SystemExit):
        harness.parse_args(args + ["--optimizer-probe"])


def test_probe_is_explicit_on_actual_checkpoint_case():
    assert harness.parse_args(command() + ["--optimizer-probe"]).optimizer_probe
    assert not harness.parse_args(command()).optimizer_probe


@pytest.mark.parametrize("arm,options", list(harness.ARMS.items()))
def test_arm_changes_preserve_reference_execution_and_optimizer_ownership(arm, options):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 4)
            self.readout = torch.nn.Linear(4, 7, bias=False)
            self.readout.weight = self.embedding.weight
            self.vector = torch.nn.Parameter(torch.ones(4))
            self.backbone = SimpleNamespace(backbone=SimpleNamespace())
            self.config = NextLatConfig(model_dim=32, vocab_chunk_size=128)

    model = Model()
    config = model.config
    harness.set_arm(model, arm)
    base = model.backbone.backbone
    assert base.ordinary_attention_backend == "sdpa"
    assert base.ordinary_pointwise_backend == "compiled"
    assert base.ordinary_checkpoint_layers is None and base.reuse_rope
    assert base.ordinary_rope_backend == options[0]
    assert model.config == replace(config, ce_chunk_size=2048)
    optimizer, _ = harness.optimizer_for(model, arm)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters}) == 2
    for group in optimizer.param_groups:
        assert group["fused"] is options[1] and group["foreach"] is False
        assert group["betas"] == (.9, .95) and group["eps"] == 1e-8


def test_optimizer_only_comparison_requires_bitwise_forward_and_gradients():
    assert harness.exact_comparison("control", "fused-adam")
    assert harness.exact_comparison("dao-rope", "dao-rope-fused-adam")
    assert not harness.exact_comparison("control", "dao-rope")


def test_full_ce_applies_to_profile_update_as_well_as_timed_updates(monkeypatch):
    original = make_batch(2, 32, vocab_size=67, supervision="half")
    seen = []
    def changed(tokenizer, case, update):
        seen.append(update)
        return original
    monkeypatch.setattr(harness, "changed_batch", changed)
    for update in (*range(8), 9):
        batch = harness.batch_for(None, None, update)
        assert torch.equal(batch.ce_mask, batch.valid_mask)
        assert batch.input_ids is original.input_ids
        assert batch.latent_mask is original.latent_mask
        assert batch.kl_mask is original.kl_mask
    assert seen == [*range(8), 9]


def test_optimizer_probe_failure_is_not_a_deferrable_model_roundoff_miss():
    check = {"name": "fixed_gradient_scalar_vs_fused_adamw", "passed": False,
             "ownership_matches": True, "finite": True, "counts_equal": True,
             "losses": {"ce": {}}, "outputs": {"x": {}}, "gradients": {"w": {}}}
    assert not harness.can_continue_after_failure(check, enabled=True)


def test_runtime_sources_include_injected_factory_and_probe():
    assert {"scripts/olmo_rt_efficiency.py", "scripts/olmo_ordinary_optimizer_probe.py",
            "scripts/olmo_ordinary_fusions.py", "cdrm/pretrained/lm_training.py",
            "cdrm/pretrained/olmo_rope.py"} <= set(harness.SOURCES)
