"""Original A5 SEQ with only ALiBi replaced by the existing OLMo RoPE.

The original builder remains authoritative for both configuration and initial
parameters. No model math is reimplemented: the supported sequential OLMo path
normalizes full-width Q/K, splits heads, applies Q/K RoPE, and attends causally.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from olmo.config import ModelConfig
from olmo.model import OLMo
from scripts.rt_a5_common import (
    build_model, canonical_parameter_sha256, model_config, parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_rope/base.json"
CONFIG_DELTA = {"alibi": {"from": True, "to": False},
                "rope": {"from": False, "to": True}}


def rope_model_config(width=512):
    """Resolve the original model and require exactly the two PE changes."""
    original = model_config("seq", width)
    raw = json.loads(CONFIG_PATH.read_text())
    raw.update(d_model=width, n_heads=width // 64, n_kv_heads=width // 64,
               mlp_hidden_size=4 * width, init_device="cpu")
    config = ModelConfig(**raw)
    before, after = asdict(original), asdict(config)
    delta = {key: {"from": before[key], "to": after[key]}
             for key in before if before[key] != after[key]}
    if delta != CONFIG_DELTA:
        raise ValueError(f"RoPE control must change only ALiBi/RoPE: {delta}")
    return config


def build_rope_model(width=512, seed=1234, device="cpu"):
    """Strictly copy the original CPU SEQ initialization, preserving caller RNG."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = rope_model_config(width)
    source = build_model("seq", width=width, seed=seed, device="cpu")
    # Construction creates the supported RoPE caches/modules. Its temporary
    # parameter draws cannot affect caller RNG or the copied canonical state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = OLMo(config).to(dtype=torch.float32)
    model.load_state_dict(source.state_dict(), strict=True)
    original_hash = source.a5_initialization["canonical_sha256"]
    if canonical_parameter_sha256(model) != original_hash:
        raise AssertionError("RoPE initialization differs from original SEQ")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != parameter_count(width):
        raise AssertionError("RoPE changed the original parameter topology")
    model.a5_initialization = {
        "schema": "rt-a5-seq-rope-initialization-v1",
        "architecture": "seq_rope", "seed": seed, "width": width,
        "parameter_count": count, "canonical_sha256": original_hash,
        "paired_base_architecture": "seq",
        "paired_base_canonical_sha256": original_hash,
        "paired_base_initialization": dict(source.a5_initialization),
        "config_delta": json.loads(json.dumps(CONFIG_DELTA)),
        "conversion": None,
        "rule": "Strict copy of original CPU Mitchell SEQ state; only ALiBi replaced by OLMo RoPE",
        "position_encoding": {
            "implementation": "olmo.model.RotaryEmbedding",
            "positions": "0..T-1", "theta": config.rope_theta,
            "full_precision": config.rope_full_precision,
            "layout": "split halves, full head dimension",
            "order": "full-width learned Q/K normalization, split heads, rotate Q/K",
            "value_rotation": False, "learned_position_parameters": False,
        },
    }
    return model.to(device=device, dtype=torch.float32)
