"""Bounded RT+NextLat pilot: identity-centered value path and fixed sinusoids.

This is an initialization heuristic for existing RT matrices. RT does not have
IDS4's input-dependent affine transition, and I+N(0,1/D) on W_V and W_O does
not make the full recurrent state update close to identity. Neither the RT
recurrence nor NextLat's objective/inference route is reimplemented here.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_nextlat_variant/base.json"


def variant_configuration(width: int) -> dict:
    """Resolve the new pilot's explicit initialization and position convention."""
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64")
    config = json.loads(CONFIG_PATH.read_text())
    initialization = config["initialization"]
    multiplier = initialization["noise_multiplier"]
    noise_seed = initialization["noise_seed"]
    if (not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool)
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("noise_multiplier must be finite and positive")
    if type(noise_seed) is not int or noise_seed < 0:
        raise ValueError("noise_seed must be a nonnegative integer")
    config["width"] = width
    initialization["noise_std"] = multiplier / math.sqrt(width)
    return config


class FixedSinusoidalPositions(nn.Module):
    """Parameter-free, length-independent standard FP32 sinusoidal positions.

The deterministic frequency buffer is nonpersistent: source/config define it,
so no new checkpoint or optimizer state is introduced. The original token
embedding is unscaled. This module replaces the optional learned wpe module
used by the existing OLMo forward; it is not an embedding forward hook.
"""

    def __init__(self, width: int, base: float = 10000.0):
        super().__init__()
        if type(width) is not int or width < 2 or width % 2:
            raise ValueError("Sinusoidal width must be positive and even")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("Sinusoidal base must be finite and greater than one")
        self.width = width
        self.base = float(base)
        frequencies = self.base ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dtype != torch.long or positions.ndim < 1:
            raise TypeError("Sinusoidal positions must be a non-scalar int64 tensor")
        angles = positions.to(dtype=torch.float32).unsqueeze(-1) * self.inverse_frequencies
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def build_variant_model(architecture: str = "rt", width: int = 512, seed: int = 1234,
                        predictor_seed: int = 1235, device: str | torch.device = "cpu",
                        *, backend: str = "tiled",
                        predictor_hidden_width: int | None = None):
    """Build the paired baseline, then change precisely four parameter tensors.

Each W_V occupies the lower D rows of the block's fused K/V projection. K
rows, Q, all norms/MLPs, embeddings/head and the NextLat predictor retain the
baseline's exact initialization. A separate CPU generator owns all noise.
"""
    if architecture != "rt":
        raise ValueError("The combined pilot supports only architecture='rt'")
    if any(type(value) is not int or value < 0 for value in (seed, predictor_seed)):
        raise ValueError("Initialization seeds must be nonnegative integers")
    config = variant_configuration(width)
    model = build_nextlat_model(
        architecture, width=width, seed=seed, predictor_seed=predictor_seed,
        device="cpu", backend=backend, predictor_hidden_width=predictor_hidden_width,
    )
    baseline = copy.deepcopy(model.nextlat_initialization)
    baseline_keys = list(model.state_dict())
    baseline_names = list(dict(model.named_parameters()))
    generator = torch.Generator(device="cpu").manual_seed(config["initialization"]["noise_seed"])
    noise_std = config["initialization"]["noise_std"]
    identity = torch.eye(width, dtype=torch.float32)
    changed = []
    with torch.no_grad():
        for index, block in enumerate(model.backbone.transformer.blocks):
            if (block.kv_proj.weight.shape != (2 * width, width)
                    or block.attn_out.weight.shape != (width, width)):
                raise ValueError("Value-path initialization requires full-width square V and O")
            prefix = f"backbone.transformer.blocks.{index}."
            for name, parameter in (("kv_proj.weight", block.kv_proj.weight[width:]),
                                    ("attn_out.weight", block.attn_out.weight)):
                noise = torch.randn((width, width), generator=generator, dtype=torch.float32)
                parameter.copy_(identity + noise_std * noise)
                changed.append({"parameter": prefix + name,
                                "rows": [width, 2 * width] if name == "kv_proj.weight" else [0, width]})

    # The paired builder used ALiBi, so no learned wpe was ever constructed.
    # Each block owns its own config; disable ALiBi in every copy explicitly.
    model.backbone.config.alibi = False
    for block in model.backbone.transformer.blocks:
        block.config.alibi = False
        if block.config.rope:
            raise AssertionError("The sinusoidal pilot must not enable RoPE")
    if model.backbone.config.rope or "wpe" in model.backbone.transformer:
        raise AssertionError("Expected the original ALiBi backbone without positional parameters")
    model.backbone.transformer["wpe"] = FixedSinusoidalPositions(
        width, base=config["position_encoding"]["base"])
    if (list(model.state_dict()) != baseline_keys
            or list(dict(model.named_parameters())) != baseline_names):
        raise AssertionError("The combined pilot changed model parameter or checkpoint topology")

    final_backbone_sha = canonical_parameter_sha256(model.backbone)
    model.backbone.a5_initialization = {
        **copy.deepcopy(baseline["backbone"]),
        "schema": "rt-a5-nextlat-variant-backbone-initialization-v1",
        "canonical_sha256": final_backbone_sha,
        "paired_base_initialization": copy.deepcopy(baseline["backbone"]),
        "rule": "Original RT parameters, then identity-centered W_V/W_O; fixed additive sinusoids",
    }
    model.variant_config = config
    model.nextlat_initialization = {
        **baseline,
        "schema": "rt-a5-nextlat-variant-initialization-v1",
        "backbone": copy.deepcopy(model.backbone.a5_initialization),
        "canonical_sha256": final_backbone_sha,
        "model_parameter_sha256": _parameter_sha256(model),
        "baseline_initialization": baseline,
        "variant_config": copy.deepcopy(config),
        "changed_parameter_slices": changed,
        "rule": "Paired baseline with isolated identity-centered value-path noise and fixed sinusoids",
    }
    return model.to(device=device, dtype=torch.float32)
