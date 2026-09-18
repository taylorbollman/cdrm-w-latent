"""Six-layer first-window RT with a permanent-value embedding bypass.

Only the second block permanent value receives the attached token projection. All 61
original six-layer backbone/predictor tensors and the original forward/objective
semantics are preserved; the sole new parameter is the independently seeded Pe.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from olmo.model import OLMo
from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_embedding_injection import EmbeddingInjectedBackbone, make_projection
from scripts.rt_a5_l1r_depth import build_model as build_depth_model, configuration as depth_configuration
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_value_bypass import replace_value_bypass_block


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_six_layer_value/base.json"
SCHEDULES = json.loads(CONFIG_PATH.read_text())["schedules"]
VARIANTS = tuple(SCHEDULES)


def coefficient(update: int, variant: str = "constant") -> float:
    if type(update) is not int or variant not in VARIANTS:
        raise ValueError("Coefficient requires an integer update and a supported variant")
    if variant == "constant":
        return .01
    return .01 * min(max(update, 0) / 20000, 1)


def set_update(model, update: int) -> float:
    if type(update) is not int or update < 0:
        raise ValueError("Applied schedule update must be a nonnegative integer")
    variant = model.experiment_config["variant"]
    value = coefficient(update, variant)
    model.backbone.injection_coefficient = value
    model.schedule_update = update
    return value


def configuration(width: int = 512, *, variant: str = "constant", projection_seed: int = 1236) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Variant must be constant or linear")
    if type(projection_seed) is not int or projection_seed < 0:
        raise ValueError("projection_seed must be a nonnegative integer")
    config = depth_configuration(width, n_layers=6)
    config.update(json.loads(CONFIG_PATH.read_text()))
    config.pop("schedules")
    config.update(variant=variant, projection_seed=projection_seed,
                  initial_coefficient=coefficient(0, variant), schedule=copy.deepcopy(SCHEDULES[variant]))
    config["injection"] = "block1 permanent V_t = W_V h_t + lambda(update) * P_e(raw token embedding_t); permanent K and temporary self K/V unchanged"
    return config


class SixLayerValueBackbone(EmbeddingInjectedBackbone):
    """Reuse the existing scoped bypass-routing forward without changing it."""

    def __init__(self, original: OLMo, projection: nn.Linear, *, coefficient: float,
                 variant: str = "value"):
        config = original.config
        if variant != "value":
            raise ValueError("Only permanent-value injection is supported")
        if (config.n_layers != 6 or config.block_group_size != 1 or not config.alibi
                or config.rope or config.embedding_layer_norm or config.embedding_dropout
                or config.scale_emb_init or "wpe" in original.transformer
                or original.activation_checkpointing_strategy is not None):
            raise ValueError("Embedding injection requires the unchanged six-layer A5 embedding/ALiBi path")
        # Initialize OLMo's ordinary non-parameter state/cache, then adopt every
        # original transformer module. The temporary constructor draw is scoped.
        with torch.random.fork_rng(devices=[]):
            OLMo.__init__(self, copy.deepcopy(config), init_params=False)
        self.transformer = original.transformer
        self.a5_initialization = copy.deepcopy(original.a5_initialization)
        self.embedding_projection = projection
        self.injection_coefficient = coefficient
        self.injection_variant = variant
        self._injection_in_progress = False
        self.train(original.training)


def build_model(width: int = 512, seed: int = 1234, predictor_seed: int = 1235,
                device: str | torch.device = "cpu", *, projection_seed: int = 1236,
                variant: str = "constant", backend: str = "tiled",
                predictor_hidden_width: int | None = None):
    config = configuration(width, variant=variant, projection_seed=projection_seed)
    initial_coefficient = coefficient(0, variant)
    model = build_depth_model(width, seed, predictor_seed, "cpu", n_layers=6,
        backend=backend, predictor_hidden_width=predictor_hidden_width)
    reference = copy.deepcopy(model.nextlat_initialization)
    original_names = list(dict(model.named_parameters()))
    transformer_hash = _parameter_sha256(model.backbone.transformer)
    projection = make_projection(width, projection_seed)
    backbone = SixLayerValueBackbone(model.backbone, projection,
        coefficient=initial_coefficient)
    backbone.transformer.blocks[1] = replace_value_bypass_block(backbone.transformer.blocks[1])
    model.backbone = backbone
    names = list(dict(model.named_parameters()))
    if ([name for name in names if name != "backbone.embedding_projection.weight"] != original_names
            or len(names) != len(original_names) + 1
            or _parameter_sha256(backbone.transformer) != transformer_hash
            or _parameter_sha256(model.predictor) != reference["predictor_sha256"]):
        raise AssertionError("Embedding routing changed an original learned tensor or its ownership")
    canonical = canonical_parameter_sha256(backbone)
    backbone_count = sum(p.numel() for p in backbone.parameters())
    backbone.a5_initialization = {
        "schema": "rt-a5-six-layer-value-backbone-initialization-v1",
        "architecture": "rt", "seed": seed, "width": width, "n_layers": 6,
        "variant": variant, "parameter_count": backbone_count,
        "canonical_sha256": canonical, "baseline_canonical_sha256": reference["canonical_sha256"],
        "baseline_initialization": copy.deepcopy(reference["backbone"]),
        "baseline_parameter_tensors_changed": [],
        "projection_seed": projection_seed, "projection_sha256": _parameter_sha256(projection),
        "projection_parameter_count": width * width, "coefficient": initial_coefficient,
        "coefficient_learned": False,
        "two_or_four_layer_initialization_identity_claimed": False,
    }
    model.experiment_config = config
    model.nextlat_initialization = {
        **reference, "schema": "rt-a5-six-layer-value-initialization-v1",
        "variant": variant, "backbone": copy.deepcopy(backbone.a5_initialization),
        "canonical_sha256": canonical, "backbone_parameter_count": backbone_count,
        "backbone_parameter_tensors": len(list(backbone.parameters())),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_tensors": len(list(model.parameters())),
        "model_parameter_sha256": _parameter_sha256(model),
        "reference_six_layer_initialization": reference,
        "shared_six_layer_model_sha256": reference["model_parameter_sha256"],
        "shared_transformer_sha256": transformer_hash,
        "baseline_parameter_tensors_changed": [], "predictor_initialization_paired": True,
        "projection_seed": projection_seed, "projection_sha256": _parameter_sha256(projection),
        "projection_parameter_count": width * width, "coefficient": initial_coefficient,
        "coefficient_learned": False, "initial_coefficient": initial_coefficient,
        "schedule": copy.deepcopy(SCHEDULES[variant]), "experiment_config": copy.deepcopy(config),
        "rule": config["initialization_pairing"],
    }
    set_update(model, 0)
    return model.to(device=device, dtype=torch.float32)
