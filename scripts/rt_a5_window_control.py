"""The winning first-window/full RT backbone with pure CE and no NextLat module.

Construction reuses the frozen winning factory to retain identical learned
backbone tensors and block ownership. Its temporary CPU predictor is discarded
before returning the bare backbone, so it never enters the optimizer or GPU.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_depth_order import build_model as build_reference_model
from scripts.rt_a5_nextlat import _parameter_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_window_control/base.json"
VARIANT = "rt_window2_first_ce"


def build_model(width: int = 512, seed: int = 1234,
                device: str | torch.device = "cpu", *, backend: str = "tiled"):
    """Return the exact paired RT backbone and explicit initialization metadata."""
    reference = build_reference_model("rt_window2_first", width=width, seed=seed,
                                     predictor_seed=1235, device="cpu", backend=backend)
    hybrid_initial = copy.deepcopy(reference.nextlat_initialization)
    backbone = reference.backbone
    backbone_initial = copy.deepcopy(backbone.a5_initialization)
    expected = {name.removeprefix("backbone."): value for name, value in reference.state_dict().items()
                if name.startswith("backbone.")}
    config = json.loads(CONFIG_PATH.read_text())
    config.update(width=width, n_heads=width // 64, mlp_hidden_size=4 * width,
                  attention=copy.deepcopy(reference.experiment_config["attention"]))
    count = sum(p.numel() for p in backbone.parameters())
    canonical = canonical_parameter_sha256(backbone)
    if (list(backbone.state_dict()) != list(expected)
            or any(not torch.equal(value, expected[name]) for name, value in backbone.state_dict().items())
            or count != hybrid_initial["backbone_parameter_count"]
            or canonical != hybrid_initial["canonical_sha256"]
            or hasattr(backbone, "predictor")
            or any(name.startswith(("backbone.", "predictor.")) for name, _ in backbone.named_parameters())):
        raise AssertionError("Removing the NextLat wrapper changed the paired backbone")
    backbone.experiment_config = config
    backbone.a5_initialization = {
        "schema": "rt-a5-window-control-initialization-v1", "variant": VARIANT,
        "architecture": "rt", "seed": seed, "width": width, "n_layers": 2,
        "parameter_count": count, "backbone_parameter_count": count,
        "parameter_tensors": len(list(backbone.parameters())), "predictor_parameter_count": 0,
        "predictor_registered": False, "canonical_sha256": canonical,
        "model_parameter_sha256": _parameter_sha256(backbone),
        "reference_nextlat_initialization": hybrid_initial,
        "reference_backbone_initialization": backbone_initial,
        "exact_reference_backbone_initialization": True, "changed_backbone_parameter_slices": [],
        "experiment_config": copy.deepcopy(config), "rule": config["initialization_pairing"],
    }
    return backbone.to(device=device, dtype=torch.float32)
