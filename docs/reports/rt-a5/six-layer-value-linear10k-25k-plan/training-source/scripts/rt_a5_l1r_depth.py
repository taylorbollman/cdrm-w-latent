"""Deeper first-window RT with a canonical Mitchell draw at its actual depth.

The first recurrent block has the existing parameter-free window-2 mask; all
remaining blocks retain full recurrent attention. NextLat's wrapper, predictor
construction and objective are the original implementations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from olmo.checkpoint_conversion import convert_model
from olmo.model import BufferCache, OLMo, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import canonical_parameter_sha256, model_config
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_l1r_depth/base.json"
VARIANT = "rt_l1r_depth"


def configuration(width: int = 512, n_layers: int = 6) -> dict:
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64")
    if type(n_layers) is not int or n_layers < 2:
        raise ValueError("n_layers must be an integer >= 2, including the first windowed block")
    config = json.loads(CONFIG_PATH.read_text())
    config.update(width=width, n_heads=width // 64, mlp_hidden_size=4 * width,
                  n_layers=n_layers, full_recurrent_layers=n_layers - 1)
    config["attention"] = {
        "layers": ["self provisional K/V and immediately previous permanent output K/V"]
                  + ["full causal recurrent attention"] * (n_layers - 1),
        "recurrent_write_rho": 1.0, "gradient_truncation": False,
        "layer_indices": "zero based", "window_is_direct_read_limit_not_history_truncation": True,
    }
    return config


def backbone_parameter_count(width: int, n_layers: int) -> int:
    return 12 * n_layers * width * width + (4 * n_layers + 121) * width


def build_model(width: int = 512, seed: int = 1234, predictor_seed: int = 1235,
                device: str | torch.device = "cpu", *, n_layers: int = 6,
                backend: str = "tiled", predictor_hidden_width: int | None = None):
    """Build fresh at the declared depth; replace only the first block's mask."""
    config = configuration(width, n_layers)
    if any(type(value) is not int or value < 0 for value in (seed, predictor_seed)):
        raise ValueError("Initialization seeds must be nonnegative integers")
    # Keep the historical wrapper and separately seeded predictor unchanged.
    # Its temporary two-layer backbone is replaced before optimizer creation.
    model = build_nextlat_model("rt", width=width, seed=seed, predictor_seed=predictor_seed,
        device="cpu", backend=backend, predictor_hidden_width=predictor_hidden_width)
    reference = copy.deepcopy(model.nextlat_initialization)
    predictor_hash = _parameter_sha256(model.predictor)
    source_config = model_config("seq", width)
    target_config = model_config("rt", width, backend=backend)
    source_config.n_layers = target_config.n_layers = n_layers
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        source = OLMo(source_config).to(dtype=torch.float32)
        canonical = canonical_parameter_sha256(source)
        backbone = OLMo(target_config).to(dtype=torch.float32)
        conversion = convert_model(source, backbone).to_dict()
    if canonical_parameter_sha256(backbone) != canonical:
        raise AssertionError("Depth-specific SEQ-to-RT conversion changed canonical learned values")
    previous = backbone.transformer.blocks[0]
    replacement_type = (WindowTwoRecurrentBlockTiled if isinstance(previous, OLMoRecurrentBlockTiled)
                        else WindowTwoRecurrentAutogradBlock)
    old_names = list(dict(backbone.named_parameters()))
    old_hash = _parameter_sha256(backbone)
    with torch.random.fork_rng(devices=[]):
        replacement = replacement_type(previous.layer_id, copy.deepcopy(previous.config), BufferCache())
    replacement.load_state_dict(previous.state_dict(), strict=True)
    replacement.train(previous.training)
    backbone.transformer.blocks[0] = replacement
    if (list(dict(backbone.named_parameters())) != old_names or _parameter_sha256(backbone) != old_hash
            or canonical_parameter_sha256(backbone) != canonical):
        raise AssertionError("The first-block window replacement changed learned initialization")
    count = sum(parameter.numel() for parameter in backbone.parameters())
    if count != backbone_parameter_count(width, n_layers):
        raise AssertionError("Unexpected depth-specific backbone parameter count")
    if (len(backbone.transformer.blocks) != n_layers
            or not backbone.config.alibi or backbone.config.rope or "wpe" in backbone.transformer
            or any(not block.config.alibi or block.layer_id != i for i, block in enumerate(backbone.transformer.blocks))
            or any(parameter.dtype != torch.float32 for parameter in backbone.parameters())):
        raise AssertionError("Unexpected recurrent depth, positions, block ownership or precision")
    backbone.a5_initialization = {
        "schema": "rt-a5-l1r-depth-backbone-initialization-v1", "architecture": "rt",
        "seed": seed, "width": width, "n_layers": n_layers, "parameter_count": count,
        "canonical_sha256": canonical, "conversion": conversion,
        "two_layer_initialization_identity_claimed": False,
        "rule": "Canonical CPU Mitchell SEQ at actual depth; exhaustive weights-only conversion to RT; parameter-free first-block window",
    }
    model.backbone = backbone
    if _parameter_sha256(model.predictor) != predictor_hash or predictor_hash != reference["predictor_sha256"]:
        raise AssertionError("Depth change altered the original independently seeded predictor")
    model.experiment_config = config
    model.nextlat_initialization = {
        **reference, "schema": "rt-a5-l1r-depth-initialization-v1", "variant": VARIANT,
        "n_layers": n_layers, "backbone": copy.deepcopy(backbone.a5_initialization),
        "canonical_sha256": canonical, "backbone_parameter_count": count,
        "backbone_parameter_tensors": len(list(backbone.parameters())),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_tensors": len(list(model.parameters())), "model_parameter_sha256": _parameter_sha256(model),
        "reference_two_layer_initialization": reference,
        "two_layer_backbone_initialization_paired": False,
        "window_replacement_changed_parameter_slices": [],
        "predictor_initialization_paired": True, "experiment_config": copy.deepcopy(config),
        "rule": config["initialization_pairing"],
    }
    return model.to(device=device, dtype=torch.float32)
