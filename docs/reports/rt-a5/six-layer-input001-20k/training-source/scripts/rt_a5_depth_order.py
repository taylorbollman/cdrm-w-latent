"""Fresh A5 depth/order variants with the frozen NextLat objective and predictor.

SEQ4 uses a canonical four-block Mitchell draw, not a copied two-block prefix.
The RT variant keeps original two-block tensors in their original layer slots;
only the first block acquires the already-validated two-record attention mask.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from olmo.model import BufferCache, OLMo, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import canonical_parameter_sha256, model_config
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_depth_order/base.json"
VARIANTS = ("seq4_alibi", "rt_window2_first")


def configuration(variant: str, width: int = 512) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64")
    result = json.loads(CONFIG_PATH.read_text())
    selected = result.pop("variants")[variant]
    result.update(selected, variant=variant, width=width, n_heads=width//64,
                  mlp_hidden_size=4*width)
    result["attention"] = {
        "layers": (["ordinary full causal attention"]*4 if variant == "seq4_alibi" else
                   ["self provisional K/V and immediately previous permanent output K/V",
                    "full causal recurrent attention"]),
        "recurrent_write_rho": 1.0 if selected["architecture"] == "rt" else None,
        "gradient_truncation": False,
        "layer_indices": "zero based",
        "window_is_direct_read_limit_not_history_truncation": selected["window_layer"] is not None,
    }
    return result


def backbone_parameter_count(width: int, layers: int) -> int:
    """Bias-free GELU blocks, four learned norm vectors/block, V60 untied head."""
    return 12*layers*width*width + (4*layers+121)*width


def build_model(variant: str, width: int = 512, seed: int = 1234,
                predictor_seed: int = 1235, device: str | torch.device = "cpu", *,
                backend: str = "tiled", predictor_hidden_width: int | None = None):
    """Return A5NextLat with provenance dictionaries used by the existing trainer.

    Predictor construction and objective metadata come from the original factory.
    All construction uses isolated CPU RNG scopes. Parameter replacement happens
    before optimizer creation, retaining each RT layer's original parameter slot.
    """
    if any(type(value) is not int or value < 0 for value in (seed,predictor_seed)):
        raise ValueError("Initialization seeds must be nonnegative integers")
    config = configuration(variant,width)
    # Reuse the original wrapper/predictor construction, including its independent
    # seed and exact initialization. SEQ4 replaces only its temporary backbone.
    model = build_nextlat_model(config["architecture"],width=width,seed=seed,
        predictor_seed=predictor_seed,device="cpu",backend=backend,
        predictor_hidden_width=predictor_hidden_width)
    reference = copy.deepcopy(model.nextlat_initialization)
    reference_model_sha = _parameter_sha256(model)
    if variant == "seq4_alibi":
        backbone_config = model_config("seq",width)
        backbone_config.n_layers = 4
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            backbone = OLMo(backbone_config).to(dtype=torch.float32)
        canonical = canonical_parameter_sha256(backbone)
        count = sum(p.numel() for p in backbone.parameters())
        backbone.a5_initialization = {
            "schema":"rt-a5-depth-initialization-v1", "seed":seed, "width":width,
            "architecture":"seq", "n_layers":4, "parameter_count":count,
            "canonical_sha256":canonical, "conversion":None,
            "rule":"Canonical four-layer CPU Mitchell initialization from original A5 config with n_layers=4",
            "two_layer_initialization_identity_claimed":False,
        }
        model.backbone = backbone
    else:
        blocks = model.backbone.transformer.blocks
        previous = blocks[0]
        replacement_type = (WindowTwoRecurrentBlockTiled if isinstance(previous,OLMoRecurrentBlockTiled)
                            else WindowTwoRecurrentAutogradBlock)
        before_keys = list(model.state_dict())
        before_names = list(dict(model.named_parameters()))
        with torch.random.fork_rng(devices=[]):
            replacement = replacement_type(previous.layer_id,copy.deepcopy(previous.config),BufferCache())
        replacement.load_state_dict(previous.state_dict(),strict=True)
        replacement.train(previous.training)
        blocks[0] = replacement
        if (list(model.state_dict()) != before_keys or list(dict(model.named_parameters())) != before_names
                or _parameter_sha256(model) != reference_model_sha
                or canonical_parameter_sha256(model.backbone) != reference["canonical_sha256"]):
            raise AssertionError("First-layer mask changed original RT learned initialization")
    count = sum(p.numel() for p in model.backbone.parameters())
    if count != backbone_parameter_count(width,config["n_layers"]):
        raise AssertionError("Unexpected backbone parameter count")
    if (len(model.backbone.transformer.blocks) != config["n_layers"]
            or not model.backbone.config.alibi or model.backbone.config.rope
            or "wpe" in model.backbone.transformer
            or any(not block.config.alibi for block in model.backbone.transformer.blocks)
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise AssertionError("Unexpected depth, position encoding or model precision")
    model.experiment_config = config
    model.nextlat_initialization = {
        **reference, "schema":"rt-a5-depth-order-initialization-v1", "variant":variant,
        "n_layers":config["n_layers"], "backbone":copy.deepcopy(model.backbone.a5_initialization),
        "canonical_sha256":canonical_parameter_sha256(model.backbone),
        "backbone_parameter_count":count, "parameter_count":sum(p.numel() for p in model.parameters()),
        "model_parameter_sha256":_parameter_sha256(model), "experiment_config":copy.deepcopy(config),
        "reference_two_layer_initialization":reference,
        "exact_two_layer_learned_initialization":variant == "rt_window2_first",
        "changed_parameter_slices":[] if variant == "rt_window2_first" else None,
        "rule":config["initialization_pairing"],
    }
    return model.to(device=device,dtype=torch.float32)
