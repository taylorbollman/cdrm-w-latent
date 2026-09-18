"""Four-layer first-window RT with one attached token-embedding bypass.

The input variant adds to the second block's input, before its existing
pre-attention normalization. The value variant delegates only its permanent
value write to a separate adapter. All original learned tensors, ALiBi,
NextLat final-hidden-state semantics, and the ordinary backbone forward remain
unchanged. The sole additional parameter is the bias-free embedding projection.
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
from scripts.rt_a5_l1r_depth import build_model as build_depth_model, configuration as depth_configuration
from scripts.rt_a5_nextlat import _parameter_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_embedding_injection/base.json"
VARIANTS = ("input", "value")


def configuration(width: int = 512, *, variant: str = "input", coefficient: float = .02,
                  projection_seed: int = 1236) -> dict:
    if variant not in VARIANTS:
        raise ValueError("variant must be 'input' or 'value'")
    if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
            or coefficient < 0):
        raise ValueError("coefficient must be finite and nonnegative")
    if type(projection_seed) is not int or projection_seed < 0:
        raise ValueError("projection_seed must be a nonnegative integer")
    config = depth_configuration(width, n_layers=4)
    config.update(json.loads(CONFIG_PATH.read_text()))
    config.update(variant=variant, coefficient=float(coefficient), projection_seed=projection_seed)
    config["injection"] = (
        "block1 input = preceding block output + coefficient * P_e(raw token embedding); existing block1 normalization follows"
        if variant == "input" else
        "block1 permanent V_t = W_V h_t + coefficient * P_e(raw token embedding_t); permanent K and temporary self K/V unchanged")
    return config


def make_projection(width: int, seed: int = 1236) -> nn.Linear:
    """A reproducible normal draw isolated from backbone/predictor/caller RNG."""
    if type(width) is not int or width < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Projection width and seed must be positive/nonnegative integers")
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(width, width, bias=False, device="cpu", dtype=torch.float32)
        torch.random.default_generator.manual_seed(seed)
        nn.init.normal_(projection.weight, mean=0.0, std=width ** -.5)
    return projection


class EmbeddingInjectedBackbone(OLMo):
    """Adopt the original transformer; supply one local tensor to block1.

    A scoped pre-hook preserves the vendor's full backbone forward and final
    latent selection. The attached bypass exists only in the current call's
    closure/autograd graph. Eager serial forwards are supported; cache decoding,
    reentrant forwards and outer checkpoint replay are explicitly unsupported.
    """

    def __init__(self, original: OLMo, projection: nn.Linear, *, coefficient: float,
                 variant: str):
        config = original.config
        if (config.n_layers != 4 or config.block_group_size != 1 or not config.alibi
                or config.rope or config.embedding_layer_norm or config.embedding_dropout
                or config.scale_emb_init or "wpe" in original.transformer
                or original.activation_checkpointing_strategy is not None):
            raise ValueError("Embedding injection requires the unchanged four-layer A5 embedding/ALiBi path")
        # Initialize OLMo's ordinary non-parameter state/cache, then adopt every
        # original transformer module. The temporary constructor draw is scoped.
        with torch.random.fork_rng(devices=[]):
            super().__init__(copy.deepcopy(config), init_params=False)
        self.transformer = original.transformer
        self.a5_initialization = copy.deepcopy(original.a5_initialization)
        self.embedding_projection = projection
        self.injection_coefficient = coefficient
        self.injection_variant = variant
        self._injection_in_progress = False
        self.train(original.training)

    def forward(self, input_ids, input_embeddings=None, attention_mask=None,
                attention_bias=None, past_key_values=None, use_cache=False,
                last_logits_only=False, output_hidden_states=None,
                return_pre_logits=False, return_logits=True, doc_lens=None,
                max_doc_lens=None, output_cdrm_states=None):
        if (input_embeddings is not None or past_key_values is not None or use_cache
                or self.activation_checkpointing_strategy is not None):
            raise ValueError("Embedding injection supports raw token IDs without cache or outer checkpointing")
        if self._injection_in_progress:
            raise RuntimeError("Concurrent or reentrant embedding-injection forward is unsupported")
        options = dict(attention_mask=attention_mask, attention_bias=attention_bias,
            last_logits_only=last_logits_only, output_hidden_states=output_hidden_states,
            return_pre_logits=return_pre_logits, return_logits=return_logits,
            doc_lens=doc_lens, max_doc_lens=max_doc_lens, output_cdrm_states=output_cdrm_states)
        if self.injection_coefficient == 0:
            # An exact baseline diagnostic; the extra projection is dormant.
            return super().forward(input_ids, **options)
        self._injection_in_progress = True
        hook = None
        calls = 0
        try:
            bypass = self.injection_coefficient * self.embedding_projection(self.transformer.wte(input_ids))

            def inject(_block, args, kwargs):
                nonlocal calls
                calls += 1
                if calls != 1 or args[0].shape != bypass.shape:
                    raise RuntimeError("Expected one same-shape block1 call per backbone forward")
                if self.injection_variant == "input":
                    return (args[0] + bypass, *args[1:]), kwargs
                if "value_bypass" in kwargs:
                    raise RuntimeError("Duplicate permanent-value bypass")
                return args, {**kwargs, "value_bypass": bypass}

            hook = self.transformer.blocks[1].register_forward_pre_hook(inject, with_kwargs=True)
            output = super().forward(input_ids, **options)
            if calls != 1:
                raise RuntimeError("The embedding bypass did not reach exactly block1")
            return output
        finally:
            if hook is not None:
                hook.remove()
            self._injection_in_progress = False


def build_model(width: int = 512, seed: int = 1234, predictor_seed: int = 1235,
                device: str | torch.device = "cpu", *, projection_seed: int = 1236,
                coefficient: float = .02, variant: str = "input", backend: str = "tiled",
                predictor_hidden_width: int | None = None):
    config = configuration(width, variant=variant, coefficient=coefficient,
                           projection_seed=projection_seed)
    model = build_depth_model(width, seed, predictor_seed, "cpu", n_layers=4,
        backend=backend, predictor_hidden_width=predictor_hidden_width)
    reference = copy.deepcopy(model.nextlat_initialization)
    original_names = list(dict(model.named_parameters()))
    transformer_hash = _parameter_sha256(model.backbone.transformer)
    projection = make_projection(width, projection_seed)
    backbone = EmbeddingInjectedBackbone(model.backbone, projection,
        coefficient=float(coefficient), variant=variant)
    if variant == "value":
        from scripts.rt_a5_value_bypass import replace_value_bypass_block
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
        "schema": "rt-a5-embedding-injection-backbone-initialization-v1",
        "architecture": "rt", "seed": seed, "width": width, "n_layers": 4,
        "variant": variant, "parameter_count": backbone_count,
        "canonical_sha256": canonical, "baseline_canonical_sha256": reference["canonical_sha256"],
        "baseline_initialization": copy.deepcopy(reference["backbone"]),
        "baseline_parameter_tensors_changed": [],
        "projection_seed": projection_seed, "projection_sha256": _parameter_sha256(projection),
        "projection_parameter_count": width * width, "coefficient": float(coefficient),
        "coefficient_learned": False,
        "two_or_six_layer_initialization_identity_claimed": False,
    }
    model.experiment_config = config
    model.nextlat_initialization = {
        **reference, "schema": "rt-a5-embedding-injection-initialization-v1",
        "variant": variant, "backbone": copy.deepcopy(backbone.a5_initialization),
        "canonical_sha256": canonical, "backbone_parameter_count": backbone_count,
        "backbone_parameter_tensors": len(list(backbone.parameters())),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_tensors": len(list(model.parameters())),
        "model_parameter_sha256": _parameter_sha256(model),
        "reference_four_layer_initialization": reference,
        "shared_four_layer_model_sha256": reference["model_parameter_sha256"],
        "shared_transformer_sha256": transformer_hash,
        "baseline_parameter_tensors_changed": [], "predictor_initialization_paired": True,
        "projection_seed": projection_seed, "projection_sha256": _parameter_sha256(projection),
        "projection_parameter_count": width * width, "coefficient": float(coefficient),
        "coefficient_learned": False, "experiment_config": copy.deepcopy(config),
        "rule": config["initialization_pairing"],
    }
    return model.to(device=device, dtype=torch.float32)
