"""Paired embedding routes for the two-layer mixed A5/MAD NextLat model.

The task interface, objective and canonical initialization remain in
``rt_nextlat_tasks``. Only the upper block (index 1) changes. Input/value
addition uses an attached raw-token projection and a fixed coefficient. The
head route replaces one historical value head using its existing projection
rows; it adds neither parameters nor a gate. Existing contextual keys and
provisional self K/V retain their ordinary paths.

This adapter is separate from the frozen baseline source closure. It reuses
the previously validated permanent-write/autograd adapters and supports eager
serial FP32 forwards only, with an explicit naive reference for CPU checks.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo.model import OLMo
from cdrm.rt_nextlat_tasks import (
    TaskNextLat, build_model as build_base_model,
    read_configuration as read_base_configuration,
)
from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_embedding_injection import make_projection
from scripts.rt_a5_embedding_head import replace_embedding_head_block
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_value_bypass import replace_value_bypass_block


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("input", "value", "head")
# Additional imported implementation/configuration dependencies beyond the
# frozen mixed trainer's source_manifest. The thin trainer also captures itself.
SOURCE_PATHS = (
    "cdrm/rt_nextlat_task_embeddings.py",
    "scripts/rt_a5_embedding_injection.py",
    "scripts/rt_a5_embedding_head.py",
    "scripts/rt_a5_value_bypass.py",
    "scripts/rt_a5_l1r_depth.py",
    "configs/rt_a5_embedding_injection/base.json",
    "configs/rt_a5_l1r_depth/base.json",
    "configs/rt_nextlat_task_embeddings/input_d128.json",
    "configs/rt_nextlat_task_embeddings/value_d128.json",
    "configs/rt_nextlat_task_embeddings/head_d128.json",
)


def source_manifest() -> dict[str, str]:
    """Additional immutable source/config hashes for the variant trainer."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(SOURCE_PATHS)}


def read_configuration(config: dict | str | Path | None = None) -> dict:
    """Resolve an optional explicit route without changing the base contract.

    A missing ``embedding_injection`` key is exactly the baseline. Zero
    coefficients and ``head.enabled=False`` are diagnostic baseline limits;
    the three experiment files select active routes and no schedule.
    """
    result = read_base_configuration(config)
    if "embedding_injection" not in result:
        return result
    raw = result["embedding_injection"]
    if not isinstance(raw, dict) or raw.get("variant") not in VARIANTS:
        raise ValueError("embedding_injection must specify input, value or head")
    if type(raw.get("layer_index", 1)) is not int or raw.get("layer_index", 1) != 1:
        raise ValueError("Only the upper RT block at layer_index 1 may change")
    normalized = {"variant": raw["variant"], "layer_index": 1}
    if raw["variant"] in ("input", "value"):
        allowed = {"variant", "layer_index", "coefficient", "projection_seed"}
        coefficient = raw.get("coefficient", .01)
        seed = raw.get("projection_seed", 1237)
        if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
                or coefficient < 0):
            raise ValueError("coefficient must be finite and nonnegative")
        if type(seed) is not int or seed < 0:
            raise ValueError("projection_seed must be a nonnegative integer")
        normalized.update(coefficient=float(coefficient), projection_seed=seed)
    else:
        allowed = {"variant", "layer_index", "head_index", "enabled"}
        head = raw.get("head_index", result["backbone"]["n_heads"] - 1)
        enabled = raw.get("enabled", True)
        if type(head) is not int or not 0 <= head < result["backbone"]["n_heads"]:
            raise ValueError("head_index must name an existing attention head")
        if type(enabled) is not bool:
            raise ValueError("head enabled must be boolean")
        normalized.update(head_index=head, enabled=enabled)
    if set(raw) - allowed:
        raise ValueError("Unexpected embedding injection fields: " + ", ".join(sorted(set(raw) - allowed)))
    result["embedding_injection"] = normalized
    return result


class TaskEmbeddingBackbone(OLMo):
    """Adopt the existing transformer and attach one call-local upper input."""

    def __init__(self, original: OLMo, route: dict):
        cfg = original.config
        if (cfg.n_layers != 2 or cfg.block_group_size != 1 or not cfg.alibi
                or cfg.rope or cfg.embedding_layer_norm or cfg.embedding_dropout
                or cfg.scale_emb_init or "wpe" in original.transformer
                or original.activation_checkpointing_strategy is not None
                or cfg.include_bias or not cfg.reference_eager):
            raise ValueError("Require the unchanged two-layer FP32 eager raw-embedding path")
        with torch.random.fork_rng(devices=[]):
            super().__init__(copy.deepcopy(cfg), init_params=False)
        self.transformer = original.transformer
        self.embedding_route = copy.deepcopy(route)
        self._embedding_route_in_progress = False
        if route["variant"] in ("input", "value"):
            self.embedding_projection = make_projection(cfg.d_model, route["projection_seed"])
            if route["variant"] == "value":
                self.transformer.blocks[1] = replace_value_bypass_block(self.transformer.blocks[1])
        else:
            replacement = replace_embedding_head_block(self.transformer.blocks[1])
            # Retain the historical last-head convention (and its ALiBi slope)
            # unless an explicit diagnostic config selects another head.
            replacement.embedding_head_index = route["head_index"]
            self.transformer.blocks[1] = replacement
        self.train(original.training)

    def forward(self, input_ids, input_embeddings=None, attention_mask=None,
                attention_bias=None, past_key_values=None, use_cache=False,
                last_logits_only=False, output_hidden_states=None,
                return_pre_logits=False, return_logits=True, doc_lens=None,
                max_doc_lens=None, output_cdrm_states=None):
        if (input_embeddings is not None or past_key_values is not None or use_cache
                or doc_lens is not None or max_doc_lens is not None
                or self.activation_checkpointing_strategy is not None):
            raise ValueError("Embedding routes require raw token IDs without caches, packed documents or outer checkpointing")
        if self._embedding_route_in_progress:
            raise RuntimeError("Concurrent or reentrant embedding-route forward is unsupported")
        options = dict(attention_mask=attention_mask, attention_bias=attention_bias,
            last_logits_only=last_logits_only, output_hidden_states=output_hidden_states,
            return_pre_logits=return_pre_logits, return_logits=return_logits,
            output_cdrm_states=output_cdrm_states)
        route = self.embedding_route
        variant = route["variant"]
        if ((variant == "head" and not route["enabled"])
                or (variant != "head" and route["coefficient"] == 0)):
            return super().forward(input_ids, **options)
        self._embedding_route_in_progress = True
        hook = None
        calls = 0
        try:
            block = self.transformer.blocks[1]
            raw_embeddings = self.transformer.wte(input_ids)
            if variant == "head":
                width = self.config.d_model
                head_dim = width // self.config.n_heads
                start = width + route["head_index"] * head_dim
                # A view of the original value rows, with ordinary shared
                # gradient accumulation from both permanent and self uses.
                routed = F.linear(raw_embeddings, block.kv_proj.weight[start:start + head_dim])
            else:
                routed = route["coefficient"] * self.embedding_projection(raw_embeddings)

            def inject(_block, args, kwargs):
                nonlocal calls
                calls += 1
                if calls != 1 or args[0].shape[:2] != routed.shape[:2]:
                    raise RuntimeError("Expected one aligned upper-block invocation")
                if variant == "input":
                    if args[0].shape != routed.shape:
                        raise RuntimeError("Input projection must match the full upper input")
                    return (args[0] + routed, *args[1:]), kwargs
                name = "value_bypass" if variant == "value" else "embedding_values"
                if name in kwargs:
                    raise RuntimeError("Duplicate embedding-route input")
                return args, {**kwargs, name: routed}

            hook = block.register_forward_pre_hook(inject, with_kwargs=True)
            result = super().forward(input_ids, **options)
            if calls != 1:
                raise RuntimeError("Embedding route did not reach exactly block1")
            return result
        finally:
            if hook is not None:
                hook.remove()
            self._embedding_route_in_progress = False


def build_model(config: dict | str | Path | None = None, *, seed: int = 1234,
                predictor_seed: int = 1235, fuzzy_seed: int = 1236,
                device: str | torch.device = "cpu", backend: str = "tiled") -> TaskNextLat:
    """Keep all shared tensors and caller RNG exactly paired with baseline."""
    config = read_configuration(config)
    base_config = copy.deepcopy(config)
    route = base_config.pop("embedding_injection", None)
    model = build_base_model(base_config, seed=seed, predictor_seed=predictor_seed,
        fuzzy_seed=fuzzy_seed, device="cpu", backend=backend)
    if route is None:
        return model.to(device=device, dtype=torch.float32)
    baseline_initialization = copy.deepcopy(model.initialization)
    original_names = list(dict(model.named_parameters()))
    transformer_sha = _parameter_sha256(model.backbone.transformer)
    model.backbone = TaskEmbeddingBackbone(model.backbone, route)
    names = list(dict(model.named_parameters()))
    projection_name = "backbone.embedding_projection.weight"
    expected_extra = [] if route["variant"] == "head" else [projection_name]
    if ([name for name in names if name not in expected_extra] != original_names
            or set(names) - set(original_names) != set(expected_extra)
            or _parameter_sha256(model.backbone.transformer) != transformer_sha
            or _parameter_sha256(model.predictor) != baseline_initialization["predictor_sha256"]
            or len(list(model.named_parameters(remove_duplicate=False))) != len(names)):
        raise AssertionError("Embedding routing changed a shared tensor or its unique ownership")
    # Preserve the base factory's resolved backend configuration, plus route.
    model.task_config["embedding_injection"] = copy.deepcopy(route)
    backbone_count = sum(p.numel() for p in model.backbone.parameters())
    additional = backbone_count - baseline_initialization["backbone_parameter_count"]
    if additional != (0 if route["variant"] == "head" else model.backbone.config.d_model ** 2):
        raise AssertionError("Unexpected embedding-route parameter count")
    model.initialization = {
        **baseline_initialization,
        "schema": "rt-nextlat-task-embeddings-initialization-v1",
        "baseline_initialization": baseline_initialization,
        "shared_model_parameter_sha256": baseline_initialization["model_parameter_sha256"],
        "shared_transformer_sha256": transformer_sha,
        "baseline_parameter_tensors_changed": [],
        "predictor_initialization_paired": True,
        "embedding_injection": copy.deepcopy(route),
        "added_parameter_count": additional,
        "coefficient_learned": False,
        "canonical_sha256": canonical_parameter_sha256(model.backbone),
        "backbone_parameter_count": backbone_count,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_tensors": len(names),
        "model_parameter_sha256": _parameter_sha256(model),
        "task_config": copy.deepcopy(model.task_config),
        "rule": baseline_initialization["rule"] + "; unchanged shared tensors; "
                "upper-block-only embedding route, separately isolated projection RNG",
    }
    if expected_extra:
        model.initialization.update(
            projection_seed=route["projection_seed"],
            projection_sha256=_parameter_sha256(model.backbone.embedding_projection))
    return model.to(device=device, dtype=torch.float32)
