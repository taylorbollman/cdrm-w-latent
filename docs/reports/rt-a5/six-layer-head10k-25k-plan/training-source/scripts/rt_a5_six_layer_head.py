"""The original six-layer model with one existing embedding-access head."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from olmo.model import OLMo
from scripts.rt_a5_embedding_head import replace_embedding_head_block
from scripts.rt_a5_l1r_depth import build_model as build_depth_model, configuration as depth_configuration
from scripts.rt_a5_nextlat import _parameter_sha256


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/rt_a5_six_layer_head/base.json"


class EmbeddingHeadBackbone(OLMo):
    def __init__(self, original, *, enabled=True):
        cfg = original.config
        if (cfg.n_layers != 6 or cfg.block_group_size != 1 or not cfg.alibi
                or cfg.rope or cfg.embedding_layer_norm or cfg.embedding_dropout
                or cfg.scale_emb_init or "wpe" in original.transformer
                or original.activation_checkpointing_strategy is not None
                or cfg.include_bias):
            raise ValueError("Require the unchanged six-layer A5 raw-embedding path")
        with torch.random.fork_rng(devices=[]):
            super().__init__(copy.deepcopy(cfg), init_params=False)
        self.transformer = original.transformer
        self.a5_initialization = copy.deepcopy(original.a5_initialization)
        self.transformer.blocks[1] = replace_embedding_head_block(self.transformer.blocks[1])
        self.embedding_head_enabled = enabled
        self._head_in_progress = False
        self.train(original.training)

    def forward(self, input_ids, input_embeddings=None, attention_mask=None,
                attention_bias=None, past_key_values=None, use_cache=False,
                last_logits_only=False, output_hidden_states=None,
                return_pre_logits=False, return_logits=True, doc_lens=None,
                max_doc_lens=None, output_cdrm_states=None):
        if (input_embeddings is not None or past_key_values is not None or use_cache
                or self.activation_checkpointing_strategy is not None):
            raise ValueError("Embedding head supports raw token IDs without cached decoding or outer checkpointing")
        if self._head_in_progress:
            raise RuntimeError("Concurrent or reentrant embedding-head forward is unsupported")
        options = dict(attention_mask=attention_mask, attention_bias=attention_bias,
            last_logits_only=last_logits_only, output_hidden_states=output_hidden_states,
            return_pre_logits=return_pre_logits, return_logits=return_logits,
            doc_lens=doc_lens, max_doc_lens=max_doc_lens, output_cdrm_states=output_cdrm_states)
        if not self.embedding_head_enabled:
            return super().forward(input_ids, **options)
        self._head_in_progress = True
        hook = None
        calls = 0
        try:
            block = self.transformer.blocks[1]
            width = self.config.d_model
            head_dim = width // self.config.n_heads
            start = width + block.embedding_head_index * head_dim
            # These are views of the existing owned value rows, never another
            # Parameter registration. Their gradients combine with ordinary
            # provisional-self uses inside the recurrent block.
            embedding_values = F.linear(self.transformer.wte(input_ids),
                                        block.kv_proj.weight[start:start + head_dim])

            def inject(_block, args, kwargs):
                nonlocal calls
                calls += 1
                if calls != 1 or args[0].shape[:2] != embedding_values.shape[:2]:
                    raise RuntimeError("Expected one aligned upper-block invocation")
                if "embedding_values" in kwargs:
                    raise RuntimeError("Duplicate embedding-head input")
                return args, {**kwargs, "embedding_values": embedding_values}

            hook = block.register_forward_pre_hook(inject, with_kwargs=True)
            output = super().forward(input_ids, **options)
            if calls != 1:
                raise RuntimeError("Embedding values did not reach exactly block1")
            return output
        finally:
            if hook is not None:
                hook.remove()
            self._head_in_progress = False


def build_model(width=512, seed=1234, predictor_seed=1235, device="cpu", *,
                backend="tiled", predictor_hidden_width=None, enabled=True):
    if type(enabled) is not bool:
        raise ValueError("Embedding-head enabled flag must be boolean")
    model = build_depth_model(width, seed, predictor_seed, "cpu", n_layers=6,
        backend=backend, predictor_hidden_width=predictor_hidden_width)
    reference = copy.deepcopy(model.nextlat_initialization)
    before = _parameter_sha256(model)
    names = list(dict(model.named_parameters()))
    model.backbone = EmbeddingHeadBackbone(model.backbone, enabled=enabled)
    if _parameter_sha256(model) != before or list(dict(model.named_parameters())) != names:
        raise AssertionError("Embedding-head routing changed original parameter values or ownership")
    config = depth_configuration(width, n_layers=6)
    config.update(json.loads(CONFIG_PATH.read_text()))
    config.update(embedding_head_index=width // 64 - 1, embedding_head_enabled=enabled)
    model.experiment_config = config
    model.nextlat_initialization = {
        **reference,
        "schema": "rt-a5-six-layer-head-initialization-v1",
        "reference_six_layer_initialization": reference,
        "shared_six_layer_model_sha256": before,
        "model_parameter_sha256": _parameter_sha256(model),
        "baseline_parameter_tensors_changed": [],
        "additional_parameter_count": 0,
        "embedding_head_layer": 1,
        "embedding_head_index": config["embedding_head_index"],
        "experiment_config": copy.deepcopy(config),
        "rule": config["initialization_pairing"],
    }
    return model.to(device=device, dtype=torch.float32)
