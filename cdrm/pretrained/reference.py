"""Independent, offline CoreNet oracle for ordinary OpenELM import checks.

The model arithmetic is executed from the byte-identical upstream files next to
this module. Only CoreNet-specific imports are removed from their AST. Registry
and framework plumbing are supplied below; attention, normalization, rotary
embedding, projections, initialization, and model forward methods are not
rewritten. This deliberately does not import our OpenELM implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


CORENET_REFERENCE_REVISION = "f9f83e616a34d02c422733a06a3fe5bde63ae575"
_SOURCE_ROOT = Path(__file__).with_name("_corenet_reference")
_EXECUTION_ORDER = (
    "math_utils.py",
    "rms_norm.py",
    "rotary_embeddings.py",
    "embedding.py",
    "linear_layer.py",
    "swish.py",
    "general_gpt.py",
)


def verify_reference_sources(source_root: Path | None = None) -> dict[str, Any]:
    """Verify the pinned local source snapshot before executing any of it."""
    source_root = _SOURCE_ROOT if source_root is None else Path(source_root)
    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest["revision"] != CORENET_REFERENCE_REVISION:
        raise ValueError("Unexpected CoreNet reference revision")
    entries = {entry["file"]: entry for entry in manifest["files"]}
    if set(entries) != {*_EXECUTION_ORDER, "LICENSE"}:
        raise ValueError("Unexpected CoreNet reference file inventory")
    for name, entry in entries.items():
        data = (source_root / name).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"CoreNet reference source checksum mismatch: {name}")
    return manifest


class _Registry:
    def register(self, *args: Any, **kwargs: Any):
        return lambda cls: cls


class _FrameworkBaseLanguageModel(nn.Module):
    """Only the nn.Module/opts constructor needed by GeneralGPTModel."""

    def __init__(self, opts: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__()
        self.opts = opts


def _reference_error(message: str) -> None:
    raise ValueError(message)


@lru_cache(maxsize=1)
def _upstream_module() -> ModuleType:
    verify_reference_sources()
    module = ModuleType("cdrm.pretrained._loaded_corenet_reference")
    # dataclasses resolves annotation ownership through sys.modules.
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    registry = _Registry()
    namespace.update(
        BaseLayer=nn.Module,
        BaseLanguageModel=_FrameworkBaseLanguageModel,
        MODEL_REGISTRY=registry,
        register_norm_fn=registry.register,
        register_act_fn=registry.register,
        logger=SimpleNamespace(error=_reference_error),
    )

    def get_normalization_layer(opts: argparse.Namespace, num_features: int, norm_type: str):
        if norm_type != "rms_norm":
            raise ValueError(f"Unsupported reference normalization: {norm_type}")
        # Native build_normalization_layer defaults to eps=1e-6. The private
        # override allows small fixtures to exercise an explicitly chosen eps.
        return namespace["RMSNorm"](
            num_features, eps=getattr(opts, "_reference_rms_norm_eps", 1e-6)
        )

    def build_activation_layer(opts: argparse.Namespace, act_type: str):
        if act_type != "swish":
            raise ValueError(f"Unsupported reference activation: {act_type}")
        return namespace["Swish"]()

    namespace.update(
        get_normalization_layer=get_normalization_layer,
        build_activation_layer=build_activation_layer,
    )
    try:
        for name in _EXECUTION_ORDER:
            path = _SOURCE_ROOT / name
            tree = ast.parse(path.read_text(), filename=str(path))
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and (node.module == "corenet" or node.module.startswith("corenet."))
                )
            ]
            exec(compile(tree, str(path), "exec"), namespace)
            if name == "rms_norm.py":
                namespace["norm_layers_tuple"] = (namespace["RMSNorm"],)
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def native_reference_config(
    *, model_name: str = "OpenELM-1_1B", vocab_size: int = 32128, max_context_length: int = 2048
) -> Any:
    """Derive geometry from the original CoreNet configuration constructor."""
    upstream = _upstream_module()
    return upstream.GPTConfig.from_name(
        model_name=model_name, vocab_size=vocab_size, max_context_length=max_context_length
    )


def _fixture_config(config: Any) -> SimpleNamespace:
    """Supply explicit small fixture dimensions to the unchanged native layers."""
    return SimpleNamespace(
        vocab_size=config.vocab_size,
        max_context_length=config.max_context_length,
        num_transformer_layers=len(config.num_query_heads),
        model_dim=config.model_dim,
        head_dim=config.head_dim,
        num_query_heads=list(config.num_query_heads),
        num_kv_heads=list(config.num_kv_heads),
        ffn_multipliers=[size / config.model_dim for size in config.ffn_intermediate_sizes],
        # Fixture intermediate sizes have already been selected explicitly.
        ffn_dim_divisor=1,
        ffn_with_glu=True,
        activation_fn_name="swish",
        normalization_layer_name="rms_norm",
        normalize_qk_projections=True,
        share_input_output_layers=True,
        rope_freq_constant=config.rope_freq_constant,
        rope_max_length=max(2 * config.max_context_length, 1),
    )


def build_corenet_reference(
    config: Any | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build the independent ordinary model, preserving native state-dict keys.

    With ``config=None``, the original named 1.1B configuration determines all
    widths independently, with native vocabulary32128 and padding index32000.
    A config argument supplies explicit dimensions for small test fixtures.
    Parameters remain FP32; BF16 runtime comparisons should use autocast, as in
    native pretraining. This avoids casting the original RoPE frequency buffer.
    """
    if dtype != torch.float32:
        raise ValueError("The native oracle uses FP32 parameters; use autocast for BF16 execution")
    upstream = _upstream_module()
    model_config = native_reference_config() if config is None else _fixture_config(config)
    opts = argparse.Namespace()
    for name, value in {
        "model.language_modeling.general_gpt.model_name": "OpenELM-1_1B",
        "model.language_modeling.general_gpt.vocab_size": model_config.vocab_size,
        "model.language_modeling.general_gpt.max_context_length": model_config.max_context_length,
        "model.language_modeling.general_gpt.padding_index": 32000 if config is None else config.padding_idx,
        "_reference_rms_norm_eps": 1e-6 if config is None else config.rms_norm_eps,
    }.items():
        setattr(opts, name, value)

    # GeneralGPTModel reads its class-level config factory. Supplying this
    # factory selects fixture dimensions without changing upstream methods or
    # mutating shared upstream configuration classes.
    class ConfigFactory:
        @staticmethod
        def from_name(**kwargs: Any) -> Any:
            return model_config

    class ReferenceModel(upstream.GeneralGPTModel):
        config = ConfigFactory

    with torch.device(device):
        model = ReferenceModel(opts)
    # Native LinearLayer constructs parameters with torch.Tensor, which does
    # not honor the device context above. Materialize the complete module on
    # the requested device before use (including nonpersistent RoPE buffers).
    model.to(device=device, dtype=dtype)
    model.reference_config = model_config
    model.reference_revision = CORENET_REFERENCE_REVISION
    return model


@dataclass
class CoreNetReferenceOutput:
    logits: Tensor
    last_hidden_state: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None


def reference_forward(
    model: nn.Module,
    input_ids: Tensor,
    *,
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
    use_cache: bool = False,
) -> CoreNetReferenceOutput:
    """Capture native logits/final states and correctly invoke its cache API.

    Native CoreNet supports causal prefill and single-token cached decoding.
    For a cached chunk, this helper invokes that unchanged single-token path
    repeatedly. It does not silently use CoreNet's top-left SDPA causal mask
    for unequal query/key lengths. Padding/explicit positions are intentionally
    outside this oracle; compare those adapter extensions to unpadded examples.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
    if past_key_values is not None and not use_cache:
        raise ValueError("past_key_values requires use_cache=True")
    if past_key_values is not None and input_ids.shape[1] > 1:
        outputs = []
        for token in input_ids.split(1, dim=1):
            output = reference_forward(model, token, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            outputs.append(output)
        return CoreNetReferenceOutput(
            logits=torch.cat([out.logits for out in outputs], dim=1),
            last_hidden_state=torch.cat([out.last_hidden_state for out in outputs], dim=1),
            past_key_values=past_key_values,
        )

    hidden_states = []
    hook = model.norm.register_forward_hook(lambda _module, _args, output: hidden_states.append(output))
    try:
        if use_cache:
            native_output = model(
                {
                    "input_ids": input_ids,
                    "past_keys": None if past_key_values is None else [kv[0] for kv in past_key_values],
                    "past_values": None if past_key_values is None else [kv[1] for kv in past_key_values],
                    "use_kv_cache": True,
                    "is_causal": past_key_values is None,
                }
            )
            logits = native_output["logits"]
            cache = tuple(zip(native_output["past_keys"], native_output["past_values"]))
        else:
            logits = model(input_ids)
            cache = None
    finally:
        hook.remove()
    if len(hidden_states) != 1:
        raise RuntimeError("Expected exactly one native final-normalization output")
    return CoreNetReferenceOutput(logits, hidden_states[0], cache)
