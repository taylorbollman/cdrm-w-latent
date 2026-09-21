"""Offline ordinary OLMo oracle executing pinned, pristine upstream source.

The bundled files are byte-identical to allenai/OLMo v0.2.4 and the selected
native checkpoint config. The loader removes package-relative *imports* and
supplies their symbols from the same pinned files in an isolated namespace.
Only StrEnum is needed from util.py; its unchanged class is extracted to avoid
unrelated logging/network dependencies. No model, block, attention, LayerNorm,
RoPE, activation, initialization or forward method is rewritten. In particular,
this module never imports the project's modified ``olmo`` package or adapter.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


OLMO_REFERENCE_REVISION = "b3741bc21f1dd504838b7dbd9878ee077ded63bd"
_SOURCE_ROOT = Path(__file__).with_name("_olmo_reference")
_EXECUTION_ORDER = (
    "aliases.py", "exceptions.py", "util.py", "config.py",
    "initialization.py", "torch_util.py", "model.py",
)


def verify_olmo_reference_sources(source_root: Path | None = None) -> dict[str, Any]:
    source_root = _SOURCE_ROOT if source_root is None else Path(source_root)
    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest["revision"] != OLMO_REFERENCE_REVISION:
        raise ValueError("Unexpected OLMo reference revision")
    entries = {entry["file"]: entry for entry in manifest["files"]}
    if set(entries) != {*_EXECUTION_ORDER, "checkpoint_config.json", "LICENSE"}:
        raise ValueError("Unexpected OLMo reference file inventory")
    for name, entry in entries.items():
        data = (source_root / name).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"OLMo reference source checksum mismatch: {name}")
    return manifest


@lru_cache(maxsize=1)
def _upstream_module() -> ModuleType:
    verify_olmo_reference_sources()
    module = ModuleType("cdrm.pretrained._loaded_olmo_reference")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace["Enum"] = Enum
    try:
        for filename in _EXECUTION_ORDER:
            path = _SOURCE_ROOT / filename
            tree = ast.parse(path.read_text(), filename=str(path))
            if filename == "util.py":
                tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StrEnum"]
                if len(tree.body) != 1:
                    raise ValueError("Pinned util.py must contain exactly one StrEnum class")
            else:
                tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
            exec(compile(tree, str(path), "exec"), namespace)
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def native_olmo_reference_config(*, device: str | torch.device = "cpu") -> Any:
    """Resolve the selected checkpoint through the original ModelConfig class."""
    upstream = _upstream_module()
    raw = json.loads((_SOURCE_ROOT / "checkpoint_config.json").read_text())
    accepted = {field.name for field in fields(upstream.ModelConfig)}
    values = {name: value for name, value in raw.items() if name in accepted}
    values["init_device"] = str(device)
    return upstream.ModelConfig(**values)


def _fixture_config(config: Any, device: str | torch.device) -> Any:
    if config.layer_norm_eps != 1e-5 or config.rope_freq_constant != 10000.0:
        raise ValueError("Original OLMo's fixed LayerNorm epsilon/RoPE base cannot be changed in the native oracle")
    native = native_olmo_reference_config(device=device)
    native.d_model = config.model_dim
    native.n_heads = config.num_heads
    native.n_layers = config.num_layers
    native.mlp_hidden_size = 2 * config.mlp_intermediate_size
    native.embedding_size = config.vocab_size
    native.vocab_size = getattr(config, "tokenizer_vocab_size", config.vocab_size)
    native.max_sequence_length = config.max_context_length
    native.eos_token_id = config.eos_token_id
    native.pad_token_id = config.pad_token_id
    return native


def build_olmo_reference(
    config: Any | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    init_params: bool = True,
) -> nn.Module:
    """Build native OLMo with its original 65-key parameter layout at 1B size.

    A small adapter-shaped config supplies fixture geometry only; native
    configuration determines all architectural flags. Parameters remain FP32;
    use autocast for bounded BF16 comparisons. ``device='meta'`` is supported
    for strict checkpoint materialization with ``load_state_dict(assign=True)``.
    The original constructor toggles global SDPA flags. Restore those flags
    after construction so callers, rather than an imported reference, select
    the backend for each numerical comparison.
    """
    if dtype != torch.float32:
        raise ValueError("The native oracle uses FP32 parameters; use autocast for BF16 execution")
    upstream = _upstream_module()
    native_config = native_olmo_reference_config(device=device) if config is None else _fixture_config(config, device)
    flags = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
    try:
        model = upstream.Olmo(native_config, init_params=init_params)
    finally:
        torch.backends.cuda.enable_flash_sdp(flags[0])
        torch.backends.cuda.enable_mem_efficient_sdp(flags[1])
    model.reference_revision = OLMO_REFERENCE_REVISION
    return model


@dataclass
class OLMoReferenceOutput:
    logits: Tensor
    last_hidden_state: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None


def olmo_reference_forward(
    model: nn.Module,
    input_ids: Tensor | None = None,
    *,
    inputs_embeds: Tensor | None = None,
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
    use_cache: bool = False,
) -> OLMoReferenceOutput:
    """Call the unchanged original forward, exposing final normalized states.

    Original OLMo supports causal prefill and cached chunks. It does not expose
    inputs_embeds: that extension replaces only the embedding module's output
    with the supplied tensor via a temporary hook. The remainder of its actual
    forward executes unchanged, including the tied readout. This source-parity
    wrapper intentionally covers unpadded, default-position sequences; explicit
    mask/position extensions are checked against independent unpadded examples.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Provide exactly one of input_ids and inputs_embeds")
    handles = []
    captured = []
    if inputs_embeds is not None:
        if inputs_embeds.ndim != 3 or min(inputs_embeds.shape) == 0 or inputs_embeds.shape[-1] != model.config.d_model or not inputs_embeds.is_floating_point():
            raise ValueError("inputs_embeds must have shape [batch, length, model_dim]")
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        handles.append(model.transformer.wte.register_forward_hook(lambda module, args, output: inputs_embeds))
    assert input_ids is not None
    if input_ids.ndim != 2 or min(input_ids.shape) == 0:
        raise ValueError("input_ids must be a nonempty [batch, length] tensor")
    handles.append(model.transformer.ln_f.register_forward_hook(lambda module, args, output: captured.append(output)))
    try:
        output = model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one native final LayerNorm invocation")
    cache = tuple(output.attn_key_values) if output.attn_key_values is not None else None
    return OLMoReferenceOutput(output.logits, captured[0], cache)
