"""Validated fixed-layout finite FBT/native forwarding for CUDA graph bodies.

Construction and validation are deliberately outside capture. The tensor-only
entry reuses the native layer loop, fusion modules, and shared embedding lookup.
It owns no parameters and accepts in-place optimizer updates; layout/ownership
changes require a new preparation and capture. NextLat loss layout is separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch
from torch import Tensor

from .nextlat import NextLatBatch, _validate_batch
from .olmo_fbt import FBTMode, FBTOutput, OLMoFBT
from .olmo_rope import build_rope_tables
from .olmo_tiled import OLMoTiledRTForCausalLM
from .recurrent import RTMode


@dataclass
class PreparedFBTOutput(FBTOutput):
    """The exact shared lookup is also available for NextLat conditioning."""
    embeddings: Tensor


def _cpu(value):
    return value.detach().to(device="cpu", copy=True)


def _digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


class PreparedFBTLayout:
    """One independent document per row with fixed validity and RoPE positions.

    ``forward`` is the capture body, not an alternative validation boundary.
    Call ``validate_batch`` and ``validate_execution`` outside capture before
    copying fresh tokens and replaying. Token values and in-place parameter
    updates may change. Validity/positions/adjacency, tensor ownership, model
    configs and checkpoint settings must remain fixed. Document IDs may be
    renumbered: only validated within-row equality affects this model's math.

    This model layout does not freeze CE/latent/KL selection masks. The separate
    prepared loss layout must validate those, along with target indices/counts.
    No online/prefix cache is supported here. Padded masks remain explicit;
    their backend compatibility requires separate GPU evidence.
    """
    def __init__(self, core: OLMoFBT, batch: NextLatBatch, position_ids: Tensor | None = None):
        if not isinstance(core, OLMoFBT) or not isinstance(core.backbone, OLMoTiledRTForCausalLM):
            raise TypeError("Prepared finite forwarding requires OLMoFBT over native OLMoTiledRTForCausalLM")
        self.core = core
        self.base = core.backbone
        _validate_batch(batch, one_document_per_row=True)
        self._shape = tuple(batch.input_ids.shape)
        self._device = core.readout_weight.device
        if batch.input_ids.device != self._device:
            raise ValueError("Prepare batch on the native model device")
        cpu_ids = _cpu(batch.input_ids)
        self._check_tokens(cpu_ids)
        self._valid_cpu = _cpu(batch.valid_mask)
        docs = _cpu(batch.document_ids)
        self._eligible_cpu = self._valid_cpu[:, 1:] & self._valid_cpu[:, :-1] & (docs[:, 1:] == docs[:, :-1])
        self._explicit_positions = position_ids is not None
        if position_ids is None:
            self._positions_cpu = (self._valid_cpu.long().cumsum(-1)-1).clamp_min(0)
        else:
            self._positions_cpu = self._validate_positions(position_ids)
        self.valid_mask = self._valid_cpu.to(self._device, copy=True)
        self.position_ids = self._positions_cpu.to(self._device, copy=True)
        self.feedback_eligible = self._eligible_cpu.to(self._device, copy=True)
        self.all_tokens_valid = bool(self._valid_cpu.all())
        self.is_causal = self.all_tokens_valid
        self._static_flags = (self.all_tokens_valid, self.is_causal)
        if self.all_tokens_valid:
            self.attention_mask = None
        else:
            indices = torch.arange(self._shape[1], device=self._device)
            self.attention_mask = (indices[None] <= indices[:, None])[None, None] & self.valid_mask[:, None, None, :]
        self.rope_tables = (build_rope_tables(self.position_ids, self.base.config.head_dim,
            self.base.config.rope_freq_constant) if self.base.reuse_rope else None)
        self._rope_owner = id(self.rope_tables)
        self._owned = self._owned_tensors()
        self._owned_versions = self._owned_signature()
        # Preserve the old table allocations if someone replaces ``.data``.
        self._owned_storage_references = tuple(value.detach() for value in self._owned if value is not None)
        self._structure = self._structure_signature()
        # Hold old storages without copying bytes, so a child dtype roundtrip
        # cannot release/reuse a pointer and masquerade as unchanged ownership.
        self._storage_references = tuple(value.detach() for value in (*core.parameters(), *core.buffers()))
        self._parameter_signatures = self._parameter_signature()
        self._buffer_signatures = self._buffer_signature()

    def _check_tokens(self, ids):
        if bool(((ids < 0) | (ids >= self.core.config.vocab_size)).any()):
            raise ValueError("Every input token, including padding storage, must be inside the native vocabulary")

    def _validate_positions(self, positions, *, allow_cpu=False):
        if (not isinstance(positions, Tensor) or positions.shape != self._shape
                or positions.dtype != torch.long
                or (positions.device != self._device and not (allow_cpu and positions.device.type == "cpu"))):
            raise ValueError("Static positions must be int64 with the prepared shape/device")
        result = _cpu(positions)
        if bool((result < 0).any()):
            raise ValueError("Static positions must be nonnegative")
        return result

    def _structure_signature(self):
        return (id(self.core), id(self.core.backbone), self.core.config, self.core.fusion_config,
            self.core.fusion.norm_eps, self.base.attention_backend, self.base.attention_precision,
            self.base.ordinary_activation_checkpointing, self.base.cast_weights_once,
            self.base.tile_backend, self.base.backward_tile_backend, self.base.backward_memory,
            self.base.reuse_rope, self.base.kv_only_writes,
            tuple((name, id(module), type(module), module.training) for name, module in self.core.named_modules()))

    def _owned_tensors(self):
        tables = (None, None) if self.rope_tables is None else (self.rope_tables.cos, self.rope_tables.sin)
        return (self.valid_mask, self.position_ids, self.feedback_eligible, self.attention_mask, *tables)

    def _owned_signature(self):
        return tuple(None if value is None else (id(value), value.data_ptr(), value._version,
            tuple(value.shape), tuple(value.stride()), value.device, value.dtype, value.requires_grad)
            for value in self._owned_tensors())

    def _parameter_signature(self):
        return tuple((name, id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad) for name, value in self.core.named_parameters())

    def _buffer_signature(self):
        return tuple((name, id(value), value.data_ptr(), value._version, tuple(value.shape),
            value.dtype, value.device) for name, value in self.core.named_buffers())

    @property
    def metadata(self):
        return {"schema": "olmo-static-fbt-layout-v1", "batch_size": self._shape[0], "length": self._shape[1],
            "valid_tokens": int(self._valid_cpu.sum()), "documents": int(self._valid_cpu.any(-1).sum()),
            "all_tokens_valid": self.all_tokens_valid, "all_valid_causal_lowering_proved": self.all_tokens_valid,
            "attention_representation": "causal_without_mask" if self.all_tokens_valid else "explicit_boolean_causal_padding_mask",
            "explicit_positions": self._explicit_positions, "valid_mask_sha256": _digest(self._valid_cpu),
            "position_ids_sha256": _digest(self._positions_cpu), "feedback_eligible_sha256": _digest(self._eligible_cpu),
            "document_contract": "one independent document per row; no cache or packed documents",
            "loss_masks_owned_here": False, "reuse_rope": self.base.reuse_rope,
            "rope_tables_owned_here": self.rope_tables is not None,
            "rope_table_bytes": 0 if self.rope_tables is None else sum(
                tensor.numel() * tensor.element_size() for tensor in (self.rope_tables.cos, self.rope_tables.sin))}

    def validate_batch(self, batch: NextLatBatch, position_ids: Tensor | None = None):
        """Run on fresh batches before copying tokens/replay, never inside capture."""
        _validate_batch(batch, one_document_per_row=True)
        if (tuple(batch.input_ids.shape) != self._shape
                or (batch.input_ids.device != self._device and batch.input_ids.device.type != "cpu")):
            raise ValueError("Fresh batch shape/device differs from the prepared layout")
        self._check_tokens(_cpu(batch.input_ids))
        valid, docs = _cpu(batch.valid_mask), _cpu(batch.document_ids)
        eligible = valid[:, 1:] & valid[:, :-1] & (docs[:, 1:] == docs[:, :-1])
        if not torch.equal(valid, self._valid_cpu) or not torch.equal(eligible, self._eligible_cpu):
            raise ValueError("Fresh validity/document adjacency differs from the prepared layout")
        positions = self._positions_cpu if position_ids is None else self._validate_positions(position_ids, allow_cpu=True)
        if not torch.equal(positions, self._positions_cpu):
            raise ValueError("Fresh positions differ from the prepared layout")

    def validate_execution(self, mode: FBTMode = FBTMode(), *, expected_signature=None):
        """Verify ownership/layout/settings externally; optimizer value updates are allowed."""
        if not isinstance(mode, FBTMode):
            raise TypeError("Static mode must be an FBTMode")
        self.core._validate_rt_mode(mode.rt_mode)
        if ((self.all_tokens_valid, self.is_causal) != self._static_flags
                or id(self.rope_tables) != self._rope_owner
                or self._owned_signature() != self._owned_versions):
            raise ValueError("Prepared layout tensors changed")
        if self._structure_signature() != self._structure or self._parameter_signature() != self._parameter_signatures:
            raise ValueError("Prepared model ownership/configuration/execution flags changed")
        if self._buffer_signature() != self._buffer_signatures:
            raise ValueError("Prepared fixed model buffers changed")
        if self.core.readout_weight is not self.core.token_embeddings.weight:
            raise ValueError("Native tied embedding/readout ownership changed")
        signature = {"mode": asdict(mode), "ordinary_activation_checkpointing": self.base.ordinary_activation_checkpointing,
            "cast_weights_once": self.base.cast_weights_once, "tile_backend": self.base.tile_backend,
            "backward_tile_backend": self.base.backward_tile_backend, "backward_memory": self.base.backward_memory,
            "reuse_rope": self.base.reuse_rope, "kv_only_writes": self.base.kv_only_writes,
            "training": self.core.training, "attention_backend": self.base.attention_backend,
            "attention_precision": self.base.attention_precision, "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "autocast_enabled": torch.is_autocast_enabled(self._device.type),
            "autocast_dtype": str(torch.get_autocast_dtype(self._device.type)),
            "autocast_cache_enabled": torch.is_autocast_cache_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled()}
        if expected_signature is not None and signature != expected_signature:
            raise ValueError("Prepared mode/runtime context differs from capture")
        return signature

    def _stack(self, embeddings, mode):
        return self.base._forward_prepared(embeddings, mode=mode,
            positions=self.position_ids, key_positions=self.position_ids, key_valid=self.valid_mask,
            attention_mask=self.attention_mask, is_causal=self.is_causal,
            query_rope=self.rope_tables, key_rope=self.rope_tables,
            checkpoint_ordinary=self.base.ordinary_activation_checkpointing and self.base.training and torch.is_grad_enabled())[0]

    def forward(self, input_ids: Tensor, mode: FBTMode = FBTMode()) -> PreparedFBTOutput:
        """Tensor-only finite pass body; external validation is required before replay."""
        if input_ids.shape != self._shape or input_ids.dtype != torch.long or input_ids.device != self._device:
            raise ValueError("Static token buffer shape/dtype/device differs")
        if not isinstance(mode, FBTMode):
            raise TypeError("Static mode must be an FBTMode")
        embeddings = self.core.token_embeddings(input_ids)
        hidden = self._stack(embeddings, RTMode(()) if mode.enabled else mode.rt_mode)
        states = [hidden]
        if mode.enabled:
            for _ in range(1, mode.num_passes):
                suffix = self.core._blend(hidden[:, :-1], embeddings[:, 1:], mode.beta, self.feedback_eligible)
                fused_inputs = torch.cat((embeddings[:, :1], suffix), dim=1)
                hidden = self._stack(fused_inputs, mode.rt_mode)
                states.append(hidden)
        return PreparedFBTOutput(None, hidden, tuple(states), embeddings)
