"""NextLat language-model auxiliary training on a checkpoint-preserving backbone.

Predictor authority: JaydenTeoh/NextLat b37d3411ab9b17be8638abbddb9529f0f3a0a5f9,
``models/model_nextlat.py`` and the released 1B LM configuration. The auxiliary
predictor is never used to roll out a sequence. Backbone states already include
the native final norm. CE trains the ordinary tied readout; auxiliary KL treats
the same matrix as a constant, without detaching its embedding lookup path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


NEXTLAT_REVISION = "b37d3411ab9b17be8638abbddb9529f0f3a0a5f9"
_TERMS = ("ce", "latent", "kl")


@dataclass(frozen=True)
class NextLatConfig:
    model_dim: int
    # The released 1B LM recipe uses 1.6; the upstream dataclass default is 1.0
    # and the historical A5 experiments use a different, smaller predictor.
    proj_factor: float = 1.6
    bias: bool = False
    dropout: float = 0.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    lambda_latent: float = 1.0
    lambda_kl: float = 1.0
    seed: int = 20260921
    # Number of selected positions per projection; every chunk still uses the
    # complete vocabulary. Checkpointing bounds retained vocabulary activations.
    vocab_chunk_size: int = 32

    def __post_init__(self) -> None:
        if type(self.model_dim) is not int or self.model_dim <= 0:
            raise ValueError("model_dim must be a positive integer")
        if type(self.vocab_chunk_size) is not int or self.vocab_chunk_size <= 0:
            raise ValueError("vocab_chunk_size must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.bias) is not bool:
            raise TypeError("bias must be boolean")
        for name in ("proj_factor", "norm_eps", "init_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hidden_dim <= 0:
            raise ValueError("Source predictor rounding to a multiple of 128 yields zero hidden width")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("lambda_latent", "lambda_kl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def hidden_dim(self) -> int:
        return 128 * round((2 * self.model_dim * self.proj_factor) / 128)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> NextLatConfig:
        return cls(**values)


class _PredictorNorm(nn.Module):
    """Pinned source's LayerNorm name denotes RMSNorm when bias is false."""

    def __init__(self, width: int, config: NextLatConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(width, device="cpu", dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(width, device="cpu", dtype=torch.float32)) if config.bias else None

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class NextLatPredictor(nn.Module):
    """Three dense layers, GELU, concatenated [next embedding; state], residual.

    Initialization uses a separate CPU generator scope. Construction changes
    neither the loaded backbone nor the caller's CPU or CUDA RNG stream. Move
    the new module with ``.to(...)`` after construction when needed.
    """

    def __init__(self, config: NextLatConfig):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.hidden_state_dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
            dimensions = (2 * config.model_dim, config.hidden_dim, config.hidden_dim, config.model_dim)
            self.mlp = nn.Sequential(
                nn.Linear(dimensions[0], dimensions[1], bias=config.bias, device="cpu", dtype=torch.float32),
                nn.GELU(),
                nn.Linear(dimensions[1], dimensions[2], bias=config.bias, device="cpu", dtype=torch.float32),
                nn.GELU(),
                nn.Linear(dimensions[2], dimensions[3], bias=config.bias, device="cpu", dtype=torch.float32),
            )
            self.norm_x = _PredictorNorm(dimensions[0], config)
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=config.init_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, current_states: Tensor, next_token_embeds: Tensor) -> Tensor:
        if current_states.shape != next_token_embeds.shape or current_states.shape[-1] != self.config.model_dim:
            raise ValueError("Predictor state/embedding shapes must match and end in model_dim")
        x = torch.cat((next_token_embeds, self.hidden_state_dropout(current_states)), dim=-1)
        return current_states + self.mlp(self.norm_x(x))


@dataclass(frozen=True)
class NextLatBatch:
    """Token masks are specified at their target position, independently.

    ``ce_mask[:,s]`` selects prediction of token s from state s-1;
    ``latent_mask[:,s]`` selects prediction of state s from state s-1 and e_s;
    ``kl_mask[:,s]`` selects prediction of token s from the predicted state s-1.
    None selects all valid same-document pairs/triples for that objective.
    document_ids must be nonnegative at valid positions; padding ids are ignored.
    No implicit EOS or prompt/response policy is inferred from token IDs.
    """

    input_ids: Tensor
    valid_mask: Tensor
    document_ids: Tensor
    ce_mask: Tensor | None = None
    latent_mask: Tensor | None = None
    kl_mask: Tensor | None = None

    def to(self, device: torch.device | str) -> NextLatBatch:
        return NextLatBatch(**{name: None if value is None else value.to(device)
                             for name, value in self.__dict__.items()})


def _validate_batch(batch: NextLatBatch, *, one_document_per_row: bool = False) -> None:
    if not isinstance(batch, NextLatBatch):
        raise TypeError("batch must be NextLatBatch")
    ids = batch.input_ids
    if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
        raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
    for name in ("valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"):
        value = getattr(batch, name)
        if value is None and name not in ("valid_mask", "document_ids"):
            continue
        dtype = torch.long if name == "document_ids" else torch.bool
        if value is None or value.shape != ids.shape or value.dtype != dtype or value.device != ids.device:
            raise ValueError(f"{name} must be {dtype} with input_ids shape/device")
    if bool((batch.document_ids[batch.valid_mask] < 0).any()):
        raise ValueError("Valid tokens must have nonnegative document_ids")
    if one_document_per_row:
        for row_ids, row_valid in zip(batch.document_ids, batch.valid_mask):
            if row_ids[row_valid].unique().numel() > 1:
                raise ValueError("Packed documents are unsupported: each batch row must contain only one document")


def build_nextlat_masks(batch: NextLatBatch) -> dict[str, Tensor]:
    """Loss-only boundary masks; they do not provide attention isolation."""
    _validate_batch(batch)
    valid, docs = batch.valid_mask, batch.document_ids
    pair = valid[:, :-1] & valid[:, 1:] & (docs[:, :-1] == docs[:, 1:])
    triple = pair[:, :-1] & pair[:, 1:]
    return {
        "ce": pair if batch.ce_mask is None else pair & batch.ce_mask[:, 1:],
        "latent": pair if batch.latent_mask is None else pair & batch.latent_mask[:, 1:],
        "kl": triple if batch.kl_mask is None else triple & batch.kl_mask[:, 2:],
    }


def _objective_weights(config: NextLatConfig, enabled: bool) -> dict[str, float]:
    return {"ce": 1.0, "latent": config.lambda_latent if enabled else 0.0,
            "kl": config.lambda_kl if enabled else 0.0}


@dataclass
class NextLatLosses:
    sums: dict[str, Tensor]
    counts: dict[str, int]
    weights: dict[str, float]

    @property
    def means(self) -> dict[str, Tensor]:
        return {name: self.sums[name] / max(self.counts[name], 1) for name in _TERMS}

    @property
    def total(self) -> Tensor:
        means = self.means
        return sum(self.weights[name] * means[name] for name in _TERMS)


def _ce_chunk(states: Tensor, weight: Tensor, targets: Tensor) -> Tensor:
    # Full vocabulary, including native checkpoint's extra output rows. Loss
    # arithmetic is FP32 even if projection uses the caller's autocast policy.
    return F.cross_entropy(F.linear(states, weight).float(), targets, reduction="sum")


def _kl_chunk(predicted: Tensor, teacher: Tensor, readout: Tensor) -> Tensor:
    # Callers also detach these arguments before checkpointing. The explicit
    # no_grad scope prevents retaining a teacher vocabulary graph.
    with torch.no_grad():
        log_teacher = F.log_softmax(F.linear(teacher, readout).float(), dim=-1)
    log_student = F.log_softmax(F.linear(predicted, readout).float(), dim=-1)
    return F.kl_div(log_student, log_teacher, log_target=True, reduction="none").sum()


def _chunked_sum(function, states: Tensor, other: Tensor, third: Tensor,
                 chunk_size: int, *, weight_second: bool) -> Tensor:
    result = states.sum() * 0.0
    for start in range(0, states.shape[0], chunk_size):
        end = start + chunk_size
        args = (states[start:end], other, third[start:end]) if weight_second else (
            states[start:end], other[start:end], third)
        # Non-reentrant checkpointing saves only inputs, not the [positions,V]
        # logits/logprob buffers of every chunk until backward. It also supports
        # autograd.grad and requires no mutation of leaf parameter gradients.
        if torch.is_grad_enabled() and any(arg.requires_grad for arg in args):
            loss = checkpoint(function, *args, use_reentrant=False)
        else:
            loss = function(*args)
        result = result + loss
    return result


def compute_nextlat_loss_sums(hidden_states: Tensor, token_embeddings: Tensor,
                             readout_weight: Tensor, batch: NextLatBatch,
                             predictor: NextLatPredictor | None,
                             config: NextLatConfig, *, enabled: bool = True) -> NextLatLosses:
    """Raw objective sums for globally normalized microbatch accumulation.

    ``latent`` sums the coordinate-mean SmoothL1(beta=1) at each selected pair;
    its count is the number of pairs. CE/KL counts are selected target tokens.
    Cross-document masks here do NOT make an already computed hidden state
    document-isolated; the integrated wrapper rejects packed-document rows.
    """
    masks = build_nextlat_masks(batch)
    shape = (*batch.input_ids.shape, config.model_dim)
    if hidden_states.shape != shape or token_embeddings.shape != shape:
        raise ValueError("Hidden states and embeddings must have [batch, sequence, model_dim] shape")
    if hidden_states.device != batch.input_ids.device or token_embeddings.device != hidden_states.device:
        raise ValueError("Hidden states, embeddings, and batch must share device")
    if readout_weight.ndim != 2 or readout_weight.shape[1] != config.model_dim or readout_weight.device != hidden_states.device:
        raise ValueError("readout_weight must have [vocabulary, model_dim] shape on the hidden-state device")
    weights = _objective_weights(config, enabled)
    counts = {name: int(mask.sum().item()) if weights[name] else 0 for name, mask in masks.items()}
    zero = hidden_states.sum() * 0.0
    sums = {name: zero for name in _TERMS}
    ce = masks["ce"]
    if counts["ce"]:
        sums["ce"] = _chunked_sum(_ce_chunk, hidden_states[:, :-1][ce], readout_weight,
                                  batch.input_ids[:, 1:][ce], config.vocab_chunk_size,
                                  weight_second=True)
    if counts["latent"] or counts["kl"]:
        if predictor is None:
            raise ValueError("An enabled auxiliary objective requires a predictor")
        latent = masks["latent"] if weights["latent"] else torch.zeros_like(masks["latent"])
        kl = masks["kl"] if weights["kl"] else torch.zeros_like(masks["kl"])
        kl_pair = F.pad(kl, (0, 1))
        needed = latent | kl_pair
        predicted = predictor(hidden_states[:, :-1][needed], token_embeddings[:, 1:][needed])
        if counts["latent"]:
            chosen = latent[needed]
            targets = hidden_states[:, 1:][latent].detach()
            sums["latent"] = F.smooth_l1_loss(predicted[chosen].float(), targets.float(), reduction="none").mean(-1).sum()
        if counts["kl"]:
            sums["kl"] = _chunked_sum(_kl_chunk, predicted[kl_pair[needed]],
                                      hidden_states[:, 1:-1][kl].detach(), readout_weight.detach(),
                                      config.vocab_chunk_size, weight_second=False)
    return NextLatLosses(sums, counts, weights)


class NextLatLM(nn.Module):
    """Own one native backbone and optional predictor; no extra readout alias.

    Runtime RT mode belongs to ``backbone_kwargs`` and is independent of the
    constructor's NextLat enabled flag. Full documents only: cache/chunked state
    and packed documents need a separate attention-isolation design.
    """

    def __init__(self, backbone: nn.Module, config: NextLatConfig, *, enabled: bool = True):
        super().__init__()
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if backbone.config.model_dim != config.model_dim:
            raise ValueError("Predictor width must match backbone model_dim")
        self.backbone = backbone
        self.config = config
        self._enabled = enabled
        self.predictor = NextLatPredictor(config) if enabled and (config.lambda_latent or config.lambda_kl) else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def objective_weights(self) -> dict[str, float]:
        return _objective_weights(self.config, self.enabled)

    def counts(self, batch: NextLatBatch) -> dict[str, int]:
        _validate_batch(batch, one_document_per_row=True)
        masks, weights = build_nextlat_masks(batch), self.objective_weights()
        return {name: int(masks[name].sum().item()) if weights[name] else 0 for name in _TERMS}

    def loss_sums(self, batch: NextLatBatch, *, backbone_kwargs: Mapping[str, Any] | None = None) -> NextLatLosses:
        _validate_batch(batch, one_document_per_row=True)
        kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        forbidden = {"input_ids", "inputs_embeds", "attention_mask", "past_key_values", "use_cache", "return_logits"} & kwargs.keys()
        if forbidden:
            raise ValueError(f"NextLat owns full-document input/masking and disables caches/logits: {sorted(forbidden)}")
        # One lookup serves both branches: gradients from conditioning and the
        # backbone accumulate into the one native tied matrix in the usual way.
        embeddings = self.backbone.token_embeddings(batch.input_ids)
        output = self.backbone(inputs_embeds=embeddings, attention_mask=batch.valid_mask,
                               return_logits=False, **kwargs)
        return compute_nextlat_loss_sums(output.last_hidden_state, embeddings,
                                         self.backbone.readout_weight, batch, self.predictor,
                                         self.config, enabled=self.enabled)

    def forward(self, batch: NextLatBatch, *, backbone_kwargs: Mapping[str, Any] | None = None) -> NextLatLosses:
        return self.loss_sums(batch, backbone_kwargs=backbone_kwargs)
