"""Explicit portable campaign contracts; no training launcher or hardware defaults.

Legacy model/optimizer/scheduler entrypoints are unchanged. The campaign opts
into new pass semantics through this module. Logical data updates, rather than
GPU rank or physical batch shape, determine exposure and feedback noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import torch
from torch import nn

from .fbt_training import FBTNextLatLM
from .lm_training import optimizer_ownership
from .nextlat import NextLatConfig
from .olmo_fbt import FBTConfig, FBTMode, OLMoFBT
from .recurrent import RTMode

ARMS = ("B", "N", "F", "R", "NF", "NR", "FR", "NFR")
CHECKPOINT_REPO = "allenai/OLMo-1B"
CHECKPOINT_REVISION = "81b71efbce6f4dada57c94860301af4298bcd351"


def _positive_int(name, value, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CampaignRecipe:
    arm: str
    sequence_length: int = 1024
    rt_layers: tuple[int, ...] = (0, 15)
    effective_valid_tokens: int = 524288
    plateau_lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_tokens: int = 52428800
    warmup_start_fraction: float = 0.1
    feedback_jitter: float = 0.02
    fusion_seed: int = 20260922
    predictor_seed: int = 20260921
    jitter_seed: int = 20260928

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"Unknown campaign arm {self.arm!r}")
        for name in ("sequence_length", "effective_valid_tokens"):
            _positive_int(name, getattr(self, name))
        for name in ("warmup_tokens", "fusion_seed", "predictor_seed", "jitter_seed"):
            _positive_int(name, getattr(self, name), zero=True)
        layers = tuple(self.rt_layers)
        if not layers or any(type(i) is not int or i < 0 for i in layers) or len(set(layers)) != len(layers):
            raise ValueError("rt_layers must be distinct nonnegative layer indices")
        object.__setattr__(self, "rt_layers", tuple(sorted(layers)))
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2 or any(isinstance(b, bool) or not math.isfinite(b) or not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two finite values in [0,1)")
        for name in ("plateau_lr", "epsilon", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in ("weight_decay", "feedback_jitter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        if (isinstance(self.warmup_start_fraction, bool) or not math.isfinite(self.warmup_start_fraction)
                or not 0 < self.warmup_start_fraction <= 1):
            raise ValueError("warmup_start_fraction must be in (0,1]")

    @property
    def nextlat(self):
        return "N" in self.arm

    @property
    def feedback(self):
        return "F" in self.arm

    def mode(self):
        return FBTMode(enabled=self.feedback, num_passes=4 if self.feedback else 1,
                       beta=1.0, rt_mode=RTMode(self.rt_layers if "R" in self.arm else ()),
                       first_pass_policy="configured-rt-v1",
                       feedback_jitter=self.feedback_jitter if self.feedback else 0.0)

    def to_dict(self):
        # JSON-native values make checkpoint equality independent of a JSON round trip.
        values = json.loads(json.dumps(asdict(self)))
        return {"schema": "olmo-campaign-recipe-v1", **values,
                "source": {"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION},
                "pass_loss_policy": "campaign_v1", "fbt_passes": 4 if self.feedback else 1,
                "precision": "bf16_mixed_fp32_master", "optimizer_state": "fresh",
                "auxiliary": {"horizon": 1, "latent": 1.0, "kl": 1.0, "token_ce": 0.0},
                "decay_policy": "matrix_except_embedding_norm_bias_v1",
                "first_pass_policy": "configured-rt-v1"}

    @property
    def sha256(self):
        return _digest(self.to_dict())


def build_campaign_model(backbone, recipe: CampaignRecipe):
    """Wrap already loaded weights; never reset or convert the backbone.

    Tiny fixtures can supply a smaller sequence_length/rt_layers explicitly.
    Loading the pinned actual weights remains the caller's responsibility.
    """
    if recipe.sequence_length > backbone.config.max_context_length:
        raise ValueError("Campaign length exceeds backbone context capacity")
    if "R" in recipe.arm and max(recipe.rt_layers) >= backbone.config.num_layers:
        raise ValueError("Campaign RT selection exceeds backbone depth")
    if any(p.dtype != torch.float32 for p in backbone.parameters()):
        raise ValueError("Campaign requires FP32 master parameters")
    core = OLMoFBT(backbone, FBTConfig(seed=recipe.fusion_seed))
    if not recipe.feedback:
        # A dormant wrapper module is resident but not trainable architecture.
        core.fusion.requires_grad_(False)
    return FBTNextLatLM(core, NextLatConfig(backbone.config.model_dim,
                        seed=recipe.predictor_seed, vocab_chunk_size=128,
                        ce_chunk_size=2048), enabled=recipe.nextlat,
                        pass_loss_policy="campaign_v1")


def build_campaign_adamw(model: FBTNextLatLM, recipe: CampaignRecipe, *, fused: bool):
    """Unique tied ownership, component telemetry, and explicit decay exclusions."""
    if type(fused) is not bool:
        raise TypeError("Choose fused explicitly for the execution environment")
    embeddings = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = ("predictor" if name.startswith("predictor.") else
                     "fusion" if name.startswith("backbone.fusion.") else "backbone")
        decay = parameter.ndim >= 2 and id(parameter) not in embeddings
        key = (component, decay)
        if key not in groups:
            groups[key] = {"params": [], "param_names": [], "component": component,
                           "weight_decay": recipe.weight_decay if decay else 0.0,
                           "decay_policy": "matrix" if decay else "excluded", "lr_multiplier": 1.0}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    if not groups:
        raise ValueError("Campaign optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(list(groups.values()), lr=recipe.plateau_lr,
                                  betas=recipe.betas, eps=recipe.epsilon,
                                  weight_decay=recipe.weight_decay, fused=fused, foreach=False)
    optimizer_ownership(model, optimizer)
    return optimizer


class CampaignTokenSchedule(torch.optim.lr_scheduler.LRScheduler):
    """Warmup/plateau at deterministic whole-update valid-token boundaries.

    update_tokens is the shared logical-update plan, not physical microbatches.
    The first update starts at the requested floor. After exposure reaches the
    warmup budget, the next update uses plateau LR. Normal trainer .step() calls
    therefore advance exposure correctly without modifying its optimizer API.
    """
    def __init__(self, optimizer, update_tokens: Sequence[int], *, warmup_tokens: int,
                 start_fraction: float = 0.1):
        _positive_int("warmup_tokens", warmup_tokens, zero=True)
        if isinstance(start_fraction, bool) or not math.isfinite(start_fraction) or not 0 < start_fraction <= 1:
            raise ValueError("start_fraction must be in (0,1]")
        prefix = [0]
        for count in update_tokens:
            _positive_int("update token count", count)
            prefix.append(prefix[-1] + count)
        if len(prefix) == 1:
            raise ValueError("Schedule needs at least one logical update")
        self.token_prefix = tuple(prefix)
        self.warmup_tokens = warmup_tokens
        self.start_fraction = float(start_fraction)
        self.plan_sha256 = _digest({"tokens": list(update_tokens), "warmup_tokens": warmup_tokens,
                                    "start_fraction": self.start_fraction})
        super().__init__(optimizer)

    @property
    def completed_tokens(self):
        return self.token_prefix[min(max(self.last_epoch, 0), len(self.token_prefix) - 1)]

    def get_lr(self):
        progress = min(1.0, self.completed_tokens / self.warmup_tokens) if self.warmup_tokens else 1.0
        factor = self.start_fraction + (1.0 - self.start_fraction) * progress
        return [base * factor for base in self.base_lrs]

    def load_state_dict(self, state_dict):
        for key in ("token_prefix", "warmup_tokens", "start_fraction", "plan_sha256", "base_lrs"):
            if state_dict.get(key) != getattr(self, key):
                raise ValueError(f"Campaign token schedule {key} differs from requested resume")
        if type(state_dict.get("last_epoch")) is not int or not 0 <= state_dict["last_epoch"] < len(self.token_prefix):
            raise ValueError("Invalid completed schedule update count")
        super().load_state_dict(state_dict)


def feedback_noise_for_rows(recipe: CampaignRecipe, window_keys: Sequence[str], *,
                            logical_update: int, sequence_length: int, width: int,
                            device="cpu", dtype=torch.float32):
    """Partition-invariant CPU-generated unit jitter; no global RNG consumption.

    Window keys identify row *occurrences* in the logical update (include a
    repetition identifier if necessary). Each row/pass has its own seed. Prefix
    noise is stable across padding lengths; GPU count/rank is deliberately absent.
    CPU staging is a correctness reference, not a production throughput claim.
    """
    _positive_int("logical_update", logical_update, zero=True)
    _positive_int("sequence_length", sequence_length)
    _positive_int("width", width)
    if not window_keys or any(not isinstance(k, str) or not k for k in window_keys):
        raise ValueError("Nonempty stable window keys are required")
    if len(set(window_keys)) != len(window_keys):
        raise ValueError("Window occurrence keys must be unique within the physical batch")
    if not dtype.is_floating_point:
        raise ValueError("Feedback noise needs a floating dtype")
    if not recipe.feedback or recipe.feedback_jitter == 0:
        return None
    noise = []
    for feedback_pass in range(1, 4):
        rows = []
        for key in window_keys:
            seed = int(_digest(["campaign-jitter-v1", recipe.jitter_seed, logical_update,
                                feedback_pass, key])[:16], 16) % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            rows.append(torch.rand((sequence_length - 1, width), generator=generator).mul_(2).sub_(1))
        noise.append(torch.stack(rows).to(device=device, dtype=dtype))
    return tuple(noise)
