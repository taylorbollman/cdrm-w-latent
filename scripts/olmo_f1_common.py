"""F1 bounded integration cases; the existing model and optimizer own the math."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTConfig, FBTMode
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import build_adamw, build_warmup_scheduler
from scripts.olmo_lm_common import fixture, tensor_digest, tree_digests


@dataclass(frozen=True)
class IntegrationCase:
    name: str
    fbt: bool = False
    nextlat: bool = False
    rt_layers: tuple[int, ...] = ()
    passes: int = 2
    alpha: float = 1.0
    beta: float = 1.0
    batch_size: int = 2
    length: int = 32
    updates: int = 3
    transition: bool = False
    resume: bool = False
    profile: bool = False

    def __post_init__(self):
        object.__setattr__(self, "rt_layers", tuple(self.rt_layers))
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.name):
            raise ValueError("Case names must be nonempty safe identifiers")
        for key in ("fbt", "nextlat", "transition", "resume", "profile"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("passes", "batch_size", "length", "updates"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.length < 8 or self.length > 2048:
            raise ValueError("F1 fixtures require lengths 8 through native context 2048")
        RTMode(self.rt_layers, self.alpha)
        FBTMode(enabled=self.fbt, num_passes=self.passes, beta=self.beta)
        if self.fbt and self.passes < 2:
            raise ValueError("F1 enabled FBT cells must exercise an extra pass")
        if self.resume and self.updates != 3:
            raise ValueError("F1 resume saves update2 and replays update3")
        if self.transition and (not self.fbt or not self.rt_layers or self.updates != 3):
            raise ValueError("Transition requires RT, FBT and three updates")
        if self.profile and (self.resume or self.transition):
            raise ValueError("Profile cases have fixed modes and no checkpoint work")

    def mode(self, update: int = 0) -> FBTMode:
        if type(update) is not int or update < 0:
            raise ValueError("update must be a nonnegative integer")
        if self.transition:
            alpha, beta = ((0., 0.), (.37, .35), (1., 1.))[min(update, 2)]
        else:
            alpha, beta = self.alpha, self.beta
        return FBTMode(enabled=self.fbt, num_passes=self.passes if self.fbt else 1,
                       beta=beta, rt_mode=RTMode(self.rt_layers, alpha))


def default_cases():
    cases = []
    for fbt in (False, True):
        for rt in (False, True):
            for nextlat in (False, True):
                name = "-".join(x for x, on in (("rt", rt), ("fbt", fbt), ("nextlat", nextlat)) if on) or "ordinary"
                cases.append(IntegrationCase(name, fbt=fbt, nextlat=nextlat,
                    rt_layers=(0,) if rt else (), resume=name in ("ordinary", "rt-fbt-nextlat")))
    all_three = IntegrationCase("all-three", fbt=True, nextlat=True, rt_layers=(0,))
    cases.extend([
        replace(all_three, name="all-three-k3", passes=3),
        replace(all_three, name="all-three-fractional", alpha=.37, beta=.35),
        replace(all_three, name="all-three-two-rt-layers", rt_layers=(0, 15)),
        replace(all_three, name="all-three-transition", transition=True),
        IntegrationCase("ordinary-t128", length=128, updates=2),
        replace(all_three, name="all-three-t128", length=128, updates=2),
    ])
    for name, fbt, layers, nextlat in (
        ("ordinary", False, (), False), ("rt", False, (0,), False),
        ("fbt", True, (), False), ("all-three", True, (0,), True),
    ):
        cases.append(IntegrationCase(name+"-t512-profile", fbt=fbt, rt_layers=layers,
                     nextlat=nextlat, batch_size=1, length=512, updates=1, profile=True))
    return cases


def build_model(state, case, *, device="cuda", model_config=None, chunk_size=128,
                backend="sdpa"):
    config = OLMoConfig.native_1b() if model_config is None else model_config
    # CPU unit fixtures must not mutate the caller's source through assign=True.
    source = {k: v.clone() for k, v in state.items()} if torch.device(device).type == "cpu" else state
    base = OLMoTiledRTForCausalLM(config, attention_backend=backend,
              attention_precision="mixed", device="meta", dtype=torch.float32)
    base.load_state_dict(source, strict=True, assign=True)
    # Construct new branches on CPU, preserving the published initialization.
    core = OLMoFBT(base, FBTConfig())
    if not case.fbt:
        core.fusion.requires_grad_(False)
    model = FBTNextLatLM(core, NextLatConfig(model_dim=config.model_dim,
                        vocab_chunk_size=chunk_size), enabled=case.nextlat).to(device)
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise AssertionError("Native input/readout tying was lost")
    return model.train()


def build_optimizer(model, lr=1e-5):
    optimizer = build_adamw(model, lr=lr, betas=(.9, .95), eps=1e-8, weight_decay=.1)
    return optimizer, build_warmup_scheduler(optimizer, warmup_updates=2)


def changed_fixture(tokenizer, case, update, *, device="cuda"):
    """Deterministic changed real-text tokens; EOS, padding and masks stay fixed.

    These are operational fixtures, not a language-quality dataset. Rotation
    deliberately changes the input each update without consuming global RNG.
    """
    batch = fixture(tokenizer, batch_size=case.batch_size, length=case.length,
                    padded=True, device=device)
    ids = batch.input_ids.clone()
    for row in range(case.batch_size):
        count = int(batch.valid_mask[row].sum())
        ids[row, :count-1] = ids[row, :count-1].roll((update * 3 + row) % (count-1))
    return replace(batch, input_ids=ids)


def split_rows(batch):
    """Unequal valid-count microbatches retain one document per row."""
    return [NextLatBatch(**{key: None if value is None else value[row:row+1]
            for key, value in vars(batch).items()})
            for row in range(batch.input_ids.shape[0])]


def active_names(model, mode):
    """Expected participation at positive default CE/auxiliary/pass weights."""
    fusion_active = mode.enabled and mode.num_passes > 1 and mode.beta > 0
    return {name for name, p in model.named_parameters()
            if p.requires_grad and (not name.startswith("backbone.fusion.") or fusion_active)}


def inference_names(model, case):
    return {name for name, _ in model.named_parameters()
            if not name.startswith("predictor.")
            and (case.fbt or not name.startswith("backbone.fusion."))}


class GradientObserver:
    """Read-only leaf hooks; detached reductions are materialized after the step."""
    def __init__(self, model):
        self.values = {}
        self.handles = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                def observe(gradient, name=name):
                    value = gradient.detach()
                    self.values.setdefault(name, []).append(
                        (torch.isfinite(value).all(), value.float().norm(), value.float().abs().max()))
                self.handles.append(parameter.register_hook(observe))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self, expected):
        tensors = {name: {"finite": all(bool(v[0]) for v in values),
                         "backward_contributions": len(values),
                         "contribution_norm_sum": sum(float(v[1]) for v in values),
                         "max_abs": max(float(v[2]) for v in values)}
                   for name, values in self.values.items()}
        missing, unexpected = sorted(set(expected)-tensors.keys()), sorted(tensors.keys()-set(expected))
        return {"tensors": tensors, "missing": missing, "unexpected": unexpected,
                "passed": not missing and not unexpected and all(v["finite"] for v in tensors.values()),
                "norm_scope": "sum of per-backward contribution norms, not accumulated-gradient norm"}


def state_health(model, optimizer):
    bad_parameters = [n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
    bad_moments = [f"{i}/{k}" for i, values in enumerate(optimizer.state.values())
                   for k, v in values.items() if isinstance(v, torch.Tensor)
                   and not bool(torch.isfinite(v).all())]
    return {"nonfinite_parameters": bad_parameters, "nonfinite_optimizer_tensors": bad_moments,
            "passed": not bad_parameters and not bad_moments}


def boundary_digests(model, optimizer, scheduler, counters):
    return tree_digests({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "counters": asdict(counters)})


def rng_probe(device):
    return {"cpu": tensor_digest(torch.rand(7)),
            "device": tensor_digest(torch.rand(7, device=device))}


def state_change(before, after, active):
    if before.keys() != after.keys():
        raise AssertionError("Model state keys changed")
    changed = sorted(name for name in before if before[name] != after[name])
    unchanged_active = sorted(set(active)-set(changed))
    unexpected = sorted(set(changed)-set(active))
    return {"changed": changed, "unchanged_active": unchanged_active,
            "unexpected_changes": unexpected, "passed": not unchanged_active and not unexpected}
