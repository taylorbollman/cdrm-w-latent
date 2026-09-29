"""Pass-weighted FBT language-model objectives with independent NextLat switch.

The historical default preserves L0 + gamma * mean(extra-pass losses). The
opt-in campaign_v1 policy normalizes CE across passes with half its mass on the
first pass and averages auxiliary terms uniformly over all passes. Each term
retains its existing valid-position denominator and NextLat weight. One shared
predictor processes every pass. It is never a recurrent rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .nextlat import (NextLatBatch, NextLatConfig, NextLatLM, NextLatLosses,
                      _validate_batch, compute_nextlat_loss_sums)
from .document_policy import ISOLATED_DOCUMENTS
from .olmo_fbt import FBTMode


@dataclass
class FBTNextLatLosses(NextLatLosses):
    """Combined raw sums and unweighted per-pass observations.

    ``counts`` count positions in one pass, not passes times positions. This
    lets the existing optimizer platform normalize unequal microbatches once
    per objective. ``means`` is therefore the pass-weighted objective mean;
    inspect ``pass_losses[p].means`` for a particular pass's CE/NLL.
    """
    pass_losses: tuple[NextLatLosses, ...]
    # Historical public field: the CE coefficients. Use term_pass_coefficients
    # when inspecting a policy with distinct CE and auxiliary pass weights.
    pass_coefficients: tuple[float, ...]
    term_pass_coefficients: dict[str, tuple[float, ...]]
    pass_loss_policy: str


def _validate_pass_loss_policy(pass_loss_policy: str, gamma: float) -> None:
    if pass_loss_policy not in ("legacy", "campaign_v1"):
        raise ValueError("FBT pass_loss_policy must be legacy or campaign_v1")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("FBT extra-pass gamma must be finite and nonnegative")
    if pass_loss_policy != "legacy" and gamma != 1.0:
        raise ValueError("campaign_v1 defines fixed pass weights and requires gamma=1")


def aggregate_pass_losses(pass_losses: Sequence[NextLatLosses], *, gamma: float = 1.0,
                          pass_loss_policy: str = "legacy") -> FBTNextLatLosses:
    """Weight raw sums per term, retaining one-pass position denominators.

    For campaign_v1, CE coefficients are (.5, .5/(K-1), ...) and latent/KL
    coefficients are (1/K, ...). K1 is the unchanged single-pass objective.
    """
    _validate_pass_loss_policy(pass_loss_policy, gamma)
    passes = tuple(pass_losses)
    if not passes or any(not isinstance(loss, NextLatLosses) for loss in passes):
        raise ValueError("FBT aggregation needs at least one NextLatLosses record")
    first = passes[0]
    names = {"ce", "latent", "kl"}
    if set(first.sums) != names or set(first.counts) != names or set(first.weights) != names:
        raise ValueError("FBT objective records must define CE, latent and KL")
    for loss in passes:
        if loss.counts != first.counts or loss.weights != first.weights or set(loss.sums) != names:
            raise ValueError("FBT passes must have identical masks/counts and objective weights")
    count = len(passes)
    if pass_loss_policy == "legacy" or count == 1:
        coefficients = (1.0,) if count == 1 else (1.0,) + (float(gamma)/(count-1),) * (count-1)
        term_coefficients = {name: coefficients for name in first.sums}
    else:
        coefficients = (.5,) + (.5/(count-1),) * (count-1)
        term_coefficients = {"ce": coefficients, "latent": (1.0/count,) * count,
                             "kl": (1.0/count,) * count}
    sums = {name: sum(coefficient * loss.sums[name]
                      for coefficient, loss in zip(term_coefficients[name], passes))
            for name in first.sums}
    return FBTNextLatLosses(sums, dict(first.counts), dict(first.weights), passes,
                           coefficients, term_coefficients, pass_loss_policy)


class FBTNextLatLM(NextLatLM):
    """Wrap OLMoFBT with the unchanged horizon-one NextLat implementation.

    FBT/RT execution is selected by the core's immutable runtime mode passed
    through ``backbone_kwargs``. ``enabled`` independently selects NextLat.
    Preserve pass_loss_policy and gamma alongside the NextLat/core/mode
    configuration in checkpoints; neither is a learned parameter or changes
    valid-position counts. The policy is read-only after construction.
    """
    def __init__(self, backbone, config: NextLatConfig, *, enabled: bool = True,
                 gamma: float = 1.0, pass_loss_policy: str = "legacy"):
        _validate_pass_loss_policy(pass_loss_policy, gamma)
        super().__init__(backbone, config, enabled=enabled)
        self._gamma = float(gamma)
        self._pass_loss_policy = pass_loss_policy

    @property
    def pass_loss_policy(self) -> str:
        return self._pass_loss_policy

    @property
    def gamma(self) -> float:
        return self._gamma

    def loss_sums(self, batch: NextLatBatch, *, backbone_kwargs: Mapping[str, Any] | None = None) -> FBTNextLatLosses:
        _validate_batch(batch, one_document_per_row=self.config.document_policy == ISOLATED_DOCUMENTS)
        kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        mode = kwargs.get("mode", FBTMode())
        if not isinstance(mode, FBTMode):
            raise TypeError("FBT NextLat requires an FBTMode")
        if mode.document_policy != self.config.document_policy:
            raise ValueError("FBT forward and NextLat loss document_policy must agree")
        forbidden = {"input_ids", "inputs_embeds", "attention_mask", "document_ids", "past_key_values",
                     "use_cache", "return_logits"} & kwargs.keys()
        if forbidden:
            raise ValueError(f"FBT NextLat owns full-document input/masks and disables caches/logits: {sorted(forbidden)}")
        embeddings = self.backbone.token_embeddings(batch.input_ids)
        output = self.backbone(inputs_embeds=embeddings, attention_mask=batch.valid_mask,
                               document_ids=batch.document_ids, return_logits=False, **kwargs)
        states = output.pass_hidden_states
        if not isinstance(states, tuple) or not states:
            raise ValueError("FBT core must return a nonempty tuple of per-pass final-normalized states")
        losses = tuple(compute_nextlat_loss_sums(hidden, embeddings, self.backbone.readout_weight,
                       batch, self.predictor, self.config, enabled=self.enabled) for hidden in states)
        return aggregate_pass_losses(losses, gamma=self.gamma, pass_loss_policy=self.pass_loss_policy)
