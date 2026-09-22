"""Pass-weighted FBT language-model objectives with independent NextLat switch.

The FBT source uses L0 + mean(extra-pass losses), rather than a mean over all
passes. We preserve that policy separately for CE, latent regression and KL;
each term retains its existing valid-position denominator and NextLat weight.
One shared predictor processes every pass. It is never a recurrent rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .nextlat import (NextLatBatch, NextLatConfig, NextLatLM, NextLatLosses,
                      _validate_batch, compute_nextlat_loss_sums)


@dataclass
class FBTNextLatLosses(NextLatLosses):
    """Combined raw sums and unweighted per-pass observations.

    ``counts`` count positions in one pass, not passes times positions. This
    lets the existing optimizer platform normalize unequal microbatches once
    per objective. ``means`` is therefore the pass-weighted objective mean;
    inspect ``pass_losses[p].means`` for a particular pass's CE/NLL.
    """
    pass_losses: tuple[NextLatLosses, ...]
    pass_coefficients: tuple[float, ...]


def aggregate_pass_losses(pass_losses: Sequence[NextLatLosses], *, gamma: float = 1.0) -> FBTNextLatLosses:
    """Return S0 + gamma * mean(extra sums); K1 has no additional term."""
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("FBT extra-pass gamma must be finite and nonnegative")
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
    coefficients = (1.0,) if len(passes) == 1 else (1.0,) + (float(gamma)/(len(passes)-1),) * (len(passes)-1)
    sums = {name: sum(coefficient * loss.sums[name] for coefficient, loss in zip(coefficients, passes))
            for name in first.sums}
    return FBTNextLatLosses(sums, dict(first.counts), dict(first.weights), passes, coefficients)


class FBTNextLatLM(NextLatLM):
    """Wrap OLMoFBT with the unchanged horizon-one NextLat implementation.

    FBT/RT execution is selected by the core's immutable runtime mode passed
    through ``backbone_kwargs``. ``enabled`` independently selects NextLat.
    Preserve gamma alongside the NextLat/core/mode configuration in checkpoints;
    it is not a learned parameter and does not change valid-position counts.
    """
    def __init__(self, backbone, config: NextLatConfig, *, enabled: bool = True, gamma: float = 1.0):
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("FBT extra-pass gamma must be finite and nonnegative")
        super().__init__(backbone, config, enabled=enabled)
        self._gamma = float(gamma)

    @property
    def gamma(self) -> float:
        return self._gamma

    def loss_sums(self, batch: NextLatBatch, *, backbone_kwargs: Mapping[str, Any] | None = None) -> FBTNextLatLosses:
        _validate_batch(batch, one_document_per_row=True)
        kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
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
        return aggregate_pass_losses(losses, gamma=self.gamma)
