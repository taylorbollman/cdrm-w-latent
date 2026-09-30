"""Observe fixed-state feedback sensitivity through the canonical model forward.

The caller supplies the existing common-FP32 evaluation_runtime/no_grad scope.
Hooks observe the actual fusion and native-stack inputs; they do not replace
forward execution. Entropies are diagnostic readouts on exactly the KL-aligned
positions, not new losses or an alternate latent recurrence.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math

import torch
from torch.nn import functional as F

from .campaign_recipe import CampaignRecipe
from .document_policy import feedback_eligibility
from .fbt_training import FBTNextLatLM
from .lm_training import TERMS
from .nextlat import build_nextlat_masks


def _finite(value):
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("Nonfinite forward diagnostic observation")
    return scalar


def _fingerprint(tensor):
    raw = tensor.detach().contiguous().cpu()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def _hidden_geometry(hidden, valid):
    selected = hidden[valid].double()
    count = selected.shape[0]
    if not count:
        return {"positions": 0, "elements": 0, "rms": None, "position_centered_rms": None,
                "position_centered_rms_over_rms": None}
    rms = _finite(selected.square().mean().sqrt())
    centered = _finite((selected-selected.mean(0, keepdim=True)).square().mean().sqrt())
    return {"positions": count, "elements": selected.numel(), "rms": rms,
            "position_centered_rms": centered,
            "position_centered_rms_over_rms": centered/rms if rms else None}


def _paired_geometry(value, token_input, eligible):
    if value.shape != token_input.shape or value.shape[:-1] != eligible.shape:
        raise ValueError("Feedback geometry shape mismatch")
    a, b = value[eligible].double(), token_input[eligible].double()
    count, elements = a.shape[0], a.numel()
    if not count:
        return {"positions": 0, "elements": 0, "value_rms": None, "token_input_rms": None,
                "rms_ratio": None, "cosine": None, "difference_rms": None,
                "dot": 0., "value_squared_sum": 0., "token_input_squared_sum": 0.}
    squared_a, squared_b = _finite(a.square().sum()), _finite(b.square().sum())
    dot = _finite((a*b).sum())
    rms_a, rms_b = math.sqrt(squared_a/elements), math.sqrt(squared_b/elements)
    return {"positions": count, "elements": elements, "value_rms": rms_a,
            "token_input_rms": rms_b, "rms_ratio": rms_a/rms_b if rms_b else None,
            "cosine": max(-1., min(1., dot/math.sqrt(squared_a*squared_b))) if squared_a and squared_b else None,
            "difference_rms": _finite((a-b).square().mean().sqrt()),
            "dot": dot, "value_squared_sum": squared_a, "token_input_squared_sum": squared_b}


def _entropy(states, readout, chunk_size):
    count, total = states.shape[0], 0.
    maximum, tolerance = math.log(readout.shape[0]), 1e-5
    for start in range(0, count, chunk_size):
        log_prob = F.log_softmax(F.linear(states[start:start+chunk_size], readout).float(), dim=-1)
        entropy = -(log_prob.exp()*log_prob).sum(-1)
        total += _finite(entropy.double().sum())
        if bool(((entropy < -tolerance) | (entropy > maximum+tolerance)).any()):
            raise ValueError("Readout entropy outside [0, log(vocabulary)] tolerance")
    return {"count": count, "sum_nats": total, "mean_nats": total/count if count else None,
            "vocabulary_rows": readout.shape[0], "projection_chunk_positions": chunk_size,
            "bounds_checked_per_position": True, "bounds_tolerance_nats": tolerance}


def feedback_forward_probe(model, recipe, batch, beta):
    """Return JSON observations for one physical B1 K4 NF/NFR canonical forward.

    Position-centered RMS subtracts the feature-wise mean over valid positions
    in this one row. Raw fusion geometry excludes the first token and any
    feedback-ineligible suffix positions. Blended geometry observes executed
    stack inputs, including exact beta-zero bypass; no synthetic blend is used.
    The caller reduces numerators/counts across rows, not local means.
    """
    if (not isinstance(model, FBTNextLatLM) or not isinstance(recipe, CampaignRecipe)
            or recipe.arm not in ("NF", "NFR") or not model.enabled
            or model.pass_loss_policy != "campaign_v1" or model.gamma != 1.
            or model.config.document_policy != recipe.document_policy):
        raise ValueError("Forward probe requires a matching K4 NF/NFR campaign model")
    if torch.is_grad_enabled() or any(module.training for module in model.modules()):
        raise ValueError("Use the common FP32 evaluation_runtime/no_grad scope")
    parameters = tuple(model.parameters())
    device = parameters[0].device
    if any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise ValueError("Forward diagnostic requires unchanged FP32 parameters on one device")
    if (torch.is_autocast_enabled(device.type) or model.backbone.backbone.attention_precision != "fp32"
            or model.backbone.backbone.rt_implementation != "native"):
        raise ValueError("Forward diagnostic requires native FP32 evaluation execution")
    if batch.input_ids.shape != (1, recipe.sequence_length) or batch.input_ids.device != device:
        raise ValueError("Forward diagnostic requires physical B1 at the declared sequence length/device")
    mode = replace(recipe.mode(), feedback_jitter=0., beta=beta)
    masks = build_nextlat_masks(batch, document_policy=recipe.document_policy)
    eligible = feedback_eligibility(batch.valid_mask, batch.document_ids, recipe.document_policy)
    observed, fusion_records, stack_inputs = [], [], []

    def observe_fbt(module, args, kwargs, output):
        observed.append((kwargs["inputs_embeds"].detach(), tuple(h.detach() for h in output.pass_hidden_states)))

    def observe_fusion(module, args, output):
        fusion_records.append((args[1].detach(), output.detach()))

    def observe_stack(module, args, kwargs):
        stack_inputs.append(kwargs["inputs_embeds"].detach())

    handles = [model.backbone.register_forward_hook(observe_fbt, with_kwargs=True),
               model.backbone.fusion.register_forward_hook(observe_fusion),
               model.backbone.backbone.register_forward_pre_hook(observe_stack, with_kwargs=True)]
    try:
        losses = model.loss_sums(batch, backbone_kwargs={"mode": mode, "feedback_noise": None,
                                                       "right_padded_causal": True})
    finally:
        for handle in handles:
            handle.remove()
    if (len(observed) != 1 or len(observed[0][1]) != 4 or len(stack_inputs) != 4
            or len(losses.pass_losses) != 4 or len(fusion_records) != (0 if mode.beta == 0 else 3)):
        raise ValueError("Unexpected canonical forward/fusion/stack invocation count")
    embeddings, hidden_states = observed[0]
    counts, weights = model.counts(batch), model.objective_weights()
    if losses.counts != counts or losses.weights != weights:
        raise ValueError("Canonical objective counts or weights differ")
    kl_mask = masks["kl"]
    if int(kl_mask.sum()) != counts["kl"]:
        raise ValueError("Entropy eligibility must exactly match active NextLat KL triples")
    readout = model.backbone.readout_weight.detach()
    chunk_size = model.config.effective_ce_chunk_size
    passes = []
    for index, (hidden, loss) in enumerate(zip(hidden_states, losses.pass_losses)):
        if hidden.dtype != torch.float32 or hidden.requires_grad or loss.counts != counts:
            raise ValueError("Expected detached FP32 states and identical pass denominators")
        sums = {term: _finite(loss.sums[term]) for term in TERMS}
        # Same source/next-embedding and detached teacher coordinates as the
        # unchanged KL loss. All projections use every native vocabulary row.
        sources = hidden[:, :-2][kl_mask]
        next_embeddings = embeddings[:, 1:-1][kl_mask]
        teacher = hidden[:, 1:-1][kl_mask]
        predicted = model.predictor(sources, next_embeddings) if sources.shape[0] else sources
        passes.append({"index": index, "sums": sums, "counts": dict(counts),
            "means": {term: sums[term]/counts[term] if counts[term] else None for term in TERMS},
            "hidden_sha256": _fingerprint(hidden), "hidden_shape": list(hidden.shape), "hidden_dtype": str(hidden.dtype),
            "hidden_geometry": _hidden_geometry(hidden, batch.valid_mask),
            "teacher_entropy_on_kl_positions": _entropy(teacher, readout, chunk_size),
            "student_entropy_on_kl_positions": _entropy(predicted, readout, chunk_size)})
    feedback = []
    for index in range(1, 4):
        token_input = embeddings[:, 1:]
        raw_geometry = None
        if mode.beta:
            recorded_input, fused = fusion_records[index-1]
            if not torch.equal(recorded_input, token_input):
                raise ValueError("Fusion token input differs from the original embedding suffix")
            raw_geometry = _paired_geometry(fused, recorded_input, eligible)
        feedback.append({"index": index, "eligible_positions": int(eligible.sum()),
            "raw_fusion_vs_token_input": raw_geometry,
            "actual_blended_vs_token_input": _paired_geometry(stack_inputs[index][:, 1:], token_input, eligible),
            "actual_stack_input_sha256": _fingerprint(stack_inputs[index])})
    aggregate_sums = {term: _finite(losses.sums[term]) for term in TERMS}
    beta_zero_exact = (all(row["hidden_sha256"] == passes[0]["hidden_sha256"] for row in passes)
                       and all(torch.equal(value, stack_inputs[0]) for value in stack_inputs)) if mode.beta == 0 else None
    if beta_zero_exact is False:
        raise ValueError("Beta-zero control failed exact pass/input equality")
    return {"schema": "olmo-feedback-forward-probe-v1", "arm": recipe.arm, "mode": asdict(mode),
        "precision": "common_fp32_no_jitter", "physical_batch": 1, "input_tokens": int(batch.valid_mask.sum()),
        "passes": passes, "counts": dict(counts), "weights": dict(weights),
        "aggregate_sums": aggregate_sums,
        "aggregate_means": {term: aggregate_sums[term]/counts[term] if counts[term] else None for term in TERMS},
        "term_pass_coefficients": {term: list(value) for term, value in losses.term_pass_coefficients.items()},
        "feedback": feedback, "fusion_forward_calls": len(fusion_records), "native_stack_forward_calls": len(stack_inputs),
        "beta_zero_exact_pass_and_input_control": beta_zero_exact,
        "hidden_hash_scope": "Complete contiguous FP32 [1,T,D] bytes, including padded positions; shape/dtype reported separately",
        "entropy_scope": "Diagnostic full-vocabulary teacher/student recomputation on actual KL-aligned triples; predictor batch regrouping can change FP32 rounding, so this is not exact logged-KL reconstruction or sampled-generation entropy",
        "geometry_scope": "Hidden centering is within this B1 row over valid positions. Fusion/blend cosine flattens all eligible position-feature elements; it is not mean token cosine or cosine of pooled means",
        "optimizer_updates": 0, "model_forward_calls": len(observed)}
