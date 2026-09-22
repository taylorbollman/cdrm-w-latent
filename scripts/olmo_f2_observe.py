"""Bounded, detached F2 activation observations without replacing model math.

Capture one finite FBT/ordinary forward, then exit before backward. Ordinary
module hooks do not see functional tiled blocks, so a scoped wrapper also sees
the actual native tiled projections/rotations. All snapshots live on CPU.
Reconstructed attention is a diagnostic FP32 dot product of observed operands;
it is not the private cuDNN softmax or a bitwise replay of dyadic mixed matmuls.
The observer is for an exclusive diagnostic process, never throughput timing.
"""
from __future__ import annotations

import math
import threading

import torch

from cdrm.pretrained import olmo as ordinary_impl
from cdrm.pretrained import olmo_tiled as tiled_impl
from cdrm.pretrained.olmo_fbt import OLMoFBT


_OBSERVER_LOCK = threading.Lock()


def _snapshot(value):
    return value.detach().to(device="cpu", copy=True)


@torch.no_grad()
def distribution(value):
    """Finite-only scalar summaries, with excluded nonfinite counts explicit."""
    flat = value.detach().double().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    result = {"count": flat.numel(), "finite_count": finite.numel(),
              "nonfinite_count": flat.numel() - finite.numel()}
    if not finite.numel():
        return result | {key: None for key in ("min", "mean", "p50", "p95", "p99", "p999", "max")}
    quantiles = torch.quantile(finite, torch.tensor([.5, .95, .99, .999], device=finite.device, dtype=finite.dtype))
    return result | {"min": float(finite.min()), "mean": float(finite.mean()),
                     **{name: float(number) for name, number in zip(("p50", "p95", "p99", "p999"), quantiles)},
                     "max": float(finite.max())}


@torch.no_grad()
def activation_summary(value, valid, *, token_axis=1):
    """Summarize [B,T,D] states or [B,H,T,D] operands on valid tokens only."""
    if value.device.type != "cpu" or valid.device.type != "cpu":
        raise ValueError("F2 activation summaries require detached CPU snapshots")
    if value.ndim not in (3, 4) or token_axis not in (1, 2):
        raise ValueError("Expected batched token states or attention-head operands")
    # Statistics must not manufacture overflow from large but finite FP32
    # activations. This CPU reduction precision does not alter model execution.
    ordered = value.movedim(token_axis, 1).double()
    if valid.dtype != torch.bool or tuple(ordered.shape[:2]) != tuple(valid.shape):
        raise ValueError("Activation validity must align with batch/token axes")
    selected = ordered[valid]
    finite = bool(torch.isfinite(selected).all())
    rms = selected.square().mean(-1).sqrt()
    token_rms = ordered.square().flatten(2).mean(-1).sqrt()
    by_position = []
    for position in range(valid.shape[1]):
        chosen = token_rms[:, position][valid[:, position]]
        by_position.append({"position": position, "valid_rows": chosen.numel(),
                            "rms": float(chosen.square().mean().sqrt())
                            if chosen.numel() and bool(torch.isfinite(chosen).all()) else None})
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "valid_token_count": int(valid.sum()), "finite": finite,
              "component_values": distribution(selected),
              "absolute_components": distribution(selected.abs()),
              "token_or_head_rms": distribution(rms),
              "rms": float(selected.square().mean().sqrt()) if selected.numel() and finite else None,
              "rms_by_position": by_position}
    if value.ndim == 4:
        result["rms_by_head"] = [float(v) for v in selected.square().mean((0, 2)).sqrt()] if selected.numel() and finite else None
    return result


@torch.no_grad()
def reconstructed_attention_summary(query, permanent_key, temporary_key, valid, *,
                                    mask=None, is_causal=True):
    """FP32 diagnostic scores with permanent history and temporary diagonal.

    All operands are actual post-RoPE CPU snapshots. Ordinary attention passes
    the same input-derived key for both key arguments. Query padding, key
    padding, future entries and self-only queries are excluded where relevant.
    No cache prefixes or packed-document layouts are supported by this probe.
    """
    if any(value.device.type != "cpu" for value in (query, permanent_key, temporary_key, valid)):
        raise ValueError("Attention reconstruction requires CPU snapshots")
    if query.ndim != 4 or query.shape != permanent_key.shape or query.shape != temporary_key.shape:
        raise ValueError("Expected matching [batch,heads,tokens,head_dim] operands")
    batch, heads, length, width = query.shape
    if valid.dtype != torch.bool or valid.shape != (batch, length):
        raise ValueError("Attention validity must have [batch,tokens] shape")
    scores = query.float() @ permanent_key.float().transpose(-1, -2) / math.sqrt(width)
    indices = torch.arange(length)
    scores[:, :, indices, indices] = (query.float() * temporary_key.float()).sum(-1) / math.sqrt(width)
    allowed = valid[:, None, :, None] & valid[:, None, None, :]
    if is_causal:
        allowed = allowed & (indices[None, :] <= indices[:, None])[None, None]
    if mask is not None:
        if mask.device.type != "cpu" or mask.dtype != torch.bool:
            raise ValueError("Only actual boolean ordinary-attention masks are supported")
        allowed = allowed & mask
    allowed = allowed.expand(batch, heads, length, length)
    support = allowed.sum(-1)
    score_summary = distribution(scores[allowed])
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(support.unsqueeze(-1) > 0, masked, torch.zeros_like(masked))
    log_probabilities = safe.log_softmax(-1)
    probabilities = log_probabilities.exp() * allowed
    entropy = -torch.where(probabilities > 0, probabilities * log_probabilities, torch.zeros_like(probabilities)).sum(-1)
    eligible = support > 1
    maximum = probabilities.max(-1).values
    self_probability = probabilities[:, :, indices, indices]
    history = allowed.clone()
    history[:, :, indices, indices] = False
    historical_max = scores.masked_fill(~history, -torch.inf).max(-1).values
    self_scores = scores[:, :, indices, indices]
    gap_eligible = eligible & allowed[:, :, indices, indices] & history.any(-1)
    selected_entropy = entropy[eligible]
    normalized_entropy = selected_entropy / support[eligible].float().log()
    return {
        "arithmetic": "detached CPU FP32 score/softmax reconstruction from actual post-RoPE operands",
        "limitations": [
            "Not cuDNN's private probabilities or exact mixed-precision dyadic tile scores.",
            "Historical score arithmetic is deliberately FP32 for scale diagnostics; runtime RT mixed matmul outputs can round to BF16.",
            "Only valid queries/keys are summarized; saturation summaries require more than one allowed key.",
        ],
        "finite": score_summary["nonfinite_count"] == 0 and bool(torch.isfinite(probabilities[allowed]).all()),
        "valid_head_queries": int((support > 0).sum()),
        "self_only_head_queries_excluded": int((support == 1).sum()),
        "eligible_head_queries": int(eligible.sum()),
        "allowed_key_count": distribution(support[support > 0]),
        "scaled_logits": score_summary,
        "absolute_scaled_logits": distribution(scores[allowed].abs()),
        "maximum_probability": distribution(maximum[eligible]),
        "entropy_nats": distribution(selected_entropy),
        "entropy_fraction_of_uniform": distribution(normalized_entropy),
        "self_probability": distribution(self_probability[eligible]),
        "temporary_self_minus_best_history_logit": distribution((self_scores - historical_max)[gap_eligible]),
    }


class ActivationObserver:
    """Observe one finite forward; exit before backward and call ``report``.

    ``model`` may be FBTNextLatLM or its OLMoFBT core. Hooks return None and
    function wrappers return the original objects unchanged. The global scoped
    wrappers require an exclusive diagnostic process; nested observers fail.
    State snapshots, not production computations, are detached and copied.
    """
    def __init__(self, model, *, layers=(0, 1, 15), max_batch_size=2,
                 max_length=128, max_passes=4):
        core = model if isinstance(model, OLMoFBT) else getattr(model, "backbone", None)
        if not isinstance(core, OLMoFBT):
            raise TypeError("F2 observer expects OLMoFBT or its NextLat training wrapper")
        self.core, self.base = core, core.backbone
        layers = tuple(layers)
        if not layers or len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(self.base.layers) for i in layers):
            raise ValueError("Observed layers must be unique valid layer indices")
        for value in (max_batch_size, max_length, max_passes):
            if type(value) is not int or value <= 0:
                raise ValueError("Observation bounds must be positive integers")
        self.layers = tuple(sorted(layers))
        self.max_batch_size, self.max_length, self.max_passes = max_batch_size, max_length, max_passes
        self._layer_ids = {id(layer): i for i, layer in enumerate(self.base.layers)}
        self._handles, self._patches = [], []
        self._entered = self._closed = self._aborted = False
        self._core_calls = 0
        self._current_pass = self._ordinary = self._rt = None
        self.passes, self.records, self.fusions = [], [], []

    def _patch(self, module, name, wrapper_factory):
        original = getattr(module, name)
        self._patches.append((module, name, original))
        setattr(module, name, wrapper_factory(original))

    def _close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, name, original in reversed(self._patches):
            setattr(module, name, original)
        self._patches.clear()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("An activation observer is single-use")
        if not _OBSERVER_LOCK.acquire(blocking=False):
            raise RuntimeError("Nested/concurrent activation observers are unsupported")
        self._entered = True
        try:
            self._handles.extend([
                self.core.register_forward_pre_hook(self._core_pre, with_kwargs=True),
                self.base.register_forward_pre_hook(self._stack_pre, with_kwargs=True),
                self.base.register_forward_hook(self._stack_post, with_kwargs=True),
                self.core.fusion.register_forward_hook(self._fusion_post),
            ])
            for index in self.layers:
                layer = self.base.layers[index]
                self._handles.extend([
                    layer.register_forward_pre_hook(self._ordinary_pre, with_kwargs=True),
                    layer.register_forward_hook(self._ordinary_post, with_kwargs=True),
                    layer.att_proj.register_forward_hook(self._ordinary_projection),
                ])
            self._patch(ordinary_impl, "_apply_rope", lambda original: self._rope_wrapper(original, tiled=False))
            self._patch(tiled_impl, "_apply_rope", lambda original: self._rope_wrapper(original, tiled=True))
            self._patch(tiled_impl, "_project", self._project_wrapper)
            self._patch(tiled_impl, "tiled_recurrent_layer", self._tiled_wrapper)
        except BaseException:
            self._close()
            self._closed = self._aborted = True
            _OBSERVER_LOCK.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._close()
        self._closed = True
        self._aborted = exc_type is not None
        _OBSERVER_LOCK.release()
        return False

    def _core_pre(self, module, args, kwargs):
        self._core_calls += 1
        if self._core_calls != 1:
            raise ValueError("Observe one finite forward per context, then exit before backward")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("F2 activation observation excludes cached execution")

    def _stack_pre(self, module, args, kwargs):
        x = kwargs.get("inputs_embeds")
        if x is None or self._current_pass is not None:
            raise ValueError("Expected a nonnested finite stack with explicit embeddings")
        batch, length = x.shape[:2]
        if batch > self.max_batch_size or length > self.max_length or len(self.passes) >= self.max_passes:
            raise ValueError("Activation observation exceeds its declared batch/length/pass bounds")
        valid = kwargs.get("attention_mask")
        if valid is None:
            valid = torch.ones((batch, length), device=x.device, dtype=torch.bool)
        if valid.shape != (batch, length):
            raise ValueError("Activation observer does not accept cache prefixes")
        record = {"pass": len(self.passes), "input": _snapshot(x), "valid": _snapshot(valid.bool())}
        self.passes.append(record)
        self._current_pass = record

    def _stack_post(self, module, args, kwargs, output):
        self._current_pass["output"] = _snapshot(output.last_hidden_state)
        self._current_pass = None

    def _new_record(self, layer, x, *, kind, kwargs):
        if self._current_pass is None:
            raise ValueError("Observed layer outside its finite stack invocation")
        record = {"pass": self._current_pass["pass"], "layer": self._layer_ids[id(layer)],
                  "kind": kind, "input": _snapshot(x), "valid": self._current_pass["valid"],
                  "query_positions": _snapshot(kwargs["query_positions"]),
                  "key_positions": _snapshot(kwargs["key_positions"]), "_rope_calls": 0}
        self.records.append(record)
        return record

    def _ordinary_pre(self, layer, args, kwargs):
        if kwargs.get("past") is not None or self._ordinary is not None:
            raise ValueError("Unexpected cached/nested ordinary layer observation")
        record = self._new_record(layer, args[0], kind="ordinary", kwargs=kwargs)
        record["mask"] = None if kwargs["mask"] is None else _snapshot(kwargs["mask"])
        record["is_causal"] = kwargs["is_causal"]
        record["backend"] = kwargs["attention_backend"]
        self._ordinary = record

    def _ordinary_projection(self, module, args, output):
        record = self._ordinary
        if record is None:
            return
        config = self.base.config
        parts = output.split(config.model_dim, dim=-1)
        qkv = tuple(part.view(output.shape[0], output.shape[1], config.num_heads, config.head_dim).transpose(1, 2) for part in parts)
        self._capture_temporary(record, qkv)

    @staticmethod
    def _capture_temporary(record, qkv):
        for name, tensor in zip(("query_pre_rope", "temporary_key_pre_rope", "temporary_value"), qkv):
            record[name] = _snapshot(tensor)

    def _ordinary_post(self, layer, args, kwargs, output):
        record = self._ordinary
        if record["_rope_calls"] != 2:
            raise ValueError("Ordinary RoPE call pattern changed; observation labels need review")
        record["output"] = _snapshot(output[0])
        record["permanent_key_pre_rope"] = _snapshot(output[1][0])
        record["permanent_value"] = _snapshot(output[1][1])
        record["permanent_key_post_rope"] = record["temporary_key_post_rope"]
        self._ordinary = None

    def _rope_wrapper(self, original, *, tiled):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            record = self._rt if tiled else self._ordinary
            if record is not None:
                call = record["_rope_calls"]
                if call == 0:
                    record["query_post_rope"] = _snapshot(result)
                elif call == 1:
                    record["temporary_key_post_rope"] = _snapshot(result)
                elif tiled:
                    record["_permanent_rope"].append(_snapshot(result))
                else:
                    raise ValueError("Unexpected extra ordinary RoPE call")
                record["_rope_calls"] += 1
            return result
        return wrapped

    def _project_wrapper(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            record = self._rt
            if record is not None:
                if record["_project_calls"] == 0:
                    self._capture_temporary(record, result)
                else:
                    record["_memory_sources"].append(_snapshot(args[0]))
                record["_project_calls"] += 1
            return result
        return wrapped

    def _tiled_wrapper(self, original):
        def wrapped(layer, x, **kwargs):
            if id(layer) not in self._layer_ids or self._layer_ids[id(layer)] not in self.layers:
                return original(layer, x, **kwargs)
            if kwargs.get("past") is not None or self._rt is not None:
                raise ValueError("Unexpected cached/nested tiled layer observation")
            record = self._new_record(layer, x, kind="rt", kwargs=kwargs)
            record.update(alpha=kwargs["alpha"], backend="native eager tiled",
                          attention_precision=kwargs.get("attention_precision", "mixed"),
                          _project_calls=0, _permanent_rope=[], _memory_sources=[])
            self._rt = record
            try:
                result = original(layer, x, **kwargs)
                if record["_project_calls"] != x.shape[1] + 1 or record["_rope_calls"] != x.shape[1] + 2:
                    raise ValueError("Tiled projection/rotation call pattern changed; review observer")
                record["output"] = _snapshot(result[0])
                record["permanent_key_pre_rope"] = _snapshot(result[1][0])
                record["permanent_value"] = _snapshot(result[1][1])
                record["permanent_key_post_rope"] = torch.cat(record.pop("_permanent_rope"), dim=-2)
                record["memory_source"] = torch.cat(record.pop("_memory_sources"), dim=1)
                return result
            finally:
                self._rt = None
        return wrapped

    def _fusion_post(self, module, args, output):
        if not self.passes or self._current_pass is not None:
            raise ValueError("Fusion observation is outside the finite inter-pass boundary")
        valid = self.passes[0]["valid"]
        self.fusions.append({"destination_pass": len(self.passes),
                             "previous_top_state": _snapshot(args[0]),
                             "token_embedding": _snapshot(args[1]),
                             "fused_output": _snapshot(output),
                             "valid": valid[:, :-1] & valid[:, 1:]})

    @torch.no_grad()
    def report(self):
        if not self._closed or self._aborted or self._core_calls != 1:
            raise RuntimeError("Report requires one successfully completed observer context")
        expected_pairs = {(p["pass"], layer) for p in self.passes for layer in self.layers}
        if len(self.records) != len(expected_pairs) or {(r["pass"], r["layer"]) for r in self.records} != expected_pairs:
            raise ValueError("Observed pass/layer coverage is incomplete")
        records = []
        for record in self.records:
            valid = record["valid"]
            row = {key: record[key] for key in ("pass", "layer", "kind", "backend")}
            row.update(input=activation_summary(record["input"], valid),
                       output=activation_summary(record["output"], valid),
                       query_positions=record["query_positions"].tolist(),
                       key_positions=record["key_positions"].tolist())
            for name in ("query_pre_rope", "query_post_rope", "temporary_key_pre_rope", "temporary_key_post_rope",
                         "temporary_value", "permanent_key_pre_rope", "permanent_key_post_rope", "permanent_value"):
                row[name] = activation_summary(record[name], valid, token_axis=2)
            if record["kind"] == "rt":
                row.update(alpha=record["alpha"], attention_precision=record["attention_precision"],
                           memory_source=activation_summary(record["memory_source"], valid))
            row["attention"] = reconstructed_attention_summary(
                record["query_post_rope"], record["permanent_key_post_rope"],
                record["temporary_key_post_rope"], valid,
                mask=record.get("mask"), is_causal=record.get("is_causal", True))
            records.append(row)
        return {
            "schema": "olmo-f2-activation-observation-v1", "observed_layers": list(self.layers),
            "pass_count": len(self.passes), "records": records,
            "passes": [{"pass": p["pass"], "input": activation_summary(p["input"], p["valid"]),
                        "final_normalized_output": activation_summary(p["output"], p["valid"])} for p in self.passes],
            "fusion": [{"destination_pass": f["destination_pass"],
                         **{name: activation_summary(f[name], f["valid"]) for name in
                            ("previous_top_state", "token_embedding", "fused_output")}} for f in self.fusions],
            "scope": "One finite forward; detached CPU observations; no inference-cache or backward-replay observations",
            "reconstruction": "FP32 diagnostic dot products, not literal fused/dyadic runtime probabilities",
        }
