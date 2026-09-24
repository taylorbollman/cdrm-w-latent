"""Fixed-layout FBT/RT/NextLat training with optional CUDA graph replay.

Only the forward, canonical objective and backward are captured. Input checks,
copies, nonfinite checks, clipping, AdamW and scheduling remain outside capture.
An immutable layout is a contract: different masks/shapes/modes need a new plan.
There is one physical batch per update; accumulation is deliberately unsupported.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import math
import torch

from .fbt_training import FBTNextLatLM, aggregate_pass_losses
from .lm_training import (LMTrainingConfig, TrainingCounters, TERMS,
                          optimizer_ownership, optimizer_state_bytes)
from .olmo_static import PreparedFBTLayout
from .static_nextlat import PreparedNextLatLayout, compute_static_nextlat_loss_sums


def normalized_objective(result):
    """Exactly the canonical single-microbatch optimizer's operation order."""
    parts = [result.sums[term] * (result.weights[term] / result.counts[term])
             for term in TERMS if result.counts[term] and result.weights[term]]
    if not parts:
        raise ValueError("Update has no valid positively weighted objective")
    return sum(parts)


@contextmanager
def _preparation_phase(observer, phase):
    """Notify only outside capture; preserve the actual failure if observation fails."""
    if observer is not None:
        observer(phase, "begin")
    try:
        yield
    except BaseException as error:
        if observer is not None:
            try:
                observer(phase, "error")
            except Exception as observation_error:
                error.add_note(f"Preparation observer also failed: {observation_error}")
        raise
    else:
        if observer is not None:
            observer(phase, "end")


class StaticFBTTraining:
    """Prepared tensor execution; capture is explicit and never automatic.

    Initialize on the final device with the final training/attention/checkpoint
    settings. Weight updates in existing storage are allowed. Parameter storage,
    objective configuration and fixed batch structure must remain unchanged.
    Grad buffers are allocated only for parameters observed in the fixed graph;
    unused trainable parameters keep grad=None and avoid unintended Adam decay.
    """

    def __init__(self, model, batch, *, mode, config=LMTrainingConfig()):
        if not isinstance(model, FBTNextLatLM):
            raise TypeError("Static training requires FBTNextLatLM")
        if not model.training or not torch.is_grad_enabled():
            raise ValueError("Prepare static training in grad-enabled train mode")
        if model.config.dropout:
            raise ValueError("Static training currently requires zero predictor dropout")
        self.model, self.mode, self.config = model, mode, config
        device = next(model.parameters()).device
        if config.precision == "bf16_mixed" and device.type != "cuda":
            raise ValueError("BF16 mixed static training requires CUDA")
        self.batch = batch.to(device)
        self.batch = replace(self.batch, input_ids=self.batch.input_ids.clone())
        self.forward_layout = PreparedFBTLayout(model.backbone, self.batch)
        self.loss_layout = PreparedNextLatLayout.from_batch(self.batch, model.config, enabled=model.enabled)
        self.counts = dict(self.loss_layout.counts)
        self.weights = dict(model.objective_weights())
        if not any(self.counts[t] and self.weights[t] for t in TERMS):
            raise ValueError("Update has no valid positively weighted objective")
        self._objective_contract = (model.config, model.enabled, model.gamma, self.config)
        self._forward_signature = self.forward_layout.validate_execution(self.mode)
        self._parameter_contract = self._parameter_signature()
        self._module_contract = self._module_signature()
        self._math_contract = self._math_signature()
        self._input_pointer = self.batch.input_ids.data_ptr()
        self.documents = int(batch.valid_mask.any(-1).sum())
        self.input_tokens = int(batch.valid_mask.sum())
        self.graph = None
        self.graph_result = None
        self.active_names = None
        self.gradient_addresses = None
        self.warmup_backward_calls = 0
        self.capture_backward_calls = 0
        self.replay_calls = 0

    def _parameter_signature(self):
        return tuple((name, id(p), p.data_ptr(), p.shape, p.dtype, p.device, p.requires_grad)
                     for name, p in self.model.named_parameters())

    def _module_signature(self):
        return tuple((n, id(m), type(m), m.training) for n, m in self.model.named_modules())

    @staticmethod
    def _math_signature():
        return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())

    def validate_execution(self):
        if not self.model.training or not torch.is_grad_enabled():
            raise ValueError("Static training requires grad-enabled train mode")
        if self._parameter_contract != self._parameter_signature():
            raise ValueError("Static model parameter storage/ownership changed; prepare again")
        if self._module_contract != self._module_signature() or self._math_contract != self._math_signature():
            raise ValueError("Static module/runtime settings changed; prepare again")
        if self._objective_contract != (self.model.config, self.model.enabled, self.model.gamma, self.config):
            raise ValueError("Static objective configuration changed; prepare again")
        if self.batch.input_ids.data_ptr() != self._input_pointer:
            raise ValueError("Static token buffer was replaced")
        self.forward_layout.validate_execution(self.mode, expected_signature=self._forward_signature)
        self.loss_layout.validate_integrity()
        if self.counts != self.loss_layout.counts or self.weights != self.loss_layout.weights:
            raise ValueError("Static objective metadata changed; prepare again")
        if self.gradient_addresses is not None:
            actual = {n: None if p.grad is None else p.grad.data_ptr()
                      for n, p in self.model.named_parameters()}
            if actual != self.gradient_addresses:
                raise ValueError("Persistent gradient buffers changed; prepare again")

    def load_batch(self, batch):
        """Validate before mutating token storage. CPU batches avoid GPU syncs."""
        self.validate_execution()
        self.forward_layout.validate_batch(batch)
        self.loss_layout.validate_batch(batch)
        self.batch.input_ids.copy_(batch.input_ids)

    def loss_sums(self):
        """Tensor-only body; validations and mutable-state checks are external."""
        output = self.forward_layout.forward(self.batch.input_ids, mode=self.mode)
        losses = tuple(compute_static_nextlat_loss_sums(hidden, output.embeddings,
            self.model.backbone.readout_weight, self.batch.input_ids, self.model.predictor,
            self.model.config, self.loss_layout, enabled=self.model.enabled)
            for hidden in output.pass_hidden_states)
        return aggregate_pass_losses(losses, gamma=self.model.gamma)

    def _tensor_backward(self):
        for _, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        # Prevent capture from freezing an autocast copy of a trainable weight.
        with torch.autocast(self.batch.input_ids.device.type, dtype=torch.bfloat16,
                            enabled=self.config.precision == "bf16_mixed", cache_enabled=False):
            result = self.loss_sums()
            objective = normalized_objective(result)
        objective.backward()
        return result

    def initialize_gradients(self):
        self.validate_execution()
        if self.active_names is not None:
            return
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError("Initialize static training at a cleared gradient boundary")
        self._tensor_backward()
        self.warmup_backward_calls += 1
        self.active_names = tuple(n for n, p in self.model.named_parameters() if p.grad is not None)
        self.gradient_addresses = {n: None if p.grad is None else p.grad.data_ptr()
                                   for n, p in self.model.named_parameters()}
        if not self.active_names:
            raise ValueError("Static objective has no participating parameter")

    def capture(self, *, warmup=10, release_transient_cache=False, phase_observer=None):
        """Prepare/capture without changing the canonical tensor computation.

        Optional ``phase_observer(phase, event)`` receives begin/end/error for
        gradient_initialization, warmup, capture, and optional transient_cleanup.
        It can persist synchronized allocator/device snapshots and reset peak
        counters, but must be read-only with respect to model/layout/gradient
        tensors. Callbacks never run inside the graph context; failed capture
        observation runs only after that context exits. Observation is disabled
        by default and this method itself never resets allocator peak counters.

        ``release_transient_cache`` explicitly collects unreachable Python
        objects and releases unused allocator cache after synchronized warmup,
        before graph creation. It cannot release live model/optimizer/gradient
        tensors or graph-owned storage, and is not a model memory optimization.
        The default retains the historical allocator preparation behavior.
        """
        if self.batch.input_ids.device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA; no CPU fallback")
        if type(warmup) is not int or warmup < 1:
            raise ValueError("Warmup must be a positive integer")
        if type(release_transient_cache) is not bool:
            raise TypeError("release_transient_cache must be boolean")
        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("phase_observer must be callable or None")
        if self.graph is not None:
            raise ValueError("This training plan already owns a captured graph")
        with _preparation_phase(phase_observer, "gradient_initialization"):
            self.initialize_gradients()
        with _preparation_phase(phase_observer, "warmup"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self._tensor_backward()
                    self.warmup_backward_calls += 1
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        if release_transient_cache:
            with _preparation_phase(phase_observer, "transient_cleanup"):
                gc.collect()
                torch.cuda.empty_cache()
        with _preparation_phase(phase_observer, "capture"):
            self.validate_execution()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = self._tensor_backward()
            torch.cuda.synchronize()
            self.graph, self.graph_result = graph, result
            self.capture_backward_calls += 1
            self.validate_execution()

    def backward(self, *, replay=False):
        self.initialize_gradients()
        self.validate_execution()
        if replay:
            if self.graph is None:
                raise ValueError("Capture a graph before replay")
            self.graph.replay()
            self.replay_calls += 1
            return self.graph_result
        return self._tensor_backward()

    def optimizer_step(self, optimizer, batch, *, replay=False, scheduler=None, counters=None):
        """Canonical one-batch update; persistent buffers survive boundaries."""
        optimizer_ownership(self.model, optimizer)
        counters = TrainingCounters() if counters is None else counters
        self.load_batch(batch)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        try:
            result = self.backward(replay=replay)
            totals = dict(zip(TERMS, torch.stack([result.sums[t].detach().float() for t in TERMS]).cpu().tolist()))
            if not all(math.isfinite(v) for v in totals.values()):
                raise FloatingPointError("Nonfinite static objective sum")
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters,
                float("inf") if self.config.max_grad_norm is None else self.config.max_grad_norm,
                error_if_nonfinite=True, foreach=False)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        except BaseException:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            raise
        counters.optimizer_updates += 1
        counters.microbatches += 1
        counters.documents += self.documents
        counters.input_tokens += self.input_tokens
        counters.ce_positions += self.counts["ce"]
        counters.latent_pairs += self.counts["latent"]
        counters.kl_triples += self.counts["kl"]
        means = {t: totals[t] / self.counts[t] if self.counts[t] else 0.0 for t in TERMS}
        return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True,
            "loss_sums": totals, "counts": dict(self.counts), "loss_means": means,
            "objective_weights": dict(self.weights), "objective": sum(self.weights[t]*means[t] for t in TERMS),
            "gradient_norm_before_clip": float(norm), "max_grad_norm": self.config.max_grad_norm,
            "lr_used": learning_rates, "lr_next": [g["lr"] for g in optimizer.param_groups],
            "optimizer_state_bytes_by_device": optimizer_state_bytes(optimizer), "counters": asdict(counters)}
