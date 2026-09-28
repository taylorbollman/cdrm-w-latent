"""Refillable campaign objective and local CUDA-graph accumulation.

No distributed collectives are performed here. The objective adapter exposes
world-size-scaled global normalization for a future actual DDP graph runner;
the local runner only accepts world_size=1. Capture never includes Adam or the
data/token clock. Legacy fixed-layout graph trainers remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from collections.abc import Sequence
from contextlib import contextmanager

import torch
from torch import nn

from .campaign_losses import DynamicNextLatLayout, compute_dynamic_nextlat_loss_sums
from .distributed_training import _normalization_contract, sum_objective_counts
from .fbt_training import FBTNextLatLM
from .lm_training import LMTrainingConfig, TERMS, TrainingCounters, optimizer_ownership
from .nextlat import NextLatBatch
from .olmo_static import PreparedFBTLayout
from .static_nextlat import _cpu_batch
from .static_training import StaticFBTTraining, _preparation_phase


class CampaignObjective(nn.Module):
    """Stable tensor storage with validated external batch/noise/count refills.

    All positive objective weights require positive global update counts. Local
    batches may have no targets at all. This intentionally fixes active parameter
    participation; changing global objective participation needs another plan.
    Validate/load outside capture; forward contains no tensor-to-host reads.
    """

    def __init__(self, model, batch, *, mode, global_counts, world_size=1,
                 feedback_noise=None, config=LMTrainingConfig()):
        super().__init__()
        if not isinstance(model, FBTNextLatLM) or model.pass_loss_policy != "campaign_v1":
            raise TypeError("CampaignObjective requires campaign_v1 FBTNextLatLM")
        if not model.training or not torch.is_grad_enabled() or model.config.dropout:
            raise ValueError("Campaign graphs require grad-enabled training with zero dropout")
        self.model, self.mode, self.config = model, mode, config
        self.training = model.training
        self.device = next(model.parameters()).device
        if any(p.device != self.device or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Campaign master parameters must be FP32 on one device")
        if config.precision == "bf16_mixed" and self.device.type != "cuda":
            raise ValueError("BF16 mixed campaign execution requires CUDA")
        cpu = _cpu_batch(batch)
        if cpu.input_ids.shape[1] < 3:
            raise ValueError("Campaign graph storage length must be at least three")
        self.batch = self._explicit_masks(cpu).to(self.device)
        self.forward_layout = PreparedFBTLayout(model.backbone, self.batch, right_padded_causal=True)
        self.loss_layout = DynamicNextLatLayout.from_batch(self.batch, model.config, enabled=model.enabled)
        self.weights = dict(model.objective_weights())
        self.world_size = world_size
        self.feedback_noise = self._checked_noise(feedback_noise)
        if self.feedback_noise is not None:
            self.feedback_noise = tuple(value.to(self.device).clone() for value in self.feedback_noise)
        self.coefficients = torch.zeros(len(TERMS), device=self.device, dtype=torch.float32)
        self.global_counts = {}
        self._set_normalization(global_counts)
        self.documents = int(cpu.valid_mask.any(-1).sum())
        self.input_tokens = int(cpu.valid_mask.sum())
        self._parameter_contract = self._parameter_signature()
        self._module_contract = self._module_signature()
        self._math_contract = StaticFBTTraining._math_signature()
        self._objective_contract = (model.config, model.enabled, model.gamma, model.pass_loss_policy,
                                    mode, config, world_size, tuple(self.weights.items()))
        self._forward_contract = self.forward_layout.validate_execution(mode, feedback_noise=self.feedback_noise)
        self._freeze_buffers()

    @property
    def counts(self):
        return self.loss_layout.counts

    @staticmethod
    def _explicit_masks(batch):
        return replace(batch, **{name: (torch.ones_like(batch.valid_mask) if getattr(batch, name) is None
                                        else getattr(batch, name))
                                 for name in ("ce_mask", "latent_mask", "kl_mask")})

    def _parameter_signature(self):
        return tuple((n, id(p), p.data_ptr(), tuple(p.shape), tuple(p.stride()), p.dtype, p.device, p.requires_grad)
                     for n, p in self.model.named_parameters())

    def _module_signature(self):
        return tuple((n, id(m), type(m), m.training) for n, m in self.model.named_modules())

    def owned_inputs(self):
        return (*self.batch.__dict__.values(), self.coefficients, *(self.feedback_noise or ()))

    def _buffer_signature(self):
        return tuple((id(v), v.data_ptr(), tuple(v.shape), tuple(v.stride()), v.dtype, v.device,
                      v.requires_grad, v._version) for v in self.owned_inputs())

    def _freeze_buffers(self):
        self._buffer_contract = self._buffer_signature()
        self._normalization_contract = (tuple(self.global_counts.items()), self.world_size)

    def _checked_noise(self, noise):
        # Accept CPU source buffers; validate values before copying into graph storage.
        if noise is not None:
            if not isinstance(noise, (tuple, list)) or not noise:
                raise ValueError("Feedback noise must be a nonempty tuple or None")
            devices = {value.device for value in noise if isinstance(value, torch.Tensor)}
            if len(devices) != 1 or next(iter(devices)) not in (torch.device("cpu"), self.device):
                raise ValueError("Feedback noise must share CPU or execution device")
            device = next(iter(devices))
        else:
            device = self.device
        return self.model.backbone._validate_feedback_noise(noise, self.mode,
            (self.batch.input_ids.shape[0], self.batch.input_ids.shape[1]-1, self.model.config.model_dim),
            device, self.model.backbone.token_embeddings.weight.dtype)

    def _validated_normalization(self, counts):
        _, counts = _normalization_contract(self.counts, self.weights, counts, self.world_size)
        if any(self.weights[t] > 0 and counts[t] <= 0 for t in TERMS):
            raise ValueError("Every enabled objective needs positive global update targets")
        return counts

    def _set_normalization(self, counts):
        counts = self._validated_normalization(counts)
        values = [self.weights[t] * self.world_size / counts[t] if self.weights[t] else 0.0 for t in TERMS]
        self.coefficients.copy_(torch.tensor(values, device=self.device, dtype=torch.float32))
        self.global_counts = counts

    def validate_execution(self):
        if not self.training or not self.model.training or not torch.is_grad_enabled():
            raise ValueError("Campaign objective requires grad-enabled train mode")
        if self._parameter_contract != self._parameter_signature() or self._module_contract != self._module_signature():
            raise ValueError("Campaign parameter/module ownership or runtime changed")
        if self._math_contract != StaticFBTTraining._math_signature():
            raise ValueError("Campaign floating-point runtime changed")
        if self._objective_contract != (self.model.config, self.model.enabled, self.model.gamma,
                self.model.pass_loss_policy, self.mode, self.config, self.world_size, tuple(self.weights.items())):
            raise ValueError("Campaign objective settings changed")
        if self._buffer_contract != self._buffer_signature():
            raise ValueError("Campaign input storage changed outside validated refill")
        if self._normalization_contract != (tuple(self.global_counts.items()), self.world_size):
            raise ValueError("Campaign normalization metadata changed")
        self.forward_layout.validate_execution(self.mode, expected_signature=self._forward_contract,
                                               feedback_noise=self.feedback_noise)
        self.loss_layout.validate_integrity()

    def validate_batch(self, batch, *, feedback_noise=None, global_counts=None):
        """Preflight without mutating any owned input buffer."""
        self.validate_execution()
        cpu = _cpu_batch(batch)
        self.forward_layout.validate_replacement_batch(cpu)
        # A temporary CPU loss layout is also a complete structural preflight.
        layout = DynamicNextLatLayout.from_batch(cpu, self.model.config, enabled=self.model.enabled)
        noise = self._checked_noise(feedback_noise)
        counts = self.global_counts if global_counts is None else global_counts
        _, counts = _normalization_contract(layout.counts, self.weights, counts, self.world_size)
        if any(self.weights[t] > 0 and counts[t] <= 0 for t in TERMS):
            raise ValueError("Every enabled objective needs positive global update targets")
        return cpu, noise, counts

    def load_batch(self, batch, *, feedback_noise=None, global_counts=None):
        cpu, noise, counts = self.validate_batch(batch, feedback_noise=feedback_noise, global_counts=global_counts)
        self.forward_layout.load_batch(cpu)
        self.loss_layout.load_batch(cpu)
        explicit = self._explicit_masks(cpu)
        for name, destination in self.batch.__dict__.items():
            destination.copy_(getattr(explicit, name))
        if self.feedback_noise is not None:
            for destination, source in zip(self.feedback_noise, noise):
                destination.copy_(source)
        self._set_normalization(counts)
        self.documents = int(cpu.valid_mask.any(-1).sum())
        self.input_tokens = int(cpu.valid_mask.sum())
        self._freeze_buffers()
        self.validate_execution()

    def forward(self, input_ids=None):
        if input_ids is not None and input_ids is not self.batch.input_ids:
            raise ValueError("Forward requires owned campaign token storage")
        output = self.forward_layout.forward(self.batch.input_ids, self.mode, feedback_noise=self.feedback_noise)
        passes = tuple(compute_dynamic_nextlat_loss_sums(hidden, output.embeddings,
            self.model.backbone.readout_weight, self.batch.input_ids, self.model.predictor,
            self.model.config, self.loss_layout, enabled=self.model.enabled)
            for hidden in output.pass_hidden_states)
        count = len(passes)
        ce_weights = (1.0,) if count == 1 else (.5,) + (.5 / (count-1),) * (count-1)
        sums = {t: sum(p[t] * (ce_weights[i] if t == "ce" else 1.0/count)
                       for i, p in enumerate(passes)) for t in TERMS}
        objective = sum(sums[t] * self.coefficients[i] for i, t in enumerate(TERMS))
        return {"objective": objective, "loss_sums": {t: sums[t].detach() for t in TERMS}}


class CampaignGraphTraining:
    """One-device accumulation using one refillable forward/backward graph.

    Each replay adds to persistent gradients. Zero only at logical-update
    boundaries, outside capture. No extra division by microbatch count occurs.
    Actual DDP/NCCL capture needs its own runner/qualification on multiple GPUs.
    """

    def __init__(self, adapter: CampaignObjective):
        if not isinstance(adapter, CampaignObjective) or adapter.world_size != 1:
            raise ValueError("Local campaign graph runner requires world_size=1")
        adapter.validate_execution()
        self.adapter, self.model = adapter, adapter.model
        self.graph = self.graph_result = self.stream = None
        self.active_names = self.gradient_addresses = None
        self.warmup_backward_calls = self.capture_backward_calls = self.replay_calls = 0
        self._failed = False
        self._at_update_boundary = True
        self._checkpoint_open = False

    def _addresses(self):
        return {n: None if p.grad is None else p.grad.data_ptr() for n, p in self.model.named_parameters()}

    def validate_execution(self):
        if self._failed:
            raise RuntimeError("Failed campaign runner must be reconstructed")
        if self._checkpoint_open:
            raise RuntimeError("Campaign execution is disabled during checkpoint publication")
        self.adapter.validate_execution()
        if self.gradient_addresses is not None and self._addresses() != self.gradient_addresses:
            raise ValueError("Campaign persistent gradient storage changed")

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def discard_backward(self):
        """Explicitly discard an unstepped diagnostic backward; no data clock advances."""
        self.validate_execution()
        self.zero_grad()
        self._at_update_boundary = True

    @contextmanager
    def checkpoint_boundary(self):
        """Expose grad=None to the existing atomic saver, retaining graph storage.

        Only use for checkpoint publication, not load/mutation/execution. On
        process restart construct a fresh adapter/runner and recapture graphs.
        This local helper makes no distributed checkpoint/recovery claim.
        """
        self.validate_execution()
        if not self._at_update_boundary or any(p.grad is not None and bool(p.grad.any())
                                               for p in self.model.parameters()):
            raise ValueError("Checkpoint requires a completed or explicitly discarded update boundary")
        gradients = [(p, p.grad) for p in self.model.parameters()]
        self._checkpoint_open = True
        try:
            for p, _ in gradients:
                p.grad = None
            yield
        finally:
            for p, gradient in gradients:
                p.grad = gradient
            self._checkpoint_open = False
            self.validate_execution()

    def _tensor_backward(self):
        with torch.autocast(self.adapter.device.type, dtype=torch.bfloat16,
                            enabled=self.adapter.config.precision == "bf16_mixed", cache_enabled=False):
            result = self.adapter(self.adapter.batch.input_ids)
        result["objective"].backward()
        # Replay needs scalar storage, not the completed Python autograd graph.
        # Keeping its grad_fn alive also keeps AccumulateGrad's capture-stream
        # association alive during a later eager diagnostic on another stream.
        return {**result, "objective": result["objective"].detach()}

    def initialize_gradients(self):
        self.validate_execution()
        if self.active_names is not None:
            return
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError("Prepare campaign graph at a cleared gradient boundary")
        self._tensor_backward()
        self.warmup_backward_calls += 1
        self.active_names = tuple(n for n, p in self.model.named_parameters() if p.grad is not None)
        expected = tuple(n for n, p in self.model.named_parameters() if p.requires_grad)
        if self.active_names != expected:
            raise ValueError("Campaign graph did not participate in all active parameters")
        self.gradient_addresses = self._addresses()
        self.zero_grad()

    def capture(self, *, warmup=10, phase_observer=None):
        if self.adapter.device.type != "cuda":
            raise ValueError("Campaign CUDA graph requires CUDA; no CPU fallback")
        if type(warmup) is not int or warmup < 10:
            raise ValueError("Campaign capture requires at least ten warmup backward calls")
        if self.graph is not None:
            raise ValueError("Campaign graph is already captured")
        if not self._at_update_boundary:
            raise ValueError("Discard pending diagnostic backward before capture")
        try:
            with _preparation_phase(phase_observer, "gradient_initialization"):
                self.initialize_gradients()
            self.stream = torch.cuda.Stream(device=self.adapter.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.adapter.device))
            with _preparation_phase(phase_observer, "warmup"):
                with torch.cuda.stream(self.stream):
                    for _ in range(warmup):
                        self.zero_grad()
                        self._tensor_backward()
                        self.warmup_backward_calls += 1
                torch.cuda.current_stream(self.adapter.device).wait_stream(self.stream)
                torch.cuda.synchronize(self.adapter.device)
            self.zero_grad()
            self.validate_execution()
            with _preparation_phase(phase_observer, "capture"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    result = self._tensor_backward()
                torch.cuda.synchronize(self.adapter.device)
                self.graph, self.graph_result = graph, result
                self.capture_backward_calls += 1
                self.zero_grad()
                self.validate_execution()
        except BaseException:
            self._failed = True
            raise

    def backward(self, microbatches: Sequence[NextLatBatch], *, feedback_noises=None, replay=False):
        """Replace prior diagnostic gradients with one accumulated update, no Adam.

        Caller-owned noise tensors must remain unchanged until this call returns.
        CPU batch metadata is snapshotted before gradient mutation.
        """
        if type(replay) is not bool:
            raise TypeError("replay must be boolean")
        if not microbatches:
            raise ValueError("A logical update needs at least one physical microbatch")
        if feedback_noises is None:
            feedback_noises = [None] * len(microbatches)
        if len(feedback_noises) != len(microbatches):
            raise ValueError("One feedback-noise tuple is needed per microbatch")
        if replay and self.graph is None:
            raise ValueError("Capture campaign graph before replay")
        self.validate_execution()
        counts = sum_objective_counts([self.model.counts(batch) for batch in microbatches])
        # Preflight every input before any gradient mutation.
        checked = [self.adapter.validate_batch(batch, feedback_noise=noise, global_counts=counts)
                   for batch, noise in zip(microbatches, feedback_noises)]
        try:
            self._at_update_boundary = False
            self.initialize_gradients()
            self.zero_grad()
            totals = torch.zeros(len(TERMS), device=self.adapter.device)
            documents = tokens = 0
            for cpu, noise, _ in checked:
                self.adapter.load_batch(cpu, feedback_noise=noise, global_counts=counts)
                if replay:
                    self.graph.replay()
                    self.replay_calls += 1
                    result = self.graph_result
                else:
                    result = self._tensor_backward()
                # Accumulate before graph output storage is overwritten by replay.
                totals.add_(torch.stack([result["loss_sums"][t] for t in TERMS]))
                documents += self.adapter.documents
                tokens += self.adapter.input_tokens
            self.validate_execution()
            if not bool(torch.isfinite(totals).all()) or not all(
                    bool(torch.isfinite(p.grad).all()) for p in self.model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite accumulated campaign loss or gradients")
            sums = dict(zip(TERMS, totals.cpu().tolist()))
            return {"loss_sums": sums, "counts": counts,
                    "objective": sum(sums[t] * self.adapter.weights[t] / counts[t] for t in TERMS if counts[t]),
                    "microbatches": len(microbatches), "documents": documents, "input_tokens": tokens}
        except BaseException:
            self.zero_grad()
            self._failed = True
            raise

    def optimizer_step(self, optimizer, microbatches, *, feedback_noises=None, replay=False,
                       scheduler=None, counters=None):
        optimizer_ownership(self.model, optimizer)
        counters = TrainingCounters() if counters is None else counters
        if not isinstance(counters, TrainingCounters):
            raise TypeError("counters must be TrainingCounters")
        TrainingCounters(**asdict(counters))
        valid_tokens = sum(int(batch.valid_mask.sum()) for batch in microbatches)
        if scheduler is not None and hasattr(scheduler, "validate_next_update"):
            scheduler.validate_next_update(valid_tokens)
        result = self.backward(microbatches, feedback_noises=feedback_noises, replay=replay)
        try:
            rates = [group["lr"] for group in optimizer.param_groups]
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                float("inf") if self.adapter.config.max_grad_norm is None else self.adapter.config.max_grad_norm,
                error_if_nonfinite=True, foreach=False)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            counters.optimizer_updates += 1
            counters.microbatches += result["microbatches"]
            counters.documents += result["documents"]
            counters.input_tokens += result["input_tokens"]
            counters.ce_positions += result["counts"]["ce"]
            counters.latent_pairs += result["counts"]["latent"]
            counters.kl_triples += result["counts"]["kl"]
            self._at_update_boundary = True
            return {**result, "gradient_norm_before_clip": float(norm), "lr_used": rates,
                    "lr_next": [g["lr"] for g in optimizer.param_groups], "counters": asdict(counters)}
        except BaseException:
            self._failed = True
            raise
        finally:
            self.zero_grad()
