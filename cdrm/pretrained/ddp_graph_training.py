"""Fixed-layout native OLMo training through actual DDP CUDA-graph capture.

The caller owns process-group initialization, rank-coordinated error handling,
health checks, clipping, optimizer/scheduler steps and checkpointing. This module
captures the real DDP forward/objective/backward, including reducer collectives;
it is not local graph computation followed by an explicit gradient all-reduce.
One physical batch per rank/update is supported. Masks, denominators, modes and
parameter participation stay fixed; changes require a fresh DDP wrapper/graph.
"""
from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .distributed_training import _normalization_contract
from .lm_training import LMTrainingConfig, TERMS
from .static_training import StaticFBTTraining, _preparation_phase


class PreparedDDPObjective(nn.Module):
    """Capture-safe objective adapter; validate and load batches outside forward.

    Canonical parameters are registered exactly once as ``model``. ``plan`` is
    ordinary Python execution state and owns no duplicate module registration.
    Do not call its gradient initialization/backward methods: actual DDP owns
    those operations and may rebuild gradient buckets during warmup.

    The prepared single-device layout currently requires some positively weighted
    local objective on every rank. Individual auxiliary terms may be locally
    empty; the caller declares the globally active parameter names to the runner.
    """

    def __init__(self, model, batch, *, mode, global_counts: Mapping[str, int],
                 world_size: int, config=LMTrainingConfig()):
        super().__init__()
        self.model = model
        self.plan = StaticFBTTraining(model, batch, mode=mode, config=config)
        self.training = model.training
        _, counts = _normalization_contract(self.plan.counts, self.plan.weights,
                                            global_counts, world_size)
        self.world_size = world_size
        self._global_counts = tuple(counts[t] for t in TERMS)
        self._coefficients = tuple((t, self.plan.weights[t] * world_size / counts[t])
                                   for t in TERMS if counts[t] and self.plan.weights[t])
        self._contract = (self.world_size, self._global_counts, self._coefficients)

    @property
    def global_counts(self):
        return dict(zip(TERMS, self._global_counts))

    @property
    def counts(self):
        return dict(self.plan.counts)

    def validate_execution(self):
        if not self.training or not torch.is_grad_enabled():
            raise ValueError("Prepared DDP requires grad-enabled train mode")
        if (self.world_size, self._global_counts, self._coefficients) != self._contract:
            raise ValueError("Prepared DDP normalization changed; prepare again")
        self.plan.validate_execution()

    def load_batch(self, batch):
        self.validate_execution()
        self.plan.load_batch(batch)

    def forward(self, input_ids=None):
        # No tensor-to-host reads, input validation or denominator collectives.
        # The Python structure and scalar coefficients are fixed at preparation.
        # GPU DDP's input mover in the installed build rejects an empty argument
        # tuple. Pass the existing static token storage through its real forward
        # boundary; no data copy or extra differentiable branch is introduced.
        if input_ids is not None and input_ids is not self.plan.batch.input_ids:
            raise ValueError("DDP input must be the prepared static token storage")
        result = self.plan.loss_sums()
        objective = sum(result.sums[t] * coefficient for t, coefficient in self._coefficients)
        return {"objective": objective,
                "loss_sums": {t: result.sums[t].detach() for t in TERMS},
                "pass_loss_sums": tuple({t: p.sums[t].detach() for t in TERMS}
                                         for p in result.pass_losses),
                "pass_coefficients": result.pass_coefficients}


def _active_contract(model, expected_active_names):
    if (not isinstance(expected_active_names, Sequence)
            or isinstance(expected_active_names, (str, bytes))
            or not expected_active_names
            or any(not isinstance(n, str) for n in expected_active_names)
            or len(set(expected_active_names)) != len(expected_active_names)):
        raise ValueError("Declare a nonempty unique sequence of globally active parameter names")
    parameters = dict(model.named_parameters())
    unknown = set(expected_active_names) - parameters.keys()
    frozen = {n for n in expected_active_names if n in parameters and not parameters[n].requires_grad}
    if unknown or frozen:
        raise ValueError(f"Invalid active parameter names: unknown={sorted(unknown)}, frozen={sorted(frozen)}")
    return tuple(n for n in parameters if n in set(expected_active_names))


class DDPGraphTraining:
    """Capture one fixed-layout DDP forward/backward on each participating rank.

    Construct one instance per process/GPU with an initialized NCCL process group.
    ``capture`` creates DDP on its dedicated side stream and performs at least 11
    actual DDP eager backward calls, without optimizer updates. Gradients are
    zeroed in place. Only after reducer warmup are addresses/participation frozen.

    All ranks must enter capture and replay in the same order. An external
    launcher timeout is required for failure containment; rank-local exceptions
    cannot safely repair a partially failed NCCL collective. No optimizer or
    process-group lifecycle is managed here. CUDA graphs are reconstructed on
    checkpoint resume, never serialized.
    """

    def __init__(self, adapter: PreparedDDPObjective, *, expected_active_names,
                 process_group=None, gradient_as_bucket_view=False, bucket_cap_mb=25):
        if not isinstance(adapter, PreparedDDPObjective):
            raise TypeError("DDPGraphTraining requires PreparedDDPObjective")
        if type(gradient_as_bucket_view) is not bool:
            raise TypeError("gradient_as_bucket_view must be boolean")
        if (isinstance(bucket_cap_mb, bool) or not isinstance(bucket_cap_mb, (int, float))
                or not math.isfinite(bucket_cap_mb) or bucket_cap_mb <= 0):
            raise ValueError("bucket_cap_mb must be finite and positive")
        adapter.validate_execution()
        self.adapter, self.model = adapter, adapter.model
        self.expected_active_names = _active_contract(self.model, expected_active_names)
        self.process_group = process_group
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_mb = float(bucket_cap_mb)
        self.ddp = self.stream = self.graph = self.graph_result = None
        self.gradient_addresses = None
        self.active_names = None
        self.warmup_backward_calls = self.capture_backward_calls = self.replay_calls = 0
        self._capture_started = False
        self._failed = False
        self._parameters = tuple(self.model.named_parameters())
        self._runner_contract = (id(adapter), id(self.model), id(process_group),
                                 self.expected_active_names, gradient_as_bucket_view, self.bucket_cap_mb)
        self._ddp_contract = None

    @property
    def device(self):
        return self.adapter.plan.batch.input_ids.device

    @property
    def metadata(self):
        return {"schema": "olmo-ddp-graph-training-v1", "execution": "actual-ddp-captured-backward",
                "world_size": self.adapter.world_size, "local_counts": self.adapter.counts,
                "global_counts": self.adapter.global_counts,
                "expected_active_names": self.expected_active_names,
                "active_names": self.active_names, "static_graph": True,
                "find_unused_parameters": False, "broadcast_buffers": False,
                "gradient_as_bucket_view": self.gradient_as_bucket_view,
                "bucket_cap_mb": self.bucket_cap_mb,
                "warmup_backward_calls": self.warmup_backward_calls,
                "capture_backward_calls": self.capture_backward_calls,
                "replay_calls": self.replay_calls, "failed": self._failed}

    def _addresses(self):
        return {n: None if p.grad is None else p.grad.data_ptr() for n, p in self._parameters}

    def _ddp_signature(self):
        if self.ddp is None:
            return None
        return (id(self.ddp), id(self.ddp.module), id(self.ddp.process_group),
                self.ddp.training, self.ddp.static_graph, self.ddp.find_unused_parameters,
                self.ddp.broadcast_buffers, self.ddp.gradient_as_bucket_view,
                self.ddp.require_backward_grad_sync)

    def validate_execution(self):
        if self._failed:
            raise RuntimeError("Distributed graph preparation/execution failed; recreate the process group and plan")
        if (id(self.adapter), id(self.model), id(self.process_group), self.expected_active_names,
                self.gradient_as_bucket_view, self.bucket_cap_mb) != self._runner_contract:
            raise ValueError("Distributed graph runner settings changed; prepare again")
        self.adapter.validate_execution()
        if self._ddp_contract is not None and self._ddp_signature() != self._ddp_contract:
            raise ValueError("DDP wrapper or reducer settings changed; prepare again")
        if self.gradient_addresses is not None and self._addresses() != self.gradient_addresses:
            raise ValueError("Persistent distributed gradient buffers changed; prepare again")

    def load_batch(self, batch):
        self.validate_execution()
        self.adapter.load_batch(batch)

    def _tensor_backward(self):
        for _, parameter in self._parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.adapter.plan.config.precision == "bf16_mixed", cache_enabled=False):
            result = self.ddp(self.adapter.plan.batch.input_ids)
        result["objective"].backward()
        return result

    def _validate_distributed_contract(self):
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("Initialize the NCCL process group before distributed graph capture")
        if dist.get_backend(self.process_group) != "nccl":
            raise ValueError("Distributed CUDA graph capture requires an NCCL process group")
        actual_world_size = dist.get_world_size(self.process_group)
        shared = {"world_size": self.adapter.world_size,
                  "counts": self.adapter.global_counts, "weights": self.adapter.plan.weights,
                  "mode": asdict(self.adapter.plan.mode), "nextlat": asdict(self.model.config),
                  "enabled": self.model.enabled, "gamma": self.model.gamma,
                  "training": asdict(self.adapter.plan.config),
                  "expected_active_names": self.expected_active_names,
                  "gradient_as_bucket_view": self.gradient_as_bucket_view,
                  "bucket_cap_mb": self.bucket_cap_mb,
                  "parameters": [(n, tuple(p.shape), str(p.dtype), p.requires_grad)
                                 for n, p in self._parameters]}
        records = [None] * actual_world_size
        dist.all_gather_object(records, shared, group=self.process_group)
        if any(record != shared for record in records):
            raise ValueError("Ranks disagree on static objective/model/reducer configuration")
        if actual_world_size != self.adapter.world_size:
            raise ValueError("Prepared world_size differs from the process group")
        # Counts are for one physical batch per rank, not accumulation. Agree on
        # supplied denominators first, so all ranks reject mismatches before
        # moving into a different collective sequence.
        local = torch.tensor([self.adapter.counts[t] for t in TERMS], dtype=torch.int64, device=self.device)
        dist.all_reduce(local, group=self.process_group)
        if local.cpu().tolist() != list(self.adapter._global_counts):
            raise ValueError("Global counts must equal all ranks' fixed local batch counts")

    def _freeze_gradients(self):
        active = tuple(n for n, p in self._parameters if p.grad is not None)
        if active != self.expected_active_names:
            raise ValueError("Warmup gradients differ from declared global parameter participation: "
                             f"missing={sorted(set(self.expected_active_names)-set(active))}, "
                             f"unexpected={sorted(set(active)-set(self.expected_active_names))}")
        self.active_names = active
        self.gradient_addresses = self._addresses()

    def prepare(self, *, warmup=11, phase_observer=None):
        """Construct and warm real DDP; permits eager Adam setup before capture."""
        if self.device.type != "cuda":
            raise ValueError("Distributed CUDA graph capture requires CUDA; no CPU fallback")
        if type(warmup) is not int or warmup < 11:
            raise ValueError("At least 11 DDP-enabled eager warmup iterations are required")
        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("phase_observer must be callable or None")
        if self.ddp is not None or self._capture_started:
            raise ValueError("Distributed preparation was already attempted; create a fresh wrapper and plan")
        self.validate_execution()
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("Set the rank's CUDA device before constructing distributed graphs")
        if any(p.grad is not None for _, p in self._parameters):
            raise ValueError("Start distributed capture at a cleared gradient boundary")
        try:
            with _preparation_phase(phase_observer, "distributed_contract"):
                self._validate_distributed_contract()
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with _preparation_phase(phase_observer, "ddp_construction"):
                with torch.cuda.stream(self.stream):
                    self.ddp = DistributedDataParallel(self.adapter, device_ids=[self.device.index],
                        output_device=self.device.index, process_group=self.process_group,
                        broadcast_buffers=False, find_unused_parameters=False, static_graph=True,
                        gradient_as_bucket_view=self.gradient_as_bucket_view, bucket_cap_mb=self.bucket_cap_mb)
                self._ddp_contract = self._ddp_signature()
            with _preparation_phase(phase_observer, "ddp_warmup"):
                recent_addresses = []
                with torch.cuda.stream(self.stream):
                    for _ in range(warmup):
                        self._tensor_backward()
                        self.warmup_backward_calls += 1
                        recent_addresses.append(self._addresses())
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
                torch.cuda.synchronize(self.device)
                if any(addresses != recent_addresses[-1] for addresses in recent_addresses[-3:]):
                    raise ValueError("DDP gradient addresses did not stabilize during warmup")
                self._freeze_gradients()
                self.validate_execution()
        except BaseException:
            self._failed = True
            raise

    def capture(self, *, warmup=11, release_transient_cache=False, phase_observer=None):
        if type(warmup) is not int or warmup < 11:
            raise ValueError("At least 11 DDP-enabled eager warmup iterations are required")
        if type(release_transient_cache) is not bool:
            raise TypeError("release_transient_cache must be boolean")
        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("phase_observer must be callable or None")
        if self._capture_started:
            raise ValueError("Distributed capture was already attempted; create a fresh wrapper and plan")
        self.validate_execution()
        if self.ddp is None:
            self.prepare(warmup=warmup, phase_observer=phase_observer)
        self.validate_execution()
        self._capture_started = True
        try:
            if release_transient_cache:
                with _preparation_phase(phase_observer, "transient_cleanup"):
                    gc.collect()
                    torch.cuda.empty_cache()
            with _preparation_phase(phase_observer, "ddp_capture"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    result = self._tensor_backward()
                torch.cuda.synchronize(self.device)
                self.graph, self.graph_result = graph, result
                self.capture_backward_calls += 1
                self.validate_execution()
        except BaseException:
            self._failed = True
            raise

    def backward(self, *, replay=True):
        """Overwrite gradients; caller reduces health and clips/steps afterwards.

        ``replay=False`` executes the identical actual-DDP prepared computation
        eagerly after preparation. It is useful for bounded same-candidate checks,
        but eager work beside a large graph pool can need extra device memory.
        """
        if type(replay) is not bool:
            raise TypeError("replay must be boolean")
        self.validate_execution()
        if replay and self.graph is None:
            raise ValueError("Capture the distributed graph before replay backward")
        if self.ddp is None:
            raise ValueError("Prepare DDP before eager backward")
        try:
            if replay:
                self.graph.replay()
                self.replay_calls += 1
                result = self.graph_result
            else:
                result = self._tensor_backward()
            self.validate_execution()
            return result
        except BaseException:
            self._failed = True
            raise
