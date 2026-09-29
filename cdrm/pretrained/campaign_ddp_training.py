"""Refillable campaign accumulation through real DDP forward/backward graphs.

There are two independent CUDA graphs: local ``no_sync`` forward/backward and
the final synchronized forward/backward (including NCCL reducer operations).
Neither graph clears gradients. A logical update clears once, replays the local
graph M-1 times, and then the synchronized graph once. Optimizer/checkpoint/token
clocks stay outside capture. CPU/Gloo eager execution is an explicit diagnostic;
it does not qualify CUDA/NCCL capture.

All ranks must invoke methods in the same order. An external launcher timeout
must terminate the whole job after a fault inside collective execution; a local
exception cannot repair an in-flight NCCL collective. Reconstruct on restart.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .campaign_training import CampaignObjective
from .ddp_training import CoordinatedUpdateError
from .distributed_training import sum_objective_counts
from .lm_training import TERMS, TrainingCounters, _plain, optimizer_ownership
from .static_training import _preparation_phase


class CampaignDDPGraphTraining:
    """Actual DDP with stable inputs/gradients and global per-term denominators.

    Dense campaign objectives must participate in every trainable parameter,
    including empty local slots. Every positive objective weight needs positive
    global update counts. Physical B/T and model participation remain fixed;
    number of accumulation slots, masks, jitter and denominators may change.
    Bucket views and shared graph pools are intentionally not enabled.
    """

    def __init__(self, adapter: CampaignObjective, *, process_group=None, bucket_cap_mb=25):
        if not isinstance(adapter, CampaignObjective):
            raise TypeError("CampaignDDPGraphTraining requires CampaignObjective")
        if (isinstance(bucket_cap_mb, bool) or not isinstance(bucket_cap_mb, (int, float))
                or not math.isfinite(bucket_cap_mb) or bucket_cap_mb <= 0):
            raise ValueError("bucket_cap_mb must be finite and positive")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Initialize the process group before constructing a campaign DDP runner")
        self.adapter, self.model = adapter, adapter.model
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.bucket_cap_mb = float(bucket_cap_mb)
        self.ddp = self.stream = None
        self.local_graph = self.sync_graph = None
        self.local_result = self.sync_result = None
        self.gradient_addresses = self.active_names = None
        self.warmup_backward_calls = self.capture_backward_calls = self.replay_calls = 0
        self.local_replay_calls = self.sync_replay_calls = 0
        self._capture_started = self._failed = self._checkpoint_open = False
        self._at_update_boundary = True
        self._pending = None
        self._parameters = tuple(self.model.named_parameters())
        self._runner_contract = (id(adapter), id(self.model), id(process_group),
                                 self.world_size, self.bucket_cap_mb)
        self._ddp_contract = None

    @property
    def device(self):
        return self.adapter.device

    @property
    def metadata(self):
        return {"schema": "olmo-campaign-ddp-graph-training-v1",
                "execution": "actual-ddp-two-captured-backwards",
                "world_size": self.world_size, "global_counts": dict(self.adapter.global_counts),
                "local_counts": dict(self.adapter.counts), "active_names": self.active_names,
                "static_graph": True, "find_unused_parameters": False,
                "broadcast_buffers": False, "gradient_as_bucket_view": False,
                "bucket_cap_mb": self.bucket_cap_mb, "separate_graph_pools": True,
                "warmup_backward_calls": self.warmup_backward_calls,
                "capture_backward_calls": self.capture_backward_calls,
                "replay_calls": self.replay_calls, "local_replay_calls": self.local_replay_calls,
                "sync_replay_calls": self.sync_replay_calls, "failed": self._failed}

    def _gather(self, value):
        result = [None] * self.world_size
        dist.all_gather_object(result, value, group=self.process_group)
        return result

    def _coordinate_error(self, error, phase):
        errors = self._gather(error)
        failures = [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value is not None]
        if failures:
            raise CoordinatedUpdateError(f"{phase}: " + "; ".join(failures))

    def _same(self, value, phase):
        values = self._gather(value)
        if any(other != values[0] for other in values[1:]):
            raise CoordinatedUpdateError(f"{phase}: rank configurations disagree")

    def _addresses(self):
        return {n: None if p.grad is None else p.grad.data_ptr() for n, p in self._parameters}

    def _ddp_signature(self):
        if self.ddp is None:
            return None
        return (id(self.ddp), id(self.ddp.module), id(self.ddp.process_group), self.ddp.training,
                self.ddp.static_graph, self.ddp.find_unused_parameters, self.ddp.broadcast_buffers,
                self.ddp.gradient_as_bucket_view, self.ddp.require_backward_grad_sync)

    def validate_execution(self):
        if self._failed:
            raise RuntimeError("Failed campaign DDP runner requires a fresh process group and runner")
        if self._checkpoint_open:
            raise RuntimeError("Campaign execution is disabled during checkpoint publication")
        if self._runner_contract != (id(self.adapter), id(self.model), id(self.process_group),
                                     self.world_size, self.bucket_cap_mb):
            raise ValueError("Campaign distributed runner settings changed")
        self.adapter.validate_execution()
        if self._ddp_contract is not None and self._ddp_signature() != self._ddp_contract:
            raise ValueError("Campaign DDP wrapper/reducer settings changed")
        if self.gradient_addresses is not None and self._addresses() != self.gradient_addresses:
            raise ValueError("Campaign persistent distributed gradient storage changed")

    def zero_grad(self):
        for _, p in self._parameters:
            if p.grad is not None:
                p.grad.zero_()

    def _invalidate(self):
        self.zero_grad()
        self._pending = None
        self._failed = True

    @contextmanager
    def _execution_stream(self):
        if self.stream is None:
            yield
        else:
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                yield
            current.wait_stream(self.stream)

    def _tensor_backward(self, *, synchronize):
        # no_sync must contain the DDP forward as well as its backward. Merely
        # surrounding replay cannot remove collectives from an existing graph.
        with (nullcontext() if synchronize else self.ddp.no_sync()):
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                               enabled=self.adapter.config.precision == "bf16_mixed", cache_enabled=False):
                result = self.ddp(self.adapter.batch.input_ids)
            result["objective"].backward()
        return {**result, "objective": result["objective"].detach()}

    def _prepare_contract(self, warmup):
        error = None
        try:
            self.validate_execution()
            if type(warmup) is not int or warmup < 11:
                raise ValueError("At least eleven synchronized DDP warmup calls are required")
            if self.ddp is not None or self._capture_started:
                raise ValueError("Campaign DDP preparation already attempted")
            if self.adapter.world_size != self.world_size:
                raise ValueError("Adapter world_size differs from process group")
            expected_backend = "nccl" if self.device.type == "cuda" else "gloo"
            if self.device.type not in ("cpu", "cuda") or dist.get_backend(self.process_group) != expected_backend:
                raise ValueError("Campaign CUDA requires NCCL; CPU diagnostics require Gloo")
            if self.device.type == "cuda" and torch.cuda.current_device() != self.device.index:
                raise ValueError("Set this rank's CUDA device before DDP preparation")
            if any(p.grad is not None for _, p in self._parameters):
                raise ValueError("Prepare at a cleared gradient boundary")
            contract = {"shape": tuple(self.adapter.batch.input_ids.shape), "world_size": self.world_size,
                        "counts": self.adapter.global_counts, "weights": self.adapter.weights,
                        "mode": asdict(self.adapter.mode), "config": asdict(self.adapter.config),
                        "nextlat": self.model.config.to_dict(), "enabled": self.model.enabled,
                        "gamma": self.model.gamma, "policy": self.model.pass_loss_policy,
                        "bucket_cap_mb": self.bucket_cap_mb, "warmup": warmup,
                        "parameters": [(n, tuple(p.shape), str(p.dtype), p.requires_grad)
                                       for n, p in self._parameters]}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "DDP preparation preflight")
        self._same(contract, "DDP preparation contract")

    def prepare(self, *, warmup=11, phase_observer=None):
        """Warm synchronized DDP first, then representative no_sync accumulation."""
        try:
            with _preparation_phase(phase_observer, "distributed_contract"):
                self._prepare_contract(warmup)
            if self.device.type == "cuda":
                self.stream = torch.cuda.Stream(device=self.device)
            with _preparation_phase(phase_observer, "ddp_construction"), self._execution_stream():
                ids = [self.device.index] if self.device.type == "cuda" else None
                self.ddp = DistributedDataParallel(self.adapter, device_ids=ids,
                    output_device=self.device.index if ids else None, process_group=self.process_group,
                    broadcast_buffers=False, find_unused_parameters=False, static_graph=True,
                    gradient_as_bucket_view=False, bucket_cap_mb=self.bucket_cap_mb)
                self._ddp_contract = self._ddp_signature()
            with _preparation_phase(phase_observer, "ddp_warmup"), self._execution_stream():
                addresses = []
                # Initial static_graph no_sync backward is unsupported by the
                # reducer. Establish synchronized iteration state first.
                for _ in range(warmup):
                    self.zero_grad()
                    self._tensor_backward(synchronize=True)
                    self.warmup_backward_calls += 1
                    addresses.append(self._addresses())
                for _ in range(3):
                    self.zero_grad()
                    for sync in (False, False, True):
                        self._tensor_backward(synchronize=sync)
                        self.warmup_backward_calls += 1
                        addresses.append(self._addresses())
                active = tuple(n for n, p in self._parameters if p.grad is not None)
                expected = tuple(n for n, p in self._parameters if p.requires_grad)
                error = None
                if active != expected:
                    error = f"Warmup parameter participation differs: missing={sorted(set(expected)-set(active))}"
                elif any(value != addresses[-1] for value in addresses[-9:]):
                    error = "DDP gradient addresses did not stabilize across accumulated warmup"
                self._coordinate_error(error, "DDP warmup ownership")
                self.active_names, self.gradient_addresses = active, self._addresses()
                self.zero_grad()
            self.validate_execution()
        except BaseException:
            self._invalidate()
            raise

    def capture(self, *, warmup=11, release_transient_cache=False, phase_observer=None):
        """Capture separate local and synchronized graphs without gradient zeros."""
        error = None
        try:
            self.validate_execution()
            if self.device.type != "cuda":
                raise ValueError("Campaign DDP CUDA graph requires CUDA/NCCL; no CPU fallback")
            if type(warmup) is not int or warmup < 11:
                raise ValueError("At least eleven synchronized DDP warmup calls are required")
            if type(release_transient_cache) is not bool:
                raise TypeError("release_transient_cache must be boolean")
            if self._capture_started or not self._at_update_boundary:
                raise ValueError("Capture needs a fresh graph at a completed/discarded update boundary")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "capture preflight")
        self._same((warmup, release_transient_cache, self.ddp is not None), "capture contract")
        try:
            if self.ddp is None:
                self.prepare(warmup=warmup, phase_observer=phase_observer)
            self._capture_started = True
            if release_transient_cache:
                gc.collect()
                torch.cuda.empty_cache()
            for name, synchronize in (("local", False), ("sync", True)):
                self.zero_grad()
                with _preparation_phase(phase_observer, f"ddp_{name}_capture"):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        result = self._tensor_backward(synchronize=synchronize)
                    torch.cuda.synchronize(self.device)
                    setattr(self, f"{name}_graph", graph)
                    setattr(self, f"{name}_result", result)
                    self.capture_backward_calls += 1
                    self.zero_grad()
                    self.validate_execution()
        except BaseException:
            self._invalidate()
            raise

    def _update_preflight(self, microbatches, feedback_noises, replay):
        error = None
        try:
            self.validate_execution()
            if type(replay) is not bool:
                raise TypeError("replay must be boolean")
            if self.ddp is None:
                raise ValueError("Prepare DDP before eager backward")
            if replay and (self.local_graph is None or self.sync_graph is None):
                raise ValueError("Capture both campaign graphs before replay")
            batches = list(microbatches)
            if not batches:
                raise ValueError("At least one physical microbatch is required")
            noises = [None] * len(batches) if feedback_noises is None else list(feedback_noises)
            if len(noises) != len(batches):
                raise ValueError("One feedback-noise tuple is required per physical microbatch")
            local_counts = sum_objective_counts([self.model.counts(batch) for batch in batches])
            local_tokens = sum(int(batch.valid_mask.sum()) for batch in batches)
            local_documents = sum(int(batch.valid_mask.any(-1).sum()) for batch in batches)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "update preflight")
        self._same((len(batches), replay), "update accumulation contract")
        values = torch.tensor([local_counts[t] for t in TERMS] + [len(batches), local_documents, local_tokens],
                              device=self.device, dtype=torch.int64)
        dist.all_reduce(values, group=self.process_group)
        counts_and_stats = values.tolist()
        counts = dict(zip(TERMS, counts_and_stats[:3]))
        error = None
        try:
            checked = [self.adapter.validate_batch(batch, feedback_noise=noise, global_counts=counts)
                       for batch, noise in zip(batches, noises)]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "all-microbatch input preflight")
        return checked, counts, counts_and_stats[3:]

    def _healthy(self, flag, phase):
        flag = flag.to(device=self.device, dtype=torch.int32).reshape(())
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(flag):
            raise CoordinatedUpdateError(f"{phase}: a rank reported nonfinite values")

    def backward(self, microbatches, *, feedback_noises=None, replay=True):
        """Replace diagnostic gradients with one global update, without Adam.

        All-rank validation covers every slot before input/gradient mutation.
        Noise sources must remain unchanged until the synchronous call returns.
        Counters report global tokens/documents/microbatches, not per-rank values.
        """
        try:
            checked, counts, stats = self._update_preflight(microbatches, feedback_noises, replay)
            self._at_update_boundary = False
            totals = torch.zeros(len(TERMS), device=self.device, dtype=torch.float64)
            with self._execution_stream():
                self.zero_grad()
                for index, (cpu, noise, _) in enumerate(checked):
                    self.adapter.load_batch(cpu, feedback_noise=noise, global_counts=counts)
                    synchronize = index + 1 == len(checked)
                    if replay:
                        graph = self.sync_graph if synchronize else self.local_graph
                        graph.replay()
                        result = self.sync_result if synchronize else self.local_result
                        self.replay_calls += 1
                        self.sync_replay_calls += int(synchronize)
                        self.local_replay_calls += int(not synchronize)
                    else:
                        result = self._tensor_backward(synchronize=synchronize)
                    totals.add_(torch.stack([result["loss_sums"][t] for t in TERMS]).to(torch.float64))
            self.validate_execution()
            health = torch.stack([torch.isfinite(p.grad).all() for _, p in self._parameters
                                  if p.grad is not None]).all() & torch.isfinite(totals).all()
            self._healthy(health, "global gradient/loss health")
            dist.all_reduce(totals, group=self.process_group)
            self._healthy(torch.isfinite(totals).all(), "reduced metrics health")
            sums = dict(zip(TERMS, totals.cpu().tolist()))
            result = {"loss_sums": sums, "counts": counts,
                    "objective": sum(sums[t] * self.adapter.weights[t] / counts[t] for t in TERMS if counts[t]),
                    "microbatches": stats[0], "documents": stats[1], "input_tokens": stats[2],
                    "local_microbatches": len(checked), "world_size": self.world_size}
            self._pending = result
            return result
        except BaseException:
            self._invalidate()
            raise

    def discard_backward(self):
        error = None
        try:
            self.validate_execution()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "discard preflight")
        self.zero_grad()
        self._pending = None
        self._at_update_boundary = True

    @contextmanager
    def checkpoint_boundary(self):
        """Publish canonical state with grad=None, then restore captured storage.

        All ranks enter this context. The checkpoint writer owns durable atomic
        publication and its own rank coordination. Loading is not allowed here;
        fresh-process resume rebuilds the adapter/DDP/graphs after state load.
        """
        error = None
        try:
            self.validate_execution()
            if not self._at_update_boundary or any(p.grad is not None and bool(p.grad.any())
                                                   for _, p in self._parameters):
                raise ValueError("Checkpoint requires a completed/discarded update boundary")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._coordinate_error(error, "checkpoint boundary")
        gradients = [(p, p.grad) for _, p in self._parameters]
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

    def step(self, result, optimizer, *, scheduler=None, counters=None):
        """Clip/step one pending backward, after optional read-only inspection."""
        try:
            error = None
            try:
                self.validate_execution()
                if result is not self._pending or result is None:
                    raise ValueError("Unknown or consumed pending backward result")
                ownership = optimizer_ownership(self.model, optimizer)
                counters = TrainingCounters() if counters is None else counters
                if not isinstance(counters, TrainingCounters):
                    raise TypeError("counters must be TrainingCounters")
                TrainingCounters(**asdict(counters))
                contract = {"counters": asdict(counters), "ownership": ownership,
                            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                            "groups": [_plain({k: v for k, v in g.items() if k != "params"})
                                       for g in optimizer.param_groups],
                            "scheduler": None if scheduler is None else _plain(scheduler.state_dict())}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._coordinate_error(error, "optimizer preflight")
            self._same(contract, "optimizer contract")
            error = None
            try:
                if scheduler is not None and hasattr(scheduler, "validate_next_update"):
                    scheduler.validate_next_update(result["input_tokens"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._coordinate_error(error, "scheduler update preflight")
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                float("inf") if self.adapter.config.max_grad_norm is None else self.adapter.config.max_grad_norm,
                error_if_nonfinite=False, foreach=False)
            self._healthy(torch.isfinite(norm), "clipped gradient norm")
            rates = [g["lr"] for g in optimizer.param_groups]
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
            self.zero_grad()
            self._pending = None
            self._at_update_boundary = True
            return {**result, "gradient_norm_before_clip": float(norm), "lr_used": rates,
                    "lr_next": [g["lr"] for g in optimizer.param_groups], "counters": asdict(counters)}
        except BaseException:
            self._invalidate()
            raise

    def optimizer_step(self, optimizer, microbatches, *, feedback_noises=None, replay=True,
                       scheduler=None, counters=None):
        """Reduce, clip once, step once, and advance global token accounting once."""
        result = self.backward(microbatches, feedback_noises=feedback_noises, replay=replay)
        return self.step(result, optimizer, scheduler=scheduler, counters=counters)
