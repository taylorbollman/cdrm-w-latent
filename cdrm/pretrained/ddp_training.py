"""Conservative eager DDP updates for the canonical FBT/RT/NextLat objective.

The caller owns process-group initialization, the canonical model/optimizer,
checkpoints and process failure handling. This runner uses real DDP forward and
``no_sync``; it never differentiates ``ddp.module.model`` behind DDP's back.
No CUDA-graph, sharding or custom communication-hook support is implied.

Loss/preflight/gradient failures are coordinated while every rank can still
reach the same collective. Arbitrary CUDA faults, OOMs or exceptions *inside*
DDP forward/backward can strand an in-flight collective; the launcher must
terminate the whole job in that case. A failed runner cannot be reused.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .distributed_training import ObjectiveForwardAdapter, sum_objective_counts
from .fbt_training import FBTNextLatLM
from .lm_training import (LMTrainingConfig, TERMS, TrainingCounters,
                          _batch_statistics, _plain, optimizer_ownership,
                          optimizer_state_bytes)


class CoordinatedUpdateError(RuntimeError):
    """All participating ranks rejected an update before optimizer.step."""


@dataclass(frozen=True)
class DDPBackwardResult:
    """Reduced unclipped gradients remain on the canonical model.

    The record is valid only for the trainer that returned it, and only until
    its single ``step`` or ``discard``. Counts/metrics/counters are global over
    every rank and every microbatch, counting input tokens once, not FBT passes.
    """
    global_counts: dict[str, int]
    loss_sums: dict[str, float]
    objective_weights: dict[str, float]
    global_microbatches: int
    global_documents: int
    global_input_tokens: int
    local_microbatches: int
    config: LMTrainingConfig


class EagerDDPTrainer:
    """Default-average DDP, dynamic unused parameters, FP32 persistent state.

    All ranks must invoke methods in the same order with the same number of
    accumulation microbatches. Masks/counts and numbers of rows can differ.
    Parameters unused globally retain ``grad=None`` and hence neither receive
    invented Adam updates nor decoupled weight decay. The caller must not
    mutate the wrapper/reducer or install a nonstandard communication hook.
    """

    def __init__(self, model: FBTNextLatLM, *, process_group=None,
                 device_ids=None, bucket_cap_mb: float = 25):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("EagerDDPTrainer requires an initialized process group")
        if not isinstance(model, FBTNextLatLM):
            raise TypeError("Expected canonical FBTNextLatLM")
        self.group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.model = model
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError("Model has no parameters")
        self.device = parameters[0].device
        if any(p.device != self.device or p.dtype != torch.float32 for p in parameters):
            raise ValueError("All canonical model parameters must be FP32 on one device")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU/Gloo diagnostics or CUDA execution is supported")
        if not math.isfinite(bucket_cap_mb) or bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive and finite")
        if device_ids is None and self.device.type == "cuda":
            device_ids = [self.device.index]
        self.ddp = DistributedDataParallel(ObjectiveForwardAdapter(model),
            process_group=process_group, device_ids=device_ids,
            broadcast_buffers=False, find_unused_parameters=True,
            gradient_as_bucket_view=False, bucket_cap_mb=bucket_cap_mb,
            static_graph=False)
        self._pending: DDPBackwardResult | None = None
        self._failed = False

    def _gather(self, value):
        records = [None] * self.world_size
        dist.all_gather_object(records, value, group=self.group)
        return records

    def _coordinate_error(self, error: str | None, phase: str):
        records = self._gather(error)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(records)
                    if message is not None]
        if failures:
            raise CoordinatedUpdateError(f"{phase}: " + "; ".join(failures))

    def _same(self, value, phase: str):
        values = self._gather(value)
        if any(item != values[0] for item in values[1:]):
            raise CoordinatedUpdateError(f"{phase}: rank configurations disagree")

    def _healthy(self, healthy: torch.Tensor, phase: str):
        flag = healthy.to(device=self.device, dtype=torch.int32).reshape(())
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        if not bool(flag):
            raise CoordinatedUpdateError(f"{phase}: a rank reported nonfinite values")

    def _invalidate(self):
        self.model.zero_grad(set_to_none=True)
        self._pending = None
        self._failed = True

    def assert_update_boundary(self):
        """Local assertion useful before coordinated checkpointing."""
        if self._failed:
            raise RuntimeError("Failed DDP runner must be reconstructed")
        if self._pending is not None or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Operation requires a cleared optimizer-update boundary")

    def _gradient_health(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        presence = torch.tensor([p.grad is not None for p in parameters],
                                device=self.device, dtype=torch.int32)
        minimum, maximum = presence.clone(), presence.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=self.group)
        if not torch.equal(minimum, maximum):
            raise CoordinatedUpdateError("Reduced gradient ownership differs across ranks")
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise CoordinatedUpdateError("No globally active parameter gradients")
        if any(g.is_sparse for g in gradients):
            raise CoordinatedUpdateError("Sparse gradients are unsupported")
        self._healthy(torch.stack([torch.isfinite(g).all() for g in gradients]).all(),
                      "gradient health")

    def backward(self, microbatches: Sequence, *,
                 config: LMTrainingConfig = LMTrainingConfig(),
                 backbone_kwargs: Mapping | None = None) -> DDPBackwardResult:
        """Reduce global raw gradients without clipping or updating any state.

        Each term differentiates ``world_size * local_sum / global_count``;
        DDP's default average supplies the matching global mean. Counts are
        independent for CE, latent and KL; pass weighting stays canonical.
        There is no additional division by accumulation length.
        """
        try:
            error = None
            try:
                self.assert_update_boundary()
                microbatches = list(microbatches)
                if not microbatches:
                    raise ValueError("At least one microbatch is required per rank")
                if not isinstance(config, LMTrainingConfig):
                    raise TypeError("config must be LMTrainingConfig")
                if config.precision == "bf16_mixed" and self.device.type != "cuda":
                    raise ValueError("BF16 mixed execution requires CUDA")
                kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
                weights = dict(self.model.objective_weights())
                if set(weights) != set(TERMS) or any(not math.isfinite(w) or w < 0 for w in weights.values()):
                    raise ValueError("Objective weights must be finite and nonnegative")
                local_counts = sum_objective_counts([self.model.counts(batch) for batch in microbatches])
                stats = [_batch_statistics(batch) for batch in microbatches]
                for batch in microbatches:
                    if batch.input_ids.device != self.device:
                        raise ValueError("Every batch must be on the model device")
                contract = {"microbatches": len(microbatches), "config": asdict(config),
                            "weights": weights, "backbone_kwargs": _plain(kwargs),
                            "objective_config": {"nextlat": self.model.config.to_dict(),
                                                 "enabled": self.model.enabled,
                                                 "gamma": self.model.gamma}}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._coordinate_error(error, "update preflight")
            self._same(contract, "update contract")
            values = [local_counts[t] for t in TERMS] + [len(microbatches),
                     sum(x[0] for x in stats), sum(x[1] for x in stats)]
            totals = torch.tensor(values, device=self.device, dtype=torch.int64)
            dist.all_reduce(totals, group=self.group)
            values = totals.tolist()
            counts = dict(zip(TERMS, values[:3]))
            if not any(counts[t] and weights[t] for t in TERMS):
                raise CoordinatedUpdateError("Update has no globally valid positively weighted objective")
            if any(counts[t] and not weights[t] for t in TERMS):
                raise CoordinatedUpdateError("Disabled objectives must have zero global counts")
            self.ddp.train()
            self.model.zero_grad(set_to_none=True)
            loss_totals = torch.zeros(len(TERMS), device=self.device, dtype=torch.float64)
            for index, batch in enumerate(microbatches):
                context = self.ddp.no_sync() if index + 1 < len(microbatches) else nullcontext()
                with context:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16,
                         enabled=config.precision == "bf16_mixed", cache_enabled=False):
                        result = self.ddp(batch, global_counts=counts, world_size=self.world_size,
                                          backbone_kwargs=kwargs)
                        objective = result["objective"]
                    local_sums = torch.stack([result["loss_sums"][t] for t in TERMS])
                    self._healthy(torch.isfinite(local_sums).all() & torch.isfinite(objective.detach()),
                                  f"microbatch {index} loss")
                    self._coordinate_error(None if objective.requires_grad else "Objective has no gradient",
                                           f"microbatch {index} objective")
                    loss_totals += local_sums.to(torch.float64)
                    objective.backward()
            self._gradient_health()
            dist.all_reduce(loss_totals, group=self.group)
            self._healthy(torch.isfinite(loss_totals).all(), "global metrics")
            record = DDPBackwardResult(counts, dict(zip(TERMS, loss_totals.tolist())), weights,
                values[3], values[4], values[5], len(microbatches), config)
            self._pending = record
            return record
        except BaseException:
            self._invalidate()
            raise

    def discard(self, result: DDPBackwardResult):
        """Collectively finish a successful backward-only diagnostic."""
        try:
            self._coordinate_error(None if result is self._pending else "Unknown pending result", "discard")
            self.model.zero_grad(set_to_none=True)
            self._pending = None
        except BaseException:
            self._invalidate()
            raise

    def step(self, result: DDPBackwardResult, optimizer, *, scheduler=None,
             counters: TrainingCounters | None = None) -> dict:
        """Clip the completed reduction once, then step every rank.

        Post-backward inspection may read gradients but must not mutate them.
        Health is rechecked here to catch accidental/nonfinite edits before any
        optimizer step. Failures *during* optimizer/scheduler stepping cannot be
        transactionally rolled back; reload the preceding durable checkpoint.
        """
        try:
            error = None
            try:
                if self._failed or result is not self._pending:
                    raise ValueError("Unknown or consumed pending backward result")
                ownership = optimizer_ownership(self.model, optimizer)
                counters = TrainingCounters() if counters is None else counters
                if not isinstance(counters, TrainingCounters):
                    raise TypeError("counters must be TrainingCounters")
                contract = {"counters": asdict(counters), "ownership": ownership,
                            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                            "groups": [_plain({k: v for k, v in group.items() if k != "params"})
                                       for group in optimizer.param_groups],
                            "scheduler": None if scheduler is None else _plain(scheduler.state_dict())}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._coordinate_error(error, "optimizer preflight")
            self._same(contract, "optimizer contract")
            self._gradient_health()
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                float("inf") if result.config.max_grad_norm is None else result.config.max_grad_norm,
                error_if_nonfinite=False, foreach=False)
            self._healthy(torch.isfinite(norm), "gradient norm")
            rates = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.model.zero_grad(set_to_none=True)
            self._pending = None
            counters.optimizer_updates += 1
            counters.microbatches += result.global_microbatches
            counters.documents += result.global_documents
            counters.input_tokens += result.global_input_tokens
            counters.ce_positions += result.global_counts["ce"]
            counters.latent_pairs += result.global_counts["latent"]
            counters.kl_triples += result.global_counts["kl"]
            means = {t: result.loss_sums[t] / result.global_counts[t] if result.global_counts[t] else 0.
                     for t in TERMS}
            return {"schema": "olmo-ddp-eager-step-v1", "update_completed": True,
                "world_size": self.world_size, "loss_sums": result.loss_sums,
                "counts": result.global_counts, "loss_means": means,
                "objective_weights": result.objective_weights,
                "objective": sum(result.objective_weights[t] * means[t] for t in TERMS),
                "gradient_norm_before_clip": float(norm), "max_grad_norm": result.config.max_grad_norm,
                "lr_used": rates, "lr_next": [g["lr"] for g in optimizer.param_groups],
                "optimizer_state_bytes_by_device": optimizer_state_bytes(optimizer),
                "counters": asdict(counters)}
        except BaseException:
            self._invalidate()
            raise

    def optimizer_step(self, optimizer, microbatches: Sequence, *,
                       config: LMTrainingConfig = LMTrainingConfig(), backbone_kwargs=None,
                       scheduler=None, counters: TrainingCounters | None = None) -> dict:
        result = self.backward(microbatches, config=config, backbone_kwargs=backbone_kwargs)
        return self.step(result, optimizer, scheduler=scheduler, counters=counters)
