"""Opt-in native ZeRO-1 AdamW with explicit consolidated checkpoint support.

Parameters and gradients remain replicated. Native PyTorch partitions Adam
state and broadcasts updated parameter shards; there is no overlap hook,
parameter rebinding, offload, ZeRO-2, or optimizer CUDA graph here. The installed
native loader moves scalar Adam steps to CPU even for fused CUDA AdamW and
duplicates local state in the outer optimizer. Only that loader is replaced:
the local torch AdamW loader retains its own dtype/device policy.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from .distributed_checkpoint import (save_distributed_checkpoint, load_distributed_checkpoint,
                                     _coordinated)
from .lm_training import build_adamw, optimizer_ownership, optimizer_state_bytes, _plain

ZERO1_SCHEMA = "olmo-native-zero1-adamw-v1"


class FP32Zero1AdamW(ZeroRedundancyOptimizer):
    """Native partition/update/broadcast, with a safe global-to-local loader.

    Construct with ``build_zero1_adamw`` so the canonical global parameter
    groups, names and no-decay policy match replicated AdamW. Global state_dict
    requires a fresh collective consolidate_state_dict(to=...) as in PyTorch.
    Local state is ``optim.state``; outer ``state`` is intentionally empty.
    """

    def clear_consolidated_state(self):
        # Native consolidation caches full CPU moments on its target rank.
        # They are not needed for subsequent training after durable publication.
        self._all_state_dicts = []

    def step(self, closure=None, **kwargs):
        self.clear_consolidated_state()
        return super().step(closure=closure, **kwargs)

    def load_state_dict(self, state_dict):
        """Load only this rank's moments through the underlying AdamW loader.

        The caller's consolidated mapping remains unchanged. No global moments
        are materialized on the GPU, and no duplicated outer state is retained.
        Global group metadata/names remain global, as in native ZeRO-1; local
        Adam's integer parameter IDs refer only to its native shard.
        """
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"state", "param_groups"}:
            raise ValueError("Expected consolidated ZeRO-1 optimizer state")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Consolidated optimizer group count differs")
        global_ids = {}
        seen = set()
        for saved, current in zip(saved_groups, self.param_groups):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("Consolidated optimizer parameter count differs")
            if saved.get("param_names") != current.get("param_names"):
                raise ValueError("Consolidated optimizer parameter names/order differ")
            for identifier, parameter in zip(saved["params"], current["params"]):
                if identifier in seen:
                    raise ValueError("Consolidated optimizer has duplicate parameter IDs")
                seen.add(identifier)
                global_ids[id(parameter)] = identifier
        if not set(state_dict["state"]).issubset(seen):
            raise ValueError("Consolidated optimizer contains foreign state")
        template = self.optim.state_dict()
        local_state, local_groups = {}, []
        for saved, current, local in zip(saved_groups, self.optim.param_groups, template["param_groups"]):
            group = copy.deepcopy({key: value for key, value in saved.items() if key != "params"})
            group["params"] = list(local["params"])
            local_groups.append(group)
            for parameter, identifier in zip(current["params"], local["params"]):
                global_id = global_ids[id(parameter)]
                if global_id in state_dict["state"]:
                    value = state_dict["state"][global_id]
                    if not isinstance(value, Mapping):
                        raise ValueError("Consolidated optimizer state contains a missing/nonmapping shard")
                    local_state[identifier] = copy.deepcopy(value)
        self.optim.load_state_dict({"state": local_state, "param_groups": local_groups})
        self._sync_param_groups(saved_groups, self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)
        self.state.clear()
        self.clear_consolidated_state()
        for parameter, state in self.optim.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                value = state.get(name)
                if value is not None and (value.device != parameter.device or value.dtype != torch.float32):
                    raise ValueError("Restored local Adam moments must remain FP32 on the parameter device")
            step = state.get("step")
            if step is not None:
                group = next(group for group in self.optim.param_groups
                             if any(p is parameter for p in group["params"]))
                device = parameter.device if group.get("fused") or group.get("capturable") else torch.device("cpu")
                if step.device != device:
                    raise ValueError("Local Adam loader did not restore the required step-counter device")


def build_zero1_adamw(model, *, lr: float, betas=(.9, .95), eps=1e-8,
                      weight_decay=.1, fused=True, process_group=None):
    """Preserve existing canonical AdamW groups while sharding only state."""
    if not dist.is_initialized():
        raise RuntimeError("ZeRO-1 requires an initialized process group")
    if type(fused) is not bool:
        raise TypeError("fused must be boolean")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("ZeRO-1 preparation requires FP32 model parameters")
    reference = build_adamw(model, lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, fused=fused, foreach=False)
    groups = [{key: list(value) if key in ("params", "param_names") else value
               for key, value in group.items()} for group in reference.param_groups]
    optimizer = FP32Zero1AdamW(groups, optimizer_class=torch.optim.AdamW,
        process_group=process_group, parameters_as_bucket_view=False, overlap_with_ddp=False,
        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused, foreach=False)
    optimizer_ownership(model, optimizer)
    return optimizer


def _require_zero1(model, optimizer):
    if type(optimizer) is not FP32Zero1AdamW:
        raise TypeError("Expected the explicit FP32Zero1AdamW optimizer")
    if optimizer._overlap_with_ddp or optimizer.parameters_as_bucket_view:
        raise ValueError("ZeRO-1 overlap or parameter bucket views are outside this validated scope")
    optimizer_ownership(model, optimizer)
    if optimizer.state:
        raise ValueError("Outer ZeRO-1 optimizer unexpectedly retains duplicate state")


def zero1_state_inventory(model, optimizer):
    """Actual rank-local optimizer storage, separate from replicated tensors.

    No collectives are performed; gather these small inventories to check the
    partition union and sum bytes. Native partitioning is at parameter-tensor
    granularity, so unequal per-rank totals are possible.
    """
    _require_zero1(model, optimizer)
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    parameters = [parameter for group in optimizer.optim.param_groups for parameter in group["params"]]
    states = []
    for parameter, state in optimizer.optim.state.items():
        states.append({"name": by_id[id(parameter)],
            "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "device": str(value.device), "bytes": value.numel()*value.element_size()}
                        for name, value in state.items() if isinstance(value, torch.Tensor)}})
    return {"schema": ZERO1_SCHEMA, "rank": optimizer.rank, "world_size": optimizer.world_size,
            "local_owned_names": [by_id[id(parameter)] for parameter in parameters],
            "local_owned_numel": sum(parameter.numel() for parameter in parameters),
            "local_state_bytes_by_device": optimizer_state_bytes(optimizer.optim),
            "outer_state_bytes_by_device": optimizer_state_bytes(optimizer),
            "consolidation_cache_present": bool(optimizer._all_state_dicts), "states": states}


def zero1_checkpoint_configuration(model, optimizer, configuration):
    _require_zero1(model, optimizer)
    configuration = _plain(configuration)
    if not isinstance(configuration, dict) or "_zero1" in configuration:
        raise ValueError("Training configuration must be a mapping without reserved _zero1 key")
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return {**configuration, "_zero1": {"schema": ZERO1_SCHEMA,
        "optimizer": type(optimizer).__module__+"."+type(optimizer).__qualname__,
        "local_optimizer": "torch.optim.adamw.AdamW", "world_size": optimizer.world_size,
        "overlap_with_ddp": False, "parameters_as_bucket_view": False,
        "checkpoint_layout": "consolidated-global-optimizer-rank-zero",
        "partition": [{"name": by_id[id(parameter)], "rank": rank}
                      for parameter, rank in optimizer._param_to_rank.items()]}}


def save_zero1_checkpoint(directory, model, optimizer, *, configuration, group=None, **kwargs):
    """Collectively consolidate, save a complete checkpoint, release CPU cache.

    Uses the actual optimizer subclass descriptor and an explicit ZeRO-1 policy
    in checkpoint configuration. Per-rank RNG/cursors and same-world validation
    are provided by the distributed checkpoint helper. Consolidation can have a
    transient communication/memory peak; callers must release graph pools
    beforehand. Clearing gradients alone does not release a live graph's pool.
    """
    group = optimizer.process_group if group is None else group
    def prepare():
        if group is not optimizer.process_group:
            raise ValueError("Checkpoint group must match the ZeRO-1 process group")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Save ZeRO-1 only with cleared gradients at an update boundary")
        return zero1_checkpoint_configuration(model, optimizer, configuration)
    augmented = _coordinated("prepare ZeRO-1 consolidation", prepare, group)
    try:
        optimizer.consolidate_state_dict(to=0)
        # Native consolidation makes nonblocking device-to-host copies. Ensure
        # all retained CPU bytes are ready before rank zero serializes them.
        device = next(model.parameters()).device
        _coordinated("finish ZeRO-1 consolidation copies",
                     lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None, group)
        return save_distributed_checkpoint(directory, model, optimizer, configuration=augmented,
                                            group=group, **kwargs)
    finally:
        optimizer.clear_consolidated_state()


def load_zero1_checkpoint(directory, model, optimizer, *, configuration, group=None, **kwargs):
    """Restore the same world's canonical state and only this rank's Adam shard."""
    group = optimizer.process_group if group is None else group
    def prepare():
        if group is not optimizer.process_group:
            raise ValueError("Checkpoint group must match the ZeRO-1 process group")
        return zero1_checkpoint_configuration(model, optimizer, configuration)
    augmented = _coordinated("prepare ZeRO-1 restore", prepare, group)
    return load_distributed_checkpoint(directory, model, optimizer, configuration=augmented,
                                       group=group, **kwargs)
