"""Untimed exact checks without an eager backward beside a live CUDA graph.

This optional harness helper changes validation order only. Eager references
precede capture; the final changed-weight check consumes a fresh replay and
releases the graph before eager execution. Call that final check after profiles.
No optimizer update, parameter copy, or full GPU reference clone occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc

import torch

from scripts.olmo_f1_common import active_names


@dataclass(frozen=True)
class CPUReference:
    """Detached CPU values and tensor-free ownership/storage metadata."""

    losses: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    parameter_storage: tuple
    gradient_storage: tuple
    declared_names: frozenset[str]
    expected_names: frozenset[str]


def _tensor_storage(tensor):
    if tensor is None:
        return None
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device)


def _storage_contract(plan):
    parameters = tuple(plan.model.named_parameters())
    return (
        tuple((name, _tensor_storage(parameter), parameter.requires_grad)
              for name, parameter in parameters),
        tuple((name, _tensor_storage(parameter.grad)) for name, parameter in parameters),
    )


def _loss_tensors(result):
    # Preserve every key, including unexpected candidate sums, for exact key
    # ownership checks. Do not clone these graph-owned tensors on the GPU.
    return {f"pass{index}/{term}": value
            for index, loss in enumerate(result.pass_losses)
            for term, value in loss.sums.items()}


def _cpu_copy(tensor):
    # copy=True also makes independent references in the CPU orchestration tests.
    return tensor.detach().to(device="cpu", copy=True)


def _equal_cpu(tensor, reference):
    return (tensor.dtype == reference.dtype and tensor.shape == reference.shape
            and torch.equal(_cpu_copy(tensor), reference))


def _snapshot(plan, result, storage):
    losses = {name: _cpu_copy(value) for name, value in _loss_tensors(result).items()}
    gradients = {name: _cpu_copy(parameter.grad)
                 for name, parameter in plan.model.named_parameters()
                 if parameter.grad is not None}
    declared = frozenset(plan.active_names)
    expected = frozenset(active_names(plan.model, plan.mode))
    if not losses or not gradients or set(gradients) != declared or declared != expected:
        raise ValueError("CPU reference has invalid loss or gradient ownership")
    if _storage_contract(plan) != storage:
        raise ValueError("Parameter or persistent gradient storage changed during reference execution")
    return CPUReference(losses, gradients, *storage, declared, expected)


def _validate_reference(plan, reference):
    if not isinstance(reference, CPUReference):
        raise TypeError("Expected a CPUReference")
    if (not reference.losses or not reference.gradients
            or any(value.device.type != "cpu" or value.requires_grad
                   for value in (*reference.losses.values(), *reference.gradients.values()))):
        raise ValueError("Reference values must be detached nonempty CPU tensors")
    if (set(reference.gradients) != reference.declared_names
            or reference.declared_names != reference.expected_names):
        raise ValueError("CPU reference has invalid gradient ownership")
    if _storage_contract(plan) != (reference.parameter_storage, reference.gradient_storage):
        raise ValueError("Parameter or persistent gradient storage differs from the CPU reference")


def snapshot_eager_cpu(plan, batch):
    """Load a batch and snapshot eager losses/gradients before graph capture.

    The caller restores the desired capture batch after preparing references.
    References remain valid while capture and token changes leave weights and
    the persistent parameter/gradient buffers unchanged.
    """
    if plan.graph is not None:
        raise ValueError("Eager reference requires no live graph")
    plan.initialize_gradients()
    storage = _storage_contract(plan)
    plan.load_batch(batch)
    result = plan.backward(replay=False)
    try:
        return _snapshot(plan, result, storage)
    finally:
        del result


def _compare(plan, reference, result, name):
    losses = _loss_tensors(result)
    loss_names_match = set(losses) == set(reference.losses)
    loss_checks = {key: {"bitwise_equal": key in losses and key in reference.losses
                        and _equal_cpu(losses[key], reference.losses[key])}
                   for key in sorted(set(losses) | set(reference.losses))}
    parameters = dict(plan.model.named_parameters())
    gradient_names = {key for key, parameter in parameters.items() if parameter.grad is not None}
    gradient_checks = {key: {"bitwise_equal": key in gradient_names and key in reference.gradients
                            and _equal_cpu(parameters[key].grad, reference.gradients[key])}
                       for key in sorted(gradient_names | set(reference.gradients))}
    ownership = (gradient_names == set(reference.gradients) == set(plan.active_names)
                 == active_names(plan.model, plan.mode) == reference.declared_names
                 == reference.expected_names)
    storage_matches = _storage_contract(plan) == (reference.parameter_storage, reference.gradient_storage)
    exact = loss_names_match and all(row["bitwise_equal"]
                                    for row in (*loss_checks.values(), *gradient_checks.values()))
    return {"name": name, "passed": ownership and storage_matches and exact,
            "all_bitwise_equal": exact, "loss_names_match": loss_names_match,
            "ownership_matches": ownership, "storage_matches": storage_matches,
            "losses": loss_checks, "gradients": gradient_checks,
            "reference_storage": "CPU full gradients; one tensor streamed at a time; untimed"}


def replay_compare_cpu(plan, refs, name, *, replays=1):
    """Compare every replay against a saved eager reference for current tokens."""
    if type(replays) is not int or replays < 1:
        raise ValueError("Require at least one graph replay")
    if plan.graph is None:
        raise ValueError("Replay comparison requires a live graph")
    _validate_reference(plan, refs)
    checks = []
    for _ in range(replays):
        result = plan.backward(replay=True)
        try:
            checks.append(_compare(plan, refs, result, name))
        finally:
            del result
    # Checking every replay also catches a stale first replay followed by a
    # correct overwrite. Retain per-replay diagnostics on any failed gate.
    check = dict(checks[-1])
    for key in ("passed", "all_bitwise_equal", "loss_names_match", "ownership_matches", "storage_matches"):
        check[key] = all(row[key] for row in checks)
    check["replays_checked"] = replays
    if not check["passed"]:
        check["replay_checks"] = checks
    return check


def snapshot_replay_cpu(plan):
    """Snapshot one fresh replay at final weights, retaining no graph outputs."""
    if plan.graph is None:
        raise ValueError("Replay reference requires a live graph")
    plan.validate_execution()
    storage = _storage_contract(plan)
    result = plan.backward(replay=True)
    try:
        return _snapshot(plan, result, storage)
    finally:
        del result


def release_graph_then_compare_eager_cpu(plan, refs, name):
    """Terminal check after profiles: free graph storage, then compare eager.

    The caller must hold no other graph/output references. This helper owns no
    such reference on entry; ``snapshot_replay_cpu`` returns CPU tensors only.
    Model, optimizer and persistent gradient storage stay live throughout.
    """
    if plan.graph is None:
        raise ValueError("Terminal validation requires a live graph")
    _validate_reference(plan, refs)
    plan.validate_execution()
    torch.cuda.synchronize()
    plan.graph_result = None
    graph = plan.graph
    graph.reset()
    plan.graph = None
    del graph
    gc.collect()
    torch.cuda.empty_cache()
    if _storage_contract(plan) != (refs.parameter_storage, refs.gradient_storage):
        raise ValueError("Parameter or persistent gradient storage changed during graph release")
    plan.validate_execution()
    result = plan.backward(replay=False)
    try:
        check = _compare(plan, refs, result, name)
    finally:
        del result
    plan.validate_execution()
    check["graph_released_before_eager"] = True
    return check
