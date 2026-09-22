"""Read-only parameter and operator accounting for bounded F1 diagnostics.

These helpers do not select kernels, change model math or install hooks. Activity
and inference membership are supplied by the caller: neither can be inferred
from ``requires_grad`` or optimizer ownership. Profiler times are observations
of recorded operators, not synchronized step timings or FLOP measurements.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from torch import nn


def _registered(module: nn.Module, *, buffers: bool = False):
    iterator = module.named_buffers if buffers else module.named_parameters
    tensors, records, aliases = {}, {}, {}
    for name, tensor in iterator(remove_duplicate=False):
        identity = id(tensor)
        aliases[name] = identity
        if identity not in tensors:
            tensors[identity] = tensor
            records[identity] = {
                "name": name, "aliases": [], "shape": list(tensor.shape),
                "dtype": str(tensor.dtype), "device": str(tensor.device),
                "requires_grad": bool(tensor.requires_grad),
                "parameter_count": int(tensor.numel()),
                "parameter_bytes": int(tensor.numel() * tensor.element_size()),
            }
        records[identity]["aliases"].append(name)
    return tensors, records, aliases


def _selected_ids(names, aliases, *, label):
    if names is None:
        return None
    if isinstance(names, str):
        raise ValueError(f"{label} must be an iterable of parameter names, not a string")
    selected = set()
    for name in names:
        if not isinstance(name, str) or name not in aliases:
            raise ValueError(f"Unknown {label} parameter name: {name!r}")
        selected.add(aliases[name])
    return selected


def _scope(identities, records):
    if identities is None:
        return None
    rows = [record for identity, record in records.items() if identity in identities]
    return {
        "tensor_count": len(rows),
        "parameter_count": sum(row["parameter_count"] for row in rows),
        "parameter_bytes": sum(row["parameter_bytes"] for row in rows),
        "names": [row["name"] for row in rows],
    }


def _resident_storage(tensors, records):
    """Count allocated full storages once, including storage outside a view.

    Distinct Parameter objects can share storage and still be distinct optimizer
    objects. Parameter-element counts and allocation-byte counts therefore have
    deliberately separate meanings. Meta tensors have no resident allocation.
    """
    seen, by_device, unavailable, shared = {}, {}, [], []
    for identity, tensor in tensors.items():
        name = records[identity]["name"]
        if tensor.device.type == "meta":
            unavailable.append(name)
            continue
        try:
            storage = tensor.untyped_storage()
            size = int(storage.nbytes())
            key = (str(tensor.device), int(storage.data_ptr()), size)
        except (NotImplementedError, RuntimeError):
            unavailable.append(name)
            continue
        # Zero-length allocations carry no bytes; their pointer need not be unique.
        if key in seen:
            if size:
                shared.append({"name": name, "shares_storage_with": seen[key]})
            continue
        seen[key] = name
        by_device[key[0]] = by_device.get(key[0], 0) + size
    return {
        "known_allocated_bytes": sum(by_device.values()),
        "by_device_bytes": by_device,
        "complete": not unavailable,
        "unavailable_names": unavailable,
        "shared_storage_objects": shared,
        "scope": "Unique full parameter storages; excludes gradients, optimizer state, buffers and activations.",
    }


def parameter_accounting(
    model: nn.Module,
    *,
    optimizer=None,
    active_parameter_names: Iterable[str] | None = None,
    inference_parameter_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Account for tied/shared tensors and distinct execution/ownership scopes.

    ``active_parameter_names`` should come from observed backward hooks for the
    recorded diagnostic (including hooks receiving an intentionally zero gradient).
    ``inference_parameter_names`` describes the caller's selected inference path;
    e.g. a training-only NextLat predictor is excluded. Names may use any registered
    alias. Missing selections are unknown (``None``), not an assertion of zero.

    An optimizer may own registered but inactive or frozen parameters. Those are
    counted faithfully; foreign or duplicate ownership is rejected. No optimizer
    is constructed and no parameter, gradient, state or trainability flag changes.
    """
    tensors, records, aliases = _registered(model)
    active = _selected_ids(active_parameter_names, aliases, label="active")
    inference = _selected_ids(inference_parameter_names, aliases, label="inference")
    trainable = {identity for identity, tensor in tensors.items() if tensor.requires_grad}
    owned, ownership_groups = None, None
    if optimizer is not None:
        owned, ownership_groups = set(), []
        for group in optimizer.param_groups:
            group_names = []
            for tensor in group["params"]:
                identity = id(tensor)
                if identity not in tensors:
                    raise ValueError("Optimizer owns a foreign parameter")
                if identity in owned:
                    raise ValueError("Optimizer has duplicate parameter ownership")
                owned.add(identity)
                group_names.append(records[identity]["name"])
            ownership_groups.append(group_names)
    buffer_tensors, buffer_records, _ = _registered(model, buffers=True)
    buffer_scope = _scope(set(buffer_tensors), buffer_records)
    # Buffer values are elements, not model parameters.
    buffer_scope["element_count"] = buffer_scope.pop("parameter_count")
    buffer_scope["tensor_bytes"] = buffer_scope.pop("parameter_bytes")
    return {
        "schema": "olmo-f1-parameter-accounting-v1",
        "registered": _scope(set(tensors), records),
        "requires_grad": _scope(trainable, records),
        "active": _scope(active, records),
        "optimizer_owned": _scope(owned, records),
        "inference": _scope(inference, records),
        "optimizer_groups": ownership_groups,
        "requires_grad_not_optimizer_owned": (
            None if owned is None else _scope(trainable - owned, records)),
        "optimizer_owned_not_requires_grad": (
            None if owned is None else _scope(owned - trainable, records)),
        "optimizer_owned_not_observed_active": (
            None if owned is None or active is None else _scope(owned - active, records)),
        "parameter_records": list(records.values()),
        "resident_parameter_storage": _resident_storage(tensors, records),
        "registered_buffers": buffer_scope,
        "limitations": [
            "Parameter counts deduplicate Parameter objects, retaining tied aliases. Distinct objects sharing storage remain separate parameters.",
            "Registered parameters are resident in the model even if inactive; allocated storage is deduplicated separately and may include unused view storage.",
            "Active membership is caller-observed backward participation, not a nonzero-gradient test or a prediction of later activity.",
            "Inference membership is caller-specified for this execution mode; requires_grad and optimizer ownership do not establish it.",
            "Parameter bytes describe stored tensor dtype, not autocast operand precision or total training memory.",
        ],
    }


def _field(entry, name, default=None):
    return entry.get(name, default) if isinstance(entry, Mapping) else getattr(entry, name, default)


def _duration(entry, name, *, fallback=None):
    value = _field(entry, name)
    if value is None and fallback is not None:
        value = _field(entry, fallback)
    if value is None:
        return 0.0
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Profiler {name} must be finite and nonnegative")
    return result


def _operator_record(entry):
    name = _field(entry, "key", _field(entry, "name"))
    if not isinstance(name, str) or not name:
        raise ValueError("Profiler entry requires a nonempty key or name")
    count_value = _field(entry, "count", 1)
    count = int(count_value)
    if isinstance(count_value, bool) or count != count_value or count < 0:
        raise ValueError("Profiler count must be a nonnegative integer")
    return {
        "name": name, "count": count,
        "self_cpu_time_us": _duration(entry, "self_cpu_time_total"),
        "cpu_time_us": _duration(entry, "cpu_time_total"),
        "self_device_time_us": _duration(entry, "self_device_time_total", fallback="self_cuda_time_total"),
        "device_time_us": _duration(entry, "device_time_total", fallback="cuda_time_total"),
    }


def _attention_categories(name):
    lower = name.lower()
    categories = []
    if "scaled_dot_product_attention" in lower and not any(
        token in lower for token in ("_math", "_flash", "_efficient", "_cudnn")
    ):
        categories.append("sdpa_dispatch")
    if "scaled_dot_product_flash_attention_for_cpu" in lower:
        categories.append("cpu_flash_attention")
    elif "scaled_dot_product_flash_attention" in lower:
        categories.append("pytorch_flash_sdpa")
    if "scaled_dot_product_cudnn_attention" in lower or (
        "cudnn" in lower and ("sdpa" in lower or "attention" in lower)
    ):
        categories.append("cudnn_sdpa")
    if "scaled_dot_product_efficient_attention" in lower:
        categories.append("efficient_sdpa")
    if "scaled_dot_product_attention_math" in lower:
        categories.append("math_sdpa")
    if "flash_attn" in lower or "flashattn" in lower:
        categories.append("flash_named_external_operator_or_kernel")
    # Kernel spelling does not identify the library/version. Keep this separate
    # from the PyTorch SDPA operator and cuDNN categories above.
    if "flash" in lower and "scaled_dot_product" not in lower:
        categories.append("flash_named_kernel_or_operator")
    return categories


def summarize_profiler(
    profiler_or_entries,
    *,
    rt_selected_layers: Iterable[int] = (),
    limit: int = 12,
) -> dict[str, Any]:
    """Summarize actual recorded dispatch, without equating SDPA with Flash.

    Accept a torch profiler (``key_averages()``) or an iterable of equivalent
    event objects/dicts. Top lists rank exclusive self time, so parent operators'
    inclusive times are not incorrectly added as independent elapsed work.
    CUDA kernel names only appear if present in the supplied event inventory.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    layers = list(rt_selected_layers)
    if any(type(layer) is not int or layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("rt_selected_layers must contain unique nonnegative integer indices")
    if callable(getattr(profiler_or_entries, "key_averages", None)):
        entries = profiler_or_entries.key_averages()
        inventory = "profiler.key_averages"
    else:
        entries = profiler_or_entries
        inventory = "caller_supplied_entries"
    # Profiler group_by_input_shape can yield several records with the same key.
    # Aggregate those for this brief name-level report without dropping counts.
    grouped = {}
    for entry in entries:
        row = _operator_record(entry)
        if row["name"] not in grouped:
            grouped[row["name"]] = row
        else:
            current = grouped[row["name"]]
            for key in row:
                if key != "name":
                    current[key] += row[key]
    rows = list(grouped.values())
    categories = {
        name: [] for name in (
            "sdpa_dispatch", "pytorch_flash_sdpa", "cudnn_sdpa", "efficient_sdpa", "math_sdpa",
            "cpu_flash_attention", "flash_named_external_operator_or_kernel", "flash_named_kernel_or_operator",
        )
    }
    for row in rows:
        for name in _attention_categories(row["name"]):
            categories[name].append(dict(row))
    dispatch = {
        name: {"observed": bool(evidence), "event_count": sum(row["count"] for row in evidence),
               "evidence": sorted(evidence, key=lambda row: row["name"])}
        for name, evidence in categories.items()
    }
    return {
        "schema": "olmo-f1-profiler-summary-v1",
        "event_inventory": inventory,
        "operator_key_count": len(rows),
        "event_count": sum(row["count"] for row in rows),
        "attention_dispatch": dispatch,
        "top_cpu_operators": sorted(
            (row for row in rows if row["self_cpu_time_us"] > 0),
            key=lambda row: (-row["self_cpu_time_us"], row["name"]),
        )[:limit],
        "top_device_operators": sorted(
            (row for row in rows if row["self_device_time_us"] > 0),
            key=lambda row: (-row["self_device_time_us"], row["name"]),
        )[:limit],
        "rt_selected_layers": layers,
        "rt_attention_implementation": "eager_pytorch_dyadic_tiles" if layers else "not_selected",
        "limitations": [
            "Only operators in this recorded scope are evidence; absent dispatch names do not establish a backend elsewhere.",
            "Generic scaled_dot_product_attention is a dispatcher, not proof of Flash. cuDNN fused SDPA does not establish use of the flash-attn package or FA4.",
            "Flash-named kernels do not by themselves identify a library/version, device, dtype or backward coverage.",
            "Selected native RT layers use eager PyTorch tiles/custom backward; fused ordinary SDPA events do not establish fused RT execution.",
            "Top operator times use exclusive self CPU/device microseconds. Inclusive parent time is retained separately; nested event counts are not layer counts.",
            "Profiler time is not synchronized wall time, utilization, a throughput benchmark or a FLOP count; overlapping work and profiler overhead can distort interpretation.",
            "key_averages may omit individual CUDA kernel names; unknown dispatch remains unestablished instead of being inferred from configuration.",
        ],
    }
