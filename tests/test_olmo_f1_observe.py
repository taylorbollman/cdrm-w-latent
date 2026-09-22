"""Parameter ownership and dispatch reporting without running a model or GPU."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.olmo_f1_observe import parameter_accounting, summarize_profiler


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 3)
        self.readout = nn.Linear(3, 7, bias=False)
        self.readout.weight = self.embedding.weight
        self.fusion = nn.Linear(3, 3, bias=False)
        self.predictor = nn.Linear(3, 3, bias=False)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer("scale", torch.tensor(0.5))


def _ownership(*groups):
    return SimpleNamespace(param_groups=[{"params": list(group)} for group in groups])


def test_scopes_tied_aliases_inactive_branches_and_no_mutation():
    model = TinyTiedModel()
    optimizer = _ownership(model.parameters())
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    report = parameter_accounting(
        model, optimizer=optimizer,
        active_parameter_names=["readout.weight", "embedding.weight", "predictor.weight"],
        inference_parameter_names=["embedding.weight", "readout.weight"],
    )
    assert report["registered"]["parameter_count"] == 41
    assert report["registered"]["tensor_count"] == 4
    assert report["requires_grad"]["parameter_count"] == 39
    assert report["active"]["parameter_count"] == 30
    assert report["optimizer_owned"]["parameter_count"] == 41
    assert report["inference"]["parameter_count"] == 21
    assert report["inference"]["names"] == ["embedding.weight"]
    assert report["optimizer_owned_not_requires_grad"]["names"] == ["frozen"]
    assert report["optimizer_owned_not_observed_active"]["parameter_count"] == 11
    tied = next(row for row in report["parameter_records"] if row["name"] == "embedding.weight")
    assert tied["aliases"] == ["embedding.weight", "readout.weight"]
    assert report["resident_parameter_storage"]["known_allocated_bytes"] == 41 * 4
    assert report["resident_parameter_storage"]["complete"]
    assert report["registered_buffers"]["element_count"] == 1
    assert report["registered_buffers"]["tensor_bytes"] == 4
    assert model.readout.weight is model.embedding.weight
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    json.dumps(report, allow_nan=False)


def test_unmeasured_is_distinct_from_empty_and_optimizer_ownership_is_optional():
    model = TinyTiedModel()
    report = parameter_accounting(model)
    assert report["active"] is None and report["inference"] is None
    assert report["optimizer_owned"] is None and report["optimizer_groups"] is None
    empty = parameter_accounting(model, active_parameter_names=[], inference_parameter_names=[])
    assert empty["active"]["parameter_count"] == empty["inference"]["parameter_count"] == 0


def test_separate_parameters_sharing_storage_count_allocations_once():
    model = nn.Module()
    backing = torch.arange(16, dtype=torch.float32)
    model.first = nn.Parameter(backing[:8])
    model.second = nn.Parameter(backing[4:12])
    model.third_alias = model.first
    report = parameter_accounting(model)
    assert report["registered"]["parameter_count"] == 16
    assert report["registered"]["tensor_count"] == 2
    assert report["resident_parameter_storage"]["known_allocated_bytes"] == 16 * 4
    assert report["resident_parameter_storage"]["shared_storage_objects"] == [
        {"name": "second", "shares_storage_with": "first"}]


def test_stored_dtype_and_meta_are_reported_without_invented_memory():
    model = nn.Module()
    model.bf16 = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    model.unmaterialized = nn.Parameter(torch.empty(7, device="meta"))
    report = parameter_accounting(model)
    assert report["registered"]["parameter_bytes"] == 3 * 2 + 7 * 4
    assert report["resident_parameter_storage"]["known_allocated_bytes"] == 6
    assert not report["resident_parameter_storage"]["complete"]
    assert report["resident_parameter_storage"]["unavailable_names"] == ["unmaterialized"]


@pytest.mark.parametrize("field", ["active_parameter_names", "inference_parameter_names"])
@pytest.mark.parametrize("names", [["missing.weight"], "embedding.weight", [None]])
def test_bad_scope_names_fail(field, names):
    with pytest.raises(ValueError):
        parameter_accounting(TinyTiedModel(), **{field: names})


def test_foreign_and_duplicate_optimizer_ownership_fail():
    model = TinyTiedModel()
    with pytest.raises(ValueError, match="foreign"):
        parameter_accounting(model, optimizer=_ownership([nn.Parameter(torch.zeros(2))]))
    with pytest.raises(ValueError, match="duplicate"):
        parameter_accounting(model, optimizer=_ownership([model.embedding.weight], [model.readout.weight]))


def test_optimizer_subset_is_reported_and_does_not_infer_activity():
    model = TinyTiedModel()
    report = parameter_accounting(model, optimizer=_ownership([model.predictor.weight]))
    assert report["optimizer_owned"]["parameter_count"] == 9
    assert report["requires_grad_not_optimizer_owned"]["parameter_count"] == 30
    assert report["active"] is None


def _event(name, *, count=1, cpu=0., device=0., inclusive_cpu=0., inclusive_device=0.):
    return {"key": name, "count": count, "self_cpu_time_total": cpu,
            "self_device_time_total": device, "cpu_time_total": inclusive_cpu,
            "device_time_total": inclusive_device}


def test_generic_dispatch_does_not_claim_flash_or_math_for_eager_rt():
    report = summarize_profiler([
        _event("aten::scaled_dot_product_attention", count=3, cpu=2, inclusive_device=50),
        _event("aten::matmul", cpu=5, device=30),
        _event("aten::softmax", cpu=4, device=20),
    ], rt_selected_layers=(0, 15))
    dispatch = report["attention_dispatch"]
    assert dispatch["sdpa_dispatch"]["event_count"] == 3
    assert not dispatch["pytorch_flash_sdpa"]["observed"]
    assert not dispatch["cudnn_sdpa"]["observed"]
    assert not dispatch["math_sdpa"]["observed"]
    assert report["rt_attention_implementation"] == "eager_pytorch_dyadic_tiles"
    assert report["rt_selected_layers"] == [0, 15]
    assert report["top_device_operators"][0]["name"] == "aten::matmul"
    assert all(row["name"] != "aten::scaled_dot_product_attention" for row in report["top_device_operators"])
    json.dumps(report, allow_nan=False)


def test_cudnn_flash_math_and_efficient_are_separate_with_backward_evidence():
    names = [
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::_scaled_dot_product_cudnn_attention_backward",
        "cudnn_generated_native_sdpa_sm90_flash_fprop_kernel",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_efficient_attention",
        "flash_attn::_flash_attn_forward",
    ]
    report = summarize_profiler([_event(name, count=2) for name in names])
    dispatch = report["attention_dispatch"]
    assert dispatch["cudnn_sdpa"]["event_count"] == 6
    assert dispatch["pytorch_flash_sdpa"]["event_count"] == 4
    assert dispatch["math_sdpa"]["event_count"] == 2
    assert dispatch["efficient_sdpa"]["event_count"] == 2
    assert dispatch["flash_named_external_operator_or_kernel"]["event_count"] == 2
    assert not dispatch["sdpa_dispatch"]["observed"]


def test_cpu_flash_is_not_reported_as_cuda_flash_sdpa():
    report = summarize_profiler([_event("aten::_scaled_dot_product_flash_attention_for_cpu")])
    assert report["attention_dispatch"]["cpu_flash_attention"]["observed"]
    assert not report["attention_dispatch"]["pytorch_flash_sdpa"]["observed"]


def test_profiler_object_and_legacy_cuda_attributes_aggregate_and_sort():
    entries = [
        SimpleNamespace(key="aten::mm", count=2, self_cpu_time_total=3., cpu_time_total=4.,
                        self_cuda_time_total=8., cuda_time_total=20.),
        _event("aten::mm", count=3, cpu=4., device=9., inclusive_cpu=5., inclusive_device=30.),
        _event("parent", cpu=1., device=0., inclusive_cpu=1000., inclusive_device=1000.),
        _event("other", cpu=9., device=11.),
    ]
    profile = SimpleNamespace(key_averages=lambda: entries)
    report = summarize_profiler(profile, limit=1)
    assert report["event_inventory"] == "profiler.key_averages"
    assert report["operator_key_count"] == 3 and report["event_count"] == 7
    assert report["top_cpu_operators"][0]["name"] == "other"
    row = report["top_device_operators"][0]
    assert row["name"] == "aten::mm" and row["count"] == 5
    assert row["self_device_time_us"] == 17. and row["device_time_us"] == 50.
    assert report["rt_attention_implementation"] == "not_selected"


def test_current_device_fields_take_precedence_even_when_zero():
    event = _event("x")
    event.update(self_cuda_time_total=99., cuda_time_total=100.)
    report = summarize_profiler([event])
    assert report["top_device_operators"] == []


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": True}, {"rt_selected_layers": [-1]},
                                    {"rt_selected_layers": [0, 0]}, {"rt_selected_layers": [True]}])
def test_bad_profiler_options_fail(kwargs):
    with pytest.raises(ValueError):
        summarize_profiler([], **kwargs)


@pytest.mark.parametrize("field,value", [("count", -1), ("count", 1.5), ("self_cpu_time_total", float("nan")),
                                        ("self_device_time_total", -1.), ("device_time_total", float("inf"))])
def test_bad_profiler_values_fail(field, value):
    event = _event("x"); event[field] = value
    with pytest.raises(ValueError):
        summarize_profiler([event])
