"""CPU contracts for references that never overlap eager work with a live graph.

These fixtures exercise real autograd and persistent buffers; they do not claim
CUDA capture, kernel parity, or large-batch capacity clearance.
"""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from scripts import olmo_large_batch_validation as validation


class LossResult:
    def __init__(self, passes):
        self.pass_losses = [SimpleNamespace(sums=sums) for sums in passes]


class Graph:
    def __init__(self, result, events):
        self.result = result
        self.events = events

    def reset(self):
        self.events.append("reset")
        self.result = None


class CPUPlan:
    """A deliberately permissive validator lets the helper detect mutations."""

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.linear = torch.nn.Linear(2, 2)
        self.model.backbone = torch.nn.Module()
        self.model.backbone.fusion = torch.nn.Linear(2, 2, bias=False)
        self.model.register_parameter("frozen", torch.nn.Parameter(
            torch.ones(1), requires_grad=False))
        with torch.no_grad():
            self.model.linear.weight.copy_(torch.tensor([[.5, -.25], [.125, .75]]))
            self.model.linear.bias.copy_(torch.tensor([.25, -.5]))
        self.mode = SimpleNamespace(enabled=False, num_passes=1, beta=1.)
        self.inputs = torch.zeros(2, 2)
        self.graph = self.graph_result = None
        self.active_names = self.gradient_addresses = None
        self.events = []
        self.mutation = None
        self.mutate_replay = True
        self.before_eager = None
        self.retired_tensors = []
        self.last_eager_result = None

    def validate_execution(self):
        self.events.append(("validate", self.graph is not None))

    def load_batch(self, batch):
        self.events.append("load")
        self.inputs.copy_(batch)

    def initialize_gradients(self):
        if self.active_names is not None:
            return
        assert self.graph is None
        self.events.append("initialize")
        self._calculate()
        self.active_names = tuple(name for name, parameter in self.model.named_parameters()
                                  if parameter.grad is not None)
        self.gradient_addresses = {name: None if p.grad is None else p.grad.data_ptr()
                                   for name, p in self.model.named_parameters()}

    def _calculate(self, *, accumulate=False):
        if not accumulate:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
        values = self.model.linear(self.inputs)
        passes = [
            {"ce": values.square().sum(), "latent": values.sum(),
             "kl": values[0].sum() * .125, "extra": values[1].sum() * .25},
            {"ce": values.square().sum() * .5, "latent": values.sum() * .5,
             "kl": values[0].sum() * .0625},
        ]
        sum(loss for sums in passes for loss in sums.values()).backward()
        return LossResult(passes)

    def capture(self):
        assert self.active_names is not None and self.graph is None
        self.graph_result = self._calculate()
        self.graph = Graph(self.graph_result, self.events)

    def backward(self, *, replay=False):
        assert self.active_names is not None, "helper must initialize persistent gradients"
        assert (self.graph is not None) is replay, "eager execution overlapped a live graph"
        self.events.append("replay" if replay else "eager")
        if not replay and self.before_eager is not None:
            self.before_eager()
        mutate = replay is self.mutate_replay
        result = self._calculate(accumulate=mutate and self.mutation == "accumulate")
        if replay:
            with torch.no_grad():
                for old_pass, new_pass in zip(self.graph_result.pass_losses, result.pass_losses):
                    for name, value in new_pass.sums.items():
                        old_pass.sums[name].copy_(value)
            result = self.graph_result
        else:
            self.last_eager_result = result
        if mutate:
            self._mutate(result)
        return result

    def _mutate(self, result):
        if self.mutation == "gradient":
            self.model.linear.weight.grad[0, 0].add_(.25)
        elif self.mutation == "loss":
            with torch.no_grad():
                result.pass_losses[1].sums["ce"].add_(.25)
        elif self.mutation == "missing_loss":
            del result.pass_losses[0].sums["extra"]
        elif self.mutation == "unexpected_loss":
            result.pass_losses[1].sums["unexpected"] = torch.tensor(1.)
        elif self.mutation == "missing_gradient":
            self.model.linear.bias.grad = None
        elif self.mutation == "unexpected_gradient":
            self.model.backbone.fusion.weight.grad = torch.ones_like(
                self.model.backbone.fusion.weight)
        elif self.mutation == "parameter_storage":
            self.retired_tensors.append(self.model.linear.weight.data)
            self.model.linear.weight.data = self.model.linear.weight.data.clone()
        elif self.mutation == "gradient_storage":
            self.retired_tensors.append(self.model.linear.weight.grad)
            self.model.linear.weight.grad = self.model.linear.weight.grad.clone()


@pytest.fixture
def plan(monkeypatch):
    plan = CPUPlan()
    collect = gc.collect
    monkeypatch.setattr(validation.torch.cuda, "synchronize",
                        lambda *args, **kwargs: plan.events.append("sync"))
    monkeypatch.setattr(validation.torch.cuda, "empty_cache",
                        lambda: plan.events.append("empty_cache"))

    def collect_and_record(*args, **kwargs):
        plan.events.append("collect")
        return collect(*args, **kwargs)

    monkeypatch.setattr(validation.gc, "collect", collect_and_record)
    return plan


def batch(index=0):
    return torch.tensor([[1. + index, 2.], [3., 4. - index]])


def captured_reference(plan):
    reference = validation.snapshot_eager_cpu(plan, batch())
    plan.capture()
    return reference


def assert_passed(check):
    assert check["passed"]
    assert check["all_bitwise_equal"]
    assert check["ownership_matches"]
    assert check["loss_names_match"]
    assert check["storage_matches"]
    assert set(check["gradients"]) == {"linear.weight", "linear.bias"}


def test_eager_references_are_independent_cpu_copies_of_all_loss_terms(plan):
    reference = validation.snapshot_eager_cpu(plan, batch())
    assert isinstance(reference, validation.CPUReference)
    assert set(reference.losses) == {
        "pass0/ce", "pass0/latent", "pass0/kl", "pass0/extra",
        "pass1/ce", "pass1/latent", "pass1/kl",
    }
    assert set(reference.gradients) == {"linear.weight", "linear.bias"}
    assert set(reference.declared_names) == set(reference.expected_names) == set(reference.gradients)
    assert all(value.device.type == "cpu" and not value.requires_grad
               for value in [*reference.losses.values(), *reference.gradients.values()])
    expected_losses = {name: value.clone() for name, value in reference.losses.items()}
    expected_grads = {name: value.clone() for name, value in reference.gradients.items()}
    parameters = dict(plan.model.named_parameters())
    for name, value in reference.gradients.items():
        assert value.data_ptr() != parameters[name].grad.data_ptr()
        parameters[name].grad.fill_(-99.)
    with torch.no_grad():
        for loss in plan.last_eager_result.pass_losses:
            for value in loss.sums.values():
                value.fill_(-99.)
    assert all(torch.equal(reference.losses[name], value) for name, value in expected_losses.items())
    assert all(torch.equal(reference.gradients[name], value) for name, value in expected_grads.items())
    assert plan.events.count("initialize") == 1
    assert plan.events.index("load") < plan.events.index("eager")
    assert plan.model.backbone.fusion.weight.grad is None
    assert plan.model.frozen.grad is None


def test_changed_tokens_use_precomputed_references_and_repeat_without_accumulation(plan):
    original = validation.snapshot_eager_cpu(plan, batch())
    changed = validation.snapshot_eager_cpu(plan, batch(1))
    assert not torch.equal(original.losses["pass0/ce"], changed.losses["pass0/ce"])
    pointers = {name: p.grad.data_ptr() for name, p in plan.model.named_parameters()
                if p.grad is not None}
    plan.capture()
    for index, reference in [(0, original), (1, changed)]:
        plan.load_batch(batch(index))
        before = len(plan.events)
        check = validation.replay_compare_cpu(plan, reference, f"tokens-{index}", replays=2)
        assert_passed(check)
        assert check["name"] == f"tokens-{index}"
        assert plan.events[before:].count("replay") == 2
        assert "eager" not in plan.events[before:]
    assert plan.events.count("initialize") == 1
    assert pointers == {name: p.grad.data_ptr() for name, p in plan.model.named_parameters()
                        if p.grad is not None}


@pytest.mark.parametrize("mutation", [
    "gradient", "loss", "accumulate", "missing_loss", "unexpected_loss",
    "missing_gradient", "unexpected_gradient", "parameter_storage", "gradient_storage",
])
def test_replay_comparison_rejects_changed_results_ownership_and_storage(plan, mutation):
    reference = captured_reference(plan)
    plan.mutation = mutation
    check = validation.replay_compare_cpu(plan, reference, mutation)
    assert not check["passed"]
    if mutation in {"missing_loss", "unexpected_loss"}:
        assert not check["loss_names_match"]
    if mutation in {"missing_gradient", "unexpected_gradient"}:
        assert not check["ownership_matches"]
    if mutation.endswith("storage"):
        assert not check["storage_matches"]


def test_first_bad_replay_cannot_be_hidden_by_a_correct_final_replay(plan, monkeypatch):
    reference = captured_reference(plan)
    backward = plan.backward
    calls = []

    def corrupt_first_only(*, replay=False):
        calls.append(replay)
        plan.mutation = "gradient" if len(calls) == 1 else None
        return backward(replay=replay)

    monkeypatch.setattr(plan, "backward", corrupt_first_only)
    check = validation.replay_compare_cpu(plan, reference, "first-replay", replays=2)
    assert calls == [True, True]
    assert check["replays_checked"] == 2
    assert not check["passed"] and not check["all_bitwise_equal"]
    assert not check["replay_checks"][0]["passed"]
    assert_passed(check["replay_checks"][1])


@pytest.mark.parametrize("replays", [0, -1, True, False, 1.5, "2", None])
def test_invalid_replay_counts_fail_before_execution(plan, replays):
    reference = captured_reference(plan)
    before = list(plan.events)
    with pytest.raises((TypeError, ValueError)):
        validation.replay_compare_cpu(plan, reference, "invalid", replays=replays)
    assert plan.events == before


def test_reference_capture_and_comparison_enforce_graph_lifetime(plan):
    reference = validation.snapshot_eager_cpu(plan, batch())
    before = plan.events.count("eager")
    with pytest.raises(ValueError):
        validation.snapshot_replay_cpu(plan)
    with pytest.raises(ValueError):
        validation.replay_compare_cpu(plan, reference, "uncaptured")
    with pytest.raises(ValueError):
        validation.release_graph_then_compare_eager_cpu(plan, reference, "uncaptured")
    plan.capture()
    with pytest.raises(ValueError):
        validation.snapshot_eager_cpu(plan, batch(1))
    assert plan.events.count("eager") == before
    assert "replay" not in plan.events


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("mutation", [
    "missing_gradient", "unexpected_gradient", "parameter_storage", "gradient_storage",
])
def test_invalid_reference_ownership_or_storage_is_rejected(plan, replay, mutation):
    validation.snapshot_eager_cpu(plan, batch())
    if replay:
        plan.capture()
    plan.mutation, plan.mutate_replay = mutation, replay
    with pytest.raises(ValueError):
        if replay:
            validation.snapshot_replay_cpu(plan)
        else:
            validation.snapshot_eager_cpu(plan, batch())


@pytest.mark.parametrize("declaration", ["declared", "expected"])
def test_reference_requires_declared_expected_and_observed_names_to_agree(plan, monkeypatch, declaration):
    validation.snapshot_eager_cpu(plan, batch())
    if declaration == "declared":
        plan.active_names = ("linear.weight",)
    else:
        monkeypatch.setattr(validation, "active_names", lambda model, mode: {"linear.weight"})
    with pytest.raises(ValueError):
        validation.snapshot_eager_cpu(plan, batch())


def test_updated_weights_get_fresh_replay_reference_then_release_before_eager(plan):
    old_reference = captured_reference(plan)
    with torch.no_grad():
        plan.model.linear.weight.add_(.25)
        plan.model.linear.bias.sub_(.125)
    before = plan.events.count("replay")
    updated_reference = validation.snapshot_replay_cpu(plan)
    assert plan.events.count("replay") == before + 1
    assert not torch.equal(old_reference.losses["pass0/ce"], updated_reference.losses["pass0/ce"])
    graph_ref, result_ref = weakref.ref(plan.graph), weakref.ref(plan.graph_result)
    parameter_ids = {name: (id(p), p.data_ptr()) for name, p in plan.model.named_parameters()}
    gradient_pointers = {name: None if p.grad is None else p.grad.data_ptr()
                         for name, p in plan.model.named_parameters()}
    plan.events.clear()

    def assert_released():
        assert plan.graph is None and plan.graph_result is None
        assert graph_ref() is None and result_ref() is None
        assert all(event in plan.events for event in ("sync", "reset", "collect", "empty_cache"))
        assert plan.events.index("reset") < plan.events.index("collect") < plan.events.index("empty_cache")
        assert ("validate", True) in plan.events[:plan.events.index("reset")]
        assert ("validate", False) in plan.events[plan.events.index("reset"):]

    plan.before_eager = assert_released
    check = validation.release_graph_then_compare_eager_cpu(plan, updated_reference, "updated-weights")
    assert_passed(check)
    assert check["name"] == "updated-weights"
    assert plan.events.count("eager") == 1 and "replay" not in plan.events
    assert parameter_ids == {name: (id(p), p.data_ptr()) for name, p in plan.model.named_parameters()}
    assert gradient_pointers == {name: None if p.grad is None else p.grad.data_ptr()
                                 for name, p in plan.model.named_parameters()}


@pytest.mark.parametrize("mutation", ["gradient", "loss", "parameter_storage", "gradient_storage"])
def test_final_eager_comparison_rejects_mutations_after_teardown(plan, mutation):
    captured_reference(plan)
    reference = validation.snapshot_replay_cpu(plan)
    plan.mutation, plan.mutate_replay = mutation, False
    check = validation.release_graph_then_compare_eager_cpu(plan, reference, mutation)
    assert not check["passed"]
    assert plan.graph is None and plan.graph_result is None
    if mutation.endswith("storage"):
        assert not check["storage_matches"]


def test_final_comparison_checks_reference_storage_before_destroying_graph(plan):
    reference = captured_reference(plan)
    graph = plan.graph
    plan.retired_tensors.append(plan.model.linear.weight.grad)
    plan.model.linear.weight.grad = plan.model.linear.weight.grad.clone()
    with pytest.raises(ValueError):
        validation.release_graph_then_compare_eager_cpu(plan, reference, "replaced-before-release")
    assert plan.graph is graph and plan.graph_result is not None
    assert "reset" not in plan.events


def test_storage_corruption_during_graph_release_fails_before_eager(plan, monkeypatch):
    reference = captured_reference(plan)
    reset = plan.graph.reset

    def reset_and_replace_gradient():
        reset()
        plan.retired_tensors.append(plan.model.linear.weight.grad)
        plan.model.linear.weight.grad = plan.model.linear.weight.grad.clone()

    monkeypatch.setattr(plan.graph, "reset", reset_and_replace_gradient)
    before = plan.events.count("eager")
    with pytest.raises(ValueError, match="storage"):
        validation.release_graph_then_compare_eager_cpu(plan, reference, "storage-during-release")
    assert plan.graph is None and plan.graph_result is None
    assert "reset" in plan.events
    assert plan.events.count("eager") == before
