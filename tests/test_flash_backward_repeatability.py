"""CPU checks for diagnostic aggregation and pre-CUDA determinism setup."""
import math

import pytest
import torch

from scripts import olmo_flash_backward_repeatability as probe


def test_chunked_geometry_and_global_aggregation():
    first = probe.compare_tensor(torch.tensor([0., 1.]), torch.tensor([1., 0.]), chunk_elements=1)
    second = probe.compare_tensor(torch.tensor([0., 4.]), torch.tensor([0., 2.]), chunk_elements=1)
    assert first["cosine"] == 0
    assert first["relative_l2"] == pytest.approx(math.sqrt(2))
    assert first["changed_elements"] == 2
    combined = probe.combine_geometry([first, second])
    assert combined["relative_l2"] == pytest.approx(math.sqrt(6/5))
    assert combined["max_absolute_error"] == 2
    assert combined["changed_elements"] == 3
    zero = probe.compare_tensor(torch.zeros(4), torch.zeros(4), chunk_elements=2)
    assert zero["relative_l2"] == 0 and zero["cosine"] is None
    assert not probe.compare_tensor(torch.tensor([float("nan")]), torch.zeros(1))["finite"]
    with pytest.raises(ValueError, match="shape and dtype"):
        probe.compare_tensor(torch.ones(2), torch.ones(3))


@pytest.mark.parametrize("enabled", [False, True])
def test_determinism_rejects_late_configuration_for_both_modes(monkeypatch, enabled):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(probe, "configure_determinism", lambda _: pytest.fail("must reject before helper"))
    with pytest.raises(RuntimeError, match="before CUDA"):
        probe.configure_before_cuda(enabled)


def test_determinism_calls_existing_helper_before_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(probe, "configure_determinism", lambda enabled: calls.append(enabled) or {"enabled": enabled})
    assert probe.configure_before_cuda(True) == {"enabled": True}
    assert calls == [True]
    with pytest.raises(TypeError):
        probe.configure_before_cuda(1)


@pytest.mark.parametrize("length,det", [(16, 0), (16, 1), (1024, 0), (1024, 1)])
def test_bounded_cli_accepts_only_four_configurations(length, det):
    args = probe.parse_args(["--length", str(length), "--deterministic", str(det), "--output-dir", "/tmp/flash-probe"])
    assert args.length == length and args.deterministic == det


@pytest.mark.parametrize("options", [["--length", "512", "--deterministic", "0"],
                                     ["--length", "1024", "--deterministic", "2"],
                                     ["--length", "16", "--deterministic", "1", "--steps", "20"]])
def test_cli_rejects_scope_expansion(options):
    with pytest.raises(SystemExit):
        probe.parse_args(["--output-dir", "/tmp/flash-probe", *options])


def test_noncontiguous_head_views_compare_with_contiguous_reference():
    value = torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 2, 4).transpose(1, 2)
    assert not value.is_contiguous()
    comparison = probe.compare_tensor(value, value.contiguous(), chunk_elements=7)
    assert comparison["finite"] and comparison["relative_l2"] == 0
    assert comparison["changed_elements"] == 0
