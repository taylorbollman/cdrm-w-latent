"""Mitchell-initialized RT+NextLat with selectable positions and a layer-2 window.

The window limits direct attention reads, not recurrent credit assignment:
token t reads its own provisional input K/V and the permanent output K/V of
token t-1. That previous output can contain information from arbitrarily early
tokens. Both the forward scan and custom backward receive the same exact mask.
The historical vendor implementation and all learned initial tensors are kept.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from olmo.model import BufferCache, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled
from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_nextlat import _parameter_sha256, build_nextlat_model
from scripts.rt_a5_nextlat_variant import FixedSinusoidalPositions


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_window/base.json"


def window_configuration(width: int, position_encoding: str = "sinusoidal",
                         second_layer_window: int | None = None) -> dict:
    """Resolve one of the two-stage pilot's explicit architecture settings."""
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64")
    if position_encoding not in ("alibi", "sinusoidal"):
        raise ValueError("position_encoding must be 'alibi' or 'sinusoidal'")
    if second_layer_window is not None and (
            type(second_layer_window) is not int or second_layer_window != 2):
        raise ValueError("second_layer_window must be None or the integer 2")
    config = json.loads(CONFIG_PATH.read_text())
    config.update(width=width, position_encoding=position_encoding,
                  second_layer_window=second_layer_window)
    config["position_details"] = config.pop("position_options")[position_encoding]
    config["attention"] = {
        "first_layer": "full causal recurrent attention",
        "second_layer": ("full causal recurrent attention" if second_layer_window is None
                         else "self provisional K/V and immediately previous permanent output K/V"),
        "second_layer_allowed_keys": "0 <= key <= query" if second_layer_window is None
                                     else "max(0, query - 1) <= key <= query",
        "first_token": "self provisional K/V only",
        "recurrent_write_rho": 1.0,
        "gradient_truncation": False,
        "implementation": ("original recurrent block" if second_layer_window is None else
                           "parameter-free block subclass supplies additive mask to original forward/backward"),
    }
    return config


def window_two_attention_bias(x: torch.Tensor,
                              attention_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Return [batch-or-1, heads-or-1, query, key] additive bias with exact exclusions.

The valid diagonal retains its original bias (zero for this pilot), so even a
tile with no allowed permanent keys retains the finite provisional-self state.
The caller's bias is never mutated; ALiBi values on the two permitted keys are
preserved. Construction occurs inside the block, after OLMo's mask handling.
"""
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("Window attention requires nonempty [batch, length, width] inputs")
    length = x.shape[1]
    if attention_bias is None:
        attention_bias = torch.zeros((1, 1, length, length), device=x.device, dtype=x.dtype)
    elif (attention_bias.ndim != 4 or attention_bias.shape[-2:] != (length, length)
          or attention_bias.shape[0] not in (1, x.shape[0])
          or attention_bias.device != x.device or not attention_bias.is_floating_point()):
        raise ValueError("Window bias must be floating [batch-or-1, heads-or-1, length, length] on input device")
    if attention_bias.requires_grad:
        raise ValueError("This bounded window pilot does not support learned attention biases")
    query = torch.arange(length, device=x.device)[:, None]
    key = torch.arange(length, device=x.device)[None, :]
    disallowed = (key > query) | (key < query - 1)
    return attention_bias.masked_fill(disallowed, float("-inf"))


class _WindowTwoMixin:
    attention_window = 2

    def _real_forward(self, x: torch.Tensor,
                      attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        return super()._real_forward(x, window_two_attention_bias(x, attention_bias))


class WindowTwoRecurrentAutogradBlock(_WindowTwoMixin, OLMoRecurrentAutogradBlock):
    """Ordinary-autograd semantic reference using the frozen RT scan."""


class WindowTwoRecurrentBlockTiled(_WindowTwoMixin, OLMoRecurrentBlockTiled):
    """Frozen tiled forward/custom backward with a two-key attention mask."""


def build_window_model(architecture: str = "rt", width: int = 512, seed: int = 1234,
                       predictor_seed: int = 1235, device: str | torch.device = "cpu",
                       *, backend: str = "tiled", predictor_hidden_width: int | None = None,
                       position_encoding: str = "sinusoidal",
                       second_layer_window: int | None = None):
    """Build the original paired Mitchell model, changing no learned tensor.

Block replacement is performed before optimizer construction. The subclass
owns the same canonical parameter names and fresh Pre/Post views of its own
parameters. An isolated CPU RNG scope makes replacement RNG-neutral.
"""
    if architecture != "rt":
        raise ValueError("The position/window pilot supports only architecture='rt'")
    if any(type(value) is not int or value < 0 for value in (seed, predictor_seed)):
        raise ValueError("Initialization seeds must be nonnegative integers")
    config = window_configuration(width, position_encoding, second_layer_window)
    model = build_nextlat_model(
        architecture, width=width, seed=seed, predictor_seed=predictor_seed, device="cpu",
        backend=backend, predictor_hidden_width=predictor_hidden_width,
    )
    baseline = copy.deepcopy(model.nextlat_initialization)
    baseline_keys = list(model.state_dict())
    baseline_names = list(dict(model.named_parameters()))
    baseline_model_sha = _parameter_sha256(model)
    blocks = model.backbone.transformer.blocks
    if len(blocks) != 2 or model.backbone.config.rope or "wpe" in model.backbone.transformer:
        raise AssertionError("Expected the original two-block ALiBi backbone without learned positions")
    if position_encoding == "sinusoidal":
        model.backbone.config.alibi = False
        for block in blocks:
            block.config.alibi = False
        model.backbone.transformer["wpe"] = FixedSinusoidalPositions(
            width, base=config["position_details"]["base"])
    if second_layer_window == 2:
        previous = blocks[1]
        replacement_type = (WindowTwoRecurrentBlockTiled
                            if isinstance(previous, OLMoRecurrentBlockTiled)
                            else WindowTwoRecurrentAutogradBlock)
        with torch.random.fork_rng(devices=[]):
            replacement = replacement_type(previous.layer_id, copy.deepcopy(previous.config), BufferCache())
        replacement.load_state_dict(previous.state_dict(), strict=True)
        replacement.train(previous.training)
        blocks[1] = replacement
    if (list(model.state_dict()) != baseline_keys
            or list(dict(model.named_parameters())) != baseline_names
            or _parameter_sha256(model) != baseline_model_sha
            or canonical_parameter_sha256(model.backbone) != baseline["canonical_sha256"]):
        raise AssertionError("Position/window construction changed the original learned initialization")
    model.experiment_config = config
    model.nextlat_initialization = {
        **baseline,
        "schema": "rt-a5-window-initialization-v1",
        "baseline_initialization": baseline,
        "model_parameter_sha256": baseline_model_sha,
        "experiment_config": copy.deepcopy(config),
        "changed_parameter_slices": [],
        "rule": "Exact original Mitchell backbone and original predictor initialization; parameter-free positions/window only",
    }
    return model.to(device=device, dtype=torch.float32)
