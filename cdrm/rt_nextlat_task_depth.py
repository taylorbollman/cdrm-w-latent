"""Fresh three-layer restricted-first RT for the unchanged mixed-task objective.

Only backbone depth changes from the frozen two-layer task factory. Initialization
uses a canonical SEQ model at the requested depth: Mitchell's depth-dependent
scaling and RNG consumption therefore need not match the two-layer backbone.
The independently seeded NextLat predictor remains exactly matched by width.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from olmo.checkpoint_conversion import convert_model
from olmo.config import ModelConfig
from olmo.model import BufferCache, OLMo
from cdrm.rt_nextlat_tasks import TaskNextLat, read_configuration as read_base_configuration
from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_nextlat import NextLatDynamicsModel, _parameter_sha256, nextlat_configuration
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_nextlat_tasks/fuzzy_d128_l3.json"
SOURCE_PATHS = ("cdrm/rt_nextlat_task_depth.py", "configs/rt_nextlat_tasks/fuzzy_d128_l3.json")


def source_manifest() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(SOURCE_PATHS)}


def read_configuration(config: dict | str | Path | None = None) -> dict:
    """Load the explicit three-layer route while retaining the base task rules."""
    if config is None:
        config = CONFIG_PATH
    result = copy.deepcopy(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    raw = result["backbone"]
    if result.get("schema") != "rt-nextlat-tasks-depth-model-v1":
        raise ValueError("Expected the explicit three-layer task model schema")
    if type(raw.get("n_layers")) is not int or raw["n_layers"] != 3:
        raise ValueError("Expected exactly three RT layers")
    if "embedding_injection" in result:
        raise ValueError("This depth pilot has no embedding injection")
    expected = {"block_type": "recurrent", "block_group_size": 1,
                "recurrent_backend": "tiled", "recurrent_layers": None,
                "ordinary_attention_precision_policy": "legacy",
                "recurrent_precision_policy": "legacy"}
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected three standard FP32 tiled RT blocks before window replacement")
    # Reuse every unchanged two-layer task constraint through an independent
    # validation view. The returned configuration retains its actual depth.
    reference = copy.deepcopy(result)
    reference["schema"] = "rt-nextlat-tasks-model-v1"
    reference["backbone"]["n_layers"] = 2
    read_base_configuration(reference)
    return result


def build_model(config: dict | str | Path | None = None, *, seed: int = 1234,
                predictor_seed: int = 1235, fuzzy_seed: int = 1236,
                device: str | torch.device = "cpu", backend: str = "tiled") -> TaskNextLat:
    """Create a canonical V60 SEQ draw at the actual depth, then convert to RT.

    The head count is read from this experiment's configuration, not inferred
    using the older A5 factory's fixed 64-dimensional-head assumption. All CPU
    initialization streams are isolated from training and each other.
    """
    config = read_configuration(config)
    if any(type(value) is not int or value < 0 for value in (seed, predictor_seed, fuzzy_seed)):
        raise ValueError("Initialization seeds must be nonnegative integers")
    if backend == "autograd":
        backend = "naive"
    if backend not in ("tiled", "naive"):
        raise ValueError("backend must be 'tiled' or 'naive'")
    raw = copy.deepcopy(config["backbone"])
    raw.update(init_device="cpu", recurrent_backend=backend,
               block_type="recurrent" if backend == "tiled" else "recurrent_autograd")
    config["backbone"] = copy.deepcopy(raw)
    width = raw["d_model"]
    depth = raw["n_layers"]
    source_config = ModelConfig(**{**raw, "vocab_size": 60, "pad_token_id": 0,
                                  "block_type": "sequential"})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        source = OLMo(source_config).to(dtype=torch.float32)
    original_sha = canonical_parameter_sha256(source)
    # Appended rows use the exact Mitchell table rule: truncated N(0, D^-1),
    # cutoff three standard deviations. No existing learned tensor is redrawn.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(fuzzy_seed)
        for name, size_attribute in (("wte", "num_embeddings"), ("ff_out", "out_features")):
            module = source.transformer[name]
            rows = torch.empty(16, width, dtype=torch.float32)
            std = 1 / math.sqrt(width)
            nn.init.trunc_normal_(rows, mean=0.0, std=std, a=-3 * std, b=3 * std)
            module.weight = nn.Parameter(torch.cat((module.weight.detach(), rows), dim=0))
            setattr(module, size_attribute, 76)
    source.config.vocab_size = 76
    source.config.pad_token_id = 75
    for block in source.transformer.blocks:
        block.config.vocab_size = 76
        block.config.pad_token_id = 75
    expanded_sha = canonical_parameter_sha256(source)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backbone = OLMo(ModelConfig(**raw)).to(dtype=torch.float32)
        conversion = convert_model(source, backbone).to_dict()
        previous = backbone.transformer.blocks[0]
        block_class = WindowTwoRecurrentBlockTiled if backend == "tiled" else WindowTwoRecurrentAutogradBlock
        replacement = block_class(previous.layer_id, copy.deepcopy(previous.config), BufferCache())
        replacement.load_state_dict(previous.state_dict(), strict=True)
        backbone.transformer.blocks[0] = replacement
    if canonical_parameter_sha256(backbone) != expanded_sha:
        raise AssertionError("RT conversion/window replacement changed canonical initialized tensors")
    nl_config = nextlat_configuration(width, config["predictor_hidden_width"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(predictor_seed)
        predictor = NextLatDynamicsModel(
            width, config["predictor_hidden_width"],
            epsilon=nl_config["predictor"]["normalization_epsilon"],
            initialization_std=nl_config["predictor"]["initialization_std"],
        ).to(dtype=torch.float32)
    model = TaskNextLat(backbone, predictor)
    backbone_count = sum(p.numel() for p in backbone.parameters())
    predictor_count = sum(p.numel() for p in predictor.parameters())
    # Each GELU RT block: 12 D^2 weights + four D-wide norms.
    # The final norm and two V76 tables add another 153 D parameters.
    if backbone_count != depth * (12 * width * width + 4 * width) + 153 * width:
        raise AssertionError("Unexpected depth-task backbone parameter count")
    if "wpe" in backbone.transformer or any(p.dtype != torch.float32 for p in model.parameters()):
        raise AssertionError("Expected no position embedding and exclusively FP32 parameters")
    model.task_config = config
    model.nextlat_config = nl_config
    model.initialization = {
        "schema": "rt-nextlat-tasks-depth-initialization-v1",
        "depth": depth,
        "depth_pairing": "Fresh canonical initialization at actual depth; no exact shared-weight pairing across depths",
        "seed": seed, "predictor_seed": predictor_seed, "fuzzy_seed": fuzzy_seed,
        "canonical_v60_sha256": original_sha, "canonical_sha256": expanded_sha,
        "model_parameter_sha256": _parameter_sha256(model),
        "predictor_sha256": _parameter_sha256(predictor), "conversion": conversion,
        "backbone_parameter_count": backbone_count, "predictor_parameter_count": predictor_count,
        "parameter_count": backbone_count + predictor_count,
        "rule": "Fresh actual-depth canonical V60 CPU Mitchell SEQ; independently appended Mitchell Fuzzy rows; "
                "exhaustive RT conversion; parameter-preserving first-layer window; isolated NextLat RNG",
        "task_config": copy.deepcopy(config), "nextlat_config": copy.deepcopy(nl_config),
    }
    return model.to(device=device, dtype=torch.float32)

