"""Task-local readouts for restricted-first RT + the existing NextLat objective.

Input IDs use one shared 76-row table (A5 0:60, MAD Fuzzy 60:76). Labels
remain local to each task. Native MAD inputs/labels are already shifted by
their generator; this module never shifts CE targets or masks native padding.
Inference runs only the RT backbone, never an autonomous predictor rollout.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from olmo.checkpoint_conversion import convert_model
from olmo.config import ModelConfig
from olmo.model import BufferCache, OLMo
from scripts.rt_a5_common import canonical_parameter_sha256
from scripts.rt_a5_nextlat import NextLatDynamicsModel, _parameter_sha256, nextlat_configuration
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_nextlat_tasks/fuzzy_d128.json"
TASK_RANGES = {"a5": (0, 60), "fuzzy": (60, 76)}


def read_configuration(config: dict | str | Path | None = None) -> dict:
    """Load an independent serializable model configuration."""
    if config is None:
        config = CONFIG_PATH
    result = copy.deepcopy(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    raw = result["backbone"]
    if result["schema"] != "rt-nextlat-tasks-model-v1":
        raise ValueError("Unsupported task model schema")
    if result["task_ranges"] != {key: list(value) for key, value in TASK_RANGES.items()}:
        raise ValueError("This interface requires A5 0:60 and Fuzzy 60:76")
    if (raw["n_layers"] != 2 or raw["vocab_size"] != 76 or raw["embedding_size"] is not None
            or result["window_layer"] != 0 or result["window_size"] != 2
            or raw["n_heads"] != raw["n_kv_heads"]
            or raw["d_model"] % raw["n_heads"] or raw["mlp_hidden_size"] != 4 * raw["d_model"]
            or raw["max_sequence_length"] < 2):
        raise ValueError("Expected two full-width-QKV RT blocks, first window two, and FFN ratio four")
    expected = {"alibi": True, "rope": False, "init_fn": "mitchell", "precision": "fp32",
                "reference_eager": True, "cdrm_enabled": False, "weight_tying": False,
                "include_bias": False, "embedding_layer_norm": False, "recurrent_write_rho": 1.0,
                "attention_dropout": 0.0, "residual_dropout": 0.0, "embedding_dropout": 0.0}
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("This pilot retains the Mitchell/ALiBi FP32 eager RT contract")
    return result


def _task_range(task: str) -> tuple[int, int]:
    try:
        return TASK_RANGES[task]
    except KeyError as exc:
        raise ValueError("task must be 'a5' or 'fuzzy'") from exc


def encode_inputs(local_ids: torch.Tensor, task: str = "fuzzy") -> torch.Tensor:
    """Map validated task-local input symbols into the shared embedding table.

    The data adapter validates symbol ranges once; this hot-path helper does
    not synchronize a GPU to revalidate their minimum and maximum each batch.
    """
    offset, _ = _task_range(task)
    if local_ids.dtype != torch.long:
        raise TypeError("Input symbols must have int64 dtype")
    return local_ids + offset


class TaskNextLat(nn.Module):
    def __init__(self, backbone: nn.Module, predictor: NextLatDynamicsModel):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def forward(self, input_ids: torch.Tensor, task: str = "fuzzy") -> torch.Tensor:
        """Return task-local backbone logits without invoking the predictor."""
        return task_logits(self, input_ids, task)


def build_model(config: dict | str | Path | None = None, *, seed: int = 1234,
                predictor_seed: int = 1235, fuzzy_seed: int = 1236,
                device: str | torch.device = "cpu", backend: str = "tiled") -> TaskNextLat:
    """Preserve a canonical V60 SEQ draw; append Fuzzy rows, then convert to RT.

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
    # Two GELU RT blocks, 4 norm vectors/block, final norm, and two V76 tables.
    if backbone_count != 24 * width * width + 161 * width:
        raise AssertionError("Unexpected two-block task-backbone parameter count")
    if "wpe" in backbone.transformer or any(p.dtype != torch.float32 for p in model.parameters()):
        raise AssertionError("Expected no position embedding and exclusively FP32 parameters")
    model.task_config = config
    model.nextlat_config = nl_config
    model.initialization = {
        "schema": "rt-nextlat-tasks-initialization-v1",
        "seed": seed, "predictor_seed": predictor_seed, "fuzzy_seed": fuzzy_seed,
        "canonical_v60_sha256": original_sha, "canonical_sha256": expanded_sha,
        "model_parameter_sha256": _parameter_sha256(model),
        "predictor_sha256": _parameter_sha256(predictor), "conversion": conversion,
        "backbone_parameter_count": backbone_count, "predictor_parameter_count": predictor_count,
        "parameter_count": backbone_count + predictor_count,
        "rule": "Canonical V60 CPU Mitchell SEQ; independently appended Mitchell Fuzzy rows; "
                "exhaustive RT conversion; parameter-preserving first-layer window; isolated NextLat RNG",
        "task_config": copy.deepcopy(config), "nextlat_config": copy.deepcopy(nl_config),
    }
    return model.to(device=device, dtype=torch.float32)


def _validate_input_shape(input_ids: torch.Tensor) -> None:
    if input_ids.ndim != 2 or not input_ids.numel():
        raise ValueError("Inputs must be nonempty [batch, length]")
    if input_ids.dtype != torch.long:
        raise TypeError("Input symbols must have int64 dtype")


def task_logits(model: TaskNextLat, input_ids: torch.Tensor, task: str = "fuzzy") -> torch.Tensor:
    """Select the task's output rows before softmax; predictor is never run."""
    _validate_input_shape(input_ids)
    start, stop = _task_range(task)
    return model.backbone(input_ids).logits[..., start:stop]


def task_loss(model: TaskNextLat, input_ids: torch.Tensor, local_labels: torch.Tensor,
              task: str = "fuzzy", latent_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Native aligned CE plus NextLat averaged over every B*(T-1)*D coordinate.

    CE accepts native ignore labels -100. They never mask NextLat transitions:
    its conditioning token is the actual next input, including native padding.
    Only the next latent's target role is detached. Weight zero bypasses the
    predictor entirely, preserving the backbone-only CE and its gradients.
    """
    _validate_input_shape(input_ids)
    start, stop = _task_range(task)
    if local_labels.shape != input_ids.shape or local_labels.dtype != torch.long:
        raise ValueError("Labels must be task-local int64 [batch, length], aligned to inputs")
    if not isinstance(latent_weight, (int, float)) or not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if latent_weight > 0 and input_ids.shape[1] < 2:
        raise ValueError("NextLat requires at least two input positions")
    output = model.backbone(input_ids, return_pre_logits=latent_weight > 0)
    logits = output.logits[..., start:stop]
    if logits.dtype != torch.float32:
        raise TypeError("This pilot requires FP32 logits")
    ce = F.cross_entropy(logits.reshape(-1, stop - start), local_labels.reshape(-1), ignore_index=-100)
    if latent_weight == 0:
        return {"loss": ce, "ce": ce, "latent": ce.new_zeros(()), "logits": logits}
    hidden = output.pre_logits
    if hidden is None or hidden.dtype != torch.float32:
        raise TypeError("NextLat requires attached FP32 post-final-normalization latents")
    next_embeddings = model.backbone.transformer.wte(input_ids[:, 1:])
    predicted = model.predictor(hidden[:, :-1], next_embeddings)
    latent = F.smooth_l1_loss(predicted, hidden[:, 1:].detach(), beta=1.0, reduction="mean")
    return {"loss": ce + latent_weight * latent, "ce": ce, "latent": latent, "logits": logits}
