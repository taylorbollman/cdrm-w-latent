"""NextLat's A5 auxiliary objective for the existing SEQ/RT backbones.

Inference runs the backbone only. The predictor is a training-time module;
there is no free-running latent evaluation or KL training objective here.

The dynamics network and loss semantics are adapted from Jayden Teoh's
NextLat release, commit b37d3411ab9b17be8638abbddb9529f0f3a0a5f9,
models/model_nextlat.py and config/a5/nextlat_a5.yaml:
https://github.com/JaydenTeoh/NextLat/tree/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9

Upstream license (MIT):
Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from scripts.rt_a5_common import build_model, task_loss


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "configs/rt_a5_nextlat/base.json"


def nextlat_configuration(width: int, predictor_hidden_width: int | None = None) -> dict:
    """Resolve the pinned A5 predictor configuration without changing RT config."""
    config = json.loads(BASE_CONFIG_PATH.read_text())
    predictor = config["predictor"]
    if predictor_hidden_width is None:
        multiple = predictor["hidden_width_multiple"]
        predictor_hidden_width = multiple * round(
            predictor["projection_factor"] * 2 * width / multiple)
    if type(predictor_hidden_width) is not int or predictor_hidden_width < 1:
        raise ValueError("predictor_hidden_width must be positive; for D64 supply an explicit width")
    config["backbone_width"] = width
    predictor.update(input_width=2 * width, hidden_width=predictor_hidden_width,
                     output_width=width)
    return config


class NextLatDynamicsModel(nn.Module):
    """Released A5 residual GELU MLP with RMSNorm on the concatenated input."""

    def __init__(self, width: int, hidden_width: int, *, epsilon: float = 1e-5,
                 initialization_std: float = 0.02):
        super().__init__()
        if type(width) is not int or width < 1 or type(hidden_width) is not int or hidden_width < 1:
            raise ValueError("Latent and predictor widths must be positive integers")
        self.width = width
        self.hidden_width = hidden_width
        self.norm_x = nn.RMSNorm(2 * width, eps=epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(2 * width, hidden_width, bias=False), nn.GELU(),
            nn.Linear(hidden_width, hidden_width, bias=False), nn.GELU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=initialization_std)

    def forward(self, current_states: torch.Tensor,
                next_token_embeds: torch.Tensor) -> torch.Tensor:
        if current_states.shape != next_token_embeds.shape or current_states.shape[-1] != self.width:
            raise ValueError("Current latents and next embeddings must share shape [..., width]")
        joined = torch.cat((next_token_embeds, current_states), dim=-1)
        return current_states + self.mlp(self.norm_x(joined))


def _parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class A5NextLat(nn.Module):
    """Training wrapper whose ordinary forward never invokes its predictor."""

    def __init__(self, backbone: nn.Module, predictor: NextLatDynamicsModel):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def forward(self, inputs: torch.Tensor):
        return self.backbone(inputs)


def build_nextlat_model(architecture: str, width: int = 512, seed: int = 1234,
                        predictor_seed: int = 1235, device: str | torch.device = "cpu",
                        *, backend: str = "tiled",
                        predictor_hidden_width: int | None = None) -> A5NextLat:
    """Pair backbone initialization with pure RT/SEQ and isolate predictor RNG."""
    config = nextlat_configuration(width, predictor_hidden_width)
    backbone = build_model(architecture, width=width, seed=seed, device="cpu", backend=backend)
    predictor_config = config["predictor"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(predictor_seed)
        predictor = NextLatDynamicsModel(
            width, predictor_config["hidden_width"],
            epsilon=predictor_config["normalization_epsilon"],
            initialization_std=predictor_config["initialization_std"],
        ).to(dtype=torch.float32)
    model = A5NextLat(backbone, predictor)
    predictor_count = sum(p.numel() for p in predictor.parameters())
    model.nextlat_config = config
    model.nextlat_initialization = {
        "schema": "rt-a5-nextlat-initialization-v1", "architecture": architecture,
        "seed": seed, "predictor_seed": predictor_seed, "width": width,
        "backbone": backbone.a5_initialization,
        "canonical_sha256": backbone.a5_initialization["canonical_sha256"],
        "backbone_parameter_count": backbone.a5_initialization["parameter_count"],
        "predictor_parameter_count": predictor_count,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "predictor_sha256": _parameter_sha256(predictor),
        "predictor_config": predictor_config,
        "rule": "Original canonical SEQ/RT initialization; separate CPU predictor RNG scope",
    }
    return model.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _one_step_diagnostics(model: A5NextLat, hidden: torch.Tensor,
                          predicted: torch.Tensor, logits: torch.Tensor,
                          labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Detached scalar training diagnostics; no recurrent predictor rollout."""
    hidden, predicted, logits = hidden.detach(), predicted.detach(), logits.detach()
    target = hidden[:, 1:]
    difference = predicted - target
    predicted_logits = F.linear(predicted, model.backbone.transformer.ff_out.weight.detach())
    teacher_logp = F.log_softmax(logits[:, 1:], dim=-1)
    predicted_logp = F.log_softmax(predicted_logits, dim=-1)
    return {
        "latent_rms": hidden.square().mean().sqrt(),
        "predicted_latent_rms": predicted.square().mean().sqrt(),
        "latent_variation_rms": hidden.flatten(0, 1).var(dim=0, correction=0).mean().sqrt(),
        "latent_relative_l2": difference.norm() / target.norm().clamp_min(1e-12),
        "latent_cosine": F.cosine_similarity(predicted, target, dim=-1).mean(),
        "predicted_state_ce": F.cross_entropy(predicted_logits.reshape(-1, predicted_logits.shape[-1]),
                                              labels[:, 1:].reshape(-1)),
        "teacher_predicted_kl": F.kl_div(predicted_logp, teacher_logp, log_target=True,
                                         reduction="none").sum(dim=-1).mean(),
    }


def nextlat_objective(model: A5NextLat, inputs: torch.Tensor, labels: torch.Tensor,
                     latent_weight: float = 1.0, diagnostics: bool = False) -> dict:
    """Same-position state CE plus attached one-step prediction of stopped h[t+1].

    The regression mean includes every latent coordinate in every within-word
    transition. Source latents and the shared next-operation embeddings retain
    gradients. The target role alone is detached. Setting the auxiliary weight
    to zero bypasses the predictor entirely, including diagnostic computation.
    """
    if not isinstance(latent_weight, (int, float)) or not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if inputs.ndim != 2 or inputs.shape != labels.shape or not inputs.numel():
        raise ValueError("Inputs and same-position labels must be nonempty [batch, length] tensors")
    if inputs.dtype != torch.long or labels.dtype != torch.long:
        raise TypeError("A5 inputs and labels must have int64 dtype")
    if latent_weight > 0 and inputs.shape[1] < 2:
        raise ValueError("A positive NextLat objective needs at least two operations")
    output = model.backbone(inputs, return_pre_logits=latent_weight > 0)
    state_loss = task_loss(output.logits, labels)
    if latent_weight == 0:
        return {"loss": state_loss, "state_loss": state_loss,
                "latent_loss": state_loss.new_zeros(()), "logits": output.logits,
                "diagnostics": {}}
    hidden = output.pre_logits
    if hidden is None or hidden.dtype != torch.float32:
        raise TypeError("NextLat requires attached FP32 post-final-normalization latents")
    next_embeddings = model.backbone.transformer.wte(inputs[:, 1:])
    predicted = model.predictor(hidden[:, :-1], next_embeddings)
    latent_loss = F.smooth_l1_loss(predicted, hidden[:, 1:].detach(), beta=1.0, reduction="mean")
    result = {"loss": state_loss + latent_weight * latent_loss,
              "state_loss": state_loss, "latent_loss": latent_loss,
              "logits": output.logits, "diagnostics": {}}
    if diagnostics:
        result["diagnostics"] = _one_step_diagnostics(model, hidden, predicted, output.logits, labels)
    return result
