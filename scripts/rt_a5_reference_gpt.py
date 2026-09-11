"""Released NextLat A5 GPT architecture, using our existing training harness.

Only model math is ported. No upstream trainer, optimizer, data loader, loss
or evaluation code is used. The caller selects our existing FP32 math-SDPA
context and uses the same unshifted A5 CE as the other experiments.

Adapted from NextLat commit b37d3411ab9b17be8638abbddb9529f0f3a0a5f9,
models/model_gpt.py and models/model_base.py:
https://github.com/JaydenTeoh/NextLat/tree/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9

MIT License
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
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rt_a5_reference/base.json"
REFERENCE_COMMIT = "b37d3411ab9b17be8638abbddb9529f0f3a0a5f9"
REFERENCE_SOURCES = {
    "models/model_gpt.py": "1913ba5c514ce0f5f4accc06ac09c93f33ea9fe947d42ff251bbadb97ead3619",
    "models/model_base.py": "3c22766cf04ddd84ff18b1d207e287e5399324cd2005c6622bf473d3c199ed4a",
}


def reference_config(width: int = 512) -> dict:
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64")
    config = json.loads(CONFIG_PATH.read_text())
    config.update(n_embd=width, n_head=width // 64,
                  mlp_hidden_width=128 * round((8 * width / 3) / 128))
    return config


def reference_parameter_count(width: int = 512) -> int:
    config = reference_config(width)
    # Two blocks: four attention matrices, three SwiGLU matrices, two norms.
    # Then untied embedding/head and one final norm.
    return (config["n_layer"] * (4 * width * width + 3 * width * config["mlp_hidden_width"] + 2 * width)
            + 2 * config["vocab_size"] * width + width)


class ReferenceRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.eps = epsilon
        self.weight = nn.Parameter(torch.ones(width))
        self.register_parameter("bias", None)

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class ReferenceRotaryEmbedding:
    """Source RoPE: positions 1..T, full head, split halves, FP32 frequencies."""

    def __init__(self, max_seq_len, head_dim, base_freq=10000):
        self.max_seq_len = (max_seq_len + 255) // 256 * 256
        self.head_dim = head_dim
        self.base_freq = base_freq
        self.cos_lookup = None
        self.sin_lookup = None

    @torch.no_grad()
    def __call__(self, positions):
        if self.cos_lookup is None or self.cos_lookup.device != positions.device:
            with torch.autocast(device_type=positions.device.type, enabled=False):
                dims = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=positions.device).float()
                frequencies = 1.0 / (self.base_freq ** (dims / self.head_dim))
                sequence = torch.arange(self.max_seq_len, dtype=torch.int64, device=positions.device).float()
                angles = torch.outer(sequence, frequencies)
                angles = torch.cat((angles, angles), dim=-1)
                self.cos_lookup, self.sin_lookup = angles.cos(), angles.sin()
        return self.cos_lookup[positions], self.sin_lookup[positions]

    @staticmethod
    def apply(q, k, rope):
        cosine, sine = (value.unsqueeze(1) for value in rope)
        def rotate_half(value):
            left, right = value.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)
        return q * cosine + rotate_half(q) * sine, k * cosine + rotate_half(k) * sine


class ReferenceAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["n_embd"]
        # Preserve source construction and module registration order for RNG.
        self.c_attn = nn.Linear(width, 3 * width, bias=False)
        self.c_proj = nn.Linear(width, width, bias=False)
        self.attn_dropout = nn.Identity()
        self.resid_dropout = nn.Identity()
        self.n_head, self.n_embd = config["n_head"], width

    def forward(self, x, mask, rope):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = width // self.n_head
        k = k.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        q = q.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        q, k = ReferenceRotaryEmbedding.apply(q, k, rope)
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.view(batch, 1, length, length), dropout_p=0.0, is_causal=False)
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(attention))


class ReferenceSwiGLU(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden_width, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return F.silu(gate) * up


class ReferenceMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = ReferenceSwiGLU(config["n_embd"], config["mlp_hidden_width"])
        self.down = nn.Linear(config["mlp_hidden_width"], config["n_embd"], bias=False)
        self.dropout = nn.Identity()

    def forward(self, x):
        return self.dropout(self.down(self.up(x)))


class ReferenceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = ReferenceRMSNorm(config["n_embd"], config["normalization_epsilon"])
        self.attn = ReferenceAttention(config)
        self.ln_2 = ReferenceRMSNorm(config["n_embd"], config["normalization_epsilon"])
        self.mlp = ReferenceMLP(config)

    def forward(self, x, mask, rope):
        x = x + self.attn(self.ln_1(x), mask, rope)
        return x + self.mlp(self.ln_2(x))


class ReferenceGPTOutput(NamedTuple):
    logits: torch.Tensor
    pre_logits: torch.Tensor | None = None


class ReferenceGPT(nn.Module):
    """Unpadded A5 operations in, same-position state logits out."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.rotary_embedding = ReferenceRotaryEmbedding(
            config["block_size"] + 1, config["n_embd"] // config["n_head"], config["rope_base"])
        self.transformer = nn.ModuleDict(dict(
            blocks=nn.ModuleList([ReferenceBlock(config) for _ in range(config["n_layer"])]),
            norm=ReferenceRMSNorm(config["n_embd"], config["normalization_epsilon"]),
        ))
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config["initialization_std"])

    def forward(self, inputs, *, return_pre_logits=False):
        if inputs.ndim != 2 or not inputs.numel() or inputs.shape[1] > self.config["block_size"]:
            raise ValueError("Expected nonempty [batch, length] operations within the context cap")
        if inputs.dtype != torch.long:
            raise TypeError("A5 operation IDs must be int64")
        batch, length = inputs.shape
        # There are no document delimiters in A5. Identity ID 0 remains ordinary.
        # Source create_position_indices() gives 1..T when EOS sentinel 60 is absent.
        positions = torch.arange(1, length + 1, device=inputs.device).unsqueeze(0).expand(batch, -1)
        rope = self.rotary_embedding(positions)
        mask = torch.ones(batch, length, length, dtype=torch.bool, device=inputs.device).tril()
        x = self.token_embedding(inputs)
        for block in self.transformer.blocks:
            x = block(x, mask, rope)
        hidden = self.transformer.norm(x)
        return ReferenceGPTOutput(self.lm_head(hidden), hidden if return_pre_logits else None)


def reference_parameter_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_reference_gpt(width=512, seed=1234, device="cpu"):
    """Build with source CPU initialization order, preserving caller RNG state."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = reference_config(width)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ReferenceGPT(config).to(dtype=torch.float32)
    count = sum(p.numel() for p in model.parameters())
    if count != reference_parameter_count(width):
        raise AssertionError("Reference architecture parameter count differs")
    model.a5_initialization = {
        "schema": "rt-a5-reference-gpt-initialization-v1", "architecture": "reference_gpt",
        "seed": seed, "width": width, "parameter_count": count,
        "canonical_sha256": reference_parameter_sha256(model), "conversion": None,
        "reference_commit": REFERENCE_COMMIT, "reference_sources": dict(REFERENCE_SOURCES),
        "initialization_family": "released_gpt_normal_0.02",
        "rule": "CPU source module construction/reset order; all Linear/Embedding weights Normal(0,.02), norm scales one; no cross-architecture weight pairing claim",
    }
    return model.to(device=device, dtype=torch.float32)
