"""Bounded O3 fixtures and a dense loss oracle independent of chunked losses."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import torch
from torch.nn import functional as F

from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, NextLatLM
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_validation import PROMPTS

ROOT = Path(__file__).resolve().parents[1]


def build_model(state, *, enabled=True, chunk_size=8, backend="math", attention_precision="mixed"):
    backbone = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), attention_backend=backend,
                                     attention_precision=attention_precision, device="meta", dtype=torch.float32)
    backbone.load_state_dict(state, strict=True, assign=True)
    config = NextLatConfig(model_dim=backbone.config.model_dim, vocab_chunk_size=chunk_size)
    return NextLatLM(backbone, config, enabled=enabled).to("cuda").eval()


def verify_nextlat_sources():
    directory = ROOT / "cdrm/pretrained/_nextlat_reference"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["revision"] != "b37d3411ab9b17be8638abbddb9529f0f3a0a5f9":
        raise ValueError("Unexpected NextLat source revision")
    for row in manifest["files"]:
        data = (directory / row["file"]).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Pinned NextLat source bytes changed")
    return manifest


def fixture(tokenizer, *, batch_size=1, length=16, padded=False, device="cuda"):
    """One document/row; explicit EOS, right padding, independent response masks."""
    rows, masks, documents, responses = [], [], [], []
    for row in range(batch_size):
        count = length - (5 if padded and row % 2 else 0)
        text = (PROMPTS[row % len(PROMPTS)] + "\n") * (length // 20 + 2)
        tokens = tokenizer.encode(text)[:count - 1] + [50279]
        assert len(tokens) == count
        rows.append(tokens + [1] * (length - count))
        masks.append([True] * count + [False] * (length - count))
        documents.append([row] * count + [-1] * (length - count))
        # A literal prompt/response boundary, independently separate from the
        # all-position latent mask; no benchmark or SFT quality is implied.
        responses.append([False] * (count // 2) + [True] * (count - count // 2) + [False] * (length - count))
    tensor = lambda value, dtype: torch.tensor(value, device=device, dtype=dtype)
    return NextLatBatch(
        input_ids=tensor(rows, torch.long), valid_mask=tensor(masks, torch.bool),
        document_ids=tensor(documents, torch.long), ce_mask=tensor(responses, torch.bool),
        latent_mask=tensor(masks, torch.bool), kl_mask=tensor(responses, torch.bool),
    )


def fixture_record(batch):
    return {name: None if getattr(batch, name) is None else getattr(batch, name).detach().cpu().tolist()
            for name in ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")}


def dense_objective(model, batch, mode):
    """Independent dense vocabulary/coordinate reductions, for short checks only."""
    output = model.backbone(batch.input_ids, attention_mask=batch.valid_mask, mode=mode, return_logits=False)
    hidden = output.last_hidden_state
    weight = model.backbone.readout_weight
    valid, docs = batch.valid_mask, batch.document_ids
    pair = valid[:, :-1] & valid[:, 1:] & (docs[:, :-1] == docs[:, 1:])
    triple = pair[:, :-1] & pair[:, 1:]
    ce = pair if batch.ce_mask is None else pair & batch.ce_mask[:, 1:]
    latent = pair if batch.latent_mask is None else pair & batch.latent_mask[:, 1:]
    kl = triple if batch.kl_mask is None else triple & batch.kl_mask[:, 2:]
    counts = {"ce": int(ce.sum()), "latent": int(latent.sum()), "kl": int(kl.sum())}
    zero = hidden.sum() * 0
    logits = F.linear(hidden[:, :-1], weight).float()
    ce_sum = F.cross_entropy(logits[ce], batch.input_ids[:, 1:][ce], reduction="sum") if counts["ce"] else zero
    sums = {"ce": ce_sum, "latent": zero, "kl": zero}
    weights = model.objective_weights()
    if weights["latent"] or weights["kl"]:
        predicted = model.predictor(hidden[:, :-1], model.backbone.token_embeddings(batch.input_ids[:, 1:]))
        if weights["latent"] and counts["latent"]:
            errors = F.smooth_l1_loss(predicted.float(), hidden[:, 1:].detach().float(), beta=1.0, reduction="none").mean(-1)
            sums["latent"] = errors[latent].sum()
        if weights["kl"] and counts["kl"]:
            teacher = F.log_softmax(F.linear(hidden[:, 1:-1].detach(), weight.detach()).float(), dim=-1)
            predicted_logp = F.log_softmax(F.linear(predicted[:, :-1], weight.detach()).float(), dim=-1)
            errors = F.kl_div(predicted_logp, teacher, log_target=True, reduction="none").sum(-1)
            sums["kl"] = errors[kl].sum()
    for name in ("latent", "kl"):
        if not weights[name]:
            counts[name] = 0
    means = {name: value / max(1, counts[name]) for name, value in sums.items()}
    total = sum(weights[name] * value for name, value in means.items())
    return total, {"sums": sums, "means": means, "counts": counts, "weights": weights}


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digests(model):
    return {name: tensor_digest(value) for name, value in model.state_dict().items()}


def tree_digests(value):
    """Small exact-comparison manifest, including optimizer values without dumps."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {str(k): tree_digests(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_digests(v) for v in value]
    return value
