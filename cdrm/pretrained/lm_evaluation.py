"""Bounded native LM evaluation without the training-only NextLat predictor.

Each row is an independent document. CE selection follows the LM target-token
contract, and aggregate NLL is weighted by valid target tokens, never by batches
or document means. Full-vocabulary logits exist only for a position chunk.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .nextlat import NextLatBatch, NextLatLM
from .olmo_recurrent import OLMoRTForCausalLM
from .recurrent import RTMode


def _ce_mask(batch: NextLatBatch) -> torch.Tensor:
    """Validate document isolation and compute the next-token evaluation mask."""
    if not isinstance(batch, NextLatBatch):
        raise TypeError("Evaluation batches must be NextLatBatch instances")
    ids = batch.input_ids
    if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
        raise ValueError("input_ids must be nonempty int64 [batch, sequence]")
    for name, dtype in (("valid_mask", torch.bool), ("document_ids", torch.long),
                        ("ce_mask", torch.bool)):
        value = getattr(batch, name)
        if value is None and name == "ce_mask":
            continue
        if value is None or value.shape != ids.shape or value.dtype != dtype or value.device != ids.device:
            raise ValueError(f"{name} must have input_ids shape/device and dtype {dtype}")
    if bool((batch.document_ids[batch.valid_mask] < 0).any()):
        raise ValueError("Valid tokens require nonnegative document_ids")
    for documents, valid in zip(batch.document_ids, batch.valid_mask):
        if documents[valid].unique().numel() > 1:
            raise ValueError("Packed documents are unsupported: each row must contain one document")
    valid, docs = batch.valid_mask, batch.document_ids
    selected = valid[:, :-1] & valid[:, 1:] & (docs[:, :-1] == docs[:, 1:])
    if batch.ce_mask is not None:
        selected = selected & batch.ce_mask[:, 1:]
    return selected


def evaluate_batches(model: NextLatLM, batches: Iterable[NextLatBatch], *,
                     mode: RTMode | None = None, precision: str = "bf16_mixed",
                     chunk_size: int = 128, include_document_records: bool = False) -> dict:
    """Evaluate full documents and return JSON-compatible finite CE metrics.

    ``mode`` is required for an RT-capable backbone, including an empty-layer
    ordinary mode. A plain ordinary backbone accepts None or an empty RTMode.
    ``chunk_size`` counts selected positions, each projected against the whole
    tied vocabulary. Accuracy is a fraction in [0,1]; perplexity is exp(mean_nll).

    Batches must already reside on the backbone device. Only validity, document
    identity and CE target-position selection affect evaluation; auxiliary masks
    and the predictor are unused. One document per padded row is required.
    Optional document records describe each nonempty row occurrence, identified
    by batch/row indices and caller document_id; repeated IDs are not merged.
    A row without a CE target has count 0 and mean_nll None.

    All caller module training flags are restored even on failure, and parameter
    values/existing gradients are untouched. Evaluation itself draws no random
    numbers and disables dropout. The supplied batch iterator remains caller
    owned; any RNG used to generate its batches is outside this guarantee.
    Empty evaluation or nonfinite/overflowing metrics raise rather than emitting
    a misleading finite clamp. BF16 mixed execution requires CUDA.
    """
    if not isinstance(model, NextLatLM):
        raise TypeError("model must be NextLatLM")
    if precision not in ("fp32", "bf16_mixed"):
        raise ValueError("precision must be fp32 or bf16_mixed")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(include_document_records) is not bool:
        raise TypeError("include_document_records must be boolean")
    if mode is not None and not isinstance(mode, RTMode):
        raise TypeError("mode must be RTMode or None")
    recurrent = isinstance(model.backbone, OLMoRTForCausalLM)
    if recurrent and mode is None:
        raise ValueError("RT-capable evaluation requires an explicit RTMode")
    if not recurrent and mode is not None and mode.selected_layers:
        raise ValueError("An ordinary backbone cannot select recurrent layers")
    kwargs = {"mode": mode} if recurrent else {}
    weight = model.backbone.readout_weight
    device = weight.device
    if precision == "bf16_mixed" and device.type != "cuda":
        raise ValueError("BF16 mixed evaluation requires CUDA; use fp32 for intentional CPU checks")

    flags = [(module, module.training) for module in model.modules()]
    ce_sum, ce_count, correct, documents, input_tokens, batch_count = 0.0, 0, 0, 0, 0, 0
    records = []
    try:
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                selected = _ce_mask(batch)
                if batch.input_ids.device != device:
                    raise ValueError("Evaluation batches must be on the backbone device")
                batch_count += 1
                input_tokens += int(batch.valid_mask.sum())
                row_counts = selected.sum(dim=1)
                row_sums = torch.zeros(batch.input_ids.shape[0], device=device, dtype=torch.float64)
                row_correct = torch.zeros_like(row_counts)
                if bool(selected.any()):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16_mixed"):
                        hidden = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
                                                return_logits=False, **kwargs).last_hidden_state
                        states = hidden[:, :-1][selected]
                        targets = batch.input_ids[:, 1:][selected]
                        rows = selected.nonzero(as_tuple=True)[0]
                        for start in range(0, targets.numel(), chunk_size):
                            stop = start + chunk_size
                            logits = F.linear(states[start:stop], weight).float()
                            nll = F.cross_entropy(logits, targets[start:stop], reduction="none")
                            if not bool(torch.isfinite(nll).all()):
                                raise FloatingPointError("Evaluation produced nonfinite token NLL")
                            row_sums.index_add_(0, rows[start:stop], nll.double())
                            hits = logits.argmax(dim=-1).eq(targets[start:stop]).long()
                            row_correct.index_add_(0, rows[start:stop], hits)
                values = row_sums.cpu().tolist()
                counts = row_counts.cpu().tolist()
                hits = row_correct.cpu().tolist()
                ce_sum += math.fsum(values)
                ce_count += sum(counts)
                correct += sum(hits)
                for row_index, row_valid in enumerate(batch.valid_mask):
                    if not bool(row_valid.any()):
                        continue
                    documents += 1
                    if include_document_records:
                        count = counts[row_index]
                        records.append({
                            "batch_index": batch_index, "row_index": row_index,
                            "document_id": int(batch.document_ids[row_index][row_valid][0]),
                            "ce_sum": values[row_index], "ce_count": count,
                            "mean_nll": values[row_index] / count if count else None,
                            "next_token_correct": hits[row_index],
                        })
            if ce_count == 0:
                raise ValueError("Evaluation has no valid CE target positions")
            mean_nll = ce_sum / ce_count
            try:
                perplexity = math.exp(mean_nll)
            except OverflowError as error:
                raise FloatingPointError("Evaluation perplexity overflows finite JSON numbers") from error
            if not all(math.isfinite(value) for value in (ce_sum, mean_nll, perplexity)):
                raise FloatingPointError("Evaluation produced nonfinite summary metrics")
            result = {
                "schema": "olmo-lm-evaluation-v1", "precision": precision,
                "mode": None if mode is None else asdict(mode),
                "positions_per_chunk": chunk_size,
                "batches": batch_count, "documents": documents, "input_tokens": input_tokens,
                "ce_sum": ce_sum, "ce_count": ce_count, "mean_nll": mean_nll,
                "perplexity": perplexity, "next_token_correct": correct,
                "next_token_accuracy": correct / ce_count,
            }
            if include_document_records:
                result["document_records"] = records
            return result
    finally:
        # Calling train(flag) per module would recursively overwrite mixed
        # child states. Restore exactly the flag of each original module.
        for module, flag in flags:
            module.training = flag
