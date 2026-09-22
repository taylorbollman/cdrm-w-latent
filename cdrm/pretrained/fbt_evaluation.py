"""Per-pass held-out NLL for FBT-only finite and exact-online execution.

Every pass is a separate inference measurement. The training objective's
pass-0-plus-extra-pass weighting never enters the reported token NLL.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .fbt_training import FBTNextLatLM
from .lm_evaluation import _ce_mask
from .nextlat import NextLatBatch
from .olmo_fbt import FBTMode, FBTOnlineMode, OLMoFBT


def _summary(ce_sum, ce_count, correct):
    if ce_count == 0:
        raise ValueError("Evaluation has no valid CE target positions")
    mean_nll = ce_sum / ce_count
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise FloatingPointError("Evaluation perplexity overflows finite JSON numbers") from error
    if not all(math.isfinite(value) for value in (ce_sum, mean_nll, perplexity)):
        raise FloatingPointError("Evaluation produced nonfinite summary metrics")
    return dict(ce_sum=ce_sum, ce_count=ce_count, mean_nll=mean_nll,
                perplexity=perplexity, next_token_correct=correct,
                next_token_accuracy=correct / ce_count)


def evaluate_fbt_batches(
    model: OLMoFBT | FBTNextLatLM, batches: Iterable[NextLatBatch], *,
    mode: FBTMode | FBTOnlineMode, precision: str = "bf16_mixed",
    chunk_size: int = 128, include_document_records: bool = False,
) -> dict:
    """Return full-vocabulary, token-weighted CE metrics for each finite pass.

    ``FBTMode`` selects finite shared-stack passes; ``FBTOnlineMode`` selects
    the exact online reference and returns one record with ``pass_index=None``.
    RT layer selections are rejected in this FBT-only pilot evaluator. K1 or
    disabled finite feedback returns one ordinary pass. No NextLat forward,
    predictor, auxiliary loss or aggregate multi-pass training loss is used.

    Batches are independent one-document padded rows on the model device.
    ``chunk_size`` bounds positions projected against the entire tied vocabulary.
    CE uses valid adjacent same-document pairs and target-token CE selection.
    Optional per-document records use the O4 names/IDs, with separate records
    for repeated document IDs across windows; callers may cluster by that ID.
    Unscored nonempty rows have count zero and ``mean_nll=None``.

    The driver owns any online row/length limit and must pass identical batches
    when comparing finite and online results. This function does not truncate,
    draw samples, consume RNG, or retain/export online caches between rows or
    batches. All module training flags are restored on success and failure;
    parameter values and existing gradient buffers remain untouched. The batch
    iterator is caller-owned and may itself consume RNG outside this guarantee.
    BF16 requires CUDA; intentional CPU checks use ``precision='fp32'``.
    """
    if isinstance(model, OLMoFBT):
        core = model
    elif isinstance(model, FBTNextLatLM) and isinstance(model.backbone, OLMoFBT):
        core = model.backbone
    else:
        raise TypeError("model must be OLMoFBT or FBTNextLatLM containing OLMoFBT")
    if not isinstance(mode, (FBTMode, FBTOnlineMode)):
        raise TypeError("mode must be FBTMode or FBTOnlineMode")
    if mode.rt_mode.selected_layers:
        raise ValueError("This FBT-only evaluator requires an empty RT layer selection")
    if precision not in ("fp32", "bf16_mixed"):
        raise ValueError("precision must be fp32 or bf16_mixed")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(include_document_records) is not bool:
        raise TypeError("include_document_records must be boolean")
    weight = core.readout_weight
    device = weight.device
    if precision == "bf16_mixed" and device.type != "cuda":
        raise ValueError("BF16 mixed evaluation requires CUDA; use fp32 for intentional CPU checks")
    online = isinstance(mode, FBTOnlineMode)
    pass_count = 1 if online or not mode.enabled else mode.num_passes
    totals = [dict(ce_sum=0.0, ce_count=0, correct=0, records=[]) for _ in range(pass_count)]
    batch_count = documents = input_tokens = 0
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                selected = _ce_mask(batch)
                if batch.input_ids.device != device:
                    raise ValueError("Evaluation batches must be on the backbone device")
                batch_count += 1
                input_tokens += int(batch.valid_mask.sum())
                document_rows = [i for i, valid in enumerate(batch.valid_mask) if bool(valid.any())]
                documents += len(document_rows)
                row_counts = selected.sum(dim=1).cpu().tolist()
                target_count = sum(row_counts)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16_mixed"):
                    states = None
                    if target_count:
                        kwargs = dict(attention_mask=batch.valid_mask, document_ids=batch.document_ids,
                                      mode=mode, return_logits=False)
                        if online:
                            output = core.forward_online(batch.input_ids, use_cache=False, **kwargs)
                            states = (output.last_hidden_state,)
                        else:
                            output = core(batch.input_ids, **kwargs)
                            states = output.pass_hidden_states
                        if not isinstance(states, tuple) or len(states) != pass_count:
                            raise ValueError("FBT execution returned an unexpected number of pass states")
                    targets = batch.input_ids[:, 1:][selected]
                    rows = selected.nonzero(as_tuple=True)[0]
                    for pass_index, total in enumerate(totals):
                        row_sums = torch.zeros(batch.input_ids.shape[0], device=device, dtype=torch.float64)
                        row_correct = torch.zeros(batch.input_ids.shape[0], device=device, dtype=torch.long)
                        if states is not None:
                            hidden = states[pass_index][:, :-1][selected]
                            for start in range(0, target_count, chunk_size):
                                stop = start + chunk_size
                                logits = F.linear(hidden[start:stop], weight).float()
                                nll = F.cross_entropy(logits, targets[start:stop], reduction="none")
                                if not bool(torch.isfinite(nll).all()):
                                    raise FloatingPointError("Evaluation produced nonfinite token NLL")
                                row_sums.index_add_(0, rows[start:stop], nll.double())
                                hits = logits.argmax(-1).eq(targets[start:stop]).long()
                                row_correct.index_add_(0, rows[start:stop], hits)
                        values = row_sums.cpu().tolist()
                        hits = row_correct.cpu().tolist()
                        total["ce_sum"] += math.fsum(values)
                        total["ce_count"] += target_count
                        total["correct"] += sum(hits)
                        if include_document_records:
                            for row_index in document_rows:
                                count = row_counts[row_index]
                                valid = batch.valid_mask[row_index]
                                total["records"].append({
                                    "batch_index": batch_index, "row_index": row_index,
                                    "document_id": int(batch.document_ids[row_index][valid][0]),
                                    "ce_sum": values[row_index], "ce_count": count,
                                    "mean_nll": values[row_index] / count if count else None,
                                    "next_token_correct": hits[row_index],
                                })
            execution = "online" if online else "finite"
            metadata = dict(precision=precision, mode=asdict(mode), execution=execution,
                            positions_per_chunk=chunk_size, batches=batch_count,
                            documents=documents, input_tokens=input_tokens)
            passes = []
            for index, total in enumerate(totals):
                result = dict(schema="olmo-fbt-pass-evaluation-v1", **metadata,
                              pass_index=None if online else index,
                              **_summary(total["ce_sum"], total["ce_count"], total["correct"]))
                if include_document_records:
                    result["document_records"] = total["records"]
                passes.append(result)
            return dict(schema="olmo-fbt-evaluation-v1", **metadata, passes=passes)
    finally:
        for module, flag in flags:
            module.training = flag
