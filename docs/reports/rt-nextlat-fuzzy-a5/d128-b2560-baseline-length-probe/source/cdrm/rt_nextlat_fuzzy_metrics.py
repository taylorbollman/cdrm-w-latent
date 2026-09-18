"""Streaming, native-mask MAD fuzzy-recall metrics without model dependencies.

All token IDs here are the native MAD IDs (0..15). Model vocabulary offsets are
the caller's concern. Metadata selects reporting regions; it is never fed into
the model or used to replace MAD's native training/evaluation labels.
"""
from __future__ import annotations

import numpy as np

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX


DISTANCE_BINS = ((1, 16), (17, 64), (65, 128), (129, 256), (257, 512), (513, None))


def evaluation_metadata(dataset):
    """Build masks once; distance is between current/last matching key starts.

    A motif exact match requires every *native scored* value token in that
    key/value occurrence. Unseen terminal queries remain in the main metrics.
    Teacher-forced value prefixes are visible, so first-value-token accuracy is
    also reported separately from continuation-token accuracy.
    """
    if dataset.manifest["task"] != FUZZY_TASK:
        raise ValueError("Expected native fuzzy-in-context-recall data")
    answer = dataset.answer_labels != IGNORE_INDEX
    shape = answer.shape
    first = np.zeros(shape, dtype=bool)
    terminal = np.zeros(shape, dtype=bool)
    group_ids = np.full(shape, -1, dtype=np.int64)
    distances = np.full(shape, -1, dtype=np.int64)
    previous_occurrences = np.zeros(shape, dtype=np.int64)
    group_id = 0
    for row, record in enumerate(dataset.metadata):
        earlier = {}
        repetitions = {}
        for pair in record["pairs"]:
            key = tuple(pair["key"])
            start = int(pair["value_start"]) - 1
            stop = start + len(pair["value"])
            positions = np.arange(start, stop)
            if start < 0 or stop > shape[1]:
                raise ValueError("Fuzzy metadata contains invalid value positions")
            positions = positions[answer[row, positions]]
            if positions.size:
                if np.any(group_ids[row, positions] != -1):
                    raise ValueError("Overlapping scored fuzzy motifs")
                group_ids[row, positions] = group_id
                group_id += 1
                first[row, start] = answer[row, start]
                if pair["final_query"]:
                    terminal[row, positions] = True
                if key in earlier:
                    distances[row, positions] = int(pair["key_start"]) - earlier[key]
                previous_occurrences[row, positions] = repetitions.get(key, 0)
            if not pair["final_query"]:
                earlier[key] = int(pair["key_start"])
                repetitions[key] = repetitions.get(key, 0) + 1
    if not np.array_equal(group_ids >= 0, answer):
        raise ValueError("Fuzzy metadata does not cover exactly the native answer mask")
    available = dataset.oracle_available_mask
    if not np.array_equal((distances >= 1) & answer, available):
        raise ValueError("Historical-distance annotation disagrees with oracle availability")
    masks = {
        "answer": answer,
        "first_value_token": first,
        "continuation_value_token": answer & ~first,
        "terminal_probe": terminal,
        "terminal_first_value_token": terminal & first,
        "known_history": available,
        "unavailable_history": answer & ~available,
    }
    for low, high in DISTANCE_BINS:
        name = f"distance_{low}_{high}" if high is not None else f"distance_{low}_plus"
        masks[name] = answer & (distances >= low) & ((distances <= high) if high is not None else True)
    for count in (1, 2):
        masks[f"prior_occurrences_{count}"] = answer & (previous_occurrences == count)
    masks["prior_occurrences_3_plus"] = answer & (previous_occurrences >= 3)
    return {"masks": masks, "answer_group_ids": group_ids,
            "distance": distances, "previous_occurrences": previous_occurrences,
            "motifs": group_id,
            "distance_definition": "Current key start minus most recent earlier matching key start, in input tokens; unavailable mappings have distance -1."}


class FuzzyMetrics:
    """Accumulate full-sequence batches, with native predictions and optional CE.

    Example: ``meter.update(predictions, per_token_ce, indices=batch_indices)``.
    ``indices`` addresses rows in the original dataset; every update includes all
    T positions for those rows. ``compute()`` returns flat JSON/W&B-ready fields.
    CE may be unspecified outside scored positions. This class performs no
    softmax and never transfers tensors or launches model work.
    """

    def __init__(self, dataset, metadata=None):
        self.dataset = dataset
        self.metadata = evaluation_metadata(dataset) if metadata is None else metadata
        self.masks = self.metadata["masks"]
        self.answer_group_ids = self.metadata["answer_group_ids"]
        self.reset()

    def reset(self):
        self.counts = {name: {"correct": 0, "tokens": 0, "ce_sum": 0.0, "ce_tokens": 0}
                       for name in self.masks}
        self.examples = self.exact_sequences = self.motifs = self.exact_motifs = 0

    def update(self, predictions, per_token_ce=None, *, indices=None):
        predictions = np.asarray(predictions)
        if indices is None:
            indices = np.arange(len(self.dataset))
        elif isinstance(indices, slice):
            indices = np.arange(len(self.dataset))[indices]
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        if indices.ndim != 1 or not len(indices) or np.any((indices < 0) | (indices >= len(self.dataset))):
            raise ValueError("Expected a nonempty one-dimensional dataset row selection")
        labels = self.dataset.answer_labels[indices]
        if predictions.shape != labels.shape:
            raise ValueError("Predictions must have the complete selected [B,T] shape")
        valid = self.masks["answer"][indices]
        if np.any((predictions[valid] < 0) | (predictions[valid] >= 16)):
            raise ValueError("Scored predictions must be native MAD IDs 0..15")
        correct = predictions == labels
        ce = None if per_token_ce is None else np.asarray(per_token_ce, dtype=np.float64)
        if ce is not None and (ce.shape != labels.shape or not np.isfinite(ce[valid]).all()):
            raise ValueError("Per-token CE must match [B,T] and be finite on native scored positions")
        for name, full_mask in self.masks.items():
            mask = full_mask[indices]
            count = self.counts[name]
            tokens = int(mask.sum())
            count["correct"] += int((correct & mask).sum())
            count["tokens"] += tokens
            if ce is not None:
                count["ce_sum"] += float(ce[mask].sum())
                count["ce_tokens"] += tokens
        self.examples += len(indices)
        self.exact_sequences += int(np.all(correct | ~valid, axis=1).sum())
        # Local group IDs also distinguish repeated row selections in a batch.
        for row, dataset_row in enumerate(indices):
            groups = self.answer_group_ids[dataset_row, valid[row]]
            unique, inverse = np.unique(groups, return_inverse=True)
            incorrect = np.bincount(inverse, weights=(~correct[row, valid[row]]).astype(np.int64),
                                    minlength=len(unique))
            self.motifs += len(unique)
            self.exact_motifs += int((incorrect == 0).sum())

    def compute(self):
        result = {"examples": self.examples, "exact_sequences": self.exact_sequences,
                  "sequence_exact_match": self.exact_sequences / self.examples if self.examples else None,
                  "answer_motifs": self.motifs, "exact_answer_motifs": self.exact_motifs,
                  "answer_motif_exact_match": self.exact_motifs / self.motifs if self.motifs else None}
        for name, count in self.counts.items():
            result[f"{name}_correct"] = count["correct"]
            result[f"{name}_tokens"] = count["tokens"]
            result[f"{name}_accuracy"] = count["correct"] / count["tokens"] if count["tokens"] else None
            result[f"{name}_ce"] = count["ce_sum"] / count["ce_tokens"] if count["ce_tokens"] else None
            result[f"{name}_ce_tokens"] = count["ce_tokens"]
        result["oracle_coverage"] = (self.counts["known_history"]["tokens"] / self.counts["answer"]["tokens"]
                                     if self.counts["answer"]["tokens"] else None)
        return result
