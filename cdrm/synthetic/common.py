"""Shared data contract; generation and audits do not require CUDA or PyTorch."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json

import numpy as np


@dataclass
class SyntheticBatch:
    input_ids: np.ndarray
    labels: np.ndarray
    metadata: list[dict]

    def __post_init__(self):
        self.input_ids = np.asarray(self.input_ids, dtype=np.int64)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if self.input_ids.ndim != 2 or self.labels.shape != self.input_ids.shape:
            raise ValueError("Expected equally shaped [examples, sequence] input IDs and aligned labels")
        if len(self.metadata) != len(self.input_ids):
            raise ValueError("Every example needs metadata")
        if np.any(self.input_ids < 0) or np.any((self.labels < 0) & (self.labels != -100)):
            raise ValueError("Negative IDs are reserved for ignored labels (-100)")
        if len(self.input_ids) and np.any((self.labels != -100).sum(axis=1) == 0):
            raise ValueError("Every synthetic example must contain at least one scored answer")

    def take(self, indices):
        indices = np.arange(len(self.input_ids))[indices]
        indices = np.atleast_1d(indices)
        return SyntheticBatch(self.input_ids[indices], self.labels[indices],
                              [self.metadata[int(i)] for i in indices])

    @property
    def sha256(self):
        digest = hashlib.sha256()
        for array in (self.input_ids, self.labels):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def concatenate(batches):
    return SyntheticBatch(np.concatenate([b.input_ids for b in batches]),
                          np.concatenate([b.labels for b in batches]),
                          [m for b in batches for m in b.metadata])


def save_batch(path: Path, batch: SyntheticBatch):
    """Store fixed fixtures without Python pickle; metadata is a separate JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input_ids=batch.input_ids, labels=batch.labels)
    path.with_suffix(".metadata.json").write_text(json.dumps(batch.metadata, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": batch.sha256, "examples": len(batch.input_ids),
            "sequence_length": int(batch.input_ids.shape[1]),
            "supervised_targets": int((batch.labels != -100).sum())}


def load_batch(path: Path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return SyntheticBatch(data["input_ids"], data["labels"],
                              json.loads(path.with_suffix(".metadata.json").read_text()))


def score_predictions(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and aligned labels must have the same shape")
    valid = labels != -100
    count = int(valid.sum())
    if not count or np.any(valid.sum(axis=1) == 0):
        raise ValueError("Answer metrics require supervised answers in every example")
    correct = (predictions == labels) & valid
    return {"answer_accuracy": float(correct.sum() / count),
            "sequence_accuracy": float(np.all(correct | ~valid, axis=1).mean()),
            "correct_answers": int(correct.sum()), "supervised_targets": count,
            "examples": len(labels)}
