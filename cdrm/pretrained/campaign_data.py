"""Portable campaign data contracts over ordered, already-tokenized documents.

This module does not download/tokenize Dolma, select a source mixture, cycle an
exhausted corpus, or start distributed workers. A future disk-backed loader can
use the same window/update contract. All identifiers and counters are independent
of physical batching and rank count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import struct
from typing import Iterable, Sequence

SCHEMA = "olmo-campaign-data-v1"


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _token_digest(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


@dataclass(frozen=True)
class SourcePin:
    """One immutable source artifact, not a mutable dataset branch alone."""

    name: str
    uri: str
    revision: str
    sha256: str

    def __post_init__(self):
        for name in ("name", "uri", "revision"):
            _string(getattr(self, name), name)
        _sha(self.sha256, "source sha256")


@dataclass(frozen=True)
class TokenizerPin:
    repo: str
    revision: str
    sha256: str
    add_special_tokens: bool = False

    def __post_init__(self):
        _string(self.repo, "tokenizer repo")
        _string(self.revision, "tokenizer revision")
        _sha(self.sha256, "tokenizer sha256")
        if self.add_special_tokens is not False:
            raise ValueError("Tokenize without implicit special-token insertion")


@dataclass(frozen=True)
class TokenizedDocument:
    source: SourcePin
    document_id: str
    split: str
    text_sha256: str
    tokens: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.source, SourcePin):
            raise TypeError("source must be SourcePin")
        _string(self.document_id, "document_id")
        _string(self.split, "split")
        _sha(self.text_sha256, "document text_sha256")
        object.__setattr__(self, "tokens", tuple(self.tokens))
        if not self.tokens or any(type(t) is not int or not 0 <= t < 2**32 for t in self.tokens):
            raise ValueError("Document tokens must be a nonempty sequence of uint32 integers")


@dataclass(frozen=True)
class TokenCounts:
    """Corpus-unique means first presentation in this noncycling ordered stream.

    Overlap is charged even when its earlier presentation was in the previous
    logical update. Counts are for one model pass; FBT does not multiply the
    campaign input-token budget.
    """

    rows: int = 0
    new_unique_tokens: int = 0
    presented_tokens: int = 0
    overlap_tokens: int = 0
    ce_targets: int = 0
    latent_pairs: int = 0
    kl_triples: int = 0
    omitted_boundary_kl_triples: int = 0

    def __add__(self, other: TokenCounts) -> TokenCounts:
        if not isinstance(other, TokenCounts):
            return NotImplemented
        return TokenCounts(**{key: getattr(self, key) + getattr(other, key) for key in self.__dataclass_fields__})


@dataclass(frozen=True)
class DocumentWindow:
    key: str
    split: str
    document_index: int
    start: int
    tokens: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def counts(self) -> TokenCounts:
        overlap = int(self.start > 0)
        return TokenCounts(1, self.length - overlap, self.length, overlap,
                           self.length - 1, self.length - 1, max(self.length - 2, 0), overlap)


def count_windows(rows: Iterable[DocumentWindow]) -> TokenCounts:
    total = TokenCounts()
    for row in rows:
        total = total + row.counts
    return total


@dataclass(frozen=True)
class DataCursor:
    manifest_sha256: str
    split: str
    next_window: int = 0


@dataclass(frozen=True)
class LogicalUpdate:
    rows: tuple[DocumentWindow, ...]
    start_cursor: DataCursor
    next_cursor: DataCursor
    target_valid_tokens: int

    @property
    def counts(self) -> TokenCounts:
        return count_windows(self.rows)

    @property
    def reaches_target(self) -> bool:
        return self.counts.presented_tokens >= self.target_valid_tokens


class CampaignData:
    """A bounded in-memory reference adapter; caller supplies canonical order.

    Exact duplicate raw text or normalized tokens are rejected, within and across
    splits, rather than silently deduplicated/reassigned. Provenance hashes are
    declarations from the caller: this adapter cannot verify absent raw files.
    """

    def __init__(self, documents: Iterable[TokenizedDocument], *, tokenizer: TokenizerPin,
                 length: int = 1024, eos_id: int = 50279, pad_id: int = 1, vocab_size: int = 50304):
        if not isinstance(tokenizer, TokenizerPin):
            raise TypeError("tokenizer must be TokenizerPin")
        if type(length) is not int or not 3 <= length <= 1024:
            raise ValueError("length must be an integer in 3..1024")
        _positive(vocab_size, "vocab_size")
        if any(type(t) is not int or not 0 <= t < vocab_size for t in (eos_id, pad_id)) or eos_id == pad_id:
            raise ValueError("EOS and PAD must be distinct in-vocabulary integer IDs")
        self.length, self.eos_id, self.pad_id = length, eos_id, pad_id
        self._windows: dict[str, tuple[DocumentWindow, ...]] = {}
        rows_by_split: dict[str, list[DocumentWindow]] = {}
        sources, records, identities, texts, token_hashes = {}, [], set(), set(), set()
        for index, document in enumerate(documents):
            if not isinstance(document, TokenizedDocument):
                raise TypeError("documents must contain TokenizedDocument values")
            identity = (document.source.name, document.document_id)
            if identity in identities:
                raise ValueError("Duplicate source/document identity")
            identities.add(identity)
            source = asdict(document.source)
            if document.source.name in sources and sources[document.source.name] != source:
                raise ValueError("Conflicting source provenance for one source name")
            sources[document.source.name] = source
            tokens = document.tokens
            if any(t >= vocab_size for t in tokens):
                raise ValueError("Document contains an out-of-vocabulary token")
            # A tokenizer/caller may already have supplied the actual terminal
            # EOS. Keep exactly that one, never append it again or silently strip
            # internal EOS that could mark a packed-document boundary.
            has_eos = tokens[-1] == eos_id
            content = tokens[:-1] if has_eos else tokens
            if not content or eos_id in content:
                raise ValueError("Document must have content and at most one terminal EOS")
            normalized = tokens if has_eos else tokens + (eos_id,)
            token_sha = _token_digest(normalized)
            if document.text_sha256 in texts or token_sha in token_hashes:
                raise ValueError("Duplicate document text/tokens within or across splits")
            texts.add(document.text_sha256)
            token_hashes.add(token_sha)
            record = {"source": document.source.name, "document_id": document.document_id,
                      "split": document.split, "text_sha256": document.text_sha256,
                      "token_sha256": token_sha, "token_count": len(normalized), "eos_appended": not has_eos}
            records.append(record)
            start = 0
            while start < len(normalized) - 1:
                window_tokens = normalized[start:start + length]
                key = _digest({"source": source, "document": record, "tokenizer": asdict(tokenizer),
                               "start": start, "length": len(window_tokens)})
                rows_by_split.setdefault(document.split, []).append(
                    DocumentWindow(key, document.split, index, start, window_tokens))
                if start + len(window_tokens) == len(normalized):
                    break
                start += length - 1
        if not records:
            raise ValueError("At least one document is required")
        self._windows = {split: tuple(rows) for split, rows in rows_by_split.items()}
        self._window_by_key = {row.key: row for rows in self._windows.values() for row in rows}
        manifest = {"schema": SCHEMA, "length": length, "eos_id": eos_id, "pad_id": pad_id,
                    "vocab_size": vocab_size, "tokenizer": asdict(tokenizer),
                    "sources": [sources[name] for name in sorted(sources)], "documents": records,
                    "preprocessing": {"order": "caller supplied; no shuffle or cycling", "packing": False,
                                      "window_context_overlap": 1, "window_stride": length - 1,
                                      "eos": "append only if absent at actual document end; reject internal EOS",
                                      "budget_unit": "valid presented input tokens including overlap and EOS",
                                      "duplicate_policy": "reject exact raw-text and normalized-token duplicates"},
                    "splits": {split: asdict(count_windows(rows)) for split, rows in self._windows.items()}}
        self._manifest_json = _json(manifest)
        self.manifest_sha256 = hashlib.sha256(self._manifest_json.encode("utf-8")).hexdigest()

    @property
    def manifest(self) -> dict:
        return json.loads(self._manifest_json)

    def windows(self, split: str) -> tuple[DocumentWindow, ...]:
        if split not in self._windows:
            raise ValueError("Unknown split")
        return self._windows[split]

    def cursor(self, split: str) -> DataCursor:
        self.windows(split)
        return DataCursor(self.manifest_sha256, split)

    def restore_cursor(self, record: dict) -> DataCursor:
        if not isinstance(record, dict) or set(record) != {"manifest_sha256", "split", "next_window"}:
            raise ValueError("Invalid data cursor fields")
        cursor = DataCursor(**record)
        self._check_cursor(cursor)
        return cursor

    def _check_cursor(self, cursor: DataCursor) -> None:
        if not isinstance(cursor, DataCursor) or cursor.manifest_sha256 != self.manifest_sha256:
            raise ValueError("Cursor manifest fingerprint differs")
        rows = self.windows(cursor.split)
        if type(cursor.next_window) is not int or not 0 <= cursor.next_window <= len(rows):
            raise ValueError("Cursor window position is out of bounds")

    def next_update(self, cursor: DataCursor, target_valid_tokens: int) -> LogicalUpdate:
        """Consume whole rows until target or exhaustion; never drop the tail.

        Overshoot is strictly less than maximum row length. A final short update
        is returned explicitly (reaches_target=False); the following call raises
        StopIteration. Save next_cursor only with its completed optimizer update.
        """
        self._check_cursor(cursor)
        _positive(target_valid_tokens, "target_valid_tokens")
        rows = self.windows(cursor.split)
        if cursor.next_window == len(rows):
            raise StopIteration("Prepared data exhausted; implicit cycling is forbidden")
        end, count = cursor.next_window, 0
        while end < len(rows) and count < target_valid_tokens:
            count += rows[end].length
            end += 1
        return LogicalUpdate(rows[cursor.next_window:end], cursor,
                             DataCursor(self.manifest_sha256, cursor.split, end), target_valid_tokens)

    def batch(self, rows: Sequence[DocumentWindow], *, pad_to: int | None = None,
              physical_batch_size: int | None = None):
        """Right-pad real rows, optionally adding fully loss-masked dummy rows.

        Dummy rows execute no valid token or target. Their execution/collective
        handling must be qualified by the trainer; this is only the data contract.
        An empty rank slot requires an explicit positive physical_batch_size.
        """
        import torch
        from .nextlat import NextLatBatch

        rows = tuple(rows)
        width = self.length if pad_to is None else pad_to
        _positive(width, "pad_to")
        if width > self.length or any(not isinstance(row, DocumentWindow) or row.length > width for row in rows):
            raise ValueError("Rows must be DocumentWindow values fitting the configured padding width")
        if any(self._window_by_key.get(row.key) != row for row in rows):
            raise ValueError("Window provenance differs from this prepared corpus")
        size = len(rows) if physical_batch_size is None else physical_batch_size
        _positive(size, "physical_batch_size")
        if size < len(rows):
            raise ValueError("physical_batch_size cannot discard rows")
        ids = torch.full((size, width), self.pad_id, dtype=torch.long)
        valid = torch.zeros((size, width), dtype=torch.bool)
        docs = torch.full((size, width), -1, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :row.length] = torch.tensor(row.tokens, dtype=torch.long)
            valid[index, :row.length] = True
            docs[index, :row.length] = row.document_index
        return NextLatBatch(ids, valid, docs)


def partition_update(update: LogicalUpdate, *, world_size: int, physical_batch_size: int
                     ) -> tuple[tuple[tuple[DocumentWindow, ...], ...], ...]:
    """Return [microstep][rank][row], including explicit empty final rank slots.

    Flattening microstep, rank, then row exactly restores the canonical update.
    Every rank receives the same microstep count; neither duplication nor rank
    count changes which examples the optimizer update consumes.
    """
    _positive(world_size, "world_size")
    _positive(physical_batch_size, "physical_batch_size")
    if not isinstance(update, LogicalUpdate) or not update.rows:
        raise ValueError("update must be a nonempty LogicalUpdate")
    capacity = world_size * physical_batch_size
    return tuple(tuple(update.rows[start + rank * physical_batch_size:start + (rank + 1) * physical_batch_size]
                       for rank in range(world_size)) for start in range(0, len(update.rows), capacity))
