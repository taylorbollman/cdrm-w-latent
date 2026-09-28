"""Bounded local raw-document preflight; not a production corpus preparation job.

All supplied source bytes are verified before any tokenization. Splits are an
explicit caller choice, determined by raw UTF-8 text hash rather than source row
or rank. There are no downloads, implicit truncation, source mixing, or sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Iterable

from .campaign_data import CampaignData, SourcePin, TokenizerPin, TokenizedDocument


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class SplitPolicy:
    """Required explicit allocation; no default held-out ratio is implied."""

    seed: int
    weights: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Split seed must be a nonnegative integer")
        weights = tuple(tuple(item) for item in self.weights)
        if not weights or any(len(item) != 2 for item in weights):
            raise ValueError("Split weights must contain (split, weight) pairs")
        names = []
        for name, weight in weights:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Split names must be nonempty strings")
            _positive(weight, "split weight")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("Split names must be unique")
        object.__setattr__(self, "weights", weights)

    def split_for_text_hash(self, text_sha256: str) -> str:
        if len(text_sha256) != 64 or any(c not in "0123456789abcdef" for c in text_sha256):
            raise ValueError("Expected lowercase UTF-8 text SHA256")
        draw = int.from_bytes(hashlib.sha256(f"{self.seed}:{text_sha256}".encode()).digest(), "big")
        draw %= sum(weight for _, weight in self.weights)
        for name, weight in self.weights:
            if draw < weight:
                return name
            draw -= weight
        raise AssertionError("Unreachable split allocation")


@dataclass(frozen=True)
class LocalJSONLSource:
    pin: SourcePin
    path: Path
    text_field: str = "text"
    id_field: str = "id"
    compression: str = "none"

    def __post_init__(self):
        if not isinstance(self.pin, SourcePin):
            raise TypeError("pin must be SourcePin")
        object.__setattr__(self, "path", Path(self.path))
        for name in (self.text_field, self.id_field):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("JSONL field names must be nonempty strings")
        if self.text_field == self.id_field:
            raise ValueError("Text and document identity fields must differ")
        if self.compression not in ("none", "gzip"):
            raise ValueError("compression must be none or gzip")

    def manifest_record(self):
        # Local relocation does not change artifact identity or preprocessing.
        return {"pin": asdict(self.pin), "text_field": self.text_field,
                "id_field": self.id_field, "compression": self.compression}


@dataclass(frozen=True)
class PreflightIngestResult:
    data: CampaignData
    _manifest_json: str

    @property
    def manifest(self):
        return json.loads(self._manifest_json)

    @property
    def manifest_sha256(self):
        return hashlib.sha256(self._manifest_json.encode()).hexdigest()


def prepare_local_preflight(sources: Iterable[LocalJSONLSource], *,
                            tokenizer: TokenizerPin, split_policy: SplitPolicy,
                            tokenizer_path: str | Path | None = None,
                            tokenize: Callable | None = None,
                            length: int = 1024, eos_id: int = 50279,
                            pad_id: int = 1, vocab_size: int = 50304,
                            max_documents: int = 1024,
                            max_source_bytes: int = 16 * 1024 * 1024,
                            max_uncompressed_bytes: int = 16 * 1024 * 1024,
                            max_tokens: int = 1_000_000) -> PreflightIngestResult:
    """Verify bounded raw inputs, tokenize verbatim text, and build CampaignData.

    Exactly one of tokenizer_path or tokenize is required. A tokenizer JSON path
    is verified against tokenizer.sha256 and loaded with implicit special-token
    insertion disabled. A supplied callable receives add_special_tokens=False;
    its implementation is explicitly *declared, unverified* provenance.

    Bounds apply to the complete provided sources; exceeding one raises, rather
    than publishing a selected prefix or pretending to prepare a large corpus.
    The returned ingest manifest must be pinned in outer run/checkpoint config
    alongside CampaignData's cursor fingerprint to bind preprocessing settings.
    """
    if not isinstance(tokenizer, TokenizerPin) or not isinstance(split_policy, SplitPolicy):
        raise TypeError("Require TokenizerPin and explicit SplitPolicy")
    if (tokenizer_path is None) == (tokenize is None):
        raise ValueError("Supply exactly one tokenizer_path or tokenize callable")
    if tokenize is not None and not callable(tokenize):
        raise TypeError("tokenize must be callable")
    for name, value in (("max_documents", max_documents), ("max_source_bytes", max_source_bytes),
                        ("max_uncompressed_bytes", max_uncompressed_bytes), ("max_tokens", max_tokens)):
        _positive(value, name)
    sources = tuple(sources)
    if not sources or any(not isinstance(source, LocalJSONLSource) for source in sources):
        raise ValueError("Provide a nonempty sequence of LocalJSONLSource values")
    if len({source.pin.name for source in sources}) != len(sources):
        raise ValueError("Each local source artifact must have a distinct source name")

    # Retain the verified bounded byte snapshots so a file edit after validation
    # cannot silently change the material that gets parsed/tokenized.
    verified, total_source_bytes = [], 0
    for source in sources:
        path = source.path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular non-symlink file: {source.pin.name}")
        if path.stat().st_size > max_source_bytes - total_source_bytes:
            raise ValueError("Local preflight source-byte bound exceeded; use bounded shards")
        with path.open("rb") as handle:
            raw = handle.read(max_source_bytes - total_source_bytes + 1)
        total_source_bytes += len(raw)
        if total_source_bytes > max_source_bytes:
            raise ValueError("Local preflight source-byte bound exceeded; use bounded shards")
        if hashlib.sha256(raw).hexdigest() != source.pin.sha256:
            raise ValueError(f"Source bytes differ from pinned SHA256: {source.pin.name}")
        verified.append((source, raw))

    verification = "declared_callable_unverified"
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        path = Path(tokenizer_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Tokenizer must be a regular non-symlink JSON file")
        tokenizer_bytes = path.read_bytes()
        if hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer.sha256:
            raise ValueError("Tokenizer bytes differ from pinned SHA256")
        native = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        if native.token_to_id("<|endoftext|>") != eos_id:
            raise ValueError("Tokenizer terminal EOS differs from configured native ID")
        tokenize = native.encode
        verification = "verified_tokenizer_json_bytes"

    documents, row_records = [], []
    total_uncompressed, total_tokens = 0, 0
    for source, raw in verified:
        if source.compression == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
                    contents = handle.read(max_uncompressed_bytes - total_uncompressed + 1)
            except (OSError, EOFError) as error:
                raise ValueError(f"Invalid gzip source: {source.pin.name}") from error
        else:
            contents = raw
        total_uncompressed += len(contents)
        if total_uncompressed > max_uncompressed_bytes:
            raise ValueError("Local preflight uncompressed-byte bound exceeded")
        try:
            # JSONL records end at LF. Unicode line/paragraph separators may
            # legally occur inside JSON strings and must not split a record.
            lines = contents.decode("utf-8").split("\n")
            if lines[-1] == "":
                lines.pop()
        except UnicodeDecodeError as error:
            raise ValueError(f"Source is not valid UTF-8: {source.pin.name}") from error
        if not lines:
            raise ValueError(f"Source contains no JSONL documents: {source.pin.name}")
        for line_number, line in enumerate(lines, 1):
            label = f"{source.pin.name}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row: {label}") from error
            if not isinstance(row, dict) or source.text_field not in row or source.id_field not in row:
                raise ValueError(f"Required text/id fields missing from JSONL object: {label}")
            text, identity = row[source.text_field], row[source.id_field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Document text must be a nonempty string: {label}")
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"Document id must be a nonempty string: {label}")
            if len(documents) == max_documents:
                raise ValueError("Local preflight document bound exceeded; no prefix is published")
            text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            encoded = tokenize(text, add_special_tokens=False)
            tokens = tuple(encoded.ids if hasattr(encoded, "ids") else encoded)
            total_tokens += len(tokens) + int(not tokens or tokens[-1] != eos_id)
            if total_tokens > max_tokens:
                raise ValueError("Local preflight normalized-token bound exceeded")
            documents.append(TokenizedDocument(source.pin, identity,
                             split_policy.split_for_text_hash(text_sha), text_sha, tokens))
            row_records.append({"source": source.pin.name, "line": line_number,
                                "document_id": identity, "text_sha256": text_sha})

    data = CampaignData(documents, tokenizer=tokenizer, length=length,
                        eos_id=eos_id, pad_id=pad_id, vocab_size=vocab_size)
    manifest = {"schema": "olmo-campaign-local-preflight-v1",
                "scope": "bounded local input preflight; no corpus mixture or production preparation",
                "sources": [source.manifest_record() for source in sources],
                "source_bytes_verified": True, "tokenizer_verification": verification,
                "preprocessing": {"text": "verbatim JSON string, UTF-8 hash; no Unicode normalization",
                                  "order": "supplied source order, then JSONL line order",
                                  "special_tokens": "encode(add_special_tokens=False); CampaignData owns EOS",
                                  "split_algorithm": "SHA256(seed:raw-text-SHA256), unsigned big-endian modulo sum(weights)",
                                  "split_policy": asdict(split_policy),
                                  "unknown_json_fields": "ignored", "empty_or_invalid_rows": "reject"},
                "bounds": {"max_documents": max_documents, "max_source_bytes": max_source_bytes,
                           "max_uncompressed_bytes": max_uncompressed_bytes, "max_tokens": max_tokens},
                "counts": {"documents": len(documents), "source_bytes": total_source_bytes,
                           "uncompressed_bytes": total_uncompressed, "normalized_tokens": total_tokens},
                "row_provenance": row_records,
                "data_manifest_sha256": data.manifest_sha256, "data_manifest": data.manifest}
    return PreflightIngestResult(data, _canonical(manifest))
