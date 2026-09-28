"""Restartable, CPU-only complete-document preparation, independent of packing.

Published shards are immutable. A shard contains uint16 little-endian tokens and
one provenance record per raw input row, including skipped exact-token duplicates.
No context length, model mask, shuffle, source mixture, or training budget is
chosen here. Memory scales with one document, not with the corpus.
"""
from __future__ import annotations

from array import array
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Iterator, Sequence

from .campaign_data import SourcePin, TokenizedDocument
from .campaign_ingest import LocalJSONLSource, SplitPolicy
from .olmo_artifacts import FILE_SPECS, REPO_ID, REVISION

SCHEMA = "olmo-tokenized-documents-v1"
EOS_ID = 50279
VOCAB_SIZE = 50280


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular non-symlink file: {path}")


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    _write(temporary, _json(value))
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _u16(tokens: Sequence[int]) -> bytes:
    values = array("H", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _decode(raw: bytes) -> tuple[int, ...]:
    values = array("H")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(values)


def _cursor(source_index=0, next_line=1):
    return {"source_index": source_index, "next_line": next_line}


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _load_tokenizer(path: Path):
    """Production preprocessing has no unverified callable tokenizer escape."""
    from tokenizers import Tokenizer
    _regular(path)
    expected_size, expected_sha = FILE_SPECS["tokenizer.json"]
    raw = path.read_bytes()
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Tokenizer differs from pinned OLMo tokenizer bytes")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
        raise ValueError("Pinned tokenizer vocabulary differs")
    if tokenizer.token_to_id("<|endoftext|>") != EOS_ID:
        raise ValueError("Pinned tokenizer EOS differs")
    # Be explicit even if the serialized native graph does not enable these.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


@contextmanager
def _verified_sources(sources):
    """Hash every complete source before tokenizing any; retain the verified FDs.

    Source files are immutable inputs. fstat guards reject concurrent edits
    before publishing each shard, including replaced paths and changed mtimes.
    """
    with ExitStack() as stack:
        opened = []
        for source in sources:
            _regular(source.path)
            handle = stack.enter_context(source.path.open("rb"))
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != source.pin.sha256:
                raise ValueError(f"Source bytes differ from pinned SHA256: {source.pin.name}")
            handle.seek(0)
            opened.append((source, handle, before))

        def check():
            for source, handle, before in opened:
                _regular(source.path)
                for current in (os.fstat(handle.fileno()), source.path.stat()):
                    attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                    if any(getattr(current, k) != getattr(before, k) for k in attrs):
                        raise ValueError(f"Source changed during preparation: {source.pin.name}")
        check()
        yield opened, check


def _config(sources, split_policy, max_documents_per_shard, target_tokens_per_shard,
            max_record_bytes):
    return {"schema": SCHEMA, "sources": [s.manifest_record() for s in sources],
            "tokenizer": {"repo": REPO_ID, "revision": REVISION,
                          "sha256": FILE_SPECS["tokenizer.json"][1],
                          "add_special_tokens": False, "padding": False,
                          "truncation": False},
            "split_policy": asdict(split_policy),
            "split_identity": "sha256_of_uint16_le_content_tokens_without_one_terminal_eos",
            "deduplication": "first_occurrence_of_split_identity_global_to_this_artifact",
            "document_identity_policy": "reject_conflicting_tokens_for_same_source_and_id",
            "token_dtype": "uint16_le", "vocab_size": VOCAB_SIZE, "eos_id": EOS_ID,
            "eos_policy": "append_one_if_absent_preserve_embedded_and_existing_terminal_eos",
            "empty_text_policy": "preserve_as_eos_only_document_subject_to_exact_dedup",
            "max_documents_per_shard": max_documents_per_shard,
            "target_tokens_per_shard": target_tokens_per_shard,
            "max_record_bytes": max_record_bytes}


def _read_config(root):
    path = root / "config.json"
    _regular(path)
    config = json.loads(path.read_bytes())
    if config.get("schema") != SCHEMA:
        raise ValueError("Document shard config schema differs")
    return config, _sha(path)


def _shards(root, config_sha):
    paths = sorted(root.glob("shard-*"))
    previous = None
    cursor = _cursor()
    for index, path in enumerate(paths):
        if path.name != f"shard-{index:06d}" or path.is_symlink() or not path.is_dir():
            raise ValueError("Shards must be contiguous regular directories")
        _regular(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_bytes())
        if (manifest.get("schema") != SCHEMA or manifest.get("shard_index") != index
                or manifest.get("config_sha256") != config_sha
                or manifest.get("previous_manifest_sha256") != previous
                or manifest.get("start_cursor") != cursor):
            raise ValueError("Shard manifest identity, chain or cursor differs")
        if set(manifest.get("files", {})) != {"tokens.bin", "documents.jsonl"}:
            raise ValueError("Shard file inventory differs")
        for name, expected in manifest["files"].items():
            target = path / name
            _regular(target)
            if (target.stat().st_size != expected["size_bytes"]
                    or _sha(target) != expected["sha256"]):
                raise ValueError(f"Shard bytes differ from manifest: {path.name}/{name}")
        previous = _sha(path / "manifest.json")
        cursor = manifest["end_cursor"]
        yield path, manifest, previous


def _records(path, manifest, config, index):
    """Validate offsets, token hashes and duplicate links with disk-backed index."""
    tokens = path / "tokens.bin"
    offset = rows = unique = duplicates = 0
    policy = SplitPolicy(config["split_policy"]["seed"], config["split_policy"]["weights"])
    last_cursor = manifest["start_cursor"]
    with tokens.open("rb") as token_file, (path / "documents.jsonl").open("rb") as metadata:
        for raw in metadata:
            record = json.loads(raw)
            source_index, line = record["source_index"], record["source_line"]
            if (type(source_index) is not int or not 0 <= source_index < len(config["sources"])
                    or type(line) is not int or line < 1
                    or source_index < last_cursor["source_index"]
                    or (source_index == last_cursor["source_index"] and line != last_cursor["next_line"])
                    or (source_index > last_cursor["source_index"] and line != 1)):
                raise ValueError("Invalid source row cursor sequence")
            last_cursor = _cursor(source_index, line + 1)
            content_hash = record["content_token_sha256"]
            _index_identity(index, source_index, record["document_id"], content_hash)
            if record["split"] != policy.split_for_text_hash(content_hash):
                raise ValueError("Document split differs from content identity")
            existing = index.execute("SELECT document_key FROM seen WHERE content_hash=?", (content_hash,)).fetchone()
            if record["kind"] == "duplicate":
                if existing is None or record["duplicate_of"] != existing[0]:
                    raise ValueError("Duplicate document reference differs")
                duplicates += 1
            elif record["kind"] == "document":
                if existing is not None or record["token_offset"] != offset:
                    raise ValueError("Document token offset or duplicate identity differs")
                count = record["token_count"]
                if type(count) is not int or count < 1:
                    raise ValueError("Invalid document token count")
                token_raw = token_file.read(count * 2)
                values = _decode(token_raw)
                if (len(values) != count or values[-1] != EOS_ID
                        or any(t >= VOCAB_SIZE for t in values)
                        or hashlib.sha256(token_raw).hexdigest() != record["token_sha256"]
                        or hashlib.sha256(token_raw[:-2]).hexdigest() != content_hash):
                    raise ValueError("Document token bytes or content hash differs")
                if record["embedded_eos_count"] != values[:-1].count(EOS_ID):
                    raise ValueError("Embedded EOS accounting differs")
                index.execute("INSERT INTO seen VALUES (?,?)", (content_hash, record["document_key"]))
                unique += 1
                offset += count
            else:
                raise ValueError("Unknown document metadata record kind")
            rows += 1
            yield record
        if token_file.read(1):
            raise ValueError("Unreferenced document token bytes")
    end = manifest["end_cursor"]
    if (end != last_cursor and (end["next_line"] != 1
            or not last_cursor["source_index"] < end["source_index"] <= len(config["sources"]))):
        raise ValueError("Shard terminal source cursor differs")
    if (rows, unique, duplicates, offset) != tuple(manifest[k] for k in
                                                  ("rows", "documents", "duplicate_rows", "tokens")):
        raise ValueError("Shard counts differ from stored records")


def _index_identity(index, source_index, document_id, content_hash):
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document identity must be a nonempty string")
    previous = index.execute("SELECT content_hash FROM identities WHERE source_index=? AND document_id=?",
                             (source_index, document_id)).fetchone()
    if previous is not None and previous[0] != content_hash:
        raise ValueError("Conflicting content tokens for the same source/document identity")
    index.execute("INSERT OR IGNORE INTO identities VALUES (?,?,?)",
                  (source_index, document_id, content_hash))


@contextmanager
def _index():
    # Temporary on-disk index bounds corpus-scale dedup RAM and is rebuilt from
    # immutable metadata, so it cannot become stale after an interrupted commit.
    with tempfile.TemporaryDirectory(prefix="olmo-document-index-") as folder:
        connection = sqlite3.connect(str(Path(folder) / "index.sqlite"))
        try:
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("CREATE TABLE seen (content_hash TEXT PRIMARY KEY, document_key TEXT NOT NULL)")
            connection.execute("CREATE TABLE identities (source_index INTEGER, document_id TEXT, content_hash TEXT NOT NULL, PRIMARY KEY(source_index, document_id))")
            yield connection
        finally:
            connection.close()


def verify_document_shards(output_dir: str | Path) -> dict:
    """Verify every byte/hash/offset and return the committed durable state."""
    root = Path(output_dir)
    config, config_sha = _read_config(root)
    summary = {"schema": SCHEMA, "config_sha256": config_sha, "completed": False,
               "shards": [], "rows": 0, "documents": 0, "duplicate_rows": 0,
               "tokens": 0, "cursor": _cursor()}
    with _index() as index:
        for path, manifest, manifest_sha in _shards(root, config_sha):
            for _ in _records(path, manifest, config, index):
                pass
            summary["shards"].append({"path": path.name, "manifest_sha256": manifest_sha})
            for key in ("rows", "documents", "duplicate_rows", "tokens"):
                summary[key] += manifest[key]
            summary["cursor"] = manifest["end_cursor"]
    final = root / "manifest.json"
    if final.exists():
        _regular(final)
        expected = dict(summary, completed=True, cursor=_cursor(len(config["sources"])))
        if json.loads(final.read_bytes()) != expected:
            raise ValueError("Completed document manifest differs from committed shards")
        summary = expected
    return summary


def iter_documents(output_dir: str | Path, *, verify: bool = True) -> Iterator[TokenizedDocument]:
    """Read unique complete documents lazily; excludes explicit duplicate rows."""
    root = Path(output_dir)
    if verify:
        verify_document_shards(root)
    config, config_sha = _read_config(root)
    for path, _, _ in _shards(root, config_sha):
        with (path / "tokens.bin").open("rb") as tokens, (path / "documents.jsonl").open("rb") as records:
            for raw in records:
                record = json.loads(raw)
                if record["kind"] != "document":
                    continue
                tokens.seek(record["token_offset"] * 2)
                yield TokenizedDocument(
                    SourcePin(**config["sources"][record["source_index"]]["pin"]),
                    record["document_id"], record["split"], record["text_sha256"],
                    _decode(tokens.read(record["token_count"] * 2)))


def prepare_document_shards(sources: Sequence[LocalJSONLSource], output_dir: str | Path, *,
                            tokenizer_path: str | Path, split_policy: SplitPolicy,
                            max_documents_per_shard: int = 4096,
                            target_tokens_per_shard: int = 4_000_000,
                            max_record_bytes: int = 64 * 1024 * 1024,
                            max_new_shards: int | None = None) -> dict:
    """Prepare complete documents from fully pinned local JSONL/gzip sources.

    Token and document limits are per-shard commit thresholds: a complete long
    document may cross the token target. max_record_bytes rejects giant input
    records explicitly; nothing is truncated. max_documents_per_shard counts raw
    rows, including duplicates. max_new_shards allows bounded resumable work.

    Terminal EOS is appended iff absent. Exactly one terminal EOS is excluded
    from the dedup/split identity; all other EOS tokens remain literal content.
    Empty text is retained as an EOS-only document (and deduplicated normally).
    Splits and exact-token dedup apply globally within this prepared artifact.
    This is not a near-duplicate detector or a complete corpus decontamination.
    """
    sources = tuple(sources)
    if (not sources or any(not isinstance(s, LocalJSONLSource) for s in sources)
            or len({s.pin.name for s in sources}) != len(sources)):
        raise ValueError("Provide distinct named LocalJSONLSource artifacts")
    if not isinstance(split_policy, SplitPolicy):
        raise TypeError("Provide an explicit SplitPolicy")
    for name, value in (("max_documents_per_shard", max_documents_per_shard),
                        ("target_tokens_per_shard", target_tokens_per_shard),
                        ("max_record_bytes", max_record_bytes)):
        _positive(value, name)
    if max_new_shards is not None:
        _positive(max_new_shards, "max_new_shards")
    tokenizer = _load_tokenizer(Path(tokenizer_path))
    config = _config(sources, split_policy, max_documents_per_shard,
                     target_tokens_per_shard, max_record_bytes)
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as lock, _verified_sources(sources) as verified:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        opened, check_sources = verified
        config_path = root / "config.json"
        if config_path.exists():
            _regular(config_path)
            if config_path.read_bytes() != _json(config):
                raise ValueError("Resume configuration or source pins differ")
        else:
            if list(root.glob("shard-*")) or (root / "manifest.json").exists():
                raise ValueError("Existing output has shards but lacks config")
            _atomic_json(config_path, config)
        config_sha = _sha(config_path)
        summary = verify_document_shards(root)
        if summary["completed"]:
            return summary
        for abandoned in root.glob(".pending-*"):
            if abandoned.is_symlink() or not abandoned.is_dir():
                raise ValueError("Unsafe unfinished shard path")
            shutil.rmtree(abandoned)
        with _index() as index:
            for path, manifest, _ in _shards(root, config_sha):
                for _ in _records(path, manifest, config, index):
                    pass
            next_cursor = dict(summary["cursor"])
            start_cursor = dict(next_cursor)
            pending = None
            token_file = metadata_file = None
            rows = documents = duplicate_rows = token_count = new_shards = 0

            def commit(end_cursor):
                nonlocal pending, token_file, metadata_file, rows, documents
                nonlocal duplicate_rows, token_count, new_shards, start_cursor
                if pending is None:
                    return
                check_sources()
                for handle in (token_file, metadata_file):
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                shard_index = len(summary["shards"])
                manifest = {"schema": SCHEMA, "shard_index": shard_index,
                            "config_sha256": config_sha,
                            "previous_manifest_sha256": summary["shards"][-1]["manifest_sha256"] if summary["shards"] else None,
                            "start_cursor": start_cursor, "end_cursor": end_cursor,
                            "rows": rows, "documents": documents,
                            "duplicate_rows": duplicate_rows, "tokens": token_count,
                            "files": {name: {"size_bytes": (pending / name).stat().st_size,
                                              "sha256": _sha(pending / name)}
                                      for name in ("tokens.bin", "documents.jsonl")}}
                _write(pending / "manifest.json", _json(manifest))
                _sync_dir(pending)
                target = root / f"shard-{shard_index:06d}"
                os.rename(pending, target)
                _sync_dir(root)
                summary["shards"].append({"path": target.name, "manifest_sha256": _sha(target / "manifest.json")})
                for key in ("rows", "documents", "duplicate_rows", "tokens"):
                    summary[key] += manifest[key]
                summary["cursor"] = end_cursor
                index.commit()
                new_shards += 1
                start_cursor = dict(end_cursor)
                pending = token_file = metadata_file = None
                rows = documents = duplicate_rows = token_count = 0

            try:
                for source_index, (source, raw_handle, _) in enumerate(opened):
                    if source_index < next_cursor["source_index"]:
                        continue
                    reader = gzip.GzipFile(fileobj=raw_handle, mode="rb") if source.compression == "gzip" else raw_handle
                    try:
                        line_number = 0
                        while True:
                            line = reader.readline(max_record_bytes + 1)
                            if not line:
                                break
                            line_number += 1
                            if len(line) > max_record_bytes:
                                raise ValueError(f"Source record exceeds max_record_bytes: {source.pin.name}:{line_number}")
                            if source_index == next_cursor["source_index"] and line_number < next_cursor["next_line"]:
                                continue
                            record = json.loads(line)
                            text = record.get(source.text_field)
                            document_id = record.get(source.id_field)
                            if not isinstance(text, str) or not isinstance(document_id, str) or not document_id.strip():
                                raise ValueError(f"Source row needs string text and nonempty ID: {source.pin.name}:{line_number}")
                            values = tokenizer.encode(text, add_special_tokens=False).ids
                            if any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in values):
                                raise ValueError("Tokenizer returned out-of-vocabulary tokens")
                            had_eos = bool(values and values[-1] == EOS_ID)
                            if not had_eos:
                                values.append(EOS_ID)
                            token_bytes = _u16(values)
                            content_hash = hashlib.sha256(token_bytes[:-2]).hexdigest()
                            _index_identity(index, source_index, document_id, content_hash)
                            document_key = hashlib.sha256(_json({"source": source.pin.name,
                                                                "line": line_number, "id": document_id})).hexdigest()
                            metadata = {"source_index": source_index, "source_line": line_number,
                                        "document_id": document_id, "document_key": document_key,
                                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                        "content_token_sha256": content_hash,
                                        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
                                        "split": split_policy.split_for_text_hash(content_hash),
                                        "had_terminal_eos": had_eos,
                                        "embedded_eos_count": values[:-1].count(EOS_ID)}
                            if pending is None:
                                pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
                                token_file = (pending / "tokens.bin").open("xb")
                                metadata_file = (pending / "documents.jsonl").open("xb")
                            existing = index.execute("SELECT document_key FROM seen WHERE content_hash=?", (content_hash,)).fetchone()
                            if existing:
                                metadata.update(kind="duplicate", duplicate_of=existing[0])
                                duplicate_rows += 1
                            else:
                                metadata.update(kind="document", token_offset=token_count, token_count=len(values))
                                token_file.write(token_bytes)
                                token_count += len(values)
                                documents += 1
                                index.execute("INSERT INTO seen VALUES (?,?)", (content_hash, document_key))
                            metadata_file.write(_json(metadata))
                            rows += 1
                            current_cursor = _cursor(source_index, line_number + 1)
                            if rows >= max_documents_per_shard or token_count >= target_tokens_per_shard:
                                commit(current_cursor)
                                if max_new_shards is not None and new_shards >= max_new_shards:
                                    return summary
                        if source_index == next_cursor["source_index"] and line_number + 1 < next_cursor["next_line"]:
                            raise ValueError("Resume source cursor exceeds source rows")
                    finally:
                        if reader is not raw_handle:
                            reader.close()
                final_cursor = _cursor(len(sources))
                commit(final_cursor)
                check_sources()
                summary.update(completed=True, cursor=final_cursor)
                _atomic_json(root / "manifest.json", summary)
                return summary
            finally:
                for handle in (token_file, metadata_file):
                    if handle is not None and not handle.closed:
                        handle.close()
