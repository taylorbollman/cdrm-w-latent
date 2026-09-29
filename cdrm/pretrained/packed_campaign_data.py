"""Verified, disk-backed OLMo stream chunks and an explicit committed cursor.

Complete documents are concatenated in canonical shard/record order within an
existing split. Chunks have stride T (no overlap), preserve true document IDs,
and retain the final right-padded tail. Literal EOS tokens do not create a
boundary. This module does not tokenize, shuffle, choose a mixture or cycle.
Token memory scales with requested chunks/batches, never with the corpus.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

from .document_shards import verify_document_shards

SCHEMA = "olmo-packed-campaign-data-v1"
DOCUMENT_POLICY = "continuous-stream-v1"
POLICY = {
    "document_policy": DOCUMENT_POLICY,
    "order": "canonical_shard_then_record_order_filtered_by_existing_split",
    "stride": "context_length_no_overlap",
    "eos": "preserve_stored_terminal_and_embedded_eos_do_not_infer_boundaries",
    "ce": "all_valid_within_chunk_adjacent_pairs_including_document_boundaries",
    "latent": "same_actual_document_adjacent_pairs",
    "kl": "same_actual_document_three_position_windows",
    "attention_rt": "continuous_causal_within_chunk_reset_at_each_row",
    "fbt": "feedback_and_keyed_jitter_across_valid_adjacency_including_document_boundaries",
    "positions": "zero_based_per_chunk",
    "tail": "retain_final_partial_chunk_with_right_padding",
    "budget": "global_valid_input_tokens_once_before_rank_partition",
    "cycling": False,
}


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _stat(path):
    if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
        raise ValueError(f"Expected immutable regular file: {path}")
    value = path.stat()
    return tuple(getattr(value, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _sha(path):
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    if _stat(path) != before:
        raise ValueError(f"File changed while verifying bytes: {path}")
    return digest.hexdigest()


def _verified_inventory(root, summary):
    config = json.loads((root / "config.json").read_bytes())
    expected = {
        "config.json": {"sha256": summary["config_sha256"], "size_bytes": (root/"config.json").stat().st_size},
        "manifest.json": {"sha256": _sha(root/"manifest.json"), "size_bytes": (root/"manifest.json").stat().st_size},
    }
    for shard in summary["shards"]:
        relative = shard["path"] + "/manifest.json"
        path = root/relative
        expected[relative] = {"sha256": shard["manifest_sha256"], "size_bytes": path.stat().st_size}
        manifest = json.loads(path.read_bytes())
        for name, pin in manifest["files"].items():
            expected[shard["path"] + "/" + name] = dict(pin)
    signatures = {}
    for relative, pin in expected.items():
        path = root/relative
        before = _stat(path)
        if before[2] != pin["size_bytes"] or _sha(path) != pin["sha256"]:
            raise ValueError(f"Corpus bytes differ from verified inventory: {relative}")
        signatures[relative] = before
    return config, expected, signatures


def _check_files(root, signatures):
    for relative, expected in signatures.items():
        if _stat(root/relative) != expected:
            raise ValueError(f"Immutable corpus/index file changed: {relative}")


def build_packed_index(corpus_dir, index_dir, *, split, length=1024,
                       document_policy=DOCUMENT_POLICY, pad_id=1):
    """Atomically publish metadata only; every token remains in verified shards.

    Publication is all-or-nothing. An interrupted unpublished index can be
    rebuilt without changing tokenization or corpus files. Existing index
    directories are never overwritten. Opening requires byte verification.
    """
    corpus, output = Path(corpus_dir), Path(index_dir)
    if corpus.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError("Corpus must be a real directory and index destination must be new")
    if type(length) is not int or not 3 <= length <= 2048:
        raise ValueError("Chunk length must be an integer in 3..2048")
    if document_policy != DOCUMENT_POLICY:
        raise ValueError("Packed data requires explicit continuous-stream-v1 policy")
    summary = verify_document_shards(corpus)
    if not summary["completed"]:
        raise ValueError("Packed index requires a completed immutable document corpus")
    config, inventory, signatures = _verified_inventory(corpus, summary)
    if not isinstance(split, str) or split not in dict(config["split_policy"]["weights"]):
        raise ValueError("Split must already exist in the source preparation policy")
    if type(pad_id) is not int or not 0 <= pad_id < config["vocab_size"] or pad_id == config["eos_id"]:
        raise ValueError("PAD must be an in-vocabulary integer distinct from EOS")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-packed-index-", dir=output.parent))
    connection = None
    try:
        db = pending/"documents.sqlite"
        connection = sqlite3.connect(str(db))
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("""CREATE TABLE documents (
            ordinal INTEGER PRIMARY KEY, document_index INTEGER NOT NULL UNIQUE,
            stream_start INTEGER NOT NULL UNIQUE, stream_end INTEGER NOT NULL,
            shard TEXT NOT NULL, token_offset INTEGER NOT NULL, token_count INTEGER NOT NULL,
            source_index INTEGER NOT NULL, source_name TEXT NOT NULL, source_line INTEGER NOT NULL,
            document_id TEXT NOT NULL, document_key TEXT NOT NULL, token_sha256 TEXT NOT NULL)""")
        connection.execute("CREATE INDEX stream_end ON documents(stream_end)")
        stream = ordinal = corpus_document_index = 0
        order_digest = hashlib.sha256()
        for shard in summary["shards"]:
            with (corpus/shard["path"]/"documents.jsonl").open("rb") as handle:
                for raw in handle:
                    record = json.loads(raw)
                    if record["kind"] != "document":
                        continue
                    document_index = corpus_document_index
                    corpus_document_index += 1
                    if record["split"] != split:
                        continue
                    count = record["token_count"]
                    selected = {"ordinal": ordinal, "document_index": document_index,
                        "stream_start": stream, "stream_end": stream+count, "shard": shard["path"],
                        "token_offset": record["token_offset"], "token_count": count,
                        "source_index": record["source_index"],
                        "source_name": config["sources"][record["source_index"]]["pin"]["name"],
                        "source_line": record["source_line"], "document_id": record["document_id"],
                        "document_key": record["document_key"], "token_sha256": record["token_sha256"]}
                    connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(selected.values()))
                    order_digest.update(_json(selected))
                    stream += count
                    ordinal += 1
        if not ordinal:
            raise ValueError("Selected existing split contains no unique documents")
        connection.commit()
        connection.close()
        connection = None
        _check_files(corpus, signatures)
        contract = {"schema": SCHEMA, "split": split, "length": length, "pad_id": pad_id,
            "eos_id": config["eos_id"], "vocab_size": config["vocab_size"],
            "token_dtype": config["token_dtype"], "tokenizer": config["tokenizer"],
            "corpus_manifest_sha256": inventory["manifest.json"]["sha256"],
            "corpus_config_sha256": summary["config_sha256"], "corpus_files": inventory,
            "order_sha256": order_digest.hexdigest(), "policy": POLICY,
            "selected_documents": ordinal, "total_tokens": stream,
            "total_chunks": (stream+length-1)//length, "final_chunk_tokens": (stream-1) % length + 1}
        manifest = {**contract, "identity_sha256": _digest(contract),
            "index": {"path": "documents.sqlite", "size_bytes": db.stat().st_size, "sha256": _sha(db)}}
        with (pending/"manifest.json").open("xb") as handle:
            handle.write(_json(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        with db.open("rb") as handle:
            os.fsync(handle.fileno())
        fd = os.open(pending, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(pending, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return manifest
    finally:
        if connection is not None:
            connection.close()
        if pending.exists():
            shutil.rmtree(pending)


@dataclass(frozen=True)
class PackedCounts:
    packed_rows: int = 0
    valid_tokens: int = 0
    ce_targets: int = 0
    latent_pairs: int = 0
    kl_triples: int = 0
    cross_document_ce_targets: int = 0
    excluded_boundary_latent_pairs: int = 0
    excluded_boundary_kl_triples: int = 0
    omitted_cross_chunk_ce_targets: int = 0
    omitted_cross_chunk_latent_pairs: int = 0
    omitted_cross_chunk_kl_triples: int = 0
    tail_padding_tokens: int = 0
    document_segments: int = 0
    document_completions: int = 0

    def __add__(self, other):
        if not isinstance(other, PackedCounts):
            return NotImplemented
        return PackedCounts(**{f.name: getattr(self, f.name)+getattr(other, f.name) for f in fields(self)})

    @property
    def objective_counts(self):
        return {"ce": self.ce_targets, "latent": self.latent_pairs, "kl": self.kl_triples}


@dataclass(frozen=True)
class ChunkDescriptor:
    index: int
    key: str
    stream_start: int
    length: int


@dataclass(frozen=True)
class DocumentSegment:
    document_index: int
    document_key: str
    document_id: str
    source_index: int
    source_name: str
    source_line: int
    shard: str
    token_file_offset: int
    document_token_offset: int
    chunk_offset: int
    length: int
    completes_document: bool


@dataclass(frozen=True)
class PackedChunk:
    descriptor: ChunkDescriptor
    tokens: tuple[int, ...]
    document_ids: tuple[int, ...]
    segments: tuple[DocumentSegment, ...]


@dataclass(frozen=True)
class PackedCursor:
    manifest_sha256: str
    split: str
    next_chunk: int = 0
    next_update: int = 0


@dataclass(frozen=True)
class PackedLogicalUpdate:
    rows: tuple[ChunkDescriptor, ...]
    start_cursor: PackedCursor
    next_cursor: PackedCursor
    target_valid_tokens: int
    counts: PackedCounts
    unique_document_count: int

    @property
    def reaches_target(self):
        return self.counts.valid_tokens >= self.target_valid_tokens

    @property
    def overshoot_tokens(self):
        return max(0, self.counts.valid_tokens-self.target_valid_tokens)


@dataclass(frozen=True)
class PackedRankBatches:
    batches: tuple
    keys: tuple[tuple[str, ...], ...]
    counts: PackedCounts
    physical_rows: int
    empty_rows: int
    padding_tokens: int

    @property
    def accounting(self):
        return {**asdict(self.counts), "physical_rows": self.physical_rows,
                "empty_rows": self.empty_rows, "padding_tokens": self.padding_tokens}


class PackedCampaignData:
    """Read-only metadata index and bounded random reads of pinned token shards.

    At most eight token FDs are cached. Chunk/update descriptors contain no
    tokens. Opening hashes all pinned corpus/index bytes; later operations
    reject inode/size/mtime/ctime changes. The cursor advances only on commit,
    never on reads, partitioning or prefetch via peek_update.
    """

    def __init__(self, corpus_dir, index_dir):
        self.corpus, self.index_dir = Path(corpus_dir), Path(index_dir)
        if self.corpus.is_symlink() or self.index_dir.is_symlink():
            raise ValueError("Corpus and index directories must not be symlinks")
        manifest_path = self.index_dir/"manifest.json"
        manifest_signature = _stat(manifest_path)
        self._manifest_bytes = manifest_path.read_bytes()
        if _stat(manifest_path) != manifest_signature:
            raise ValueError("Packed manifest changed while reading")
        manifest = json.loads(self._manifest_bytes)
        self.manifest_sha256 = hashlib.sha256(self._manifest_bytes).hexdigest()
        if manifest.get("schema") != SCHEMA or manifest.get("policy") != POLICY:
            raise ValueError("Packed manifest schema or stream policy differs")
        contract = {k: v for k, v in manifest.items() if k not in ("identity_sha256", "index")}
        if manifest.get("identity_sha256") != _digest(contract):
            raise ValueError("Packed canonical corpus/order/policy identity differs")
        self.length, self.split = manifest["length"], manifest["split"]
        self.total_tokens, self.total_chunks = manifest["total_tokens"], manifest["total_chunks"]
        if (type(self.length) is not int or not 3 <= self.length <= 2048
                or type(self.total_tokens) is not int or self.total_tokens <= 0
                or self.total_chunks != (self.total_tokens+self.length-1)//self.length):
            raise ValueError("Packed manifest dimensions differ")
        self._corpus_signatures = {}
        for relative, pin in manifest["corpus_files"].items():
            if (Path(relative).is_absolute() or ".." in Path(relative).parts
                    or not relative or Path(relative).as_posix() != relative):
                raise ValueError("Unsafe corpus inventory path")
            path = self.corpus/relative
            signature = _stat(path)
            if signature[2] != pin["size_bytes"] or _sha(path) != pin["sha256"]:
                raise ValueError(f"Corpus bytes differ from packed index: {relative}")
            self._corpus_signatures[relative] = signature
        if manifest["index"]["path"] != "documents.sqlite":
            raise ValueError("Packed metadata index path differs")
        db = self.index_dir/"documents.sqlite"
        db_signature = _stat(db)
        if db_signature[2] != manifest["index"]["size_bytes"] or _sha(db) != manifest["index"]["sha256"]:
            raise ValueError("Packed metadata index bytes differ")
        # Retain the signatures from before verification. Capturing fresh ones
        # here could adopt an unverified replacement made after either read.
        self._index_signatures = {"manifest.json": manifest_signature, "documents.sqlite": db_signature}
        _check_files(self.index_dir, self._index_signatures)
        self._connection = sqlite3.connect(db.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA cache_size=-2048")
        self._token_fds = OrderedDict()
        self._closed = False
        self._cursor = PackedCursor(self.manifest_sha256, self.split)
        self._identity = manifest["identity_sha256"]
        self.pad_id = manifest["pad_id"]
        self._reader_contract = self._contract()
        try:
            self.validate_integrity()
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self):
        return json.loads(self._manifest_bytes)

    def validate_integrity(self):
        if self._closed:
            raise ValueError("Packed data reader is closed")
        _check_files(self.corpus, self._corpus_signatures)
        _check_files(self.index_dir, self._index_signatures)
        if self._reader_contract != self._contract():
            raise ValueError("Packed reader runtime contract changed")

    def _contract(self):
        return (self.length, self.split, self.total_tokens, self.total_chunks,
                self.pad_id, self._identity, self.manifest_sha256)

    def close(self):
        if not self._closed:
            for handle in self._token_fds.values():
                handle.close()
            self._token_fds.clear()
            self._connection.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _descriptor(self, index):
        if type(index) is not int or not 0 <= index < self.total_chunks:
            raise ValueError("Chunk index out of bounds")
        start = index*self.length
        length = min(self.length, self.total_tokens-start)
        return ChunkDescriptor(index, _digest({"identity": self._identity, "index": index,
            "stream_start": start, "length": length}), start, length)

    def descriptor(self, index):
        self.validate_integrity()
        return self._descriptor(index)

    def _segments(self, row):
        if not isinstance(row, ChunkDescriptor) or row != self._descriptor(row.index):
            raise ValueError("Chunk descriptor provenance differs")
        end = row.stream_start+row.length
        records = self._connection.execute(
            "SELECT * FROM documents WHERE stream_end>? AND stream_start<? ORDER BY ordinal",
            (row.stream_start, end))
        segments = []
        for record in records:
            start, stop = max(record["stream_start"], row.stream_start), min(record["stream_end"], end)
            offset = start-record["stream_start"]
            segments.append(DocumentSegment(record["document_index"], record["document_key"],
                record["document_id"], record["source_index"], record["source_name"], record["source_line"],
                record["shard"], record["token_offset"]+offset, offset, start-row.stream_start,
                stop-start, stop == record["stream_end"]))
        if sum(s.length for s in segments) != row.length:
            raise ValueError("Packed metadata index does not cover chunk")
        return tuple(segments)

    def _document_at(self, position):
        if not 0 <= position < self.total_tokens:
            return None
        record = self._connection.execute(
            "SELECT document_index FROM documents WHERE stream_start<=? ORDER BY stream_start DESC LIMIT 1",
            (position,)).fetchone()
        return record[0]

    def _chunk_counts(self, row, segments=None):
        segments = self._segments(row) if segments is None else segments
        latent = sum(max(s.length-1, 0) for s in segments)
        kl = sum(max(s.length-2, 0) for s in segments)
        end = row.stream_start+row.length
        omitted_ce = int(end < self.total_tokens)
        omitted_latent = int(bool(omitted_ce) and self._document_at(end-1) == self._document_at(end))
        omitted_kl = sum(int(start >= 0 and start+2 < self.total_tokens
                            and len({self._document_at(p) for p in range(start, start+3)}) == 1)
                         for start in (end-2, end-1)) if omitted_ce else 0
        ce = max(row.length-1, 0)
        return PackedCounts(1, row.length, ce, latent, kl, ce-latent, ce-latent,
            max(row.length-2, 0)-kl, omitted_ce, omitted_latent, omitted_kl,
            self.length-row.length, len(segments), sum(s.completes_document for s in segments))

    def chunk_counts(self, row):
        self.validate_integrity()
        return self._chunk_counts(row)

    def _token_slice(self, segment):
        relative = segment.shard+"/tokens.bin"
        path = self.corpus/relative
        expected = self._corpus_signatures.get(relative)
        if expected is None or _stat(path) != expected:
            raise ValueError("Token file changed from pinned corpus")
        handle = self._token_fds.pop(relative, None)
        if handle is None:
            handle = path.open("rb")
        self._token_fds[relative] = handle
        while len(self._token_fds) > 8:
            _, discarded = self._token_fds.popitem(last=False)
            discarded.close()
        current = os.fstat(handle.fileno())
        signature = tuple(getattr(current, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
        if signature != expected:
            raise ValueError("Open token FD differs from pinned corpus")
        raw = os.pread(handle.fileno(), segment.length*2, segment.token_file_offset*2)
        if len(raw) != segment.length*2 or _stat(path) != expected:
            raise ValueError("Token bytes changed during bounded read")
        values = array("H")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return tuple(values)

    def _read_chunk(self, row):
        segments = self._segments(row)
        tokens, ids = [], []
        for segment in segments:
            tokens.extend(self._token_slice(segment))
            ids.extend([segment.document_index]*segment.length)
        return PackedChunk(row, tuple(tokens), tuple(ids), segments)

    def read_chunk(self, index):
        self.validate_integrity()
        return self._read_chunk(self._descriptor(index))

    def cursor(self):
        self.validate_integrity()
        return self._cursor

    def _check_cursor(self, cursor):
        if (not isinstance(cursor, PackedCursor) or cursor.manifest_sha256 != self.manifest_sha256
                or cursor.split != self.split or type(cursor.next_chunk) is not int
                or not 0 <= cursor.next_chunk <= self.total_chunks or type(cursor.next_update) is not int
                or not 0 <= cursor.next_update <= cursor.next_chunk
                or (cursor.next_chunk == 0) != (cursor.next_update == 0)):
            raise ValueError("Packed cursor identity, split or committed position differs")

    def restore_cursor(self, record):
        self.validate_integrity()
        if not isinstance(record, dict) or set(record) != {f.name for f in fields(PackedCursor)}:
            raise ValueError("Invalid packed cursor fields")
        cursor = PackedCursor(**record)
        self._check_cursor(cursor)
        if self._cursor.next_chunk or self._cursor.next_update:
            raise ValueError("Restore the cursor only into a fresh reader")
        self._cursor = cursor
        return cursor

    def peek_update(self, cursor, target_valid_tokens):
        self.validate_integrity()
        self._check_cursor(cursor)
        _positive(target_valid_tokens, "target_valid_tokens")
        if cursor.next_chunk == self.total_chunks:
            return None  # Explicit exhaustion; never silently cycle the corpus.
        end = min(self.total_chunks, cursor.next_chunk+(target_valid_tokens+self.length-1)//self.length)
        rows = tuple(self._descriptor(index) for index in range(cursor.next_chunk, end))
        counts = PackedCounts()
        documents = set()
        for row in rows:
            segments = self._segments(row)
            counts += self._chunk_counts(row, segments)
            documents.update(s.document_index for s in segments)
        return PackedLogicalUpdate(rows, cursor,
            PackedCursor(self.manifest_sha256, self.split, end, cursor.next_update+1),
            target_valid_tokens, counts, len(documents))

    def _check_update(self, update):
        if not isinstance(update, PackedLogicalUpdate) or update != self.peek_update(update.start_cursor, update.target_valid_tokens):
            raise ValueError("Logical update membership/counts/cursors differ from canonical stream")

    def commit(self, cursor, update):
        """Caller invokes only after the associated optimizer update completes."""
        self.validate_integrity()
        self._check_update(update)
        if cursor != self._cursor or update.start_cursor != cursor:
            raise ValueError("Cannot commit a stale, out-of-order or unconsumed update")
        self._cursor = update.next_cursor
        return self._cursor

    def partition(self, update, *, world_size, physical_batch_size):
        self._check_update(update)
        _positive(world_size, "world_size")
        _positive(physical_batch_size, "physical_batch_size")
        capacity = world_size*physical_batch_size
        return tuple(tuple(update.rows[start+r*physical_batch_size:start+(r+1)*physical_batch_size]
                           for r in range(world_size)) for start in range(0, len(update.rows), capacity))

    def batch(self, rows, *, physical_batch_size):
        import torch
        from .nextlat import NextLatBatch
        self.validate_integrity()
        rows = tuple(rows)
        _positive(physical_batch_size, "physical_batch_size")
        if len(rows) > physical_batch_size:
            raise ValueError("Physical batch cannot discard real chunks")
        ids = torch.full((physical_batch_size, self.length), self.pad_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        document_ids = torch.full_like(ids, -1)
        for i, row in enumerate(rows):
            chunk = self._read_chunk(row)
            ids[i, :row.length] = torch.tensor(chunk.tokens)
            valid[i, :row.length] = True
            document_ids[i, :row.length] = torch.tensor(chunk.document_ids)
        return NextLatBatch(ids, valid, document_ids, valid.clone(), valid.clone(), valid.clone())

    def rank_batches(self, update, *, rank, world_size, physical_batch_size):
        slots = self.partition(update, world_size=world_size, physical_batch_size=physical_batch_size)
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("Rank is outside the requested world size")
        rows = tuple(slot[rank] for slot in slots)
        counts = PackedCounts()
        for slot in rows:
            for row in slot:
                counts += self._chunk_counts(row)
        physical_rows = len(rows)*physical_batch_size
        return PackedRankBatches(tuple(self.batch(slot, physical_batch_size=physical_batch_size) for slot in rows),
            tuple(tuple(row.key for row in slot) for slot in rows), counts,
            physical_rows, physical_rows-counts.packed_rows, physical_rows*self.length-counts.valid_tokens)
