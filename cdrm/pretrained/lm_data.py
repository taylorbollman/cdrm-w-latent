"""Pinned O4 code adaptation data, with isolated document windows and cursors.

CodeSearchNet functions are used verbatim. WikiText rows are reassembled into
articles at top-level headings. Each document receives one explicit native EOS.
Windows overlap by one context token: every CE/latent pair appears once, while
one KL triple per window boundary is intentionally omitted. No document packing,
implicit cycling, code execution, or special-token insertion by the tokenizer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .artifacts import sha256_file, write_json
from .olmo_artifacts import FILE_SPECS, REPO_ID as TOKENIZER_REPO, REVISION as TOKENIZER_REVISION

SCHEMA = "olmo-lm-data-v1"
CSN_REPO = "code-search-net/code_search_net"
CSN_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
WIKI_REPO = "Salesforce/wikitext"
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EOS_ID, PAD_ID = 50279, 1
SPLITS = ("train", "dev", "test", "retention_dev", "retention_test")
# repo, revision, repo-relative path, actual downloadable bytes, published LFS SHA256
RAW_SPECS = {
    "train": (CSN_REPO, CSN_REVISION, "python/train-00000-of-00001.parquet", 521785752,
              "ad9e3a4ab10c2c1d8926d2b26ca2bfcc3aadda1477ba29a933391f93806b9fed"),
    "dev": (CSN_REPO, CSN_REVISION, "python/validation-00000-of-00001.parquet", 30742555,
            "22eaacb46ed7e74d582409b85692ef63f5a43e99f9395c2eb736b5c8451422bb"),
    "test": (CSN_REPO, CSN_REVISION, "python/test-00000-of-00001.parquet", 28744352,
             "3167e79ee7f081d825bf97b96d3a6b2d96428b00f6a98125be943384d8afae5f"),
    "retention_dev": (WIKI_REPO, WIKI_REVISION, "wikitext-2-raw-v1/validation-00000-of-00001.parquet", 657209,
                      "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"),
    "retention_test": (WIKI_REPO, WIKI_REVISION, "wikitext-2-raw-v1/test-00000-of-00001.parquet", 732610,
                       "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
}


def _source_record(spec):
    repo, revision, path, size, sha = spec
    return {"repo": repo, "revision": revision, "path": path, "size_bytes": size, "sha256": sha,
            "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"}


def _raw_path(root, spec):
    return Path(root) / ("code_search_net" if spec[0] == CSN_REPO else "wikitext") / spec[2]


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document_order(record, seed):
    return hashlib.sha256(f"{seed}:{record['text_sha256']}".encode()).digest()


def _read_code(path):
    import pyarrow.parquet as pq
    records = []
    columns = ["func_code_string", "repository_name", "func_code_url"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            text = row["func_code_string"]
            if not isinstance(text, str) or not isinstance(row["repository_name"], str):
                raise ValueError("CodeSearchNet text/repository fields must be strings")
            records.append({"text": text, "text_sha256": _text_hash(text), "source_row": len(records),
                            "repository": row["repository_name"], "source_url": row["func_code_url"]})
    return records


def _read_wikitext(path):
    import pyarrow.parquet as pq
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    articles, lines, first = [], [], 0
    def finish():
        # Preserve each nonempty source line exactly; blank source rows supply
        # one newline. This is recorded preprocessing, not benchmark perplexity.
        text = "".join(line if line else "\n" for line in lines)
        if text.strip():
            articles.append({"text": text, "text_sha256": _text_hash(text), "source_row": first,
                             "source_row_end": first + len(lines), "repository": None, "source_url": None})
    for index, text in enumerate(rows):
        if not isinstance(text, str):
            raise ValueError("WikiText rows must contain string text")
        if re.fullmatch(r"\s*=\s*[^=]+?\s*=\s*", text):
            finish(); lines, first = [], index
        lines.append(text)
    finish()
    return articles


def _windows(token_count, length):
    start = 0
    while start < token_count - 1:
        count = min(length, token_count - start)
        yield start, count
        if start + count == token_count:
            break
        start += length - 1


def prepare_lm_data(*, raw_root, tokenizer_path, output_dir, train_token_budget=25_000_000,
                    seed=20260922, length=512):
    """Verify pinned raw bytes and publish one deterministic, noncycling corpus.

    Budget is unique CE targets, not padded tokens or repeated overlap tokens.
    The final selected document is retained whole, so the budget may overrun.
    Test splits are prepared for reproducibility but must remain unevaluated
    until the final comparison has been frozen.
    """
    import pyarrow
    import tokenizers
    from tokenizers import Tokenizer
    if type(train_token_budget) is not int or train_token_budget <= 0:
        raise ValueError("train_token_budget must be a positive integer")
    if type(length) is not int or not 3 <= length <= 512 or type(seed) is not int:
        raise ValueError("length must be 3..512 and seed must be an integer")
    output_dir, tokenizer_path = Path(output_dir), Path(tokenizer_path)
    if output_dir.exists():
        raise FileExistsError(f"Prepared dataset already exists: {output_dir}")
    raw_sources = {}
    for split, spec in RAW_SPECS.items():
        path = _raw_path(raw_root, spec)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != spec[3] or sha256_file(path) != spec[4]:
            raise ValueError(f"Raw dataset bytes differ from pinned source: {split}")
        raw_sources[split] = _source_record(spec)
    if tokenizer_path.is_symlink() or sha256_file(tokenizer_path) != FILE_SPECS["tokenizer.json"][1]:
        raise ValueError("Tokenizer bytes differ from native OLMo pin")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|endoftext|>") != EOS_ID:
        raise ValueError("Native tokenizer EOS differs")
    documents = {split: (_read_code(_raw_path(raw_root, spec)) if split in ("train", "dev", "test")
                         else _read_wikitext(_raw_path(raw_root, spec))) for split, spec in RAW_SPECS.items()}
    repository_sets = {split: {record["repository"] for record in documents[split]} for split in ("train", "dev", "test")}
    overlap = {f"{a}/{b}": sorted(repository_sets[a] & repository_sets[b])
               for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    if any(overlap.values()):
        raise ValueError("Official CodeSearchNet splits share repository identifiers")
    output_dir.mkdir(parents=True, exist_ok=False)
    seen_text, seen_tokens, next_doc_id = set(), set(), 0
    split_records, files = {}, {}
    # Heldout takes precedence over training; reserved final test takes precedence
    # over development. Both raw-byte and native-token duplicate identities count.
    priority = ("test", "dev", "retention_test", "retention_dev", "train")
    for split in priority:
        ordered = sorted(documents.pop(split), key=lambda row: (_document_order(row, seed), row["source_row"]))
        stats = {"source_documents": len(ordered), "documents": 0, "windows": 0, "ce_targets": 0,
                 "unique_input_tokens": 0, "window_input_tokens": 0, "latent_pairs": 0, "kl_triples": 0,
                 "kl_boundary_triples_omitted": 0, "raw_duplicates_skipped": 0, "token_duplicates_skipped": 0,
                 "empty_documents_skipped": 0}
        token_path, doc_path, window_path = (output_dir / f"{split}.{suffix}" for suffix in ("tokens.bin", "documents.jsonl", "windows.npy"))
        window_rows, token_offset = [], 0
        with token_path.open("xb") as token_file, doc_path.open("x", encoding="utf-8") as doc_file:
            for record in ordered:
                if not record["text"].strip():
                    stats["empty_documents_skipped"] += 1; continue
                text_sha = record["text_sha256"]
                if text_sha in seen_text:
                    stats["raw_duplicates_skipped"] += 1; continue
                ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
                ids.append(EOS_ID)
                tokens = np.asarray(ids, dtype="<u4")
                token_sha = hashlib.sha256(tokens.tobytes()).hexdigest()
                if token_sha in seen_tokens:
                    seen_text.add(text_sha)
                    stats["token_duplicates_skipped"] += 1; continue
                if len(tokens) < 2:
                    stats["empty_documents_skipped"] += 1; continue
                seen_text.add(text_sha); seen_tokens.add(token_sha)
                windows = list(_windows(len(tokens), length))
                tokens.tofile(token_file)
                for start, count in windows:
                    window_rows.append((token_offset + start, count, next_doc_id, start))
                metadata = {key: value for key, value in record.items() if key != "text"}
                metadata.update(document_id=next_doc_id, token_sha256=token_sha, token_count=len(tokens),
                                token_offset=token_offset, window_start=len(window_rows)-len(windows), windows=len(windows))
                doc_file.write(json.dumps(metadata, sort_keys=True) + "\n")
                next_doc_id += 1; token_offset += len(tokens)
                stats["documents"] += 1; stats["windows"] += len(windows)
                stats["ce_targets"] += len(tokens)-1; stats["latent_pairs"] += len(tokens)-1
                stats["unique_input_tokens"] += len(tokens)
                stats["window_input_tokens"] += sum(count for _, count in windows)
                stats["kl_triples"] += sum(max(count-2, 0) for _, count in windows)
                stats["kl_boundary_triples_omitted"] += max(len(windows)-1, 0)
                if split == "train" and stats["ce_targets"] >= train_token_budget:
                    break
        if stats["windows"] == 0 or (split == "train" and stats["ce_targets"] < train_token_budget):
            raise ValueError(f"Insufficient unique document targets for {split}")
        np.save(window_path, np.asarray(window_rows, dtype="<i8"), allow_pickle=False)
        for path in (token_path, doc_path, window_path):
            files[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        split_records[split] = stats
        print({"prepared_split": split, **stats}, flush=True)
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed, "length": length, "pad_id": PAD_ID, "eos_id": EOS_ID,
                "train_token_budget": train_token_budget, "budget_unit": "unique CE targets",
                "token_dtype": "<u4", "window_columns": ["token_offset", "length", "document_id", "document_token_start"],
                "raw_sources": raw_sources, "files": files, "splits": split_records,
                "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
                              "sha256": FILE_SPECS["tokenizer.json"][1], "add_special_tokens": False},
                "source_sha256": sha256_file(Path(__file__)),
                "versions": {"numpy": np.__version__, "pyarrow": pyarrow.__version__, "tokenizers": tokenizers.__version__},
                "preprocessing": {"code_field": "func_code_string", "code_text": "unchanged",
                                  "document_eos": "append one at actual end, none at cropped boundary",
                                  "window_stride": length-1, "window_context_overlap": 1, "packing": False,
                                  "wikitext": "top-level heading articles; concatenate original nonempty rows, blank row becomes newline",
                                  "dedup_priority": list(priority), "dedup_keys": ["UTF-8 SHA256", "native uint32 token SHA256"],
                                  "order": "SHA256(seed:UTF-8 SHA256), then original row index"},
                "repository_overlap": overlap, "final_test_evaluated": False,
                "licenses": {"code_search_net": "individual source repository licenses; HF does not include per-example license",
                             "wikitext_hub_metadata": ["cc-by-sa-3.0", "gfdl"],
                             "wikitext_card_caveat": "README body instead links CC BY-SA 4.0; retain source attribution"}}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


class PreparedLMData:
    """Verified read-only tokens; integer cursor names the next training window."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        self._windows, self._tokens = {}, {}
        for split in SPLITS:
            self._windows[split] = np.load(self.root / f"{split}.windows.npy", mmap_mode="r", allow_pickle=False)
            self._tokens[split] = np.memmap(self.root / f"{split}.tokens.bin", dtype="<u4", mode="r")
        self.split_sizes = {split: len(self._windows[split]) for split in SPLITS}

    def lengths(self, split):
        return self._windows[split][:, 1]

    @property
    def train_lengths(self):
        return self.lengths("train")

    def batch(self, split, indices, device="cpu"):
        import torch
        from .nextlat import NextLatBatch
        if split not in SPLITS:
            raise ValueError("Unknown prepared split")
        indices = list(indices)
        if not indices or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0 or i >= self.split_sizes[split] for i in indices):
            raise ValueError("Window indices must be a nonempty in-range integer sequence")
        shape = (len(indices), self.manifest["length"])
        ids = np.full(shape, PAD_ID, dtype=np.int64)
        valid = np.zeros(shape, dtype=np.bool_)
        docs = np.full(shape, -1, dtype=np.int64)
        for row, index in enumerate(indices):
            offset, count, document, _ = self._windows[split][index]
            ids[row, :count] = self._tokens[split][offset:offset+count]
            valid[row, :count] = True; docs[row, :count] = document
        return NextLatBatch(torch.from_numpy(ids).to(device), torch.from_numpy(valid).to(device), torch.from_numpy(docs).to(device))

    def next_train_batch(self, cursor, batch_size, device="cpu"):
        if type(cursor) is not int or cursor < 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Cursor/batch size must be nonnegative/positive integers")
        end = cursor + batch_size
        if end > self.split_sizes["train"]:
            raise StopIteration("Prepared training data exhausted; implicit cycling is forbidden")
        return self.batch("train", range(cursor, end), device=device), end


def load_lm_data(root, *, expected_manifest_sha256=None):
    """Check exact provenance and every prepared file once before memory mapping."""
    root = Path(root)
    if expected_manifest_sha256 is not None and sha256_file(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Prepared data manifest fingerprint differs")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("pad_id") != PAD_ID or manifest.get("eos_id") != EOS_ID:
        raise ValueError("Prepared data schema or native special IDs differ")
    length = manifest.get("length")
    if type(length) is not int or not 3 <= length <= 512 or manifest.get("token_dtype") != "<u4":
        raise ValueError("Prepared data length/token dtype differs")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("window_stride") != length-1 or preprocessing.get("window_context_overlap") != 1 or preprocessing.get("packing") is not False:
        raise ValueError("Prepared data window/packing contract differs")
    if manifest.get("raw_sources") != {split: _source_record(spec) for split, spec in RAW_SPECS.items()}:
        raise ValueError("Prepared data raw source provenance differs")
    if manifest.get("tokenizer") != {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
                                    "sha256": FILE_SPECS["tokenizer.json"][1], "add_special_tokens": False}:
        raise ValueError("Prepared tokenizer provenance differs")
    expected_files = {f"{split}.{suffix}" for split in SPLITS for suffix in ("tokens.bin", "documents.jsonl", "windows.npy")}
    if set(manifest.get("files", {})) != expected_files or set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("Prepared data inventory differs")
    if manifest.get("source_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Prepared data preprocessing source differs")
    for name, record in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared data file integrity differs: {name}")
    result = PreparedLMData(root, manifest)
    for split in SPLITS:
        rows, tokens = result._windows[split], result._tokens[split]
        if rows.dtype != np.dtype("<i8") or rows.ndim != 2 or rows.shape[1] != 4 or len(rows) != manifest["splits"][split]["windows"]:
            raise ValueError("Prepared window table shape/count differs")
        if bool(((rows[:, 0] < 0) | (rows[:, 1] < 2) | (rows[:, 1] > manifest["length"]) |
                 (rows[:, 0] + rows[:, 1] > len(tokens)) | (rows[:, 2] < 0) | (rows[:, 3] < 0)).any()):
            raise ValueError("Prepared window table bounds differ")
        documents = [json.loads(line) for line in (root / f"{split}.documents.jsonl").read_text().splitlines()]
        if len(documents) != manifest["splits"][split]["documents"]:
            raise ValueError("Prepared document count differs")
        next_offset, next_window = 0, 0
        for document in documents:
            count, offset, first = document["token_count"], document["token_offset"], document["window_start"]
            expected = [(offset+start, size, document["document_id"], start) for start, size in _windows(count, length)]
            if offset != next_offset or first != next_window or document["windows"] != len(expected):
                raise ValueError("Prepared document continuity differs")
            if not np.array_equal(rows[first:first+len(expected)], np.asarray(expected, dtype="<i8")):
                raise ValueError("Prepared windows cross or misidentify document boundaries")
            if count < 2 or offset+count > len(tokens) or tokens[offset+count-1] != EOS_ID:
                raise ValueError("Prepared document end/EOS differs")
            next_offset += count; next_window += len(expected)
        if next_offset != len(tokens) or next_window != len(rows):
            raise ValueError("Prepared document coverage differs")
        stats = manifest["splits"][split]
        if stats["ce_targets"] != int((rows[:, 1]-1).sum()) or stats["window_input_tokens"] != int(rows[:, 1].sum()):
            raise ValueError("Prepared target/input token counts differ")
    return result
