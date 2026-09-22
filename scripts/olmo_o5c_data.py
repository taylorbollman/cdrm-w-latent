#!/usr/bin/env python3
"""Fresh O5c code/general training plans with exact per-update CE quotas.

Each segment is one document's contiguous tokens. Splitting a window carries
one context token across the split, conserving every selected CE target once.
Input-token counts differ from CE counts by the number of segment rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from cdrm.pretrained import lm_data as lm
from cdrm.pretrained.artifacts import sha256_file, write_json

SCHEMA = "olmo-o5c-data-v1"
ARMS = ("code", "mixed")
DOMAINS = ("code", "general")
GENERAL_SPEC = (lm.WIKI_REPO, lm.WIKI_REVISION, "wikitext-2-raw-v1/train-00000-of-00001.parquet", 6357543,
                "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7")
SEGMENT_COLUMNS = ["domain", "source_window", "start_in_window", "length", "document_id"]
FILES = {"general.tokens.bin", "general.windows.npy", "general.documents.jsonl", "code.plan.npy", "mixed.plan.npy"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _hashes():
    return {"scripts/olmo_o5c_data.py": sha256_file(Path(__file__)), "cdrm/pretrained/lm_data.py": sha256_file(Path(lm.__file__))}


def _int(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def build_plan(code_windows, general_windows, *, arm, start_code_window, updates, ce_per_update):
    """Partition fresh windows into quota-sized updates without cycling targets."""
    require(arm in ARMS, "Unknown O5c arm")
    _int(start_code_window, "start_code_window")
    _int(updates, "updates", 1); _int(ce_per_update, "ce_per_update", 2)
    require(ce_per_update % 2 == 0, "Mixed CE quota must divide evenly")
    windows = [code_windows, general_windows]
    positions = [[start_code_window, 0], [0, 0]]
    segments = []
    row_prefix, token_prefix, ce_prefix = [0], [0], [0]
    domain_prefixes = {name: {key: [0] for key in ("ce_positions", "input_tokens", "documents")} for name in DOMAINS}
    for _ in range(updates):
        totals = {name: {key: 0 for key in ("ce_positions", "input_tokens", "documents")} for name in DOMAINS}
        # Both arms cut code at the same half-update CE boundaries. Combining
        # two chunks into a code-only update must not give its shared targets
        # longer context than the mixed arm receives.
        half = ce_per_update // 2
        chunks = [(0, half), (0, half)] if arm == "code" else [(0, half), (1, half)]
        for domain, quota in chunks:
            remaining = quota
            while remaining:
                window, start = positions[domain]
                require(window < len(windows[domain]), f"Insufficient fresh {DOMAINS[domain]} CE targets; cycling is forbidden")
                source = windows[domain][window]
                available = int(source[1]) - 1 - start
                require(available > 0, "Invalid source window/cursor")
                take = min(remaining, available)
                segments.append([domain, window, start, take + 1, int(source[2])])
                stats = totals[DOMAINS[domain]]
                stats["ce_positions"] += take; stats["input_tokens"] += take + 1; stats["documents"] += 1
                remaining -= take
                positions[domain] = [window + 1, 0] if take == available else [window, start + take]
        row_prefix.append(len(segments))
        token_prefix.append(token_prefix[-1] + sum(value["input_tokens"] for value in totals.values()))
        ce_prefix.append(ce_prefix[-1] + ce_per_update)
        for domain in DOMAINS:
            for key in totals[domain]:
                values = domain_prefixes[domain][key]
                values.append(values[-1] + totals[domain][key])
    plan = {"arm": arm, "total_updates": updates, "ce_per_update": ce_per_update,
            "batch_ce_prefix": ce_prefix, "batch_token_prefix": token_prefix, "batch_row_prefix": row_prefix,
            "domain_prefixes": domain_prefixes, "source_end_cursors": {name: positions[i] for i, name in enumerate(DOMAINS)},
            "row_order": "two code quota chunks per code update; code then general quota chunks per mixed update",
            "segment_quota_ce": ce_per_update // 2,
            "shared_code_context": "identical code segments and context resets at every half-update CE quota in both arms",
            "documents_counter_unit": "independent segment rows, not unique original documents"}
    return np.asarray(segments, dtype="<i8"), plan


def _base_document_hashes(base):
    texts, tokens, maximum_id = set(), set(), -1
    for split in lm.SPLITS:
        for line in (base.root / f"{split}.documents.jsonl").read_text().splitlines():
            row = json.loads(line)
            texts.add(row["text_sha256"]); tokens.add(row["token_sha256"])
            maximum_id = max(maximum_id, row["document_id"])
    return texts, tokens, maximum_id


def _prepare_general(raw_path, tokenizer_path, output, base, seed):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    require(tokenizer.token_to_id("<|endoftext|>") == lm.EOS_ID, "Native tokenizer EOS differs")
    records = sorted(lm._read_wikitext(raw_path), key=lambda row: (lm._document_order(row, seed), row["source_row"]))
    seen_text, seen_tokens, maximum_id = _base_document_hashes(base)
    offset, windows, documents = 0, [], []
    stats = {"source_documents": len(records), "byte_hash_duplicates_skipped": 0, "token_hash_duplicates_skipped": 0,
             "empty_documents_skipped": 0}
    with (output / "general.tokens.bin").open("xb") as stream:
        for record in records:
            if not record["text"].strip():
                stats["empty_documents_skipped"] += 1; continue
            if record["text_sha256"] in seen_text:
                stats["byte_hash_duplicates_skipped"] += 1; continue
            ids = np.asarray([*tokenizer.encode(record["text"], add_special_tokens=False).ids, lm.EOS_ID], dtype="<u4")
            token_sha = hashlib.sha256(ids.tobytes()).hexdigest()
            if token_sha in seen_tokens:
                seen_text.add(record["text_sha256"])
                stats["token_hash_duplicates_skipped"] += 1; continue
            if len(ids) < 2:
                stats["empty_documents_skipped"] += 1; continue
            seen_text.add(record["text_sha256"]); seen_tokens.add(token_sha)
            document_id = maximum_id + 1 + len(documents)
            pieces = list(lm._windows(len(ids), 512))
            metadata = {key: value for key, value in record.items() if key != "text"}
            metadata.update(document_id=document_id, token_sha256=token_sha, token_count=len(ids), token_offset=offset,
                            window_start=len(windows), windows=len(pieces))
            windows.extend([offset + start, length, document_id, start] for start, length in pieces)
            ids.tofile(stream); offset += len(ids); documents.append(metadata)
    require(bool(documents), "No general-text documents remain after exclusion")
    rows = np.asarray(windows, dtype="<i8")
    np.save(output / "general.windows.npy", rows, allow_pickle=False)
    (output / "general.documents.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in documents))
    stats.update(documents=len(documents), windows=len(rows), unique_input_tokens=offset,
                 window_input_tokens=int(rows[:, 1].sum()), ce_targets=int((rows[:, 1] - 1).sum()))
    return rows, stats


def prepare_data(*, base_root, raw_general, tokenizer_path, previous_report, output_dir,
                 updates=512, ce_per_update=8192, seed=20260922):
    base = lm.load_lm_data(base_root)
    output, raw, tokenizer_path, previous_report = map(Path, (output_dir, raw_general, tokenizer_path, previous_report))
    require(not output.exists(), "Output dataset already exists; choose a new lineage")
    require(raw.is_file() and not raw.is_symlink() and raw.stat().st_size == GENERAL_SPEC[3]
            and sha256_file(raw) == GENERAL_SPEC[4], "General raw bytes differ from pinned WikiText train")
    require(tokenizer_path.is_file() and not tokenizer_path.is_symlink()
            and sha256_file(tokenizer_path) == lm.FILE_SPECS["tokenizer.json"][1], "Tokenizer differs from native pin")
    report = json.loads(previous_report.read_text())
    require(report.get("schema") == "olmo-o5b-arm-v1" and report.get("arm") == "fbt"
            and report.get("status") == "completed" and report.get("finished_utc"), "Require completed O5b FBT parent report")
    require(report.get("configuration", {}).get("data_manifest_sha256") == base.manifest_sha256,
            "O5b parent consumed a different prepared corpus")
    start = _int(report.get("data_cursor"), "parent cursor")
    require(start == report.get("counters", {}).get("documents") and start < base.split_sizes["train"], "Parent window cursor differs")
    code = base._windows["train"]
    require(start == 0 or code[start, 2] != code[start - 1, 2], "Fresh-code boundary is inside a previously consumed document")
    require(int(code[start, 3]) == 0, "Fresh code must begin at a document boundary")
    output.mkdir(parents=True, exist_ok=False)
    general, stats = _prepare_general(raw, tokenizer_path, output, base, seed)
    plans = {}
    for arm in ARMS:
        segments, plan = build_plan(code, general, arm=arm, start_code_window=start, updates=updates, ce_per_update=ce_per_update)
        np.save(output / f"{arm}.plan.npy", segments, allow_pickle=False)
        plans[arm] = plan
    files = {name: {"sha256": sha256_file(output / name), "size_bytes": (output / name).stat().st_size} for name in sorted(FILES)}
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "length": 512, "pad_id": lm.PAD_ID, "eos_id": lm.EOS_ID, "token_dtype": "<u4",
                "source_hashes": _hashes(), "base_manifest_sha256": base.manifest_sha256,
                "base_prepared_files": base.manifest["files"], "tokenizer": base.manifest["tokenizer"],
                "base_raw_sources": base.manifest.get("raw_sources", {}),
                "versions": {"numpy": np.__version__, "pyarrow": importlib.metadata.version("pyarrow"),
                             "tokenizers": importlib.metadata.version("tokenizers")},
                "general_raw_source": lm._source_record(GENERAL_SPEC), "general_statistics": stats,
                "parent": {"report_sha256": sha256_file(previous_report), "arm": "fbt", "consumed_windows": start,
                           "completed_counters": report["counters"]}, "files": files,
                "fresh_code": {"start_window": start, "remaining_windows": len(code) - start,
                               "remaining_input_tokens": int(code[start:, 1].sum()),
                               "remaining_ce_targets": int((code[start:, 1] - 1).sum()), "starts_new_document": True},
                "plans": plans, "segment_columns": SEGMENT_COLUMNS,
                "preprocessing": {"general_text": base.manifest["preprocessing"]["wikitext"],
                                  "order": "SHA256(seed:UTF-8 SHA256), then original row index",
                                  "dedup": "Exclude general documents matching any base train/dev/test full UTF-8 or native-token SHA256; then deduplicate general itself",
                                  "window_stride": 511, "packing": False,
                                  "segmentation": "Split quota boundaries with one carried context token; every selected CE target once; never append EOS at a segment/crop boundary",
                                  "document_eos": "Native EOS once at actual source document end"},
                "licenses": base.manifest["licenses"], "test_evaluated": False,
                "qualification": "Exact full-document hash exclusion, not semantic or substring deduplication. General text is WikiText train, not broad representative web text. "
                    "Original-pretraining overlap is unknown. Updates match CE target counts; valid input-token and segment-row counts differ. No document packing or implicit cycling."}
    write_json(output / "manifest.json", manifest)
    return manifest


class O5cData:
    def __init__(self, root, base, manifest):
        self.root, self.base, self.manifest = Path(root), base, manifest
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        self.general_tokens = np.memmap(self.root / "general.tokens.bin", dtype="<u4", mode="r")
        self.general_windows = np.load(self.root / "general.windows.npy", mmap_mode="r", allow_pickle=False)
        self.segments = {arm: np.load(self.root / f"{arm}.plan.npy", mmap_mode="r", allow_pickle=False) for arm in ARMS}

    def plan(self, arm):
        require(arm in ARMS, "Unknown O5c arm")
        return self.manifest["plans"][arm]

    def batch(self, arm, indices, device="cpu"):
        import torch
        from cdrm.pretrained.nextlat import NextLatBatch
        require(arm in ARMS, "Unknown O5c arm")
        indices = list(indices)
        require(bool(indices) and all(type(i) is int and 0 <= i < len(self.segments[arm]) for i in indices), "Invalid segment selection")
        shape = (len(indices), 512)
        ids = np.full(shape, lm.PAD_ID, dtype=np.int64)
        valid = np.zeros(shape, dtype=np.bool_)
        docs = np.full(shape, -1, dtype=np.int64)
        for row, index in enumerate(indices):
            domain, window, start, length, document = self.segments[arm][index]
            windows = self.base._windows["train"] if domain == 0 else self.general_windows
            tokens = self.base._tokens["train"] if domain == 0 else self.general_tokens
            offset = windows[window, 0] + start
            ids[row, :length] = tokens[offset:offset + length]
            valid[row, :length] = True; docs[row, :length] = document
        return NextLatBatch(torch.from_numpy(ids).to(device), torch.from_numpy(valid).to(device), torch.from_numpy(docs).to(device))

    def batch_for_update(self, arm, update, device="cpu"):
        _int(update, "update")
        plan = self.plan(arm)
        if update >= plan["total_updates"]:
            raise StopIteration("Frozen data schedule exhausted; no implicit cycling")
        bounds = plan["batch_row_prefix"]
        return self.batch(arm, range(bounds[update], bounds[update + 1]), device)

    def next_train_batch(self, arm, cursor, device="cpu"):
        return self.batch_for_update(arm, cursor, device), cursor + 1


def load_o5c_data(root, base_root, *, expected_manifest_sha256=None):
    root = Path(root)
    require((root / "manifest.json").is_file() and not (root / "manifest.json").is_symlink(), "Missing regular prepared manifest")
    if expected_manifest_sha256 is not None:
        require(sha256_file(root / "manifest.json") == expected_manifest_sha256, "O5c manifest fingerprint differs")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest.get("schema") == SCHEMA and manifest.get("source_hashes") == _hashes(), "O5c preparation source/schema differs")
    require(manifest.get("general_raw_source") == lm._source_record(GENERAL_SPEC), "WikiText train provenance differs")
    require(manifest.get("length") == 512 and manifest.get("pad_id") == lm.PAD_ID and manifest.get("eos_id") == lm.EOS_ID
            and manifest.get("token_dtype") == "<u4" and manifest.get("segment_columns") == SEGMENT_COLUMNS, "Prepared geometry differs")
    require(set(manifest.get("files", {})) == FILES and set(manifest.get("plans", {})) == set(ARMS), "Prepared inventory differs")
    base = lm.load_lm_data(base_root, expected_manifest_sha256=manifest["base_manifest_sha256"])
    require(manifest.get("base_prepared_files") == base.manifest["files"] and manifest.get("tokenizer") == base.manifest["tokenizer"], "Base prepared artifacts/tokenizer differ")
    for name, value in manifest["files"].items():
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == value["size_bytes"]
                and sha256_file(path) == value["sha256"], f"Prepared data bytes differ: {name}")
    result = O5cData(root, base, manifest)
    general = result.general_windows
    require(general.dtype == np.dtype("<i8") and general.ndim == 2 and general.shape[1] == 4, "General window table differs")
    texts, token_hashes, maximum_id = _base_document_hashes(base)
    offset = first = 0
    documents = [json.loads(line) for line in (root / "general.documents.jsonl").read_text().splitlines()]
    for index, row in enumerate(documents):
        count, document = row["token_count"], row["document_id"]
        expected = np.asarray([[offset + start, length, document, start] for start, length in lm._windows(count, 512)], dtype="<i8")
        require(count >= 2 and row["token_offset"] == offset and row["window_start"] == first and row["windows"] == len(expected)
                and document == maximum_id + 1 + index, "General document continuity/identity differs")
        require(np.array_equal(general[first:first + len(expected)], expected), "General windows cross document boundaries")
        tokens = result.general_tokens[offset:offset + count]
        require(len(tokens) == count and tokens[-1] == lm.EOS_ID and hashlib.sha256(tokens.tobytes()).hexdigest() == row["token_sha256"], "General token bytes/EOS differ")
        require(row["text_sha256"] not in texts and row["token_sha256"] not in token_hashes, "General document overlaps existing train/dev/test")
        texts.add(row["text_sha256"]); token_hashes.add(row["token_sha256"])
        offset += count; first += len(expected)
    require(offset == len(result.general_tokens) and first == len(general), "General document coverage differs")
    statistics = manifest["general_statistics"]
    require(statistics["documents"] == len(documents) and statistics["windows"] == len(general)
            and statistics["unique_input_tokens"] == offset and statistics["window_input_tokens"] == int(general[:, 1].sum())
            and statistics["ce_targets"] == int((general[:, 1] - 1).sum()), "General statistics differ from prepared bytes")
    start = manifest["fresh_code"]["start_window"]
    require(start == manifest["parent"]["consumed_windows"] == manifest["parent"]["completed_counters"]["documents"], "Fresh parent cursor differs")
    code = base._windows["train"]
    require(0 <= start < len(code) and int(code[start, 3]) == 0 and (start == 0 or code[start, 2] != code[start - 1, 2]), "Fresh-code start repeats a consumed document")
    require(manifest["fresh_code"] == {"start_window": start, "remaining_windows": len(code) - start,
            "remaining_input_tokens": int(code[start:, 1].sum()), "remaining_ce_targets": int((code[start:, 1] - 1).sum()),
            "starts_new_document": True}, "Fresh-code statistics differ")
    for arm in ARMS:
        plan = result.plan(arm)
        expected, expected_plan = build_plan(code, general, arm=arm, start_code_window=start,
                                             updates=plan["total_updates"], ce_per_update=plan["ce_per_update"])
        require(result.segments[arm].dtype == np.dtype("<i8") and np.array_equal(expected, result.segments[arm])
                and expected_plan == plan, "Frozen segmentation or token quota plan differs")
    require(result.plan("code")["total_updates"] == result.plan("mixed")["total_updates"]
            and result.plan("code")["ce_per_update"] == result.plan("mixed")["ce_per_update"], "Arm target/update budgets differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-root", "raw-general", "tokenizer", "previous-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--ce-per-update", type=int, default=8192)
    args = parser.parse_args()
    manifest = prepare_data(base_root=args.base_root, raw_general=args.raw_general, tokenizer_path=args.tokenizer,
                            previous_report=args.previous_report, output_dir=args.output_dir,
                            updates=args.updates, ce_per_update=args.ce_per_update)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "general": manifest["general_statistics"],
                      "plans": {arm: {"updates": plan["total_updates"], "ce": plan["batch_ce_prefix"][-1],
                                      "input_tokens": plan["batch_token_prefix"][-1], "rows": plan["batch_row_prefix"][-1]}
                                for arm, plan in manifest["plans"].items()}}, indent=2))


if __name__ == "__main__":
    main()
