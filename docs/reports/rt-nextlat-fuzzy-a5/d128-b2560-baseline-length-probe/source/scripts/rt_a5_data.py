#!/usr/bin/env python3
"""Native A5 prefix-state datasets; this module does not import PyTorch.

The mathematical/serialization contract follows the pinned sources in
docs/rt-a5-experiment-plan.md. This is an independent implementation, not a
copy of either repository's generator or training code.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "rt-a5-data-v1"
GENERATOR_VERSION = "native-a5-pcg64-v1"
MANIFEST = "manifest.json"
ROLES = ("train", "dev", "train36", "ood_dev", "confirmation")
_VALIDATED: dict[str, tuple] = {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def alphabet() -> np.ndarray:
    """ID-ordered even permutations, with one-based images and identity ID 0."""
    rows = [p for p in itertools.permutations(range(1, 6))
            if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
    result = np.asarray(rows, dtype=np.uint8)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=1)
def multiplication_table() -> np.ndarray:
    """table[p,q] encodes p(q(i)): the rightmost permutation acts first."""
    elements = alphabet()
    ids = {tuple(row): index for index, row in enumerate(elements)}
    result = np.asarray([[ids[tuple(p[q - 1])] for q in elements]
                         for p in elements], dtype=np.uint8)
    result.flags.writeable = False
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _words(words, *, name: str = "words") -> np.ndarray:
    words = np.asarray(words)
    if words.ndim != 2 or words.shape[1] < 1 or words.dtype.kind not in "ui":
        raise ValueError(f"{name} must be a two-dimensional integer array with nonempty words")
    if words.size and (words.min() < 0 or words.max() >= 60):
        raise ValueError(f"{name} contains an operation outside IDs 0..59")
    return np.ascontiguousarray(words, dtype=np.uint8)


def _keys(words: np.ndarray, length: int | None = None) -> np.ndarray:
    words = _words(words)
    length = words.shape[1] if length is None else length
    if length < 1 or length > words.shape[1]:
        raise ValueError("Invalid comparison-prefix length")
    return np.ascontiguousarray(words[:, :length]).view(np.dtype((np.void, length))).ravel()


def prefix_labels(words: np.ndarray) -> np.ndarray:
    """State after each input operation: no BOS, target shift, or feedback."""
    words = _words(words)
    table = multiplication_table()
    labels = np.empty_like(words)
    state = np.zeros(len(words), dtype=np.uint8)
    for position in range(words.shape[1]):
        state = table[state, words[:, position]]
        labels[:, position] = state
    return labels


def generate_unique_words(count: int, length: int, seed: int, *,
                          forbidden_prefixes: np.ndarray | None = None,
                          return_stats: bool = False):
    """Uniform operation draws, retaining first occurrences in draw order.

    Rejected complete duplicates and held-out-prefix collisions are recorded.
    Sampling differs deliberately from the upstream Python/set-based generator;
    alphabet, composition, unique-word contract and stored data are authoritative.
    """
    count = _positive_integer(count, "count")
    length = _positive_integer(length, "length")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if count > 60 ** length:
        raise ValueError("Requested more unique words than exist")
    forbidden = None
    prefix_length = None
    if forbidden_prefixes is not None:
        forbidden_prefixes = _words(forbidden_prefixes, name="forbidden_prefixes")
        prefix_length = forbidden_prefixes.shape[1]
        if prefix_length > length:
            raise ValueError("Forbidden prefixes are longer than generated words")
        forbidden = np.unique(_keys(forbidden_prefixes))
        if count > (60 ** prefix_length - len(forbidden)) * 60 ** (length - prefix_length):
            raise ValueError("Too few words remain after forbidden-prefix exclusion")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    accepted = np.empty((0, length), dtype=np.uint8)
    accepted_keys = _keys(accepted)
    stats = {"drawn": 0, "duplicate_words_rejected": 0, "forbidden_prefixes_rejected": 0}
    for _ in range(1000):
        remaining = count - len(accepted)
        if not remaining:
            break
        candidate = rng.integers(0, 60, size=(remaining, length), dtype=np.uint8)
        stats["drawn"] += len(candidate)
        if forbidden is not None:
            excluded = np.isin(_keys(candidate, prefix_length), forbidden)
            stats["forbidden_prefixes_rejected"] += int(excluded.sum())
            candidate = candidate[~excluded]
        keys = _keys(candidate)
        _, first = np.unique(keys, return_index=True)
        first.sort()  # np.unique's key order must not become the training order.
        stats["duplicate_words_rejected"] += len(candidate) - len(first)
        candidate, keys = candidate[first], keys[first]
        if len(accepted_keys):
            duplicate = np.isin(keys, accepted_keys)
            stats["duplicate_words_rejected"] += int(duplicate.sum())
            candidate, keys = candidate[~duplicate], keys[~duplicate]
        accepted = np.concatenate((accepted, candidate))
        accepted_keys = np.concatenate((accepted_keys, keys))
    if len(accepted) != count:
        raise ValueError("Unique-word sampling did not converge; reduce the requested population")
    return (accepted, stats) if return_stats else accepted


def split_indices(count: int, train_fraction: float = .8, seed: int = 42):
    """ShuffleSplit-style ordering: shuffled development rows first, then train."""
    count = _positive_integer(count, "count")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    # Complementing .8 in floating point can change ceil for exact small fixtures.
    development_count = int(math.ceil(round(count * (1 - train_fraction), 12)))
    if development_count >= count:
        raise ValueError("Both training and development need at least one row")
    order = np.random.RandomState(seed).permutation(count).astype(np.int64)
    return order[development_count:], order[:development_count]


def leakage_audit(splits: dict[str, np.ndarray], prefix_length: int = 12) -> dict:
    """Audit all role pairs at the training length and equal-length whole words."""
    keys = {name: np.unique(_keys(rows, prefix_length)) for name, rows in splits.items()}
    result = {"prefix_length": prefix_length, "prefix_intersections": {},
              "complete_word_intersections": {}, "duplicate_words": {},
              "within_split_duplicate_prefixes": {}}
    for name, rows in splits.items():
        result["duplicate_words"][name] = len(rows) - len(np.unique(_keys(rows)))
        result["within_split_duplicate_prefixes"][name] = len(rows) - len(keys[name])
    for left, right in itertools.combinations(splits, 2):
        pair = f"{left}__{right}"
        result["prefix_intersections"][pair] = int(np.intersect1d(keys[left], keys[right]).size)
        if splits[left].shape[1] == splits[right].shape[1]:
            result["complete_word_intersections"][pair] = int(np.intersect1d(
                _keys(splits[left]), _keys(splits[right])).size)
    return result


def _save_array(root: Path, name: str, array: np.ndarray, files: dict) -> str:
    path = root / name
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    files[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size,
                   "shape": list(array.shape), "dtype": str(array.dtype)}
    return name


def prepare_dataset(output_dir: str | Path, *, pool_size: int = 1_000_000,
                    confirmation_size: int = 102_400, ood_eval_size: int = 102_400,
                    short_length: int = 12, long_length: int = 36,
                    train_fraction: float = .8, seed12: int = 444, seed36: int = 445,
                    confirmation_seed: int = 446, split_seed: int = 42) -> dict:
    """Create a fresh immutable corpus, split manifests, and untouched final set."""
    pool_size = _positive_integer(pool_size, "pool_size")
    confirmation_size = _positive_integer(confirmation_size, "confirmation_size")
    ood_eval_size = _positive_integer(ood_eval_size, "ood_eval_size")
    short_length = _positive_integer(short_length, "short_length")
    long_length = _positive_integer(long_length, "long_length")
    if long_length <= short_length:
        raise ValueError("long_length must exceed short_length")
    if len({seed12, seed36, confirmation_seed}) != 3:
        raise ValueError("Independent length-12, length-36 and confirmation streams need distinct seeds")
    train_ids, dev_ids = split_indices(pool_size, train_fraction, split_seed)
    if ood_eval_size > len(dev_ids):
        raise ValueError("ood_eval_size exceeds the length-36 development split")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    files: dict = {}
    _save_array(root, "alphabet.npy", alphabet(), files)
    _save_array(root, "multiplication_table.npy", multiplication_table(), files)
    short, short_stats = generate_unique_words(pool_size, short_length, seed12, return_stats=True)
    # Exclude full training-length prefixes of both short-word roles. Shorter
    # common prefixes are expected and intentionally remain unrestricted.
    long, long_stats = generate_unique_words(pool_size, long_length, seed36,
                                             forbidden_prefixes=short, return_stats=True)
    excluded_confirmation = np.concatenate((short, long[:, :short_length]))
    confirmation, confirmation_stats = generate_unique_words(
        confirmation_size, long_length, confirmation_seed,
        forbidden_prefixes=excluded_confirmation, return_stats=True)
    arrays = {"train": short[train_ids], "dev": short[dev_ids],
              "train36": long[train_ids], "ood_dev": long[dev_ids],
              "confirmation": confirmation}
    audit = leakage_audit(arrays, short_length)
    # Any complete-word overlap or short/long/final leakage is a hard error.
    # Distinct long words can legitimately share a short prefix; report those
    # (including the unused train-at-36 control) instead of hiding them.
    if any(audit["duplicate_words"].values()) or any(audit["complete_word_intersections"].values()):
        raise AssertionError("Complete-word disjointness audit failed")
    for pair, overlap in audit["prefix_intersections"].items():
        if overlap and pair != "train36__ood_dev":
            raise AssertionError(f"Training-prefix leakage: {pair}")
    roles = {}
    for role, inputs in arrays.items():
        labels = prefix_labels(inputs)
        ids = train_ids if role in ("train", "train36") else dev_ids
        if role == "confirmation":
            ids = np.arange(confirmation_size, dtype=np.int64)
        roles[role] = {
            "inputs": _save_array(root, f"{role}.inputs.npy", inputs, files),
            "labels": _save_array(root, f"{role}.labels.npy", labels, files),
            "source_indices": _save_array(root, f"{role}.indices.npy", ids, files),
            "stored_rows": len(inputs), "rows": ood_eval_size if role == "ood_dev" else len(inputs),
            "length": inputs.shape[1],
            "source_pool": "confirmation" if role == "confirmation" else
                ("short" if role in ("train", "dev") else "long"),
            "allowed_use": "frozen_final_confirmation_only" if role == "confirmation" else
                ("reserved_train_at_36_control" if role == "train36" else role),
        }
    manifest = {
        "schema": SCHEMA, "status": "complete", "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": file_sha256(__file__), "numpy_version": np.__version__,
        "rng": "numpy.random.Generator(PCG64); uint8 uniform draws in retained draw order",
        "task": {"group": "A5", "vocab_size": 60, "identity_id": 0,
                 "alphabet_order": "lexicographic even permutations of (1,2,3,4,5)",
                 "composition": "p(q(i)); state_t = state_(t-1) compose input_t",
                 "label_alignment": "same position, after input operation; no BOS or shift"},
        "sampling": {"pool_size": pool_size, "short_length": short_length,
                     "long_length": long_length, "confirmation_size": confirmation_size,
                     "seed12": seed12, "seed36": seed36, "confirmation_seed": confirmation_seed,
                     "draw_statistics": {"short": short_stats, "long": long_stats,
                                         "confirmation": confirmation_stats},
                     "long_prefix_exclusion": "all short-pool complete words",
                     "confirmation_prefix_exclusion": "all short-pool words and all long-pool short prefixes"},
        "split": {"seed": split_seed, "train_fraction": train_fraction,
                  "algorithm": "RandomState(seed).permutation(pool_size); ceil((1-fraction)*N) dev first; rest train",
                  "ood_eval_size": ood_eval_size, "ood_selection": "first ood_eval_size rows of shuffled long dev split"},
        "confirmation_evaluated": False, "roles": roles, "leakage_audit": audit, "files": files,
        "references": {
            "nextlat": "https://github.com/JaydenTeoh/NextLat/tree/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9",
            "word_problem": "https://github.com/jopetty/word-problem/tree/8f910f92e1c70455dcd9376f56032dfc55126188"},
    }
    with (root / MANIFEST).open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    validate_manifest(root)
    return manifest


def validate_manifest(root: str | Path) -> dict:
    """Validate hashes once per unchanged manifest/files within this process."""
    root = Path(root).resolve()
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete A5 dataset manifest")
    if set(manifest.get("roles", {})) != set(ROLES) or not manifest.get("files"):
        raise ValueError("Dataset manifest has missing or unexpected roles/files")
    paths = [manifest_path]
    for name in manifest["files"]:
        path = (root / name).resolve()
        if path.parent != root or path.suffix != ".npy":
            raise ValueError("Dataset manifest path escapes the corpus directory")
        paths.append(path)
    fingerprint = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns,
                         path.stat().st_ctime_ns) for path in paths)
    if _VALIDATED.get(str(root)) == fingerprint:
        return manifest
    for name, metadata in manifest["files"].items():
        path = root / name
        if path.stat().st_size != metadata["bytes"] or file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Dataset file hash/size differs: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
            raise ValueError(f"Dataset array shape/dtype differs: {name}")
    for role, metadata in manifest["roles"].items():
        for field in ("inputs", "labels", "source_indices"):
            if metadata.get(field) not in manifest["files"]:
                raise ValueError(f"Missing {role} {field} file")
        shape = [metadata["stored_rows"], metadata["length"]]
        if not 0 < metadata["rows"] <= metadata["stored_rows"]:
            raise ValueError(f"Invalid visible row count: {role}")
        for field in ("inputs", "labels"):
            entry = manifest["files"][metadata[field]]
            if entry["shape"] != shape or entry["dtype"] != "uint8":
                raise ValueError(f"Invalid A5 input/label schema: {role}")
        entry = manifest["files"][metadata["source_indices"]]
        if entry["shape"] != [metadata["stored_rows"]] or entry["dtype"] != "int64":
            raise ValueError(f"Invalid split-index schema: {role}")
    if not np.array_equal(np.load(root / "alphabet.npy", allow_pickle=False), alphabet()):
        raise ValueError("Dataset alphabet does not match the A5 ID contract")
    if not np.array_equal(np.load(root / "multiplication_table.npy", allow_pickle=False), multiplication_table()):
        raise ValueError("Dataset multiplication table does not match the A5 composition contract")
    _VALIDATED[str(root)] = fingerprint
    return manifest


def load_split(root: str | Path, role: str, *, verify: bool = True):
    """Read-only uint8 input/label memmaps in their frozen split order.

    The caller must explicitly select confirmation; normal training/evaluation
    code should reject that role until a separate final protocol is frozen.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown A5 split {role!r}; expected one of {ROLES}")
    root = Path(root)
    manifest = validate_manifest(root) if verify else json.loads((root / MANIFEST).read_text())
    metadata = manifest["roles"][role]
    rows = metadata["rows"]
    return tuple(np.load(root / metadata[field], mmap_mode="r", allow_pickle=False)[:rows]
                 for field in ("inputs", "labels"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create a fresh, immutable A5 dataset directory")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--pool-size", type=int, default=1_000_000)
    prepare.add_argument("--confirmation-size", type=int, default=102_400)
    prepare.add_argument("--ood-eval-size", type=int, default=102_400)
    prepare.add_argument("--short-length", type=int, default=12)
    prepare.add_argument("--long-length", type=int, default=36)
    prepare.add_argument("--train-fraction", type=float, default=.8)
    prepare.add_argument("--seed12", type=int, default=444)
    prepare.add_argument("--seed36", type=int, default=445)
    prepare.add_argument("--confirmation-seed", type=int, default=446)
    prepare.add_argument("--split-seed", type=int, default=42)
    arguments = vars(parser.parse_args())
    arguments.pop("command")
    manifest = prepare_dataset(**arguments)
    print(json.dumps({"status": manifest["status"], "output_dir": str(arguments["output_dir"]),
                      "roles": {key: {"rows": row["rows"], "length": row["length"]}
                                for key, row in manifest["roles"].items()},
                      "leakage_audit": manifest["leakage_audit"]}, indent=2))


if __name__ == "__main__":
    main()
