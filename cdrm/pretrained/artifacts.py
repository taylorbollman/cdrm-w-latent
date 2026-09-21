"""Pinned Stage A OpenELM artifacts; downloads never execute remote model code."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any
import urllib.request


CORENET_REVISION = "f9f83e616a34d02c422733a06a3fe5bde63ae575"
HF_REVISION = "ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f"
TOKENIZER_REPO = "meta-llama/Llama-2-7b-hf"
TOKENIZER_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
# Observed from an authorized download of tokenizer.model at the above revision.
TOKENIZER_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
CHECKPOINT_URL = (
    "https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/"
    "openelm/pretrained/1.1B/checkpoint_epoch_0_iter_299999.pt"
)
CHECKPOINT_FILENAME = "checkpoint_epoch_0_iter_299999.pt"
CHECKPOINT_SIZE = 4_320_718_260
# Acquired from the pinned Apple URL in Stage A, not an Apple-published checksum.
CHECKPOINT_SHA256 = "0e79fef600d022da33111f86dae4e7f4a4fd1d2e9045364c5ccd4c2283c4d9c6"
CORENET_PATHS = (
    "LICENSE",
    "projects/openelm/pretraining_configs/openelm_1_1B.yaml",
    "projects/openelm/README.md",
    "projects/openelm/README-pretraining.md",
    "corenet/modeling/models/language_modeling/general_gpt.py",
    "corenet/modeling/layers/normalization/rms_norm.py",
    "corenet/modeling/layers/rotary_embeddings.py",
    "corenet/modeling/layers/linear_layer.py",
    "corenet/modeling/layers/embedding.py",
    "corenet/modeling/layers/activation/swish.py",
    "corenet/utils/math_utils.py",
    "corenet/utils/checkpoint_utils.py",
    "corenet/data/text_tokenizer/sentencepiece_tokenizer.py",
    "corenet/loss_fn/language_modeling/cross_entropy.py",
)
HF_PATHS = ("config.json", "configuration_openelm.py", "modeling_openelm.py")


@dataclass(frozen=True)
class ArtifactSpec:
    url: str
    size_bytes: int | None = None
    sha256: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _record_path(path: Path) -> Path:
    return path.with_name(path.name + ".artifact.json")


def validate_artifact(path: str | Path, spec: ArtifactSpec) -> dict[str, Any]:
    """Rehash existing bytes; a same-size file without provenance is not accepted."""
    path = Path(path)
    record_path = _record_path(path)
    if not record_path.is_file():
        raise ValueError(f"Existing artifact has no integrity record: {path}")
    record = json.loads(record_path.read_text())
    if record.get("url") != spec.url:
        raise ValueError(f"Artifact source differs from the pinned source: {path}")
    size = path.stat().st_size
    if size != record.get("size_bytes") or (spec.size_bytes is not None and size != spec.size_bytes):
        raise ValueError(f"Artifact byte count differs from its manifest: {path}")
    digest = sha256_file(path)
    if digest != record.get("sha256") or (spec.sha256 is not None and digest != spec.sha256):
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    return record


def download_artifact(spec: ArtifactSpec, destination: str | Path) -> dict[str, Any]:
    """Resume verified partial bytes; publish only a complete, hashed artifact.

    The SHA256 of Apple's model-only file is recorded on first acquisition;
    Apple does not publish an authoritative SHA256 in the checkpoint table.
    Subsequent uses check that recorded digest and the pinned byte count.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return validate_artifact(destination, spec)
    with urllib.request.urlopen(urllib.request.Request(spec.url, method="HEAD"), timeout=60) as response:
        size_header = response.headers.get("Content-Length")
        remote_size = int(size_header) if size_header is not None else None
        identity = {
            "url": spec.url,
            "size_bytes": remote_size,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    if spec.size_bytes is not None and remote_size not in (None, spec.size_bytes):
        raise ValueError("Remote byte count differs from the pinned artifact")
    total = spec.size_bytes if spec.size_bytes is not None else remote_size
    if total is None:
        raise ValueError("Download requires an expected size or Content-Length")
    identity["size_bytes"] = total
    partial = destination.with_name(destination.name + ".part")
    partial_record = partial.with_name(partial.name + ".json")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        if not partial_record.exists() or json.loads(partial_record.read_text()) != identity:
            raise ValueError("Partial artifact has missing or changed remote identity; refusing to append")
        if offset > total:
            raise ValueError("Partial artifact is larger than the expected download")
        if not identity["etag"] and not identity["last_modified"]:
            raise ValueError("Cannot safely resume without ETag or Last-Modified")
    write_json(partial_record, identity)
    if offset < total:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            headers["If-Range"] = identity["etag"] or identity["last_modified"]
        with urllib.request.urlopen(urllib.request.Request(spec.url, headers=headers), timeout=60) as response:
            status = response.status
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if match is None or int(match[1]) != offset or int(match[3]) != total or int(match[2]) != total - 1:
                    raise ValueError("Invalid Content-Range on resumed download")
                mode = "ab" if offset else "wb"
            elif status == 200:
                # A server that ignores Range supplies the full object; restart rather than append.
                mode = "wb"
            else:
                raise ValueError(f"Unexpected download status: {status}")
            if identity["etag"] and response.headers.get("ETag") != identity["etag"]:
                raise ValueError("Remote ETag changed between HEAD and GET")
            with partial.open(mode) as output:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    if output.tell() > total:
                        raise ValueError("Download exceeds the expected byte count")
                output.flush()
                os.fsync(output.fileno())
    if partial.stat().st_size != total:
        raise ValueError("Incomplete download retained as .part; rerun to resume")
    digest = sha256_file(partial)
    if spec.sha256 is not None and digest != spec.sha256:
        raise ValueError("Downloaded artifact SHA256 differs from the pinned digest")
    record = {**identity, "sha256": digest}
    write_json(_record_path(destination), record)
    partial.replace(destination)
    partial_record.unlink(missing_ok=True)
    return record


def prepare_sources(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    records = {}
    for family, revision, paths, base in (
        ("corenet", CORENET_REVISION, CORENET_PATHS, f"https://raw.githubusercontent.com/apple/corenet/{CORENET_REVISION}/"),
        ("huggingface", HF_REVISION, HF_PATHS, f"https://huggingface.co/apple/OpenELM-1_1B/resolve/{HF_REVISION}/"),
    ):
        for relative in paths:
            path = root / "sources" / family / relative
            records[str(path.relative_to(root))] = {
                **download_artifact(ArtifactSpec(base + relative), path), "revision": revision,
            }
    manifest = {"schema_version": 1, "artifacts": records}
    write_json(root / "source_manifest.json", manifest)
    return manifest


def prepare_checkpoint(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "checkpoint" / CHECKPOINT_FILENAME
    return {"path": str(path.relative_to(root)), **download_artifact(ArtifactSpec(CHECKPOINT_URL, CHECKPOINT_SIZE, CHECKPOINT_SHA256), path)}


def prepare_tokenizer(root: str | Path, local_model: str | Path | None = None) -> dict[str, Any]:
    """Acquire only the official, pinned Llama tokenizer or byte-identical local copy."""
    root = Path(root)
    if local_model is None:
        from huggingface_hub import hf_hub_download

        try:
            local_model = hf_hub_download(
                repo_id=TOKENIZER_REPO, revision=TOKENIZER_REVISION, filename="tokenizer.model",
                cache_dir=str(root / "hf-cache"),
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise RuntimeError(
                "Official Llama tokenizer acquisition failed "
                f"({type(exc).__name__}, HTTP {status}); configure authorized HF access "
                "or provide a legitimately obtained byte-identical --tokenizer-model."
            ) from None
    local_model = Path(local_model)
    digest = sha256_file(local_model)
    if digest != TOKENIZER_SHA256:
        raise ValueError("Tokenizer bytes differ from the pinned official Llama v1/v2 model; refusing substitution")
    path = root / "tokenizer" / "tokenizer.model"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256_file(path) != digest:
        raise ValueError("Existing tokenizer differs from the pinned tokenizer")
    if not path.exists():
        temporary = path.with_name(path.name + ".tmp")
        shutil.copyfile(local_model, temporary)
        temporary.replace(path)
    tokenizer = OpenELMTokenizer(path)
    manifest = {
        "schema_version": 1, "repo_id": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
        "path": str(path.relative_to(root)), "sha256": digest, "size_bytes": path.stat().st_size,
        "sentencepiece_vocab_size": tokenizer.processor.vocab_size(),
        "model_vocab_size": 32128, "unk_id": 0, "bos_id": 1, "eos_id": 2,
        "sentencepiece_pad_id": -1, "pad_id": 32000,
        "unused_output_ids": [32001, 32127],
        "normalization": "ftfy.fix_text(text, normalization='NFC')",
        "append_bos": True, "append_eos": True,
        "tokenizer_contract_source": f"corenet@{CORENET_REVISION}/corenet/data/text_tokenizer/sentencepiece_tokenizer.py",
    }
    write_json(root / "tokenizer_manifest.json", manifest)
    return manifest


class OpenELMTokenizer:
    """The pinned native pretraining text contract, without HF added-token defaults."""

    pad_id = 32000
    bos_id = 1
    eos_id = 2

    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        if sha256_file(model_path) != TOKENIZER_SHA256:
            raise ValueError("Tokenizer model SHA256 does not match the pinned official file")
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        actual = (self.processor.vocab_size(), self.processor.unk_id(), self.processor.bos_id(), self.processor.eos_id(), self.processor.pad_id())
        if actual != (32000, 0, 1, 2, -1):
            raise ValueError(f"Unexpected tokenizer vocabulary/special-token contract: {actual}")

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        try:
            import ftfy
        except ModuleNotFoundError:
            raise RuntimeError("Native OpenELM tokenization requires ftfy; install the pinned Docker dependencies") from None
        ids = self.processor.Encode(ftfy.fix_text(text, normalization="NFC"))
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: list[int]) -> str:
        if any(token < 0 or token >= 32000 for token in ids):
            raise ValueError("Cannot decode padding or hardware-padding output IDs as SentencePiece text")
        return self.processor.Decode(ids)


def load_native_state_dict(
    path: str | Path, *, expected_vocab_size: int = 32128, expected_model_dim: int = 2048,
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Load a model-only native state dict on CPU; never crop, remap, or reinitialize."""
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Native model-only checkpoint must be a nonempty state dict")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Expected a tensor-only model state dict, not a wrapped training/optimizer checkpoint")
    embedding = state.get("token_embeddings.weight")
    if embedding is None or tuple(embedding.shape) != (expected_vocab_size, expected_model_dim):
        raise ValueError("Native embedding dimensions differ; no vocabulary cropping or prefix remapping is allowed")
    if expected_shapes is not None:
        if set(state) != set(expected_shapes):
            raise ValueError(f"Native state keys differ: missing={sorted(set(expected_shapes) - set(state))}, unexpected={sorted(set(state) - set(expected_shapes))}")
        for key, shape in expected_shapes.items():
            if tuple(state[key].shape) != tuple(shape):
                raise ValueError(f"Native tensor shape differs: {key}: {tuple(state[key].shape)} != {tuple(shape)}")
    for key, value in state.items():
        if value.device.type != "cpu" or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Native checkpoint contains a nonfinite or non-floating CPU tensor: {key}")
    return dict(state)


def state_dict_manifest(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()} for key, value in state.items()},
    }


def refresh_manifest(root: str | Path) -> dict[str, Any]:
    """Consolidate completed component records; relative paths survive host/container moves."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = {
            "schema_version": 1, "model": "OpenELM-1_1B", "checkpoint_step": 300000,
            "checkpoint_iteration_index": 299999,
            "corenet_revision": CORENET_REVISION, "hf_revision": HF_REVISION,
            "native_vocab_size": 32128, "native_pad_id": 32000,
        }
        for key, name in (("sources", "source_manifest.json"), ("tokenizer", "tokenizer_manifest.json"), ("checkpoint_inspection", "checkpoint_inspection.json")):
            if (root / name).exists():
                result[key] = json.loads((root / name).read_text())
        checkpoint = root / "checkpoint" / CHECKPOINT_FILENAME
        if checkpoint.exists() and _record_path(checkpoint).exists():
            result["checkpoint"] = {
                "path": str(checkpoint.relative_to(root)),
                **json.loads(_record_path(checkpoint).read_text()),
            }
        result["preparation_complete"] = all(key in result for key in ("sources", "tokenizer", "checkpoint"))
        write_json(root / "manifest.json", result)
        return result


def validate_prepared_manifest(root: str | Path) -> dict[str, Any]:
    """Verify all completed Stage A files before using an existing preparation."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("corenet_revision") != CORENET_REVISION
        or manifest.get("hf_revision") != HF_REVISION
        or manifest.get("model") != "OpenELM-1_1B"
        or manifest.get("checkpoint_step") != 300000
        or manifest.get("native_vocab_size") != 32128
        or manifest.get("native_pad_id") != 32000
    ):
        raise ValueError("Prepared model/source contract differs from the Stage A pins")
    if not all(key in manifest for key in ("sources", "tokenizer", "checkpoint")):
        raise ValueError("Artifact preparation is incomplete")
    checkpoint = root / "checkpoint" / CHECKPOINT_FILENAME
    record = validate_artifact(checkpoint, ArtifactSpec(CHECKPOINT_URL, CHECKPOINT_SIZE, CHECKPOINT_SHA256))
    if manifest["checkpoint"] != {"path": str(checkpoint.relative_to(root)), **record}:
        raise ValueError("Consolidated checkpoint manifest differs from its integrity record")
    expected_sources = {
        **{f"sources/corenet/{path}": (f"https://raw.githubusercontent.com/apple/corenet/{CORENET_REVISION}/{path}", CORENET_REVISION) for path in CORENET_PATHS},
        **{f"sources/huggingface/{path}": (f"https://huggingface.co/apple/OpenELM-1_1B/resolve/{HF_REVISION}/{path}", HF_REVISION) for path in HF_PATHS},
    }
    source_records = manifest["sources"]["artifacts"]
    if set(source_records) != set(expected_sources):
        raise ValueError("Prepared source snapshot set differs from the Stage A contract")
    for path, (url, revision) in expected_sources.items():
        record = validate_artifact(root / path, ArtifactSpec(url))
        if source_records[path] != {**record, "revision": revision}:
            raise ValueError(f"Consolidated source manifest differs: {path}")
    tokenizer_path = root / "tokenizer" / "tokenizer.model"
    tokenizer = manifest["tokenizer"]
    if (
        sha256_file(tokenizer_path) != TOKENIZER_SHA256
        or tokenizer.get("sha256") != TOKENIZER_SHA256
        or tokenizer.get("revision") != TOKENIZER_REVISION
        or tokenizer.get("repo_id") != TOKENIZER_REPO
        or tokenizer.get("path") != "tokenizer/tokenizer.model"
        or tokenizer.get("model_vocab_size") != 32128
        or tokenizer.get("pad_id") != 32000
        or tokenizer.get("sentencepiece_vocab_size") != 32000
        or tokenizer.get("unk_id") != 0
        or tokenizer.get("bos_id") != 1
        or tokenizer.get("eos_id") != 2
        or tokenizer.get("append_bos") is not True
        or tokenizer.get("append_eos") is not True
    ):
        raise ValueError("Prepared tokenizer differs from the pinned contract")
    return manifest
