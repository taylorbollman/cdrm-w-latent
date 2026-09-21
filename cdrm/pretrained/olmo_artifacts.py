"""Immutable original OLMo-1B step-60k artifacts; never execute Hub model code.

The native model is authoritative.  The nearby official Transformers conversion
is provenance only until actual tensor/function correspondence has been checked.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from cdrm.pretrained.artifacts import sha256_file, write_json

REPO_ID = "allenai/OLMo-1B"
REVISION = "81b71efbce6f4dada57c94860301af4298bcd351"
BRANCH_LABEL = "step60000-tokens252B"
SOURCE_REVISION = "b3741bc21f1dd504838b7dbd9878ee077ded63bd"
CHECKPOINT_FILENAME = "model.safetensors"
CHECKPOINT_SIZE = 4_707_065_440
CHECKPOINT_SHA256 = "ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c"
PARAMETER_COUNT = 1_176_764_416
FILE_SPECS = {
    CHECKPOINT_FILENAME: (CHECKPOINT_SIZE, CHECKPOINT_SHA256),
    "config.json": (1321, "89767baeb3784222457a8e18e7821640bd45021cb7d4b3b1c0d7de5a7ab59c9d"),
    "tokenizer.json": (2115113, "9ad33b4b39a9f83973c3f8c42a01948dd5b877a28ac9a5356956c4ff4ed0b714"),
    "tokenizer_config.json": (5492, "2076c0f5842818df6372ac9699df1056cee7a44e01650677599e0d1d57de5755"),
    "special_tokens_map.json": (65, "b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2"),
}
MANIFEST_FILENAME = "artifact-manifest.json"
MANIFEST_SCHEMA = "olmo1b-native-step60000-artifacts-v1"
TOKENIZER_FIXTURES = (
    "Hello, world!",
    "cafe\u0301 and café",
    "def add(x, y):\n    return x + y\n",
    "<|endoftext|>",
)


def native_tensor_shapes() -> dict[str, tuple[int, ...]]:
    result = {"transformer.wte.weight": (50304, 2048)}
    for layer in range(16):
        for name, shape in (
            ("att_proj", (6144, 2048)), ("attn_out", (2048, 2048)),
            ("ff_proj", (16384, 2048)), ("ff_out", (2048, 8192)),
        ):
            result[f"transformer.blocks.{layer}.{name}.weight"] = shape
    return result


def _file_records(root: Path) -> dict[str, Any]:
    result = {}
    for name, (expected_size, expected_sha) in FILE_SPECS.items():
        path = root / "native" / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular pinned OLMo artifact: {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"Pinned artifact byte count differs: {name}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(f"Pinned artifact SHA256 differs: {name}")
        result[f"native/{name}"] = {
            "size_bytes": expected_size, "sha256": observed_sha,
            "url": f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}",
            "hash_basis": "Hub LFS SHA256" if name == CHECKPOINT_FILENAME else "pinned metadata bytes",
        }
    return result


class OLMoNativeTokenizer:
    """Pinned original tokenizers graph, including NFC; no added BOS/EOS by default.

    Native OLMo preprocessing appends document EOS outside the tokenizer.  A caller
    may request one explicitly.  This wrapper never adds a nonexistent BOS,
    applies ftfy, pads, truncates, or alters native tokenizer special-token rules.
    """
    eos_token_id = 50279
    pad_token_id = 1
    vocab_size = 50280

    def __init__(self, tokenizer_json: str | Path):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        if self._tokenizer.get_vocab_size(with_added_tokens=True) != self.vocab_size:
            raise ValueError("Unexpected native OLMo tokenizer vocabulary")
        if self._tokenizer.token_to_id("<|endoftext|>") != self.eos_token_id:
            raise ValueError("Unexpected native OLMo EOS ID")
        if self._tokenizer.token_to_id("<|padding|>") != self.pad_token_id:
            raise ValueError("Unexpected native OLMo padding ID")

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        result = self._tokenizer.encode(text, add_special_tokens=False).ids
        return result + [self.eos_token_id] if add_eos else result

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def fixture_records(self) -> list[dict[str, Any]]:
        return [{"text": text, "token_ids": self.encode(text),
                 "decoded": self.decode(self.encode(text)), "adds_eos": False}
                for text in TOKENIZER_FIXTURES]


def prepare_artifacts(root: str | Path) -> dict[str, Any]:
    """Download missing pinned files via hf CLI, hash complete bytes, freeze manifest.

    Existing bytes with a wrong hash are rejected, not silently overwritten.
    A successful rerun validates the existing immutable manifest rather than
    changing its provenance.  Full model loading is a separate CPU operation.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        return validate_prepared_manifest(root)
    native = root / "native"
    native.mkdir(exist_ok=True)
    missing = []
    for name, (expected_size, expected_sha) in FILE_SPECS.items():
        path = native / name
        if not path.exists():
            missing.append(name)
        elif path.is_symlink() or path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise ValueError(f"Existing artifact differs from immutable pin: {name}")
    if missing:
        subprocess.run([
            "hf", "download", REPO_ID, *missing, "--revision", REVISION,
            "--local-dir", str(native.resolve()),
        ], check=True)
    records = _file_records(root)
    tokenizer = OLMoNativeTokenizer(native / "tokenizer.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID, "revision": REVISION, "branch_label": BRANCH_LABEL,
        "source_revision": SOURCE_REVISION,
        "checkpoint_status": "complete bytes hash verified; model execution not implied",
        "token_count": {"native_branch_label": "252B", "official_hf_conversion_branch_label": "251B",
                        "optimizer_step": 60000, "exact_training_counter_verified": False},
        "artifacts": records,
        "checkpoint": {"path": "native/" + CHECKPOINT_FILENAME, **records["native/" + CHECKPOINT_FILENAME]},
        "raw_native_config": json.loads((native / "config.json").read_text()),
        "tokenizer": {"vocab_size": 50280, "embedding_rows": 50304,
                      "eos_token_id": 50279, "pad_token_id": 1, "automatic_bos": False,
                      "automatic_eos": False, "fixtures": tokenizer.fixture_records()},
        "load_contract": {"state_key_transform": "strip exactly the outer model. prefix",
                          "expected_tensors": 65, "expected_parameters": PARAMETER_COUNT,
                          "dtype": "float32", "tied_embedding_readout": True,
                          "embedding_padding_idx": None, "qk_norm": False},
    }
    write_json(manifest_path, manifest)
    return manifest


def validate_prepared_manifest(root: str | Path) -> dict[str, Any]:
    """Offline rehash of all full artifacts against immutable pins and manifest."""
    root = Path(root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    expected = {"schema": MANIFEST_SCHEMA, "repo_id": REPO_ID, "revision": REVISION,
                "branch_label": BRANCH_LABEL, "source_revision": SOURCE_REVISION}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared OLMo manifest provenance differs from pinned source")
    if manifest.get("artifacts") != _file_records(root):
        raise ValueError("Prepared OLMo manifest artifact records differ from verified bytes")
    checkpoint = {"path": "native/" + CHECKPOINT_FILENAME, **manifest["artifacts"]["native/" + CHECKPOINT_FILENAME]}
    if manifest.get("checkpoint") != checkpoint:
        raise ValueError("Prepared OLMo checkpoint identity differs from verified bytes")
    if manifest.get("raw_native_config") != json.loads((root / "native/config.json").read_text()):
        raise ValueError("Prepared OLMo raw config differs from pinned bytes")
    tokenizer = OLMoNativeTokenizer(root / "native/tokenizer.json")
    expected_tokenizer = {"vocab_size": 50280, "embedding_rows": 50304,
                          "eos_token_id": 50279, "pad_token_id": 1, "automatic_bos": False,
                          "automatic_eos": False, "fixtures": tokenizer.fixture_records()}
    if manifest.get("tokenizer") != expected_tokenizer:
        raise ValueError("Prepared tokenizer manifest differs from pinned native behavior")
    expected_contract = {"state_key_transform": "strip exactly the outer model. prefix",
                         "expected_tensors": 65, "expected_parameters": PARAMETER_COUNT,
                         "dtype": "float32", "tied_embedding_readout": True,
                         "embedding_padding_idx": None, "qk_norm": False}
    if manifest.get("load_contract") != expected_contract:
        raise ValueError("Prepared OLMo load contract differs from immutable native contract")
    return manifest


def validate_native_tensors(raw_state: Mapping[str, Any], *,
                            expected_shapes: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """Validate the exact native tensor layout before exposing local model keys."""
    import torch
    expected = dict(expected_shapes) if expected_shapes is not None else native_tensor_shapes()
    if not raw_state or any(not key.startswith("model.") for key in raw_state):
        raise ValueError("Native checkpoint keys must have exactly the model. wrapper prefix")
    state = {key.removeprefix("model."): value for key, value in raw_state.items()}
    if set(state) != set(expected):
        raise ValueError(f"Native tensor key mismatch: missing={sorted(set(expected)-set(state))}; "
                         f"unexpected={sorted(set(state)-set(expected))}")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor native state value: {key}")
        if tuple(tensor.shape) != tuple(expected[key]):
            raise ValueError(f"Native tensor shape mismatch: {key}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Native tensor must remain stored float32: {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite native tensor: {key}")
    return state


def load_native_state_dict(root: str | Path, *,
                           expected_shapes: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """Strictly load complete hashed native safetensors on CPU; no remote code."""
    from safetensors.torch import load_file
    validate_prepared_manifest(root)
    raw = load_file(str(Path(root) / "native" / CHECKPOINT_FILENAME), device="cpu")
    return validate_native_tensors(raw, expected_shapes=expected_shapes)


def load_native_tokenizer(root: str | Path) -> OLMoNativeTokenizer:
    # Verify only the tokenizer's pinned files here; no need to hash 4.7GB to encode.
    root = Path(root)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = root / "native" / name
        expected_size, expected_sha = FILE_SPECS[name]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise ValueError(f"Native tokenizer file differs from pinned source: {name}")
    return OLMoNativeTokenizer(root / "native/tokenizer.json")
