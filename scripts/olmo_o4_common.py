"""O4 bounded learning-pilot provenance, tracking and immutable retention."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_lm_profile import SOURCE_FILES as O3_SOURCES

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"ordinary": (False, False), "ordinary-nextlat": (False, True),
        "rt": (True, False), "rt-nextlat": (True, True)}
SOURCE_FILES = tuple(sorted(set(O3_SOURCES) | {
    "cdrm/pretrained/lm_data.py", "cdrm/pretrained/lm_evaluation.py",
    "cdrm/pretrained/lm_schedule.py", "scripts/olmo_o4_common.py",
    "scripts/olmo_o4_train.py", "scripts/olmo_o4_preflight.py",
}))


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def preserve_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


class PilotTracker(OnlineTracker):
    """Resume the same online curve while checkpoint state owns training RNG."""
    def start(self, config, *, run_id=None):
        if run_id is None:
            return super().start(config)
        def initialize():
            import wandb
            if wandb.run is not None:
                raise RuntimeError("An existing W&B run would mix experiment lineages")
            self._run = wandb.init(project=self.record["project"], entity=self.record["entity"],
                group=self.record["group"], name=self.record["name"], id=run_id, resume="must",
                mode="online", config=config, dir=str(self.output_dir),
                settings=wandb.Settings(init_timeout=60, save_code=False, disable_git=True, x_disable_stats=True))
            if self._run is None or self._run.settings.mode != "online" or not self._run.url:
                raise RuntimeError("W&B resume did not open an online run")
            self._run.define_metric("update")
            self._run.define_metric("train/*", step_metric="update")
            self._run.define_metric("dev/*", step_metric="update")
            self.record.update(status="running", run_id=self._run.id,
                               run_url=self._run.url, project_url=self._run.get_project_url())
        self._call("resume", initialize)


def retain_file(path: Path, uri: str, *, expected_sha256=None):
    """Upload to a fresh object generation and verify bytes; never overwrite."""
    import base64
    from google.cloud import storage
    if not uri.startswith("gs://") or not path.is_file() or path.is_symlink():
        raise ValueError("Retention requires a regular local file and gs:// object URI")
    bucket_name, key = uri[5:].split("/", 1)
    if not key or ".." in key.split("/"):
        raise ValueError("Invalid retention object key")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024**2):
            sha.update(chunk); md5.update(chunk)
    digest = sha.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Checkpoint changed before retention")
    size, md5_b64 = path.stat().st_size, base64.b64encode(md5.digest()).decode()
    blob = storage.Client().bucket(bucket_name).blob(key)
    if not blob.exists():
        blob.metadata = {"sha256": digest, "artifact_schema": "olmo-o4-pilot-v1"}
        blob.upload_from_filename(str(path), if_generation_match=0, timeout=1200, checksum="md5")
    blob.reload()
    if blob.size != size or blob.md5_hash != md5_b64 or (blob.metadata or {}).get("sha256") != digest:
        raise ValueError("Existing remote object differs from selected local bytes")
    return {"uri": uri, "generation": str(blob.generation), "sha256": digest,
            "size_bytes": size, "md5_base64": md5_b64,
            "verification": "GCS generation, size, server MD5 and SHA256 metadata verified"}


def write_event(directory, record):
    with (Path(directory) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
