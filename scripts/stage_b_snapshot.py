#!/usr/bin/env python3
"""Archive explicit project source paths, never credentials or runtime caches."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / (args.name + ".tar.gz")
    metadata = args.output_dir / (args.name + ".json")
    if archive.exists() or metadata.exists():
        raise FileExistsError("Use a new snapshot name; existing source records are immutable")
    folders = ["cdrm", "scripts", "configs/stage_b", "tests", "vendors/zoology",
               "vendors/mad-lab", "recurrent-transformer/olmo"]
    files = {root / name for name in ["AGENTS.md", "README.md", "docker/Dockerfile", "docker/requirements-docker.txt",
             "recurrent-transformer/pyproject.toml", "recurrent-transformer/LICENSE",
             "docs/inputs/README.md", "docs/inputs/coding_agent_review_amendments.md",
             "docs/inputs/recurrent_transformer_initial_run_protocol.md",
             "docs/reports/stage-b/modal_value_baseline.py",
             "docs/inputs/recurrent_transformer_coding_agent_brief.md", "docs/semantic-decisions.md",
             "docs/stage-b-plan.md", "docs/stage-b-state-task.md", "docs/stage-b-usage.md"] if (root / name).is_file()}
    for folder in folders:
        files.update(path for path in (root / folder).rglob("*") if path.is_file()
                     and not any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts)
                     and not path.name.startswith(".env") and path.suffix not in {".pyc", ".pyo"})
    ordered = sorted(files)
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in ordered}
    with tarfile.open(archive, "w:gz") as handle:
        for path in ordered:
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    record = {"recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "source_sha256": hashes, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "archive_bytes": archive.stat().st_size, "archive": str(archive),
              "parent_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "fork_revision": subprocess.check_output(["git", "-C", str(root / "recurrent-transformer"), "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_changes_included": True}
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "files": len(ordered), "bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
