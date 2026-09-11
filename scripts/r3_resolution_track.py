#!/usr/bin/env python3
"""Publish retained R3 dense-gradient diagnostics to online W&B without rerunning them.

Run in the CPU container with CDRM_DOCKER_GPUS=none. The source reports and tensor
packets are read-only; publication metadata goes into a new sibling directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from experiment_tracking import OnlineTracker, add_wandb_arguments


METRICS = ("relative_l2", "max_error_over_reference_rms", "max_absolute_error")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def diagnostic_rows(report):
    """Preserve each error's named reference; never reinterpret a pass threshold."""
    tables = {}

    def add(table, label, metrics, **metadata):
        row = {"label": label, **metadata,
               **{name: metrics[name] for name in METRICS},
               "exact": metrics["exact"], "finite": metrics["finite"]}
        tables.setdefault(table, []).append(row)

    synthetic = report.get("synthetic", {})
    for arm, values in synthetic.get("arms", {}).items():
        add("synthetic_weight_gradients_vs_fp64", arm, values["weight_gradient_vs_fp64"],
            cache_enabled=values["cache_enabled"], mode=values["mode"])
    for arm, values in synthetic.get("explicit_reductions_vs_fp64", {}).items():
        add("synthetic_reductions_vs_fp64", arm, values)

    actual = report.get("actual", {})
    for arm, values in actual.get("arms", {}).items():
        for layer, kinds in values["dense"].items():
            for kind, record in kinds.items():
                add("isolated_dense_weight_gradients_vs_fixed_operand_fp64",
                    f"{arm}/{layer}/{kind}", record["actual_weight_gradient_vs_fixed_operand_fp64"],
                    arm=arm, layer=layer, kind=kind, cache_enabled=values["cache_enabled"])
    for pair, comparisons in actual.get("comparisons", {}).items():
        for name, values in comparisons.get("parameters", {}).items():
            add("isolated_parameter_comparisons", f"{pair}/{name}", values, pair=pair, parameter=name)
        for name in ("output", "input_gradient"):
            add("isolated_boundary_comparisons", f"{pair}/{name}", comparisons[name], pair=pair, tensor=name)
    return tables


def publish_rows(tracker, prefix, tables):
    import wandb
    for category, rows in tables.items():
        columns = list(rows[0])
        table = wandb.Table(columns=columns, data=[[row[name] for name in columns] for row in rows])
        payload = {f"{prefix}/{category}/table": table}
        if len(rows) <= 40:
            payload[f"{prefix}/{category}/relative_error"] = wandb.plot.bar(
                table, "label", "relative_l2", title=f"{prefix}: {category} (relative L2)")
            payload[f"{prefix}/{category}/maximum_error"] = wandb.plot.bar(
                table, "label", "max_error_over_reference_rms",
                title=f"{prefix}: {category} (maximum error / reference RMS)")
        tracker.log(payload)
        tracker.summary({f"{prefix}/{category}/{row['label']}/{metric}": row[metric]
                         for row in rows for metric in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True,
                        help="Retained dense-probe report.json; repeat for multiple reports")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if not args.wandb_project:
        parser.error("Publication requires --wandb-project")
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Enter the project CPU container before publishing retained metrics")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new publication output directory")
    if len({path.resolve() for path in args.report}) != len(args.report):
        raise ValueError("Each source report may be included only once")
    sources, records = [], []
    for path in args.report:
        path = path.resolve()
        if path.parent in output.parents:
            raise ValueError("Publication output must be outside each retained source run directory")
        report = json.loads(path.read_text())
        if report.get("schema") != "r3-dense-gradient-attribution-v1" or report.get("status") != "diagnostics_complete":
            raise ValueError(f"Require a completed dense-gradient diagnostic report: {path}")
        tensor_path = path.parent / "tensors.pt"
        if digest(tensor_path) != report["tensors_sha256"]:
            raise ValueError(f"Retained tensor packet digest differs: {tensor_path}")
        sources.append({"path": str(path), "sha256": digest(path),
                        "tensor_sha256": report["tensors_sha256"],
                        "original_recorded_at_utc": report.get("provenance", {}).get("recorded_at_utc"),
                        "settings": report["settings"], "scope": report["scope"],
                        "fixture": report.get("fixture"), "synthetic_shape": report.get("synthetic", {}).get("shape")})
        records.append(report)
    output.mkdir(parents=True)
    publication = {"schema": "r3-retained-diagnostics-wandb-publication-v1", "status": "publishing",
                   "scope": "Post-hoc visualization of retained NUM evidence; no new numerical run or acceptance decision",
                   "source_reports": sources,
                   "source_sha256": {"scripts/r3_resolution_track.py": digest(Path(__file__)),
                                      "scripts/experiment_tracking.py": digest(Path(__file__).with_name("experiment_tracking.py"))}}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                            group=args.wandb_group, name=args.wandb_run_name, output_dir=output)
    publication["wandb"] = tracker.record
    started = time.monotonic()
    try:
        tracker.start({"evidence_class": "NUM-republication", "source_reports": sources,
                       "scope": publication["scope"]})
        print(json.dumps({"wandb_run_url": tracker.record["run_url"]}), flush=True)
        for index, (source, report) in enumerate(zip(sources, records)):
            prefix = f"{index + 1:02d}-{Path(source['path']).parent.name}"
            publish_rows(tracker, prefix, diagnostic_rows(report))
            arms = report.get("actual", {}).get("arms", {})
            credit = {name: values["write_gradient_norms"] for name, values in arms.items()
                      if values["write_gradient_norms"] and all(v is not None for v in values["write_gradient_norms"])}
            if credit:
                import wandb
                tracker.log({f"{prefix}/persistent_write_gradient_norms": wandb.plot.line_series(
                    xs=[list(range(len(values))) for values in credit.values()], ys=list(credit.values()),
                    keys=list(credit), title=f"{prefix}: gradient norm per permanent write", xname="position")})
        if any(digest(Path(source["path"])) != source["sha256"] for source in sources):
            raise RuntimeError("A retained report changed during publication")
        publication["status"] = "published"
        tracker.summary({"publication_status": "published", "retained_reports": len(sources)})
    except BaseException as error:
        publication.update(status="publication_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            tracker.finish(succeeded=publication["status"] == "published")
        except Exception as error:
            publication.update(status="publication_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            publication["elapsed_seconds"] = time.monotonic() - started
            (output / "report.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
