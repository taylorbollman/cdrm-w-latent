#!/usr/bin/env python3
"""Interpret an extended fresh mixed pilot without changing its historical gate.

Saved evidence only: the original reporter and its retained output stay intact.
This report supersedes the original report's gate-based interpretation of the
extended budget; figures and checkpoint observations remain the same.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts import rt_nextlat_a5_fuzzy_report as legacy


SCHEMA = "rt-nextlat-a5-fuzzy-extension-report-v1"


def extension_observations(evaluations, endpoint):
    legacy.require(type(endpoint) is int and endpoint > 10000,
                   "Extension reporting requires an actual endpoint beyond 10k")
    full = []
    for metric in evaluations:
        if metric.get("task") == "a5" and metric.get("role") == "ood_dev":
            legacy.checked_a5_metric(metric)
            legacy.require(metric["update"] <= endpoint, "Evaluation exceeds endpoint")
            if metric["rows"] == 102400 and metric["update"] > 0:
                full.append(metric)
    full.sort(key=lambda row: row["update"])
    legacy.require(len({row["update"] for row in full}) == len(full),
                   "Duplicate full L36 evaluation")
    legacy.require(any(row["update"] == endpoint for row in full),
                   "Actual endpoint requires a full L36 evaluation")

    def observation(rows):
        first = next((row for row in rows if row["whole_word_correct"] > 0), None)
        return {"observed_positive": first is not None,
                "first_positive_update": None if first is None else first["update"],
                "first_positive_whole_word_correct": None if first is None else first["whole_word_correct"],
                "first_positive_rows": None if first is None else first["rows"],
                "first_positive_accuracy": None if first is None else first["whole_word_exact_match"],
                "full_evaluation_updates": [row["update"] for row in rows]}

    final = next(row for row in full if row["update"] == endpoint)
    return {"actual_endpoint": endpoint, "full_budget": observation(full),
            "post_10k": observation([row for row in full if row["update"] > 10000]),
            "endpoint": {"update": endpoint, "whole_word_correct": final["whole_word_correct"],
                         "rows": final["rows"], "whole_word_exact_match": final["whole_word_exact_match"]},
            "interpretation": "Observed development evidence, not a new stopping gate or endpoint selection. First observation is not the exact onset of learning."}


def interpretation_markdown(summary):
    gate, observed = summary["a5_positive_gate"], summary["a5_extension_observations"]
    text = ["## Historical 10k gate and extended-budget observations", "",
            "The original continuation gate applies only through update 10,000 and is preserved unchanged."]
    if gate["passed"]:
        text.append(f"That historical gate passed at update {gate['first_positive_update']:,}.")
    else:
        text.append("That historical gate did not pass: no positive full L36 observation was recorded by 10k. "
                    "This does not determine what happened during the extension.")
    first = observed["full_budget"]
    if first["observed_positive"]:
        text.append(f"Across the actual extended budget, the first positive full L36 observation was at "
                    f"**update {first['first_positive_update']:,}**: "
                    f"{first['first_positive_whole_word_correct']:,}/{first['first_positive_rows']:,} words "
                    f"({100*first['first_positive_accuracy']:.6f}%).")
    else:
        text.append(f"No full L36 evaluation recorded a positive whole-word count through the actual "
                    f"endpoint of {observed['actual_endpoint']:,} updates.")
    final = observed["endpoint"]
    text.append(f"At the actual endpoint, L36 whole-word accuracy is "
                f"**{100*final['whole_word_exact_match']:.6f}%** "
                f"({final['whole_word_correct']:,}/{final['rows']:,}). "
                "Earlier positive observations and endpoint performance are reported separately.")
    text.append("A positive finite-sample development count establishes observed correct length-36 words; "
                "it does not establish reliable or general state tracking. The first observed full positive is not the exact onset of learning. "
                "Final confirmation remains unused. "
                "The unchanged same-checkpoint endpoint table reports Fuzzy performance alongside A5.")
    return "\n\n".join(text) + "\n\n"


def build_report(args):
    source = args.legacy_report.resolve()
    evidence = legacy.read_json(source / "evidence.json")
    published = legacy.read_json(source / "report.json")
    legacy.require(evidence.get("schema") == legacy.SCHEMA and evidence.get("status") == "complete"
                   and evidence.get("mode") == "mixed", "Require a completed legacy mixed report")
    legacy.require(all(published.get(key) == value for key, value in evidence.items()),
                   "Published legacy report differs from its immutable evidence")
    legacy.require(evidence["source_sha256"] == legacy.sha(Path(legacy.__file__)),
                   "Legacy reporter source differs")
    # The completed legacy report already performed strict lineage validation.
    # Preserve its evidence and hashes; this supplement changes interpretation.
    legacy.require(legacy.a5_gate(evidence["evaluations"]) == evidence["a5_positive_gate"],
                   "Historical 10k gate differs")
    final = evidence["final"]["a5/ood_dev"]
    legacy.require(final["update"] == evidence["endpoint"]
                   and final["rows"] == 102400
                   and final["checkpoint"]["sha256"] == evidence["checkpoint"]["sha256"],
                   "Endpoint metric differs from the legacy endpoint checkpoint")
    summary = dict(evidence)
    summary.update(schema=SCHEMA,
                   a5_extension_observations=extension_observations(evidence["evaluations"], evidence["endpoint"]),
                   historical_a5_10k_gate=evidence["a5_positive_gate"],
                   authority="This report supersedes the legacy report's extended-budget interpretation; historical metrics, gate and figures are unchanged.",
                   legacy_report={"directory": str(source), "report_sha256": legacy.sha(source / "report.json"),
                                  "evidence_sha256": legacy.sha(source / "evidence.json"),
                                  "source_sha256": evidence["source_sha256"],
                                  "report_wandb": published.get("report_wandb")},
                   source_code=str(Path(__file__).resolve()), source_sha256=legacy.sha(Path(__file__)))
    summary["qualification"] = ("Single initialization and reused development data; no final confirmation. "
        "The historical 10k continuation gate is preserved separately from observed performance through the actual endpoint. "
        "Only common retained checkpoints match control training exposure; controls are not extended or extrapolated.")
    markdown = (source / "report.md").read_text()
    start, stop = "## A5 continuation gate", "## Controls and qualifications"
    legacy.require(markdown.count(start) == markdown.count(stop) == 1,
                   "Unexpected legacy narrative; refusing an ambiguous replacement")
    before, remaining = markdown.split(start, 1)
    _, after = remaining.split(stop, 1)
    markdown = ("**Authoritative extended-budget interpretation.** The historical 10k gate is unchanged; "
                "this report supersedes the earlier gate-based conclusion about the extended budget.\n\n"
                + before + interpretation_markdown(summary) + stop + after)
    markdown = markdown.replace(evidence["qualification"], summary["qualification"])
    args.output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for stem in summary["figures"]:
        legacy.require(Path(stem).name == stem, "Invalid figure filename")
        for suffix in ("png", "pdf"):
            name = f"{stem}.{suffix}"
            shutil.copy2(source / name, args.output / name)
            copied[name] = legacy.sha(args.output / name)
    summary["figure_sha256"] = copied
    (args.output / "report.md").write_text(markdown)
    shutil.copy2(source / "evidence.json", args.output / "legacy-evidence.json")
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    summary = build_report(args)
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", output_dir=args.output,
                                name=f"d128-mixed-extension-authoritative-{summary['endpoint']}")
        try:
            tracker.start({"scope": summary["authority"], "endpoint": summary["endpoint"]})
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png"))
                         for name in summary["figures"]})
            tracker.summary({"historical_a5_10k_gate": summary["historical_a5_10k_gate"],
                             "a5_extension_observations": summary["a5_extension_observations"],
                             "endpoint_metrics": summary["final"], "endpoint": summary["endpoint"]})
            artifact = wandb.Artifact(f"d128-mixed-extension-{tracker.record['run_id']}", type="development-report")
            for path in sorted(args.output.iterdir()):
                if path.is_file() and path.suffix in (".json", ".md", ".png", ".pdf"):
                    artifact.add_file(str(path), name=path.name)
            tracker._call("artifact logging", lambda: tracker._run.log_artifact(artifact))
            tracker.finish(succeeded=True)
            summary["report_wandb"] = tracker.record
        except BaseException:
            tracker.finish(succeeded=False)
            raise
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": summary["endpoint"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
