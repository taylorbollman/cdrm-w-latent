#!/usr/bin/env python3
"""Saved-evidence report for the explicitly user-stopped window-2 RT + NextLat 80k checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts import rt_a5_window_budget_report as continuing
from scripts import rt_a5_nextlat_budget_report as budget
from scripts import rt_a5_nextlat_stopped_report as stopped
from scripts import rt_a5_window_report as window
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, STEPS, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT, LINEAGE = continuing.ROOT, continuing.LINEAGE
WINDOW, FULL = continuing.WINDOW, continuing.FULL
SCHEMA = "rt-a5-window-stopped-budget-comparison-v1"
REVISION_SCHEMA = "rt-a5-window-endpoint-revision-v1"
ENDPOINT = FULL_ENDPOINT = 80000
NEW_STEPS = tuple(s for s in budget.NEW_STEPS if s <= ENDPOINT)
REPORTING_SOURCES = ("scripts/rt_a5_window_stopped_report.py", *continuing.REPORTING_SOURCES)


def validate_revision(revision,protocol_path,directory):
    if revision.get("schema") != REVISION_SCHEMA:
        raise ValueError("Explicit window user-stop revision is required")
    # The receipt has its own architecture-specific schema. Reuse only the
    # identical common stop fields/file-binding guard after checking that schema.
    common = dict(revision,schema="rt-a5-nextlat-endpoint-revision-v1")
    return stopped.validate_revision(common,protocol_path,directory)


def validate_stopped_history(history,raw_report,revision):
    if raw_report.get("schema") != window.TRAIN_SCHEMA:
        raise ValueError("Stopped evidence must identify the window RT + NextLat trainer")
    # Window training uses the original history/loss/exception-handler function
    # objects. Adapt only the schema for the shared history guard; raw evidence
    # remains unchanged and is hash-bound by the separate window stop receipt.
    common = dict(raw_report,schema=budget.TRAIN_SCHEMA)
    return stopped.validate_stopped_history(history,common,revision)


def read_stopped(directory,protocol,revision,evidence):
    raw,report_file = read_input(directory / "report.json")
    packet = json.loads(raw)
    history_raw,history_file = read_input(directory / "history.jsonl")
    if report_file != evidence["report"] or history_file != evidence["history"]:
        raise ValueError("Window evidence changed after explicit revision verification")
    history = [json.loads(line) for line in history_raw.splitlines() if line.strip()]
    accepted,tail = validate_stopped_history(history,packet,revision)
    if packet["contract"] != protocol["strict_contract"] or packet["source_files"] != protocol["source_files"]:
        raise ValueError("Window stopped-run strict contract or frozen source 52 differs")
    for relative,expected in packet["source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid frozen source path/hash")
        if hash_file(directory / "source" / relative)["sha256"] != expected:
            raise ValueError(f"Window stopped-run source changed: {relative}")
    config_raw,config_file = read_input(directory / "config.json")
    config = json.loads(config_raw)
    continuing.validate_config(config,protocol,pilot=False)
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != protocol["strict_contract"]["data_manifest_sha256"]:
        raise ValueError("Window stopped-run data manifest differs")
    if [r["completed_updates"] for r in packet["checkpoints"]] != list(NEW_STEPS):
        raise ValueError("Window retained checkpoint schedule differs from accepted 80k scope")
    checkpoints = {}
    for item in packet["checkpoints"]:
        if item.get("examples_seen") != item["completed_updates"]*1024:
            raise ValueError("Window retained checkpoint word exposure differs")
        checkpoints[str(item["completed_updates"])] = stopped.verify_file(item)
    if checkpoints[str(ENDPOINT)] != evidence["accepted_checkpoint"]:
        raise ValueError("Window accepted checkpoint differs from stop receipt")
    budget.verify_evaluations(packet,10000,packet["completed_updates"],NEW_STEPS)
    return {"report":packet,"history":accepted,"raw_history":history,"tail":tail,"checkpoints":checkpoints,
            "input_files":{"report":report_file,"history":history_file,"run_config":config_file,"data_manifest":manifest}}


def make_summary(pilot_dir,run_dir,protocol_path,revision_path):
    directory = local_path(run_dir).resolve()
    raw,revision_file = read_input(local_path(revision_path))
    revision = json.loads(raw)
    evidence = validate_revision(revision,protocol_path,directory)
    protocol = json.loads(Path(evidence["protocol"]["path"]).read_text())
    continuing.validate_protocol(protocol)
    pilot = window.read_window(local_path(pilot_dir),WINDOW)
    continuing.validate_config(json.loads(Path(pilot["input_files"]["run_config"]["path"]).read_text()),protocol,pilot=True)
    budget.validate_history(pilot["history"],pilot["report"],0,10000)
    budget.verify_evaluations(pilot["report"],0,10000,STEPS)
    continuation = read_stopped(directory,protocol,revision,evidence)
    history = continuing.verify_join(pilot,continuation,protocol,ENDPOINT)
    reference = protocol["full_attention_reference"]
    full_lineage = local_path(reference["lineage"])
    full = stopped.make_summary(ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat",
        full_lineage / "train-rt-nextlat",full_lineage / "protocol.json",local_path(reference["revision"]),
        ROOT / ".runtime/rt-a5/20260911T154748Z/train-rt",ROOT / ".runtime/rt-a5/20260911T171239Z-rt100k/train-rt")
    continuing.compare_full_reference(protocol["strict_contract"],protocol["source_files"],pilot["report"]["initialization"],
                                      pilot["history"]+continuation["raw_history"],full)
    steps = [*STEPS,*NEW_STEPS]
    curves,metrics,checkpoints = {},{},{"0":pilot["checkpoints"]["0"]}
    for step in steps:
        source = pilot if step <= 10000 else continuation
        checkpoints[str(step)] = source["checkpoints"][str(step)]
        curves[str(step)],metrics[str(step)] = {},{}
        for role in ROLES:
            selected = [m for m in source["report"]["evaluations"] if (m["update"],m["role"]) == (step,role)]
            if len(selected) != 1 or selected[0]["rows"] != 102400:
                raise ValueError("Missing full window checkpoint development evaluation")
            metrics[str(step)][role] = selected[0]
            curves[str(step)][role] = metric_rows(selected[0],WINDOW)
    packet = continuation["report"]
    result = {"schema":SCHEMA,"primary_update":ENDPOINT,"reported_through_update":ENDPOINT,
        "original_planned_endpoint":100000,"planned_endpoint_reached":False,"accepted_endpoint_completed":True,
        "selection_kind":"user_selected_after_development_review","matched_checkpoint_updates":steps,
        "full_attention_max_accepted_update":FULL_ENDPOINT,"full_attention_100k_available":False,"window_100k_available":False,
        "scope":"Both window2 and full-attention 80k endpoints were selected by the user after development review; both original 100k plans remain unchanged as historical protocols. No further training in either lineage.",
        "confirmation_evaluated":False,"latent_rollout_evaluated":False,"evaluation_route":"backbone_only",
        "protocol":protocol,"protocol_input":evidence["protocol"],"endpoint_revision":revision,"revision_input":revision_file,
        "revision_evidence":evidence,"contract":protocol["strict_contract"],"source_files":protocol["source_files"],
        "initialization":pilot["report"]["initialization"],"order_chain":history[-1]["order_chain"],
        "parent_checkpoint":packet["parent_checkpoint"],
        "arms":{WINDOW:{"label":"ALiBi, layer2 window2, RT + NextLat (user-stopped80k)","checkpoint_updates":steps,
            "curves":curves,"metrics":metrics,"checkpoints":checkpoints,"training_curve":training_curve(history,True),
            "inputs":{"pilot":pilot["input_files"],"continuation":continuation["input_files"]},
            "wandb":{"pilot":pilot["report"]["wandb"],"continuation":packet["wandb"]}},
            FULL:{"label":"ALiBi, full attention, RT + NextLat (user-stopped80k)","checkpoint_updates":full["checkpoint_updates"],
                "curves":full["curves"],"metrics":full["historical_metrics"],"checkpoints":full["checkpoints"],
                "training_curve":full["training_curve"],"inputs":full["inputs"],"wandb":full["historical_wandb"]}},
        "full_attention_reference_provenance":{"schema":full["schema"],"selection_kind":full["selection_kind"],
            "endpoint_revision":full["endpoint_revision"],"revision_input":full["revision_input"],
            "revision_evidence":full["revision_evidence"],"contract":full["contract"],"source_files":full["source_files"],
            "stop":full["stop"],"pure_rt_order_inputs":full["pure_rt_reference"]["inputs"]},
        "stop":{"kind":"user_requested","raw_status":packet["status"],"raw_error_type":packet["error_type"],
            "raw_completed_updates":packet["completed_updates"],"excluded_completed_updates":len(continuation["tail"]),
            "excluded_evaluation_records":sum(m["update"]>ENDPOINT for m in packet["evaluations"]),
            "excluded_one_step_diagnostic_records":sum(m["update"]>ENDPOINT for m in packet["one_step_diagnostics"]),
            "raw_report_counter_scope":"completed_updates includes unsaved tail; order_chain and train_seconds describe retained 80k"},
        "budget":{"total_updates":ENDPOINT,"additional_updates":70000,"original_planned_updates":100000,
            "total_word_presentations":ENDPOINT*1024,"unique_training_words":800000,"nominal_training_passes":ENDPOINT*1024/800000},
        "timing":{"accepted_total_training_seconds":sum(r["seconds"] for r in history),
            "accepted_continuation_training_seconds":sum(r["seconds"] for r in continuation["history"]),
            "excluded_tail_training_seconds":sum(r["seconds"] for r in continuation["tail"]),
            "raw_continuation_training_seconds":sum(r["seconds"] for r in continuation["raw_history"]),
            "raw_continuation_job_elapsed_seconds":packet["elapsed_seconds"]},
        "training_bin_updates":100,"training_seconds":sum(r["seconds"] for r in history),
        "metric_definitions":{"E":"Every state through t correct","A":"Only state at t correct","M":"Mean token correctness through t"},
        "intervals":"Pointwise Wilson 95% across words for E/A; no seed or paired-difference uncertainty; no M interval"}
    result["checkpoint_summary"] = continuing.checkpoint_summary(result)
    return result


def metric_table(summary):
    rows = continuing.metric_table(summary)
    if any(r["update"] > ENDPOINT for r in rows):
        raise ValueError("Uncheckpointed window tail or unavailable 100k outcome leaked into comparison")
    return rows


def plot_rows(summary,arm,step):
    if step > ENDPOINT:
        raise ValueError("Both accepted experiments end at 80k; no 100k comparison is available")
    return continuing.plot_rows(summary,arm,step)


def markdown_report(summary):
    stop,timing = summary["stop"],summary["timing"]
    lines = ["# Window2 RT + NextLat: user-selected 80k checkpoint", "",
        "The user stopped the original Mitchell + ALiBi, layer-2 window2 RT + NextLat continuation at its retained "
        "**80,000-update checkpoint after reviewing development curves**. The original prospective plan was 100k. "
        "The full-attention RT + NextLat reference was also stopped at 80k after development review. These are retrospective "
        "user-selected endpoints with matched training budgets; neither model has a completed 100k outcome.", "",
        "Layer 1 retains full-prefix recurrent attention. Layer 2 reads temporary self K/V and the immediately preceding "
        "output's permanent K/V, with recurrent gradients attached. The window limits direct reads, not the full history "
        "encoded in recurrent states. Mitchell initialization, ALiBi, D512/H8/GELU-FFN2048, LayerNorm/full-width QK norm, "
        "rho1, state CE plus weight-one latent SmoothL1, and full FP32 are unchanged. Total 7,407,104 parameters. "
        "Only the target latent is detached; model accuracy uses the RT backbone without autonomous predictor rollout.", "",
        "The window model resumed its original 10k checkpoint with model, Adam, RNG and absolute word-order state. "
        "Accepted training adds 70,000 updates, totaling 81,920,000 word presentations over 800,000 unique length-12 words "
        "(102.4 nominal passes). The reporter performs no model inference or training.", "",
        "| Model | Updates | L12 whole word | E(13) | E(14) | E(16) | M(36) | E(36) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["update"] not in (10000,ENDPOINT):
            continue
        label = "Window2 RT + NextLat" if row["arm"] == WINDOW else "Full attention RT + NextLat"
        lines.append(f"| {label} | {row['update']:,} | " + " | ".join(f"{100*row[k]:.4f}%" for k in
            ("dev_whole_word_exact_match","E13","E14","E16","M36","E36")) + " |")
    lines += ["", "E(t) requires every state through t to be correct. A(t) checks only state t; M(t) averages correctness "
        "through t. OOD curves use prefixes of the same 102,400 frozen length-36 development words. Full and boundary plots "
        "use identical rows. L12 development is a separate short-word set. The 1/60 guessing reference applies only to A(t).", "",
        "![Window 10k versus 80k, full range](length-full.png)", "", "![Identical endpoint rows, boundary view](length-boundary.png)", "",
        "![Matched 80k comparison](matched-budget-boundary.png)", "", "![Exactness versus training budget](exactness-vs-updates.png)", "",
        "![Accepted training losses](training-losses.png)", "",
        "| Updates | Window E(13) | Full E(13) | Window E(14) | Full E(14) | Window E(16) | Full E(16) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_key = {(r["arm"],r["update"]):r for r in summary["checkpoint_summary"]}
    for step in summary["matched_checkpoint_updates"]:
        values = [f"{100*by_key[(arm,step)][f'E{length}']:.4f}%" for length in (13,14,16) for arm in (WINDOW,FULL)]
        lines.append(f"| {step:,} | " + " | ".join(values) + " |")
    lines += ["", f"The requested SIGINT arrived after update {stop['raw_completed_updates']:,}; "
        f"**{stop['excluded_completed_updates']:,} complete uncheckpointed window updates after 80k** and any interrupted in-flight "
        f"step are excluded from accepted results. The {stop['excluded_evaluation_records']} later routine evaluation records and "
        f"{stop['excluded_one_step_diagnostic_records']} later one-step diagnostic records remain in raw evidence but are excluded from comparisons.", "",
        "The deliberate stop is recorded as `failed`/`KeyboardInterrupt` by the unchanged trainer and as "
        "`synced_failed_experiment` by W&B. The separate window endpoint-revision receipt binds the protocol, raw history/report, "
        "accepted checkpoint and termination evidence. Only this explicit user-directed stop is accepted; arbitrary failures are not "
        "reclassified. The raw completed counter includes the unsaved tail, while raw order/time counters remain those of 80k.", "",
        f"Accepted window training-loop time: {timing['accepted_total_training_seconds']/3600:.3f}h total, with "
        f"{timing['accepted_continuation_training_seconds']/3600:.3f}h in the continuation. Excluded tail: "
        f"{timing['excluded_tail_training_seconds']:.3f}s. Raw continuation training time: "
        f"{timing['raw_continuation_training_seconds']/3600:.3f}h; raw job elapsed: "
        f"{timing['raw_continuation_job_elapsed_seconds']/3600:.3f}h including evaluation/checkpoint/logging/interruption overhead.", "",
        "Every reported checkpoint uses 102,400 words per development role. This is one seed with endpoints selected after "
        "development inspection, not a confirmed convergence result. Pointwise Wilson 95% intervals describe sampling over words "
        "for E/A, not seed variability, simultaneous coverage or paired differences. Zero E means no successes in this sample. "
        "Final confirmation and autonomous latent rollout remain **unevaluated**.", "",
        "[Exact counts](metrics.csv) · [Checkpoint summary](checkpoint-summary.csv) · [Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def plot_results(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figures = {}

    def save(fig,name):
        figures[name] = {}
        for suffix in ("png","pdf"):
            figures[name][suffix] = f"{name}.{suffix}"
            fig.savefig(output / figures[name][suffix],dpi=200)
        plt.close(fig)

    def lengths(name,series,limits,title):
        fig,axes = plt.subplots(1,3,figsize=(13.5,4.4),constrained_layout=True)
        for arm,step,label,color in series:
            rows = plot_rows(summary,arm,step)
            for axis,key in zip(axes,("E","A","M")):
                axis.plot([r["length"] for r in rows],[r[key] for r in rows],label=label,color=color)
                if key != "M":
                    axis.fill_between([r["length"] for r in rows],[r[f"{key}_low95"] for r in rows],
                                      [r[f"{key}_high95"] for r in rows],color=color,alpha=.15)
                axis.set(xlim=limits,ylim=(-.025,1.025),xlabel="Prefix length t of same length-36 words",
                         title={"E":"E: every state through t correct","A":"A: only state t correct","M":"M: mean token accuracy through t"}[key])
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.axvline(12,color="black",linestyle=":",alpha=.45)
                axis.grid(alpha=.18)
                axis.legend(fontsize=8)
        axes[1].axhline(1/60,color="gray",linestyle="--",linewidth=.8)
        fig.suptitle(title)
        save(fig,name)

    primary = [(WINDOW,10000,"Window2,10k","#C68423"),(WINDOW,80000,"Window2,80k","#3465A4")]
    for name,limits in (("length-full",(1,36)),("length-boundary",(10,18))):
        lengths(name,primary,limits,"Window2 RT + NextLat · user-selected 80k versus its10k checkpoint · 102,400 OOD words")
    lengths("matched-budget-boundary",[(FULL,80000,"Full attention,80k","#C68423"),(WINDOW,80000,"Window2,80k","#3465A4")],
            (10,18),"Matched 80k budgets · both endpoints selected after development review")
    fig,axes = plt.subplots(1,3,figsize=(13.5,4.2),constrained_layout=True)
    for axis,length in zip(axes,(13,14,16)):
        for arm,color in ((FULL,"#C68423"),(WINDOW,"#3465A4")):
            steps = summary["arms"][arm]["checkpoint_updates"]
            axis.plot([s/1000 for s in steps],[plot_rows(summary,arm,s)[length-1]["E"] for s in steps],
                      marker="o",markersize=3,color=color,label="Full attention" if arm == FULL else "Window2")
        axis.set(xlabel="Total optimizer updates (thousands)",title=f"E({length})",ylim=(-.025,1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    fig.suptitle("Full and window2 RT + NextLat · both user-selected 80k endpoints; raw tails excluded")
    save(fig,"exactness-vs-updates")
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for axis,key,title in zip(axes,("state_ce","latent_loss"),("Training state CE","Training latent SmoothL1")):
        for arm,color in ((FULL,"#C68423"),(WINDOW,"#3465A4")):
            points = summary["arms"][arm]["training_curve"]
            axis.plot([p["update"]/1000 for p in points],[p[key] for p in points],color=color,
                      label="Full attention (accepted through 80k)" if arm == FULL else "Window2",linewidth=1)
        axis.set(xlabel="Total optimizer updates (thousands)",ylabel=title)
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    fig.suptitle("Complete accepted histories · nonoverlapping 100-update means · full-attention unsaved tail excluded")
    save(fig,"training-losses")
    return figures


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh window-budget report directory")
    summary = make_summary(args.pilot_dir,args.run_dir,args.protocol,args.revision)
    rows = metric_table(summary)
    output.mkdir(parents=True)
    records = []
    for arm in (WINDOW,FULL):
        for phase,inputs in summary["arms"][arm]["inputs"].items():
            records += [(Path("inputs") / arm / phase / Path(r["path"]).name,r) for r in inputs.values()]
    records += [(Path("inputs/window-endpoint-revision.json"),summary["revision_input"]),
                (Path("inputs/window-training-exit-code.txt"),summary["revision_evidence"]["termination_exit"])]
    ref = summary["full_attention_reference_provenance"]
    for phase,inputs in ref["pure_rt_order_inputs"].items():
        records += [(Path("inputs/pure-rt-order") / phase / Path(r["path"]).name,r) for k,r in inputs.items() if k != "endpoint_checkpoint"]
    records += [(Path("inputs/protocol.json"),summary["protocol_input"]),
                (Path("inputs/full-reference/endpoint-revision.json"),ref["revision_input"]),
                (Path("inputs/full-reference/protocol.json"),ref["revision_evidence"]["protocol"]),
                (Path("inputs/full-reference/training-exit-code.txt"),ref["revision_evidence"]["termination_exit"])]
    for relative,record in records:
        target = output / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(record["path"],target)
        if any(hash_file(target)[k] != record[k] for k in ("sha256","bytes")):
            raise ValueError("Window/full input changed during evidence copy")
    summary["reporting_sources"] = {}
    for relative in REPORTING_SOURCES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT / relative,target)
        summary["reporting_sources"][relative] = hash_file(target)
    summary["reporting_source_sha256"] = _digest_dict({k:v["sha256"] for k,v in summary["reporting_sources"].items()})
    budget.write_csv(output / "metrics.csv",rows,CSV_COLUMNS)
    with (output / "metrics.csv").open() as stream:
        if list(csv.DictReader(stream)) != [{k:str(r[k]) for k in CSV_COLUMNS} for r in rows]:
            raise ValueError("CSV differs from identical endpoint plot rows")
    budget.write_csv(output / "checkpoint-summary.csv",summary["checkpoint_summary"],summary["checkpoint_summary"][0].keys())
    for arm in (WINDOW,FULL):
        points = summary["arms"][arm]["training_curve"]
        budget.write_csv(output / f"training-{arm}.csv",points,points[0].keys())
    write_json(output / "summary.json",summary)
    write_json(output / "plot-data.json",summary)
    figures = plot_results(summary,output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking",entity="taylorbollman",output_dir=output,
                            group=args.wandb_group,name="rt-nextlat-window2-user-selected 80k-report")
    result = {**summary,"status":"running","figures":figures}
    try:
        tracker.start({k:summary[k] for k in ("schema","scope","contract","budget","primary_update","selection_kind","stop")})
        import wandb
        tracker.log({"report/metrics":wandb.Table(columns=list(CSV_COLUMNS),data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}":wandb.Image(str(output / files["png"])) for name,files in figures.items()}})
        bins = {arm:{p["update"]:p for p in summary["arms"][arm]["training_curve"]} for arm in (WINDOW,FULL)}
        for step in sorted(bins[WINDOW]):
            values = {"update":step}
            for arm in (WINDOW,FULL):
                if step in bins[arm]:
                    values.update({f"train/{arm}/{key}":value for key,value in bins[arm][step].items()
                                   if key not in ("update","first_update","updates_in_bin")})
                if str(step) in summary["arms"][arm]["curves"]:
                    for length in budget.KEY_LENGTHS:
                        row = plot_rows(summary,arm,step)[length-1]
                        values.update({f"dev/{arm}/prefix_{length}/{key}":row[key] for key in ("E","A","M")})
            tracker.log(values)
        tracker.summary({"primary_update":ENDPOINT,"full_attention_100k_available":False,"confirmation_evaluated":False,
                         "latent_rollout_evaluated":False,"checkpoint_summary":summary["checkpoint_summary"]})
        tracker.finish(succeeded=True)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p != output / "report.json"}
        write_json(output / "report.json",result)
    print(json.dumps({"status":result["status"],"output_dir":str(output),"wandb":tracker.record["run_url"]}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir",default=str(ROOT / ".runtime/rt-a5/20260914T175404Z-nextlat-mitchell-window/train-window2"))
    parser.add_argument("--run-dir",default=str(LINEAGE / "train-window2"))
    parser.add_argument("--protocol",default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--revision",default=str(LINEAGE / "endpoint-revision.json"))
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--wandb-group",default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
