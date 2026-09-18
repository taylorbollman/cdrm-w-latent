#!/usr/bin/env python3
"""CPU-only window-2 RT + NextLat 100k report, with matched full-RT references through 80k."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from scripts import rt_a5_nextlat_budget_report as budget
from scripts import rt_a5_nextlat_stopped_report as stopped
from scripts import rt_a5_window_report as window
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, STEPS, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT = budget.ROOT
LINEAGE = ROOT / ".runtime/rt-a5/20260914T201057Z-rt-nextlat-window2-100k"
SCHEMA = "rt-a5-window-budget-comparison-v1"
WINDOW, FULL = "rt_nextlat_window2", "rt_nextlat"
ENDPOINT, FULL_ENDPOINT = 100000, 80000
SOURCE_SHA = "9d12d61e994bdad57567cb1d30fd36f1ea3296c94f7aa401d574ecbb34d339c2"
PARENT_SHA = "9385386a9a559b397f144ce7aab7132d016459a857b247c69f4bf83311ecb6dc"
PARENT_REPORT_SHA = "343fe8b31e6bab944eb00a074b1fc57375de77cc3534cb0ea82dbe9db3b9add0"
REPORTING_SOURCES = tuple(dict.fromkeys(("scripts/rt_a5_window_budget_report.py", *stopped.REPORTING_SOURCES,
                                        "scripts/rt_a5_window_report.py", "scripts/rt_a5_nextlat_variant_report.py")))


def validate_protocol(protocol):
    expected = {"schema":"rt-a5-window-budget-protocol-v1", "start_update":10000, "endpoint":ENDPOINT,
                "additional_updates":90000, "checkpoint_steps":list(budget.NEW_STEPS), "batch_size":1024,
                "full_evaluation_rows":102400, "confirmation_evaluated":False, "latent_rollout_evaluated":False,
                "parent_checkpoint_sha256":PARENT_SHA, "parent_report_sha256":PARENT_REPORT_SHA}
    for key,value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Frozen window continuation protocol differs: {key}")
    if len(protocol["source_files"]) != 52 or _digest_dict(protocol["source_files"]) != SOURCE_SHA:
        raise ValueError("Frozen52-source window lineage differs")
    contract = protocol["strict_contract"]
    window.validate_window_contract(contract)
    if (contract.get("source_sha256") != SOURCE_SHA or contract["experiment_config"]["position_encoding"] != "alibi"
            or contract["experiment_config"]["second_layer_window"] != 2):
        raise ValueError("Expected original Mitchell ALiBi layer-2 window2")
    if protocol["full_attention_reference"].get("accepted_endpoint") != FULL_ENDPOINT:
        raise ValueError("Full-attention reference scope must end at accepted80k")


def validate_config(config, protocol, *, pilot):
    budget.verify_run_config(config, protocol, pilot=pilot)
    if config.get("position_encoding") != "alibi" or config.get("second_layer_window") != 2:
        raise ValueError("Window continuation CLI changed positions or direct-read window")


def read_continuation(directory, protocol, *, through_update=ENDPOINT):
    if through_update not in (20000, ENDPOINT):
        raise ValueError("Only final100k or bounded read-only20k preflight is supported")
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    final = through_update == ENDPOINT
    if report.get("schema") != window.TRAIN_SCHEMA or report.get("status") not in (("complete",) if final else ("running","complete")):
        raise ValueError("Window report must complete100k; running allowed only for explicit20k preflight")
    observed = report.get("completed_updates",0)
    if (type(observed) is not int or observed < through_update or observed > ENDPOINT
            or report.get("start_update") != 10000 or report.get("endpoint") != ENDPOINT):
        raise ValueError("Window continuation interval differs or requested checkpoint not completed")
    if final and (observed != ENDPOINT or report.get("wandb",{}).get("status") != "synced"):
        raise ValueError("Window100k endpoint and online evidence must be complete")
    if report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous latent rollout must remain unevaluated")
    if report["contract"] != protocol["strict_contract"] or report["source_files"] != protocol["source_files"]:
        raise ValueError("Window strict contract or frozen training sources differ")
    for relative,expected in report["source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid source snapshot path/hash")
        if hash_file(directory / "source" / relative)["sha256"] != expected:
            raise ValueError(f"Window frozen source changed: {relative}")
    raw_config, config_file = read_input(directory / "config.json")
    config = json.loads(raw_config)
    validate_config(config, protocol, pilot=False)
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != report["contract"]["data_manifest_sha256"]:
        raise ValueError("Window data manifest differs")
    if final:
        raw_history, history_file = read_input(directory / "history.jsonl")
    else:
        with (directory / "history.jsonl").open("rb") as stream:
            raw_history = b"".join(stream.readline() for _ in range(observed-10000))
        history_file = {"path":str(directory / "history.jsonl"), "bytes":len(raw_history),
                        "sha256":hashlib.sha256(raw_history).hexdigest(), "snapshot":"prefix through observed saved update"}
    history = [json.loads(line) for line in raw_history.splitlines() if line.strip()]
    budget.validate_history(history, report, 10000, observed)
    expected_steps = [step for step in budget.NEW_STEPS if step <= observed]
    if [c["completed_updates"] for c in report["checkpoints"]] != expected_steps:
        raise ValueError("Window retained checkpoint schedule differs")
    checkpoints = {}
    for item in report["checkpoints"]:
        if item.get("examples_seen") != item["completed_updates"]*1024:
            raise ValueError("Window checkpoint exposure differs")
        checkpoints[str(item["completed_updates"])] = stopped.verify_file(item)
    budget.verify_evaluations(report,10000,observed,budget.NEW_STEPS)
    return {"report":report, "history":history, "checkpoints":checkpoints,
            "input_files":{"report":report_file,"history":history_file,"run_config":config_file,"data_manifest":manifest}}


def verify_join(pilot, continuation, protocol, through_update):
    left,right = pilot["report"],continuation["report"]
    if (left["start_update"],left["completed_updates"],right["start_update"],right["endpoint"]) != (0,10000,10000,ENDPOINT):
        raise ValueError("Expected original window0->10k and exact10k->100k continuation")
    if left["contract"] != right["contract"] or right["contract"] != protocol["strict_contract"]:
        raise ValueError("Window resume strict contracts differ")
    if left["initialization"] != right["initialization"]:
        raise ValueError("Window initialization lineage changed during resume")
    if left["source_files"] != right["source_files"] or right["source_files"] != protocol["source_files"]:
        raise ValueError("Window source snapshots differ across resume")
    parent = right.get("parent_checkpoint") or {}
    expected = pilot["checkpoints"]["10000"]
    if (parent.get("sha256") != protocol["parent_checkpoint_sha256"] or parent.get("sha256") != expected["sha256"]
            or local_path(parent["path"]).resolve() != local_path(expected["path"]).resolve()
            or pilot["input_files"]["report"]["sha256"] != protocol["parent_report_sha256"]):
        raise ValueError("Window resume parent identity differs")
    joined = pilot["history"] + [row for row in continuation["history"] if row["update"] <= through_update]
    if len(joined) != through_update or any(row["update"] != i for i,row in enumerate(joined,1)):
        raise ValueError("Window joined history must contain each update exactly once")
    return joined


def compare_full_reference(window_contract, window_sources, window_initialization, history, full):
    window.compare_contracts(full["contract"],window_contract)
    if _digest_dict(window_sources) != window_contract["source_sha256"]:
        raise ValueError("Window source map differs from its contract")
    window.validate_source_lineage(full["source_files"],window_sources)
    if (window_initialization.get("baseline_initialization") != full["initialization"]
            or window_initialization.get("changed_parameter_slices") != []):
        raise ValueError("Window and full RT + NextLat original initialization differs")
    if (full.get("primary_update") != FULL_ENDPOINT or full.get("selection_kind") != "user_selected_after_development_review"
            or full.get("planned_endpoint_reached") is not False):
        raise ValueError("Full RT reference must retain its user-selected 80k qualification")
    if any(int(s) > FULL_ENDPOINT for s in full["curves"]):
        raise ValueError("Full RT reference contains unavailable later retained outcomes")
    full_history = []
    for phase in ("pilot","continuation"):
        path = full["inputs"][phase]["history"]["path"]
        full_history.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    accepted = [r for r in full_history if r["update"] <= min(len(history),FULL_ENDPOINT)]
    if [r["order_chain"] for r in accepted] != [r["order_chain"] for r in history[:len(accepted)]]:
        raise ValueError("Window and full RT minibatch order differs at matched budgets")
    # The pure RT reference has independently retained100k history; use its order
    # evidence only to cover window updates beyond the accepted full-NextLat80k.
    independent = []
    for phase in ("pilot","continuation"):
        path = full["pure_rt_reference"]["inputs"][phase]["history"]["path"]
        independent.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    if [r["order_chain"] for r in independent[:len(history)]] != [r["order_chain"] for r in history]:
        raise ValueError("Window full data-order lineage differs from independent100k history")


def make_summary(pilot_dir,run_dir,protocol_path,*,through_update=ENDPOINT):
    raw,protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    validate_protocol(protocol)
    pilot = window.read_window(local_path(pilot_dir),WINDOW)
    validate_config(json.loads(Path(pilot["input_files"]["run_config"]["path"]).read_text()),protocol,pilot=True)
    budget.validate_history(pilot["history"],pilot["report"],0,10000)
    budget.verify_evaluations(pilot["report"],0,10000,STEPS)
    continuation = read_continuation(local_path(run_dir),protocol,through_update=through_update)
    history = verify_join(pilot,continuation,protocol,through_update)
    reference = protocol["full_attention_reference"]
    full_lineage = local_path(reference["lineage"])
    full = stopped.make_summary(ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat",
        full_lineage / "train-rt-nextlat",full_lineage / "protocol.json",local_path(reference["revision"]),
        ROOT / ".runtime/rt-a5/20260911T154748Z/train-rt",ROOT / ".runtime/rt-a5/20260911T171239Z-rt100k/train-rt")
    compare_full_reference(protocol["strict_contract"],protocol["source_files"],
                           pilot["report"]["initialization"],history,full)
    steps = [*STEPS,*(s for s in budget.NEW_STEPS if s <= through_update)]
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
    shared_steps = [s for s in steps if s <= FULL_ENDPOINT]
    result = {"schema":SCHEMA,"primary_update":ENDPOINT,"reported_through_update":through_update,
        "endpoint_completed":through_update == ENDPOINT,"full_attention_max_accepted_update":FULL_ENDPOINT,
        "full_attention_100k_available":False,"matched_checkpoint_updates":shared_steps,
        "scope":"Window2 fixed 100k endpoint; intermediate checkpoints diagnostic. Full-attention RT+NextLat stopped at user-selected 80k after development review; no full-attention 100k comparator.",
        "confirmation_evaluated":False,"latent_rollout_evaluated":False,"evaluation_route":"backbone_only",
        "protocol":protocol,"protocol_input":protocol_file,"contract":protocol["strict_contract"],
        "source_files":protocol["source_files"],"initialization":pilot["report"]["initialization"],
        "order_chain":history[-1]["order_chain"],"parent_checkpoint":continuation["report"]["parent_checkpoint"],
        "arms":{WINDOW:{"label":"ALiBi, layer 2 window2, RT + NextLat","checkpoint_updates":steps,
            "curves":curves,"metrics":metrics,"checkpoints":checkpoints,
            "training_curve":training_curve(history,True),"inputs":{"pilot":pilot["input_files"],"continuation":continuation["input_files"]},
            "wandb":{"pilot":pilot["report"]["wandb"],"continuation":continuation["report"]["wandb"]}},
            FULL:{"label":"ALiBi, full attention, RT + NextLat (user-stopped80k)","checkpoint_updates":full["checkpoint_updates"],
                "curves":full["curves"],"metrics":full["historical_metrics"],"checkpoints":full["checkpoints"],
                "training_curve":full["training_curve"],"inputs":full["inputs"],"wandb":full["historical_wandb"]}},
        "full_attention_reference_provenance":{"schema":full["schema"],"selection_kind":full["selection_kind"],
            "endpoint_revision":full["endpoint_revision"],"revision_input":full["revision_input"],
            "revision_evidence":full["revision_evidence"],"contract":full["contract"],"source_files":full["source_files"],
            "stop":full["stop"],"pure_rt_order_inputs":full["pure_rt_reference"]["inputs"]},
        "budget":{"total_updates":through_update,"additional_updates":through_update-10000,
            "total_word_presentations":through_update*1024,"unique_training_words":800000,
            "nominal_training_passes":through_update*1024/800000},
        "training_bin_updates":100,"training_seconds":sum(r["seconds"] for r in history),
        "metric_definitions":{"E":"Every state through t correct","A":"Only state at t correct","M":"Mean token correctness through t"},
        "intervals":"Pointwise Wilson 95% across words for E/A; no training-seed or paired-difference uncertainty; no M interval"}
    result["checkpoint_summary"] = checkpoint_summary(result)
    return result


def metric_table(summary):
    rows = [r for arm in (WINDOW,FULL) for step in summary["arms"][arm]["checkpoint_updates"]
            for role in ROLES for r in summary["arms"][arm]["curves"][str(step)][role]]
    expected = sum(len(summary["arms"][arm]["checkpoint_updates"])*48 for arm in (WINDOW,FULL))
    if len(rows) != expected or len({(r["arm"],r["update"],r["role"],r["length"]) for r in rows}) != expected:
        raise ValueError("Missing or duplicate window/full comparison rows")
    if any(r["arm"] == FULL and r["update"] > FULL_ENDPOINT for r in rows):
        raise ValueError("Unavailable full-attention 100k or unsaved tail leaked into comparison")
    return rows


def plot_rows(summary,arm,step):
    if arm == FULL and step > FULL_ENDPOINT:
        raise ValueError("Full-attention100k comparator is unavailable")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [r["length"] for r in rows] != list(range(1,37)):
        raise ValueError("Plot requires every prefix position exactly once")
    return rows


def checkpoint_summary(summary):
    rows = []
    for arm in (WINDOW,FULL):
        packet = summary["arms"][arm]
        for step in packet["checkpoint_updates"]:
            dev,ood = packet["metrics"][str(step)]["dev"],packet["curves"][str(step)]["ood_dev"]
            rows.append({"arm":arm,"update":step,"matched_budget_available":step in summary["matched_checkpoint_updates"],
                "dev_token_accuracy":dev["token_accuracy"],"dev_whole_word_exact_match":dev["whole_word_exact_match"],
                "ood_ce":packet["metrics"][str(step)]["ood_dev"]["ce"],"A36":ood[-1]["A"],"M36":ood[-1]["M"],
                **{f"E{t}":ood[t-1]["E"] for t in budget.KEY_LENGTHS}})
    return rows


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

    primary = [(WINDOW,10000,"Window2,10k","#C68423"),(WINDOW,100000,"Window2,100k","#3465A4")]
    for name,limits in (("length-full",(1,36)),("length-boundary",(10,18))):
        lengths(name,primary,limits,"Window2 RT + NextLat · fixed 100k versus its 10k checkpoint · 102,400 OOD words")
    lengths("matched-budget-boundary",[(FULL,80000,"Full attention,80k","#C68423"),(WINDOW,80000,"Window2,80k","#3465A4")],
            (10,18),"Matched 80k budgets · full attention endpoint was user-selected after development review")
    fig,axes = plt.subplots(1,3,figsize=(13.5,4.2),constrained_layout=True)
    for axis,length in zip(axes,(13,14,16)):
        for arm,color in ((FULL,"#C68423"),(WINDOW,"#3465A4")):
            steps = summary["arms"][arm]["checkpoint_updates"]
            axis.plot([s/1000 for s in steps],[plot_rows(summary,arm,s)[length-1]["E"] for s in steps],
                      marker="o",markersize=3,color=color,label="Full attention (ends80k)" if arm == FULL else "Window2")
        axis.set(xlabel="Total optimizer updates (thousands)",title=f"E({length})",ylim=(-.025,1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    fig.suptitle("Full and window2 RT + NextLat · matched budgets through 80k; no full-attention 100k endpoint")
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


def markdown_report(summary):
    lines = ["# RT + NextLat, layer-2 window2: fixed 100k continuation", "",
        "The original **Mitchell + ALiBi, layer-2 window2 RT + NextLat** checkpoint was resumed from 10k to the "
        "prospectively fixed **100,000-update endpoint**, retaining model, Adam, RNG and absolute data-order state. "
        "Intermediate checkpoints are diagnostics; no best-checkpoint selection is used for this run.", "",
        "Layer 1 retains full-prefix recurrent attention. At layer 2, each step reads temporary self K/V and the immediately "
        "preceding output's permanent K/V. Recurrent gradients remain attached, so this is a restriction on direct reads, "
        "not two-token memory. Mitchell initialization, ALiBi, D512/H8/GELU-FFN2048, LayerNorm/QK norm, rho1, "
        "state CE plus weight-one latent SmoothL1 and full FP32 remain unchanged. Total 7,407,104 parameters. "
        "Evaluation uses only the RT backbone; the NextLat predictor is not autonomously rolled out.", "",
        "**Full-attention RT + NextLat has no 100k comparator.** That run was stopped by the user at its retained 80k checkpoint "
        "after reviewing development curves. Equal-budget comparisons are available only through 80k. The full-attention "
        "unsaved tail is excluded, and its retrospective endpoint qualification is retained.", "",
        "| Model | Updates | Same-budget counterpart | L12 whole word | E(13) | E(14) | E(16) | M(36) | E(36) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["update"] not in (10000,80000,100000):
            continue
        label = "Window2 RT + NextLat" if row["arm"] == WINDOW else "Full attention RT + NextLat"
        values = [row[k] for k in ("dev_whole_word_exact_match","E13","E14","E16","M36","E36")]
        lines.append(f"| {label} | {row['update']:,} | {'yes' if row['matched_budget_available'] else 'unavailable'} | "
                     + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "E(t) requires every state through t to be correct, A(t) checks only state t, and M(t) averages correctness "
        "through t. OOD curves use prefixes of the same 102,400 length-36 development words. Full and boundary figures consume "
        "identical rows. L12 development is a separate short-word set. The 1/60 guessing reference applies only to isolated A(t).", "",
        "![Window 10k versus 100k, full range](length-full.png)", "", "![Same window endpoint curves, boundary view](length-boundary.png)", "",
        "![Matched 80k comparison](matched-budget-boundary.png)", "", "![Exactness versus training budget](exactness-vs-updates.png)", "",
        "![Training state and latent losses](training-losses.png)", "",
        "| Updates | Window E(13) | Full E(13) | Window E(14) | Full E(14) | Window E(16) | Full E(16) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by_key = {(r["arm"],r["update"]):r for r in summary["checkpoint_summary"]}
    for step in summary["arms"][WINDOW]["checkpoint_updates"]:
        values = []
        for length in (13,14,16):
            for arm in (WINDOW,FULL):
                row = by_key.get((arm,step))
                values.append("unavailable" if row is None else f"{100*row[f'E{length}']:.4f}%")
        lines.append(f"| {step:,} | " + " | ".join(values) + " |")
    lines += ["", "Every reported checkpoint evaluation uses102,400 words per role. Source 52 snapshots, strict resume contract, "
        "checkpoint hashes, original initialization, integer metric counts and every minibatch-order hash were checked. "
        "Full-attention source 46 and its explicit user-stop revision were independently revalidated by the stopped-run reader. "
        "Existing pure RT 100k history is used only as additional data-order evidence; no pure RT outcome is substituted for a "
        "missing full-attention RT + NextLat 100k outcome.", "",
        "This is one development seed, following earlier development-driven selection of ALiBi. Pointwise Wilson 95% bands "
        "describe word sampling for E/A, not seed variation, simultaneous coverage or paired differences. Zero observed E does "
        "not prove population impossibility. Final confirmation and autonomous latent rollout remain **unevaluated**.", "",
        f"Window training exposure: {summary['budget']['total_word_presentations']:,} words over 800,000 unique training words "
        f"({summary['budget']['nominal_training_passes']:g} nominal passes). Recorded update-loop time: "
        f"{summary['training_seconds']/3600:.3f}h; evaluation/checkpoint/logging overhead is excluded.", "",
        "[Exact metric counts](metrics.csv) · [Checkpoint summary](checkpoint-summary.csv) · [Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh window-budget report directory")
    summary = make_summary(args.pilot_dir,args.run_dir,args.protocol)
    rows = metric_table(summary)
    output.mkdir(parents=True)
    records = []
    for arm in (WINDOW,FULL):
        for phase,inputs in summary["arms"][arm]["inputs"].items():
            records += [(Path("inputs") / arm / phase / Path(r["path"]).name,r) for r in inputs.values()]
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
                            group=args.wandb_group,name="rt-nextlat-window2-budget100k-report")
    result = {**summary,"status":"running","figures":figures}
    try:
        tracker.start({k:summary[k] for k in ("schema","scope","contract","budget","primary_update","full_attention_100k_available")})
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
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--wandb-group",default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
