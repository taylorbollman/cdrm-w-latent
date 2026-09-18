#!/usr/bin/env python3
"""CPU-only four-arm A5 comparison at 80k, preserving initialization and stop qualifications."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

from scripts import rt_a5_nextlat_budget_report as budget
from scripts import rt_a5_window_stopped_report as stopped_window
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json

ROOT = budget.ROOT
LINEAGE = ROOT / ".runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k"
SCHEMA = "rt-a5-depth-order-comparison-v1"
TRAIN_SCHEMA = "rt-a5-depth-order-training-v1"
SOURCE_SHA = "cc9fbfa50ebd57387f4ad95803bb2db9d42b69a13875268c0400759858fe4176"
FRESH = ("seq4_alibi","rt_window2_first")
REFERENCES = ("rt_nextlat","rt_nextlat_window2")
ARMS = (*FRESH,*REFERENCES)
ENDPOINT = 80000
STEPS = (1000,5000,10000,20000,25000,30000,40000,50000,60000,70000,80000)
ADDITIONS = {"scripts/rt_a5_depth_order.py","scripts/rt_a5_depth_order_train.py","configs/rt_a5_depth_order/base.json"}
LABELS = {"seq4_alibi":"SEQ4 + NextLat", "rt_window2_first":"RT: first layer window 2 + NextLat",
          "rt_nextlat":"RT: both layers full + NextLat", "rt_nextlat_window2":"RT: second layer window 2 + NextLat"}
COLORS = {"seq4_alibi":"#7B5BA7","rt_window2_first":"#24938C","rt_nextlat":"#C68423","rt_nextlat_window2":"#3465A4"}
COUNTS = {"seq4_alibi":(12653056,13702656,39),"rt_window2_first":(6357504,7407104,25)}
REPORTING_SOURCES = ("scripts/rt_a5_depth_order_report.py",*stopped_window.REPORTING_SOURCES)


def validate_protocol(protocol):
    expected = {"schema":"rt-a5-depth-order-protocol-v1","arm_order":list(FRESH),"start_update":0,"endpoint":ENDPOINT,
        "checkpoint_steps":[0,*STEPS],"batch_size":1024,"training_words":800000,"training_length":12,"ood_length":36,
        "full_evaluation_rows":102400,"evaluation_every":500,"evaluation_rows":4096,"diagnostic_rows":1024,
        "confirmation_evaluated":False,"latent_rollout_evaluated":False,"evaluation_route":"backbone_only"}
    for key,value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Depth/order prospective protocol differs: {key}")
    sources = protocol["source_files"]
    if len(sources) != 55 or _digest_dict(sources) != SOURCE_SHA or protocol.get("source_sha256") != SOURCE_SHA:
        raise ValueError("Frozen 55-source depth/order lineage differs")
    if not ADDITIONS.issubset(sources):
        raise ValueError("Depth/order source additions missing")
    historical = {k:v for k,v in sources.items() if k not in ADDITIONS}
    if _digest_dict(historical) != stopped_window.continuing.SOURCE_SHA:
        raise ValueError("Historical 52 training sources changed")
    for variant in FRESH:
        packet = protocol["arms"][variant]
        backbone,total,tensors = COUNTS[variant]
        if packet.get("variant") != variant or packet.get("parameter_count") != total or packet.get("parameter_tensors") != tensors:
            raise ValueError("Protocol variant parameter/tensor counts differ")
        validate_contract(packet["strict_contract"],variant)
        initial = packet["initialization"]
        if (initial.get("parameter_count") != total or initial.get("backbone_parameter_count") != backbone
                or initial.get("predictor_parameter_count") != 1049600
                or initial.get("schema") != "rt-a5-depth-order-initialization-v1"):
            raise ValueError("Protocol initialization dimensions differ")
    for item in (protocol["data_manifest"],protocol["preflight"],*protocol["references"].values()):
        stopped_window.stopped.verify_file(item)


def validate_contract(contract,variant):
    seq = variant == "seq4_alibi"
    expected = {"schema":TRAIN_SCHEMA,"variant":variant,"architecture":"seq" if seq else "rt",
        "source_sha256":SOURCE_SHA,"width":512,"batch_size":1024,"train_rows":800000,"length":12,
        "seed":1234,"predictor_seed":1235,"data_order_seed":1234,"precision":"fp32","tf32":False,
        "compile":False,"cuda_graphs":False,"evaluation_route":"backbone_only","latent_rollout_evaluated":False,
        "training_step":"scripts.rt_a5_nextlat_train.train_step (same function object)",
        "evaluation":"scripts.rt_a5_train.evaluate_arrays (same function object)",
        "one_step_diagnostics":"scripts.rt_a5_nextlat_train.evaluate_diagnostics (same function object)"}
    for key,value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Depth/order contract differs: {key}")
    config,experiment = contract["model_config"],contract["experiment_config"]
    for key,value in {"n_layers":4 if seq else 2,"block_type":"sequential" if seq else "recurrent",
        "d_model":512,"n_heads":8,"mlp_hidden_size":2048,"activation_type":"gelu","init_fn":"mitchell",
        "alibi":True,"rope":False,"cdrm_enabled":False,"recurrent_layers":None,"recurrent_write_rho":1.0}.items():
        if config.get(key) != value:
            raise ValueError(f"Depth/order backbone differs: {key}")
    for key,value in {"variant":variant,"architecture":"seq" if seq else "rt","n_layers":4 if seq else 2,
        "window_layer":None if seq else 0,"window_length":None if seq else 2,"position_encoding":"alibi",
        "learned_position_parameters":False,"evaluation_route":"backbone_only","latent_rollout_evaluated":False}.items():
        if experiment.get(key) != value:
            raise ValueError(f"Depth/order resolved experiment differs: {key}")
    expected_attention = {"layers":["ordinary full causal attention"]*4 if seq else
        ["self provisional K/V and immediately previous permanent output K/V","full causal recurrent attention"],
        "recurrent_write_rho":None if seq else 1.0,"gradient_truncation":False,"layer_indices":"zero based",
        "window_is_direct_read_limit_not_history_truncation":not seq}
    if experiment.get("attention") != expected_attention:
        raise ValueError("Depth/order attention location or gradient semantics differ")


def shared_contract(contract):
    """Remove only declared architecture/depth/provenance differences before pairing."""
    result = copy.deepcopy(contract)
    for key in ("schema","variant","source_sha256","experiment_config","training_step","evaluation","one_step_diagnostics","architecture"):
        result.pop(key,None)
    result["model_config"].pop("block_type")
    result["model_config"].pop("n_layers")
    return result


def read_fresh(variant,protocol,*,through_update=ENDPOINT):
    if variant not in FRESH or through_update not in (1000,ENDPOINT):
        raise ValueError("Use a declared fresh arm at final 80k or a bounded read-only 1k preflight")
    expected = protocol["arms"][variant]
    directory = local_path(expected["directory"]).resolve()
    raw,report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    final = through_update == ENDPOINT
    if (report.get("schema") != TRAIN_SCHEMA or report.get("status") not in (("complete",) if final else ("running","complete"))
            or report.get("start_update") != 0 or report.get("parent_checkpoint") is not None or report.get("endpoint") != ENDPOINT):
        raise ValueError("Require a fresh declared 0->80k run; no resume or stopped endpoint substitution")
    observed = report.get("completed_updates",0)
    if type(observed) is not int or not through_update <= observed <= ENDPOINT:
        raise ValueError("Requested fresh-arm checkpoint has not completed")
    if final and (observed != ENDPOINT or report.get("wandb",{}).get("status") != "synced"):
        raise ValueError("Both fresh 80k endpoints must complete with synced online evidence")
    if report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous latent rollout must remain unevaluated")
    if (report["contract"] != expected["strict_contract"] or report["initialization"] != expected["initialization"]
            or report["source_files"] != protocol["source_files"]):
        raise ValueError("Fresh-arm strict contract, initialization or frozen sources differ from prospective protocol")
    for relative,wanted in report["source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(wanted):
            raise ValueError("Invalid frozen source path/hash")
        if hash_file(directory / "source" / path)["sha256"] != wanted:
            raise ValueError(f"Fresh-arm frozen source changed: {relative}")
    config_raw,config_file = read_input(directory / "config.json")
    config = json.loads(config_raw)
    if config != expected["resolved_args"]:
        raise ValueError("Fresh-arm executed CLI differs from frozen prospective arguments")
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != report["contract"]["data_manifest_sha256"]:
        raise ValueError("Fresh-arm data manifest differs")
    if final:
        history_raw,history_file = read_input(directory / "history.jsonl")
    else:
        with (directory / "history.jsonl").open("rb") as stream:
            history_raw = b"".join(stream.readline() for _ in range(observed))
        history_file = {"path":str(directory / "history.jsonl"),"bytes":len(history_raw),
            "sha256":hashlib.sha256(history_raw).hexdigest(),"snapshot":"prefix through observed saved update"}
    history = [json.loads(line) for line in history_raw.splitlines() if line.strip()]
    budget.validate_history(history,report,0,observed)
    budget.verify_evaluations(report,0,observed,STEPS)
    expected_checkpoints = [0,*(s for s in STEPS if s <= observed)]
    if [r["completed_updates"] for r in report["checkpoints"]] != expected_checkpoints:
        raise ValueError("Fresh-arm checkpoint schedule differs")
    checkpoints = {}
    for item in report["checkpoints"]:
        if item.get("examples_seen") != item["completed_updates"]*1024:
            raise ValueError("Fresh-arm checkpoint word exposure differs")
        checkpoints[str(item["completed_updates"])] = stopped_window.stopped.verify_file(item)
    selected_steps = [s for s in STEPS if s <= through_update]
    curves,metrics = {},{}
    for step in selected_steps:
        curves[str(step)],metrics[str(step)] = {},{}
        for role in ROLES:
            selected = [m for m in report["evaluations"] if (m["update"],m["role"]) == (step,role)]
            if len(selected) != 1 or selected[0]["rows"] != 102400:
                raise ValueError("Missing full checkpoint development evaluation")
            metrics[str(step)][role] = selected[0]
            curves[str(step)][role] = metric_rows(selected[0],variant)
    return {"label":LABELS[variant],"contract":report["contract"],"initialization":report["initialization"],
        "source_files":report["source_files"],"checkpoint_updates":selected_steps,"curves":curves,"metrics":metrics,
        "checkpoints":checkpoints,"history":history[:through_update],"training_curve":training_curve(history[:through_update],True),
        "inputs":{"training":{"report":report_file,"history":history_file,"run_config":config_file,"data_manifest":manifest}},
        "wandb":{"training":report["wandb"]},"parameter_count":expected["parameter_count"],
        "parameter_tensors":expected["parameter_tensors"],"selection_kind":"prospectively_fixed80k"}


def read_references(protocol):
    ref = protocol["references"]["rt_window_second"]
    revision_path = local_path(ref["path"])
    lineage = revision_path.parent
    result = stopped_window.make_summary(ROOT / ".runtime/rt-a5/20260914T175404Z-nextlat-mitchell-window/train-window2",
                                        lineage / "train-window2",lineage / "protocol.json",revision_path)
    full_record = result["full_attention_reference_provenance"]["revision_input"]
    if (full_record["sha256"] != protocol["references"]["rt_full"]["sha256"]
            or result["revision_input"]["sha256"] != ref["sha256"]):
        raise ValueError("Historical user-stop revisions differ from prospective pair protocol")
    arms = result["arms"]
    for arm in REFERENCES:
        packet = arms[arm]
        packet["label"] = LABELS[arm]
        packet["parameter_count"],packet["parameter_tensors"] = 7407104,25
        packet["selection_kind"] = "user_selected_after_development_review"
        raw_history = []
        for inputs in packet["inputs"].values():
            raw_history.extend(json.loads(line) for line in Path(inputs["history"]["path"]).read_text().splitlines() if line.strip())
        packet["history"] = [r for r in raw_history if r["update"] <= ENDPOINT]
        if len(packet["history"]) != ENDPOINT or any(r["update"] != i for i,r in enumerate(packet["history"],1)):
            raise ValueError("Historical accepted reference history differs")
    arms[REFERENCES[1]].update(contract=result["contract"],initialization=result["initialization"],source_files=result["source_files"])
    full = result["full_attention_reference_provenance"]
    arms[REFERENCES[0]].update(contract=full["contract"],initialization=result["initialization"]["baseline_initialization"],source_files=full["source_files"])
    return arms,{"window_second":{"revision_input":result["revision_input"],"revision_evidence":result["revision_evidence"],
        "endpoint_revision":result["endpoint_revision"],"stop":result["stop"]},"full":full}


def compare_arm(variant,arm,reference):
    if shared_contract(arm["contract"]) != shared_contract(reference["contract"]):
        raise ValueError("Shared data, runtime, width, objective or optimizer contract differs")
    expected_order = [r["order_chain"] for r in reference["history"][:len(arm["history"])]]
    if [r["order_chain"] for r in arm["history"]] != expected_order:
        raise ValueError("Fresh and historical per-update minibatch order differs")
    initial,base = arm["initialization"],reference["initialization"]
    if initial["predictor_sha256"] != base["predictor_sha256"] or initial["predictor_config"] != base["predictor_config"]:
        raise ValueError("NextLat predictor initialization or architecture differs")
    if variant == "rt_window2_first":
        if (initial.get("reference_two_layer_initialization") != base
                or initial.get("exact_two_layer_learned_initialization") is not True
                or initial.get("changed_parameter_slices") != [] or initial["canonical_sha256"] != base["canonical_sha256"]):
            raise ValueError("First-window RT must preserve exact original RT2 learned initialization")
    elif (initial.get("exact_two_layer_learned_initialization") is not False
            or initial.get("changed_parameter_slices") is not None
            or initial["backbone"].get("two_layer_initialization_identity_claimed") is not False
            or initial["canonical_sha256"] == base["canonical_sha256"]):
        raise ValueError("SEQ4 must retain its distinct canonical four-layer initialization qualification")


def make_summary(protocol_path):
    raw,protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    validate_protocol(protocol)
    references,provenance = read_references(protocol)
    arms = {variant:read_fresh(variant,protocol) for variant in FRESH}
    arms.update(references)
    for variant in FRESH:
        compare_arm(variant,arms[variant],arms[REFERENCES[0]])
    if arms[FRESH[1]]["initialization"]["model_parameter_sha256"] != arms[REFERENCES[1]]["initialization"]["model_parameter_sha256"]:
        raise ValueError("First/second-window RT initial learned tensor digest differs")
    for key,value in arms[REFERENCES[1]]["source_files"].items():
        if protocol["source_files"].get(key) != value:
            raise ValueError("Shared historical training source changed")
    order_chain = arms[FRESH[0]]["history"][-1]["order_chain"]
    for arm in ARMS:
        arms[arm].pop("history")
    result = {"schema":SCHEMA,"primary_update":ENDPOINT,"checkpoint_updates":list(STEPS),"protocol":protocol,
        "protocol_input":protocol_file,"source_files":protocol["source_files"],"arms":arms,
        "reference_provenance":provenance,"confirmation_evaluated":False,"latent_rollout_evaluated":False,
        "evaluation_route":"backbone_only","order_chain":order_chain,
        "scope":"Two fresh prospectively fixed 80k endpoints compared with two historical user-selected 80k endpoints. Shared width and exposure; SEQ4 has different depth, parameter count, compute and backbone initialization.",
        "comparison":{"shared_update_budget":True,"shared_parameter_budget":False,"shared_compute_budget":False,
            "rt_initialization_paired":True,"seq4_backbone_initialization_paired":False,
            "new_endpoint_selection":"prospectively_fixed80k","reference_endpoint_selection":"user_selected_after_development_review"},
        "budget":{"total_updates_per_arm":ENDPOINT,"word_presentations_per_arm":ENDPOINT*1024,
            "training_words":800000,"nominal_training_passes_per_arm":ENDPOINT*1024/800000},
        "training_bin_updates":100,"metric_definitions":{"E":"Every state through t correct","A":"Only state at t correct","M":"Mean token correctness through t"},
        "intervals":"Pointwise Wilson 95% across words for E/A; no seed or paired-difference uncertainty; no M interval"}
    result["checkpoint_summary"] = checkpoint_summary(result)
    return result


def metric_table(summary):
    rows = [r for arm in ARMS for step in summary["arms"][arm]["checkpoint_updates"]
            for role in ROLES for r in summary["arms"][arm]["curves"][str(step)][role]]
    expected = len(ARMS)*len(STEPS)*48
    if len(rows) != expected or len({(r["arm"],r["update"],r["role"],r["length"]) for r in rows}) != expected:
        raise ValueError("Four-arm metric table contains missing or duplicate positions")
    if any(r["update"] not in STEPS for r in rows):
        raise ValueError("Unsaved tail or unavailable endpoint leaked into comparison")
    return rows


def plot_rows(summary,arm,step):
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Plot requires a declared arm and retained matched checkpoint")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [r["length"] for r in rows] != list(range(1,37)):
        raise ValueError("Plot must contain every prefix exactly once")
    return rows


def checkpoint_summary(summary):
    result = []
    for arm in ARMS:
        packet = summary["arms"][arm]
        for step in packet["checkpoint_updates"]:
            dev,ood = packet["metrics"][str(step)]["dev"],packet["curves"][str(step)]["ood_dev"]
            result.append({"arm":arm,"update":step,"parameter_count":packet["parameter_count"],
                "selection_kind":packet["selection_kind"],"dev_token_accuracy":dev["token_accuracy"],
                "dev_whole_word_exact_match":dev["whole_word_exact_match"],"ood_ce":packet["metrics"][str(step)]["ood_dev"]["ce"],
                "A36":ood[-1]["A"],"M36":ood[-1]["M"],**{f"E{t}":ood[t-1]["E"] for t in budget.KEY_LENGTHS}})
    return result


def plot_results(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    figures = {}
    short = {FRESH[0]:"SEQ4",FRESH[1]:"RT, first window2",REFERENCES[0]:"RT, full/full",REFERENCES[1]:"RT, second window2"}
    def save(fig,name):
        figures[name] = {}
        for suffix in ("png","pdf"):
            figures[name][suffix] = f"{name}.{suffix}"
            fig.savefig(output / figures[name][suffix],dpi=200)
        plt.close(fig)
    for name,step,limits in (("length-full",80000,(1,36)),("length-boundary",80000,(10,18)),("length-boundary-10k",10000,(10,18))):
        fig,axes = plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
        for arm in ARMS:
            rows = plot_rows(summary,arm,step)
            for axis,key in zip(axes,("E","A","M")):
                axis.plot([r["length"] for r in rows],[r[key] for r in rows],label=short[arm],color=COLORS[arm])
                if key != "M":
                    axis.fill_between([r["length"] for r in rows],[r[f"{key}_low95"] for r in rows],
                        [r[f"{key}_high95"] for r in rows],color=COLORS[arm],alpha=.12)
                axis.set(xlim=limits,ylim=(-.025,1.025),xlabel="Prefix length t of same length-36 words",
                    title={"E":"E: every state through t correct","A":"A: only state t correct","M":"M: mean token accuracy through t"}[key])
                axis.axvline(12,color="black",linestyle=":",alpha=.45)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.grid(alpha=.18)
                axis.legend(fontsize=7)
        axes[1].axhline(1/60,color="gray",linestyle="--",linewidth=.8)
        fig.suptitle(f"All models trained with NextLat · matched {step//1000}k updates · SEQ4 has more parameters\nNew 80k endpoints prospective; historical RT full/full and second-window endpoints selected after development review")
        save(fig,name)
    fig,axes = plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    for axis,length in zip(axes,(13,14,16)):
        for arm in ARMS:
            axis.plot([s/1000 for s in STEPS],[plot_rows(summary,arm,s)[length-1]["E"] for s in STEPS],
                marker="o",markersize=2.5,label=short[arm],color=COLORS[arm])
        axis.set(xlabel="Optimizer updates (thousands)",title=f"E({length})",ylim=(-.025,1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
        axis.legend(fontsize=7)
    fig.suptitle("A5 + NextLat · common development prefixes across matched update budgets")
    save(fig,"exactness-vs-updates")
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for axis,key,label in zip(axes,("state_ce","latent_loss"),("Training state CE","Training latent SmoothL1")):
        for arm in ARMS:
            points = summary["arms"][arm]["training_curve"]
            axis.plot([p["update"]/1000 for p in points],[p[key] for p in points],label=short[arm],color=COLORS[arm],linewidth=1)
        axis.set(xlabel="Optimizer updates (thousands)",ylabel=label)
        axis.grid(alpha=.18)
        axis.legend(fontsize=7)
    fig.suptitle("Accepted histories through 80k · nonoverlapping 100-update means · historical unsaved tails excluded")
    save(fig,"training-losses")
    return figures


def markdown_report(summary):
    lines = ["# A5 + NextLat: four-layer Transformer and recurrent window location", "",
        "The two new models completed prospectively fixed **80,000-update** runs: SEQ4 with ALiBi + NextLat, followed by "
        "two RT layers with window 2 in the first layer and full attention in the second. Historical RT full/full and full/second-window "
        "references are their retained 80k checkpoints, selected by the user after development review. Both historical unsaved tails are excluded.", "",
        "All models use D512/H8/GELU-FFN2048, LayerNorm/full-width QK norm, original Mitchell initialization, ALiBi, full FP32, "
        "the same A5 data/order, batch 1024, AdamW recipe, and unchanged state CE plus weight-one next-latent SmoothL1 objective. "
        "Only target latents are detached. Inference uses the backbone, without autonomous predictor rollout.", "",
        "| Model | Blocks | Parameters, including predictor | Learned tensors | Backbone initialization |",
        "| --- | ---: | ---: | ---: | --- |",
        "| SEQ4 + NextLat |4 ordinary |13,702,656 |39 | Fresh canonical four-layer Mitchell draw |",
        "| Each RT + NextLat variant |2 recurrent |7,407,104 |25 | Exact original RT2 learned tensors at matching layer indices |", "",
        "This is a same-width, same-update-budget comparison, **not a parameter- or FLOP-matched comparison**. SEQ4 has a different "
        "depth, parameter count, compute requirement and canonical backbone initialization. Its common seed does not make its tensors "
        "identical to RT2. The RT window variants preserve the original learned initialization; window location changes direct attention reads, "
        "with gradients still attached through recurrent states. All arms retain the same predictor initialization.", "",
        "| Model | Updates | L12 whole word | E(13) | E(14) | E(16) | M(36) | E(36) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for step in (10000,ENDPOINT):
        for row in summary["checkpoint_summary"]:
            if row["update"] == step:
                lines.append(f"| {LABELS[row['arm']]} | {step:,} | " + " | ".join(f"{100*row[k]:.4f}%" for k in
                    ("dev_whole_word_exact_match","E13","E14","E16","M36","E36")) + " |")
    full_stop = summary["reference_provenance"]["full"]["stop"]
    window_stop = summary["reference_provenance"]["window_second"]["stop"]
    lines += ["", f"The historical full/full job logged {full_stop['raw_completed_updates']:,} updates, of which "
        f"{full_stop['excluded_completed_updates']:,} were an unsaved tail beyond its accepted checkpoint. "
        f"The second-window job logged {window_stop['raw_completed_updates']:,}, excluding its "
        f"{window_stop['excluded_completed_updates']:,}-update tail. Their original protocols planned 100k; "
        "neither historical arm has a completed 100k outcome. These later stopping choices are retrospective, "
        "while both new 80k endpoints were fixed before training.", "", "E(t) requires all states through t correct; A(t) checks only state t; M(t) averages token correctness through t. "
        "Every reported checkpoint uses 102,400 words per development role. OOD curves are prefixes of the same frozen length-36 outputs, "
        "not separate samples for each length. L12 development is a separate set. The 80k full-range and boundary figures use identical rows; "
        "the additional 10k boundary figure is explicitly labeled with its own checkpoint.", "",
        "![All four 80k endpoints](length-full.png)", "", "![Identical 80k rows, boundary view](length-boundary.png)", "",
        "![Four 10k checkpoints, diagnostic reference](length-boundary-10k.png)", "", "![Exactness across update budgets](exactness-vs-updates.png)", "",
        "![Training losses](training-losses.png)", "",
        "The accepted histories contain updates 1–80,000 exactly once per arm, with matching minibatch-order hashes. Source 55 and "
        "the historical 52/46-source subsets, executed arguments, model/objective/runtime contracts, initializations and all checkpoint hashes "
        "are verified. Both explicit historical stop revisions remain bound to raw logs, accepted checkpoints and excluded tails. "
        "This reporter performs no training or model inference.", "",
        "These are one-seed development results. New endpoints were fixed prospectively, while historical endpoints and the choice of "
        "these experiments followed development inspection. Pointwise Wilson 95% bands quantify sampling over words for E/A, not seed "
        "variability, simultaneous coverage or paired differences. Zero E means zero successes in this sample. Final confirmation and "
        "autonomous latent rollout remain **unevaluated**.", "",
        "Each accepted 80k run represents 81,920,000 word presentations over 800,000 unique training words (102.4 nominal passes). "
        "Shared update count does not equate model FLOPs or wall time. Intermediate checkpoints are diagnostics, not selected substitutes "
        "for the new 80k endpoints.", "",
        "[Exact metric counts](metrics.csv) · [All checkpoint summaries](checkpoint-summary.csv) · [Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def input_records(summary):
    """Copy all read-only provenance inputs, including the independent pure-RT order reference."""
    records = [(Path("inputs/protocol.json"),summary["protocol_input"])]
    for arm in ARMS:
        for phase,inputs in summary["arms"][arm]["inputs"].items():
            records += [(Path("inputs") / arm / phase / Path(r["path"]).name,r) for r in inputs.values()]
    for name,reference in summary["reference_provenance"].items():
        records += [(Path("inputs/reference-revisions") / name / "endpoint-revision.json",reference["revision_input"]),
                    (Path("inputs/reference-revisions") / name / "protocol.json",reference["revision_evidence"]["protocol"]),
                    (Path("inputs/reference-revisions") / name / "training-exit-code.txt",reference["revision_evidence"]["termination_exit"])]
    for phase,inputs in summary["reference_provenance"]["full"]["pure_rt_order_inputs"].items():
        records += [(Path("inputs/pure-rt-order") / phase / Path(record["path"]).name,record)
                    for key,record in inputs.items() if key != "endpoint_checkpoint"]
    for name in ("data_manifest","preflight"):
        record = summary["protocol"][name]
        records.append((Path("inputs/prospective") / f"{name}.json",record))
    return records


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh depth/order comparison report directory")
    summary = make_summary(args.protocol)
    rows = metric_table(summary)
    output.mkdir(parents=True)
    records = input_records(summary)
    for relative,record in records:
        source = local_path(record["path"])
        target = output / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if any(hash_file(target)[k] != record[k] for k in ("sha256","bytes")):
            raise ValueError("Depth/order input changed during evidence copy")
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
            raise ValueError("CSV differs from exact full/boundary curve rows")
    budget.write_csv(output / "checkpoint-summary.csv",summary["checkpoint_summary"],summary["checkpoint_summary"][0].keys())
    for arm in ARMS:
        points = summary["arms"][arm]["training_curve"]
        budget.write_csv(output / f"training-{arm}.csv",points,points[0].keys())
    write_json(output / "summary.json",summary)
    write_json(output / "plot-data.json",summary)
    figures = plot_results(summary,output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking",entity="taylorbollman",output_dir=output,
                            group=args.wandb_group,name="nextlat-depth-order-four-arm80k-report")
    result = {**summary,"status":"running","figures":figures}
    try:
        tracker.start({k:summary[k] for k in ("schema","scope","comparison","budget","primary_update")})
        import wandb
        tracker.log({"report/metrics":wandb.Table(columns=list(CSV_COLUMNS),data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
            **{f"report/{name}":wandb.Image(str(output / files["png"])) for name,files in figures.items()}})
        bins = {arm:{p["update"]:p for p in summary["arms"][arm]["training_curve"]} for arm in ARMS}
        for step in range(100,ENDPOINT+1,100):
            values = {"update":step}
            for arm in ARMS:
                values.update({f"train/{arm}/{k}":v for k,v in bins[arm][step].items() if k not in ("update","first_update","updates_in_bin")})
                if step in STEPS:
                    for length in budget.KEY_LENGTHS:
                        row = plot_rows(summary,arm,step)[length-1]
                        values.update({f"dev/{arm}/prefix_{length}/{key}":row[key] for key in ("E","A","M")})
            tracker.log(values)
        tracker.summary({"primary_update":ENDPOINT,"comparison":summary["comparison"],"confirmation_evaluated":False,
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
    parser.add_argument("--protocol",default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--wandb-group",default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
