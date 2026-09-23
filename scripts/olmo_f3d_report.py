#!/usr/bin/env python3
"""Validate and summarize bounded RT backward-memory evidence without scope promotion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from scripts import olmo_f3b_report as prior
from scripts import olmo_f3c_report as f3c

SCHEMAS = {"olmo-f3d-native-v1": "native", "olmo-f3d-probe-v1": "probe"}
PROTOCOL = "docs/reports/olmo1b-f3d/protocol.md"
require, number, digest = prior.require, prior.number, prior.digest


def source_lineage(report, directory, project_root):
    lineage = prior.source_lineage({k:v for k,v in report.items() if k != "protocol_sha256"}, directory, project_root)
    expected = report.get("protocol_sha256")
    require(isinstance(expected, str) and len(expected) == 64, "Missing frozen F3d protocol hash")
    found = []
    for path, label in ((directory/"protocol.md", "protocol.md"),
            (directory/"source-snapshot"/PROTOCOL, "source-snapshot/"+PROTOCOL),
            (project_root/PROTOCOL, "current:"+PROTOCOL)):
        if not path.is_file(): continue
        matches = digest(path) == expected
        if not label.startswith("current:"): require(matches, "Corrupt F3d protocol snapshot")
        if matches: found.append(label)
    require(found, "Unreconstructable F3d protocol")
    lineage["protocol"] = {"sha256": expected, "verified_at": found}
    return lineage


def declared_checks(report):
    adapted = {**report, "schema": "olmo-f3b-native-v1" if SCHEMAS[report["schema"]] == "native" else "olmo-f3b-tile-probe-v1"}
    return prior.declared_checks(adapted)


def configuration_check(report, kind):
    c = report["configuration"]
    if kind == "probe":
        require(c.get("primary_forward_backend") == c.get("primary_backward_backend") == "triton"
                and c.get("primary_cast_weights_once") is True
                and c.get("control_backward_memory") == "materialized"
                and c.get("candidate_backward_memory") == "recompute", "Probe primary configuration differs")
    else:
        require(c.get("variant") in ("reference", "recompute") and c.get("cast_weights_once") is True
                and c.get("forward_tile_backend") == c.get("backward_tile_backend") == "triton",
                "Native primary execution variant differs")
        require(c.get("backward_memory") == ("recompute" if c["variant"] == "recompute" else "materialized"),
                "Backward memory flag contradicts variant")
        require(report.get("stage") == c.get("stage"), "Native stage/configuration differs")


def native_checks(report):
    # F3c's mathematical and full-Adam checks are unchanged. Its internal
    # 'triton' spelling selects the mixed engineering gate, not an execution flag.
    adapted = {**report, "configuration": {**report["configuration"],
        "variant": "triton" if report["configuration"]["variant"] == "recompute" else "reference"}}
    return f3c.native_checks(adapted)


def output_check(row):
    require(row.get("passed") is True and row.get("finite") is True and row.get("shape_matches") is True,
            "Primary reconstruction contains a failed comparison")
    require(row.get("relative_l2_limit") == 1/64 and row.get("max_error_reference_max_limit") == 1/16,
            "Reconstruction budget changed")
    require(number(row.get("relative_l2"), "reconstruction relative error") <= 1/64
            and number(row.get("max_error_reference_max"), "reconstruction maximum error") <= 1/16,
            "Reconstruction budget exceeded")
    if number(row.get("reference_max_abs"), "reconstruction reference maximum") == 0:
        require(number(row.get("max_abs"), "reconstruction absolute error") == 0, "Zero reference requires exact zero")


def no_full_shape(observer):
    require(observer.get("no_full_attention_shape") is True and observer.get("full_attention_outputs") == [],
            "Candidate observer recorded full attention-shaped tensor")
    require(type(observer.get("tensor_outputs")) is int and observer["tensor_outputs"] > 0, "Empty shape observation")


def probe_checks(report):
    checks, stage = report["checks"], report["configuration"].get("stage")
    rows = [r for r in checks if r.get("kind") == "frozen_reconstruction"]
    blocks = [r for r in checks if r.get("kind") == "native_tiny_block"]
    memories = [r for r in checks if r.get("kind") == "isolated_reconstruction_memory"]
    rectangles = [r for r in checks if r.get("kind") == "frozen_long_history"]
    require(len(rows)+len(blocks)+len(memories)+len(rectangles) == len(checks), "Unknown probe check kind")
    expected = {"all": (48,14,3,4), "rows": (48,0,0,4), "blocks": (0,14,0,0), "memory": (0,0,3,0)}.get(stage)
    require(expected == (len(rows),len(blocks),len(memories),len(rectangles)), "Passed probe omitted requested cases")
    if rows:
        require({(r["configuration"]["length"], r["configuration"]["prefix"], r["configuration"]["variant"]) for r in rows}
                == {(t,p,v) for t in (1,9,17,33,65,129) for p in (0,3)
                    for v in ("normal","all_masked","strided_masked","large_scores")}, "Changed reconstruction coverage")
    if blocks:
        require({(r["configuration"]["length"],r["configuration"]["prefix"],r["configuration"]["alpha"]) for r in blocks}
                == {(t,p,a) for t in (9,17) for p in (0,3) for a in (0.,.37,1.)} | {(65,3,1.),(129,3,1.)},
                "Changed block geometry/alpha coverage")
    for row in rows:
        for key in ("attention_vs_f3c_control","diagonal_vs_f3c_control"): output_check(row[key])
        require(row.get("maximum_matches_empty_pattern") is True and row.get("finite_denominator_diagonal_attention") is True,
                "Invalid reconstruction row statistics")
        historical = row.get("historical_gradients_vs_f3c_control")
        config = row["configuration"]
        require((historical is None) == (config["length"] == 1 and config["prefix"] == 0), "Missing historical gradient check")
        if historical is not None: f3c.engineering_gradients(historical, ("dkey","dvalue"))
        require(row.get("candidate_recomputed_backward_calls") == row.get("expected_recomputed_backward_calls") == int(historical is not None), "Reconstruction dispatch count differs")
    if rectangles:
        require({(r["configuration"]["rows"],r["configuration"]["columns"]) for r in rectangles} == {(257,255),(513,511),(1024,1024),(2048,3)}, "Changed long-rectangle coverage")
    for row in rectangles:
        f3c.engineering_gradients(row["candidate_vs_f3c_control"], ("dkey","dvalue"))
        require(row.get("candidate_recomputed_backward_calls") == row.get("expected_recomputed_backward_calls") == 1, "Long rectangle did not execute recompute kernel")
    for row in blocks:
        c = row["configuration"]; length, prefix = c["length"], c["prefix"]
        require(c.get("primary_forward_backend") == c.get("primary_backward_backend") == "triton"
                and c.get("cast_weights_once") is True and c.get("control_backward_memory") == "materialized"
                and c.get("candidate_backward_memory") == "recompute", "Block comparison changed extra flags")
        require(row.get("primary_forward_and_cache_bitwise_equal") is True, "Backward change altered forward/cache")
        expected_calls = length-1+int(prefix>0)
        require(row.get("candidate_fused_forward_calls") == row.get("control_fused_forward_calls")
                == row.get("expected_fused_forward_calls") == expected_calls
                and row.get("candidate_recomputed_backward_calls") == row.get("expected_recomputed_backward_calls") == expected_calls
                and row.get("control_fused_backward_calls") == length-1
                and row.get("candidate_fused_backward_calls") == row.get("control_recomputed_backward_calls") == 0,
                "Block dispatch counts disagree")
        names = {"input","att_proj.weight","attn_out.weight","ff_proj.weight","ff_out.weight"}
        if prefix: names |= {"prefix_key","prefix_value"}
        primary = row["candidate_vs_f3c_control"]
        f3c.engineering_gradients(primary, names)
        require(primary.get("gradient_ownership_matches") is True and set(primary.get("outputs",{})) == {"hidden","key","value"}
                and all(r.get("bitwise_equal") is True and r.get("finite") is True for r in primary["outputs"].values()),
                "Primary output/gradient ownership differs")
        if length >= 65: no_full_shape(row["backward_shape_observer"])
    if memories:
        require({r["configuration"]["length"] for r in memories} == {512,1024,2048}, "Changed memory length coverage")
    for row in memories:
        for key in ("attention_vs_f3c_control","diagonal_vs_f3c_control"): output_check(row[key])
        control, candidate = row["control"], row["candidate"]
        no_full_shape(candidate["shape_observer"])
        require(control["shape_observer"].get("no_full_attention_shape") is False
                and control["shape_observer"].get("full_attention_outputs"), "Control lacked materialized attention observation")
        require(control.get("all_outputs_finite") is True and candidate.get("all_outputs_finite") is True, "Nonfinite memory probe")
        for arm in (control,candidate):
            peak = number(arm.get("peak_allocated_bytes"), "peak bytes", positive=True)
            before = number(arm.get("resident_before_bytes"), "resident bytes")
            require(peak-before == number(arm.get("peak_above_resident_bytes"), "incremental peak", positive=True), "Memory arithmetic differs")
        ratio = candidate["peak_above_resident_bytes"]/control["peak_above_resident_bytes"]
        require(ratio < 1 and math.isclose(number(row.get("peak_ratio_candidate_to_control"), "peak ratio"), ratio, rel_tol=1e-10),
                "Memory reduction claim contradicts observations")
    return {"row_count":len(rows),"block_count":len(blocks),"memory_count":len(memories),"long_rectangle_count":len(rectangles),
        "max_primary_block_global_gradient_relative_l2":max((r["candidate_vs_f3c_control"]["global_gradient_relative_l2"] for r in blocks),default=None),
        "isolated_reconstruction_memory":memories}


def summarize(runtime_dirs, *, project_root=ROOT, allow_incomplete=False):
    project_root = Path(project_root)
    summary = {"schema":"olmo-f3d-summary-v1","generated_utc":datetime.now(timezone.utc).isoformat(),"status":"passed",
        "runs":[],"full_steps":[],"correctness":[],"probes":[],"failed_diagnostics":[],"incomplete":[],
        "successful_f3d_optimizer_updates":{"eager":0,"graph":0,"total":0},
        "primary_reference":"F3c forward/backward fusion and cast reuse with materialized probability/error matrices",
        "scope":"Backward workspace implementation; no native model/loss/QK math changes and no quality or all-layer RT claim."}
    seen = set()
    for item in runtime_dirs:
        directory = Path(item).resolve(); path = directory/"report.json"
        require(directory.name not in seen,"Duplicate runtime name"); seen.add(directory.name)
        if not path.is_file():
            require(allow_incomplete and directory.is_dir(),"Missing runtime report")
            summary["incomplete"].append({"run":directory.name,"status":"no_report_yet"}); continue
        report = json.loads(path.read_text()); require(report.get("schema") in SCHEMAS,"Unknown F3d schema")
        kind,status = SCHEMAS[report["schema"]],report.get("status")
        if status == "running":
            require(allow_incomplete,"Active report requires explicit preview")
            summary["incomplete"].append({"run":directory.name,"status":status}); continue
        require(status in ("passed","failed","capture_blocked") and report.get("finished_utc"),"Report is not completed")
        checks = declared_checks(report)
        run = {"name":directory.name,"kind":kind,"status":status,"configuration":report.get("configuration",{}),
            "runtime":report.get("runtime"),"source_lineage":source_lineage(report,directory,project_root),
            "report_path":str(path),"report_sha256":digest(path),"checks":checks,
            "wandb_url":report.get("wandb",{}).get("run_url"),"used_for_performance":False}
        summary["runs"].append(run)
        if status != "passed":
            summary["failed_diagnostics"].append({"run":directory.name,"status":status,"error_type":report.get("error_type"),
                "error_message":report.get("error_message"),"declared_failed_checks":[c["name"] for c in checks if not c["passed"]],
                "excluded_partial_performance":bool(report.get("capacity") or any(c["record"].get("kind")=="isolated_reconstruction_memory" for c in checks))})
            continue
        configuration_check(report,kind)
        if kind == "native":
            require(report.get("checkpoint",{}).get("sha256") == prior.CHECKPOINT_SHA256,"Wrong native checkpoint")
            if report["stage"] == "correctness":
                summary["correctness"].append({"run":directory.name,"configuration":report["configuration"],**native_checks(report)})
            elif report["stage"] == "capacity":
                require({c["name"] for c in checks} == {"finite_complete_updates"},"Missing finite-update gate")
                summary["full_steps"].append(prior.full_step_record(report,directory.name)); run["used_for_performance"]=True
            else: raise ValueError("Unknown native stage")
            for name,updates in (("eager",3),("graph",3),("total",6)): summary["successful_f3d_optimizer_updates"][name]+=updates
        else:
            summary["probes"].append({"run":directory.name,**probe_checks(report)})
    require(summary["runs"] or summary["incomplete"],"No F3d runs supplied")
    if summary["incomplete"]: summary["status"]="partial_preview"
    elif summary["failed_diagnostics"]: summary["status"]="completed_with_failed_diagnostics"
    summary["declared_check_count"]=sum(len(r["checks"]) for r in summary["runs"])
    summary["declared_checks_passed"]=sum(c["passed"] for r in summary["runs"] for c in r["checks"])
    return summary


def markdown(summary):
    lines=["# F3d bounded RT backward workspace","",f"Status: **{summary['status']}**.","",
        "The primary control is F3c fused forward/backward with cast reuse. The candidate changes only backward memory policy. "
        "Native parameters, recurrence, RoPE, Q/K treatment and losses stay fixed. Full-model checks select RT layer0 only; "
        "quality, all-layer RT, multi-GPU and broader graph configurations remain untested here.","",
        "Primary gradient budgets: global relative L2≤1/64, per tensor≤1/32, maximum error/reference maximum≤1/16. "
        "Exact-zero references require exact zero. Same-candidate graph and full-Adam checks retain stricter requirements. "
        "Nested stricter/FP32 diagnostic failures remain in summary.json and original reports.","",
        "| Run | Scope | Status | Passed / declared |","| --- | --- | --- | ---: |"]
    for run in summary["runs"]:
        name=f"[{run['name']}]({run['wandb_url']})" if run.get("wandb_url") else run["name"]
        lines.append(f"| {name} | {run['kind']} | {run['status']} | {sum(c['passed'] for c in run['checks'])}/{len(run['checks'])} |")
    updates=summary["successful_f3d_optimizer_updates"]
    lines += ["",f"Successful native runs contain {updates['total']} physical optimizer updates ({updates['eager']} eager + "
        f"{updates['graph']} graph). Warmup/backward-only probes and failed attempts do not count as successful updates.","",
        "| Native check | B/T | Initial losses exact | Gradient relative L2 | Same-candidate graph exact | Full Adam exact |",
        "| --- | ---: | --- | ---: | --- | --- |"]
    for row in summary["correctness"]:
        c=row["configuration"]
        lines.append(f"| {row['run']} | {c['batch_size']}/{c['length']} | {row['forward_losses_bitwise_equal']} | "
            f"{row['global_gradient_relative_l2']} | {row['candidate_graph_checks_bitwise']} | {row['complete_updates_exact']} |")
    lines += ["","## Complete-update measurements","",
        "Wall time includes input validation/copy, graph forward/loss/backward, clipping, AdamW and scheduler. "
        "Peak allocated includes setup; reserved peak and current reserved are distinct. These are full-model measurements, "
        "separate from reconstruction-only workspace probes.","",
        "| Run | Policy | Case | B/T | Input tokens/s | Seconds/update | Allocated / reserved peak GiB | Current reserved GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    for row in summary["full_steps"]:
        lines.append(f"| {row['run']} | {row['variant']} | {row['case']} | {row['batch_size']}/{row['length']} | "
            f"{row['input_tokens_per_second']:,.0f} | {row['full_step']['median_wall_seconds']:.4f} | "
            f"{fmt(row['peak_allocated_gib'])} / {fmt(row['peak_reserved_gib'])} | {fmt(row['current_reserved_gib'])} |")
    lines += ["","## Isolated reconstruction workspace","",
        "The shape observer inspects Torch outputs/views, not device-private Triton scratch. Source review and allocated peaks "
        "complement it. These figures cover attention reconstruction, not total model memory.","",
        "| Probe | Length | Control incremental peak MiB | Candidate incremental peak MiB | Candidate/control |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for probe in summary["probes"]:
        for row in probe["isolated_reconstruction_memory"]:
            lines.append(f"| {probe['run']} | {row['configuration']['length']} | {row['control']['peak_above_resident_bytes']/2**20:.3f} | "
                f"{row['candidate']['peak_above_resident_bytes']/2**20:.3f} | {row['peak_ratio_candidate_to_control']:.4f} |")
    for failure in summary["failed_diagnostics"]:
        lines += ["",f"Failed diagnostic `{failure['run']}`: {failure.get('error_type')}: {failure.get('error_message')}. "
            "Its partial performance is excluded from successful results; raw records and source snapshots remain retained."]
    return "\n".join(lines)+"\n"


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True); paths=[]
    if summary["full_steps"]:
        rows=summary["full_steps"]; fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        labels=[f"{r['case']} B{r['batch_size']} T{r['length']}\n{r['variant']}" for r in rows]
        for axis,key,title in ((axes[0],"input_tokens_per_second","Complete-update input tokens/s"),(axes[1],"peak_allocated_gib","Full-model allocated peak GiB")):
            axis.bar(range(len(rows)),[r[key] for r in rows]); axis.set_xticks(range(len(rows)),labels,rotation=30,ha="right"); axis.set_title(title)
        for suffix in ("pdf","png"):
            path=output/f"full-update-throughput-memory.{suffix}";fig.savefig(path,dpi=180);paths.append(str(path))
        plt.close(fig)
    measurements=[(probe["run"],row) for probe in summary["probes"] for row in probe["isolated_reconstruction_memory"]]
    if measurements:
        fig,axis=plt.subplots(figsize=(7,4),layout="constrained")
        for run in dict.fromkeys(name for name,_ in measurements):
            rows=sorted((r for name,r in measurements if name==run),key=lambda r:r["configuration"]["length"])
            for arm in ("control","candidate"):
                axis.plot([r["configuration"]["length"] for r in rows],
                    [r[arm]["peak_above_resident_bytes"]/2**20 for r in rows],marker="o",label=f"{run}: {arm}")
        axis.set(xlabel="Sequence length",ylabel="Incremental allocated peak (MiB)",title="Isolated attention reconstruction; B2 H4 D64")
        axis.legend(fontsize=8);axis.grid(alpha=.25)
        for suffix in ("pdf","png"):
            path=output/f"isolated-reconstruction-memory.{suffix}";fig.savefig(path,dpi=180);paths.append(str(path))
        plt.close(fig)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args(argv);summary=summarize(args.runtime_dir,allow_incomplete=args.allow_incomplete)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/"summary.json",summary)
    (args.output_dir/"results.md").write_text(markdown(summary));plot(summary,args.output_dir)
    print(json.dumps({"status":summary["status"],"checks":summary["declared_check_count"]}))


if __name__=="__main__": main()
