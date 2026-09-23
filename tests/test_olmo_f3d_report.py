"""F3d summary gates preserve failed diagnostics and separate measurement scopes."""
import copy
import json
from pathlib import Path
import pytest
from scripts import olmo_f3d_report as reporter


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value)


def tensor_check():
    return {"passed":True,"finite":True,"bitwise_equal":True,"relative_l2":0.,"max_abs":0.,
        "reference_max_abs":1.,"relative_l2_limit":1e-5,"max_abs_limit":1.1e-5}


def output_check():
    return {"passed":True,"finite":True,"shape_matches":True,"bitwise_equal":True,"relative_l2":0.,
        "max_abs":0.,"reference_max_abs":1.,"relative_l2_limit":1/64,"max_error_reference_max_limit":1/16,
        "max_error_reference_max":0.}


def engineering(names):
    return {"passed":True,"global_gradient_relative_l2":0.,"global_gradient_relative_l2_limit":1/64,
        "gradients":{n:{**output_check(),"relative_l2_limit":1/32} for n in names}}


def full_step_data():
    return {"full_step":{"wall_seconds":[2.]*3,"cuda_seconds":[1.8]*3,"median_wall_seconds":2.,"median_cuda_seconds":1.8},
        "input_tokens_per_second":16384.,"peak_allocated_gib":40.,"peak_reserved_gib":64.,"current_reserved_gib":43.,
        "health":{"passed":True},"records":[{"update_completed":True,"counts":{"ce":16384,"latent":32704,"kl":16384},
            "counters":{"optimizer_updates":s,"input_tokens":s*32768}} for s in (4,5,6)]}


@pytest.fixture
def evidence(tmp_path):
    project,directory=tmp_path/"project",tmp_path/"runtime"/"capacity"
    write(project/"scripts/runtime.py","source");write(project/reporter.PROTOCOL,"protocol")
    report={"schema":"olmo-f3d-native-v1","status":"passed","finished_utc":"now","stage":"capacity",
        "configuration":{"case":"combined","stage":"capacity","variant":"recompute","batch_size":64,"length":512,
            "cast_weights_once":True,"forward_tile_backend":"triton","backward_tile_backend":"triton","backward_memory":"recompute"},
        "checkpoint":{"sha256":reporter.prior.CHECKPOINT_SHA256},
        "source_hashes":{"scripts/runtime.py":reporter.digest(project/"scripts/runtime.py")},
        "protocol_sha256":reporter.digest(project/reporter.PROTOCOL),
        "checks":[{"name":"finite_complete_updates","passed":True}],"capacity":full_step_data()}
    write(directory/"report.json",report)
    return project,directory,report


def summarize(evidence,**kwargs):
    project,directory,_=evidence
    return reporter.summarize([directory],project_root=project,**kwargs)


def test_capacity_counts_six_updates_and_scopes_memory(evidence,tmp_path):
    summary=summarize(evidence)
    assert summary["successful_f3d_optimizer_updates"]=={"eager":3,"graph":3,"total":6}
    assert summary["full_steps"][0]["input_tokens_per_second"]==16384.
    text=reporter.markdown(summary)
    assert "reconstruction-only workspace" in text and "RT layer0 only" in text
    assert all(Path(p).stat().st_size>1000 for p in reporter.plot(summary,tmp_path/"plots"))


@pytest.mark.parametrize("change",["forward","backward","memory","variant","stage","counter","tokens","health","timing","checkpoint"])
def test_capacity_cannot_override_contradictory_evidence(evidence,change):
    _,directory,r=evidence
    if change=="forward":r["configuration"]["forward_tile_backend"]="eager"
    if change=="backward":r["configuration"]["backward_tile_backend"]="eager"
    if change=="memory":r["configuration"]["backward_memory"]="materialized"
    if change=="variant":r["configuration"]["variant"]="triton"
    if change=="stage":r["stage"]="correctness"
    if change=="counter":r["capacity"]["records"][-1]["counters"]["optimizer_updates"]=5
    if change=="tokens":r["capacity"]["input_tokens_per_second"]*=2
    if change=="health":r["capacity"]["health"]["passed"]=False
    if change=="timing":r["capacity"]["full_step"]["median_wall_seconds"]=0
    if change=="checkpoint":r["checkpoint"]["sha256"]="a"*64
    write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)


def native_report(original):
    r=copy.deepcopy(original);r["stage"]=r["configuration"]["stage"]="correctness"
    r["configuration"]["batch_size"]=8
    row={"passed":True,"all_bitwise_equal":True,"ownership_matches":True,
        "gradients":{"weight":tensor_check()},"losses":{"pass0/ce":tensor_check()}}
    r["checks"]=[{**copy.deepcopy(row),"name":name} for name in sorted(reporter.prior.NATIVE_CHECKS-{"complete_adamw_update_parity"})]
    first=next(v for v in r["checks"] if v["name"]=="same_state_variant_vs_reference")
    first.update(forward_losses_bitwise_equal=True,dispatch_matches=True,reference_fused_backward_calls=0,
        candidate_fused_backward_calls=511,expected_candidate_fused_backward_calls=511,global_gradient_relative_l2=0.,
        mixed_gradient_screen={"weight":{"delta_sq":0.,"reference_sq":3.,"relative_l2":0.,"max_relative":0.}},
        mixed_loss_screen={"pass0/ce":{"relative_l2":0.,"max_relative":0.}})
    r["checks"].append({"name":"complete_adamw_update_parity","passed":True,"metrics_exact":True,
        "model_optimizer_scheduler_counters_exact":True,"weights_changed":True,"updates_per_arm":3,"physical_optimizer_updates":6,
        "arms":[{"replay":replay,"metrics":[{"update_completed":True} for _ in range(3)],
            "boundary":{"model":"same","optimizer":"same"},"health":{"passed":True}} for replay in (False,True)]})
    return r


@pytest.mark.parametrize("change",[None,"strict_only","loss","dispatch","global_sums","optimizer","nonfinite","zero_reference"])
def test_native_mixed_screen_and_full_update_parity(evidence,change):
    _,directory,original=evidence;r=native_report(original)
    row=next(v for v in r["checks"] if v["name"]=="same_state_variant_vs_reference")
    if change=="strict_only":row["gradients"]["weight"]["passed"]=False
    if change=="loss":row["forward_losses_bitwise_equal"]=False
    if change=="dispatch":row["candidate_fused_backward_calls"]=0
    if change=="global_sums":row["mixed_gradient_screen"]["weight"]["delta_sq"]=.0001
    if change=="optimizer":r["checks"][-1]["arms"][1]["boundary"]["optimizer"]="different"
    if change=="nonfinite":row["gradients"]["weight"]["finite"]=False
    if change=="zero_reference":row["gradients"]["weight"].update(reference_max_abs=0.,max_abs=1e-40)
    write(directory/"report.json",r)
    if change not in (None,"strict_only"):
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        assert summary["correctness"][0]["observed_fused_backward_calls"]==511
        if change: assert next(c for c in summary["runs"][0]["checks"] if c["name"] == "same_state_variant_vs_reference")["nested_diagnostic_failures"]


def memory_report(original):
    r=copy.deepcopy(original);r.update(schema="olmo-f3d-probe-v1",stage="complete",configuration={"stage":"memory",
        "primary_forward_backend":"triton","primary_backward_backend":"triton","primary_cast_weights_once":True,
        "control_backward_memory":"materialized","candidate_backward_memory":"recompute"},checks=[])
    r.pop("checkpoint")
    for length in (512,1024,2048):
        control={"resident_before_bytes":100,"peak_allocated_bytes":1100,"peak_above_resident_bytes":1000,"all_outputs_finite":True,
            "shape_observer":{"no_full_attention_shape":False,"full_attention_outputs":[{"shape":[length,length]}],"tensor_outputs":10}}
        candidate={"resident_before_bytes":100,"peak_allocated_bytes":200,"peak_above_resident_bytes":100,"all_outputs_finite":True,
            "shape_observer":{"no_full_attention_shape":True,"full_attention_outputs":[],"tensor_outputs":10}}
        r["checks"].append({"name":f"memory{length}","kind":"isolated_reconstruction_memory","passed":True,
            "configuration":{"length":length},"control":control,"candidate":candidate,"peak_ratio_candidate_to_control":.1,
            "attention_vs_f3c_control":output_check(),"diagonal_vs_f3c_control":output_check(),
            "strict_diagnostic":{"passed":False}})
    return r


@pytest.mark.parametrize("change",[None,"full_shape","empty_observer","arithmetic","ratio","output","missing_length","kind"])
def test_memory_scope_enforces_measured_arithmetic_and_keeps_strict_flags(evidence,change):
    _,directory,original=evidence;r=memory_report(original);row=r["checks"][0]
    if change=="full_shape":row["candidate"]["shape_observer"]["full_attention_outputs"]=[{}]
    if change=="empty_observer":row["candidate"]["shape_observer"]["tensor_outputs"]=0
    if change=="arithmetic":row["candidate"]["peak_allocated_bytes"]=201
    if change=="ratio":row["peak_ratio_candidate_to_control"]=.2
    if change=="output":row["attention_vs_f3c_control"]["relative_l2_limit"]=.5
    if change=="missing_length":r["checks"].pop()
    if change=="kind":row["kind"]="other"
    write(directory/"report.json",r)
    if change:
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        assert summary["probes"][0]["memory_count"]==3 and summary["successful_f3d_optimizer_updates"]["total"]==0
        assert summary["runs"][0]["checks"][0]["nested_diagnostic_failures"]


def test_failed_attempt_keeps_partial_checks_but_excludes_all_timing(evidence):
    _,directory,r=evidence;r.update(status="failed",error_type="RuntimeError",error_message="capture failed")
    write(directory/"report.json",r);summary=summarize(evidence)
    assert summary["status"]=="completed_with_failed_diagnostics"
    assert not summary["full_steps"] and not summary["correctness"]
    assert summary["successful_f3d_optimizer_updates"]["total"]==0
    assert summary["failed_diagnostics"][0]["excluded_partial_performance"]


def test_incomplete_requires_preview(evidence):
    _,directory,r=evidence;r["status"]="running";r.pop("finished_utc");write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)
    assert summarize(evidence,allow_incomplete=True)["status"]=="partial_preview"


def test_corrupt_historical_snapshot_rejected_even_if_current_source_matches(evidence):
    _,directory,_=evidence;write(directory/"source-snapshot/scripts/runtime.py","corrupt")
    with pytest.raises(ValueError):summarize(evidence)


def block_report(original):
    r=memory_report(original);r["configuration"]["stage"]="blocks";r["checks"]=[]
    cases=[(t,p,a) for t in (9,17) for p in (0,3) for a in (0.,.37,1.)]+[(65,3,1.),(129,3,1.)]
    for length,prefix,alpha in cases:
        names={"input","att_proj.weight","attn_out.weight","ff_proj.weight","ff_out.weight"}
        if prefix:names|={"prefix_key","prefix_value"}
        primary=engineering(names);primary.update(gradient_ownership_matches=True,
            outputs={n:output_check() for n in ("hidden","key","value")})
        count=length-1+int(prefix>0)
        row={"name":f"block{length}-{prefix}-{alpha}","kind":"native_tiny_block","passed":True,
            "configuration":{"length":length,"prefix":prefix,"alpha":alpha,"primary_forward_backend":"triton",
                "primary_backward_backend":"triton","cast_weights_once":True,"control_backward_memory":"materialized","candidate_backward_memory":"recompute"},
            "candidate_vs_f3c_control":primary,"primary_forward_and_cache_bitwise_equal":True,
            "candidate_fused_forward_calls":count,"control_fused_forward_calls":count,"expected_fused_forward_calls":count,
            "candidate_recomputed_backward_calls":count,"expected_recomputed_backward_calls":count,
            "control_fused_backward_calls":length-1,"candidate_fused_backward_calls":0,"control_recomputed_backward_calls":0,
            "candidate_vs_full_fp32":{"passed":False,"global_gradient_relative_l2":.03}}
        if length>=65:row["backward_shape_observer"]={"no_full_attention_shape":True,"full_attention_outputs":[],"tensor_outputs":200}
        r["checks"].append(row)
    return r


@pytest.mark.parametrize("change",[None,"dispatch","inventory","forward","global","tensor","zero","geometry","workspace"])
def test_block_primary_gates_do_not_promote_secondary_fp32_control(evidence,change):
    _,directory,original=evidence;r=block_report(original);row=r["checks"][0]
    if change=="dispatch":row["candidate_recomputed_backward_calls"]-=1
    if change=="inventory":row["candidate_vs_f3c_control"]["gradients"].pop("input")
    if change=="forward":row["candidate_vs_f3c_control"]["outputs"]["hidden"]["bitwise_equal"]=False
    if change=="global":row["candidate_vs_f3c_control"]["global_gradient_relative_l2"]=.02
    if change=="tensor":row["candidate_vs_f3c_control"]["gradients"]["input"]["relative_l2_limit"]=.5
    if change=="zero":row["candidate_vs_f3c_control"]["gradients"]["input"].update(reference_max_abs=0.,max_abs=1e-40)
    if change=="geometry":row["configuration"]["alpha"]=.4
    if change=="workspace":r["checks"][-1]["backward_shape_observer"]["full_attention_outputs"]=[{}]
    write(directory/"report.json",r)
    if change:
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        assert summary["probes"][0]["block_count"]==14
        assert summary["runs"][0]["checks"][0]["nested_diagnostic_failures"]


def rows_report(original):
    r=memory_report(original);r["configuration"]["stage"]="rows";r["checks"]=[]
    for length in (1,9,17,33,65,129):
        for prefix in (0,3):
            for variant in ("normal","all_masked","strided_masked","large_scores"):
                history=length!=1 or prefix!=0
                r["checks"].append({"name":f"row{length}-{prefix}-{variant}","kind":"frozen_reconstruction","passed":True,
                    "configuration":{"length":length,"prefix":prefix,"variant":variant},
                    "attention_vs_f3c_control":output_check(),"diagonal_vs_f3c_control":output_check(),
                    "historical_gradients_vs_f3c_control":engineering(("dkey","dvalue")) if history else None,
                    "maximum_matches_empty_pattern":True,"finite_denominator_diagonal_attention":True,
                    "candidate_recomputed_backward_calls":int(history),"expected_recomputed_backward_calls":int(history)})
    for rows,columns in ((257,255),(513,511),(1024,1024),(2048,3)):
        r["checks"].append({"name":f"long{rows}-{columns}","kind":"frozen_long_history","passed":True,
            "configuration":{"rows":rows,"columns":columns},"candidate_vs_f3c_control":engineering(("dkey","dvalue")),
            "candidate_recomputed_backward_calls":1,"expected_recomputed_backward_calls":1})
    return r


@pytest.mark.parametrize("change",[None,"row_dispatch","missing_history","long_dispatch","missing_case","geometry"])
def test_frozen_rows_and_long_rectangles_have_complete_geometry_and_dispatch(evidence,change):
    _,directory,original=evidence;r=rows_report(original)
    if change=="row_dispatch":r["checks"][0]["candidate_recomputed_backward_calls"]=1
    if change=="missing_history":r["checks"][4]["historical_gradients_vs_f3c_control"]=None
    if change=="long_dispatch":r["checks"][-1]["candidate_recomputed_backward_calls"]=0
    if change=="missing_case":r["checks"].pop()
    if change=="geometry":r["checks"][-1]["configuration"]["rows"]=2047
    write(directory/"report.json",r)
    if change:
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        assert summary["probes"][0]["row_count"]==48 and summary["probes"][0]["long_rectangle_count"]==4
