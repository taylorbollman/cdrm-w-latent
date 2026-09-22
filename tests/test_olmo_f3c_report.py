"""F3c reports enforce isolated backward comparison and keep diagnostics visible."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import olmo_f3c_report as reporter
from scripts.olmo_f3c_tile_probe import block_comparison


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)


def timing(seconds):
    return {"wall_seconds": [seconds]*3, "cuda_seconds": [seconds*.9]*3,
        "median_wall_seconds": seconds, "median_cuda_seconds": seconds*.9}


def tensor_check():
    return {"passed": True, "finite": True, "bitwise_equal": True, "relative_l2": 0.,
        "max_abs": 0., "reference_max_abs": 1., "relative_l2_limit": 1e-5, "max_abs_limit": 1.1e-5}


def full_step_data():
    return {"full_step": timing(2.), "input_tokens_per_second": 16384., "peak_allocated_gib": 40.,
        "peak_reserved_gib": 64., "current_reserved_gib": 43., "health": {"passed": True},
        "records": [{"update_completed": True, "counts": {"ce": 16384, "latent": 32704, "kl": 16384},
            "counters": {"optimizer_updates": step, "input_tokens": step*32768}} for step in (4,5,6)]}


def profile_event(name, device="CUDA", count=2, microseconds=100):
    return {"name": name, "device_type": "DeviceType."+device, "count": count,
        "self_device_us": microseconds, "device_total_us": microseconds, "cpu_total_us": 0, "self_cpu_us": 0}


@pytest.fixture
def evidence(tmp_path):
    project, directory = tmp_path/"project", tmp_path/"runtime"/"profile"
    write(project/"scripts/runtime.py", "frozen source")
    write(project/reporter.PROTOCOL, "frozen protocol")
    expected = {"RT/local_writer_vjp":512,"RT/local_finish_vjp":512,"RT/batched_parameter_vjp":1,
        "RT/attention_reconstruction":1,"RT/reverse_historical_tile":511,"RT/final_attention_matmul":4}
    report = {"schema":"olmo-f3c-profile-v1","status":"passed","finished_utc":"now",
        "configuration":{"case":"combined","variant":"reference","batch_size":64,"length":512,
            "cast_weights_once":True,"tile_backend":"triton","backward_tile_backend":"eager"},
        "checkpoint":{"sha256":reporter.prior.CHECKPOINT_SHA256},
        "source_hashes":{"scripts/runtime.py":reporter.digest(project/"scripts/runtime.py")},
        "protocol_sha256":reporter.digest(project/reporter.PROTOCOL),
        "checks":[{"name":"observer_neutrality","passed":True,"ownership_matches":True,"loss_keys_match":True,
                    "nonidentical_losses":[],"nonidentical_gradients":[]},
                  {"name":"expected_backward_annotations","passed":True,"expected":expected,"actual":copy.deepcopy(expected)},
                  {"name":"finite_complete_updates","passed":True}],
        "annotation_calls":expected,"completed_optimizer_updates":6,
        "optimizer_update_scope":{"eager":3,"graph":3,"total":6},
        "graph_profile":[profile_event("gemm",count=7,microseconds=900),profile_event("RT/local_writer_vjp",count=512,microseconds=10000)],
        "annotated_eager_profile":[profile_event("RT/local_writer_vjp"),profile_event("RT/local_writer_vjp",device="CPU")],
        **full_step_data()}
    write(directory/"report.json",report)
    return project,directory,report


def summarize(evidence, **kwargs):
    project,directory,_=evidence
    return reporter.summarize([directory],project_root=project,**kwargs)


def test_profile_counts_updates_neutrality_and_kernel_scope(evidence):
    summary=summarize(evidence)
    assert summary["status"]=="passed"
    assert summary["successful_f3c_optimizer_updates"]=={"eager":3,"graph":3,"total":6}
    assert summary["profiles"][0]["graph_kernel_invocations"]==7
    assert summary["profiles"][0]["graph_kernel_self_device_us"]==900
    assert len(summary["profiles"][0]["annotated_eager_ranges"])==2
    assert summary["full_steps"][0]["variant"]=="reference"
    text=reporter.markdown(summary)
    assert "F3b fused historical forward" in text
    assert "does not remove quadratic backward storage" in text
    assert "host submission gaps" in text


@pytest.mark.parametrize("change",["observer","annotation","update_count","forward_flag","backward_flag","timing","checkpoint","gate"])
def test_passing_profile_cannot_override_contradiction(evidence,change):
    _,directory,data=evidence
    if change=="observer":data["checks"][0]["nonidentical_gradients"]=["weight"]
    elif change=="annotation":data["checks"][1]["actual"]["RT/local_writer_vjp"]-=1
    elif change=="update_count":data["completed_optimizer_updates"]=7
    elif change=="forward_flag":data["configuration"]["tile_backend"]="eager"
    elif change=="backward_flag":data["configuration"]["backward_tile_backend"]="triton"
    elif change=="timing":data["input_tokens_per_second"]=1
    elif change=="checkpoint":data["checkpoint"]["sha256"]="0"*64
    elif change=="gate":data["checks"][0]["passed"]=False
    write(directory/"report.json",data)
    with pytest.raises(ValueError):summarize(evidence)


def test_failed_profile_excludes_partial_speed_and_updates(evidence,tmp_path):
    _,directory,data=evidence
    data.update(status="failed",error_type="RuntimeError",error_message="bad candidate",stage="profile")
    data["input_tokens_per_second"]=float("nan")
    write(directory/"report.json",data)
    summary=summarize(evidence)
    assert summary["status"]=="completed_with_failed_diagnostics"
    assert not summary["full_steps"] and not summary["profiles"]
    assert summary["successful_f3c_optimizer_updates"]["total"]==0
    assert summary["failed_diagnostics"][0]["excluded_partial_performance"]
    write(tmp_path/"throughput.png","stale")
    assert reporter.plot(summary,tmp_path)==[] and not (tmp_path/"throughput.png").exists()


def test_historical_sources_and_protocol_use_exact_snapshots(evidence):
    project,directory,_=evidence
    write(directory/"source-snapshot/scripts/runtime.py","frozen source")
    write(directory/"protocol.md","frozen protocol")
    write(project/"scripts/runtime.py","new source")
    write(project/reporter.PROTOCOL,"new protocol")
    summary=summarize(evidence)
    assert not summary["runs"][0]["source_lineage"]["sources"]["scripts/runtime.py"]["current_matches"]
    write(directory/"protocol.md","corrupt protocol")
    with pytest.raises(ValueError,match="Corrupt"):
        summarize(evidence)


def blocks_report(data):
    result=copy.deepcopy(data)
    result.update(schema="olmo-f3c-tile-probe-v1",configuration={"stage":"blocks","primary_forward_backend":"triton",
        "primary_cast_weights_once":True,"control_backward_backend":"eager","candidate_backward_backend":"triton"})
    result["checks"]=[]
    for length in (9,17):
        for prefix in (0,3):
            for alpha in (0.,.37,1.):
                gradients={name:torch.ones(3) for name in ("input","att_proj.weight","attn_out.weight","ff_proj.weight","ff_out.weight")}
                if prefix:gradients.update(prefix_key=torch.ones(3),prefix_value=torch.ones(3))
                tensors={"outputs":{name:torch.ones(3) for name in ("hidden","key","value")},"gradients":gradients}
                primary=block_comparison(tensors,tensors)
                result["checks"].append({"name":f"block{length}-{prefix}-{alpha}","kind":"native_tiny_block","passed":True,
                    "configuration":{"length":length,"prefix":prefix,"alpha":alpha,"primary_forward_backend":"triton","cast_weights_once":True,
                        "candidate_backward_backend":"triton","control_backward_backend":"eager"},
                    "candidate_vs_f3b_control":primary,
                    "candidate_vs_full_fp32":{"passed":False,"global_gradient_relative_l2":.03},
                    "primary_forward_and_cache_bitwise_equal":True,
                    "candidate_fused_forward_calls":length-1+int(prefix>0),"control_fused_forward_calls":length-1+int(prefix>0),
                    "expected_fused_forward_calls":length-1+int(prefix>0),
                    "candidate_fused_backward_calls":length-1,"expected_fused_backward_calls":length-1,"control_fused_backward_calls":0})
    return result


def test_secondary_fp32_flags_remain_visible_without_overriding_primary(evidence):
    _,directory,data=evidence
    write(directory/"report.json",blocks_report(data))
    summary=summarize(evidence)
    assert summary["status"]=="passed" and summary["declared_checks_passed"]==12
    assert summary["tile_probes"][0]["block_count"]==12
    assert summary["tile_probes"][0]["max_fp32_diagnostic_block_global_gradient_relative_l2"]==.03
    assert summary["runs"][0]["checks"][0]["nested_diagnostic_failures"]


@pytest.mark.parametrize("change",["forward","dispatch","per_tensor","global","zero","geometry"])
def test_primary_blocks_enforce_exact_forward_dispatch_and_fixed_gradient_gates(evidence,change):
    _,directory,original=evidence
    data=blocks_report(original);row=data["checks"][0]
    if change=="forward":row["primary_forward_and_cache_bitwise_equal"]=False
    elif change=="dispatch":row["candidate_fused_backward_calls"]=0
    elif change=="per_tensor":row["candidate_vs_f3b_control"]["gradients"]["input"]["relative_l2_limit"]=.5
    elif change=="global":row["candidate_vs_f3b_control"]["global_gradient_relative_l2"]=.02
    elif change=="zero":row["candidate_vs_f3b_control"]["gradients"]["input"].update(reference_max_abs=0.,max_abs=1e-40)
    elif change=="geometry":row["configuration"]["alpha"]=.123
    write(directory/"report.json",data)
    with pytest.raises(ValueError):summarize(evidence)


def native_report(data):
    result=copy.deepcopy(data)
    result.update(schema="olmo-f3c-native-v1",configuration={"case":"combined","variant":"triton","stage":"correctness",
        "batch_size":8,"length":512,"cast_weights_once":True,"forward_tile_backend":"triton","backward_tile_backend":"triton"})
    row={"passed":True,"all_bitwise_equal":True,"ownership_matches":True,
        "gradients":{"weight":tensor_check()},"losses":{"pass0/ce":tensor_check()}}
    result["checks"]=[{**copy.deepcopy(row),"name":name} for name in sorted(reporter.prior.NATIVE_CHECKS-{"complete_adamw_update_parity"})]
    first=next(r for r in result["checks"] if r["name"]=="same_state_variant_vs_reference")
    first.update(forward_losses_bitwise_equal=True,dispatch_matches=True,reference_fused_backward_calls=0,
        candidate_fused_backward_calls=511,expected_candidate_fused_backward_calls=511,global_gradient_relative_l2=0.,
        mixed_gradient_screen={"weight":{"delta_sq":0.,"reference_sq":3.,"relative_l2":0.,"max_relative":0.}},
        mixed_loss_screen={"pass0/ce":{"relative_l2":0.,"max_relative":0.}})
    result["checks"].append({"name":"complete_adamw_update_parity","passed":True,"metrics_exact":True,
        "model_optimizer_scheduler_counters_exact":True,"weights_changed":True,"updates_per_arm":3,"physical_optimizer_updates":6,
        "arms":[{"replay":replay,"metrics":[{"update_completed":True} for _ in range(3)],
            "boundary":{"model":"same","optimizer":"same"},"health":{"passed":True}} for replay in (False,True)]})
    return result


@pytest.mark.parametrize("change",[None,"loss","fused_calls","global_sums","optimizer","nonfinite","zero_reference"])
def test_native_checks_verify_forward_dispatch_global_math_and_exact_updates(evidence,change):
    _,directory,original=evidence
    data=native_report(original)
    row=next(r for r in data["checks"] if r["name"]=="same_state_variant_vs_reference")
    if change=="loss":row["forward_losses_bitwise_equal"]=False
    elif change=="fused_calls":row["candidate_fused_backward_calls"]=0
    elif change=="global_sums":row["mixed_gradient_screen"]["weight"]["delta_sq"]=.0001
    elif change=="optimizer":data["checks"][-1]["arms"][1]["boundary"]["optimizer"]="different"
    elif change=="nonfinite":row["gradients"]["weight"]["finite"]=False
    elif change=="zero_reference":row["gradients"]["weight"].update(reference_max_abs=0.,max_abs=1e-40)
    write(directory/"report.json",data)
    if change:
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        assert summary["correctness"][0]["observed_fused_backward_calls"]==511
        assert summary["correctness"][0]["forward_losses_bitwise_equal"]
        assert summary["successful_f3c_optimizer_updates"]["total"]==6


def test_incomplete_runs_require_preview_and_never_create_performance(evidence):
    _,directory,data=evidence
    data.update(status="running");data.pop("finished_utc")
    write(directory/"report.json",data)
    with pytest.raises(ValueError):summarize(evidence)
    summary=summarize(evidence,allow_incomplete=True)
    assert summary["status"]=="partial_preview" and not summary["full_steps"]


def test_plot_exports_separate_full_step_and_helper_scopes(evidence,tmp_path):
    summary=summarize(evidence)
    paths=reporter.plot(summary,tmp_path/"output")
    assert len(paths)==2 and all(Path(p).stat().st_size>1000 for p in paths)


@pytest.mark.parametrize("wrong_variant",[False,True])
def test_historical_f3b_baseline_is_explicit_and_excluded_from_f3c_update_counts(evidence,wrong_variant):
    project,directory,data=evidence
    historical=copy.deepcopy(data)
    historical.update(schema="olmo-f3b-native-v1",configuration={"case":"rt","variant":"reference" if wrong_variant else "triton",
        "stage":"capacity","batch_size":128,"length":512},checks=[{"name":"finite_complete_updates","passed":True}])
    historical["capacity"]=full_step_data()
    historical["capacity"]["input_tokens_per_second"]=32768.
    for record in historical["capacity"]["records"]:
        step=record["counters"]["optimizer_updates"]
        record["counters"]["input_tokens"]=step*65536
        record["counts"]={"ce":32768,"latent":0,"kl":0}
    write(project/reporter.prior.PROTOCOL,"frozen F3b protocol")
    historical["protocol_sha256"]=reporter.digest(project/reporter.prior.PROTOCOL)
    baseline=directory.parent/"historical-f3b-rt"
    write(baseline/"report.json",historical)
    if wrong_variant:
        with pytest.raises(ValueError,match="Historical comparison"):
            summarize(evidence,f3b_baselines=[baseline])
    else:
        summary=summarize(evidence,f3b_baselines=[baseline])
        assert len(summary["runs"])==1 and len(summary["historical_f3b_lineage"])==1
        assert summary["successful_f3c_optimizer_updates"]["total"]==6
        assert len(summary["full_steps"])==2
        assert summary["full_steps"][-1]["historical"] is True
        assert summary["full_steps"][-1]["variant"]=="F3b reference"
