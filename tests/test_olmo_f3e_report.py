"""Multi-layer evidence rejects contradictory dispatch, resource and parity claims."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from scripts import olmo_f3e_report as reporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict,list)) else value)
    return path


def tensor_check():
    return {"passed":True, "finite":True, "bitwise_equal":True, "relative_l2":0., "max_abs":0.,
            "reference_max_abs":1., "relative_l2_limit":1e-5, "max_abs_limit":1.1e-5}


def metric_fixture(config, counts, step):
    weights={"ce":1.,"latent":float(config["case"] != "rt"),"kl":float(config["case"] != "rt")}
    means={key:1. if count else 0. for key,count in counts.items()}
    return {"schema":"olmo-lm-optimizer-step-v1","update_completed":True,"counts":copy.deepcopy(counts),
        "loss_means":means,"loss_sums":{key:count*means[key] for key,count in counts.items()},
        "objective_weights":weights,"objective":sum(means[key]*weights[key] for key in counts),
        "gradient_norm_before_clip":1.,"max_grad_norm":1.,"lr_used":[1e-5],"lr_next":[1e-5],
        "counters":{"optimizer_updates":step,"microbatches":step,"documents":step*config["batch_size"],
            "input_tokens":step*config["batch_size"]*config["length"],"ce_positions":step*counts["ce"],
            "latent_pairs":step*counts["latent"],"kl_triples":step*counts["kl"]}}


def make_report(*, stage="capacity", layout="spread2", case="combined", length=512, batch=8):
    layers=reporter.LAYOUTS[layout]; enabled=case != "rt"
    passes=3 if case == "combined-k3" else 2
    mode=reporter.FBTMode(enabled=enabled, num_passes=passes if enabled else 1,
                          rt_mode=reporter.RTMode(tuple(layers)))
    config={"case":case,"layout":layout,"stage":stage,"variant":"recompute","batch_size":batch,"length":length,
        "selected_rt_layers":layers,"layout_role":"optional_all_layer_stress" if layout == "all16" else
            "single_layer_reference" if layout == "single" else "primary_multi_layer_integration",
        "mode":json.loads(json.dumps(asdict(mode))),"rt_block_calls_per_forward_backward":len(layers)*(passes-1 if enabled else 1),
        "cast_weights_once":True,"forward_tile_backend":"triton","backward_tile_backend":"triton","backward_memory":"recompute",
        "ordinary_activation_checkpointing":True,"ordinary_attention_backend":"deterministic_flash",
        "precision":"bf16_mixed","tf32":False,"autocast_weight_cache":False,
        "case_specification":{"name":case,"fbt":enabled,"nextlat":enabled,"rt_layers":layers,"passes":passes,
            "alpha":1.,"beta":1.,"batch_size":batch,"length":length,"updates":3,"transition":False,"resume":False,"profile":False}}
    work=reporter.LossWork(batch*(length//2),batch*(length-1) if enabled else 0,
                          batch*(length//2) if enabled else 0,batch*(length-1) if enabled else 0)
    native=reporter.OLMoConfig.native_1b(); auxiliary=reporter.NextLatConfig(native.model_dim) if enabled else None
    estimate=reporter.estimate_training_resources(native,batch_size=batch,sequence_length=length,mode=mode,
        nextlat=auxiliary,loss_work=work,ordinary_checkpointing=True,backward_memory="recompute").to_dict()
    active=estimate["parameter_counts"]["training_architecture"]
    resident=reporter.architecture_parameter_counts(native,fbt=True,nextlat=auxiliary)["training_architecture"]
    shapes=reporter.parameter_shapes(enabled)
    executed=sorted(n for n in shapes if enabled or not n.startswith("backbone.fusion."))
    resource={"analytic_matrix_work":estimate,"loss_work":asdict(work),"loss_weights":{"ce":1.,"latent":float(enabled),"kl":float(enabled)},
        "observed_parameters":{"registered_unique":resident,"resident_parameter_bytes":resident*4,"trainable":active,
            "gradient_participating":active,"optimizer_owned":active if stage == "capacity" else None,"executed_declared":active,
            "deployable_inference_declared":estimate["parameter_counts"]["deployable_inference"]},
        "named_parameter_shapes":shapes,"declared_execution_names":executed,
        "declared_inference_names":[n for n in executed if not n.startswith("predictor.")],"scope":"matrix estimate"}
    result={"schema":reporter.SCHEMA,"status":"passed","finished_utc":"now","stage":stage,"configuration":config,
        "checkpoint":{"sha256":reporter.prior.CHECKPOINT_SHA256},"resources":resource,
        "wandb":{"status":"synced","run_url":"https://wandb.ai/taylorbollman/test/runs/123"},
        "backward_preparation":{"warmup":11,"capture":1,"replay":7 if stage == "correctness" else 3}}
    counts={key:estimate["objective_positions_per_update"][key] for key in ("ce","latent","kl")}
    if stage == "capacity":
        result["checks"]=[{"name":"finite_complete_updates","passed":True}]
        result["capacity"]={"full_step":{"wall_seconds":[2.]*3,"cuda_seconds":[1.8]*3,"median_wall_seconds":2.,"median_cuda_seconds":1.8},
            "input_tokens_per_second":batch*length/2,"ce_targets_per_second":work.ce_targets/2,
            "estimated_matrix_tflops_per_second_minimum":estimate["matrix_flops_minimum"]/2e12,
            "estimated_matrix_tflops_per_second_maximum":estimate["matrix_flops_maximum"]/2e12,
            "peak_allocated_gib":40.,"peak_reserved_gib":64.,"current_reserved_gib":43.,"health":{"passed":True,"nonfinite_parameters":[],"nonfinite_optimizer_tensors":[]},
            "warmup_updates":3,"timed_updates":3,"physical_optimizer_updates":6,"backward_only_warmup":10,
            "records":[metric_fixture(config,counts,i) for i in (4,5,6)]}
        return result
    losses={f"pass{p}/{term}":tensor_check() for p in range(mode.num_passes if enabled else 1) for term in ("ce","latent","kl")}
    row={"passed":True,"all_bitwise_equal":True,"ownership_matches":True,
         "gradients":{n:tensor_check() for n in executed},"losses":losses}
    result["checks"]=[{**copy.deepcopy(row),"name":name} for name in sorted(reporter.prior.NATIVE_CHECKS-{"complete_adamw_update_parity"})]
    first=next(v for v in result["checks"] if v["name"] == "same_state_variant_vs_reference")
    eligible=sum((i & -i)<=256 for i in range(1,length)); extra=passes-1 if enabled else 1
    dispatch={}
    for arm in ("reference","candidate"):
        by_layer={str(i):{"forward_blocks":extra,"forward_tiles":extra*(length-1),"forward_fused_tiles":extra*eligible,
            "forward_eager_tiles":extra*(length-1-eligible),"backward_blocks":extra,
            "recompute_tiles":extra*(length-1) if arm == "candidate" else 0,
            "materialized_tiles":extra*(length-1) if arm == "reference" else 0,
            "materialized_fused_tiles":extra*eligible if arm == "reference" else 0} for i in layers}
        dispatch[arm]={"by_layer":by_layer,"totals":{key:sum(v[key] for v in by_layer.values()) for key in next(iter(by_layer.values()))}}
        first[arm+"_dispatch"]=copy.deepcopy(dispatch[arm]);first["expected_"+arm+"_dispatch"]=copy.deepcopy(dispatch[arm])
    first.update(forward_losses_bitwise_equal=True,dispatch_matches=True,reference_recompute_backward_calls=0,
        candidate_recompute_backward_calls=(length-1)*len(layers)*extra,
        expected_candidate_recompute_backward_calls=(length-1)*len(layers)*extra,global_gradient_relative_l2=0.,
        mixed_gradient_screen={n:{"delta_sq":0.,"reference_sq":3.,"relative_l2":0.,"max_relative":0.} for n in executed},
        mixed_loss_screen={n:{"relative_l2":0.,"max_relative":0.} for n in losses})
    result["checks"].append({"name":"complete_adamw_update_parity","passed":True,"metrics_exact":True,
        "model_optimizer_scheduler_counters_exact":True,"weights_changed":True,"updates_per_arm":3,"physical_optimizer_updates":6,
        "arms":[{"replay":replay,"metrics":[metric_fixture(config,counts,i) for i in (1,2,3)],
                 "boundary":{"model":{"tensor":"digest"},"optimizer":{"moment":"digest"},"scheduler":{"step":3},
                             "counters":metric_fixture(config,counts,3)["counters"]},"health":{"passed":True,"nonfinite_parameters":[],"nonfinite_optimizer_tensors":[]}} for replay in (False,True)]})
    return result


@pytest.fixture
def evidence(tmp_path):
    project,directory=tmp_path/"project",tmp_path/"runtime"/"native"
    write(project/"scripts/runtime.py","source");write(project/reporter.PROTOCOL,"protocol")
    report=make_report()
    for name in reporter.RUNTIME_SOURCES:write(project/name,"source "+name)
    report.update(source_hashes={name:reporter.digest(project/name) for name in (*reporter.RUNTIME_SOURCES,"scripts/runtime.py")},
                  protocol_sha256=reporter.digest(project/reporter.PROTOCOL))
    write(directory/"report.json",report)
    return project,directory,report


def summarize(evidence,**kwargs):
    project,directory,_=evidence
    return reporter.summarize([directory],project_root=project,**kwargs)


def install(evidence,report):
    _,directory,original=evidence
    report.update({key:original[key] for key in ("source_hashes","protocol_sha256")})
    write(directory/"report.json",report)


def test_capacity_scope_parameters_and_plot(evidence,tmp_path):
    summary=summarize(evidence)
    assert summary["successful_f3e_optimizer_updates"] == {"eager":3,"graph":3,"total":6}
    assert summary["capability_ledger"][0]["selected_rt_layers"] == [0,15]
    assert summary["resource_cards"][0]["observed_parameters"]["registered_unique"] == 1267879936
    text=reporter.markdown(summary)
    assert "optional stress" in text and "not hardware utilization" in text
    assert all(Path(p).stat().st_size>1000 for p in reporter.plot(summary,tmp_path/"plots"))


@pytest.mark.parametrize("layout,case,length",[("single","combined",512),("adjacent2","combined",512),
    ("spread2","combined",2048),("spread2","combined-k3",32),("spread4","combined",512),("all16","combined",32),("spread2","rt",512)])
def test_native_layer_pass_and_fallback_multiplicity(evidence,layout,case,length):
    report=make_report(stage="correctness",layout=layout,case=case,length=length); install(evidence,report)
    summary=summarize(evidence); actual=summary["correctness"][0]
    assert actual["observed_recompute_backward_calls"] == (length-1)*len(reporter.LAYOUTS[layout])*(2 if case == "combined-k3" else 1)
    assert actual["complete_updates_exact"]
    if length == 2048: assert actual["dispatch"]["candidate"]["totals"]["forward_eager_tiles"] == 6
    if case == "rt":
        inv=summary["resource_cards"][0]["observed_parameters"]
        assert inv["registered_unique"]-inv["trainable"] == 8388608


@pytest.mark.parametrize("damage",["layout","role","mode","multiplicity","backend","memory","stage","checkpoint","counter","tokens","health","timing", "ce_rate","matrix_rate","loss_counts","warmup"])
def test_capacity_rejects_contradictory_evidence(evidence,damage):
    _,directory,r=evidence;c=r["configuration"];d=r["capacity"]
    if damage == "layout":c["selected_rt_layers"]=[0]
    if damage == "role":c["layout_role"]="chosen_architecture"
    if damage == "mode":c["mode"]["num_passes"]=3
    if damage == "multiplicity":c["rt_block_calls_per_forward_backward"]=1
    if damage == "backend":c["forward_tile_backend"]="eager"
    if damage == "memory":c["backward_memory"]="materialized"
    if damage == "stage":r["stage"]="correctness"
    if damage == "checkpoint":r["checkpoint"]["sha256"]="a"*64
    if damage == "counter":d["records"][-1]["counters"]["optimizer_updates"]=5
    if damage == "tokens":d["input_tokens_per_second"]*=2
    if damage == "health":d["health"]["passed"]=False
    if damage == "timing":d["full_step"]["median_wall_seconds"]=0
    if damage == "ce_rate":d["ce_targets_per_second"]*=2
    if damage == "matrix_rate":d["estimated_matrix_tflops_per_second_minimum"]*=2
    if damage == "loss_counts":d["records"][0]["counts"]={"ce":1,"latent":1,"kl":1}
    if damage == "warmup":d["backward_only_warmup"]=1
    write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)


@pytest.mark.parametrize("damage",["flops","memory_flag","ce","union","weights","registered","trainable","resident_bytes","gradient","optimizer","inference","shapes","execution_names","inference_names","parameter_counts"])
def test_resource_card_has_actual_counts_and_parameter_ownership(evidence,damage):
    _,directory,r=evidence;c=r["resources"]
    if damage == "flops":c["analytic_matrix_work"]["matrix_flops_minimum"]+=1
    if damage == "memory_flag":c["analytic_matrix_work"]["backward_memory"]="materialized"
    if damage == "ce":c["loss_work"]["ce_targets"]+=1
    if damage == "union":c["loss_work"]["predictor_positions"]+=1
    if damage == "weights":c["loss_weights"]["latent"]=0
    if damage in ("registered","trainable","resident_bytes","gradient","optimizer","inference"):
        key={"registered":"registered_unique","trainable":"trainable","resident_bytes":"resident_parameter_bytes",
             "gradient":"gradient_participating","optimizer":"optimizer_owned","inference":"deployable_inference_declared"}[damage]
        c["observed_parameters"][key]+=1
    if damage == "shapes":c["named_parameter_shapes"].pop(next(iter(c["named_parameter_shapes"])))
    if damage == "execution_names":c["declared_execution_names"].pop()
    if damage == "inference_names":c["declared_inference_names"].pop()
    if damage == "parameter_counts":c["analytic_matrix_work"]["parameter_counts"]["backbone"]+=1
    write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)


@pytest.mark.parametrize("damage",[None,"strict_only","loss","single_layer_count","by_layer","forward_fallback","boolean_count","missing_gradient","global_sums","nonfinite","zero_reference","optimizer"])
def test_correctness_checks_remain_strict_across_layers(evidence,damage):
    r=make_report(stage="correctness",length=2048)
    first=next(v for v in r["checks"] if v["name"] == "same_state_variant_vs_reference")
    weight=next(iter(first["gradients"]))
    if damage == "strict_only":first["gradients"][weight]["passed"]=False
    if damage == "loss":first["forward_losses_bitwise_equal"]=False
    if damage == "single_layer_count":first["candidate_recompute_backward_calls"]=2047
    if damage == "by_layer":first["candidate_dispatch"]["by_layer"]["0"]["recompute_tiles"]-=1
    if damage == "forward_fallback":first["candidate_dispatch"]["by_layer"]["0"]["forward_eager_tiles"]=0
    if damage == "boolean_count":first["candidate_dispatch"]["by_layer"]["0"]["forward_blocks"]=True
    if damage == "missing_gradient":
        for check in r["checks"]:
            if "gradients" in check:check["gradients"].pop(weight)
        first["mixed_gradient_screen"].pop(weight)
    if damage == "global_sums":first["mixed_gradient_screen"][weight]["delta_sq"]=.0001
    if damage == "nonfinite":first["gradients"][weight]["finite"]=False
    if damage == "zero_reference":first["gradients"][weight].update(reference_max_abs=0.,max_abs=1e-40)
    if damage == "optimizer":r["checks"][-1]["arms"][1]["boundary"]["optimizer"]="different"
    install(evidence,r)
    if damage not in (None,"strict_only"):
        with pytest.raises(ValueError):summarize(evidence)
    else:
        summary=summarize(evidence)
        if damage:assert any(row["nested_diagnostic_failures"] for row in summary["runs"][0]["checks"])


@pytest.mark.parametrize("layout",["all16","spread2"])
def test_failed_stress_is_preserved_without_invalidating_primary(evidence,layout):
    r=make_report(layout=layout);r.update(status="failed",error_type="OutOfMemoryError",error_message="bounded stress exceeded")
    install(evidence,r);summary=summarize(evidence)
    assert summary["status"] == "completed_with_failed_diagnostics"
    assert summary["primary_status"] == ("not_assessed" if layout == "all16" else "completed_with_failed_diagnostics")
    assert not summary["full_steps"] and not summary["resource_cards"]
    assert summary["successful_f3e_optimizer_updates"]["total"] == 0
    assert summary["failed_diagnostics"][0]["excluded_partial_performance"]


def test_incomplete_requires_explicit_preview(evidence):
    _,directory,r=evidence;r["status"]="running";r.pop("finished_utc");write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)
    assert summarize(evidence,allow_incomplete=True)["status"] == "partial_preview"


def test_historical_snapshots_preserved_and_corruption_rejected(evidence):
    project,directory,r=evidence
    write(directory/"source-snapshot/scripts/runtime.py","source");write(directory/"protocol.md","protocol")
    write(project/"scripts/runtime.py","new source");write(project/reporter.PROTOCOL,"new protocol")
    assert summarize(evidence)["status"] == "passed"
    write(directory/"source-snapshot/scripts/runtime.py","corrupt")
    with pytest.raises(ValueError):summarize(evidence)


@pytest.mark.parametrize("anchor",["single","all16"])
def test_reference_or_stress_alone_does_not_claim_primary_coverage(evidence,anchor):
    report=make_report(layout=anchor);install(evidence,report)
    assert summarize(evidence)["primary_status"] == "not_assessed"


def test_failed_stress_does_not_override_completed_primary(evidence):
    project,directory,original=evidence
    failure=make_report(layout="all16")
    failure.update(status="failed",error_type="OutOfMemoryError",error_message="bounded stress exceeded",
                   source_hashes=original["source_hashes"],protocol_sha256=original["protocol_sha256"])
    failed_directory=directory.with_name("stress-failed");write(failed_directory/"report.json",failure)
    summary=reporter.summarize([directory,failed_directory],project_root=project)
    assert summary["primary_status"] == "passed"
    assert summary["status"] == "completed_with_failed_diagnostics"
    assert summary["successful_f3e_optimizer_updates"]["total"] == 6


@pytest.mark.parametrize("damage",["l2_recomputed","max_recomputed","bitwise_flag","equal_bad_metrics", "missing_boundary_scheduler", "bad_boundary_counters", "bad_health", "warmup", "missing_source"])
def test_independent_evidence_arithmetic_and_state_boundaries(evidence,damage):
    r=make_report(stage="correctness"); install(evidence,r)
    first=next(row for row in r["checks"] if row["name"] == "same_state_variant_vs_reference")
    weight=next(iter(first["gradients"]))
    update=r["checks"][-1]
    if damage == "l2_recomputed":first["mixed_gradient_screen"][weight]["relative_l2"]=.001
    if damage == "max_recomputed":first["mixed_gradient_screen"][weight]["max_relative"]=.001
    if damage == "bitwise_flag":
        graph=next(row for row in r["checks"] if row["name"] == "candidate_initial_graph")
        graph["gradients"][weight]["bitwise_equal"]=False
    if damage == "equal_bad_metrics":
        for arm in update["arms"]:arm["metrics"][0]["objective"]+=1
    if damage == "missing_boundary_scheduler":
        for arm in update["arms"]:arm["boundary"].pop("scheduler")
    if damage == "bad_boundary_counters":
        for arm in update["arms"]:arm["boundary"]["counters"]["documents"]+=1
    if damage == "bad_health":
        for arm in update["arms"]:arm["health"]["nonfinite_parameters"]=[weight]
    if damage == "warmup":r["backward_preparation"]["warmup"]=10
    if damage == "missing_source":r["source_hashes"].pop("cdrm/pretrained/static_training.py")
    write(evidence[1]/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)


@pytest.mark.parametrize("damage",["objective","documents","counter_counts","health_details","capture_replays"])
def test_capacity_validates_canonical_metrics_and_health(evidence,damage):
    _,directory,r=evidence
    if damage == "objective":r["capacity"]["records"][0]["objective"]+=1
    if damage == "documents":r["capacity"]["records"][0]["counters"]["documents"]+=1
    if damage == "counter_counts":r["capacity"]["records"][0]["counters"]["ce_positions"]+=1
    if damage == "health_details":r["capacity"]["health"].pop("nonfinite_optimizer_tensors")
    if damage == "capture_replays":r["backward_preparation"]["replay"]=7
    write(directory/"report.json",r)
    with pytest.raises(ValueError):summarize(evidence)
