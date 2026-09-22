#!/usr/bin/env python3
"""Native-checkpoint comparison and capacity of fused RT backward tiles."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import gc
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cdrm.pretrained.artifacts import write_json,sha256_file
from cdrm.pretrained.static_training import StaticFBTTraining
from cdrm.pretrained.lm_training import LMTrainingConfig,TrainingCounters
from scripts.olmo_f3_graph_training import (SOURCE_FILES as F3_SOURCES,case_for,changed_batch,
    build_optimizer,build_model,state_health,compare_graph,loss_snapshot,full_update_parity,timed,
    configure_determinism,require_container_gpu,validate_prepared_manifest,load_native_state_dict,
    load_native_tokenizer,backend_context,tensor_comparison,OnlineTracker)

SOURCES=tuple(sorted(set(F3_SOURCES)|{"scripts/olmo_f3c_validate.py","cdrm/pretrained/olmo_rt_kernels.py","cdrm/pretrained/olmo_rt_backward_kernels.py"}))
PROTOCOL=ROOT/"docs/reports/olmo1b-f3c/protocol.md"


def global_gradient_l2(rows):
    rows=list(rows)
    error=sum(row["delta_sq"] for row in rows)
    reference=sum(row["reference_sq"] for row in rows)
    return math.sqrt(error/reference) if reference else (0. if error==0 else float("inf"))


def expected_fused_backward_calls(batch,mode):
    passes=mode.num_passes-1 if mode.enabled else 1
    return (batch.input_ids.shape[1]-1)*len(mode.rt_mode.selected_layers)*passes


@contextmanager
def count_backward_tiles():
    from cdrm.pretrained import olmo_rt_backward_kernels
    original=olmo_rt_backward_kernels.backward_tile
    observed={"count":0}
    def counted(*args,**kwargs):
        observed["count"]+=1
        return original(*args,**kwargs)
    with patch.object(olmo_rt_backward_kernels,"backward_tile",counted):
        yield observed


def new_plan(model,batch,mode,variant):
    base=model.backbone.backbone
    base.ordinary_activation_checkpointing=True
    base.cast_weights_once=True
    base.tile_backend="triton"
    base.backward_tile_backend="triton" if variant=="triton" else "eager"
    return StaticFBTTraining(model,batch,mode=mode,config=LMTrainingConfig(precision="bf16_mixed"))


def compare_variant(model,batch,mode,variant):
    plan=new_plan(model,batch,mode,"reference")
    # Gradient-buffer initialization itself executes a backward. Count only
    # the following requested comparison backward in each arm.
    plan.initialize_gradients()
    with count_backward_tiles() as reference_calls:
        reference=plan.backward(replay=False)
    losses=loss_snapshot(reference)
    gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    del plan,reference
    model.zero_grad(set_to_none=True)
    plan=new_plan(model,batch,mode,variant)
    plan.initialize_gradients()
    with count_backward_tiles() as candidate_calls:
        candidate=plan.backward(replay=False)
    candidate_losses=loss_snapshot(candidate)
    checks={n:tensor_comparison(v,losses[n]) for n,v in candidate_losses.items()}
    grad_checks={n:tensor_comparison(p.grad,gradients[n]) for n,p in model.named_parameters() if p.grad is not None}
    ownership=set(grad_checks)==set(gradients)==set(plan.active_names)
    exact=all(r["bitwise_equal"] for r in [*checks.values(),*grad_checks.values()])
    forward_losses_equal=all(r["bitwise_equal"] for r in checks.values())
    expected_calls=expected_fused_backward_calls(batch,mode) if variant=="triton" else 0
    dispatch_matches=reference_calls["count"]==0 and candidate_calls["count"]==expected_calls
    # Only historical backward tiles change; both arms retain validated F3b
    # forward fusion/cast reuse. Engineering screens are recorded prospectively.
    if variant=="triton":
        def metrics(a,b):
            a,b=a.double(),b.double();delta=a-b
            return {"delta_sq":float(delta.square().sum()),"reference_sq":float(b.square().sum()),
                "relative_l2":float(delta.norm()/b.norm()) if b.norm()!=0 else (0. if delta.norm()==0 else float("inf")),
                "max_relative":float(delta.abs().max()/b.abs().max()) if b.abs().max()!=0 else (0. if delta.abs().max()==0 else float("inf"))}
        mixed={n:metrics(p.grad,gradients[n]) for n,p in model.named_parameters() if p.grad is not None}
        mixed_losses={n:metrics(v,losses[n]) for n,v in candidate_losses.items()}
        global_l2=global_gradient_l2(mixed.values())
        passed=(ownership and dispatch_matches and forward_losses_equal and global_l2<=1/64
            and all(x["relative_l2"]<=1/32 and x["max_relative"]<=1/16 for x in mixed.values())
            and all(x["relative_l2"]<=1/64 for x in mixed_losses.values()))
    else:
        mixed=None;mixed_losses=None;global_l2=None
        passed=ownership and dispatch_matches and all(r["passed"] for r in [*checks.values(),*grad_checks.values()])
    report={"name":"same_state_variant_vs_reference","passed":passed,"all_bitwise_equal":exact,
            "ownership_matches":ownership,"forward_losses_bitwise_equal":forward_losses_equal,
            "losses":checks,"gradients":grad_checks,
            "reference_fused_backward_calls":reference_calls["count"],
            "candidate_fused_backward_calls":candidate_calls["count"],
            "expected_candidate_fused_backward_calls":expected_calls,"dispatch_matches":dispatch_matches,
            "mixed_loss_screen":mixed_losses,"mixed_gradient_screen":mixed,"global_gradient_relative_l2":global_l2}
    del gradients,losses,candidate
    return plan,report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=("rt","combined"),default="combined")
    parser.add_argument("--variant",choices=("reference","triton"),default="triton")
    parser.add_argument("--stage",choices=("correctness","capacity"),default="correctness")
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--length",type=int,choices=(32,512),default=32)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--artifacts",type=Path,default=ROOT/".runtime/olmo1b-step60000/artifacts")
    args=parser.parse_args(argv)
    if not 1<=args.batch_size<=128:parser.error("bounded batch1..128")
    determinism=configure_determinism(True);runtime=require_container_gpu()
    torch.set_num_threads(4);torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        dst=args.output_dir/"source-snapshot"/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dst)
    shutil.copyfile(PROTOCOL,args.output_dir/"protocol.md")
    report={"schema":"olmo-f3c-native-v1","status":"running","stage":args.stage,
        "configuration":{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "forward_tile_backend":"triton","cast_weights_once":True,
            "backward_tile_backend":"triton" if args.variant=="triton" else "eager"},
        "runtime":{k:str(v) for k,v in runtime.items()},"determinism":determinism,
        "started_utc":datetime.now(timezone.utc).isoformat(),"checks":[],
        "source_hashes":{p:sha256_file(ROOT/p) for p in SOURCES},"protocol_sha256":sha256_file(PROTOCOL)}
    tracker=OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
        group="olmo1b-f3c-rt-backward",name="olmo-"+args.output_dir.name)
    def publish(check):
        report["checks"].append(check)
        write_json(args.output_dir/"report.json",report)
        print({"check":check["name"],"passed":check["passed"],"exact":check.get("all_bitwise_equal")},flush=True)
        tracker.log({"correctness/passed":int(check["passed"])},step=len(report["checks"]))
        if not check["passed"]:raise AssertionError(check["name"])
    try:
        manifest=validate_prepared_manifest(args.artifacts);report["checkpoint"]=manifest["checkpoint"]
        tracker.start({"configuration":report["configuration"],"checkpoint":report["checkpoint"]})
        print({"wandb":tracker.record["run_url"]},flush=True)
        state=load_native_state_dict(args.artifacts);tokenizer=load_native_tokenizer(args.artifacts)
        case=case_for(args.case,batch=args.batch_size,length=args.length)
        model=build_model(state,case)
        batch=changed_batch(tokenizer,case,0)
        with backend_context("flash"):
            if args.stage=="correctness":
                plan,comparison=compare_variant(model,batch,case.mode(),args.variant)
                publish(comparison)
                plan.capture(warmup=10)
                publish(compare_graph(plan,"candidate_initial_graph"))
                plan.load_batch(changed_batch(tokenizer,case,1))
                publish(compare_graph(plan,"candidate_changed_tokens_and_overwrite",replays=2))
                publish(full_update_parity(plan,tokenizer,case,updates=3))
                plan.load_batch(changed_batch(tokenizer,case,2))
                publish(compare_graph(plan,"candidate_changed_weights"))
            else:
                plan=new_plan(model,batch,case.mode(),args.variant)
                optimizer,scheduler=build_optimizer(model);counters=TrainingCounters()
                for step in range(3):plan.optimizer_step(optimizer,changed_batch(tokenizer,case,step),scheduler=scheduler,counters=counters)
                plan.capture(warmup=10)
                batches=[changed_batch(tokenizer,case,i+3) for i in range(3)];records=[]
                def complete():
                    records.append(plan.optimizer_step(optimizer,batches[len(records)],replay=True,scheduler=scheduler,counters=counters))
                timing=timed(complete,3)
                report["capacity"]={"full_step":timing,"input_tokens_per_second":plan.input_tokens/timing["median_wall_seconds"],
                    "records":records,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                    "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                    "current_reserved_gib":torch.cuda.memory_reserved()/2**30,"health":state_health(model,optimizer)}
                publish({"name":"finite_complete_updates","passed":report["capacity"]["health"]["passed"]})
                tracker.log({"capacity/input_tokens_per_second":report["capacity"]["input_tokens_per_second"],
                    "capacity/peak_allocated_gib":report["capacity"]["peak_allocated_gib"]},step=len(report["checks"])+1)
                print(report["capacity"],flush=True)
        assert report["source_hashes"]=={p:sha256_file(ROOT/p) for p in SOURCES}
        assert report["protocol_sha256"]==sha256_file(PROTOCOL)
        report["status"]="passed"
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error_message=str(error));raise
    finally:
        tracker.finish(succeeded=report["status"]=="passed")
        report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir/"report.json",report)


if __name__=="__main__":main()
