#!/usr/bin/env python3
"""Installed FA4/CuTE functionality check; does not replace native RT math."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cdrm.pretrained.artifacts import write_json,sha256_file
from scripts.olmo_validation import require_container_gpu
from scripts.experiment_tracking import OnlineTracker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    runtime=require_container_gpu()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={"schema":"olmo-fa4-smoke-v1","status":"running","runtime":{k:str(v) for k,v in runtime.items()},
            "started_utc":datetime.now(timezone.utc).isoformat()}
    tracker=OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
        group="olmo1b-f3b-rt-kernel",name="olmo-"+args.output_dir.name)
    try:
        import flash_attn.cute.interface as fa
        report["interface"]={"path":fa.__file__,"sha256":sha256_file(Path(fa.__file__))}
        report["versions"]={name:importlib.metadata.version(name) for name in ("flash-attn-4","nvidia-cutlass-dsl","triton")}
        tracker.start({"interface":report["interface"],"versions":report["versions"]})
        torch.manual_seed(13)
        q,k,v=[torch.randn(2,33,4,128,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        probe=torch.randn_like(q)
        out,lse=fa.flash_attn_func(q,k,v,causal=False,return_lse=True)
        gradients=torch.autograd.grad(out,(q,k,v),probe)
        refs=[x.detach().float().requires_grad_() for x in (q,k,v)]
        qr,kr,vr=[x.transpose(1,2) for x in refs]
        score=qr@kr.transpose(-1,-2)/(128**.5)
        reference=(score.softmax(-1)@vr).transpose(1,2)
        expected=torch.autograd.grad(reference,refs,probe.float())
        def error(actual,target):
            delta=actual.double()-target.double()
            return {"relative_l2":float(delta.norm()/target.double().norm()),
                    "max_relative":float(delta.abs().max()/target.double().abs().max())}
        comparisons={"output":error(out,reference),**{n:error(a,b) for n,a,b in zip(("dq","dk","dv"),gradients,expected)}}
        report["comparisons"]=comparisons
        report["lse_shape"]=list(lse.shape)
        assert all(x["relative_l2"]<1/64 and x["max_relative"]<1/16 for x in comparisons.values())
        # Standalone forward capture only: no model/RT/backward integration claim.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(10):
                fa.flash_attn_func(q.detach(),k.detach(),v.detach(),return_lse=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=fa.flash_attn_func(q.detach(),k.detach(),v.detach(),return_lse=True)
        graph.replay();torch.cuda.synchronize()
        report["capture_output_exact"]=torch.equal(out,captured[0])
        assert report["capture_output_exact"]
        report["status"]="passed"
        tracker.log({"smoke/"+n+"_relative_l2":x["relative_l2"] for n,x in comparisons.items()},step=1)
        print({"status":report["status"],"comparisons":comparisons,"capture_exact":report["capture_output_exact"]},flush=True)
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error_message=str(error))
        raise
    finally:
        tracker.finish(succeeded=report["status"]=="passed")
        report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat(),
            script_sha256=sha256_file(Path(__file__)),launcher_sha256=sha256_file(ROOT/"scripts/docker_shell.sh"))
        write_json(args.output_dir/"report.json",report)


if __name__=="__main__":main()
