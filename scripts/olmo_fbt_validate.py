#!/usr/bin/env python3
"""O5a actual-checkpoint FBT checks; no learning or long-run authorization."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer, validate_prepared_manifest
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTConfig, FBTMode, FBTOnlineMode
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.fbt_training import FBTNextLatLM
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_lm_common import fixture, fixture_record, state_digests, verify_nextlat_sources
from scripts.olmo_lm_validate import SOURCE_FILES as O3_SOURCES
from scripts.olmo_validation import require_container_gpu, autocast, comparison, descriptive, semantic_gradients

SOURCE_FILES = tuple(sorted(set(O3_SOURCES) | {
    "cdrm/pretrained/olmo_fbt.py", "cdrm/pretrained/fbt_training.py",
    "scripts/olmo_fbt_validate.py", "docs/reports/olmo1b-o5a/protocol.md",
    "cdrm/pretrained/_fbt_reference/gpt.py", "cdrm/pretrained/_fbt_reference/manifest.json",
}))


def build(state, *, tiled=True, enabled=False):
    cls = OLMoTiledRTForCausalLM if tiled else OLMoRTForCausalLM
    base = cls(OLMoConfig.native_1b(), attention_backend="math", device="meta", dtype=torch.float32)
    base.load_state_dict(state, strict=True, assign=True)
    fbt = OLMoFBT(base, FBTConfig())
    return FBTNextLatLM(fbt, NextLatConfig(model_dim=2048, vocab_chunk_size=8), enabled=enabled).to("cuda").eval()


def manual_fusion(fbt, embedding, previous):
    """Direct retained-source equations; never call production fusion/shift."""
    eps = fbt.fusion.norm_eps
    token = embedding * torch.rsqrt(embedding.square().mean(-1, keepdim=True) + eps)
    value = F.linear(previous, fbt.fusion.state_proj.weight)
    product = value * F.linear(token, fbt.fusion.token_gate.weight).sigmoid()
    return fbt.fusion.output_scale * product * torch.rsqrt(product.square().mean(-1, keepdim=True) + eps)


def manual_passes(model, batch, mode):
    fbt = model.backbone
    embedding = fbt.token_embeddings(batch.input_ids)
    base = fbt.backbone
    if not mode.enabled:
        return (base(inputs_embeds=embedding, attention_mask=batch.valid_mask,
                     mode=mode.rt_mode, return_logits=False).last_hidden_state,)
    hidden = base(inputs_embeds=embedding, attention_mask=batch.valid_mask,
                  mode=RTMode(()), return_logits=False).last_hidden_state
    passes = [hidden]
    for _ in range(1, mode.num_passes):
        columns = []
        for t in range(embedding.shape[1]):
            current = embedding[:, t:t+1]
            if t and mode.beta:
                fused = manual_fusion(fbt, current, hidden[:, t-1:t])
                allowed = (batch.valid_mask[:, t] & batch.valid_mask[:, t-1]
                           & (batch.document_ids[:, t] == batch.document_ids[:, t-1]))
                current = torch.where(allowed[:, None, None],
                                      (1-mode.beta)*current + mode.beta*fused, current)
            columns.append(current)
        hidden = base(inputs_embeds=torch.cat(columns, dim=1), attention_mask=batch.valid_mask,
                      mode=mode.rt_mode, return_logits=False).last_hidden_state
        passes.append(hidden)
    return tuple(passes)


def manual_loss(model, batch, mode):
    """Independent dense CE/latent/KL and pass reduction for tiny fixtures."""
    passes = manual_passes(model, batch, mode)
    embedding = model.backbone.token_embeddings(batch.input_ids)
    weight = model.backbone.readout_weight
    pair = (batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
            & (batch.document_ids[:, :-1] == batch.document_ids[:, 1:]))
    triple = pair[:, :-1] & pair[:, 1:]
    masks = {
        "ce": pair if batch.ce_mask is None else pair & batch.ce_mask[:, 1:],
        "latent": pair if batch.latent_mask is None else pair & batch.latent_mask[:, 1:],
        "kl": triple if batch.kl_mask is None else triple & batch.kl_mask[:, 2:],
    }
    weights = model.objective_weights()
    means = {k: [] for k in masks}
    for h in passes:
        zero = h.sum()*0
        values = {"ce": F.cross_entropy(F.linear(h[:, :-1], weight).float()[masks["ce"]],
                                        batch.input_ids[:, 1:][masks["ce"]]), "latent": zero, "kl": zero}
        if model.enabled:
            pred = model.predictor(h[:, :-1], embedding[:, 1:])
            if weights["latent"]:
                values["latent"] = F.smooth_l1_loss(pred.float(), h[:, 1:].detach().float(), reduction="none").mean(-1)[masks["latent"]].mean()
            if weights["kl"]:
                teacher = F.log_softmax(F.linear(h[:, 1:-1].detach(), weight.detach()).float(), -1)
                student = F.log_softmax(F.linear(pred[:, :-1], weight.detach()).float(), -1)
                values["kl"] = (teacher.exp() * (teacher-student)).sum(-1)[masks["kl"]].mean()
        for key in means:
            means[key].append(values[key])
    aggregate = {key: values[0] + (model.gamma * sum(values[1:])/(len(values)-1) if len(values)>1 else 0)
                 for key, values in means.items()}
    return sum(weights[key]*aggregate[key] for key in means), aggregate


def gradients(model, reference):
    rows = semantic_gradients(model, reference)
    expected = dict(reference.named_parameters())
    inactive = [name for name, p in model.named_parameters() if p.grad is None and expected[name].grad is None]
    rows["inactive_both"] = inactive
    rows["missing"] = [name for name in rows["missing"] if name not in inactive]
    rows["passed"] = not rows["missing"] and not rows["failed"]
    return rows


def finite_pair(model, reference, batch, mode):
    model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
    out = model.backbone(batch.input_ids, attention_mask=batch.valid_mask,
                         document_ids=batch.document_ids, mode=mode, return_logits=False)
    expected = manual_passes(reference, batch, mode)
    outputs = [comparison(a, b, atol=5e-5, rtol=5e-5) for a, b in zip(out.pass_hidden_states, expected)]
    # Dense CE on the final pass exercises every backbone tensor and the fusion.
    a_loss = F.cross_entropy(F.linear(out.last_hidden_state[:, :-1], model.backbone.readout_weight).float().flatten(0,1), batch.input_ids[:, 1:].flatten())
    b_loss = F.cross_entropy(F.linear(expected[-1][:, :-1], reference.backbone.readout_weight).float().flatten(0,1), batch.input_ids[:, 1:].flatten())
    a_loss.backward(); b_loss.backward()
    grad = gradients(model, reference)
    loss = comparison(a_loss,b_loss,atol=5e-5,rtol=5e-5)
    return {"kind": "finite_pass_gradient_parity", "mode": asdict(mode), "outputs": outputs,
            "loss": loss, "gradients": grad,
            "passed": len(out.pass_hidden_states)==len(expected) and all(x["passed"] for x in outputs) and loss["passed"] and grad["passed"]}


def objective_pair(model, reference, batch, mode):
    model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
    actual = model.loss_sums(batch, backbone_kwargs={"mode": mode})
    expected, means = manual_loss(reference, batch, mode)
    actual.total.backward(); expected.backward()
    loss = {key: comparison(actual.means[key], means[key], atol=5e-5, rtol=5e-5) for key in means}
    grad = gradients(model, reference)
    return {"kind": "nextlat_pass_objective_parity", "mode": asdict(mode), "nextlat": model.enabled,
            "counts": actual.counts, "loss_means": {k:float(v.detach()) for k,v in actual.means.items()},
            "loss_comparisons":loss,"gradients":grad,
            "passed": all(x["passed"] for x in loss.values()) and grad["passed"]}


@torch.no_grad()
def online_check(model, reference, batch):
    """Whole-prefix recomputation oracle, independent of online KV handling."""
    fbt = model.backbone
    mode = FBTOnlineMode(beta=.37, rt_mode=RTMode((0,),.37))
    output = fbt.forward_online(batch.input_ids, attention_mask=batch.valid_mask,
                                document_ids=batch.document_ids, mode=mode, return_logits=False)
    embedding = reference.backbone.token_embeddings(batch.input_ids)
    inputs, states = [], []
    for t in range(embedding.shape[1]):
        current = embedding[:,t:t+1]
        if t:
            current = (1-mode.beta)*current + mode.beta*manual_fusion(reference.backbone,current,states[-1])
        inputs.append(current)
        prefix = reference.backbone.backbone(inputs_embeds=torch.cat(inputs,dim=1),
                    mode=mode.rt_mode, return_logits=False).last_hidden_state
        states.append(prefix[:,-1:])
    expected = torch.cat(states,dim=1)
    parity = comparison(output.last_hidden_state, expected, atol=8e-5, rtol=8e-5)
    first = fbt.forward_online(batch.input_ids[:,:3], document_ids=batch.document_ids[:,:3],
                               mode=mode,use_cache=True,return_logits=False)
    rest = fbt.forward_online(batch.input_ids[:,3:], document_ids=batch.document_ids,
                              mode=mode,past_key_values=first.past_key_values,return_logits=False)
    chunk = comparison(torch.cat((first.last_hidden_state,rest.last_hidden_state),1),
                       output.last_hidden_state,atol=8e-5,rtol=8e-5)
    differences = []
    for passes in (1,2,3,batch.input_ids.shape[1]+1):
        finite = fbt(batch.input_ids,mode=FBTMode(num_passes=passes,beta=mode.beta,rt_mode=mode.rt_mode),return_logits=False)
        differences.append({"passes":passes, **descriptive(comparison(finite.last_hidden_state,expected,atol=0,rtol=0))})
    converged = comparison(finite.last_hidden_state,expected,atol=8e-5,rtol=8e-5)
    return {"kind":"exact_online_reference","mode":asdict(mode),"prefix_oracle":parity,
            "chunk_continuation":chunk,"finite_pass_differences":differences,"converged":converged,
            "passed":parity["passed"] and chunk["passed"] and converged["passed"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    runtime=require_container_gpu()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision("highest")
    report={"schema":"olmo-fbt-validation-v1","status":"running","runtime":runtime,"cases":[],
            "started_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"O5a B1/T8 native checkpoint correctness; no learning or capacity claim",
            "source_hashes":{p:sha256_file(ROOT/p) for p in SOURCE_FILES}}
    tracker=OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
                          group="olmo1b-step60000-fbt-reference",name="olmo-1b-o5a-validation")
    began=time.monotonic()
    try:
        manifest=validate_prepared_manifest(args.artifacts)
        report["checkpoint"]=manifest["checkpoint"]
        report["nextlat_reference"]=verify_nextlat_sources()
        directory=ROOT/"cdrm/pretrained/_fbt_reference"
        pinned=json.loads((directory/"manifest.json").read_text())
        if pinned["revision"] != "7037c60924870aca6e30fac95212b0c7caee052d":
            raise ValueError("Unexpected FBT source revision")
        for row in pinned["files"]:
            path=directory/row["file"]
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError("Pinned FBT source bytes changed")
        report["fbt_reference"]=pinned
        tracker.start({"scope":report["scope"],"checkpoint_sha256":report["checkpoint"]["sha256"],
                       "protocol":"docs/reports/olmo1b-o5a/protocol.md"})
        report["wandb"]=tracker.record
        print({"wandb":tracker.record["run_url"]},flush=True)
        state=load_native_state_dict(args.artifacts)
        batch=fixture(load_native_tokenizer(args.artifacts),length=8)
        report["fixture"]=fixture_record(batch)
        def record(row):
            report["cases"].append(row); write_json(args.output_dir/"report.json",report)
            tracker.log({"validation/passed":row["passed"],"validation/gradient_relative_l2":row.get("gradients",{}).get("relative_l2",0)},step=len(report["cases"]))
            print({"kind":row["kind"],"passed":row["passed"],"mode":row.get("mode"),"grad_l2":row.get("gradients",{}).get("relative_l2")},flush=True)
            if not row["passed"]:
                raise AssertionError("O5a case failed; inspect retained evidence")
        for enabled in (False,True):
            model=build(state,enabled=enabled); reference=build(state,tiled=False,enabled=enabled)
            report["fusion_scale"]=float(model.backbone.fusion.output_scale)
            report["fusion_config"]=asdict(model.backbone.fusion_config)
            report.setdefault("parameter_counts",{})[str(enabled)]=sum(p.numel() for p in model.parameters())
            initial=state_digests(model)
            if not enabled:
                modes=[FBTMode(num_passes=1,rt_mode=RTMode((0,),1)),
                       FBTMode(num_passes=3,beta=0),
                       FBTMode(num_passes=2,beta=.37,rt_mode=RTMode((0,),.37)),
                       FBTMode(num_passes=3,beta=1),
                       FBTMode(num_passes=3,beta=1,rt_mode=RTMode((0,),1))]
                for mode in modes: record(finite_pair(model,reference,batch,mode))
                model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
                record(online_check(model,reference,batch))
            mode=FBTMode(num_passes=2,beta=.37,rt_mode=RTMode((0,),.37))
            record(objective_pair(model,reference,batch,mode))
            if enabled:
                fp32means=report["cases"][-1]["loss_means"]
                model.zero_grad(set_to_none=True)
                with autocast("bf16_mixed"):
                    loss=model.loss_sums(batch,backbone_kwargs={"mode":mode})
                loss.total.backward()
                grad=gradients(model,reference)
                record({"kind":"bf16_mixed_observation","mode":asdict(mode),
                        "loss_means":{k:float(v.detach()) for k,v in loss.means.items()},
                        "fp32_loss_means":fp32means,"gradients":descriptive(grad),
                        "scope":"finite losses/all active gradients; differences descriptive, not a training gate",
                        "passed":bool(torch.isfinite(loss.total)) and all(bool(torch.isfinite(v)) for v in loss.means.values())
                                  and not grad["missing"] and all(r["finite"] for r in grad["tensors"].values())})
            record({"kind":"unchanged_weights_and_tying","nextlat":enabled,
                    "passed":state_digests(model)==initial and model.backbone.readout_weight is model.backbone.token_embeddings.weight})
            del model,reference;gc.collect();torch.cuda.empty_cache()
        if report["source_hashes"] != {p:sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise AssertionError("Validation sources changed during execution")
        report.update(status="passed",elapsed_seconds=time.monotonic()-began,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tracker.summary({"validation/status":"passed"})
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__)
        raise
    finally:
        try: tracker.finish(succeeded=report["status"]=="passed")
        finally:
            report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json",report)


if __name__ == "__main__":
    main()
