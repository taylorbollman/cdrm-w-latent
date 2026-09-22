"""Frozen O5b paired-pilot construction, observation, evaluation and provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.fbt_evaluation import evaluate_fbt_batches
from cdrm.pretrained.lm_training import optimizer_step, optimizer_ownership
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTConfig, FBTMode, FBTOnlineMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_fbt_validate import SOURCE_FILES as O5A_SOURCES
from scripts.olmo_o4_common import PilotTracker, preserve_rng, write_event

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ordinary", "fbt")
SOURCE_FILES = tuple(sorted(set(O5A_SOURCES) | {
    "cdrm/pretrained/lm_data.py", "cdrm/pretrained/lm_schedule.py",
    "cdrm/pretrained/fbt_evaluation.py", "cdrm/pretrained/lm_evaluation.py",
    "scripts/olmo_o4_common.py", "scripts/olmo_o5b_common.py",
    "scripts/olmo_o5b_preflight.py", "scripts/olmo_o5b_train.py",
    "docs/reports/olmo1b-o5b/protocol.md",
}))


def source_hashes():
    return {name: sha256_file(ROOT/name) for name in SOURCE_FILES}


def resume_record(report, path):
    """Resume only the latest published boundary; older branches need a new lineage."""
    records=report.get("checkpoints",[])
    if not records or Path(records[-1]["path"]).name != Path(path).name:
        raise ValueError("Resume requires the latest recorded checkpoint, not an older branch")
    return records[-1]


def validate_completed_arm(report, config, arm, prefix, runtime):
    source=report.get("source_fingerprint",{})
    counters=report.get("counters",{});schedule=config["schedule"]
    if (report.get("status")!="completed" or report.get("arm")!=arm or not report.get("finished_utc")
            or report.get("configuration")!=config or report.get("storage_prefix")!=prefix
            or source.get("code")!=config["source_hashes"] or source.get("runtime")!=runtime
            or source.get("checkpoint_sha256")!=config["checkpoint_sha256"]
            or source.get("data_manifest_sha256")!=config["data_manifest_sha256"]
            or counters.get("optimizer_updates")!=schedule["total_updates"]
            or counters.get("input_tokens")!=schedule["total_tokens"]
            or report.get("data_cursor")!=schedule["used_windows"]):
        raise ValueError("Completed arm does not match the frozen queue lineage")
    records=[r for r in report.get("checkpoints",[]) if r.get("optimizer_updates")==schedule["total_updates"]]
    if len(records)!=1:
        raise ValueError("Completed arm needs exactly one final checkpoint")
    r=records[0];storage=r.get("storage",{})
    digest=r.get("sha256")
    if (not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)
            or type(r.get("size_bytes")) is not int or r["size_bytes"]<=0
            or storage.get("sha256")!=digest or storage.get("size_bytes")!=r.get("size_bytes")
            or not storage.get("generation") or not storage.get("md5_base64") or not storage.get("verification")
            or not storage.get("uri","").startswith(prefix+"/"+arm+"/")):
        raise ValueError("Completed arm lacks a matching retained final checkpoint")


class ObservedFBTLM(FBTNextLatLM):
    """Capture detached per-pass sums without altering the optimization graph."""
    def loss_sums(self, *args, **kwargs):
        result = super().loss_sums(*args, **kwargs)
        if getattr(self, "_observations", None) is not None:
            self._observations.append([float(p.sums["ce"].detach()) for p in result.pass_losses])
        return result


def build_model(state, *, backend="sdpa"):
    base = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), attention_backend=backend,
                               attention_precision="mixed", device="meta", dtype=torch.float32)
    base.load_state_dict(state, strict=True, assign=True)
    core = OLMoFBT(base, FBTConfig())
    return ObservedFBTLM(core, NextLatConfig(model_dim=2048, vocab_chunk_size=128),
                         enabled=False, gamma=1.0).to("cuda").eval()


def make_mode(beta):
    return FBTMode(num_passes=2, beta=beta, rt_mode=RTMode(()))


def build_optimizer(model, config, *, zero_lr=False):
    groups = []
    for label, selected, lr in (("backbone", False, config["lr"]), ("fusion", True, config["fusion_lr"])):
        pairs = [(name, p) for name, p in model.named_parameters()
                 if name.startswith("backbone.fusion.") == selected and p.requires_grad]
        if not pairs:
            raise ValueError("Both native and fusion parameter groups are required")
        if any(p.ndim < 2 for _, p in pairs):
            raise ValueError("O5b expects only native/fusion matrices; revisit decay policy for new parameter types")
        groups.append({"params": [p for _,p in pairs], "param_names": [n for n,_ in pairs],
                       "group_name": label, "lr": 0.0 if zero_lr else lr,
                       "weight_decay": config["weight_decay"]})
    optimizer = torch.optim.AdamW(groups, lr=0.0 if zero_lr else config["lr"],
                    betas=tuple(config["betas"]), eps=config["eps"], foreach=False)
    optimizer_ownership(model,optimizer)
    return optimizer


def build_scheduler(optimizer, config):
    warmup = config["schedule"]["warmup_updates"]
    fusion_warmup = config["fusion_warmup_updates"]
    if warmup < 1 or fusion_warmup < 1:
        raise ValueError("O5b requires positive native and fusion warmups")
    # The first ramp update after warmup still has beta0. Fusion becomes active
    # at warmup+2, and starts with 1/fusion_warmup of its base learning rate.
    functions = [lambda completed: min(1.0, (completed+1)/warmup),
                 lambda completed: min(1.0, max(0.0,(completed-warmup)/fusion_warmup))]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,functions)
    scheduler._cdrm_warmup_updates = warmup
    return scheduler


def slice_batch(batch, row_slice=slice(None), *, length=None):
    return NextLatBatch(**{name: None if value is None else value[row_slice, :length]
                           for name,value in batch.__dict__.items()})


def microbatches(batch, physical_batch):
    if type(physical_batch) is not int or physical_batch < 1 or batch.input_ids.shape[0] % physical_batch:
        raise ValueError("Physical batch must divide the full effective batch")
    return [slice_batch(batch,slice(i,i+physical_batch)) for i in range(0,batch.input_ids.shape[0],physical_batch)]


def observed_step(model, optimizer, batches, **kwargs):
    """Use the established step; group norms are observed after global clipping."""
    model._observations = []
    norms = {}
    def observe(opt, args, kw):
        for group in opt.param_groups:
            values = [torch.linalg.vector_norm(p.grad.detach().float()) for p in group["params"] if p.grad is not None]
            norms[group["group_name"]] = float(torch.linalg.vector_norm(torch.stack(values))) if values else 0.0
    hook = optimizer.register_step_pre_hook(observe)
    try:
        result = optimizer_step(model,optimizer,batches,**kwargs)
        count = result["counts"]["ce"]
        sums = model._observations
        if not sums or any(len(row)!=2 for row in sums):
            raise AssertionError("O5b must record exactly two pass losses for every microbatch")
        result["pass_ce_means"] = [sum(row[p] for row in sums)/max(count,1) for p in range(2)]
        result["group_gradient_norm_after_clip"] = norms
        result["stack_input_tokens"] = 2 * sum(int(batch.valid_mask.sum()) for batch in batches)
        return result
    finally:
        hook.remove()
        model._observations = None


def evaluation(model, corpus, config, mode, rows, *, documents=False):
    results={}
    for split in ("dev","retention_dev"):
        count=min(rows,corpus.split_sizes[split]); b=config["eval_batch_size"]
        batches=(corpus.batch(split,range(i,min(count,i+b)),device="cuda") for i in range(0,count,b))
        results[split]=evaluate_fbt_batches(model,batches,mode=mode,precision=config["precision"],
                                          include_document_records=documents)
    return results


def online_evaluation(model, corpus, config, beta):
    results={}
    for split in ("dev","retention_dev"):
        count=min(config["online_eval_rows"],corpus.split_sizes[split]); b=config["eval_batch_size"]
        def batches():
            for i in range(0,count,b):
                yield slice_batch(corpus.batch(split,range(i,min(count,i+b)),device="cuda"),
                                   length=config["online_eval_length"])
        finite=evaluate_fbt_batches(model,batches(),mode=make_mode(beta),precision=config["precision"],
                                   include_document_records=True)
        online=evaluate_fbt_batches(model,batches(),mode=FBTOnlineMode(beta=beta,rt_mode=RTMode(())),
                                   precision=config["precision"],include_document_records=True)
        results[split]={"pass0":finite["passes"][0],"finite":finite["passes"][1],"online":online["passes"][0]}
    return results


def catastrophic(current, initial, margin):
    # Stop when the same pass deteriorates catastrophically in BOTH domains.
    return any(all(current[k]["passes"][p]["mean_nll"] > initial[k]["passes"][p]["mean_nll"]+margin
                   for k in ("dev","retention_dev")) for p in (0,1))


def finite_optimizer(model, optimizer):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters()) and all(
        bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values()
        if isinstance(v,torch.Tensor))


def retain_file(path, uri, *, expected_sha256=None):
    """Create-only checkpoint/evidence retention; verify server checksums."""
    import base64
    from google.cloud import storage
    path=Path(path)
    root="gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/"
    if not uri.startswith(root) or not path.is_file() or path.is_symlink():
        raise ValueError("O5b retention requires an explicit regular file and its designated GCS prefix")
    bucket,key=uri[5:].split("/",1)
    if ".." in key.split("/"):
        raise ValueError("Invalid retention key")
    sha,md5=hashlib.sha256(),hashlib.md5()
    with path.open("rb") as stream:
        while chunk:=stream.read(16*1024**2): sha.update(chunk);md5.update(chunk)
    digest=sha.hexdigest(); size=path.stat().st_size; encoded=base64.b64encode(md5.digest()).decode()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError("Checkpoint changed before retention")
    blob=storage.Client().bucket(bucket).blob(key)
    if not blob.exists():
        blob.metadata={"sha256":digest,"artifact_schema":"olmo-o5b-pilot-v1"}
        blob.upload_from_filename(str(path),if_generation_match=0,timeout=1200,checksum="md5")
    blob.reload()
    if blob.size!=size or blob.md5_hash!=encoded or (blob.metadata or {}).get("sha256")!=digest:
        raise ValueError("Existing remote object differs from selected local bytes")
    return {"uri":uri,"generation":str(blob.generation),"sha256":digest,"size_bytes":size,
            "md5_base64":encoded,"verification":"GCS generation, size, server MD5 and SHA256 metadata verified"}
