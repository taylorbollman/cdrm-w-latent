# Two-H100 runtime and probe usage

Implementation reference: `d88be56` (2026-09-25). Read [progress.md](progress.md)
for the current queue and [protocol.md](protocol.md) for frozen comparison
budgets. These are bounded engineering probes, not a quality-training campaign.

## Runtime contracts

`EagerDDPTrainer` in `cdrm/pretrained/ddp_training.py` wraps the canonical
`FBTNextLatLM` through a real DDP forward/objective adapter. All ranks call it
in the same order with the same number of accumulation microbatches. Local
counts, masks and row counts may differ. CE, latent and KL each use their own
global denominator; the local objective includes the world-size factor needed
for DDP's default gradient averaging. Earlier accumulation microbatches use
`no_sync()` around both forward and backward. Globally unused parameters keep
`grad=None`, preserving their Adam/weight-decay behavior.

Use `backward(microbatches, config=..., backbone_kwargs=...)` to inspect reduced
raw gradients, then `step(result, optimizer, scheduler=..., counters=...)` for
one coordinated health check, clipping, Adam and counter update. The convenience
`optimizer_step(...)` combines these. Keep optimizer ownership and checkpoint
names on the canonical model, outside the DDP wrapper. Parameters, gradients
and Adam state remain FP32 under BF16 mixed execution.

`PreparedDDPObjective` and `DDPGraphTraining` in
`cdrm/pretrained/ddp_graph_training.py` provide a separate fixed-layout path:

1. Prepare the batch layout, mode, global loss counts and active parameter set.
2. Construct DDP on a side stream and call `prepare(warmup=11)`. This executes
   real DDP forwards/backwards and lets buckets rebuild before freezing gradient
   addresses. The harness allocates Adam state with three eager updates next.
3. Capture actual forward/loss/backward **including NCCL reduction**, then use
   `load_batch(...)` and `backward(replay=True)`. Clipping, Adam, scheduler,
   health/metric collectives and parameter broadcasts remain outside capture.

The graph holds shapes, masks, document boundaries, denominators, precision,
mode and parameter participation fixed. Tokens and in-place optimizer values
can change. Do not replace gradient storage, call `zero_grad(set_to_none=True)`
while retaining the graph, or restore fixed buffers just to copy identical
values: their versions are part of the execution contract. A changed layout
requires rebuilding. `--bucket-view` is an explicit separately checked option;
it is not the default. This path makes no dynamic-padding claim.

## Checkpoint and recovery boundaries

All ranks call `save_distributed_checkpoint`/`load_distributed_checkpoint`
at a cleared update boundary. Rank zero publishes one `state.pt`, followed by
the hashed `manifest.json` commit marker. Saving is create-only: use a new
directory, including after a failed attempt. The payload contains canonical
model/optimizer/scheduler/counters plus each rank's data cursor, Python, NumPy,
CPU and local CUDA RNG state and named generators. Replicas must already agree;
the probes check this separately. Loading validates configuration, source hashes,
parameter ownership/aliases, optimizer type and world size before mutation.

Graphs and DDP reducers are execution state, not serialized checkpoint state.
Release graph references and old reducer hooks before clearing gradients and
reconstructing; after loading, rebuild DDP, warm up and recapture. The recovery
probes compare the next raw gradients, losses, complete update and RNG draws.
They reconstruct models/optimizers/DDP **inside the same process group**. They
do not establish a fresh `torchrun` restart, changed-world resharding, or saving
and restoring a live CUDA graph.

`build_zero1_adamw` returns `FP32Zero1AdamW`, an opt-in native PyTorch ZeRO-1
wrapper. Parameter and gradient replicas remain; only Adam moments/step state
are partitioned. Native partitioning, local fused Adam and parameter broadcasts
are unchanged. The installed native loader moved scalar Adam steps to CPU and
retained duplicate outer state; our loader maps global IDs to the native local
shard and delegates placement to the underlying Adam loader. This preserves
FP32 local moments and fused CUDA step counters without mutating the supplied
checkpoint. Inspect `zero1_state_inventory`, particularly `optimizer.optim`:
the outer `optimizer.state` is intentionally empty.

Use `save_zero1_checkpoint`/`load_zero1_checkpoint` rather than calling a generic
rank-zero `optimizer.state_dict()`. Saving collectively consolidates global
optimizer state on rank zero, uses the same manifest protocol, and releases
the consolidation cache. Release graph pools first: collective staging can
add a temporary GPU peak. No optimizer-overlap hook, parameter rebinding,
offload, ZeRO-2 or FSDP is enabled.

## Launching probes

From the host project checkout, enter the required container:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh
```

Run the following **inside** that container. Check for other active work first;
do not overlap probes on the same GPUs. Each output/checkpoint directory below
must be fresh. Graph NCCL settings are scoped to graph launches; retain the
external timeout because asynchronous NCCL error handling is disabled there.

```bash
test -f /.dockerenv
test "$PWD" = /workspace/cdrm-w-latent
nvidia-smi

graph_probe() {
  env TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0 \
    timeout 3600s torchrun --standalone --nproc-per-node=2 "$@"
}

# Tiny eager coverage of all eight feature combinations.
timeout 3600s torchrun --standalone --nproc-per-node=2 \
  scripts/olmo_two_gpu_validate.py --tiny --case all --length 8 --updates 3 \
  --output-dir .runtime/olmo-two-gpu/example-tiny-eager

# Actual native RT at indices 0/15; replace rt with combined for K2 + NextLat.
graph_probe scripts/olmo_two_gpu_graph.py --case rt --stage correctness \
  --batch-size 1 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-rt-graph

# Actual combined graph reconstruction and complete next-update recovery.
graph_probe scripts/olmo_two_gpu_graph_recovery.py --case combined \
  --batch-size 1 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-combined-recovery \
  --checkpoint-dir .runtime/olmo-two-gpu/checkpoints/example-combined-recovery

# ZeRO-1 fixed-gradient full-Adam comparison and consolidated recovery.
timeout 3600s torchrun --standalone --nproc-per-node=2 \
  scripts/olmo_two_gpu_zero1.py --tiny --case combined --length 8 \
  --output-dir .runtime/olmo-two-gpu/example-tiny-zero1 \
  --checkpoint-dir .runtime/olmo-two-gpu/checkpoints/example-tiny-zero1

# New graph integration probe: first tiny, then actual after its prerequisites.
graph_probe scripts/olmo_two_gpu_zero1_graph.py --tiny --case combined \
  --stage integration --batch-size 1 --length 8 \
  --output-dir .runtime/olmo-two-gpu/example-tiny-zero1-graph
graph_probe scripts/olmo_two_gpu_zero1_graph.py --case combined \
  --stage integration --batch-size 1 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-actual-zero1-graph
```

For actual eager/ZeRO-1 recovery, omit `--tiny` and use `--length 512` with B1.
The replicated eager recovery harness is `olmo_two_gpu_recovery.py` and accepts
the same output/checkpoint arguments as graph recovery. Full-model probes load
the pinned OLMo-1B step60000 artifacts; actual execution uses BF16 mixed,
ordinary checkpointing, native RoPE, rounded compiled ordinary SwiGLU, fused
Adam and optimized native RT. Tiny probes use FP32 fixtures.

The following are scaling commands, **not measured performance claims**:

```bash
# Equal global physical batch: one GPU B128 versus two GPUs B64 each.
CUDA_VISIBLE_DEVICES=0 timeout 3600s python scripts/olmo_two_gpu_single_reference.py \
  --case rt --batch-size 128 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-single-rt-b128
graph_probe scripts/olmo_two_gpu_graph.py --case rt --stage capacity \
  --batch-size 64 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-ddp-rt-b64

# Conditional ZeRO-1 capacity after correctness/recovery and graph integration.
graph_probe scripts/olmo_two_gpu_zero1_graph.py --case rt --stage capacity \
  --batch-size 128 --length 512 \
  --output-dir .runtime/olmo-two-gpu/example-zero1-rt-b128
```

Repeat matched cases with `--case combined`; investigate RT B128/B192 and
combined B128 per rank if memory permits. Use a separate output directory for
`--bucket-view`. DDP does not pool VRAM: global B512 means physical B256 on each
GPU. Accumulation changes the optimizer batch without supplying that physical
batch's RT utilization. Report aggregate input tokens/sec, per-rank setup and
steady memory, GPU-seconds/token, parameter counts and estimated matrix FLOPs.
Count original input tokens once regardless of K passes. Capacity timings include
the complete update but exclude diagnostic hashing and checkpoint I/O.

## Retention and tested scope

Freeze runtime sources throughout each probe; reports snapshot and hash them.
Rank-zero W&B tracking uses `taylorbollman/pretrained-fbt-rt-nextlat`, group
`olmo-two-gpu`; never print `.env` values. Retain failures and launcher logs.
Final `report.json` is authoritative; an earlier progress file or embedded W&B
status may still say running/false after successful completion.

After writers stop, copy the launcher log into the evidence directory. Retain
from a CPU container (`CDRM_DOCKER_GPUS=none` on the host launcher):

```bash
env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_two_gpu_retain.py \
  --input-dir .runtime/olmo-two-gpu/example-combined-recovery \
  --checkpoint-dir .runtime/olmo-two-gpu/checkpoints/example-combined-recovery \
  --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/example-combined-recovery \
  --receipt .runtime/olmo-two-gpu/retention/example-combined-recovery.json
```

Omit `--checkpoint-dir` for evidence-only stages. Unsetting the inherited ADC
path selects the working mounted credentials. Receipts verify server size/MD5
and SHA metadata; evidence archives also get a downloaded SHA check. Actual
full-state checkpoints need roughly 13–14 GiB each. Check free persistent disk,
keep one new actual checkpoint at a time, and verify GCS retention before any
local cleanup. Keep the manifest and a cleanup receipt. Local SSD is disposable.

At this documentation snapshot, tiny eight-mode eager, tiny eager/graph recovery,
tiny ZeRO-1 recovery, actual ordinary/RT eager, actual RT eager recovery and
actual RT/combined graph correctness pass. Actual combined independent eager
update 2 retains strict parameter/moment tolerance failures despite passing
raw-gradient budgets; the anchored fixed-state comparison passes unchanged
budgets. This does not clear independent trajectory equivalence or the older
native/author BF16 qualification. Actual combined graph recovery, actual ZeRO-1,
the new ZeRO-1 graph integration and capacity/scaling require their own final
reports. Check progress.md before deciding what remains; command availability
alone is not validation. No Q/K-normalization or model-quality conclusion follows.
