# OLMo F3 static-layout graph training

Read the [protocol](reports/olmo1b-f3/protocol.md),
[F2 assessment](reports/olmo1b-f2/assessment.md) and
[handoff](fbt-rt-nextlat-handoff.md). F3 extends capture to the canonical
CE/FBT/NextLat training objective. It preserves native model math and keeps
the optimizer outside capture. The completed report identifies the subset
actually validated; examples below do not establish additional coverage.

Enter the project GPU container first:

```bash
bash scripts/docker_shell.sh
test -f /.dockerenv
pwd
nvidia-smi
```

The expected directory is /workspace/cdrm-w-latent. Run all CUDA work inside
this container; do not fall back to CPU. Use new output directories for each
attempt, and keep W&B online under taylorbollman/pretrained-fbt-rt-nextlat.
Credentials come from the launcher environment and must not enter artifacts.

Start with the small correctness phase, using distinct output directories:

```bash
python scripts/olmo_f3_graph_training.py \
  --output-dir .runtime/olmo1b-step60000/f3-rt-t32-cpoff-NEW \
  --stage correctness --case rt --batch-size 1 --length 32 \
  --checkpointing off --warmup 10 --updates 3

python scripts/olmo_f3_graph_training.py \
  --output-dir .runtime/olmo1b-step60000/f3-combined-t32-cpon-NEW \
  --stage correctness --case combined --batch-size 1 --length 32 \
  --checkpointing on --warmup 10 --updates 3
```

Also check the complementary ordinary-checkpoint settings for these two modes.
The named --case combined-k3 selects three total FBT passes and can be checked
at B1/T32 if feasible. The --artifacts default is
.runtime/olmo1b-step60000/artifacts. This runner fixes RT index0, BF16 mixed
compute with FP32 parameters/gradients/Adam state, deterministic Flash SDPA and
disabled autocast weight caching; it configures determinism before CUDA starts.

After the small checks pass, explicitly test changed tokens, gradient overwrite,
changed weights and complete AdamW state parity at the larger shape:

```bash
python scripts/olmo_f3_graph_training.py \
  --output-dir .runtime/olmo1b-step60000/f3-combined-b8-t512-NEW \
  --stage correctness --case combined --batch-size 8 --length 512 \
  --checkpointing on --warmup 10 --updates 3
```

Repeat for --case rt. Keep the configuration and check outcomes with each run;
a passing B1/T32 case does not clear B8/T512 or large-batch training. Capacity
is a separate stage that measures prepared-eager and graph complete updates:

```bash
python scripts/olmo_f3_graph_training.py \
  --output-dir .runtime/olmo1b-step60000/f3-combined-b32-capacity-NEW \
  --stage capacity --case combined --batch-size 32 --length 512 \
  --checkpointing on --warmup 10 --repeats 3 --comfortable-gib 65
```

Run the corresponding RT cell. Consider B64 and then B128 only while the
preceding cell fits the comfort budget and capture succeeds. This entrypoint
measures one requested batch size; it does not launch the whole sweep. Inspect
stopped_at_memory_budget and each arm's memory fields before advancing.

Preparation validates the fixed layout outside capture. Replays may change
token values and model weights; changes to masks, positions, document layout,
shape, mode or ownership require new preparation/capture. Unsupported layouts
must fail explicitly. The graph computes forward, the canonical loss and
backward, then external clipping/AdamW/scheduler advance a complete update.
Persistent parameter gradients must be overwritten each replay. Updating
weights after capture must not reuse stale autocast weight-cache values.

The reusable public interface is
cdrm.pretrained.static_training.StaticFBTTraining. Construct it with the final
FBTNextLatLM on its final device in grad-enabled training mode, a valid fixed
layout batch, the existing FBTMode, and LMTrainingConfig. The caller owns the
optimizer, scheduler, counters and deterministic backend context; the class
does not silently choose a backend or capture during construction.

```python
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.static_training import StaticFBTTraining

# model, batch, mode, optimizer and scheduler are already configured.
model.train()
plan = StaticFBTTraining(
    model, batch, mode=mode, config=LMTrainingConfig(precision="bf16_mixed")
)
plan.capture(warmup=10)
counters = TrainingCounters()
metrics = plan.optimizer_step(
    optimizer, next_batch, replay=True, scheduler=scheduler, counters=counters
)
```

Use replay=False for the matched prepared-eager path. load_batch validates the
layout before copying tokens. backward(replay=...) performs no optimizer step;
it supports the separately timed forward/loss/backward region. Only parameters
actually participating in the fixed objective receive persistent gradients;
inactive trainable parameters keep grad=None. The preparation/capture backward
counts are separate from optimizer updates.

Existing standard checkpoint saving requires every parameter's .grad to be
None at a completed-update boundary. A live static plan instead owns persistent
gradient buffers and checks their storage identity. Saving/loading therefore
needs a lifecycle boundary: synchronize and dispose the capture/plan, clear
gradients to None, use standard checkpoint handling, and prepare/capture anew
after loading the model, optimizer, scheduler and counters. Do not clear or
replace gradient buffers and then replay the old plan. F3 does not validate
this save/load/rebuild lifecycle or serialize CUDA graphs; its in-place
complete-update comparisons are not checkpoint-resume clearance.

Use deterministic Flash SDPA and configure deterministic algorithms/cuBLAS
before CUDA initialization. Native RT tiling remains eager PyTorch operations
within the captured graph; selecting Flash for ordinary layers does not turn
it into an RT Flash kernel. Near-bitwise comparison budgets and bitwise results
remain separate. Preserve failures without relabeling them passing.

ordinary_activation_checkpointing remains an opt-in ordinary-block flag.
It must leave selected RT reconstruction untouched, and it rejects training
cache use. CUDA graph compatibility needs F3's own checkpointed checks.
Profile B32/64/128 only after smaller checks pass, stopping at an OOM or65GiB
allocated peak. Graph private pools and reserved memory are reported separately.
The capacity stage first performs three real preparation updates in each arm,
then ten backward-only warmups and three timed complete updates. The graph
arm also records its capture backward. A separate forward/loss/backward timing
does not advance the optimizer. Full-step measurements include validated input
copies, clipping, AdamW and scheduling; the region excludes these operations.
These short timings give directional measurements and remain separately labeled.
The raw capture_seconds field measures the entire plan.capture call: setup,
backward warmup, capture, synchronization and validation. In correctness runs
it also includes initial gradient discovery when not already initialized.
Interpret it as setup/warmup + capture time, not the duration of the CUDA graph
capture context alone. This preparation time is outside the timed updates.
For memory, capture_memory.allocated_gib and capture_memory.reserved_gib are
current totals immediately after setup: postcapture for the graph arm and
after warmup for the eager arm. The corresponding peak_ fields are high-water
marks through setup/capture; top-level row peak_ fields also include timed
updates and the separate region timing. Warmup allocator cache can be released
on graph entry, leaving current reserved memory well below its earlier peak.
Peak reserved memory is therefore not the continuing graph-pool footprint.
These totals include model and optimizer memory; they do not isolate the graph's
private pool. Keep current allocated/reserved and both sets of peaks distinct.

After results and assessment are complete, retain all successful and explicitly
failed diagnostics together. For example, inside the container:

```bash
python scripts/olmo_f3_retain.py \
  --runtime-dir .runtime/olmo1b-step60000/f3-SUCCESS \
  --runtime-dir .runtime/olmo1b-step60000/f3-FAILED \
  --allow-failed-diagnostics \
  --report-dir docs/reports/olmo1b-f3 \
  --output-dir .runtime/olmo1b-step60000/f3-retention-NEW \
  --storage-prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3-graph-training/20260922T000000Z/
```

Replace example paths and timestamp with the actual new run set. The retainer
verifies current successful source hashes and the existing original checkpoint's
GCS identity, then uploads only small evidence and receipts with create-only
preconditions. Failed diagnostics remain separate from passed coverage. No
model weights, optimizer checkpoints, W&B directories or secrets enter this
archive. Do not resume disposable fixture training from these measurements.
