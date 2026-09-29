# Background checkpoint retention and pilot preparation

Asynchronous retention is implemented and accepted, including exact tiny
two-GPU continuation and a completed four-update native accumulated-batch run.
Local-save regions with initialized optimizer state took about **76 seconds**;
checkpoint 2's **346-second
background retention overlapped subsequent training/evaluation callbacks**.
Full byte verification remains enabled. No 32-update learning cohort has started.

## Checkpoint behavior and interruption risk

After the normal immutable SSD snapshot and its preservation checks, training
can proceed while one background worker validates local bytes, uploads them,
performs full generation-pinned readback verification, publishes the recovery
receipt and applies the existing keep-two local retention. There is no unlimited
queue. The loop drains before another local save and at normal termination.

Cloud work runs in a fresh CPU-only child process. The training process never
wraps a concurrent upload in process-wide RNG save/restore; that would risk
rewinding live training randomness. Only the main thread updates W&B, reports
and collectives. Historical model math, optimizer, graph, data, saver and storage
verification code remain unchanged. New versioned host orchestration is explicit
in checkpoint identity, so transport changes cannot silently become same-lineage
resumes.

If the VM disappears mid-upload, recovery uses the previous verified cloud
checkpoint. A complete local SSD save is not yet cloud durability. Cloud
publication can finish before the trainer's next boundary poll records it;
persistent publication receipts and the journal remain authoritative. The first
fresh checkpoint can also overlap setup, so a new run has no new durable origin
until that first publication succeeds. Normal final termination waits for the
last checkpoint to finish.

The 480-second timeout bounds the cloud child phase. It is not a hard deadline
for local hashing, fsync, in-progress publication or a complete optimizer update.
These distinctions apply on both spot and reserved machines. A reserved machine
can justify a longer future cadence, but this milestone keeps the accepted
600-second save trigger. It is checked at completed update boundaries and resets
after local submission. It does not guarantee cloud publication every ten
minutes or limit rollback to ten minutes.

## Native measurement

The fixture uses two H100 80GB GPUs, OLMo-1B's original backbone plus the accepted
fusion128 weight import, a paired fresh predictor and fresh all-active Adam.
It retains BF16 mixed training, K4 FBT, native RT at layers 0/15 on every pass,
NextLat regression and KL, prepared CUDA graphs and activation checkpointing.
The logical batch is **524,288 valid inputs/update** at T1024, physical B12/GPU,
22 slots/GPU. Each update includes 512 real and 16 dummy rows globally; actual
counts and rank allocation match the declaration. Four updates consume
2,097,152 inputs and 2,095,104 CE targets. Prior fusion exposure remains separate.

There are **1,267,879,936 resident/trainable parameters**: 1,176,764,416 in the
backbone, 8,388,608 in fusion and 82,726,912 in the predictor. The deployable
model without the training-only predictor has 1,185,153,024 parameters.
Readout remains tied.

| Checkpoint | Selected synchronous local regions | Background worker regions | Observed overlap |
| --- | ---: | ---: | --- |
| Initial 0 | 27.48 s | 124.26 s | Preparation; completes before update 1 |
| Cadence 2 | 76.18 s | 346.45 s | 345.99 s of update 3/4 callback intervals |
| Terminal 4 | 76.41 s | 364.83 s | None; normal termination drains |

Local regions sum boundary observation, SSD serialization and postcheck, using
the maximum rank per phase. They exclude small control/report-write gaps.
Checkpoint 2's background work comprises 34.81 s of local validation, 277.58 s
in the cloud child and 33.95 s of verified publication/pruning. Update 4's callback also contains
development evaluation, so its overlap is not all training compute. Worker
interval overlap proves wall-clock concurrency, not simultaneous GPU and network
activity throughout that interval. Background time must not be added to
foreground time as if sequential. Loop blocking polls/drain total 364.69 s,
dominated by the expected terminal wait.

This addresses the recurring pause without weakening verification. There is no
matched blocking native control and no causal overall speedup claim against the
older one-slot fixture. Tiny blocking/async elapsed times of 40.64/25.12 s also vary
with cloud service timing and are not a native performance benchmark.

| Timing scope | Four-update input tokens/sec | Included / excluded |
| --- | ---: | --- |
| Compute regions | 3,743 | Forward/loss/backward/DDP plus optimizer/cursor coordination; excludes host data materialization and other gaps |
| Compute plus materialization | **3,568** | Adds host data/noise materialization; excludes evaluation, checkpoints and other host gaps |
| Complete update callbacks | 2,577 | Adds health/evidence work and final development evaluation; excludes setup, external checkpoint work and loop/logging tails |

Updates 2–4 give 3,567 inputs/s for compute plus materialization, close to the
all-four rate. Individual training/materialization regions take 146.87–147.00 s.
Graph preparation took 448.69 s; the development evaluation took 225.45 s.
The executor's 1,886.27 s elapsed time includes construction, setup, training,
evaluation, checkpoint drains and final checks, but excludes container/bootstrap
and final W&B shutdown. It is not steady-state throughput.

Allocator reservation reaches **59.06 GiB/GPU**, with sampled free memory
**12.78 GiB/GPU**. Peak allocated memory at the recorded samples is 33.35 GiB.
These are capture/update samples and cumulative allocator high-water marks,
not continuous free-memory minima or separately measured evaluation peaks.

## Adaptation concern remains open

All four updates are finite and heavily clipped:

| Update | Raw gradient norm | Estimated clip coefficient | Aggregate training CE | Latent mean | KL mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 211.12 | 0.00474 | 7.740 | 0.814 | 5.264 |
| 2 | 76.50 | 0.01307 | 6.514 | 0.807 | 4.148 |
| 3 | 73.80 | 0.01355 | 6.196 | 0.754 | 2.638 |
| 4 | 52.42 | 0.01908 | 5.666 | 0.721 | 3.155 |

The fixed 65,536-input common-FP32/no-jitter development evaluation has 65,472 CE
targets. Pass CEs are **3.18058, 7.87717, 7.71814, 7.74559**. Later-pass gaps
relative to pass 1 are +4.69659, +4.53756 and +4.56501 nats/target. Evaluation preserves
the training boundary exactly on both ranks. Falling training CE does not
establish useful refinement. This larger dev membership and exposure differ
from PR50; their endpoints are not a learning curve.

The [test ledger](test-ledger.md) records **136 passing CPU tests**, 2,847 exact
tiny blocking/async audit checks and 2,427 cloud-resume checks. All 200 frozen
runtime pins and 192 historical pins remain unchanged. Native source, budget,
loss-count, parameter-ownership, worker and evaluation assertions also pass in
the independent JSON summary. Native new-runtime byte retention is tested;
exact new-runtime cloud continuation is scoped to the tiny fixture.

The [pilot plan](pilot-plan.md) has concrete CPU-resolved B and
paired NF/NFR declarations at T1024, 524,288 inputs/update, an initial stop at 32
within a declared 128-update ceiling and a fixed 65,536-input development prefix.

Heavy clipping and worse later-pass CE remain an explicit
[open adaptation issue](open-issues.md), with review criteria at the first pilot
stop. Functional checkpoint acceptance is not useful refinement or a numerical
clearance. The four-update native run is a separate diagnostic, not the starting
checkpoint of that eventual cohort. The proposed first 32-update NFR segment
needs roughly two hours; NF's accumulated cost remains unmeasured. Review
per-pass CE gaps, first-pass trajectory, clipping and separate losses before
extending any learning ceiling. No new universal BF16 clearance is claimed.

Native run: [osqidkgt](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/osqidkgt).
Timing and per-pass charts:
[1ofs0x3r](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1ofs0x3r).
The chart upload synced successfully. An optional immediate readback of the
original run's final checkpoint summary fields failed; a later independent
read confirmed the requested values exactly, consistent with delayed read
visibility. The original failure and recovery receipt are both retained; no
training, chart upload or history was repeated.
Final report is `.runtime/olmo-pilot-async/native-nfr12-accum-01/report.json`,
SHA256 `f2e4065c7167cad0cfa22f24d3803bcdd286f9792554b0bd9f126f0a7ed9e05d`.
The independent native summary is `native-summary-01/report.json`,
SHA256 `86142ab9f44b6cd70b6da61d378b4bdffd49980e0306d940c233fe0cde154667`.
All three native checkpoints are cloud-verified; both GPUs are idle.
See [storage and recovery authorities](storage-receipt.md).
