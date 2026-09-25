# Two-H100 integration, recovery and scaling

2026-09-25. **Completed engineering milestone.** All planned functionality,
recovery and bounded capacity stages are complete and retained. No job is queued.

Native RT now works with real two-rank DDP, captured NCCL backward, and
recoverable training state in the tested configurations. At the same global
batch of 128, two H100s improve full-update throughput by **1.69× for RT** and
**1.89× for RT + K2 FBT + NextLat**. One strict comparison between independently
evolving BF16 trajectories remains failed; the corresponding fixed-state
distributed updates pass the original budgets. This milestone establishes
engineering functionality and costs, not model-quality improvement.

## Configuration and terminology

The source model is the original **OLMo-1B step 60,000**, approximately 252B
pretraining tokens: 16 layers, width 2,048, 16 heads, SwiGLU width 8,192 per
branch, tied embedding/readout and native RoPE. Its checkpoint SHA256 is
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.

- **RT:** native temporal recurrence at layer indices **0 and 15**; the other
  14 layers use ordinary attention. This is not an all-16-layer RT measurement.
- **Combined:** K2 FBT plus RT and NextLat. K2 means two backbone passes: an
  ordinary bootstrap pass followed by the feedback pass with the selected RT
  layers. Fusion projects/gates the previous pass's representation into the
  next pass. NextLat adds the training-only latent predictor and auxiliary losses.
- **Precision/execution:** BF16 mixed with FP32 model parameters, reduced
  gradients and Adam state; TF32 and autocast weight caching disabled. Native
  RoPE, rounded compiled ordinary SwiGLU, fused AdamW, ordinary activation
  checkpointing and deterministic Flash SDPA remain the selected path. Native
  RT uses its Triton tiles, recompute backward, reused casts/RoPE and K/V-only
  writes. Q/K normalization and model math are unchanged.
- **Batch:** throughput rows use T512, full valid CE and the existing NextLat
  fixture. “B64/rank” means 64 physical sequences on each GPU, global batch 128,
  with one microbatch per update. Original input tokens count once despite K2.

The machine has two H100 80GB HBM3 devices with NV18 connectivity and verified
bidirectional peer access. Container versions: PyTorch
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3 and NCCL 2.30.5. Genuine NCCL sum checks
pass from 4 bytes through 256 MiB. Median isolated 25 MiB and 256 MiB all-reduce
times are 0.133 ms and 0.918 ms; these are not measurements of training overlap.

## What was implemented and checked

The eager trainer routes forward and objective construction through DDP. It
normalizes CE, latent and KL sums separately by their global target counts,
handles accumulation with `no_sync`, clips after reduction, and preserves
`grad=None` for globally unused parameters. The graph runtime uses actual DDP
forward/backward and captures its NCCL collectives. It warms up for at least
11 backwards, freezes the rebuilt gradient storage, and keeps clipping, Adam,
scheduler and health/metric checks outside capture.

| Completed check | Scope and outcome |
| --- | --- |
| Tiny eager feature matrix | All eight RT/FBT/NextLat combinations pass three accumulated updates each. Includes unequal loss counts, local empty auxiliary objectives, predictor use only in an earlier `no_sync` microbatch, and a globally unused predictor. |
| Actual eager ordinary and RT | Two complete updates each pass against canonical rank-major references; exact rank replicas. RT raw-gradient relative L2 is 3.73e-9 then 6.48e-9. |
| Actual combined anchored updates | Two fixed-state complete updates pass unchanged budgets; gradient relative L2 is 2.34e-9 then 3.66e-9. Independent-trajectory qualification remains below. |
| Tiny combined and actual RT/combined graphs | Initial eager/graph raw loss and gradient equality plus two changed-input complete Adam comparisons pass. Four compound checks per rank in each correctness run. |
| Actual RT eager recovery | 24 coordinated checks pass; exact next raw gradients, loss, full model/Adam/scheduler/counters, cursor and local RNG draws after reconstruction. Four physical updates per rank, logical endpoint three. |
| Actual combined graph recovery | 42 coordinated checks pass. Both reference and restored branches rebuild graphs; exact next gradients, loss and complete state. Six physical updates per rank, logical endpoint five. |
| Actual combined ZeRO-1 | Two fixed-gradient comparisons with fully replicated Adam are exact; local moment ownership and bytes are checked. Consolidated checkpoint reproduces the third update exactly. 13 compound checks per rank; four distributed updates plus two reference Adam steps per rank. |
| Actual combined ZeRO-1 graph integration | Initial and changed-weight terminal raw eager/graph checks pass, along with exact parameter replicas and local moment partition/health checks. B1/rank integration is not representative large-batch throughput. |

Counts describe different scopes: a coordinated recovery check already checks
both ranks, while graph checks are repeated on replicas. They must not be added
as independent scenarios. Physical optimizer executions include duplicated
reference/recovery branches, not steps of a single learning run. Warmup and
capture backwards do not count as optimizer updates.

Recovery saves canonical model/optimizer state, scheduler/counters, per-rank
data cursors and Python/NumPy/CPU/local CUDA RNG state, with a hashed manifest
as commit marker. Loading verifies source/configuration/ownership and world
size before mutation. **These probes reconstruct models, optimizers and DDP
within the same process group.** They do not prove a fresh `torchrun` restart,
changed-world resharding or checkpointing a live graph. Static graph layouts
also do not establish dynamic padding or changing auxiliary-loss participation.

## ZeRO-1 state and checkpoint fixes

ZeRO-1 retains replicated parameters and gradients and shards only Adam state.
Native partitioning, fused local Adam and parameter broadcasts remain in use;
optimizer and broadcasts run outside the compute graph. The installed native
loader put scalar Adam steps on CPU and duplicated state in the outer optimizer.
Our explicit global-to-local loader uses the underlying Adam loader's placement
policy, preserving FP32 moments and CUDA fused-step counters without modifying
the caller's checkpoint. Actual local state is `optimizer.optim`; the outer
`optimizer.state` is intentionally empty.

Consolidation exposed a separate checkpoint bottleneck: native sender byte-to-
tensor conversion iterated over a multi-gigabyte serialized payload. A scoped
buffer-view conversion preserves serialized bytes, broadcast ordering, native
consolidation mapping and checkpoint format. It does not change optimizer math.
The corrected actual combined checkpoint save took **83.4 seconds**, including
consolidation, disk writing and hashing, excluding GCS upload. Tiny and full-model
exact recovery pass after the change. The original run's several-minute save
was only coarsely observed; we do not claim a precisely measured end-to-end
speedup. See [the transport audit](zero1-consolidation-transport.md).

At B1 graph integration, combined Adam state is approximately 5.071/5.072 GB
per rank instead of 10.143 GB replicated on each rank. These are tensor bytes,
not net VRAM savings: activations, buckets, graph pools, allocator slack and
temporary consolidation staging remain. ZeRO-2 and parameter sharding are not
implemented or established by this result.

## Matched global-batch scaling

Each pair uses the same global examples, T512 and global physical batch 128.
Single-GPU input concatenates the two rank fixtures. Each row includes three
preparation updates and five timed complete updates, with exact own eager/graph
raw checks before and after changed weights. This does not assert bitwise
equivalence across the different physical batch shapes.

| Mode | GPUs × local batch | Aggregate input tokens/s | Seconds/update | GPU-microseconds/token | Setup peak allocated GiB/GPU | Setup peak reserved GiB/GPU | Steady reserved GiB/GPU | Sampled free GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT | 1 × 128 | 28,000.26 | 2.34055 | 35.714 | 45.746 | 48.377 | 48.377 | 29.697 |
| RT | 2 × 64 | 47,384.11 | 1.38308 | 42.208 | 36.463 | 58.135 | 39.879 | 37.397 |
| Combined | 1 × 128 | 12,361.82 | 5.30148 | 80.894 | 58.095 | 65.113 | 65.113 | 12.891 |
| Combined | 2 × 64 | 23,405.97 | 2.79997 | 85.448 | 43.649 | 74.168 | 46.707 | 30.527 |

RT scales **1.6923×** and combined scales **1.8934×** in elapsed throughput.
Total GPU time per token increases **18.2%** and **5.6%**, respectively. The
second GPU therefore shortens wall time, with a modest-to-material resource
cost depending on the configuration. These bounded five-update measurements
are directional engineering estimates, not long-run throughput guarantees.

Timing includes batch validation/copy, captured compute/reduction, coordinated
health/loss checks, clipping, Adam and scheduler. It excludes fixture construction,
outer timing barriers, reporting, hashing and checkpoint I/O. Memory columns
are phase allocator peaks; device usage is sampled at phase boundaries, not
continuously. Setup can exceed steady memory substantially. Eager-warmup cache
is released before capture; a large setup reserved peak is not necessarily a
live activation requirement. Conversely, replay reuses graph-pool storage, so
the roughly 22 GiB reported as allocated during RT replay is not the full graph
footprint. Consider allocated and reserved peaks, post-capture reservation and
sampled free memory together; no one column establishes comfortable capacity.

W&B: [single RT](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/eezfn9r8),
[DDP RT](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ahbhvxmx),
[single combined](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/watsz35j),
[DDP combined](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zt5q993w).

### Larger physical batches

All rows pass initial and changed-weight terminal raw eager/graph checks and
their replica/state-ownership checks. Memory uses the larger rank value for
allocated/reserved columns and the smaller rank value for free memory, sampled
after capture/timing. ZeRO-1 steady memory includes parameter-broadcast buffers.

| Backend | Mode | Local / global batch | Status | Aggregate tokens/s | Setup peak allocated GiB/rank | Setup peak reserved GiB/rank | Steady reserved GiB/rank | Sampled free GiB/rank |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DDP | RT | 128 / 256 | Passed | 55,767.65 | 50.132 | 77.461 | 52.492 | 24.783 |
| DDP | RT | 192 / 384 | Passed | 59,034.62 | 63.806 | 77.566 | 66.813 | 10.467 |
| DDP | Combined | 128 / 256 | Passed | 24,657.19 | 62.819 | 77.559 | 73.836 | 3.365 |
| ZeRO-1 | RT | 192 / 384 | Passed | 58,187.06 | 59.432 | 77.566 | 62.775 | 14.504 |
| ZeRO-1 | Combined | 128 / 256 | Passed | 24,398.21 | 58.095 | 77.508 | 69.664 | 7.537 |

For RT, B192/rank improves throughput another 5.9% over B128/rank while reducing
sampled free memory from 24.8 to 10.5 GiB per GPU. Update times are 2.35032 and
3.33038 seconds. The larger physical matrix helps throughput, as expected from
RT's batch sensitivity; it also changes the global optimizer batch. This is a
systems comparison, not a learning-efficiency comparison. Both rows retain high
transient setup reservations even though cache release leaves usable headroom.
W&B: [RT B128/rank](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/o5rhk350)
and [RT B192/rank](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aw43q0xt).

Combined DDP B128/rank gains only 5.35% throughput over B64/rank, with
3.37 GiB free versus 30.53 GiB. It fits this fixed diagnostic, but is too tight
as a default for adding features. At B128/rank, ZeRO-1 trades approximately
1.05% throughput for 4.17 GiB more minimum sampled headroom. RT ZeRO-1 at
B192/rank trades approximately 1.44% throughput for 4.04 GiB more headroom.
These small speed differences are directional five-update observations.

Recommended operating points:

- **Development and architectural additions:** RT DDP B128/rank (55.8k/s,
  24.8 GiB free); combined DDP B64/rank (23.4k/s, 30.5 GiB free).
- **Frozen larger-batch configurations:** RT ZeRO-1 B192/rank (58.2k/s,
  14.5 GiB free); combined ZeRO-1 B128/rank (24.4k/s, 7.5 GiB free).
  The latter has moderate headroom, so additions still need a shape check.
- **Unmodified RT with replicated Adam:** DDP B192/rank also works (59.0k/s,
  10.5 GiB free). ZeRO-1 is optional; no global runtime default changed.

At the same physical B128/rank, DDP reaches approximately 1.99 times the
single-GPU rate for both modes, while processing twice the global batch. This
weak-scaling result differs from the matched-global-batch speedups above.
Do not infer faster learning per optimizer step from either systems comparison.

W&B: [combined DDP B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nmv7nu8r),
[RT ZeRO-1 B192](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/m11j5e6b),
[combined ZeRO-1 B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xsg5anc0).

No gradient bucket-view option is adopted. DDP does not pool VRAM; global B512
would require physical B256 on each GPU unless accumulated. Accumulation does
not supply a larger physical RT matrix. The previous one-GPU B256 capture failed;
this milestone does not establish two-GPU B256/rank feasibility, and does not
repeat that boundary merely to maximize usage. ZeRO-2 is deferred until its
extra memory supports a required configuration. This scope does not establish
physical B512 or all-layer RT.

## Parameters and arithmetic accounting

| Component | RT execution | Combined execution |
| --- | ---: | ---: |
| Backbone | 1,176,764,416 | 1,176,764,416 |
| Active fusion | 0 | 8,388,608 |
| NextLat predictor, training only | 0 | 82,726,912 |
| Training architecture | 1,176,764,416 | 1,267,879,936 |
| Deployable architecture, excluding predictor | 1,176,764,416 | 1,185,153,024 |

The RT harness still registers a dormant fusion module, so its actual resident
parameter count is 1,185,153,024, slightly above its executed architecture.
Resident-state ledgers count those bytes; parameter comparisons should not
silently treat them as active RT capacity. Combined has 1,267,879,936 registered
parameters. Tied tensors are counted once.

At global B128/T512, analytic matrix work per update is **636.7–677.1 TFLOPs**
for RT and **1,362.9–1,449.4 TFLOPs** for combined, identical within each matched
one-/two-GPU pair. These estimates include the modeled checkpoint/recompute
matrix work and count multiply-add as two FLOPs. They exclude normalization,
RoPE, other elementwise work, optimizer/clipping, communication, launch overhead
and hardware padding. They are not measured hardware FLOPs or utilization.

## Five retained failures and numerical qualification

1. **`actual-eager-01`: combined independent update 2.** Losses/counts and all
   raw-gradient budgets pass (global relative L2 **1.3509761003e-5**), and rank
   gradients/full state match exactly. However, **20 parameter and 7 moment
   tensors** fail the strict elementwise complete-update budgets. Maximum
   parameter absolute difference is **1.383945346e-6**; maximum parameter-tensor
   relative L2 is **9.096305447e-8**. The run remains failed. Restoring canonical
   weights/Adam/scheduler/counters between comparisons makes both fixed-state
   updates pass unchanged budgets. This supports reduction-order/BF16 trajectory
   divergence as an explanation; it does not demonstrate that such divergence
   is harmless over long training or clear the older native/author BF16 issue.
2. **`tiny-graph-01`:** the installed DDP CUDA path rejected a forward with no
   input arguments. Passing the existing static token tensor fixes the call
   contract without changing model arithmetic. Zero optimizer updates.
3. **`tiny-graph-02`:** restoring unchanged persistent buffers advanced their
   versions and triggered the prepared-state guard. Restore parameters only,
   while validating fixed-buffer equality/storage/version. This failed attempt
   executed **four optimizer updates per rank**, not zero.
4. **`tiny-graph-recovery-01`:** a missing dependency-recorder output-directory
   argument caused a setup failure; the call was corrected. Zero updates.
5. **`rt-graph-01`:** omitted required NCCL asynchronous-error settings were
   rejected before process-group initialization. Corrected launcher passes.
   Zero updates.

No numerical tolerance was loosened to turn a failed run into a passing one.
See [the fixed-state follow-up](fixed-state-followup.md) for the frozen diagnostic
decision. Successful same-candidate graph/recovery checks do not erase a failed
cross-execution comparison.

## Evidence, tests and next decision

All reports retain sources, configuration, errors and W&B identifiers. The
principal runtime pins are `fe5ff69` for early recovery, `d88be56` for graph
integration tooling, `cf8f414` for the matched RT measurements, and `bfa3649` for
the consolidation transport change. Later graph/combined measurements record
`6921c53`; the fast full-model recovery records `18eebc9`. Per-report source
hashes, not just branch HEAD, establish the implementation measured. The larger
RT DDP rows and RT ZeRO-1 record `1a67303`; the final combined ZeRO-1
row records `b615830`, with the same runtime implementation.

**Final audited inventory:** 31 completed stage reports: 26 passed and five
failed, with no unfinished stage or OOM in this milestone. All 31 stage retention
receipts are verified, including eight checkpoint stages. All 4,492 report-pinned
source snapshot pairs rehash successfully, with no archive/report-retention
mismatch. The two pre-execution setup stubs (`rt-graph-01` and
`tiny-graph-recovery-01`) lack full report source pins and remain explicitly
listed as such; their available setup/error evidence is retained.

The inventory counts 173 distributed optimizer executions (346 rank calls),
16 single-GPU benchmark executions and 50 canonical-reference Adam calls:
412 calls in total across engineering/reference branches, not a learning run.
See [storage receipt](storage-receipt.md), [test ledger](test-ledger.md) and the
retained audit for provenance. Compound checks are not summed into this count.

The final capacity-integration CPU scope has **77 passing tests**; the transport
change has a **54-test focused scope**. Earlier 91-, 129- and other reported
scopes overlap; do not sum them into a distinct-test total. GPU check scopes are
listed above rather than inflated by adding replicated or nested checks.

Evidence and checkpoints are retained under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/`,
with local receipts in `.runtime/olmo-two-gpu/retention/`. Actual generated
checkpoints are approximately 13.2–14.2 GiB each. Large local copies were removed
only after remote size/MD5/SHA-metadata verification and local hash confirmation;
manifests and cleanup receipts remain. Evidence archives additionally receive
downloaded-SHA verification. Original pretrained artifacts use the existing O1
manifest/receipt. No credential values are part of this report.

This is the planned review point. Before a long interruption-sensitive campaign,
add a fresh-process checkpoint restart rehearsal for the chosen execution path
and real data cursor. Then select a short learning pilot or longer ordinary
baseline using these measured costs. Dynamic graph layouts, additional RT
layers and changed context lengths remain scoped follow-ups. Existing Q/K and
precision qualifications remain; no model-quality conclusion follows from this
milestone. See [usage.md](usage.md) for execution and recovery contracts.
