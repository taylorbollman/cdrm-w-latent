# Allocation milestone validation

Reviewed 2026-09-30. This note records the correctness and preservation checks
for the first allocation milestone. Native throughput results belong in the
companion results report; this is not a model-quality or precision study.

## Focused CPU checks

The retained final log records **42 passing tests in 23.05 seconds**:
[`cpu-tests-final.log`](../../../.runtime/olmo-gpu-allocation/cpu-tests-final.log).
The focused suite covers the allocation benchmark, independent acceptance and
summary code. It includes:

- One-, two- and eight-rank layout arithmetic, real-row membership and
  row-keyed feedback noise. Eight-rank coverage here is CPU allocation logic,
  not eight-GPU execution.
- Real one- and two-rank CPU/Gloo execution for B and NFR. Three optimizer
  updates match an independently assembled global objective and each other in
  parameters, Adam moments and scheduler state. Physical microbatch counts may
  differ; useful tokens, loss counts and optimizer-update clocks must agree.
- Different CE, latent and KL masks, true document boundaries, unequal rank
  workloads, fully masked dummy rows and a dummy final synchronized slot.
- Authenticated clone ownership, populated Adam-state validation and rejection
  of mismatched counters, parameter order, tensor shapes and missing moments.
- Measured-region accounting, conservative rank timing, graph-release ordering,
  concurrency overlap, idle-tail accounting and rejection of incomplete,
  nonfinite, modified or unmatched evidence. Summary tests also exercise static
  figure export and declared device isolation.

These tests do not measure GPU throughput, certify eight-rank NCCL, or implement
rank-changing production checkpoint recovery.

## Independent tiny GPU acceptance

Both real CUDA/NCCL graph acceptance jobs passed for B and NFR:

- [One rank](../../../.runtime/olmo-gpu-allocation/tiny-graph1-01/report.json)
- [Two ranks](../../../.runtime/olmo-gpu-allocation/tiny-graph2-01/report.json)

Each arm executes three full Adam updates after graph preparation. The reference
uses ordinary, uncaptured model losses on the same global logical rows and
explicit per-pass coefficients. It does not use DDP normalization or the
prepared dense loss layout. Token-by-token eligibility supplies independent
global loss denominators. Keyed jitter is identical for a given real row across
partitions; changing dummy padding does not contribute to the objective.

The largest recorded relative L2 differences across the three updates and all
ranks are below. These are fractions, not percentages. Parameter displacement
means the difference from the shared initial parameters, rather than the norm
of the full pretrained weights.

| Execution | Arm | Raw-gradient relative L2 | Parameter-error / oracle displacement |
|---|---|---:|---:|
| One rank | B | 0 | 0 |
| One rank | NFR | 2.737e-6 | 5.312e-6 |
| Two ranks | B | 3.271e-7 | 2.410e-6 |
| Two ranks | NFR | 2.848e-6 | 5.707e-6 |

All reported per-tensor gradient, parameter and Adam checks passed, along with
useful-exposure accounting, scheduler equality and stable captured addresses.
This is **tiny FP32 same-precision operational acceptance**. It neither clears
the earlier full-model BF16 numerical qualifications nor establishes bitwise
trajectory equivalence when rank count changes. The tiny fixture uses the math
attention backend; native BF16/Flash operation is separately exercised by the
performance cells, without an independent native gradient-equivalence claim.

## Independent implementation review

The review read the benchmark, host supervisor and comparison producer, their
focused tests, and the shared partitioning/DDP/ordered-data code. No blocking
issue was identified for the declared B and NFR performance comparisons.

**Objective and allocation.** The shared campaign runner sums each enabled
objective's valid counts across ranks, applies the existing DDP normalization,
and clips once per logical optimizer update. New layout code changes rank
allocation and accumulation slots, not the global real-row count. Native runs
require 512 real T1024 rows per update and reject mismatched token accounting.
The ordered reader keeps corpus/index bytes pinned and retains true document
boundaries for auxiliary masks.

**Checkpoint preservation.** Native construction authenticates the source
manifest and checkpoint state, verifies model/optimizer ownership and tied
aliases, loads populated Adam moments, and restores the checkpoint's actual
model loss configuration. NFR therefore retains latent weight 1 and KL weight
0.1. The benchmark holds saved optimizer learning rates fixed, uses fresh
benchmark counters and a declared data-prefix cursor, and does not extend or
overwrite the production schedule, RNG state or data cursor. Source weights and
Adam state are cloned into disposable GPU processes. At completion the runner
rechecks the manifest hash and state-file size/mtime/inode; it does not perform
a second full state-file content hash. The initial state verification is a full
content check.

**Concurrent jobs and measurements.** The supervisor assigns disjoint GPU
visibility, independent torchrun rendezvous and new output directories. Both
single-GPU jobs finish warmup before a shared release gate opens. The comparison
uses total useful input tokens divided by the joint measured makespan, including
gaps and idle tails. It reports measured overlap and rejects unsupported
concurrency conclusions. Setup, warmup, final shutdown and checkpoint I/O are
outside this throughput interval. Peak memory is reported per GPU and is not
treated as pooled capacity.

Comparison requires matching checkpoint, objective, runtime, implementation,
physical batch, authenticated data index, starting cursor and update signatures.
The two singles additionally must have identical row-key digests. Pair-versus-
single data identity is established through the pinned deterministic data source
and execution contracts; the summary does not reconstruct and join rank digests
into an independent global row-membership hash.

**Completed-run shutdown.** The first native B pair completed finite updates
but stalled during process-group destruction because captured NCCL graphs were
still alive. Its failed supervisor record and original source snapshot are
retained and excluded from the primary comparison. The corrected wrapper
synchronizes completed work, resets both graphs, drops reducer/model references
and then destroys the process group. A run is marked completed only after this
shutdown succeeds. The replacement B pair and both B single jobs exited
successfully with the corrected source. Unknown CUDA/NCCL failures remain
externally bounded by the supervisor instead of attempting unsafe in-process
recovery.

## Remaining boundaries

- These are disposable performance clones, not production topology migration or
  a continuation past the completed NFR128 checkpoint.
- Native finite updates establish operation at the tested allocation. They do
  not establish matched BF16 optimization trajectories or model quality.
- H200 capacity/bandwidth benefits and full eight-GPU concurrent or distributed
  throughput require measurement on the target machine.
- Final native cells and their evidence retention should be recorded in the
  companion results report after their supervisors report successful completion.

## Completed native allocation evidence

All four primary cells completed with process exit0 and successful graph/reducer
teardown: Bpair02, Bsingles01, NFRpair01 and NFRsingles01. Their six native jobs
completed36 finite optimizer updates total (two warmup plus four measured per
job); all six W&B runs synced. The first Bpair01 teardown failure remains
retained and excluded from the primary comparison.

The final read-only preservation audit passed931 checks with no failures:
`.runtime/olmo-gpu-allocation/closeout-audit.json`. It verifies original B32 and
NFR128 report/manifest hashes, all422 original source-inventory entries, each
native job's source pins and supervisor report authority, completion/teardown,
measured counts, and original checkpoint inode/size/mtime preservation. Full
state content hashes were checked at each import; closeout did not add another
full15GB readback. The audit producer is retained with the runtime evidence.

Both H100s were checked inside the required container after the suite ended:
zero compute processes,0MiB used,0% utilization. No further GPU work is queued.
The final comparison summary independently checks matching contracts, ordered
input authorities, objective accounting, device isolation and overlapping
measurement windows. See results.md for throughput and its scope.
