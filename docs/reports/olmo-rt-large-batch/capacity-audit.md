# Native RT physical-batch capacity audit

Working assessment, 2026-09-24. The completed RT measurements below establish
that the original B192 failure was caused by validation beside a live graph,
and that removing this overlap permits B192. The corrected B256 attempt then
fails during capture of native RT backward. These are different limits.
Combined capacity, reverse-order repeats, conditional FA4 comparisons and the
bounded profile are still being collected; this is not their final assessment.

All sizes are GiB. These runs use the actual OLMo-1B checkpoint, physical
per-device batch at T512, two native RT layers `(0,15)`, full valid CE,
FP32 parameters/gradients/Adam moments, BF16 projections, all ordinary layers
checkpointed, and graph forward/loss/backward. The primary `compiled-native`
arm keeps native ordinary RoPE and uses rounded compiled ordinary SwiGLU and
fused AdamW. Its RT and combined integration gates passed; the separate Dao
arm's loss-screen miss remains failed. See the [protocol](protocol.md).

Input throughput counts `B*T` once per optimizer update. The runtime metric
`ce_targets_per_second` counts `B*(T-1)` unique next-token target positions per
update; label it unique, or per-pass, CE positions/s. Combined K2 supervises
those positions in both passes, but this rate does not count them twice.
The resource ledger's `analytic_matrix_work.objective_positions_across_passes`
separately includes both passes' CE/latent/KL position work. Do not compare that
doubled work ledger directly with the unique-position throughput denominator.

## Measured RT capacity

Old ordering means `validation-order=live-graph`, cleanup off, runtime
`3fdad1e`. Separated ordering means `validation-order=before-capture`, cleanup
before and after side-stream warmup, runtime `a2bc709`. These preparation
settings are separate comparison cohorts. The repeated B128 shape is the
bridge between them; it does not isolate the effects of the two switches.

| Run | Outcome / physical updates | Input tokens/s | All-phase allocated peak | All-phase reserved peak | Timing current reserved | Timing sampled free | Minimum sampled free across phase boundaries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [RT B128 old](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vvdkbb71) | passed / 8 | 28,010.0 | 45.746 | 78.059 | 76.266 | 1.795 | 0.170 |
| [RT B128 separated](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/154naiyy) | passed / 8 | 28,008.6 | 45.746 | 48.377 | 48.377 | 29.684 | 29.684 |
| [RT B192 old](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w1ytzug4) | validation OOM / 3 | — | 59.417 | 78.361 | — | — | 0.643 |
| [RT B192 separated](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4upxo3z9) | passed / 8 | 29,743.7 | 59.417 | 62.926 | 60.918 | 17.145 | 15.565 |
| [RT B256 separated](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/s0zg48mj) | capture OOM / 3 | — | 73.088 | 78.227 | — | — | 0.197 |

The phase peaks include terminal validation and health, where present; they
are not a difference between current-memory snapshots. Failed runs have no
timed graph-update throughput. Device free memory is sampled at boundaries,
not monitored continuously; the last column is not a measured continuous
minimum. The exposed device reports total capacity **79.179 GiB**.

B128 throughput changes by less than 0.01% in the single matched-shape
comparison, while allocated peak remains 45.746 GiB and reserved peak falls
by 29.682 GiB. This supports attributing the capacity recovery to preparation
and validation storage lifetimes, not reduced model tensor requirements or
changed training arithmetic. The common-shape exact gates pass.

B192 is 6.19% faster than B128 in the first separated-order pair for 50% more
physical examples. Its observed setup and timing headroom exceed the protocol's
approximately 8 GiB target. Its reverse-order repeat confirms29,748.39tokens/s against29,743.65initially,
so B192 is the selected comfortable RT point.
B256 has neither that headroom nor a successful captured optimizer update.

## Why the original B192 failure does not establish a replay limit

The old B192 report completed three preparation optimizer updates, ten
side-stream backward warmups and graph capture. At capture completion it held
60.918 GiB reserved and sampled 17.145 GiB free. The first eager validation
then attempted to allocate a 1.5 GiB ordinary compiled-SwiGLU output of shape
`[192,512,8192]` while the graph remained alive. The OOM diagnostic reported
42.60 GiB in private pools and 78.53 GiB total process usage.

The traceback reaches `compare_graph_cpu -> plan.backward(replay=False)` and
ordinary SwiGLU; the phase marked `validation_initial` failed. The old report's
coarse top-level `stage="capture"` had not been updated after capture, so that
field alone misclassifies the failure. The completed `capture_capture` phase
and failing validation traceback are decisive. Its CPU reference strategy
already avoided a full GPU gradient clone; fresh eager workspace still
coexisted with the retained graph pool.

The separated-order helper takes complete eager CPU references before capture,
checks initial and changed-token replay exactly, and deletes those references
before timing. It checks every requested overwrite. After timing and any
profiles, it snapshots a fresh graph replay at final weights, releases graph
outputs and graph ownership, and only then runs eager comparison. All loss
keys, active gradients, and persistent parameter/gradient storage are checked.
No optimizer state or required parity gate is omitted. Its 37 focused CPU tests
cover lifetime and ownership failures; the B128/B192 runs supply the actual
GPU checks. The old failed run remains retained.

## Allocator accounting and the corrected B256 boundary

Allocated, reserved and device-used memory answer different questions.
Allocated counters track tensor allocations seen by PyTorch. Reserved memory
also includes cached blocks and graph-private storage retained for replay.
Device usage includes memory outside that allocator; here timing device usage
is about 1.12 GiB above current reserved. Resetting peak counters changes the
measurement window, not the allocations or graph lifetime.

At separated B192, timing reports only 17.790 GiB peak allocated, but 60.918 GiB
reserved and 62.034 GiB device used. Graph replay reuses storage established
during capture, without replaying the allocator events that produced its
intermediates. **17.790 GiB is not the full memory requirement of the captured
backward.** Use capture/setup peaks together with current reserved/device usage
and sampled free memory when deciding headroom.

Cleanup likewise does not release live model or graph storage. In B128,
pre-warmup cleanup lowers current reserved from 46.570 to 18.811 GiB without
changing allocated 17.695 GiB. Warmup then peaks at 48.127 GiB reserved, versus
75.887 GiB in the old run. This is consistent with avoiding retained
default-stream cache while building side-stream workspace. The matched run
verifies the combined preparation change; it is not a factorial attribution
of each cleanup and validation option.

Corrected B256 completes all three eager optimizer updates and both eager CPU
reference passes at 73.025 GiB peak allocated. Its ten side-stream warmups
complete at 73.088 GiB peak allocated / 78.227 GiB reserved, with only 0.197 GiB
free at that boundary. Post-warmup cleanup reduces current reserved to
22.191 GiB before capture. Capture nevertheless fails requesting 2 GiB inside
the native RT batched `torch.autograd.grad` at
[`olmo_tiled.py`](../../../cdrm/pretrained/olmo_tiled.py), line 342, after its
full-sequence `_finish` replay. The allocation size is consistent with one
BF16 `[256,512,8192]` tensor; the Python traceback does not identify the precise
derivative buffer.

This establishes a capture boundary for the tested runtime and allocator
configuration after the known validation overlap was removed. Three successful
eager updates prevent calling it an eager-training impossibility. Capture
allocation/lifetime details may still admit improvement. It is not evidence
that ordinary attention alone caused the failure. Error snapshots occur after
exception unwinding; their current counters and the OOM diagnostic's private-
pool accounting need not describe the identical allocation instant.

## Genuine fixed and batch-scaled costs

The RT report counts 1,185,153,024 resident parameters, including the inactive
fusion branch, and 1,176,764,416 gradient/optimizer-owned parameters. Their FP32
storage is approximately 4.415 GiB parameters + 4.384 GiB gradients + 8.768 GiB
Adam moments = **17.566 GiB**, before buffers, inputs, casts, activations,
workspace and allocator overhead. The combined inventory has 1,267,879,936
active parameters: 4.723 GiB each for parameters and gradients, plus 9.446 GiB
for the moments, a **18.893 GiB** subtotal. Optimizer scalar state is tiny.
Changing batch does not remove these fixed floors.

Native RT has several additional costs visible in
[`olmo_tiled.py`](../../../cdrm/pretrained/olmo_tiled.py):

- The custom forward saves layer input and completed output at line 220.
  Native embedding/residual storage is FP32. Ordinary checkpointing retains
  its FP32 boundary inputs as well; it removes internal saved activations,
  not the entire full-sequence state at every checkpoint boundary.
- Backward rebuilds full-sequence Q/K/V projection graphs, then allocates
  FP32 attention/adjoint arrays, including `dk`, `dv`, `ga` and `gamma`
  at lines 276–280. Query/self adjoints follow at lines 324–328.
- `_finish(xg, ...)` at line 341 recreates the MLP across the complete physical
  `B*T` before one batched parameter VJP. The projection/attention graphs and
  adjoints are still in scope. `_finish` uses the native eager SiLU/product at
  lines 70–75; compiling **ordinary** SwiGLU does not fuse this native RT path.
- [`olmo_rt_memory.py`](../../../cdrm/pretrained/olmo_rt_memory.py) bounds
  attention reconstruction with row statistics and tiled scratch. It avoids
  a global `T*T` probability/error tensor, while retaining linear-sized
  attention, projections and adjoints. It does not bound the MLP replay.

Individual tensor sizes illustrate the physical-batch cost; these rows must
not be added as an exact peak inventory because views, shared storage and
lifetimes matter.

| One tensor at T512 | B128 | B192 | B256 | B512 |
| --- | ---: | ---: | ---: | ---: |
| FP32 `[B,T,2048]` state/adjoint | 0.5 | 0.75 | 1 | 2 |
| BF16 `[B,T,16384]` packed up/gate projection | 2 | 3 | 4 | 8 |
| BF16 `[B,T,8192]` MLP branch/output | 1 | 1.5 | 2 | 4 |

RT-only has 14 ordinary checkpoint calls; combined K2 has 16 in bootstrap and
14 in its feedback pass. At B512, their nominal FP32 checkpoint-input payloads
are respectively 28 and 60 GiB. These count boundary references rather than
additional storage independent of all other states: for example, an RT output
can also be the next checkpoint's input. The four saved input/output references
across two RT blocks cannot simply be added again without a lifetime inventory.

The observed full-backward allocated peaks rise by about 13.67 GiB per
additional 64 examples from B128 through the completed B256 warmup. A simple
unchanged-source linear projection gives approximately **127.8 GiB at B512**.
That is a planning estimate from these shapes, not an executed B512 result or
an exact graph-memory model. Together with the actual B256 capture failure,
it makes B512 implausible for this unchanged 1.177B model on the exposed GPU.
Accumulation leaves the per-invocation RT matrix dimensions unchanged and does
not provide B512 utilization.

## Combined scope and bounded follow-ups

Combined K2 starts with an ordinary bootstrap, then executes the feedback pass
with RT at `(0,15)`; it is not two identical RT passes. Both pass outputs and
their differentiation paths remain part of the objective, with fusion/NextLat
parameters and CE/latent/KL work added. Its boundary states and fixed optimizer
floor must be measured separately. The RT B192 operating point is not a
combined recommendation.

| Separated-order combined run | Outcome / updates | Input tokens/s | All-phase allocated peak | All-phase reserved peak | Timing current reserved | Timing sampled free |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Control B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/oh7n35js) | passed / 8 | 11,198.0 | 38.923 | 41.256 | 41.256 | 36.764 |
| [Compiled-native B64](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aqw8yhev) | passed / 8 | 11,707.6 | 38.925 | 41.713 | 41.213 | 36.809 |
| [Compiled-native B128](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mzt0ukzu) | passed / 8 | 12,410.9 | 58.095 | 65.971 | 65.113 | 12.877 |

The B64 control's pre-timing setup reserved peak is 40.871 GiB, while its
complete-update timing reaches 41.256 GiB. This is another reason to inspect
all phases when assessing headroom. Compiled-native improves throughput by
4.55% in this single matched pair, with essentially unchanged allocated peak.
Its minimum sampled free memory across all phase boundaries is 36.613 GiB; its timing boundary has
36.809 GiB free. Both B64 shapes clear all five gates and eight updates.
Combined B128 improves throughput by6.01% over candidate B64 and completes all
five gates and eight updates. Its minimum boundary-sampled free memory is
12.520GiB, above the approximately8GiB target. Reverse-order repeats confirm12,415.46tokens/s atB128 and11,698.26atB64.
B128 is the selected comfortable combined point. Conditional FA4 is assessed below.

The conditional ordinary-FA4 measurement does not remove the RT boundary:
B192 setup allocated/reserved peaks match SDPA59.417/62.926GiB, its current
reservation rises by0.75GiB, and B256 still fails graph capture after three
eager updates. Combined B128 FA4 saves1.857GiB setup reservation and1GiB current
reservation, while allocated peak stays58.095GiB. The B8 FA4 integration screens
fail (RT loss-only; combined also output/gradient failures), despite exact own
operational checks. These qualified capacity measurements do not justify
switching the working backend. See [results](results.md) for precise budgets.

The remaining improvements should be measured against this demonstrated
boundary, preserving the native model and the existing numerical budgets:

1. Completed: the matched combined curve, reverse repeats and conditional
   ordinary-FA4 comparison establish B192 RT and B128 combined. FA4 changes
   ordinary attention only and does not remove the observed RT capture limit.
2. Completed: useful-batch graph-only and complete-update profiles. See the
   [profile audit](profile-audit.md). A later bounded finish/writer fusion may
   reduce device work; any memory claim needs separate measured evidence.
3. Inspect native backward tensor lifetimes around the final batched VJP.
   Release genuinely dead attention/reconstruction references or split/chunk
   the MLP VJP only in a separately frozen experiment. Autograd may retain
   tensors after a local name is deleted, and chunking can change gradient
   reduction order; preserve raw-gradient and exact own graph/update checks.
4. Consider different checkpoint boundaries if measured saved FP32 states
   dominate. This trades recomputation and graph complexity for storage and
   requires new combination checks; reducing residual precision changes native
   arithmetic and is not a preparation-only fix.
5. A future genuine multi-GPU path may shard Adam state and gradients after its
   DDP baseline. DDP replication alone does not lower the per-device fixed
   model/state floor, and sharding that floor does not remove these batch-scaled
   activations. No sharding or additional-device result is claimed here.

Raw sources are each run's immutable `source-snapshot`, `protocol.md` and
`report.json` under `.runtime/olmo-rt-large-batch/`. The run names correspond
to the linked W&B records. Failed source revisions and reports must remain in
the final retention set; later successful preparation does not rewrite them.

## Bounded code review

Read-only review of the capture observer/cleanup changes against `main`, the
CPU-reference helper and its harness integration found no training/capture
correctness regression. Default capture preserves stream ordering and its
existing synchronizations. Enabled cleanup synchronizes before releasing unused
cache; persistent tensors remain live. Initial references are deleted before
timing, and terminal graph release follows the optional profiled optimizer
update. The tested exact ownership/storage gates remain intact.

One nonblocking failure-reporting issue remains in the frozen harness:
[`MemoryPhases.phase`](../../../scripts/olmo_rt_large_batch.py), lines 319–325,
calls `observe(name, "error")` without guarding a secondary observer failure.
If memory snapshotting or report persistence raises while handling an original
OOM, that second exception replaces the OOM and can label the run `failed`
instead of `oom`. The static capture `_preparation_phase` already preserves the
original exception correctly; the harness wrapper needs the same treatment
and a focused CPU regression test after the active GPU queue finishes. No
retained run encountered this secondary failure, so current measurements and
failure localization remain valid. No frozen runtime source was edited for
this review.
