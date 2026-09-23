# Author-derived native RoPE comparison

Prospective bounded protocol, 2026-09-23. The user authorized proceeding directly
after Stage A (merged PR23,5d3e396). No quality training or production backend
replacement is authorized by a timing result alone. Read the source audit here
and the staged plan in `docs/olmo-rt-efficiency-and-author-comparison-plan.md`.
Freeze the final implementation and this protocol before the comparison queue.

## Implementation contract

Use original OLMo-1B step60000 weights (~252B tokens). Native geometry D2048,
H16/head128, SwiGLU8192 per branch, nonaffine LayerNorm eps1e-5, no Q/K norm,
no bias/dropout, native split-half FP32 RoPE. Do not substitute the author's
smaller MLP or ALiBi. Isolated fixtures contain the first1/2/6 checkpoint blocks,
without embeddings, final norm, vocabulary losses, FBT or NextLat. A six-block
fixture is not a pretrained six-layer language model.

The author-derived candidate preserves separate temporary Q/KV GEMMs,
position-major dyadic scheduling, full materialized attention reconstruction,
per-token permanent-writer weight VJPs, chunked local MLP input VJPs and final
batched temporary/MLP VJPs. Retain native packed outer parameters. Invocation-local
leaf aliases receive nested gradients, which are returned explicitly to outer
autograd; do not write real module `.grad` inside the custom backward. This
ownership adaptation must pass shared-call/frozen/graph tests.

Rotate Q before the author's query prescaling; rotate temporary and permanent
K at their actual positions. Do not rotate V. Reuse Stage A FP32 tables and
keep them outside compiled helpers where practical. Compile the same bounded
helper boundaries as the author, not additional local MLP or whole-block code.
Record compiler specializations and fail on graph breaks/fallback/recompile-limit
overflow. Use the existing project limit64, versus upstream13; fresh processes
isolate shapes. Preserve efficient cast reuse with fresh invocation-owned
autocast scopes; no stale casts may cross warmup/capture or weight updates.

Primary author arithmetic is `author_legacy`: BF16 query prescaling and the
author's probability/adjoint rounding. Native uses post-dot scaling and its
existing FP32 probability/error policy. These are explicit differences. An
`fp32_state` author diagnostic, if needed, is labeled separately; neither is
silently substituted for the other. Native primary is Stage A `both/recompute`
with mixed attention, cast reuse and Triton historical kernels. A native
materialized-workspace case is a separate attribution diagnostic.

## Correctness and progression gates

CPU FP32: independent unchanged native scan versus adapted author scan/tiled
outputs, arbitrary-cotangent input and every packed parameter gradient; short
dyadic boundaries, offset/irregular positions, chunk counts1/4, two layers,
shared/frozen calls, parameter/checkpoint ownership, invalid metadata. Independently
count executed mm/bmm work before publishing any new author FLOP formula.
The isolated block ledger is `cdrm/pretrained/rt_block_resources.py`; CPU dispatch
counts check its forward/backward partitions. Each native-width block owns
67,108,864 parameters. The author schedule's dense training work is
50ND²+36NDM versus native K/V-only58ND²+36NDM (N=BT). Attention and native
probability recomputation are accounted separately, including the author's
pointwise singleton reverse tiles. These are logical matrix counts, not hardware
FLOP measurements, and exclude objectives, optimizer and pointwise operations.

Native-width B1/T32 FP32 comparisons must pass before throughput: relative L2
<=1e-4 or absolute L2<=2e-6, max error<=2e-6+1e-4*reference peak; global parameter
gradient relative L2<=1e-4 with the same tiny absolute fallback. Preserve failed
attempts and repair defects rather than widening budgets.

Repeat native B1/T32 using BF16 mixed and record both candidates against FP32
and each other. Paired BF16 screens retain Stage A thresholds: global gradient
relative L2<=1/64, each tensor L2<=1/32 and normalized max<=1/16; output L2<=1/64
and normalized max<=1/16; relative MSE loss difference<=1e-5. Raw-gradient probes
use a fixed Gaussian cotangent scaled by1/sqrt(B*T*D), expected L2 approximately1,
before clipping. Full-FP32 differences are diagnostic, not automatic clearance.
Then perform a paired B8/T512 tiled comparison without retaining a huge naive
sequential graph. Explicitly report which backend/precision screen passed.

Each candidate independently must have exact eager/graph outputs, loss and raw
gradients under its own arithmetic; changed inputs and repeated overwrite;
three eager versus three graph AdamW/clipping/scheduler updates from restored
state with identical final parameters, moments and counters; changed-weight
replay. Nonfinite state, FP32 disagreement or same-candidate operational failure
blocks performance claims for that candidate. A finite cross-backend BF16 gate
miss may coexist with a labeled exploratory timing, but it remains a failed
compatibility screen and cannot authorize replacement. Investigate precision
boundaries before interpreting such a speed comparison.

## Timing and bounded queue

One H10080GB, GPU commands only through the project container. FP32 parameters,
gradients and Adam states; BF16 mixed primary, TF32 off, deterministic settings.
Fixed seeded Gaussian FP32 input/target fixtures, record norm and seed. Full
updates use the same FP32 mean-squared-error objective; report these rates as
block-stack execution, not language-model training throughput.

One training CUDA graph contains persistent gradient reset, forward/objective
and backward. Input/target copies, clipping, AdamW and scheduler stay outside
and are included in complete-update wall time. Persistent input gradients and
parameter gradients are overwritten, not accumulated across updates. Before
using forward/backward CUDA-event splits, verify external event nodes in a
tiny separate container preflight. The split must use the same training graph;
do not infer it by subtracting unrelated captures. If unsupported, retain total
graph timing and label a separate profile instead.

Each fresh capacity process: three preparation updates, sufficient helper
compilation, ten backward warmups, capture, five complete-update samples and
three graph-only replays. Setup/compile/capture time and memory stay separate
from steady samples. Preserve useful author autocast caching versus native
explicit casts; changed-weight checks determine correctness, not identical
implementation mechanisms. No logging/profiling/filesystem I/O inside timers.

Sequence the queue adaptively, avoiding a broad sweep:

1. One block B1/T512 diagnostic, then paired B32/B128. At B32, include native
   control/rope/both once as a bridge to Stage A. Add one native materialized
   case at this shared point to characterize workspace/reconstruction cost.
2. Attempt paired B512 only if B128 memory permits comfortable setup; use B256
   or the smaller common point otherwise. Report physical batches, not equivalent
   accumulation. A supported OOM is a retained capacity result, not permission
   to exhaust the device through repeated maximum-batch searches.
3. Two and six all-RT blocks at B32; B128 if comfortable. Follow up close gaps
   in reverse order before claiming a small gain. Keep bwd_mlp_chunks=4 initially,
   recording it as the prior project's150M setting, not an upstream universal
   default. A larger chunk count is an explicit memory diagnostic if necessary.
4. If the alternate has a repeatable useful advantage, perform a bounded actual
   16-layer integration with RT layers0/15 and one FBT+NextLat compatibility
   check before recommending replacement. Otherwise retain native and identify
   any useful components for a separate follow-up. Do not integrate broadly just
   because the alternate exists.

Report separate forward/backward and complete-update timing, input tokens/s,
setup/steady allocated/reserved memory, compile/capture costs, parameters and
audited matrix work. Full materialization versus bounded recomputation and
different precision/compile/casting boundaries must accompany comparisons.

## Evidence and limits

W&B entity`taylorbollman`, project`pretrained-fbt-rt-nextlat`, new group
`olmo-rt-author-comparison`. Freeze code/protocol and preserve exact source pins,
snapshots, hashes, completed/failed reports, compiler counters, logs and small
plots in a verified bundle under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-rt-author-comparison/`.
Reuse the retained native checkpoint by reference. Few-update diagnostic weights
are disposable unless needed to reproduce a failure. Record checkpoints in Git
and update the handoff throughout; compilation/validation may take hours.

This does not reproduce the paper's trained architecture, published throughput
or quality; it is an author-derived implementation for native OLMo. No new Q/K
normalization, FA4 rewrite, CE/MLP fusion project, padded/cache/transition generality,
long learning, genuine multi-GPU or broader precision clearance follows here.
