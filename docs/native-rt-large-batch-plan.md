# Native RT: integration and physical-batch scaling

Updated 2026-09-24. User authorized this milestone and a conditional FA4 memory
comparison at larger batches. See the frozen run
[protocol](reports/olmo-rt-large-batch/protocol.md).

## Decision and motivation

Keep optimized native RT as the working backend. Retain the author-derived
implementation and its qualified comparison as reference evidence. Do not
restart a backend-adjudication campaign as a prerequisite for this milestone.

The user's large-batch concern is correct. RT processes the MLP one sequence
position at a time, so physical per-device batch controls its matrix dimensions.
Section 6.1 of the [RT paper](https://arxiv.org/pdf/2604.21215) identifies B512 as
a useful utilization/memory tradeoff and explicitly explains why gradient
accumulation does not supply the same utilization. Its measurements use H100s;
training is on a single device. Sections 6.1/6.2 use activation checkpointing
and CUDA graphs. The paper's main pretraining models are 150M/300M backbones;
their capacity and statistical batch limits do not automatically transfer to
our roughly 1.177B, 16-layer, width-2048 pretrained OLMo or FBT/NextLat variants.

B64 is a conservative development reference, not our established throughput
optimum. In the older [F3e two-RT-layer measurements](reports/olmo1b-f3e/results.md),
physical B64/T512 reached 19,445 input tokens/s and B128 reached 22,720, a 16.8%
increase. B128 setup peak reserved memory was 76.797 GiB, versus 49.039 GiB
current reserved after setup; allocated peak was 46.106 GiB. These are distinct
measurements, not a demonstration that the memory difference is recoverable.
That F3e capacity path did not retain the later harness's full eager-reference
gradient clones with a live graph, so do not attribute its peak to those clones.
Those runs used half-CE and predate the latest optimizations; use the paired
trend, not their absolute rates, as evidence about today's configuration.

## Scope and sequence

1. **Integrate the accepted ordinary optimizations with native RT.** Use the
   actual step-60000 checkpoint, T512, full valid CE, native RoPE/QK semantics,
   BF16 mixed with FP32 parameters/gradients/moments, and deterministic ordinary
   Flash SDPA. Carry forward rounded compiled ordinary SwiGLU, Dao ordinary
   RoPE and fused AdamW, plus validated native RT table/cast reuse, K/V-only
   writes, Triton historical tiles and recompute backward. Keep ordinary
   activation checkpointing and graph forward/loss/backward. Clipping,
   optimizer and scheduler remain outside the graph. The Dao ordinary option
   has only been measured in ordinary-only runs and needs bounded integration.
   Record every option explicitly; do not change historical resume identity.

2. **Check two representative configurations.** Start with RT-only at indices
   `(0,15)`, then the same selection with FBT K2 and NextLat. K2 retains the
   ordinary bootstrap and one feedback pass; do not silently add RT to bootstrap.
   Reuse existing tests and run small-B/T512 cross-configuration gradient checks,
   followed by each candidate's own graph/full-update checks. Preserve existing
   thresholds and qualifications. A failed new integration check receives a
   bounded investigation before scaling that candidate. A four-RT-layer spot
   check can follow at the chosen operating point; an all-layer sweep is outside
   this initial milestone.

3. **Measure the physical-batch curve, aiming toward B512.** Start B64 and B128;
   then try B192/B256/B384/B512 as memory permits, using only useful intermediate
   points. Stop escalation on an explained capacity limit; refine once below
   it if necessary. Each shape uses a fresh process. Report separate operating
   points for RT-only and combined, rather than forcing one common batch.
   Use full training updates, substantial graph/compiler warmup and repeat the
   best point against its smaller neighbor in reverse order. Keep the data,
   loss coverage, precision and optimizer recipe fixed. This tests execution,
   not learning quality or learning-rate/batch scaling.

4. **Distinguish model memory from measurement overhead.** Record loading,
   validation, preparation, capture and steady-update allocated/reserved peaks
   separately, along with device memory outside the allocator. At large batches,
   avoid retaining duplicate models and full reference gradient/state clones
   merely to time one candidate. Preserve independent small-batch correctness
   evidence and bounded large-batch operational checks. Inspect graph-pool and
   temporary-buffer lifetimes before calling a setup OOM an intrinsic training
   limit. Report any changed validation scope and rerun a common shape to verify
   the harness change itself. Do not count empty-cache calls or omitted required
   optimizer/activation state as an algorithmic memory improvement.

5. **Profile the resulting operating point before further RT fusion.** Measure
   complete-update throughput and time spent in the sequential finish/writer,
   historical attention, ordinary layers, loss and optimizer. If the leaf path
   remains material, the next bounded optimization is compilation of a pure
   contiguous finish/writer region, preserving BF16 rounding, FP32 norm/residual
   math and batched parameter VJPs. Keep recursion/cache mutation outside it.
   Compare at realistic physical batches, not just small development batches.
   A broad activation-library survey is not part of this milestone.

If larger batches hit capacity, compare ordinary FA4 against optimized Flash
SDPA at matched large-batch shapes and the failed boundary, after a bounded
integration check. Existing smaller-batch measurements showed little memory
benefit but do not settle this larger-batch question. Native RT attention stays
on its validated Triton tiles; FA4 is an ordinary-layer execution option.

## Measurements and stopping point

For each mode/shape, retain physical per-GPU batch, accumulation count, global
optimizer batch, valid input tokens/s, CE targets/s, seconds/update, phase times,
parameter counts, analytic matrix-FLOP accounting, setup/steady memory, exact
execution flags, warmup/timing counts and failed attempts. Log graphable runs to
`taylorbollman/pretrained-fbt-rt-nextlat`; retain sources, reports and evidence in
the persistent project and `gs://fast-chunks` using the existing workflow.

The review deliverable is a measured batch-throughput-memory curve and a
comfortable operating point for each configuration. If that point is below
B512, say whether the reason is a memory limit, capture/setup overhead, or a
measured throughput plateau. Leave real headroom for recovery and subsequent
features. Do not describe a memory-constrained smaller batch as intrinsically
faster, or infer that any chosen batch improves learning.

If B512 physical cannot fit after bounded memory diagnosis, record that limit
and proceed with the best comfortable measured point. Accumulation may later
match a desired optimizer batch, but does not reproduce B512 MLP utilization.
Genuine multi-GPU DDP then ZeRO1/2 remains the distributed path: DDP does not
pool memory; optimizer/gradient sharding may permit larger physical batches,
subject to separate integration checks. No change to Q/K normalization, new
quality-training campaign or claim that historical BF16 qualifications are
resolved is implied by this plan.
