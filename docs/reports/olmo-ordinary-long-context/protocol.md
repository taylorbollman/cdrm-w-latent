# Ordinary OLMo T2048: prospective single-H100 benchmark protocol

2026-09-28. User authorized base OLMo throughput/memory at sequence length 2048
versus the saved length-512 reference, and T2048 with and without FA4. User
clarified that T512 already has measurements: **do not rerun T512**, including
FA4 at that length. No RT, FBT, NextLat or quality-training experiment here.

## Fixed model and comparison

Original OLMo-1B step60000 (~252B pretraining tokens), 16 layers, D2048,
16 heads/head128, SwiGLU8192 per branch, tied50304 vocabulary, native nonaffine
normalization and no added Q/K normalization. Active backbone1,176,764,416
parameters; unused frozen32MiB fusion wrapper is separately accounted for.

Use the accepted Dao native-FP32 RoPE, rounded compiled ordinary SwiGLU,
all-layer activation checkpointing, fusedAdamW, full valid causal attention
and CE position chunks2048. BF16 mixed with FP32 parameters/gradients,
residuals/norms/Adam, TF32off and autocast weight cacheoff. Deterministic
execution. The only T2048 arm difference is ordinary attention: forced PyTorch
Flash SDPA versus installed FA4 CuTE. No dependency upgrade or kernel tuning
is planned unless required for execution; snapshot actual imported dependencies.

Use the latest single-reference harness/paired fixture from the two-GPU
milestone so its timing and capture-memory procedure match the saved reference.
One GPU, one physical microbatch/update, no gradient accumulation. T2048 has
2047 CE targets per sequence; T512 has511. Input tokens count B times T.
The fixture has independent full-valid repeated-text rows with deterministic
token changes; it is a compute benchmark, not real-data loading or quality.

Saved reference: `.runtime/olmo-ordinary-two-gpu/ordinary-dao-single-b128-01/`:
B128/T512,44,041.286tokens/s,37.518GiB peak/steady reservation and40.912GiB
sampled free. Retain its report/source hashes/GCS receipt by reference and label
it historical. Confirm current hardware/software, but do not claim same-day
interleaving. T2048/B32 has the same65,536inputtokens/update.

## Bounded measurements and checks

1. Confirm container, idle H10080GB, installed FA4/Dao sources, checkpoint
   manifest, disk headroom and W&B. Use the project launcher with installed
   Flash-Attention, not the vendor-shadowed namespace.
2. Extend only benchmark selection/reporting to ordinary T2048 and explicit
   FA4; retain old defaults and unsupported-case guards. Focused CPU tests
   check selection, dependencies and unchanged ordinary/RT behavior.
3. One B2/T2048 same-state FA4-versus-SDPA comparison with both arms using the
   same Dao/compiled/fused settings. Keep existing output/raw-gradient/loss
   budgets unchanged: globalgradientL2<=1/64, per-tensorL2<=1/32,
   max/referencepeak<=1/16; outputL2<=1/64/max<=1/16; relative loss<=1e-5.
   Preserve finite numerical-only misses as failed comparisons while finishing
   own operational checks if safe. Do not widen thresholds or start a broad
   numerics investigation. Structural, nonfinite, dispatch or own graph/update
   failures stop dependent capacity work until understood.
4. Start matched T2048 SDPA/FA4 at B16, then B32. Consider B48 or B64 only when
   headroom and throughput make them informative. Stop scaling at a useful
   plateau with comfortable memory; do not seek the last GB or require an OOM.
   Repeat the selected matched pair in reversed order to check a small speed
   difference. Aim for4primarycapacityrows+2repeats, with at most4additional
   capacity rows if the batch trend needs clarification.
5. Each row starts from the same checkpoint in a fresh process: three actual
   Adam preparation updates, eleven backward warmups and five timed complete
   updates. CUDA graphs capture forward/loss/backward; clipping, fusedAdam,
   scheduler and health checks remain outside. Timing includes input validation
   and copy, and excludes compilation, reference snapshots, logging and I/O.
6. Untimed actual dispatch must demonstrate selected ordinary attention, Dao
   RoPE and compiled SwiGLU, with no fallback. Each row checks exact own initial
   eager/graph raw gradients/loss and terminal changed-weight parity. Store
   eager references on CPU before capture; release transient cache before
   capture and release the graph before terminal eager verification. Record
   all actual optimizer updates, including any optional parity branches.

The older Dao and FA4 T2048 comparisons retain tiny strict relative-loss
failures while passing output/gradient and own graph/full-update checks. They
are not cleared by a throughput result. A new finite numeric miss may support
qualified timing, not an unqualified numerical-equivalence claim.

## Memory, evidence and closeout

Record phase-specific peak allocated/reserved memory, steady graph reservation,
sampled device-free memory and optimizer/parameter/gradient inventories. Setup
allocator cache and graph-pool reservation must not be described as live
activation sizes. Free memory is sampled, not a continuously measured minimum.

Freeze runtime sources/protocol before execution; changed runtime gets a new
revision and run directory. Preserve every attempt/failure. W&B entity
`taylorbollman`, project`pretrained-fbt-rt-nextlat`, group`olmo-ordinary-long-context`.
Local evidence `.runtime/olmo-ordinary-long-context/`, source/dependency snapshots,
raw reports, logs, plots and receipts retained in `gs://fast-chunks` using the
existing per-stage retention helper and a fresh timestamp under its required
`cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/` prefix. Original O1 weights are
already retained; these disposable short updates need no new full checkpoint.

Report matched-token T512/T2048 performance and T2048 SDPA/FA4 speed/memory,
recommend a comfortable T2048 operating point, update compaction handoffs and
close the PR. Production attention defaults do not change automatically.
