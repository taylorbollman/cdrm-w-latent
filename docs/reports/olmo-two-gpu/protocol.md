# Two-H100 functional and resource milestone

Authorized 2026-09-25. Parent plan: `docs/native-rt-single-to-two-gpu-plan.md`.
No language-quality campaign is queued. Save source and evidence after each
stage; retain complete recovery checkpoints in `gs://fast-chunks` before removing
any local copy. The VM may be interrupted. Local SSD is disposable.

## Frozen first-stage scope

Two distinct H100 80GB devices, one process per GPU, NCCL. First verify topology,
peer access, exact collective results and bounded communication timings. Then
exercise genuine eager DDP on all eight tiny RT/FBT/NextLat combinations and
the actual pretrained OLMo ordinary, RT (layers 0/15), and combined K2 models.
Use original OLMo-1B step60000, FP32 parameters/gradients/Adam moments, BF16 mixed
compute, deterministic ordinary Flash SDPA, native RoPE, rounded compiled
ordinary SwiGLU, ordinary activation checkpointing, fused AdamW, native RT's
Triton forward/recompute backward, reused casts/RoPE and KV-only writes.
Actual fixtures are B1 per rank, T512 initially; no padding/cache claims.

Compare identical rank-major microbatches against canonical one-GPU accumulation,
with independent global CE/latent/KL target denominators. Cover unequal counts,
locally empty auxiliary losses, an auxiliary branch used only before the last
`no_sync` microbatch, globally inactive parameters, tied readout and shared FBT
passes. Raw reduced gradients must be inspected before clipping. Check complete
Adam/scheduler updates, globally counted metrics and exact inter-rank replicas.
Reject nonfinite objectives/gradients collectively before any optimizer update.

Prospective reduction-order budgets (same physical microbatch math, no backend
change): raw-gradient global relative L2 <=2e-5; per-tensor relative L2 <=1e-4
and max-error/reference-max <=5e-4. Near-zero tensors use absolute floor 1e-7.
Loss sums/means agree within relative 2e-6 plus absolute 2e-6. After two updates,
parameters agree within atol 2e-7, rtol 2e-6; Adam moments within atol 1e-8,
rtol 5e-4. Any miss is retained and diagnosed, not silently tolerated. All rank
replicas and same-world-size recovery continuations require exact state equality.
Tiny FP32 mathematical tests also use the established close-element checks.

## Recovery and later stages

Save at a cleared-gradient update boundary, with one canonical model/optimizer/
scheduler copy, shared global counters, per-rank data cursor and Python/NumPy/
CPU/local-CUDA RNG. Atomic manifest is the commit marker. Compare the next update
against a reconstructed two-rank continuation, including model, moments, counters
and next RNG draws. Keep rank-zero-only W&B/artifact publication and verify GCS
size/hash metadata before local checkpoint cleanup.

Next validate real DDP forward/backward CUDA graph capture after sufficient
DDP-enabled side-stream warmup (at least11 iterations), preserving persistent
gradients and unused-parameter semantics. A graph-compute/explicit-reduction
fallback must be named distinctly if required. Then compare equal global batches
and comfortable physical per-rank batches, documenting setup and steady memory,
total/per-GPU tokens/sec and communication. Native ZeRO-1 is the initial sharding
candidate; ZeRO-2 is conditional on a useful memory benefit. These stages need
their own recorded settings and results before any readiness claim.

Full-model native-versus-author BF16 qualifications remain open. This milestone
does not reclassify those results or add Q/K normalization/model changes.
