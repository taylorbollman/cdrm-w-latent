# Combined OLMo T1024: prospective single-H100 benchmark

2026-09-28. The user is considering length 1024 as a compromise between the
saved combined-model T512 throughput and the measured T2048 cost. FA4 is set
aside. Implement a bounded T1024 Flash-SDPA benchmark of the same combined
model, then review the context-length choice. The proposed approximately
500-million-token plus SFT screen is planning context, not a training launch.

## Fixed model and execution

Original 16-layer OLMo-1B step60000 (~252B pretraining tokens), width 2048,
16 heads/head128, SwiGLU8192 per branch, tied 50304 vocabulary and native
nonaffine normalization. No added Q/K normalization. K2 FBT: ordinary bootstrap
followed by attached feedback with native RT at indices 0 and 15. Alpha, beta
and gamma are 1. Both passes have CE + NextLat SmoothL1 + KL, all coefficients
1; pass losses are summed. RT is not active in every layer or the bootstrap.

Keep native ordinary RoPE, rounded compiled ordinary SwiGLU, fused AdamW,
all ordinary-layer activation checkpointing, and native RT weight-cast/RoPE
reuse, K/V-only permanent writes, Triton tiles and recompute backward. BF16
mixed: FP32 parameters, gradients, residuals, normalization and Adam state;
TF32/autocast weight cache off. Forced PyTorch Flash SDPA for ordinary attention.
No FA4, dependency upgrade, production arithmetic or default change.

Active training parameters: 1,267,879,936 = backbone 1,176,764,416 + fusion
8,388,608 + training-only NextLat 82,726,912. Deployable with fusion:
1,185,153,024. CE position chunks 2048; KL position chunks 128. Full-valid,
independent row documents with full CE/latent supervision and response-half KL.
One physical batch/update, one GPU, no accumulation. Input tokens count once;
K2 pass-token work is twice this amount. At B64/T1024 the per-pass CE/latent/KL
counts are 65,472/65,472/32,768.

## References and measurements

Reuse the saved reports without new T512 or T2048 GPU runs:

- `.runtime/olmo-two-gpu/single-combined-b128-01/`: B128/T512,
  12,361.82 input tokens/s, 65.113 GiB reserved, 12.891 GiB sampled free.
- `.runtime/olmo-combined-long-context/sdpa-b32-01/` and `sdpa-b32-02/`:
  B32/T2048 pooled 9,184.55 input tokens/s, 69.014 GiB reserved,
  7.914 GiB sampled free.

Both references have 65,536 input tokens/update, matching B64/T1024. They are
historical measurements, not same-day interleaved repeats. Verify model sources,
software and configurations; report any mismatch. Preserve fixture and timing
procedure rather than treating equal input tokens as equal physical RT batch.

1. Confirm the required container, idle H10080GB, checkpoint identity,
   installed dependency namespace, disk headroom and W&B.
2. Extend only harness selection/reporting to combined T1024, preserving
   T2048/T512 and ordinary behavior. Require this frozen protocol and add focused
   CPU coverage. Ordinary T1024/FA4/Dao combinations are outside this extension.
3. Start B32/T1024, including its own dispatch and initial/terminal eager/graph
   checks. A separate B2 numerical campaign is unnecessary: production math is
   unchanged and each capacity run retains the existing operational checks.
4. If B32 passes with adequate setup/capture headroom, run B64/T1024, then repeat
   the useful selected batch in a fresh process. B64 gives the equal-token
   comparison. If it does not fit, retain the failure and use a bounded lower
   batch. No larger-batch/OOM hunt; at most two extra attempts if needed.
5. Each run starts from the same pretrained weights, with three actual Adam
   preparation updates, eleven requested capture warmups plus one gradient-
   preparation backward, and five timed complete updates. Graphs capture
   forward/loss/backward; clipping, fused Adam and scheduler remain outside.
   Include validation/copy and health checks; exclude compilation, logging,
   fixture generation and evidence I/O from timing.
6. Require finite losses, gradients and updates; exact own initial and changed-
   weight terminal graph/eager parity; active parameter/objective counts; actual
   attention/RT dispatch; final source/dependency integrity. Nonfinite,
   structural, dispatch or own-parity failures stop dependent capacity work.
   Existing independent BF16 qualifications are not cleared by these checks.

At T1024, across RT0/15, expected historical forward dispatch is 2,046 tiles:
2,044 Triton tiles (side <=256) and two existing eager 512x512 tiles. All 2,046
historical backward tiles use recomputed Triton. The eager tiles cover about
50.05% of historical attention pair area, not that fraction of runtime. Observe
actual dispatch; do not add kernel optimization to this benchmark.

## Evidence and interpretation

Record setup/steady allocated and reserved memory separately, sampled device-
free memory, complete-update throughput, parameters and the existing analytic
matrix-FLOP estimates. Use the established CPU reference snapshots and release
graph memory before final eager verification. Pool repeats by total tokens
divided by total elapsed time. No claims about real data loading or quality.

W&B entity `taylorbollman`, project `pretrained-fbt-rt-nextlat`, group
`olmo-combined-t1024`. Local evidence `.runtime/olmo-combined-t1024/`. Freeze
runtime/protocol before the GPU queue; preserve each attempt and retain each
completed stage in `gs://fast-chunks` with the existing stage helper and a fresh
timestamp under its required `cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/` prefix.
Starting O1 weights are already retained; disposable benchmark updates do not
need a new full checkpoint. Keep sources, logs, reports, plots and receipts.

Report T1024 throughput/memory versus both saved contexts and a compute-only
500-million-input-token time estimate, explicitly excluding SFT, evaluation,
data loading, checkpoint I/O and startup. A future value screen should include
matched FBT+NextLat with and without RT, with both token and compute-cost axes.
An early negative result would apply to that adaptation/data/budget, not prove
RT has no value generally. Context length should also fit the intended tasks;
this synthetic throughput fixture cannot select it on quality grounds.

Close the benchmark milestone and pause for review before new training,
kernel profiling/optimization, FA4, or additional hardware experiments.
