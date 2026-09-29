# Packed T1024 restart repeatability investigation

2026-09-29. The first full-model packed restart failed bitwise continuation.
A fixed-input diagnostic then isolated repeatability failure in ordinary
Flash-SDPA backward at T1024 when deterministic execution was disabled. Enabling
deterministic controls removed the isolated failure, and the corrected
full-model cloud-restored write/resume pair **passed exact continuation on both
ranks**. Recovery is qualified under this pinned execution contract; the
separate BF16 numerical comparisons remain unresolved.

## Original full-model failure

Evidence: `.runtime/olmo-packed-campaign/pretrained-write-01/report.json` and
`pretrained-resume-01/report.json`. The source configuration was NFR, K4, native
RT at layers 0/15, NextLat, BF16 mixed, T1024, B12 per rank on two H100s, using
22 microbatches per rank for a 524,288-valid-token logical update.

Checkpoint model/Adam/RNG/cursor restoration and post-capture state checks
passed. The next update's input and keyed-noise fingerprints also matched.
Forward CE/latent/KL sums and the combined objective were bitwise identical.
The only unequal scalar metric was `gradient_norm_before_clip`:
196.2971954345703 on uninterrupted continuation versus 196.29531860351562 on
fresh-process continuation. All four predictor gradient hashes matched; all 67
backbone/fusion gradient hashes differed, followed by model/optimizer state.
Both ranks showed the same result. The exactness gate correctly failed.

This pattern localized the investigation to backward execution. Token embedding
accumulation alone would not explain differing upper-layer/fusion gradients.
The packed runner had TF32 disabled but did not explicitly enable deterministic
backward algorithms. Those are separate controls.

## Fixed-input Flash diagnostic

[Diagnostic script](../../../scripts/olmo_flash_backward_repeatability.py), with
12 focused CPU tests, ran one fresh process per configuration. Each used B12,
16 heads, head dimension 128, BF16 Q/K/V and a fixed BF16 output cotangent,
causal ordinary PyTorch Flash-SDPA with dropout zero. Storage was `[B,T,H,D]`
transposed into `[B,H,T,D]`, matching ordinary attention's head-view pattern.
There was no model, optimizer, DDP, native RT, NextLat, input refill or checkpoint.

Each configuration recorded three eager backwards and three CUDA-graph replays.
All six outputs and full Q/K/V gradients were hashed; numerical comparisons used
bounded FP64 diagnostic reductions against the first eager result. The
non-deterministic condition treats differences as descriptive; the deterministic
condition requires bitwise equality. Thus a non-deterministic run's operational
success does not mean its gradients were identical.

| Length | Deterministic controls | All six forward hashes equal | All six gradient hashes equal | Maximum aggregate Q/K/V relative L2 versus first eager |
|---|---|---|---|---:|
| 16 | Off | Yes | Yes | 0 |
| 16 | On | Yes | Yes | 0 |
| 1024 | Off | Yes | **No** | 3.9178617503e-6 |
| 1024 | On | Yes | Yes | 0 |

For T1024 with determinism off, only **dQ** differed; dK and dV remained exact.
Between 427 and 451 dQ elements differed in each non-reference observation out
of 25,165,824 query elements. Maximum dQ-relative L2 was 7.1668583249e-6 and
maximum absolute error 0.001953125. Both eager repetition and repeated replay
of the same captured graph varied. This rules out a fresh-process restore or
changed graph input being necessary to produce this isolated failure.

With determinism on, all gradients and outputs were bitwise identical across
both eager and captured execution at both lengths. Off/on input hashes were
identical for each length. T1024's first deterministic gradient need not equal
its non-deterministic counterpart; the acceptance requirement concerns
repeatability under one frozen execution contract.

Run evidence and W&B:

- `flash-t16-d0-01/report.json`: [T16/off](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4rmeljzu).
- `flash-t16-d1-01/report.json`: [T16/on](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zul4zvyk).
- `flash-t1024-d0-01/report.json`: [T1024/off](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/u9l20orh).
- `flash-t1024-d1-01/report.json`: [T1024/on](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nsnyngyf).

All paths above are relative to `.runtime/olmo-packed-campaign/`. All four
runs used PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, H100 80GB and driver
580.178.04. All six listed source hashes matched across all four runs, and every
source-snapshot file was independently SHA-checked against its recorded pin.
The diagnostic script pin is
`b7c856a2db2f5766b9c28d124b5fa6b5bec29f2c5f743cf0bab3a96b8c7e2622`.
All 24 observations passed finite/input-integrity checks; source and local
input-generator state remained unchanged. An independent report audit verified
the three-eager/three-replay sequence and recomputed every exactness label from
the stored hashes.

## Corrected runner contract

Commit `e5a593b728bf2c14deef71acf958c7c92e7c3cd1` updates the packed runner to invoke
the existing `configure_determinism(True)` helper **before CUDA device selection
or initialization**. It enables deterministic algorithms, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, makes cuDNN deterministic and disables cuDNN
benchmarking. The helper rejects an already-initialized CUDA context.

The selected controls are recorded in the report, W&B configuration and
checkpoint configuration, and the helper source is included in the recovery
source fingerprint. Focused tests check initialization ordering, rejection of
late configuration and helper-source pinning. Model architecture, loss policy,
source weights, logical token budget and optimizer recipe are unchanged.

The corrected pair started and continued under this same contract. The old
failed pair remains evidence and is not relabeled as passed. New evidence:

- `pretrained-write-02/report.json`: all 13 gates passed; checkpoint saved after
  the first logical update and the next update executed on the original live
  graphs. [W&B write run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/k5ynlpin).
- `pretrained-resume-02/report.json`: all **10 gates passed** in 760.70455 seconds,
  including the fresh-process continuation from the cloud-restored checkpoint.
  Both ranks matched input/noise fingerprints, raw gradients, scalar metrics,
  full model/Adam/scheduler/counter/cursor/RNG boundary state and actual RNG draws
  bitwise. [W&B resume run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p4xtd9as).

The finalized resume-report SHA256 is
`18666d8cfed2ef79bb5a20f0f569c81c25dcf8b08b6a20842a5983f42e821603`.
The reader verified the restored checkpoint/index, and actual Adam state was
resident before DDP preparation and CUDA-graph capture. This establishes
full-model recovery for this same-world-size, same-runtime T1024/B12 case; it
does not extend the guarantee to changed hardware, world size or execution
settings.

The isolated Flash experiment and successful full-model pair support the
missing deterministic controls as a correctable cause of the observed recovery
failure. The fix enables several backend controls together, so these results
do not establish that one particular Flash kernel was the sole full-model
source of nondeterminism.

Neither repeatability result clears the separate 3.40224% isolated or 1.6953%
packed BF16 sparse/prepared gradient discrepancy, nor the larger
BF16-versus-FP32 differences described in
[precision-assessment.md](precision-assessment.md). Repeatability, agreement
between numerical implementations and training quality remain distinct claims.
