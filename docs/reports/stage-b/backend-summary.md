# Stage B backend gate and cost probes

Recorded 2026-09-06 on one NVIDIA H100 80GB, inside the project container.
These are NUM correctness fixtures and OPS random-ID cost probes, not synthetic
task learning results. The executable is
[`scripts/stage_b_backend_check.py`](../../../scripts/stage_b_backend_check.py).

## Gate decision

**Use FP32 for the initial Stage B pilot. BF16 remains blocked by the declared
numerical bounds.** Tiled R3 uses compiled helpers, four backward MLP chunks,
and rho=1. Ordinary attention uses explicit math SDPA; deterministic algorithms
are enabled and TF32 is disabled. Compiled paths did not fall back to eager.

All fixtures use 12 blocks, four full-MHA heads, GELU, MLP width four times model
width, pre-norm ALiBi, Q/K normalization, Mitchell initialization, an untied head,
and no embedding norm, biases, or dropout. The training cost probes use a
1,024-symbol vocabulary to cover the largest proposed task head. Numerical
fixtures use 256 symbols unless the table states otherwise.

### FP32 evidence

The initial unnormalized random-cotangent runs fail some elementwise gradient
bounds. Their per-tensor relative errors are small, but absolute cancellation
errors exceed `atol=2e-6` for large summed output cotangents. The failed raw runs
are retained as `backend-num-d32-t128.json`, `backend-num-d32-t256.json`,
`backend-num-d32-t512.json`, and `backend-num-d256-t16.json`.

A separately identified diagnostic normalizes the same sampled output direction
to unit L2 norm, making the directional-derivative scale independent of output
size. It keeps **the original elementwise `atol=2e-6`, `rtol=2e-5`** for logits
and gradients. Each fixture compares all 100 named parameter gradients,
including embeddings and final head, plus the gradient at the embedding output.
No missing gradient is skipped. These are initialization fixtures, not a claim
covering arbitrary trained weights or every sequence length.

| Width | Length | Batch | Vocabulary | Unit-direction FP32 gate | Worst per-tensor relative L2 gradient error |
|---:|---:|---:|---:|---|---:|
| 32 | 128 | 2 | 256 | [Pass](backend-num-d32-t128-unit-fp32.json) | 9.04e-7 |
| 32 | 256 | 2 | 256 | [Pass](backend-num-d32-t256-unit-fp32.json) | 8.33e-7 |
| 32 | 512 | 2 | 256 | [Pass](backend-num-d32-t512-unit-fp32.json) | 7.64e-7 |
| 256 | 128 | 2 | 1024 | [Pass](backend-num-d256-t128-unit-fp32.json) | 1.00e-6 |

### BF16 failures retained

The BF16 checks compare naive BF16 and tiled BF16 both with a common naive FP32
reference and with each other, using the same BF16-representable random output
cotangent. The unchanged per-tensor gradient limits are relative L2 <= 0.015625
and maximum absolute error / reference RMS <= 0.0625. These checks include input
gradients and every parameter. Logits also receive elementwise checks at
`atol=0.002`, `rtol=0.02`; applying those logit limits against FP32 is additional
to Stage A's between-BF16 logit check. The gradient failures alone block BF16.

At width 32 and length 128, worst relative L2 errors are approximately 0.0325
(naive BF16 vs FP32), 0.0285 (tiled BF16 vs FP32), and 0.0197 (between BF16
backends). Failures also occur at lengths 256/512 and in the width-256,
length-16 fixture. Neither changing the relative-error limits nor ignoring
failed parameter tensors is used to permit BF16 training. BF16 cost-probe
completion below establishes only operational execution.

The earliest width-32/length-128 raw report used automatic SDPA dispatch. All
subsequent numerical and operational reports explicitly select math SDPA.

## Update cost and memory

Training probes use width 256, four heads, MLP width 1,024, 12 blocks, length 128,
batch 64, and vocabulary 1,024. Each topology receives the same initial SEQ
weights and identical independently seeded random inputs/labels. Labels are
aligned to scored logits, with one answer every 16 positions: 512 supervised
answers per update and 8,192 input tokens. There is no extra next-token shift.

Each probe performs two warm-up updates, including initialization of AdamW
state, followed by three measured updates. Time includes forward, answer-only
CE, backward, gradient clipping at 1, and AdamW. Input preparation/transfers,
initial compilation, evaluation, checkpoint I/O, and scientific-task generation
are excluded. AdamW uses LR 1e-3, betas (0.9, 0.95), epsilon 1e-8, no weight
decay, and no fused/foreach update. BF16 probes retain FP32 model parameters
and optimizer moments. Final parameters and gradients are checked for finiteness.

| Precision | Topology | Mean update seconds | Mean forward seconds | Peak allocated GB (decimal) | Record |
|---|---|---:|---:|---:|---|
| FP32 | SEQ | 0.02912 | 0.00899 | 2.398 | [JSON](backend-bench-d256-t128-b64-seq-fp32.json) |
| FP32 | tiled R3 | 0.23269 | 0.07986 | 2.230 | [JSON](backend-bench-d256-t128-b64-r3-fp32.json) |
| BF16, numerically blocked | SEQ | 0.03122 | 0.01166 | 1.720 | [JSON](backend-bench-d256-t128-b64-seq-bf16.json) |
| BF16, numerically blocked | tiled R3 | 0.26710 | 0.09098 | 1.606 | [JSON](backend-bench-d256-t128-b64-r3-bf16.json) |

At these measured update rates, three paired tasks with 2,000 updates for each
SEQ/R3 arm project to about **0.436 H100-hours (26.2 minutes) of update work**.
This is an extrapolation, not a completed run or an end-to-end job budget. The
runner must add measured generation, evaluation, compilation, and checkpoint
costs. The three-update timing sample is deliberately small.

## Long-sequence evaluation probe

At width 256, length 512, batch 64, vocabulary 1,024, and FP32, two warm-up
forwards plus three measured inference-mode forwards complete with finite
logits. Mean forward time is [0.04041 seconds for SEQ](backend-eval-d256-t512-b64-seq-fp32.json)
and [0.27013 seconds for tiled R3](backend-eval-d256-t512-b64-r3-fp32.json).
Peak allocated memory is 0.960 GB for both. This measures prepared-data forward
execution; it does not certify full-width length-512 parameter gradients or
task generalization.

All R3 cost probes record compiled graphs (12 for training, 10 for evaluation)
and no unsupported/fallback counters. Each JSON includes resolved configuration,
hardware/software provenance, source hashes, and detailed measurements. The GPU
was released to the main runner after these probes completed.
