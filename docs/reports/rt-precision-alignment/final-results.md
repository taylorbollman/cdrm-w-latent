# Standard RT precision alignment: 500-update development decision

**Keep the protected A policy as the operational default; retain released-style
B as experimental.** Both predeclared initialization pairs completed 500
updates. B reduced update time by about 11% and reserved about 9.8 GiB less
GPU memory, but neither seed cleared the frozen 0.005-nat development margin
with its one-sided 95% upper bound. B was not nominated for adoption, so the
conditional confirmation evaluation was **not run**. The
[recorded disposition](../../../.runtime/rt-precision-alignment/20260910T191100Z/reference/development-disposition-500.json)
closes this bounded milestone without changing the model default.

The question is whether the released-style mixed precision provides useful
speed and memory savings without a consequential numerical or learning cost.
The completed checks show correct graph replay and modest trained-state
gradient/update errors at the tested anchors. Both seeds show a small positive
development CE gap for B, with different magnitudes and uncertainty. Passing
a gradient screen does not establish equivalent learning, and this evidence
does not indicate catastrophic numerical failure or final-convergence behavior.

The completed two-seed figure build is available in
[W&B](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/lbo30ro4), with
standalone [training curves](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/outcome-two-seed-500/training-curves.svg),
[development gaps](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/outcome-two-seed-500/paired-evaluation-gaps.svg)
and [trained numerical comparisons](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/outcome-two-seed-500/trained-numerics.svg).
Its [coverage report](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/outcome-two-seed-500/results.md)
includes both complete training histories and all endpoint development pairs,
and explicitly lists untested second-seed numerical anchors and confirmation.

**Frozen comparison.** The model has 12 tiled recurrent blocks, D1024,
16 heads, FFN4096, rho 1, causal ALiBi and learned Q/K normalization: 151,045,120
backbone parameters and 216,843,264 total. This is the selected width-1024
150M-backbone configuration, not the paper's width-1408 D.1 model. CDRM is
disabled. Each training update uses physical B512/T512, head-only microbatch
2, internal checkpoint recomputation with four backward MLP chunks, and no
outer activation checkpoint. Forward/backward use CUDA graphs; clipping and
AdamW run outside the graph. [Experiment plan](../../rt-precision-alignment-plan.md)
and [frozen protocol](../../../.runtime/rt-precision-alignment/20260910T191100Z/reference/protocol.json).

| Arm | Arithmetic | Role |
| --- | --- | --- |
| A | BF16 autocast with `bf16_fp32_state` | Protected control |
| B | BF16 autocast with `legacy` recurrent precision | Released-style candidate |
| C | Tiled FP32, autocast and TF32 disabled | Conditional numerical reference |

All arms retain FP32 parameters, residual storage, parameter gradients and
Adam moments. B also retains upstream FP32 running sums, max-logit state and
K/V gradient buffers. A adds FP32 attention working and reconstruction/adjoint
intermediates; B is not uniformly BF16. Ordinary-attention protections from
the earlier CDRM work are inactive here. Ownership, masking, autocast and
capture fixes were preserved; this comparison made no new model-arithmetic
change. [Dtype and change audit](change-audit.md).

Training uses a pinned, bounded C4/T5 corpus with identical token order in
A/B and initialization seeds 20260910 and 20260911. This is not a reproduction
of the authors' unavailable processed corpus. Native training loss is shifted
CE summed over 511 targets per row and divided by B×512; evaluation reports
CE per supervised token. The released schedule retains a 5,000-update warmup,
12,500-update horizon, peak LR 0.001 and alpha0/alpha_f 0.1. LR is 0.000118 at
100 and 0.00019 at 500. All 500 updates remain within the original 5,000-update
warmup. Neither endpoint tests peak LR or final convergence.

The prospective learning margin is 0.005 nats per supervised token, assessed
under common FP32 evaluation with paired token-weighted document bootstrap
intervals: 10,000 resamples, seed 20260912, one-sided 95% upper bound and
separate outcomes for both initialization seeds. Native evaluation is
secondary. Document resampling does not estimate training-seed uncertainty.
The margin and numerical screens are engineering criteria, not published
author tolerances.

**Numerical and execution findings.** Candidate captured/uncaptured checks
passed at tiny/full B2, including changed-input Adam updates and causality;
physical B512 initial loss and all 111 gradients matched exactly. The tiny
fixed-forward credit test preserved all gradients and write adjoints exactly
under cotangent scaling by 1/32 and 32. Initial B2 query/key flags remain
recorded. At initial B512, three block-9 tensors had 3.31–3.34% relative L2
error, just over the 3.125% tensor screen; all maximum-error screens passed.
The frozen-operand block-9 probe localized a 4.3098% query-adjoint discrepancy
to attention arithmetic, without establishing learning harm or one primitive
to replace. [Initial results](initial-results.md) and
[attention findings](attention-findings.md).

Every completed trained-anchor comparison below uses the same physical B512
diagnostic objective, exact starting weights and nonempty Adam state across
A/B/C. It covers all 111 parameter tensors and 216,843,264 coordinates. C fits
physically; no microbatch reference approximation is used. These numerical
probes use compiled uncaptured execution, with graph correctness established
separately.

| Trained anchor | Arithmetic vs C | Global gradient L2 | Worst tensor L2 | Worst tensor max/reference-max | Applied Adam-delta L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| A, seed0, update 100 | A | 0.5060% | 1.0697% | 1.4634% | 0.3165% |
| A, seed0, update 100 | B | 0.5068% | 1.2002% | 1.8206% | 0.3365% |
| B, seed0, update 100 | A | 0.5898% | 1.0966% | 1.7219% | 0.3087% |
| B, seed0, update 100 | B | 0.6016% | 1.2023% | 1.9060% | 0.3267% |
| A, seed0, update 500 | A | 0.8016% | 1.1128% | 2.1682% | 0.4793% |
| A, seed0, update 500 | B | 0.8034% | 1.1505% | 1.8120% | 0.5090% |
| B, seed0, update 500 | A | 0.6707% | 1.5629% | 1.5473% | 0.4655% |
| B, seed0, update 500 | B | 0.6735% | 1.5401% | 1.6338% | 0.4933% |

There are no global, tensor or trained-Adam review flags at these anchors.
The screens are 1.5625% global L2, 3.125% tensor L2, 6.25% maximum/reference
maximum and 1.5625% trained Adam-delta L2, with no absolute acceptance floor.
The update 100 probes use saved LR 0.000118; the update 500 probe uses saved LR
0.00019, rather than the next scheduled LR. At the A500 anchor clipping is
inactive in all three arms; at B500 it is active in all three, with coefficients
approximately 0.9275–0.9278. Every state is finite FP32, and Adam advances
exactly once. These are conditional update comparisons, not comparisons of
gradients from separately trained weight sets. The second-seed anchors remain
unassessed. [100-update evidence](pilot-100-results.md),
[A500 numerical report](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/trained-A-seed0-500-b512/report.json)
and [independent A500 packet audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/trained-A-seed0-500-b512-audit.json),
[B500 numerical report](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/trained-B-seed0-500-b512/report.json)
and [independent B500 packet audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/trained-B-seed0-500-b512-audit.json).

**Development learning.** Evaluations use physical B512 on the same 1,024
development sequences, 523,264 supervised targets and 1,043 documents.

| Seed / update | Evaluation | A CE | B CE | B−A | Paired 95% interval | One-sided 95% upper |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| seed0 / 100 | Common FP32 | 6.21191467 | 6.22131791 | +0.00940324 | [0.00786219, 0.01095135] | 0.01073532 |
| seed0 / 100 | Native | 6.21198226 | 6.22140499 | +0.00942273 | [0.00787450, 0.01096924] | 0.01075699 |
| seed0 / 500 | Common FP32 | 5.01842698 | 5.02735240 | +0.00892543 | [0.00722523, 0.01059407] | 0.01033497 |
| seed0 / 500 | Native | 5.01847412 | 5.02740736 | +0.00893324 | [0.00722450, 0.01060590] | 0.01034659 |
| seed1 / 500 | Common FP32 | 5.01217213 | 5.01634153 | +0.00416939 | [0.00186638, 0.00649190] | 0.00610802 |
| seed1 / 500 | Native | 5.01226576 | 5.01638325 | +0.00411749 | [0.00180205, 0.00645520] | 0.00606474 |

For seed0, B's common-FP32 deficit exceeds the 0.005 margin at both checkpoints,
and both paired intervals lie wholly above it. For seed1, the 500-update point
estimate is below the margin, but its one-sided upper bound exceeds it: this
seed does not establish degradation beyond 0.005, and it does not establish
acceptability within 0.005 either. Both seeds have positive gaps; they are
reported separately without a pooled interval or an estimate of seed-level
uncertainty. Native evaluation gives nearly the same results, so
evaluation-time rounding alone does not explain the measured gaps.
[Paired FP32 analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-100-dev-fp32/report.json),
[paired native analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-100-dev-native/report.json),
[500-update paired FP32 analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-500-dev-fp32/report.json)
and [500-update paired native analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed0-500-dev-native/report.json),
[seed1 FP32 analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed1-500-dev-fp32/report.json)
and [seed1 native analysis](../../../.runtime/rt-precision-alignment/20260910T191100Z/analysis/pair-seed1-500-dev-native/report.json).

**Recovery provenance.** The first A continuation was interrupted after
recorded update 169, with the last saved weights/Adam at 100. That directory
was preserved. The accepted continuation, `A-seed0-500-restart1`, resumes the
verified original step 100 checkpoint and records updates 101–500. Its overlap
101–169 exactly reproduces the interrupted run's saved losses, gradient norms,
learning rates and data windows. This scalar check does not claim equality
of unsaved intermediate model tensors. Independent fresh-process resume
checks compare all saved gradient, model, optimizer and RNG tensor bytes
exactly at the bounded proof step. [Recovery trajectory audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/restart-A-seed0-trajectory.json)
and [post-interruption resume gate](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/post-interruption-resume-gate.json).

The completed A continuation retains the original initialization, data,
schedule, source and runtime identity. All 400 resumed updates report finite
FP32 parameters, gradients and moments; combined with the completed 100 segment,
this covers updates 1–500. Final training CE is 5.06555916 per supervised token.
The step 500 checkpoint SHA256 is
`1e530f6635226e9d318a1cc0cdd36412c4dc9fd34c8aa8c85508d7aed146f472`.
[Closed A500 training report](../../../.runtime/rt-precision-alignment/20260910T191100Z/train/A-seed0-500-restart1/report.json)
and [W&B training history](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/aqhq91l4).

B likewise completed updates 101–500 from its original step 100 checkpoint,
with finite FP32 parameters, gradients and moments throughout. Its final
training CE is 5.07062735 per supervised token, and the step 500 checkpoint
SHA256 is `0448b7758a2cae31f91183bd0cd4285e4a5be54397bbefb80950e93081de979b`.
[Closed B500 training report](../../../.runtime/rt-precision-alignment/20260910T191100Z/train/B-seed0-500/report.json)
and [W&B training history](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/z1x8ysfe).

An independent first-seed CPU lineage audit verified the initial 216,843,264 parameter
values are bitwise identical across A/B, both initial optimizer states are
empty, all six retained checkpoint hashes match, and the four completed
segments preserve the paired data, source, model, runtime and schedule.
All 1,000 update records cover the expected data windows with finite FP32
checks and no compiler activity after capture. The interrupted segment is
excluded from merged curves, avoiding duplicated updates.
[Paired training lineage audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/paired-seed0-500-training-audit.json).

The second-seed A/B runs each completed updates 1–500 directly, without a
training restart, using the matching initialization digest
`b26b0d0a844c1c6d29083185a8289566b791a297c7e63e89e085e34192d4c922`.
Both report finite FP32 parameters, gradients and moments throughout; final
training CE is 5.04745645 for A and 5.04579954 for B. These losses on changing training batches do not replace held-out comparisons. The outcome report verifies
completed segment coverage and common source, runtime, data and schedule
across both seeds. [A seed1 training](../../../.runtime/rt-precision-alignment/20260910T191100Z/train/A-seed1-500/report.json)
and [B seed1 training](../../../.runtime/rt-precision-alignment/20260910T191100Z/train/B-seed1-500/report.json).
The independent [second-seed lineage audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/paired-seed1-500-training-audit.json)
also verifies matching initial tensors, initialization distinct from seed0,
all six step 0/100/500 checkpoint hashes and state metadata, all 1,000 update
records, finite FP32 state and unchanged compiler counters after capture.

**Practical cost.** Timings exclude capture setup but include data transfer,
graph replay, clipping, Adam and complete finite/FP32 verification. The table
omits the first two updates following each listed capture.

| Matched phase | A seconds/update | B seconds/update | B time reduction | A / B peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| seed0, resumed updates 103–500 | 6.1305 | 5.4501 | 11.1% | 49.00 / 39.17 |
| seed1, updates 3–500 | 6.1284 | 5.4463 | 11.1% | 49.00 / 39.17 |

These are sequential same-device comparisons within each paired phase,
not randomized or isolated kernel benchmarks. Host artifact retention can
contribute background I/O. The physical GPU UUID changed across the
interruption; both original 100 runs shared one H100, and both first-seed
continuations shared another H100 of the same model, capacity, capability
and driver. The two second-seed runs also share one device. Memory reservation
and sampled physical free memory are reported separately from allocation
peaks because graph replay can reuse workspace. [100-update cost and scope](pilot-100-results.md)
and the linked completed training/lineage reports retain the full measurements.

**Disposition and untested scope.** The frozen
[endpoint scope](../../../.runtime/rt-precision-alignment/20260910T191100Z/reference/endpoint-check-scope.json)
uses the two first-seed trained anchors for representative physical-B512
numerical coverage; second-seed numerical anchors are untested and are not
required absent a new consequential concern. Confirmation was conditional
on development nomination. Because neither seed clears the unchanged
one-sided-upper-bound margin, B was not nominated and **no confirmation
evaluation was run**. The unused confirmation set remains available for a
future selected comparison; there is no confirmation-quality claim here.

The useful speed/memory gain does not satisfy the current quality criterion.
Keep A as the default and B available for explicit experiments. Passing the
trained numerical screens alone does not justify a new kernel repair, nor
does it prove precision differences cannot accumulate during learning. The
full development gap is not attributed to one isolated attention primitive.
A longer comparison through the actual warmup, or an explicit decision to
accept a speed/quality tradeoff, would be a separate milestone. Current
evidence covers this bounded C4/T5 slice, these two seeds and the first
500 of 5,000 warmup updates; peak LR and final convergence remain untested.

The [final source check](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/final-source-check.json)
confirms the 59 tested source files remain unchanged. Validation comprises
152 passing integrated CPU tests and 16 separate outcome-contract checks;
the [final figure check](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/final-figures-check.json)
verifies report coverage and figure hashes. The figure builder itself makes
no automatic precision-policy decision.

Execution lineage: `.runtime/rt-precision-alignment/20260910T191100Z`.
Online records are in the [W&B project](https://wandb.ai/taylorbollman/rt-precision-alignment).
Reusable data, checkpoints, numerical packets and the actual compiler cache
have verified retention at
`gs://fast-chunks/cdrm-w-latent/rt-precision-alignment/20260910T191100Z/`.
Per-case storage receipts identify completed upload and integrity checks;
the final document/figure archive is recorded separately after the reviewed
document bytes are frozen.
