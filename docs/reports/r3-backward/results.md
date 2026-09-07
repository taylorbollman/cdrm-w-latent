# R3 FP32 backward validation

**Decision: FP32/no-accumulation path cleared for the tested regime.** The
reproduced discrepancies are consistent with ordinary FP32 evaluation-order
error and scale-sensitive coordinate thresholds. No backward defect was found;
no model, tiled-kernel, optimizer, or Stage B training-source patch was needed.
The clearance rests on the combined evidence below. Original failed criteria
remain recorded, including one logit coordinate, stricter gradient-tail checks,
and first-step Adam update screens.

This is a bounded NUM investigation requested after Stage B. It contains no new
research trajectory and does not establish an R3 learning advantage. The original
Stage B artifacts retain their status; this investigation gives no numerical
reason to invalidate or rerun the FP32 pilot.

## Tested scope and identity

The task-loss checks use MQAR with the exact Stage B aligned answer mask and
mean over scored answers, vocabulary 1024, 12 blocks, D256, H4/full MHA,
MLP1024/GELU, pre-norm ALiBi, affine Q/K normalization, recurrent block index 3,
rho=1, and four backward MLP chunks. They use T128 and physical batches 2 and 64,
at the retained R3 initialization and final update-2000 checkpoint. B64 is the
pilot's actual physical/global batch; B2 is a diagnostic subset, without
accumulation. A D32/T32/B2 mean-CE fixture and an isolated D32/T16/B2 write probe
provide additional diagnostics.

All model commands ran serially inside the project Docker container on one
H100 80 GB. Twelve commands took **241.22 seconds**, including container startup,
within the 45-minute cap. PyTorch is `2.13.0a0+8145d63` (NGC 26.06), CUDA 13.3,
kernel driver 580.173.02 with container compatibility driver 610.43.02.
FP32 parameters/computation, autocast/TF32/dropout off, deterministic algorithms
on, math SDPA, compiled tiled helpers, and no whole-model compilation/CUDA graphs
match the selected Stage B training profile. A separately labeled automatic-SDPA
run reproduces the earliest raw failure. Every tiled check observed compiled
graphs and passed the strict compiler/fallback audit. FP64 fallbacks execute only
naïve autograd, using the same FP32-representable weights promoted to FP64;
existing fixed ALiBi/mask construction is retained.

Outer revision: `a605d3c167034e2a1da8f3acc4677f3d07b9c06f`.
Fork revision: `0a82392fa394084e05f24cef09fe53dda85de71a`.
The working source snapshot and per-file hashes are retained because diagnostics
and Stage B support files include uncommitted work. The fork remains unchanged.
The four checkpoint-based comparisons verify all 14 frozen training-source
hashes, the plan hash, and the checkpoint's next-data digest before execution.
Full checkpoint hashes, token/label hashes, mapping, settings and compiler
counters are in each retained report. Checkpoints are:

```text
.runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/
  SYN-mqar-R3-seed0-init.pt
  SYN-mqar-R3-seed0-final.pt
```

The exact commands and execution times are in [execution.json](execution.json).
They use `bash scripts/docker_shell.sh bash -lc '<command>'`, first asserting
`/.dockerenv`, `/workspace/cdrm-w-latent`, and successful `nvidia-smi -L`.
The new tools are `scripts/r3_backward_validate.py`, `r3_write_path_probe.py`,
`r3_ce_fp64_reference.py`, `r3_validation_metrics.py`, and the CPU-only
`r3_first_adam_analysis.py`. The metric module's 22 CPU tests pass.

## Criteria kept fixed

The original coordinate rule is `abs(error) <= 2e-6 + 2e-5*abs(reference)`.
Separate diagnostics require relative L2 and maximum absolute error divided by
reference RMS each to be at most `2e-5`. These reuse the original relative-error
budget at tensor scale; they are engineering screens rather than a numerical
error theorem. Norm reductions and subtraction use FP64 with a documented
`1e-12` denominator floor. The predefined near-zero band is
`abs(reference) <= 0.001*reference_RMS`.

Adam diagnostics report unclipped gradients, actual clip norm/coefficient,
moments, actual deltas computed by FP64 subtraction of saved FP32 weights, and
post-step weights. The separately labeled practical update screen is relative
delta L2 at most `1e-3` and coordinate discrepancy at most `0.01*learning_rate`.
No threshold was increased after observing a failure. Flags are explained using
retained tensors and FP64 references below, rather than relabeled as passes.

## Reproduced raw-cotangent failure

The D32/T128/B2/V256 seed-937 fixture reproduces the old token and cotangent
hashes, configuration, source identities, and all 101 saved gradient summaries.
The old 61 failing tensors are 60 parameter tensors plus the embedding-output
gradient. Capturing the actual block-3 input adds a 62nd failure. The previous
"unit-direction" fixture divided the entire random `[B,T,V]` cotangent by its
L2 norm. The random values had a BF16 round-trip before testing, but these runs
performed FP32 arithmetic. That normalization did not represent masked CE.

| Raw fixture/profile | Scale | Legacy failing tensors / 102 | Worst relative L2 | Worst max error / RMS |
|---|---:|---:|---:|---:|
| Historical automatic SDPA | 1/32 | 0 | 8.80e-7 | 4.49e-6 |
| Historical automatic SDPA | 1 | 62 | 8.80e-7 | 4.49e-6 |
| Historical automatic SDPA | 32 | 81 | 8.80e-7 | 4.49e-6 |
| Strict math SDPA | 1 | 59 | 1.02e-6 | 4.87e-6 |

For both profiles and both backends, `G(alpha*v)/alpha == G(v)` exactly at
alpha=1/32 and 32 on the retained forward graphs, before clipping or Adam.
At a scale matching a valid MQAR CE cotangent norm, backend comparisons pass the
original rule. Dividing this non-power-of-two result back to raw scale produces
small rounding discrepancies in both naïve and tiled backward; all normwise
checks pass. This is a scale check, with actual CE tested independently below.

Only 90 of 906 failing automatic-profile coordinates meet the predefined
near-zero band. Most failures are small relative to tensor RMS, but they must
not all be described as near-zero entries. Naïve FP32 itself has 72 failing
gradient tensors against naïve FP64; tiled FP32 has 68. Both pass every normwise
check against FP64, with worst relative L2 1.20e-6 and 1.47e-6. The higher-precision
reference places some offending coordinates between the two FP32 values.
The combined evidence supports rounding and threshold scale sensitivity for
this fixture, rather than an incorrect backward scaling law.

See [the raw analysis](raw-analysis.md) for every scale, error magnitude, worst
tensors, coordinate examples and FP64 values, and [the existing-evidence audit](existing-evidence.md)
for the other retained raw failures and historical-source limitations. Existing
T256/T512 and D256/T16 evidence was audited rather than unnecessarily rerun.

## Actual masked CE and persistent writes

Every task case compares all **100 intended parameter gradients**, the embedding
output gradient, and the **block-3 input gradient**. All are present, finite and
shape-matched. Parameter names/order and initial values match exhaustively.
The custom tiled backward accumulates parameter gradients internally, so the
harness calls `loss.backward()` and reads every parameter's `.grad`.

| Fixture | Max gradient absolute error | Worst relative L2 | Legacy gradient failures / 102 | Max-error/RMS failures / 102 |
|---|---:|---:|---:|---:|
| Tiny initialization, D32/T32/B2 | 3.58e-7 | 1.17e-6 | 0 | 0 |
| Stage B initialization, B2 | 1.62e-7 | 1.39e-6 | 0 | 0 |
| Stage B trained, B2 | 2.24e-7 | 2.99e-6 | 0 | 8 |
| Stage B initialization, B64 | 1.68e-8 | 1.23e-6 | 0 | 0 |
| Stage B trained, B64 | 2.79e-8 | 2.07e-6 | 0 | 4 |

All five CE values pass the original rule. The B64 CE values are identical
between backends at recorded FP32 precision: 7.38333416 initially and 2.84727669
at the trained checkpoint. Four cases pass every original logit coordinate
check. Trained B64 has one failed logit coordinate out of 8,388,608:
`[26,112,518]`, naïve 0.0103462823 versus tiled 0.0103437286, error 2.55369e-6
against allowance 2.20693e-6. This remains a failure; the largest overall logit
error, 2.28882e-5, occurs elsewhere and passes its coordinate allowance.

FP64 fallback checks were triggered by the residual gradient-tail, logit and
Adam flags. All FP32 task-gradient coordinates pass the original rule against
FP64 in the tiny, initial B64, trained B2 and trained B64 cases. At trained B64,
worst relative L2 against FP64 is 6.32e-6 for naïve and 6.78e-6 for tiled.
Both arms also show stricter max-error/RMS tails against FP64, and both have
more original logit-coordinate failures against FP64 than against each other.
This supports shared FP32 numerical error; it does not turn the tail/logit
flags into passes. [The CE analysis](ce-analysis.md) records all flagged tensors
and same-coordinate FP64 values, including the single failed logit.

The isolated block-3 probe supplies cotangent only at the final output position.
All 15 earlier permanent K/V writes receive nonzero gradients in both backends
(L2 norms 0.166–0.681); earlier input-gradient L2 is about 1.5964. Every block
parameter, input, and earlier-write value/gradient passes both numerical rules.
No later ordinary block can supply an alternative history path in this probe.
The unused terminal write has expected naïve `None` versus tiled explicit zero;
that exception applies only to that intermediate, never to intended parameters.

## One actual AdamW update

Each arm begins with identical model tensors, moments and step counters.
The settings are Stage B AdamW: betas (0.9,0.95), epsilon 1e-8, zero weight decay,
clip norm 1, and `foreach=False,fused=False` for AdamW. Clipping retains the
trainer's normal dispatch. Initial fixtures use the first warmup LR 1e-5;
the trained diagnostic retains the checkpoint's final LR 1e-4 for exactly one
NUM step. It does not advance a research checkpoint or invent a schedule after
update 2000.

| Fixture | Maximum actual delta discrepancy | Worst tensor relative delta L2 | Practical update-screen failures / 100 |
|---|---:|---:|---:|
| Tiny initialization | 7.53e-7 | 1.36e-3 | 1 |
| Stage B initialization, B2 | 2.68e-7 | 1.09e-4 | 10 |
| Stage B trained, B2 | 5.96e-8 | 1.59e-4 | 0 |
| Stage B initialization, B64 | 2.68e-7 | 9.22e-5 | 14 |
| Stage B trained, B64 | 1.19e-7 | 2.86e-4 | 0 |

Clipping norms/coefficients agree exactly except initial B64's one-FP32-step
rounding difference. All moment tensors pass the original rule. Stricter
moment max-error/RMS flags remain (2 tiny, 27 initial B2, 5 trained B2,
26 initial B64, 0 trained B64). Squaring a gradient changes the error/RMS
relationship; [the first-update analysis](first-update-analysis.md) checks the
recorded moments and ideal FP64 Adam formulas directly.

The initial update flags are real Adam sensitivity, not primarily subtraction
resolution. At the tiny worst coordinate, clipped gradients are 6.565e-9 and
4.730e-9, both below epsilon=1e-8. Computing the first Adam step
`-lr*g/(abs(g)+eps)` in FP64 reproduces the 7.52e-7 discrepancy; rounding each
ideal result reproduces the stored worst-coordinate updates. The naïve FP64
reference favors tiled at this coordinate. At initial B64's worst paired
coordinate it favors naïve, and across the 14 flagged tensors' worst coordinates
it favors naïve in 8 and tiled in 6. Both FP32 arms have larger maximum ideal
update errors against FP64 than against each other. This is consistent with
near-epsilon sensitivity shared by FP32 arithmetic, with no tiled-specific
missing gradient or optimizer-state mismatch.

For initial B64, 17 of 9,974,016 parameter coordinates exceed the 1%-LR
screen; global relative delta disagreement is 3.28e-5. This quantifies sparsity
and overall magnitude without erasing the flagged coordinates. All post-step
weights pass the original rule; three initial-B64 tensors retain stricter
max-error/RMS weight flags arising from these same update discrepancies.

At the actual D256 shape, every tensor's relative update discrepancy stays
below the declared 0.1% screen. The first-update coordinate maximum is 2.68%
of the warmup LR; those 1%-LR screen failures remain visible. Both retained
trained-checkpoint comparisons pass the practical update screens. Small FP32
arithmetic differences can change trajectories over many updates; this milestone
does not claim bitwise training equivalence or repair the separate update-zero
resume restriction.

## Retention and next boundary

Fixtures, gradient/optimizer packets, FP64 references, exact commands, source
versions and detailed reports are retained under
`.runtime/r3-backward/20260906T210249Z/` on persistent home storage and archived at
`gs://fast-chunks/cdrm-w-latent/r3-backward/20260906T210249Z/`.
[The storage manifest](storage.json) records the completed checksum verification.
Existing Stage B checkpoints remain in their original local and GCS lineage.

This clears the tested FP32 R3/rho1/no-accumulation MQAR regime for further bounded
work. BF16, accumulation, training at T512, distributed execution and CDRM remain
outside this clearance. Their old failures are not evidence that this Stage B
FP32 path is invalid. No further research training was launched in this milestone.
