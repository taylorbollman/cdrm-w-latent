# 150M-family CDRM fuzzy-recall milestone

Status: first experimental milestone complete, 2026-09-08. Both 153M-family models learn native T256 fuzzy recall to
near-perfect development accuracy. All six ten-epoch calibration runs complete
successfully. The numerical prerequisites pass with the initial-Adam and strict
FP32-coordinate qualifications below. No final-test data or four-length training
sweep has run.

This implements the first milestone of the approved
[plan](../../cdrm-150m-fuzzy-recall-plan.md); commands, precision semantics and
continuation requirements are described in the
[usage guide](../../cdrm-fuzzy-recall-usage.md).

## Architecture and comparison

| Arm | Ordinary blocks / width / heads | MLP width | Unique parameters |
| --- | --- | ---: | ---: |
| Tiled CDRM, shared side fabric at blocks 3/8 | 12 / 1024 / 16 | 4096 | 153,175,040 |
| Matched standard Transformer | 12 / 1024 / 16 | 4192 | 153,437,184 |

The standard model has 0.171% more parameters. Both use native V16 untied tables,
learned LayerNorm and Q/K normalization, GELU, ALiBi and fresh initialization.
CDRM adds two separately owned D×D adapters, while sharing block 3's weights.
Rho=1, epsilon=0.1 and lambda=0.01 remain fixed. This uses the paper's 150M
architecture family on its synthetic task; Figure 6's published fuzzy curve
uses the much smaller one-layer model and is not a parameter-matched reference.

The new `ordinary_attention_precision_policy="fp32"` covers all 12 ordinary
attention blocks in both arms. Ordinary MLPs/head and tiled fabric use BF16;
the established FP32 parameter, residual/state, normalization, CE and Adam
policies remain. The model defaults to legacy behavior for old callers.

The paired initialization copies 75 shape-identical embedding/attention/norm
tensors bitwise. The 24 resized MLP tensors retain their native variance and
separate draws. Shared weights have digest
`4f6f628cac41978f0eeec4d096dcc9a12e3765728ecd394897abb5756e77a2d5`.

## Native fuzzy data

The pinned native generator and labels are unchanged. Training uses dense
aligned next-token targets, including padding; development scores the native
repeated-key and terminal-probe value positions. Training keys have length 1–3,
held-out keys length 3, and values length 1–3. Actual tensor length is T after
the generator's own shift. Earlier value tokens are teacher forced.

An independent parser supplies training answer annotations and a causal lookup
availability mask. Native generation sometimes emits terminal probes for keys
not inserted into the observed input; those examples and targets are retained.
The T256 development corpus has 16,143 scored targets; 13 targets in 9 of 1,280
examples lack that lookup history. Available-position oracle accuracy is 100%
with 99.91947% coverage; that coverage is not a universal statistical ceiling.

The query-ignoring answer-prefix shortcut scores 44.3536%, and a modal visible
value shortcut scores 17.3883%. Uniform value chance is 12.5%. These controls
make it possible to distinguish retrieval learning from simpler predictions.
The 12,800-example calibration training corpus and all numerical fixtures were
frozen before model training; no exact cross-split or cross-role input overlap
was found. Final-test data has not been generated.

## Initialization numerical evidence

These are same-state strict-FP32 versus mixed comparisons at the actual D1024,
12-block topology. All five CDRM initialization candidate cases pass the unchanged
gradient, state and initialization-Adam criteria.

| Length | Batch | Full gradient relative L2 | Independent fabric gradient relative L2 | First Adam delta relative L2 | First Adam cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 0.5506% | 0.4164% | 7.0461% | 0.997518 |
| 128 | 2 | 0.5974% | 0.4178% | 7.3126% | 0.997326 |
| 256 | 2 | 0.5558% | 0.4293% | 6.8408% | 0.997660 |
| 300 | 2 | 0.5259% | 0.4290% | 6.8226% | 0.997673 |
| 300 | 128 | 0.5764% | 0.4236% | 6.5954% | 0.997825 |

The matched Transformer also passes its full B128/T300 check: full-gradient
relative L2 0.4661%, first-Adam delta relative L2 5.9582%, and cosine 0.998225.
Neither full-batch arm has parameter tensors exceeding the gradient L2 or
maximum-error budgets. The CDRM full-batch diagnostic peaks at 50.638 GiB
allocated across its strict-FP32 and mixed passes.

The first Adam update remains qualified: passing its prospective cosine gate
does not establish close relative-L2 agreement. Between 98.97% and 99.37% of
the four B2 cases' error energy lies in the original near-zero-gradient bucket.
At B128 the fractions are 98.9894% for CDRM and 99.556% for the Transformer. Trained-state
Adam checks use the stricter 1.5625% relative-L2 limit, with results below.

T32 additionally compared naive FP32 against tiled FP32. Its full and independent
fabric gradient differences were 2.35e-7 and 3.16e-7 relative L2. Five independent
fabric parameter tensors produced 6,515 raw strict FP32 coordinate flags; the
maximum absolute difference was 3.05e-5 and all existing scale-aware checks
passed. Those raw flags remain retained. Longer cases did not repeat the naive
reference, so their passing raw screen does not supersede the T32 qualification.

T32 causality is bitwise, and T32/T300 cotangent-scaling checks pass, including
B128/T300 for CDRM. All required
compiled helpers ran with no fallback. Source and same-checkpoint audits are
clean. Full-batch initialization and the early trained state are checked separately.
These diagnostics compare precisions at identical saved states. Separate complete
FP32 training trajectories have not been run in this milestone.

## Trained-state numerical evidence

At the authoritative update-101 checkpoints, both models pass the actual B128,
T256 native training-loss comparison. The saved optimizer moments are restored
identically for FP32 and mixed precision before comparing the next update.

| Arm | Full gradient relative L2 | Independent fabric gradient relative L2 | Adam delta relative L2 | Adam cosine |
| --- | ---: | ---: | ---: | ---: |
| CDRM | 0.9903% | 0.3378% | 0.9456% | 0.999956 |
| Transformer | 0.7047% | — | 0.7466% | 0.999973 |

Both trained Adam errors are below the prospective 1.5625% limit. No parameter
tensors fail the gradient L2 or maximum-error budgets. CDRM's six state checks
and cotangent-scaling check pass; its strict-FP32 and mixed passes each capture
25 compiled graphs with no fallback. The ordinary Transformer does not require
compiled recurrence graphs. The saved states, native data and source identities are verified.
The CE differences are 2.10e-5 for CDRM and 7.72e-5 for the Transformer.

These observations clear the agreed prerequisites for the bounded T256
calibration. Trained T300 and the later final length/replicate experiment remain
separate scope. The initialization-Adam and raw FP32-coordinate qualifications
above are not removed by these trained-state passes.

## H100 profiles

Each profile used the actual update helper, two warmup updates and four measured
updates, at T300 with discarded numerical-fixture updates. No task-performance
claim is attached to these runs. Timing includes ordinary per-update metrics
and checks, and excludes compilation/warmup, periodic state monitoring, W&B and
checkpoint overhead. Memory is measured before development evaluation.

| Arm | Batch | Median seconds/update | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| CDRM | 32 | 0.794 | 11.208 |
| Transformer | 32 | 0.128 | 10.819 |
| CDRM | 64 | 0.901 | 20.417 |
| Transformer | 64 | 0.237 | 19.657 |
| CDRM | 128 | 1.128 | 38.647 |
| Transformer | 128 | 0.461 | 37.172 |

All profiles pass; B128 is frozen as the largest common tested batch, without
accumulation, after full-B128 initialization numerical checks passed.
The T300 medians give a rough 79.4-minute training-only proxy for six ten-epoch
calibration runs. The proposed 24-run main sweep has 60,000 updates at 25 epochs
and a T300-rate proxy of 13.24 hours, doubling at 50 epochs. Other lengths and
overhead have not yet been measured; these are proxies, not measured campaign
durations or proven upper bounds.

## First epoch and recovery

Both LR5e-4 calibration prefixes complete update 101, with native dense training
and no optimization failures. At the shared first-epoch endpoint (update 100),
development recall accuracy is 12.8353% for the Transformer and 12.8848% for
CDRM; development answer CE is 2.3011 and 2.3566. This remains near uniform value
chance and does not establish retrieval learning or an architecture advantage.

Each uninterrupted prefix saves updates 99 and 100, then reaches 101. Replaying
99→101 in a new process produces bitwise-equal model, optimizer, scheduler, RNG
and non-timing metrics for both arms. The LR changes from 0.0005 at update 100
to 0.0004995076687428538 at update 101, which uses epoch 2's first shuffled batch.
All 101 observed batch and index hashes match across architectures. The original
uninterrupted update-101 states remain the calibration authorities. Recovery
relies on the declared retained compiler cache and tested container/runtime.

## Ten-epoch development calibration

Each cell trains for 1,000 updates at B128/T256 on the same retained data order.
The cosine schedule has a 50-epoch horizon, no warmup and minimum LR 1e-6;
ten epochs is the frozen calibration endpoint, not the schedule endpoint.
AdamW uses betas (0.9, 0.98), epsilon 1e-8, zero weight decay and gradient
clipping at 1. The shared initialization and optimizer/runtime settings are
preserved across learning rates. Every update passes its finite-gradient check.

| Arm | Initial LR | Native token accuracy | Exact sequences | Native answer CE | Token errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 1e-4 | 99.8761% | 98.7500% | 0.042952 | 20 |
| Transformer | 5e-4 | 99.7398% | 97.1094% | 0.016890 | 42 |
| Transformer | 1e-3 | 99.7708% | 97.5000% | 0.032153 | 37 |
| CDRM | 1e-4 | 99.8637% | 98.5938% | 0.024036 | 22 |
| CDRM | 5e-4 | 99.9009% | 99.0625% | 0.009581 | 16 |
| CDRM | 1e-3 | 99.8018% | 97.7344% | 0.028614 | 32 |

All rows use 16,143 native targets and 1,280 complete examples. The prospective
selection rule uses native token accuracy at update 1,000, with native answer CE
as tie-breaker. It therefore selects Transformer LR1e-4 and CDRM LR5e-4.
Neither best-observed intermediate epochs nor the secondary coverage/precision
diagnostics change this rule. The selected CDRM endpoint makes four fewer token
errors and four fewer sequence errors than the selected Transformer endpoint.
These are one-seed development results after LR selection, not a final-test
architecture comparison or evidence that the small gap is reproducible.

At the common LR1e-4, both models already exceed 99.60% token accuracy and
95% whole-sequence exact match at update 300 (three epochs). Thus this 150M-family
T256 task has little remaining endpoint accuracy headroom. Higher rates learn
more slowly; near-ceiling accuracy also fluctuates between epochs. We retain
the declared endpoint rather than choosing the highest point on each curve.

![Native T256 development accuracy and answer CE for all six calibration runs](calibration-development.png)

Measured T256 mean update times after the first update are 0.9314 seconds for
CDRM and 0.3754 seconds for the Transformer, averaged over their three LR
settings: CDRM takes about 2.48 times as long per update. The six calibration
histories total 65.7 minutes of recorded training-update time, excluding
development evaluation, checkpointing, W&B and the separate recovery/numerical
work. These observations give no demonstrated CDRM wall-clock advantage.
Using measured T256 rates and T300 proxies for the other lengths projects
12.65 hours of training updates for the proposed 25-epoch main sweep, or
25.30 hours for 50 epochs; timings at the other lengths remain unmeasured.

The [audited aggregate and plots](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/gmibkb9d)
are online. The aggregate verifies 1,368 referenced files with zero consistency
errors. The frozen
[LR selection](../../../.runtime/cdrm-150m-fuzzy-recall/20260908T031418Z/decisions/calibration-selection-v1.json)
links every endpoint and the aggregate by SHA-256; it selects Transformer LR1e-4
and CDRM LR5e-4 before their secondary checkpoint evaluations.

## Selected-checkpoint precision and residual errors

After freezing the LR choices, both selected update-1,000 checkpoints were
reevaluated on all 1,280 retained development examples at the original B128,
T256 and execution settings. BF16 reproduces the official native metrics
exactly. Strict FP32 evaluation of the same saved weights changes **zero
predictions on all 16,143 scored targets in either model**. Weights, RNG and
gradients remain unchanged; required CDRM compiled helpers run without fallback.

| Selected arm | Native errors | Errors with earlier lookup mapping | Errors without earlier mapping | Available-target accuracy | Available-target exact sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer, LR1e-4 | 20 | 8 | 12 | 99.9504% | 99.3750% (1272/1280) |
| CDRM, LR5e-4 | 16 | 5 | 11 | 99.9690% | 99.6094% (1275/1280) |

The available subset contains 16,130 targets. The unavailable subset contains
13 targets in nine examples; Transformer predicts one of these targets
correctly, CDRM two. These are descriptive partitions of the unchanged native
score. Lack of a prior mapping does not establish a hard statistical ceiling,
and the models still make some errors on available targets.

FP32 slightly changes confidence: native answer CE decreases from 0.0429517
to 0.0429443 for Transformer and from 0.00958067 to 0.00957316 for CDRM.
There is no observed BF16 inference-accuracy penalty at these selected states.
This does not establish identical FP32/BF16 training trajectories or replace
the gradient and optimizer checks. The packet audit independently recomputes
mask partitions, native metrics, logits' predictions and precision-flip counts.

The [Transformer evaluation](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/floch77w)
and [CDRM evaluation](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/2bxainuy)
are online. Raw logits, predictions, masks and all error coordinates are retained.

## Recommended next decision

The first milestone has established a working comparison and useful numerical
evidence. Before the proposed 24-run length/seed sweep, review whether its
25/50-epoch endpoints answer the scientific question: both models already
nearly solve T256 after three epochs at a common learning rate. A bounded T300
development screen and a prospectively defined learning-efficiency comparison
are more informative next steps than immediately spending that full budget.
Report accuracy against updates, seen tokens and measured time; CDRM's extra
computation must remain visible. Use fresh final-test data only after freezing
the revised comparison, and retain the native task separately if exploring a
harder variant. This is the planned first review point; the full length/seed
sweep remains unstarted.

## Validation and provenance

- Production precision option: 97 CPU regression checks passed, 19 CUDA cases
  skipped in that CPU invocation; four targeted CUDA wrapper-parity cases passed
  for SEQ/CDRM × FP32/BF16, with bitwise logits, canonical parameter gradients and
  CDRM state parity. The CUDA compilation audits passed.
- Native task/preparation: 58 CPU checks passed, including preservation of old
  task identities, native array/mask alignment and terminal-probe edge cases.
- Calibration runner: seven focused CPU checks passed.
- Reporting helper: 14 focused CPU checks passed, including explicit separation
  of prefix/recovery roles from authoritative ten-epoch calibration endpoints.
- Selected-checkpoint evaluator: 13 focused CPU checks passed, including native
  metric replay authority, mask partitions and precision-change accounting.

The environment is an H100 80GB HBM3 in the required Docker container,
PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA runtime 13.3, with TF32 disabled and
math SDPA selected. Source snapshots, exact configurations, native arrays,
raw numerical packets, execution logs and failures are retained under
`.runtime/cdrm-150m-fuzzy-recall/20260908T031418Z`.

Online tracking is under
[taylorbollman/cdrm-150m-fuzzy-recall](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall),
group `20260908T031418Z`. Durable artifacts use
`gs://fast-chunks/cdrm-w-latent/cdrm-150m-fuzzy-recall/20260908T031418Z/`.
The [storage verification receipt](storage.json) records completion, final
manifest SHA-256, retained file counts and remote verification scope. It is
written after freezing the scientific report, avoiding a circular manifest hash.
The prior numerical milestone remains referenced through its immutable verified
manifest and receipt; those parent objects are not freshly reuploaded or rehashed.
