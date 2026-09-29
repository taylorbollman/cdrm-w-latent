# Recurrence/precision matrix results

2026-09-29. **The ordinary arm is close to FP32 on this fixture; temporal RT
and FBT each increase the measured sensitivity, and their combination has the
largest difference.** This localizes the problematic configuration more
clearly, but does not yet identify an arithmetic defect. The subsequent NF-only
FP32-fusion test improves backbone-gradient agreement modestly while worsening
final forward agreement on both records; **it is not a joint correction**.

The eight aggregate cases compare FP32 math/eager with production BF16
Flash/Triton in four arms, using unchanged initial OLMo-1B step60000 weights
and the original two B2/T16 records. Each arm retains NextLat branches with
zero auxiliary cotangents; the objective is CE only. There are **16 physical
model-backward calls**, no optimizer updates, DDP or CUDA graphs. See
[baseline controls](baseline-and-controls.md) and [protocol](protocol.md).

| Computation | Full gradient relative L2 | Shared-backbone relative L2 | Backbone cosine | BF16/FP32 backbone norm |
| --- | ---: | ---: | ---: | ---: |
| Ordinary (`N`) | 0.9804% | 0.9804% | 0.999952 | 0.9991 |
| Temporal RT (`NR`) | 25.4583% | 25.4583% | 0.976066 | 0.8437 |
| K4 FBT (`NF`) | 60.9806% | 60.8698% | 0.797300 | 0.7186 |
| K4 FBT + RT (`NFR`) | 95.9337% | 95.8461% | 0.302370 | 0.4027 |

Each comparison uses its own arm's FP32 gradient as reference. The shared
backbone contains the same 65 parameter tensors in every arm. Full gradients
also include the zero-gradient predictor and, with FBT, active fusion. Similar
full and backbone errors show that adding fusion coordinates to the reported
vector does not account for the large FBT errors.

Forward disagreement also increases across feedback passes. For FBT without
RT, valid-token hidden-state errors range from 0.85–1.01% across the two
records at pass 0 to 3.23–12.66% at pass 3. With RT, those ranges are
1.72–1.73% and 20.55–40.41%. Intermediate errors are not always monotonic.
At each precision, FBT's first pass exactly matches its corresponding
single-pass arm (`N`↔`NF`, `NR`↔`NFR`), before feedback acts. Their incoming
gradients need not match because later losses and feedback contribute.

The NFR metrics, forward fingerprints, gradient summaries and within-pair
precision geometry reproduce PR40 exactly. All shared weights, buffers,
data/noise controls, finite-execution and integrity checks pass. Predictor
gradients are zero in every case. The matrix completed in 119.94 seconds,
including source loading and diagnostic overhead:
[W&B matrix run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/lt54objk).
This duration is not a training-throughput measurement.

These are operational passes, not numerical clearance. Both precision and
their associated kernels change within each pair. The matrix implicates
neither one kernel nor one precision boundary by itself. Cross-arm gradient
norms also reflect different pass weights and active fusion, so the errors
are not additive causal contributions. One initial T16 fixture cannot
establish training stability, changed-state behavior or packed T1024 agreement.

## Fusion-only FP32 does not fix the discrepancy

The [conditional probe](fusion-protocol.md) tested NF only: two recomputed
precision references and a production-BF16 candidate with autocast disabled
only inside the existing feedback-fusion forward. Actual inputs, master
weights and return dtype stay FP32; no tensor promotion, formula change or
global autocast change is introduced. There were **three additional aggregate
cases / six physical model backwards**.

| Measurement versus full FP32 NF | Production BF16 | BF16 with FP32 fusion |
| --- | ---: | ---: |
| Full-gradient relative L2 | 60.9806% | 56.4635% |
| Backbone-gradient relative L2 | 60.8698% | 56.1640% |
| Backbone cosine | 0.797300 | 0.840608 |
| Backbone norm ratio | 0.7186 | 0.9891 |
| Fusion-gradient relative L2 | 65.2122% | 67.2421% |
| Final-pass hidden error, record 0 | 12.6627% | 14.5235% |
| Final-pass hidden error, record 1 | 3.2282% | 3.2730% |

Backbone norm matching improves considerably, but gradient direction remains
different and forward agreement does not improve together with it. This
retained mixed result therefore does not satisfy the protocol's joint
forward/gradient correction criterion. It does not show that fusion is
irrelevant; changing this boundary affects the result, but is insufficient
as the tested remedy.

Both reference cases reproduce the matrix exactly. The candidate's first pass
matches production BF16 exactly; the override executes only six fusion calls,
restores its callable and outer autocast, and leaves all model state, inputs
and RNG unchanged. All operational checks pass. The probe completed in
38.67 seconds: [W&B fusion run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/wih59gy7).

The bounded follow-up is complete and GPU changes stop here. The temporary
fusion override has been restored and is not adopted as a runtime policy.
The prior BF16 loss-layout and full-model qualifications remain open, including
NFR and packed T1024 behavior. Normalization and model equations are unchanged.
