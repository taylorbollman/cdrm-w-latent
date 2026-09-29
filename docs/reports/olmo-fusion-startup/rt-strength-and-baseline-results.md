# RT strength and packed ordinary-model calibration

2026-09-29. **Reducing RT strength to 0.25 did not remove the remaining precision
discrepancy.** The zero-strength scan agrees much more closely with FP32, near
the ordinary FBT + NextLat result. Separately, an ordinary-model packed check
establishes a measurable BF16 difference on that fixture, while adapted FBT
remains more sensitive. These findings extend the [component results](component-results.md)
and [saved-Adam/packed-context analysis](update-and-packed-results.md).

Both new stages passed all operational checks and made **zero optimizer
updates**. Finite values, correct participation and preserved state establish
functionality; they do not clear production BF16 or demonstrate task quality.

## Native RT strength

The [strength protocol](rt-strength-protocol.md) reuses the original OLMo-1B
step60000 backbone, seeded NextLat predictor and the same fusion-only update128
checkpoint as the component stage. The startup trained only two fusion matrices
with FP32 CE; it trained neither the backbone nor predictor and used no RT.
For these gradient measurements, all component parameters are trainable.

All cases use combined CE + latent + KL, four feedback passes (K4), beta1,
fixed jitter0.02 and isolated documents. The unchanged fixture is two B2/T128
records: 512 input tokens, 508 CE/latent positions and 504 KL triples. Each BF16
gradient is compared with its own FP32 reference at the **same RT strength and
objective**. The comparison retains full FP32/math/eager versus production
BF16/ordinary Flash/native Triton, FP32 masters and deterministic controls.

| Combined-loss condition | Backbone difference | Fusion difference | Predictor difference | Backbone cosine |
|---|---:|---:|---:|---:|
| NF, no RT — retained component result | 0.735% | 0.888% | 0.763% | 0.999977 |
| Native scan retained, alpha0 | **0.704%** | 0.894% | 0.728% | 0.999978 |
| RT layers0/15, alpha0.25 | **9.912%** | 14.882% | 0.985% | 0.995207 |
| RT layers0/15, alpha1 — retained component result | **12.059%** | 20.389% | 0.820% | 0.992726 |

Differences are `||g_BF16 - g_FP32||₂ / ||g_FP32||₂` within each parameter
group. Alpha controls the RT stored representation,
`(1-alpha) * layer_input + alpha * completed_layer_output`; it is distinct from
FBT's beta. The selected RT layers execute on all four passes, including the
first. Alpha0 retains the scan implementation while reaching the ordinary
model's mathematical limit, so arithmetic order can still differ from NF.

Alpha0's agreement near NF is evidence against a large discrepancy caused
merely by executing the scan path. It does not establish that full recurrence
or its backward is cleared. At alpha0.25, the backbone absolute gradient
difference is **18.691** against FP32 norm **188.577**; at alpha1 these are
**21.449 /177.868**. Their functions and gradients differ, so the relative
percentages cannot be subtracted to assign causal shares to RT or its strength.

Reducing alpha also did not improve every internal observation. Across the
eight record/pass aggregates, the largest all-valid hidden-state differences
are **1.402%** at alpha0, **4.684%** at alpha0.25 and **3.748%** at alpha1.
The corresponding largest total incoming hidden-gradient differences are
**7.700%**, **38.798%** and **48.664%**. Aggregate parameter agreement is not a
substitute for agreement at each internal boundary. The [component report](component-results.md)
also retains the larger alpha1 CE-only discrepancy and explains why its lower
combined-loss percentage cannot be credited as resolving CE behavior.

The new stage ran four aggregate/eight physical backwards in **99.71 seconds**,
including observation and hashing. No alpha1 GPU case was repeated; its retained
reference is the completed component stage. Older gradient arrays were not
retained, so this stage does not claim a direct cross-alpha vector comparison.

## Packed T1024 ordinary calibration

The [ordinary baseline protocol](packed-baseline-protocol.md) uses the exact
packed fixture from the earlier NF check: two B1/T1024 records, 2,048 valid
inputs and 2,046 CE targets. Six within-chunk targets cross document boundaries;
attention and CE follow the continuous-stream policy while auxiliary masks
respect true documents. No position predicts beyond its chunk.

| Packed CE condition | Backbone difference | Fusion difference | Backbone cosine | Backbone absolute difference / FP32 norm |
|---|---:|---:|---:|---:|
| Ordinary N, one pass — new result | **1.799%** | Inactive | 0.999838 | 0.104 /5.779 |
| Cold NF, K4 — retained result | 19.970% | 25.406% | 0.980134 | 25.979 /130.091 |
| Startup128 NF, K4 — retained result | **5.209%** | 8.175% | 0.998793 | 0.822 /15.778 |

Ordinary N uses the unchanged pretrained backbone and needs no fusion checkpoint
because fusion is inactive. Its complete output matches the retained cold NF
first pass byte for byte at each precision, with identical tokens/masks and
tied embeddings. The original backbone/predictor state is identical in all
three rows. Ordinary CE is **3.258801** versus **3.259456** nats/target.

The 1.799% result is a calibration on these packed tokens, not an acceptance
threshold. Ordinary N has a one-pass CE objective; NF uses weighted CE over four
feedback passes with different incoming gradients. Adapted NF remains above
this calibration, but subtracting their percentages does not isolate FBT's
numerical contribution. The new ordinary stage ran two aggregate/four physical
backwards in **88.75 seconds**, including observation and hashing. There was
no optimizer, distributed execution or CUDA-graph training test here.

## Verification and evidence

The new stages preserved their exact source/state/fixture contracts and cleared
gradients afterward. An independent CPU cross-report audit checked **93
conditions**, including all **774 live/snapshot source-file pairs** across these
two stages, the component stage and their three retained references. It verified
identical imported tensor state, unchanged original backbone/predictor, exact
fixture/noise identity, objective weights/counts, precision-specific forward
anchors, report links and gradient-ratio arithmetic. All checks passed.
This was a JSON/hash/arithmetic audit, not another GPU experiment.

All local paths below are under `.runtime/olmo-fusion-startup/`:

- `rt-strength-128-01/report.json`: SHA256 `888727d5dd3aacfa9d0e2557d0fbc795ab3e826b588b8e502ecd5ffa9780eada`; [W&B blc71leq](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/blc71leq).
- `packed-baseline-01/report.json`: SHA256 `ed0487f5b80b2a002aad5ab487c4e5a36574630d3b3dca02d6cbf212c246ff05`; [W&B w13p9tle](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w13p9tle).
- `component-rt-audit-01/report.json`: SHA256 `cc0895dc9eff17e84fe3d95cd4c125c278543e46729f5f9de4cfb8902777c30a`; its `audit.py` is retained alongside it.

The approved next work measures bounded actual optimizer updates and paired
continuations, with checkpoints and common-precision evaluation. These results
do not establish that RT's remaining discrepancy is harmless, justify a new
global precision budget, or demonstrate a model-quality advantage. No kernel,
normalization or permanent recurrence-strength change was adopted here.
