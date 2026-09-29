# Overnight numerical and training-readiness investigation

Work in progress, 2026-09-29. The user authorized about six hours of useful
technical work without intermediate review. This page summarizes the findings;
the individual protocols, reports and retained source snapshots give the exact
scope. No production-quality training campaign or model architecture change is
part of this work.

## What has become clearer

The large BF16 gradient differences at fresh feedback initialization are
substantially reduced by a short **FP32 fusion-only warmup**. The pretrained
OLMo backbone does not need to change to obtain this improvement. Fusion is the
learned module that combines the token input with the previous pass's shifted
top-layer state; K4 means four feedback passes through the same model weights.

We trained only its two matrices for 128 updates, totaling 1,048,576 CE targets.
This took about 15.4 minutes including setup and checkpoint retention. Original
backbone, predictor and output scale stayed unchanged. The following are
whole-backbone relative L2 differences between BF16 and FP32 gradients, with
matched weights, inputs, feedback noise and objective within each comparison:

| CE diagnostic | Fresh fusion | After warmup128 |
|---|---:|---:|
| Original short fixture | 60.87% | 0.82% |
| Fresh short development fixture | 24.33% | 1.98% |
| Fresh T128 development fixture | 144.02% | 1.63% |
| Packed T1024 development fixture | 19.97% | 5.21% |

Absolute gradient errors also fall. This is not merely a favorable change in
the denominator. Ordinary OLMo with no feedback or temporal RT gives **1.80%**
on the same packed T1024 rows, so the remaining 5.21% has a relevant baseline
and is not being called equivalent to ordinary-model precision.

At the saved warmup boundary, a counterfactual next Adam step changes the fusion
weights by vectors differing **1.68%** between precisions. Relative to the
increment beyond a zero-gradient Adam control, the difference is **4.92%**.
The latter prevents common momentum and decay from hiding the contribution of
the new gradient. This check concerns the fusion optimizer only.

See [warmup results](warmup-results.md) and
[optimizer and packed-context results](update-and-packed-results.md).

## What is still unresolved

Activating the actual NextLat latent and KL losses at T128 is compatible with
the improved feedback state: the NF combined backbone gradient difference is
**0.74%**. NF means NextLat plus feedback, without temporal RT.

Adding native temporal RT at layers 0 and 15 gives **32.44%** for CE alone and
**12.06%** for the combined objective. These are different objectives and
denominators; the smaller combined percentage is not an additive attribution or
proof that NextLat repairs CE gradients.

A controlled recurrence-strength check is informative. The selected RT layers
still execute their scan at strength zero, but write the ordinary-layer memory
source. Backbone differences for the combined objective are:

| Temporal recurrence strength | BF16 versus FP32 |
|---|---:|
| 0 | 0.70% |
| 0.25 | 9.91% |
| 1 | 12.06% |

This argues against the mere scan dispatch being the explanation on this
fixture. It supports investigating the consequences of recurrence sensitivity.
It does not establish harmlessness or clear the complete BF16 model for a long
campaign. We have not changed Q/K normalization, recurrence math, loss weights
or numerical budgets to obtain these results.

See [component results](component-results.md). The four-update matched optimizer
test of the complete NFR model is now complete. It used fresh training rows and
a fresh all-component optimizer; backbone moments did not exist in the earlier
fusion-only checkpoint.

## Independent training-readiness work

The resource ledger records parameters and analytical FLOPs for all eight
component combinations. These are accounting results, not new throughput
measurements; see [resource results](../olmo-campaign-resource-ledger/results.md).

A bounded host training loop now has actual two-GPU captured-DDP acceptance
for ordinary stopping, checkpoint upload, fresh-process continuation and a
deliberate logging failure. The first replay check caught GCS host work
consuming Python RNG after checkpoint capture. Model, Adam, data and gradients
were already exact; isolating retention RNG made the complete continuation
exact as well. A separate failure-lifetime guard lets the deliberate failure
exit promptly while preserving its text traceback and completed checkpoint.
The original failure is retained, not overwritten.

These are tiny-model operational tests. The earlier pretrained packed T1024
restart evidence remains separately scoped. Neither substitutes for BF16
gradient agreement or H200 testing.

## Saved work and active follow-ups

Authoritative warmup checkpoint:
`.runtime/olmo-fusion-startup/train-02/update-000128.pt`, SHA256
`892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc`.
It is retained with verified bytes under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/train-02/`.

Two short fusion-only trajectories completed from that same checkpoint and
Adam state: FP32 and BF16, 16 additional updates each. A fresh-process BF16
midpoint replay matched all remaining updates and the final saved boundary
exactly. Common-FP32 held-out CE at update144 differs by **0.000171 nats/target**;
the cumulative fusion update vectors differ **18.41%**. Close short-run losses
and different optimization trajectories are both part of the result. See
[continuation results](continuation-results.md).

The complete-model paired optimizer diagnostic is complete. Its first update
starts from exactly matched state and data. Backbone raw gradients differ
13.45%, clipped gradients13.42%, and actual master-weight updates17.47%; fusion
updates differ26.58%. All four updates per path are finite. Later updates include
accumulated trajectory differences and must not be described as same-state
precision checks. The fourth backbone update differs30.31%, with a larger raw
gradient discrepancy that cannot be explained only by a shrinking denominator.

Both trained endpoints improve on the fixed held-out fixture when evaluated in
FP32. FP32/BF16-trained endpoint losses are CE **5.67536/5.71905**, KL
**2.74808/2.88192** and latent **0.59827/0.61607**. The BF16 deficit is visible;
four updates and four dev documents neither establish failed training nor clear
long-run precision. See [full-model update results](nfr-updates-results.md).

The [operator recovery bundle](../olmo-campaign-lifecycle/recovery-bundle-results.md)
has verified actual cloud-restored checkpoint/index/source/data authorities and
emits an explicitly conditional launch command. Abrupt-rank termination and
exact fresh-process continuation have also completed. The active numerical
bridge uses the saved full-model endpoint on the fixed packedT1024 fixture to
compare sparse, prepared and CUDA-graph backwards, without further training.

Metrics are online in
[Weights & Biases](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat).
This page will be updated after the remaining bounded stages finish; completed
diagnostic sources stay frozen.
