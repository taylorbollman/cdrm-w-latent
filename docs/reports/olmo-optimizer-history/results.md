# Fixed-state optimizer-history result

At the saved BF16 NFR update 20 state, FP32 and BF16 produced close gradients
and candidate Adam updates on this small fixed fixture. Retaining our 20 steps
of Adam history reduced the absolute FP32/BF16 update difference relative to
resetting Adam. The result does not recover original OLMo pretraining moments,
erase the earlier startup discrepancy or establish general BF16 clearance.

The diagnostic completed in **163.85 seconds**, with no live training update or
new checkpoint. [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yezbu1wv)
is synced. The [protocol](protocol.md) fixes the selected BF16 update 20 model,
the same two B2/T128 held-out records, K4, beta 1, native RT at layers 0 and 15,
alpha 1, keyed jitter 0.02 and all three loss terms. There are 512 input tokens, 508 CE/latent
targets and 504 KL triples. This is the earlier diagnostic fixture, not the new
no-jitter packed evaluation protocol.

FP32 math/eager and BF16 mixed Flash/Triton each performed one aggregate
backward, comprising two physical backwards. The model, optimizer, schedule,
RNG, counters, fixture and parameter objects remained unchanged. All 148 current
source/snapshot pairs, 100 historical authority controls, runtime/import checks
and 11 final integrity checks passed independent inspection. Full report:
`.runtime/olmo-optimizer-history/probe-01/report.json`, SHA256
`035a417908b4c673c3b63983e8303909ec7b21866e4e3a26c45250319d5f284e`.

## Same-state gradients and losses

Relative L2 uses the FP32 path as reference. Both paths use FP32 master weights;
the comparison changes the declared execution path as well as autocast precision.

| Component | Raw gradient difference | After global clipping | Raw cosine |
|---|---:|---:|---:|
| All | 1.3687% | 1.3497% | 0.999909 |
| Backbone | 1.2894% | 1.2909% | 0.999917 |
| Fusion | 1.0638% | 1.1399% | 0.999947 |
| Predictor | 1.6678% | 1.5783% | 0.999891 |

The backbone FP32 gradient norm is 9.27478 and the absolute difference is 0.119592.
Global norms 10.31791/10.34042 are independently clipped to 1, using the original
actual clipping operation. Per-target means are:

| Term | FP32 | BF16 | BF16 minus FP32 |
|---|---:|---:|---:|
| CE | 5.667641 | 5.668042 | +0.000401 |
| KL | 1.031401 | 1.031141 | -0.000260 |
| Latent | 0.165854 | 0.165867 | +0.000013 |

The FP32 means exactly reproduce the saved BF16 endpoint's prior common-FP32
evaluation. Opposing component differences partly cancel in the combined loss.

There is also an earlier **same-fixture** comparison, before full-model
adaptation, at original backbone/predictor plus fusion weights from update 128. The
[component probe](../olmo-fusion-startup/component-results.md) used exactly the
same tokens, masks, noise, physical records, mode, loss weights/denominators,
parameter participation, precision paths and runtime. Its 130 source/snapshot
pins remain unchanged and are included in the current inventory. Report SHA256:
`59e850755c25d3e585b1787b3b7a70ec67a43a8a7edfc97ceacd0d6f75a0653d`.

| Raw gradient difference | Before full-model adaptation | BF16 update 20 state |
|---|---:|---:|
| All | 11.4280% | 1.3687% |
| Backbone | 12.0590% | 1.2894% |
| Fusion | 20.3890% | 1.0638% |
| Predictor | 0.8202% | 1.6678% |

The backbone absolute difference also falls from 21.4491 to 0.119592, while the FP32
reference norm falls from 177.8677 to 9.27478. Predictor relative error increases, though
its absolute difference falls from 0.540359 to 0.074916. This supports lower overall
precision sensitivity at this particular adapted model state on unchanged data;
it does not show uniform improvement in every component. Because all model
weights evolved, it cannot attribute that change to Adam history, fusion alone,
or any single training mechanism. Adam does not enter a fixed-state backward.

The earlier [initial full-NFR update](../olmo-fusion-startup/nfr-updates-results.md)
had a 13.447% raw backbone difference and 17.469% actual backbone update
difference. Today's 1.289% raw result is a different model state **and a different
batch**: 508 held-out CE targets versus 8,192 training targets in that first
update. It is encouraging evidence at the adapted state, not a controlled claim
that adaptation alone reduced that discrepancy tenfold. Optimizer history cannot
itself alter these fixed-state raw gradients; its effect is tested separately.

## Actual Adam updates and shared-history controls

Each candidate uses the same clipped gradient and saved current LR
`2.0575741958618166e-5`, betas 0.9/0.95, epsilon 1e-5 and per-group weight decay.
Inherited candidates retain step 20 and both moments; reset candidates start
step 0 with empty moments. Resetting Adam does not restart LR warmup. All results
use actual stored FP32 parameter changes from unfused CUDA AdamW; this does not
establish equivalence with the newer fused campaign optimizer.

Six conceptual full-model candidates are factored into 426 temporary Adam calls
over 71 canonical parameter tensors: inherited/reset crossed with FP32/BF16/zero
gradients. No temporary candidate replaces or steps the live model.

| Component | Inherited actual-update difference | Reset actual-update difference | Inherited after subtracting zero-gradient Adam | Reset after subtracting zero-gradient Adam |
|---|---:|---:|---:|---:|
| All | 0.9692% | 1.9417% | 1.6397% | 1.9417% |
| Backbone | 1.0378% | 1.9207% | 1.6405% | 1.9207% |
| Fusion | 0.3588% | 1.1403% | 1.1030% | 1.1403% |
| Predictor | 0.4789% | 2.2298% | 1.6464% | 2.2298% |

For all parameters, inherited FP32 update norm is 0.091669, with absolute
FP32/BF16 difference **0.00088849** and cosine 0.999953. Reset update norm is 0.333659,
with difference **0.00647877** and cosine 0.999812. Thus reset has 7.29 times the
absolute difference, as well as roughly twice the relative difference.

The inherited zero-gradient update still has norm 0.074289 because stored
momentum continues to act. Subtracting that common control leaves reference
norm 0.054186 and the same absolute precision difference, hence 1.6397% rather
than 0.9692%. This exposes how shared history partly enlarges the denominator;
the smaller inherited absolute difference is not a denominator-only effect.
Subtracting ordinary weight decay alone barely changes either comparison.
The zero-gradient subtraction is a diagnostic counterfactual, not an additive
causal decomposition of Adam.

Resetting history also changes the intended update substantially even within
FP32: reset/inherited norm ratio 3.64 and cosine 0.60669. These numbers therefore
support retaining optimizer continuity when resuming this trajectory; they do
not show that reset Adam fails, that inherited Adam is always preferable for a
new objective, or that original pretraining moments would behave the same way.

## Epsilon, sign changes and limits

About 63.60% of clipped FP32 coordinates have magnitude at most epsilon. After
the FP32 candidate, bias-corrected square-root second moments are at most
epsilon for 77.99% of inherited coordinates and 63.60% of reset coordinates.
Those coordinate subsets carry 51.12% and 24.89% of their respective FP32 stored
update squared norms. The update energy includes momentum/decay; these are
observations, not a measurement of causal harm from epsilon or a reason to
change it.

Strict gradient sign flips affect 0.4753% of coordinates, but only 0.0000907% of
FP32 clipped-gradient squared norm. They account for 8.98% of the **reset**
FP32/BF16 update-difference squared norm. Small-gradient sign changes are more
visible after Adam, but do not explain most of this reset difference. These
statistics do not define an acceptance threshold or justify an epsilon sweep.

The available [official original OLMo training run](https://wandb.ai/ai2-llm/OLMo-1B/runs/jis94ivf)
records step 60000 CE 2.598164 per input token at T2048. That ordinary pretraining
loss is not directly comparable to this adapted NFR/T128 fixture. Matching data,
context, objective, denominator and weight boundary is necessary; the
[checkpoint/log notes](original-checkpoint-notes.md) retain the exact sources and
current inability to verify public original optimizer retrieval.

This is one fixed adapted checkpoint and one repeatedly inspected tiny dev
fixture. It supports continued bounded functionality work without an automatic
epsilon, Q/K normalization or architecture change. It does not undo the prior
trajectory differences, validate cold-start BF16, choose a production startup,
or establish long-run quality. The separate packed evaluation milestone tests
integration and state preservation, not those unresolved claims.
