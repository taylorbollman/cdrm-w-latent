# Combined-model KL continuation

**2026-09-30: both branches have completed training and evaluation through 64;
the reduced branch's final checkpoint publication and paired audit are pending.**
Reducing KL weight from 1 to 0.1 improves development CE on every pass, with a
1.126734-nat advantage on pass 4 at update 64. Both raw auxiliary losses are
higher. This is a loss-balance tradeoff, not improvement in every objective.

NFR means **NextLat + FBT + native RT**. The model uses four FBT passes (K4),
with RT active at layers 0 and 15 of the 16-layer OLMo backbone. In this arm,
the first pass already includes RT; it is not the ordinary OLMo control.
See the [protocol](protocol.md) and [progress record](progress.md).

## Controlled comparison

Both branches restore the **same original NFR update-32 model and populated
Adam state**, along with the same data cursor, scheduler and rank RNG. Neither
starts from the completed FBT-only update-128 checkpoint. Only the external KL
coefficient changes: control 1, reduced 0.1. Latent weight remains 1. This is
continuation of an adapted state, not two fresh optimizer restarts.

Both use full fusion strength, jitter 0.02, T1024, physical batch 12 per GPU
on two H100s, and 524,288 real input tokens per update. Each adds 32 updates
(16,777,216 inputs), stopping at 64 inside the original 128-update plan. The
100-update LR warmup and all data ordering remain unchanged. Inputs are counted
once despite four passes. Training is BF16 mixed with FP32 master parameters
and Adam, CUDA graphs and activation checkpointing; the fixed 64-row development
panel uses FP32 with jitter disabled.

The repeated development measurement at update 32 agrees exactly on raw CE,
latent and KL. Both ranks' first resumed training forward at update 33 also
agrees exactly on all raw loss means. The gradient norm at that first update is
8.834785 for control and 3.237772 for reduced KL. This confirms a common forward
starting point; a smaller norm alone is not a predictive result.

## Development CE

Values are nats per target on the same 64-row panel; lower is better.

| Update | KL weight | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | shared origin | 3.043012 | 7.026269 | 7.055292 | 7.063815 |
| 48 | 1 | 2.983241 | 6.764267 | 6.839343 | 6.858271 |
| 48 | 0.1 | 2.790368 | 6.348725 | 6.456982 | 6.493014 |
| 64 | 1 | 2.930108 | 6.473414 | 6.604760 | 6.646855 |
| 64 | 0.1 | 2.774970 | 5.375003 | 5.481358 | 5.520121 |

At update 48, reduced KL improves first/fourth-pass CE by 0.192873/0.365257
nats relative to control. It improves absolute CE rather than merely shrinking
the pass gap by degrading the first pass. At 64, first/fourth-pass advantages
are **0.155138/1.126734 nats**. The fourth-minus-first gap is 3.716748 for
control and 2.745152 for reduced KL. Neither branch's feedback passes beat its
own first pass. Useful refinement remains unestablished.

The midpoint raw latent losses are
0.261257/0.043548/0.044086/0.043945 for control versus
0.341310/0.057770/0.056030/0.055411 for reduced KL. Raw KL is
2.508881/0.620044/0.638946/0.637515 versus
3.724618/0.783862/0.822513/0.820431. Thus both auxiliary losses are higher in
every pass under the reduced coefficient. Weighted objective totals are not
used to judge the two recipes. The same tradeoff persists at the terminal
evaluation:

| Raw auxiliary, update 64 | KL weight | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latent | 1 | 0.242130 | 0.043243 | 0.042835 | 0.042737 |
| Latent | 0.1 | 0.274914 | 0.088235 | 0.084884 | 0.084474 |
| KL | 1 | 2.299452 | 0.666133 | 0.683167 | 0.683961 |
| KL | 0.1 | 3.320543 | 1.191557 | 1.210429 | 1.210006 |

Both continuations completed 32 finite updates, with clipping active on all 32.
Control preclip norm min/median/max/final is
3.526242/7.714435/23.630489/6.139404; reduced is
2.095179/3.778194/7.929704/4.687100. The reduced coefficient lowers typical
gradient scale and improves measured CE, but does not eliminate clipping.

## Interpretation and remaining scope

This comparison asks whether the earlier NF loss-balance finding also appears
with native RT active. It does not measure RT's causal benefit, establish
general NextLat settings, or resolve the earlier mixed-precision equivalence
qualification. Finite updates and preservation checks are functional evidence;
they do not establish quality or an optimal LR.

All saved configurations measured before this continuation settle by K32,
including poor predictors. **The new NFR64 endpoints have not yet had their
deep pass curves measured.** The recommended next diagnostic is a small,
matched no-update K1–32 probe of both saved endpoints, before considering a
selected continuation through the existing update-100 warmup boundary. See
the [next-step assessment](../olmo-fbt-stability/next-steps.md). No further
training is queued after this pair.

Control [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ujz924fj)
and reduced [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1xu07xdf).
Checkpoint locations and verified publications are in [storage-receipt.md](storage-receipt.md);
timing definitions and parameter counts are in the
[resource ledger](../olmo-fbt-stability/resource-ledger.md).
