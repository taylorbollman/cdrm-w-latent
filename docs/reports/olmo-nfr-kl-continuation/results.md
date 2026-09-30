# Combined-model KL continuation

**Intermediate report, 2026-09-30: the reduced-KL branch is still running.**
The control has completed update 64. The matched update-48 measurements favor
reducing KL weight from 1 to 0.1 for predictive CE on every pass, while both raw
auxiliary losses are higher. Terminal results and the paired audit remain
pending. This is a loss-balance tradeoff, not improvement in every objective.

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
| 64 | 0.1 | pending | pending | pending | pending |

At update 48, reduced KL improves first/fourth-pass CE by 0.192873/0.365257
nats relative to control. It improves absolute CE rather than merely shrinking
the pass gap by degrading the first pass. However, neither branch's feedback
passes beat its own first pass. Useful refinement remains unestablished.

The midpoint raw latent losses are
0.261257/0.043548/0.044086/0.043945 for control versus
0.341310/0.057770/0.056030/0.055411 for reduced KL. Raw KL is
2.508881/0.620044/0.638946/0.637515 versus
3.724618/0.783862/0.822513/0.820431. Thus both auxiliary losses are higher in
every pass under the reduced coefficient. Weighted objective totals are not
used to judge the two recipes.

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
