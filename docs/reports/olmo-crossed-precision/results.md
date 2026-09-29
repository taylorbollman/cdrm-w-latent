# Crossed-state NF precision results

2026-09-29. **Adapted fusion transfers part of the numerical improvement to the
cold backbone. The adapted backbone with cold fusion does not reproduce the
adapted pair's close agreement.** This supports testing fusion startup, while
leaving substantial residual error and possible coadaptation unresolved.

The two new hybrids completed **four aggregate cases/eight physical backwards**
in 177.39 seconds, with no updates. [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/cwxjdnwe)
is synced. The existing cold/cold and adapted/adapted results below are retained
context, not new runs. This diagnostic duration is not training throughput.

## Gradient geometry

Each comparison is production BF16 versus FP32 within one fixed state. C means
cold and A adapted; the first letter denotes backbone, the second fusion.
Gradients are raw/unclipped, and predictor gradients remain exactly zero.

| State | Backbone relative L2 | Fusion relative L2 | Backbone absolute difference | Fusion absolute difference |
| --- | ---: | ---: | ---: | ---: |
| CC, existing | 60.8698% | 65.2122% | 332.7231 | 56.6953 |
| **CA, new** | **8.7328%** | **13.2529%** | **8.8512** | **1.2061** |
| **AC, new** | **53.8216%** | **54.4150%** | **340.3719** | **87.4234** |
| AA, existing | 0.9085% | 1.4051% | 0.4606 | 0.01853 |

CA improves both relative and absolute discrepancies against CC. Its FP32
backbone/fusion norms are 101.3558/9.1007, versus CC's 546.6146/86.9398, so the
relative improvement is not explained by a larger denominator. CA gradient
cosines are 0.996531/0.993535 for backbone/fusion.

**AC's lower relative errors are not an absolute-error improvement over CC.**
Its FP32 backbone/fusion norms rise to 632.4079/160.6605, and its absolute
discrepancies rise to 340.3719/87.4234. AC cosines are 0.882610/0.895525.
The table does not provide additive causal contributions or isolate how the
historical training process reached either component state.

CE objectives are 6.5289556 FP32 / 6.5239391 BF16 for CA and 7.1670839 /
7.1599725 for AC. These few fixture losses are diagnostic values, not a model
quality comparison.

## Forward states and position support

CA's valid-token hidden-state differences span 0.853–2.692% across the eight
record/pass sites. AC reaches **9.261%** at record0/pass3; restricting that site
to the shared union of positions with nonzero incoming gradients gives
**9.323%**. The discrepancy is therefore still present on supported positions.

CA still has **16.57%** incoming-cotangent relative error at record1/pass0,
despite its much better aggregate parameter gradients. It does not make every
intermediate sensitivity small. Its supported-position hidden error reaches
2.711%; AC's worst supported individual token reaches 21.95%.

Position observations retain direct CE prediction masks, feedback source and
destination eligibility, and actual/reference incoming-gradient support
separately. In this fixture, union support covers 19 of 21 valid positions in
record0 and 6 of 8 in record1 for every pass of both hybrids. Those masks are
observations; they do not remove positions from the backward or prove that
zero-cotangent positions are harmless in another objective/state.

The old AA record0/pass1 hidden discrepancy remains **12.437%**. Its per-position
tensors were not retained; the new hybrid observations cannot locate or clear
that earlier spike. Incoming pass-output gradients include later feedback and
are not common fixed cotangents across precision cases.

## Controls and scope

All four first-pass identities exactly match the retained diagonal with the
same backbone. Both complete component states, including fusion's saved scale,
match their designated source pins; predictor, trainability, tied readout,
fixture/noise, runtime controls and source bytes remain intact. All operational
checks pass; see the [ledger](test-ledger.md).

These are K4/beta1/jitter0.02 NF checks on two isolated B2/T16 records, with
NextLat branches present but zero auxiliary cotangents and no temporal RT.
CA's partial transfer motivates a bounded fusion-only startup experiment; it
does not prove those weights can be learned safely from the cold checkpoint.
No training schedule, architecture, precision policy or error budget is adopted.
RT/NFR, auxiliary gradients, packed T1024, optimizer/restart behavior, graphs,
DDP and training quality remain outside this result.

Recommended continuation: one bounded full-FP32 fusion-only warmup from the
original backbone and fresh fusion, then matched checkpoint precision checks.
Read [next steps](next-steps.md); no warmup has started.
