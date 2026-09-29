# Conditional full-NFR continuation

**Status:** both paths completed their 16 updates and exited successfully.
Update-12/20 checkpoints are verified in GCS. The separate CPU endpoint
comparison passed its 105 controls; an independent scalar/receipt audit passed
131 checks. The earlier [continuation results](continuation-results.md)
describe fusion-only training and remain a separate experiment.

Both precisions recover from a midpoint CE regression and finish with slightly
lower aggregate held-out CE than their shared starting state. Their held-out
results are close, with no BF16-specific deterioration evident at this endpoint.
This supports continued investigation of functionality under this short
conditional trajectory. It does not establish numerical equivalence, a BF16
quality advantage or production readiness. Close losses coexist with a
**16.27% difference in the cumulative backbone update**, so the optimization
paths are measurably different.

Plots: [per-term trajectories](nfr-continuation/loss-trajectories.pdf)
([PNG](nfr-continuation/loss-trajectories.png)) and
[per-pass held-out losses](nfr-continuation/dev-per-pass.pdf)
([PNG](nfr-continuation/dev-per-pass.png)). Online runs:
[FP32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mfzy5brd) and
[BF16](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2v1ygap2).

## Common origin and fixed scope

Both runs import the **same BF16-trained update-4 model, Adam moments and RNG**
from the [four-update diagnostic](nfr-updates-results.md), then continue to
update 20. The FP32 branch therefore retains the same earlier BF16 history.
The scheduler extension preserves its prefix and current LR rather than
resetting optimization. Full native NFR uses RT at layers 0/15 with alpha 1,
four FBT passes, beta 1, jitter 0.02 and unit combined CE/latent/KL coefficients.

Each path uses the same 16 data selections, indices 148–163, with 8,192 selected
CE targets per update: 131,072 new CE targets and 134,059 valid input tokens.
Physical batches are B8/T128 isolated-document batches. Training batches change
across optimizer updates but match between the two precisions. Their training
loss trajectories alone therefore cannot demonstrate held-out improvement.

Every evaluation below uses the same four held-out documents and FP32 math
execution, with 508 CE/latent and 504 KL targets. CE pass weights are
1/2, 1/6, 1/6, 1/6; auxiliary pass weights are uniform. All three recorded
evaluation rows per path pass their read-only checks. Both finalized reports
also pass all ten final integrity checks.

## Held-out losses

| Training path / completed update | CE, nats per target | KL | Latent | Combined |
| --- | ---: | ---: | ---: | ---: |
| Shared origin, 4 | 5.719054 | 2.881924 | 0.616070 | 9.217047 |
| FP32, 12 | 5.782321 | 1.316563 | 0.277147 | 7.376031 |
| BF16, 12 | 5.789579 | 1.311884 | 0.277135 | 7.378598 |
| FP32, 20 | 5.669885 | 1.033129 | 0.166343 | 6.869358 |
| BF16, 20 | 5.667641 | 1.031401 | 0.165854 | 6.864895 |

At update 12, CE worsens in both paths while KL and latent losses fall sharply.
The decreasing combined objective would conceal that CE regression if reported
alone. At update 20, aggregate CE is 0.04917 nats below the origin for FP32 and
0.05141 below it for BF16. The endpoint BF16-minus-FP32 CE difference is
−0.002245 nats per target. This tiny difference on four documents is not a
quality ranking.

| Unweighted CE by FBT pass | Shared origin, 4 | FP32-trained, 20 | BF16-trained, 20 |
| --- | ---: | ---: | ---: |
| Pass 1 | 3.637564 | 3.674884 | 3.670779 |
| Pass 2 | 7.833348 | 7.635007 | 7.633599 |
| Pass 3 | 7.776318 | 7.675022 | 7.675182 |
| Pass 4 | 7.791965 | 7.684629 | 7.684726 |

The weighted endpoint improvement is not uniform across passes: first-pass CE
remains slightly worse than at the origin, while the later passes improve.
Final-pass CE also briefly regresses to about 7.8725–7.8728 at update 12 before
recovering. Later-pass CE remains substantially above first-pass CE, so this
does not demonstrate beneficial refinement or improved language-model quality.

## Endpoint optimization geometry

The [endpoint comparison](nfr-comparison-protocol.md) uses the actual FP32
master parameters saved at updates 4 and 20. Each cumulative update is the
endpoint minus the common BF16 origin; BF16 is compared against FP32. This is
an endpoint trajectory comparison, not a same-state gradient comparison.

| Component | FP32 update norm | BF16 update norm | Absolute difference | Relative difference | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backbone | 1.047573 | 1.048535 | 0.170401 | 16.2663% | 0.986783 |
| Fusion | 0.092720 | 0.092686 | 0.001298 | 1.4003% | 0.999902 |
| Predictor | 0.548417 | 0.547385 | 0.007926 | 1.4453% | 0.999897 |
| All | 1.186072 | 1.186443 | 0.170590 | 14.3828% | 0.989660 |

The same backbone difference is only 0.03289% of its entire endpoint weight
norm (518.13). That denominator largely reflects pretrained weights and would
hide the more substantial difference between the updates. The update norms
are close in magnitude but differ in direction. No numerical tolerance was
invented to classify this as equivalent or harmless.

Adam moments also remain different at the endpoint:

| Component | First-moment absolute difference | First-moment relative difference | Second-moment absolute difference | Second-moment relative difference |
| --- | ---: | ---: | ---: | ---: |
| Backbone | 0.030214 | 17.8833% | 0.000078375 | 1.4619% |
| Fusion | 0.000635 | 3.6763% | 0.000000913 | 1.7285% |
| Predictor | 0.004208 | 2.6627% | 0.000027073 | 1.1369% |
| All | 0.030512 | 13.1522% | 0.000082925 | 1.4135% |

Backbone first/second-moment cosine similarities are 0.984085/0.999898.
When measured against each moment's *change from the common origin*, their
relative differences are 13.4653%/2.5727%; the absolute differences are unchanged.
The retained JSON includes all six geometry comparisons for every component.

## Evidence and recovery

Both producers finished with the same counters: 20 cumulative optimizer
updates, 167,697 valid inputs, 163,840 CE/latent targets and 162,416 KL targets.
The continuation contributes 16 updates; the first four belong to the shared
origin. Final data cursor is source selection 164. Source inventories, input
pins, token/LR schedules, optimizer ownership, module modes, frozen buffers
and evaluation immutability checks pass. W&B runs are synced.

The three comparison checkpoints were checked against their complete serialized
boundaries and independently pinned local bytes. The CPU analysis took 236.34 s,
passed 105 checks and preserved all nine analysis source/snapshot pairs; its
focused tests passed 20 cases in 3.62 s. A separate stdlib audit verified the
final report pins, scalar arithmetic and all four producer-verified cloud
receipts/local sidecars, without rereading the large checkpoint tensors.

All four new full checkpoints are 15,215,042,309 bytes each. The update-20
objects are under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/`:

| Path below prefix | Generation | SHA256 |
| --- | --- | --- |
| `nfr-continue-fp32-01/update-000020.pt` | `1790680435443040` | `82c3b738ef01d52ae5bc74bfc8f76f88015eb25b122fbd0376644633894db665` |
| `nfr-continue-bf16-01/update-000020.pt` | `1790680265436959` | `bbd00dc1b70e08cae516ab4bb0f491ff03884efcd6e2d8b92558bf45ddbd7dd6` |

Final report pins under `.runtime/olmo-fusion-startup/`:

- `nfr-continue-fp32-01/report.json`:
  `6285917a106a6337a6880172167a4dbbe228cfcc6f369d57e0b686593f88ea9f`.
- `nfr-continue-bf16-01/report.json`:
  `cef1136ca0e75d77cfd0d840eace2fc44ad267f9cc5ef96b0232ae7f6ccc400d`.
- `nfr-comparison-01/report.json`:
  `3dff8bdaf7e3136c5c3744eae18055ccfb3336dc0fc2a1c40f0a5f392abb2229`.
- `nfr-continuation-audit-01/report.json` (all four checkpoint receipts):
  `29d82bc2ae6cb9c8fe35200d1a9d3d6da27d68691804d0ea7b9c2d9cac6784ef`.

The experiment is a short, conditional, isolated-T128 functionality probe. It
does not resolve the earlier same-state gradient discrepancies or establish
long-run stability, packed-T1024 training behavior, quality gains or general
BF16 clearance. No additional numerical GPU variant is planned from this
endpoint agreement alone.
