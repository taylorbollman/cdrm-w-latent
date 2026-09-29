# Precision-localization results

2026-09-29. **The larger gradient discrepancy remains unresolved.** The new
checks separate some loss-layout rounding from backbone behavior. A subsequent
crossed-backend case identifies ordinary SDPA dispatch as the switch producing
the observed BF16 backend split in this fixture; replacing RT's Triton tiles
with eager tiles under Flash changes nothing. Neither BF16 endpoint follows
FP32 closely. A subsequent eight-site attention probe finds much smaller local
output/gradient differences on common inputs than the full-model discrepancy.
This supports investigating propagation sensitivity, but does not establish
that the discrepancy is harmless or exclude a Flash bug globally. All reported
passes below are operational passes, not numerical clearance.

These are fixed-state diagnostics on the initial pretrained NFR model: K4,
native RT at layers 0/15, T16, B2 for each of two virtual inputs. No optimizer,
DDP, CUDA graphs or training updates were used. Explicit deterministic controls
and TF32-off settings were applied. See [protocol](protocol.md) and
[test ledger](test-ledger.md) for exact scope and evidence pins.

## The BF16 eager bridge does not resolve the gap

Six aggregate gradient cases compared CE-only and combined CE+latent+KL under
FP32 math SDPA/eager RT, BF16 math SDPA/eager RT, and current BF16 Flash
SDPA/Triton RT. Each case accumulates two physical records: **12 model-backward
calls**. The BF16 eager bridge retains mixed RT attention, recomputation and
the other existing BF16 runtime settings.

| Objective | BF16 math/eager versus FP32 | BF16 Flash/Triton versus FP32 | BF16 Flash/Triton versus BF16 math/eager |
| --- | ---: | ---: | ---: |
| CE only | 100.03% | 95.93% | 79.85% |
| Combined | 81.50% | 85.96% | 75.94% |

Entries are full raw-gradient relative L2 errors, with the second named path
as the reference. For the combined objective, the BF16 math/eager gradient has
cosine 0.584 with FP32; the Flash/Triton gradient has cosine 0.511. The two BF16
paths have cosine 0.657 with each other. Thus neither precision alone nor the
combined backend switch can yet be named as the sole cause. These errors are
descriptive; no new BF16 tolerance has been introduced.

Forward states also diverge across feedback passes. For the first physical
record, current BF16 hidden-state error versus FP32 grows from **1.72% at pass
0 to 40.41% at pass 3**; the BF16 eager bridge grows from 1.59% to 22.48%.
The second record is not monotonic, so these examples do not establish a
general growth law. They do show that the observed disagreement is not limited
to the backward calculation. Recorded gradients entering each pass output
include contributions through later feedback passes, not just that pass's
local loss.

The combined scalar objectives remain much closer: 14.4347 for FP32, 14.4926
for BF16 math/eager and 14.5475 for BF16 Flash/Triton. Their proximity does not
establish gradient-direction agreement. The original FP32/current-BF16 endpoint
metrics and gradient comparisons reproduce the earlier diagnostic exactly.

The bridge completed its operational checks in 51.88 seconds:
[W&B bridge run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p6oooxmt).

## A smaller difference starts inside the auxiliary losses

The loss-level probe reused identical detached hidden/embedding values from
the current BF16 forward. Those exported tensors are actually **FP32**, as
produced by the mixed-precision model; they were not rounded to BF16 merely
because the forward used autocast. The readout and seeded predictor were
reconstructed and hash-checked. Sparse and prepared layouts then used the same
fixed values, masks, pass weights and global denominators.

Eight aggregate cases cover latent/KL × FP32/BF16 × sparse/prepared: **16
loss-only backward calls**, with no backbone forward or backward in this probe.
A cotangent here is the gradient that the auxiliary loss sends into the hidden
representation.

| Auxiliary loss | BF16 prepared versus sparse hidden-cotangent error | FP32 prepared versus sparse hidden-cotangent error |
| --- | ---: | ---: |
| Latent | 0.1416% | `6.33e-7` relative L2 |
| KL | 0.2016% | `7.83e-7` relative L2 |

The same-precision FP32 layout differences are very small. BF16 already creates
a measurable local loss-layout difference before backbone propagation. For
comparison, BF16 sparse versus FP32 sparse hidden-cotangent errors on these
fixed values are 0.725% latent and 1.654% KL.

This localizes part of the sparse/prepared discrepancy to auxiliary arithmetic,
but does not explain the much larger full-model errors. A hidden-cotangent
error and a parameter-gradient error are different measurements; their ratio
is not an identified amplification factor. CE alone still has a large backbone
discrepancy despite the previously exact BF16 CE-layout comparison.

The auxiliary probe completed its operational checks in 34.72 seconds:
[W&B auxiliary run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mdo63etu).

## Crossed backend: the ordinary attention switch reproduces the split

The [adaptive protocol](adaptive-protocol.md) added one CE-only condition:
**ordinary Flash attention with eager native RT**, keeping the mixed precision
policy unchanged. Three aggregate cases, or **six physical backward calls**,
included that new condition and two recomputed BF16 references. Both references
reproduce the prior bridge's metrics, forward fingerprints and gradient-group
summaries exactly.

| CE-only comparison | Full raw-gradient relative L2 |
| --- | ---: |
| Flash/eager RT versus Flash/Triton RT | **0: exact** |
| Flash/eager RT versus math/eager RT | 79.85% |
| Flash/Triton RT versus math/eager RT | 79.85% |

All 71 parameter-gradient tensors match exactly between the two Flash cases,
as do scalar metrics, all eight pass/record hidden-state fingerprints and the
recorded incoming cotangents. With RT held eager, switching ordinary attention
from math to Flash reproduces the entire measured BF16 backend split. Thus a
Triton-versus-eager RT implementation difference is not needed to produce that
split in this particular CE/T16 case.

This remains a full-model comparison with recurrent/feedback propagation. It
does not distinguish a local attention error from sensitivity to ordinary
attention rounding, and it does not establish which BF16 result is preferable
or clear their shared FP32 discrepancy. No combined-objective or longer-context
conclusion follows from this crossed case.

The cross completed operationally in 39.70 seconds:
[W&B crossed-backend run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mrgqe7di).

## Fixed-input attention: local differences are much smaller

The [local protocol](attention-local-protocol.md) captured actual ordinary
attention at layers 1/14, passes 0/3, in both physical records: **eight sites**.
One recomputed CE anchor required two physical model backwards and reproduced
the prior production metrics, forward fingerprints and gradient-group summaries.
The observer saw 112 original forward SDPA calls and excluded 112 checkpoint
recomputation calls.

At each site, FP32 math, BF16 math and BF16 Flash received identical Q/K/V
values and the same incoming CE gradient: **24 local vector-Jacobian products
(VJPs)**. The actual captured Q/K/V and incoming gradients were BF16. The FP32
local reference promotes those same values; it is not a fresh full-FP32 model
trajectory. All eight local Flash output hashes exactly matched their captured
production outputs.

| Quantity | BF16 Flash versus local FP32 math | BF16 math versus local FP32 math |
| --- | ---: | ---: |
| Attention output | 0.1599–0.1862% | 0.1526–0.1637% |
| Query gradient, dQ | 0.2465–1.5006% | 0.1637–0.1808% |
| Key gradient, dK | 0.2287–0.6841% | 0.1589–0.1666% |
| Value gradient, dV | 0.1736–0.2222% | 0.1605–0.1705% |

These are ranges of relative L2 across the eight sites, measured over every
entry in each call tensor. Restricting outputs to actual valid queries gives
Flash errors of 0.1287–0.1865%. Local inputs preserve native strides, but use
independent leaves with offset zero; original Q/K/V storage aliasing is not
preserved.

The measurements do not show a local discrepancy approaching the full-model
79.85% backend gradient split at these sampled sites. They are consistent with
smaller local differences growing through the model, but do not establish
where or how that happens. Eight T16 sites cannot clear every attention call,
the RT/FBT composition, a different incoming gradient, or longer-context
behavior. A local input-gradient error and a full parameter-gradient error are
different measurements, so their ratio is not an amplification estimate.

The local probe completed operationally in 32.14 seconds:
[W&B local-attention run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/s3jwgeks).

## Remaining decision

The next decision is where to test propagation and a narrowly scoped precision
intervention; no correction has been established. The original 3.40224%
isolated and 1.6953% packed
BF16 layout qualifications remain open, as does the shared FP32 discrepancy.
No architecture, Q/K-normalization policy or training configuration has been
changed on the basis of these diagnostics.
