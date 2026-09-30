# Saved-state gradient interpretation

The two NF batches and the NFR confirmation show opposition between auxiliary
gradients and **first-pass CE**. NFR's separately trained endpoint also shows
stronger opposition to combined CE, which the two NF batches do not consistently
show. The full measured joint gradient nevertheless points toward lower
first-pass and later-pass CE in all three cases. These are local tradeoffs,
not evidence that the whole training direction is locally destructive to CE.

All three authorized gradient cases are complete. No further gradient grid is
proposed.

## Measurement and validation

The fixed update-32 weights are differentiated in FP32, with TF32/autocast off,
zero feedback jitter, beta 1 and the accepted K4 loss semantics. Primary rows are
ordered train chunks 16384–16385; conditional rows are 16386–16387. Both contain
2,048 inputs and 2,046 CE targets. Primary latent/KL counts are 2,044/2,040;
conditional counts are 2,046/2,044. Each term uses its own count. These are
different, preselected row identities, not a claim of independent documents.

Let the four already weighted gradient contributions be

\[
g_1=\nabla(\tfrac12\mathrm{CE}_1),\quad
g_L=\nabla\big((\mathrm{CE}_2+\mathrm{CE}_3+\mathrm{CE}_4)/6\big),\quad
g_A=\nabla\big(\textstyle\sum_p A_p/4\big),\quad
g_Q=\nabla\big(\textstyle\sum_p Q_p/4\big).
\]

The helper uses the existing detached targets and differentiable routes,
then compares their sum with a fifth, independently computed joint objective
gradient. It does not fill `.grad`, construct Adam or apply clipping/updates.
Unique tied parameters are counted once. The backbone group is shared by all
objectives; the fusion group receives no first-pass CE gradient; the predictor
receives no CE gradient. Zero-vector cosines are undefined, not zero.

NF reconstruction relative L2 is **1.611e-6** on the primary batch and
**9.872e-7** on the conditional batch; NFR primary is **2.463e-6**. All are
below the declared 1e-4 bound. Finite gradients and all structural zero-gradient
checks pass. These validate this
decomposition, not BF16 equivalence or reconstruction of the actual large-batch
training update.

## Shared-backbone geometry

| Measurement | NF primary | NF conditional | NFR primary |
| --- | ---: | ---: | ---: |
| First CE gradient norm | 5.3083 | 3.2908 | 7.9536 |
| Later CE gradient norm | 3.9228 | 5.4207 | 4.4978 |
| Latent gradient norm | 0.5536 | 0.6659 | 0.6737 |
| KL gradient norm | 6.4127 | 6.2658 | 6.4547 |
| Combined CE gradient norm | 6.7225 | 6.4427 | 9.1286 |
| Combined auxiliary gradient norm | 6.7261 | 6.6384 | 6.9216 |
| cos(first CE, later CE) | +0.0390 | +0.0363 | -0.0022 |
| cos(first CE, latent) | -0.6650 | -0.2390 | -0.5629 |
| cos(first CE, KL) | -0.3690 | -0.2002 | -0.5736 |
| cos(combined CE, combined auxiliary) | -0.0583 | +0.2310 | -0.5361 |

First-pass and later-pass CE are almost orthogonal here. The repeated negative
first-CE/auxiliary cosines are meaningful local evidence of competing signals,
especially for KL because its norm is much larger than the latent norm. On the
backbone, the auxiliary norm is approximately 1.001/1.030 times the combined CE
norm. The overall CE/auxiliary cosine changes sign across the two batches;
therefore a claim that auxiliaries consistently oppose total NF CE would be wrong.
NFR primary does show stronger total-CE opposition. Its auxiliary norm is about
0.758 times its combined CE norm. NFR was measured on the same primary rows as
NF, but its weights have followed a different 32-update trajectory. The
comparison does not isolate adding RT at fixed weights or establish which
RT-dependent operation produced that difference.

The conditional probe was justified by selective first-pass opposition on
shared parameters, not by treating the small -0.058 primary total-CE cosine as
strong global conflict. Its result narrows that interpretation rather than
confirming a global-conflict story.

## The joint direction remains CE-descent locally

For an infinitesimal plain-gradient step `delta_theta = -eta * g_joint`, the
first-order change in a CE contribution is `-eta * dot(g_CE, g_joint)`. The
following measured backbone dots are positive:

| Dot with the measured joint gradient | NF primary | NF conditional | NFR primary |
| --- | ---: | ---: | ---: |
| First-pass CE | +14.4753 | +6.8252 | +30.7140 |
| Later-pass CE | +28.0820 | +44.5653 | +18.7439 |
| Combined CE | +42.5572 | +51.3905 | +49.4579 |

Auxiliary terms weaken the first-pass CE descent signal, but do not reverse it.
The joint first-CE dot is about 51.4%/63.0% of that contribution's own squared
norm for NF and about 48.6% for NFR. Later CE also remains positively aligned
with the joint gradient. A
positive global clipping scalar preserves those signs, though it changes the
step magnitude. Adam's coordinate-dependent preconditioning, finite steps,
BF16 arithmetic, feedback jitter and the true 524,288-input batch are outside
this diagnostic; the table does not predict their update direction or quality.

## Fusion and predictor change the global norm interpretation

Fusion's combined CE/auxiliary cosine is strongly positive on both batches:
**+0.8698 / +0.8219**. Its CE and auxiliary norms are **0.8562 / 1.0785** on the
primary batch and **0.8617 / 1.0160** on the conditional batch. These results do
not support a claim that the auxiliary signals fight fusion's measured CE
gradient. They also do not establish that fusion has adapted enough to help
language prediction. NFR fusion differs: its CE/auxiliary cosine is **-0.6459**,
but the auxiliary norm is only **0.2789**, versus CE norm **1.1432**. Its actual
joint/CE dot remains **+1.1009**. Large angular opposition alone therefore does
not mean the auxiliary gradient overwhelms fusion's CE descent direction.

The union-of-parameters auxiliary/CE norm ratio is approximately 1.72/1.64,
larger than the roughly 1.00/1.03 backbone ratios, because the predictor has
auxiliary gradients but no CE gradient. Its joint norm is 9.4843/8.3190 and
accounts for approximately 50.4%/38.9% of the joint squared norm. Large global
auxiliary norms therefore cannot be read as equally large opposition on the
shared backbone. Global gradient clipping couples these disjoint parameter
groups through one scale, but no actual optimizer step is tested here.

## What a next comparison could establish

These measurements make a small loss-balance comparison reasonable to propose;
they do not establish that lowering KL will repair later-pass prediction. KL
is the larger of the two auxiliary gradient contributions and repeatedly
opposes first-pass CE, so it is a more focused first intervention than changing
both auxiliary weights. But its direction also helps measured later-pass CE
and aligns with fusion CE. Reducing it trades off those signals and could
reduce useful predictor learning.

Any subsequent learning comparison should retain an unchanged-recipe control,
matched starting weights/optimizer history and data, and report first-pass CE,
later-pass CE and auxiliary behavior separately. The poor full-feedback route
already present at origin remains a distinct adaptation problem. This local
gradient evidence alone neither explains that original deficit nor warrants
changing Q/K normalization, the RT kernel, or precision settings.

## Evidence

The three immutable gradient records are under
`.runtime/olmo-feedback-diagnostic/`:

| Record | SHA256 |
| --- | --- |
| `nf32-gradient-primary-01/gradient.json` | `917e53f0531eff012c50ef2d109a2bb7f692772108f296f2cc38aabc8fcf0b60` |
| `nf32-gradient-conditional-01/gradient.json` | `d6e2b55cb6f6f286f928d69a42a14772fd3d69ab89c49d85ad4b24e05f9a3e85` |
| `nfr32-gradient-primary-01/gradient.json` | `2f484c7fb4f8942280f816326eceb9dab2c9f7edc12b9c78116e5909ab40f0c8` |

All values above are recomputed from stored Gram entries/norms or taken from
their declared reconstruction checks. No raw gradient vectors are retained.
