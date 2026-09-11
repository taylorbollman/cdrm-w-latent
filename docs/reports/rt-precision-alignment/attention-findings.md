# Where the legacy attention-gradient difference comes from

The bounded probe identifies a local numerical source: with the **same rounded
Q/K/V operands and the same incoming attention gradient**, the legacy query
adjoint differs from independent FP64 attention differentiation by **4.3098%
relative L2**. Independent FP32 attention differentiation on those same
operands differs by only `7.82e-7` relative L2. Forward operand drift alone
therefore cannot explain this local query-adjoint discrepancy.

This finding explains a source and scale of error; it does not establish that
the difference harms learning or identify a single operation that should be
patched. The result supports the frozen, short A/B training comparison while
retaining the protected policy as the control.

## Fixture and integrity

The [completed probe report](../../../.runtime/rt-precision-alignment/20260910T191100Z/attention/block9-seed0/report.json)
uses zero-based block 9 of the actual twelve-block D1024/H16 RT, physical
B2/T512, seed 20260910, legacy BF16, and the retained initial C4 diagnostic
fixture. It performs native shifted CE and backward, with no optimizer update
and no CUDA capture. [W&B run](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/ckr2ib2s).

Observed and observer-free loss are both exactly `10.853645324707031`. Every
one of the 111 parameter-gradient tensors, covering 216,843,264 coordinates,
matches bitwise between those executions and the earlier retained legacy
B packet. Model weights remain unchanged and observers are removed. All
recorded source snapshots match their hashes and current corresponding files;
the compiled execution reports no fallback.

The report SHA256 is
`7ea49c74b87e2ab0b097ddedd3d823e68f291e8dbf320b51130360aa5492169b`.
The 241,186,636-byte
[attention packet](../../../.runtime/rt-precision-alignment/20260910T191100Z/attention/block9-seed0/attention-packet.pt)
has SHA256
`ef9a9ce50efc96777d8c7c246f3cb55c3c0a9a48b07ede7c851cbe6a7448b24b`.
I independently verified that packet hash and recomputed the reported local
gradient errors inside an explicitly CPU-only container.

## What the reference differentiates

For one batch element and head, let \(q_t\) be the already-scaled query,
\(k_t^0,v_t^0\) the provisional self record, and \(k_s,v_s\) permanent
earlier records. The independent reference computes

\[
z_{ts}=\begin{cases}
q_t^\top k_s+b_{ts}, & s<t,\\
q_t^\top k_t^0+b_{tt}, & s=t,
\end{cases}
\qquad
a_{ts}=\frac{e^{z_{ts}}}{\sum_{j\le t}e^{z_{tj}}},
\]

\[
u_t=\sum_{s<t}a_{ts}v_s+a_{tt}v_t^0,
\qquad
\Phi=\sum_t\langle g_t,u_t\rangle.
\]

Here \(g_t\) is frozen from the actual legacy backward. It already contains
the incoming credit produced by that recurrent computation. The reference
treats the stored Q/K/V operands as independent inputs and computes the local
vector-Jacobian product, \(\nabla\Phi\), without differentiating through
\(g_t\) or regenerating a different recurrent trajectory. There is no second
query-scaling factor.

I checked the implementation's orientation: inputs are `[length,batch,head,dim]`,
the mathematical score matrix uses `[query,key]`, the native ALiBi bias is
added in that orientation, and reported probabilities are transposed to the
custom helper's `[key,query]` layout. The permanent record is excluded on the
diagonal, which uses the provisional record. Future keys are masked.

Beyond the four passing CPU oracle tests, I independently evaluated explicit
softmax/attention derivative formulas on the full saved operands in CPU FP64.
All five analytic adjoints agree with the saved autograd FP64 reference to
relative L2 between `1.44e-16` and `3.42e-15`. This checks the reference's
orientation and provisional/permanent distinction against another derivative
calculation, rather than relying only on a plausible gradient norm.

## Measured local error

The reference is FP64 attention on the fixed captured operands. Each row covers
1,048,576 adjoint coordinates. FP32 values below are fractions, not percentages.

| Adjoint | Actual legacy dtype | Legacy relative L2 error | Independent FP32 relative L2 error |
| --- | --- | ---: | ---: |
| Already-scaled query | BF16 | **4.3098%** | `7.82e-7` |
| Provisional key | BF16 | 1.7237% | `3.03e-7` |
| Provisional value | BF16 | 0.2578% | `6.68e-8` |
| Permanent key destination buffer | FP32 | 0.9387% | `4.58e-7` |
| Permanent value destination buffer | FP32 | 0.2530% | `3.29e-7` |

The query-adjoint cosine is 0.999072 and its maximum absolute error is
`1.79e-6`, against a reference maximum magnitude of `1.47e-4`. Its 4.3098%
relative error closely matches the approximately 4.3224% query-projection
parameter-gradient error that selected this block for investigation. This is
evidence of a relevant local source, not a measurement of the fraction of
whole-model parameter error caused by that source.

A supplementary CPU calculation rounds the saved FP64 query adjoint to BF16
once and measures only 0.1668% relative error. Thus the full 4.3098% difference
is substantially larger than the final BF16 output cast by itself. Intermediate
precision, reconstruction and reduction order remain the relevant combined
numerical process; this probe does not isolate one of those operations.

Reconstructed attention probabilities differ from the fixed-operand FP64
reference by 0.2155% relative L2, and reconstructed attention values by
0.2335%. The actual BF16 attention input to the forward MLP and its backward
reconstruction differ by 0.2999% relative L2, with maximum absolute difference
0.015625. Small reconstruction differences can affect subsequent derivative
calculations, but the present experiment does not separately attribute the
query-adjoint error to probabilities, attention values or adjoint contractions.

The source-based first-token cancellation hypothesis is **not observed in
this fixture**. At \(t=0\), forward and reconstructed attention match exactly;
actual and FP64 query/provisional-key adjoints are exactly zero; and the
provisional-value adjoint equals \(g_0\). The last permanent K/V record also
receives exactly zero reference and actual credit, as required by causality.
Those checks should remain counterevidence against that particular proposed
failure mechanism.

## Consequence for the current milestone

This is an explanation for a measurable local precision difference under
legacy arithmetic. It establishes neither a broken derivative formula nor
equal long-run learning. In particular, the independent FP32 reference uses
mathematical attention with a different execution order; it is not a tested
drop-in change to the tiled custom backward. A selective precision repair
would need its own complete gradient, capture, memory and performance checks.

The completed
[physical B512 A/B/C comparison](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/full-b512-seed0/report.json)
provides the relevant operational context. Full physical-batch FP32 fit, so
no microbatch-reference qualification was needed. Global gradient relative
L2 was 0.78729% for A–C and 0.78744% for B–C. Legacy retained three tensor-L2
review flags in this same block's query projection and Q/K norms, reaching
3.3389%; all tensor-maximum screens passed. B–A global gradient relative L2
was 0.1235%, with no pairwise review flags. These facts keep the flags visible
while showing that their local magnitude does not translate into a comparable
global gradient error at the intended batch size.

I recommend proceeding with the already frozen 100-update A/B warmup pilot,
using identical initialization/data and retaining both policies unchanged.
Compare their resulting weights with common FP32 evaluation as well as native
evaluation, and examine nonempty-Adam states. That directly informs whether
the observed local precision difference warrants extra FP32 work in this
application. Additional arithmetic surgery is justified if that evidence
shows a consequential deficit or if a targeted experiment is needed to
explain a new failure. The first pilot remains a short prefix of a 5000-step
warmup and cannot clear peak-LR behavior, long-run convergence or other
architectures.
