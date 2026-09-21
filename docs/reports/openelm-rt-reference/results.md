# OpenELM native RT reference: Stage B review milestone

Completed 2026-09-21. **The sequential FP32 reference passes its bounded
correctness checks on the real 1.1B checkpoint.** It is ready to serve as the
target for native tiled implementation. No weights were trained or changed.

The checkpoint-import [PR #1](https://github.com/taylorbollman/cdrm-w-latent/pull/1)
was merged at `959c225`. Stage B uses branch `feat/openelm-rt-reference`.
Its implementation/evidence commit is `6603521`, tracked by
[PR #2](https://github.com/taylorbollman/cdrm-w-latent/pull/2). Later
documentation-only changes do not alter the validated source hashes.
The user authorized direct PR closure/merge and asked us to stop at the next
review milestone. Tiled execution and learning are subsequent work.

## Model and computation

Native OpenELM-1.1B individual 300k checkpoint: **28 layers, residual width
2048, head dimension 64, 4:1 GQA, native variable attention/FFN widths, SwiGLU,
headwise Q/K RMSNorm before RoPE, and tied embeddings/readout**. All 32,128
vocabulary rows and native padding ID 32,000 are preserved. There are still
**1,080,153,600 parameters in 226 tensors**, with no new trainable parameters.

The actual-checkpoint test makes **only layer index 0 recurrent**. At a selected
layer, position t attends to persistent K/V from previous completed block
outputs and temporary self K/V from its current input. The persistent source is

\[
m_t=(1-\alpha)x_t+\alpha z_t,
\]

followed by the native attention-input normalization and K/V projections.
Here z is the complete attention-plus-MLP block output, **before model final
normalization**. Both branches remain attached to gradients. Alpha 0 executes
the actual scan and recovers ordinary causal attention mathematically; alpha 1
uses completed outputs for historical memory. The full equations and API are
in [usage](../../openelm-rt-reference-usage.md).

## Correctness evidence

The final GPU check used the H100 80GB **inside the project container**,
FP32 parameters, physical **B1**, a fixed **16-token** native-tokenizer fixture,
TF32 disabled, and math SDPA for semantic comparisons. All parameter gradients
and input-embedding activation gradients were checked through next-token CE.

The independent RT oracle rebuilds earlier memory sources at every query and
uses explicit matmul/softmax plus pinned original CoreNet blocks. It does not
call the production scan or share its persistent cache implementation.

| FP32 comparison | Logit relative L2 | Global gradient relative L2 | Maximum gradient absolute difference |
| --- | ---: | ---: | ---: |
| Alpha 0 scan vs native ordinary | 3.412e-7 | 1.269e-6 | 3.219e-6 |
| Alpha 0 scan vs independent RT oracle | 4.261e-7 | 1.176e-6 | 2.265e-6 |
| Alpha 0.37 scan vs independent RT oracle | 5.484e-7 | 2.313e-6 | 2.670e-5 |
| Alpha 1 scan vs independent RT oracle | 6.965e-7 | 1.908e-6 | 4.578e-5 |

Every case has all **226 gradient tensors** present and finite. Across the
four comparisons, the largest individual tensor's relative L2 error is
9.371e-6; the largest maximum error normalized by that tensor's maximum
reference magnitude is 1.515e-5. No claim of bitwise identity is appropriate:
the scan and independent oracle change GEMM shapes and reduction order.

The initial attempt used an elementwise gradient tolerance of
`2e-6 + 3e-4 * abs(reference_element)`. At alpha 0.37 it flagged the bottom QKV
tensor, despite global gradient relative L2 of 2.313e-6. The final run retains
that diagnostic, plus a similar alpha 1 flag in layer 6 QKV. These small errors
are consistent with reduction rounding and cancellation near zero; this is not
a proof attributing each coordinate's error to a particular kernel.

The accepted FP32 parameter-gradient budget therefore requires **both** a
relative L2 error no greater than 1e-4 (or whole error norm no greater than
2e-6 for nearly zero tensors) and maximum absolute error no greater than
`2e-6 + 1e-4 * max(abs(reference_tensor))`. This prevents one bad coordinate
from hiding in a global norm while avoiding a near-zero element's arbitrary
relative denominator. Tests demonstrate that both isolated and distributed
errors exceeding these limits are rejected. The original failed attempt is
preserved in [calibration-summary.json](calibration-summary.json).

Actual-checkpoint cache checks at all three alphas use 13 tokens in chunks
3+2+8. Maximum cached/full logit differences are 7.439e-5, 6.199e-5 and
4.292e-5 respectively. Changing future tokens changes earlier logits by
**exactly zero**. First-token temporary self semantics, native KV head counts
and incompatible-mode rejection also pass.

The scoped CPU-container suite passes **136 tests**. Its small models cover
bottom/top/multiple recurrent layers, 4:1 grouping, variable widths, nonuniform
Q/K norm gains, padding, absolute positions, fractional branch adjoints,
historical-input finite differences, full/chunked training gradients, and two
attached forwards sharing weights with different immutable modes. The composed
forward is a gradient-ownership test, not an implemented FBT experiment.

## BF16 observations and limitation

BF16 is **smoke-tested, not cleared for training**. FP32 remains the semantic
reference. The scan changes projection shapes even at alpha 0; the oracle
also reconstructs history and performs explicit FP32 attention. These paths
can have different BF16 rounding without implying different equations.

| Comparison | Logit relative L2 | Global gradient relative L2 |
| --- | ---: | ---: |
| BF16 alpha 0 scan vs BF16 native ordinary, math | 0.366% | 1.121% |
| BF16 alpha 1 scan vs BF16 independent oracle, math | 0.436% | 4.155% |
| BF16 alpha 1 scan vs BF16 independent oracle, default backend | 0.615% | 4.578% |
| BF16 alpha 1 scan vs its FP32 execution, math | 0.724% | 3.859% |
| BF16 alpha 1 scan vs its FP32 execution, default backend | 0.641% | 4.842% |

All outputs and all 226 parameter gradients remain finite. Both alpha 1 BF16
paths choose the same top token as FP32 at every fixture position. Their CE
losses are 5.93685 (math) and 5.92659 (default), versus 5.92037 in FP32.
These observations come from one short fixture; they do not establish quality,
stability over optimization, or acceptable precision at long recurrent lengths.
The default backend was requested but not profiled here, so this report makes
no specific flash/cuDNN dispatch claim for the recurrent path.

For context, the same fixture's ordinary/alpha 0 FP32 CE is 3.94575 and alpha
0.37 CE is 5.28407. Turning on recurrence changes a pretrained function before
any adaptation. This tiny-fixture loss increase is neither a task-performance
experiment nor evidence that recurrence cannot improve after training.

## Cache corrections from review

Caches now reject changed alpha/layer selection, another model, updated weights,
device/dtype conversion, changed autocast settings, changed grad/inference
context or configured attention backend. A dtype roundtrip can change weights
without changing PyTorch parameter version counters; a separate conversion
generation handles this case. Ordinary and recurrent caches are separate
types, and masks/positions are owned copies rather than aliases to mutable
caller metadata. Stage A source and validated math remain unchanged.

## Reproducibility and next step

Final report: `.runtime/openelm-rt-reference/validation-final/report.json`.
[W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/uowl622p),
[compact machine-readable summary](summary.json),
[CPU test record](test-results.txt).

GPU validation took approximately 41 seconds and peaked at 17.50 GiB allocated
and 18.23 GiB reserved. These measurements include the paired native/RT models
and their gradients; they are **not single-model training capacity estimates**.

Evidence retention uses
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-rt-reference/20260921T190151Z/`.
It reuses the previously retained 4.3GB native checkpoint by URI, object
generation and hashes rather than uploading it again. The
[storage receipt](storage-receipt.json) records verification.

The next milestone is native tiled execution/backward against this reference,
with the same immutable mode and cache contracts. Begin from the established
FP32 reference, then assess actual tiled BF16 behavior rather than transferring
these sequential-scan observations into an unsupported training claim.
Long-context/large-batch testing, optimizer adaptation, NextLat and FBT remain
subsequent scope. No GPU job or learning run needs resuming from Stage B.
