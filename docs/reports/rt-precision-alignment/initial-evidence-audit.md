# Independent audit of initial precision-alignment evidence

This audit covers the completed dtype A/B/C runs, tiny and full B2 legacy
CUDA-graph validation, B512 legacy capacity profile, tiny A/B/C comparison and
full B2 A/B/C comparison in
[`20260910T191100Z`](../../../.runtime/rt-precision-alignment/20260910T191100Z).
It is a snapshot after the first full B2 comparison closed. A B512
cross-precision comparison, second initialization and real-data training are
**not included** in this audit.

The executed legacy path has passed the available execution checks. The
full-size B2 numerical result requires review: legacy BF16 has 21 flagged
parameter tensors against FP32, chiefly Q projections and learned Q/K norms.
These are not missing/nonfinite gradients or CUDA-graph mismatches. They are
also not an unconditional numerical pass.

## Provenance and completed scope

I independently checked every recorded source snapshot against its SHA256 and
the current corresponding source for all eight completed reports. All matched.
Their W&B records are online and synced; their compiler audits report no
graph-break/unsupported fallback and retain fail-on-recompile-limit behavior.
The full B2 comparison's source set has 38 files. Its final report SHA256 is
`e2686adbb913ca5de5afb3c408a4ea3268f65f0e08bd4a2e8e912e70efe5e19f`.

| Completed report | Scope checked | Result |
| --- | --- | --- |
| [dtype A](../../../.runtime/rt-precision-alignment/20260910T191100Z/dtype/A/report.json), [B](../../../.runtime/rt-precision-alignment/20260910T191100Z/dtype/B/report.json), [C](../../../.runtime/rt-precision-alignment/20260910T191100Z/dtype/C/report.json) | D64/H4/FFN256, two recurrent blocks, B3/T16; native CE and all 21 parameter gradients | Both observed and subsequent observer-free backwards match the unobserved reference bitwise in each arm. Weights unchanged and observers removed. |
| [Tiny legacy graph](../../../.runtime/rt-precision-alignment/20260910T191100Z/graph/tiny-legacy/report.json) | Two identical-input checks and three changed-input Adam updates | Loss, all gradients, clipping norm, parameters and Adam tensors match uncaptured execution bitwise. |
| [Full legacy graph](../../../.runtime/rt-precision-alignment/20260910T191100Z/graph/full-legacy/report.json) | Actual twelve-block D1024 model, physical B2/T512; same checks | All 111 gradients/parameters and complete Adam state match bitwise. |
| [B512 legacy profile](../../../.runtime/rt-precision-alignment/20260910T191100Z/profile/b512-legacy/report.json) | Actual physical B512/T512, all twelve blocks recurrent; ten complete updates | Initial loss and all 111 raw gradients match uncaptured execution bitwise. Eight measured updates after two warmups; finite FP32 parameters/gradients/moments; no measured recompilation. |
| [Tiny A/B/C](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/tiny-seed0/report.json) | One shared initialization, random IDs and initially empty Adam state | No prospective review flags; optimizer disagreement remains material as quantified below. |
| [Full B2 A/B/C](../../../.runtime/rt-precision-alignment/20260910T191100Z/numerics/full-b2-seed0/report.json) | Actual architecture, first two frozen diagnostic-C4 rows, full physical FP32 reference | A–C has no review flags. B–C and B–A have per-tensor review flags. |

The full model configuration is all twelve tiled recurrent blocks, D1024,
H16, FFN4096, rho 1, causal ALiBi, learned Q/K normalization, no dropout,
untied 32128-row tables with 32100 valid token IDs, and four internal MLP
backward chunks. There are 151,045,120 backbone parameters and 216,843,264 total
parameters. Head microbatch 2 leaves the recurrent backbone at the stated
physical batch size. Loss is summed shifted CE divided by B×T; at T512 there
are 511 supervised targets per sequence.

All three dtype runs have identical initialization and token hashes. The tiny
numerical comparison shares that initialization but uses a different,
separately retained random token fixture. Do not compare its CE values directly
to the dtype fixture's CE. In each numerical comparison, the source loads the
same retained starting weights and optimizer state into every arm, and passes
the same saved ID matrix to every arm. The full B2 input records the exact
hash of the supplied diagnostic matrix and the selected two-row array.

For the tiny numerical comparison I additionally read the retained packets
inside an explicitly CPU-only container, checked their file hashes and shared
starting-state reference, and independently recomputed gradient relative L2,
actual parameter-delta relative L2 and delta cosine using CPU FP64 reductions.
All matched the report. Every tiny Adam state advanced exactly once. This
independent tensor recomputation was not repeated over the multi-GB full B2
packets; that result is supported here by source, report and coverage audits.
The separate
[retention-agent integrity audit](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/retention-agent-review-pre-b512.json)
also verified all full B2 starting/A/B/C packet hashes: 307 files and
13,897,696,533 bytes checked overall, zero errors. It reproduced global
gradient relative L2 and per-tensor review flags from the retained FP64
summaries, preserving all 21 B–C flags.

## Observed precision contract

The new runtime observations confirm that **legacy running maxima are FP32**
on this fixture. This agrees with the historical observation and corrects an
overly broad inference that legacy implies BF16 max state.

| Observed boundary | Protected BF16 A | Legacy BF16 B | FP32 C |
| --- | --- | --- | --- |
| Projected Q/K/V and output logits | BF16 | BF16 | FP32 |
| Scaled arithmetic query | FP32 | BF16 | FP32 |
| Initial/running max state | FP32 | FP32 | FP32 |
| Running weighted-value numerator and denominator | FP32 | FP32 | FP32 |
| Reconstructed attention probabilities and attention values | FP32 | BF16 | FP32 |
| Attention-adjoint `gs` working buffer | FP32 | BF16 | FP32 |
| K/V gradient destination buffers | FP32 | FP32 | FP32 |

These are tensor-boundary observations with demonstrated observer neutrality.
They do not identify internal fused exponential precision or GEMM accumulator
precision, and the tiny shape does not establish identical kernel selection
at D1024/T512/B512.

## Same-precision graph evidence

Both legacy graph validators preserve model state during capture, reject
changed shapes, reject `no_grad` replay and reject replaced gradient buffers.
Their changed-input updates test that graph replay uses changing tokens and
updated weights. The prefix causality check passes for the **native forward**;
it is not a separately extracted captured-prefix assertion.

At B512 the initial loss is exactly `10.842390060424805` in both executions and
every raw parameter gradient is bitwise equal. Ten captured optimizer updates
then complete on the fixed random batch. The B512 profile does not independently
compare all post-update model/Adam tensors to an uncaptured B512 control;
the complete changed-input optimizer equivalence checks are at B2 and tiny
size. Nor does same-precision graph equality establish BF16 accuracy against
FP32.

## Cross-precision results and qualifications

The table uses the native B×T loss denominator and actual post-Adam parameter
delta. Both numerical fixtures start with empty Adam moments and use an
identical-state, one-step AdamW comparison at LR 0.001 with clip norm 1.

| Fixture/comparison | Gradient relative L2 | Worst tensor relative L2 | Worst tensor max error / reference max | Adam-delta relative L2 | Adam-delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny A–C | 1.4228% | 2.1204% | 2.3439% | 13.4269% | 0.990986 |
| Tiny B–C | 1.5187% | 2.3883% | 2.5685% | 14.0788% | 0.990089 |
| Full B2 A–C | 1.0279% | 2.0199% | 2.7478% | 8.9121% | 0.996029 |
| Full B2 B–C | 1.1033% | 4.3224% | 6.2795% | 10.0610% | 0.994939 |
| Full B2 B–A | 0.4677% | 4.0251% | 6.1624% | 7.2922% | 0.997341 |

Full B2 A–C CE differs by −0.000559807 nats per B×T and B–C by
−0.000519753. Small initialization CE differences do not establish learning
equivalence. The tiny B–C Adam cosine is close to the 0.99 engineering review
trigger despite its nominal pass. The first Adam step is sensitive to gradient
signs and near-zero coordinates; it must not be described as an identical or
negligibly different update.

In full B2, 21 legacy-versus-FP32 tensors exceed the 3.125% tensor relative-L2
trigger. They are the Q projection and both Q/K normalization scales in
zero-based blocks 4, 6, 7, 8, 9, 10 and 11. Block 4 `k_norm.weight` also
exceeds the 6.25% tensor-maximum trigger, reaching 6.2795%. The largest tensor
relative error, 4.3224%, is block 9 `q_proj.weight`. The flagged tensors'
reference RMS values are roughly 1.39e-5 to 2.87e-5, all above the report's
2e-6 descriptive small-RMS marker. Thus the flags cannot simply be erased by
calling whole tensors numerically zero.

The global gradient cosine remains 0.999939 for B–C. Sampled layer-output
relative error is about 0.83% at block 0/position 128 and 0.94% at block
11/position 511; these limited samples do not show explosive forward growth.
About 97.62% of B–C Adam-difference energy lies in the predeclared near-zero
bucket, but 12,129 sign flips remain outside it. Near-zero accounting explains
much of the initial optimizer distance; it does not prove that every changed
direction is harmless or address trained Adam moments.

## Operational comparison with the earlier protected run

The prior
[protected B512 graph profile](../../../.runtime/rt-cuda-graphs/20260910T182400Z/profiles/b512-graph-confirm/report.json)
has the same hardware, random seed, exact input-ID hash and optimizer settings.
Its resolved model differs only in recurrent precision policy. All recorded
OLMo source hashes agree; the profiler and graph-helper files differ because
this milestone added the explicit policy interface. The compiler-cache
directories are different, so this is an informative historical comparison,
not a newly paired timing experiment.

| B512/T512, eight measured updates | Prior protected A | Current legacy B |
| --- | ---: | ---: |
| Mean seconds/update | 6.1393 | 5.4541 |
| Input tokens/second | 42,700 | 48,064 |
| Peak reserved memory across setup/steady phases | 49.074 GiB | 39.229 GiB |
| Minimum sampled post-update physical free memory | 27.299 GiB | 37.309 GiB |

Legacy is 11.16% lower in update latency, or 12.56% higher in throughput, in
this comparison. CUDA replay can make allocator *allocated-peak* counters miss
graph workspace; reserved memory and sampled physical free memory are the
useful evidence here. Post-update free memory is a sample, not a continuous
minimum. These random-token profiles do not establish real-data learning
quality, and future training-driver timings include more per-update
verification than this capacity probe.

## Disposition at this snapshot

The available evidence supports continuing bounded diagnosis. It does not
support replacing the protected default yet. The full B2 flags are coherent
attention-gradient differences, with modest global error and no observed
execution defect, rather than evidence by themselves that training must fail.
I would not prohibit the short 100-update pilot solely because of these
engineering flags once the planned B512 precision comparison closes without a
materially worse result. Preserve the flags and treat that pilot as evidence
gathering, not as retroactive numerical clearance.

An operand-fixed attention-adjoint check at a flagged block is a useful
bounded localization: compare the actual legacy backward with FP32/FP64
attention differentiation using identical rounded Q/K/V operands and a frozen
incoming attention gradient. That separates attention-backward rounding from
forward operand drift without changing the architecture or optimization.
The later decision also needs the second initialization, actual B512 scope,
trained nonempty-Adam states and the frozen development/confirmation data.
A 100- or 500-update pilot retains the released 5000-step warmup and therefore
does not clear peak-LR behavior or long-run convergence.
