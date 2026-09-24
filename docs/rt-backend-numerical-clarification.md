# RT backend qualification and selected ordinary execution

2026-09-24. Clarification after PR27; no new GPU experiment in this note.

## Selected RT backend

The user subsequently chose optimized native RT and to move on from backend
selection after confirming the existing speed comparison. Native is already
the library default. Keep the author-derived port as an experimental reference;
do not remove its evidence or call the BF16 compatibility issue resolved.

The actual16-layer OLMo model at B64/T512, with RT at layers0/15 and CUDA graphs,
measured native/author22,430.8/22,425.7 input tokens/s. With K2 FBT and NextLat,
the pair measured11,195.8/11,186.5. These are effectively tied five-update
measurements. Isolated B128 one/two/six-block fixtures favored author by7.5–8.6%
and used less memory; B32 favored native. Those isolated rates are not full-LM
rates. The full-model comparison predates PR26/27 ordinary optimizations.

For SwiGLU, retain the validated rounded compiled ordinary helper. For future
RT optimization, prefer a bounded compile experiment over its contiguous
per-position finish/writer work, with existing batched weight gradients retained.
Specialized library kernels remain candidates if they improve the measured
path; no claim is made that compilation is fastest among untested libraries.

## Selected ordinary configuration

The user accepted the recommendation to use Dao RoPE and fused AdamW. Use
these explicitly for upcoming ordinary T512 development runs, together with
rounded compiled SwiGLU, deterministic Flash SDPA, all-layer activation
checkpointing, native RoPE-table reuse, CE chunks2048 and the validated graph
boundary. The conservative physical batch is64. This is a future-run choice,
not an edit to historical frozen comparisons or old checkpoint identities.
The library defaults remain available for reproduction. Record all options in
each future run/checkpoint configuration. RT/FBT/NextLat and cache/online
combinations still need their own bounded integration checks.

## What 31% and 16% mean

The full-model comparisons are author-derived BF16 versus native BF16, at
identical checkpoint/auxiliary weights, examples and supervision, before
gradient clipping. The metric is

\[
\frac{\|\nabla_\theta L_{\mathrm{author}}-
          \nabla_\theta L_{\mathrm{native}}\|_2}
     {\|\nabla_\theta L_{\mathrm{native}}\|_2}.
\]

It is0.312458 for RT-only and0.162606 for K2 FBT+RT+NextLat. RT selects layers0
and15 of the16-layer OLMo model. K2 here is an ordinary bootstrap plus one
feedback/RT pass, with NextLat objectives on both. These are relative vector
differences across all participating raw parameter gradients, not fractions of
incorrect gradients, accuracy losses or errors against a full-model FP32 oracle.
The native implementation is the comparison denominator, not established truth.
The combined metric has a different objective/gradient composition and identical
ordinary-bootstrap forward outputs and losses; its smaller ratio does not prove
greater stability. Later-pass gradient contributions through that bootstrap
can still differ.

## What the 0.33% local test establishes

For block0, write output z=f(x;theta) and incoming error signal g=dL/dz. Its
parameter gradient is J_theta(x,theta)^T g. There are two sources of variation:
the local block computation/Jacobian and the signal supplied by the rest of the
model. The fixed-input/shared-cotangent test uses the same x, weights, positions
and g for both backends; each still executes its own local recurrent forward.
Their parameter-gradient relative L2 is0.003327 (0.333%). Their full-FP32 local
versions agree to4.65e-7. These are block0 tests, not full-stack FP32 tests.

The subsequent fresh capture measured:

| Quantity | Author/native difference |
| --- | ---: |
| Block0 input and positions | Bitwise identical |
| Block0 output relative L2 | 0.003639 |
| Incoming error signal g relative L2 | 0.786611 |
| Block0 parameter gradients, each with its own g | 0.443495 |
| Block0 gradients with one shared native g | 0.003327 |

Each local implementation exactly reproduces its own full-model block0
parameter gradients when supplied its own captured g. Holding the author
local computation fixed and replacing only g recreates approximately the
44% discrepancy. This is strong evidence that a large local block0 VJP
disagreement at the same input/direction is not the main explanation here.
The original full-model block0 ratio0.437355 and fresh0.443495 are different
captures, not claimed bitwise reruns.

It does not explain why g differs so much. Downstream activations, ordinary
layers, the second RT layer, attention/loss sensitivities, and backward
rounding/reconstruction can all contribute. Different forward rounding can
lead to different downstream derivatives; a downstream implementation issue
also remains possible. It is not established that forward amplification alone
explains the measurements. The separate combined discrepancy is not cleared
by the RT-only localization. Exact own graph/eager agreement establishes that
capture preserves each calculation, not that both calculations are equivalent.

Thus no demonstrated gross block0 backward bug, but no demonstrated harmlessness
or complete-model FP32 winner either. A raw-gradient difference is also not the
same as an Adam update difference after clipping and moment normalization.

## Deferred diagnostic if backend equivalence becomes necessary

The user-selected supported implementation is native; no new comparison is queued.
Its existing integration breadth and tied full-model speed are reasons to avoid
a premature switch, not numerical evidence of its superiority. Preserve a frozen
shared ordinary configuration during this diagnostic; do not fold new kernel
changes into the same comparison.

1. Obtain an actual full-stack FP32 anchor for the same failing RT-only fixture
   if feasible. If a smaller microbatch is necessary, first reproduce the
   discrepancy there and state that B8/T512 remains untested against FP32.
   Compare both BF16 paths to it, not only to each other; add gradient norm,
   direction and actual clipped-Adam update differences to raw L2.
2. Capture activation and incoming-gradient differences at layer boundaries.
   Change layer0 and layer15 backends separately, then together, to distinguish
   an early activation perturbation from the later recurrent layer or their
   interaction. Do not assume a layer with a large incoming error is its source.
3. Only where the trace implicates a region, selectively increase precision or
   align an operation/rounding boundary and see whether the discrepancy shrinks.
   If the RT-only mechanism is understood, repeat a bounded combined check.

This is a proposed next diagnostic, not a new GPU queue or long training run.
The aim is to distinguish an implementation error from mixed-precision sensitivity
and assess actual update relevance, not require bitwise agreement across kernels.

Evidence: [full integration and localization report](reports/olmo-rt-author-integration/results.md),
[original protocol](reports/olmo-rt-author-integration/protocol.md),
[localization protocol](reports/olmo-rt-author-integration/localization-protocol.md).

## Remaining ordinary kernel candidates

Dao's standalone `swiglu(gate, up)` uses our native packed `[value,gate]`
ordering and can preserve existing parameters, norm and residual operations.
It avoids the TE module ownership/normalization concerns. The installed and
currently inspected upstream function uses CUDA Jiterator forward/manual
backward with FP32 internal arithmetic and one output cast. It does not retain
the BF16 SiLU intermediate rounding preserved by our rounded compiled helper.
Compare those two functions, and possibly the broader RT finish/writer region,
before claiming an additional speedup or compatibility. The general Dao
`FusedMLP` class is not automatically a SwiGLU replacement.
[Dao activation source](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/flash_attn/ops/activations.py),
[Dao packed MLP call](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/flash_attn/modules/mlp.py).

FA4 remains a candidate for capacity as well as speed, but measured full-model
memory savings are effectively zero so far. At both B64/T512 and B16/T2048,
the matched all-checkpointed SDPA/FA4 arms have setup peak allocated/reserved
26.706/46.248GiB and steady peak18.495/46.248GiB. The setup allocation differs
by only1KiB. The isolated attention study measured speed, not memory. SDPA is
already using Flash attention; these comparisons are not against materialized
quadratic score tensors. Attention represented3.75%/9.01% of named device
time atT512/T2048 in that older reference, limiting whole-step speed gains.

If revisited, compare SDPA/FA4 on the newly selected ordinary configuration in
separate fresh processes at fixed batch/checkpointing, recording setup and
steady allocated/reserved peaks and useful batch capacity. Longer contexts may
change the tradeoff. Keep the prior small loss-screen qualification visible;
neither a memory win nor the new kernel combination is yet established.
[Retained ordinary measurements](reports/olmo-ordinary-efficiency/results.md).
