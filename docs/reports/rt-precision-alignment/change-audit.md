# Precision and correctness changes retained for standard RT

This audit compares the current code with the authors' upstream revision
`a21b42d2bc292edb86ed1b62cee4bcab809a9d21`. It separates correctness repairs from
precision choices that this milestone can evaluate. It is a source/evidence
audit, not a new numerical pass for either BF16 policy.

The current experiment is the standard model with **all twelve blocks tiled
recurrent**, D1024, H16, FFN4096, rho 1, causal ALiBi and learned Q/K
normalization. It uses native shifted CE, FP32 master parameters/residuals/
gradients/Adam moments, BF16 eligible dense operations, and four internal MLP
backward chunks. CDRM is disabled. The previous configuration with only block 3
recurrent and ordinary blocks around it is a different numerical setting.

## Source identity and meaning of `legacy`

The audited upstream is
[geniucos/recurrent-transformer at a21b42d](https://github.com/geniucos/recurrent-transformer/tree/a21b42d2bc292edb86ed1b62cee4bcab809a9d21).
The local vendor history also contains our ownership/guard repairs, recurrent
precision policy and CDRM changes. Its local `HEAD` is consequently not a
pristine upstream reference. Source snapshots and hashes for each execution,
rather than that local commit alone, identify the actual tested arithmetic.

The `legacy` branch preserves the upstream-style recurrent mixed-precision
arithmetic while retaining unrelated repairs. It does **not** mean all recurrent
state is BF16. Upstream already allocates the forward weighted-value state and
normalization denominator using the residual dtype, and K/V gradient buffers
using that same FP32 dtype. Its backward softmax explicitly computes FP32
before casting the result to the working query dtype.

Our protected `bf16_fp32_state` policy additionally promotes attention working
views and several backward intermediates, disables autocast around that
arithmetic, and preserves the actual BF16 dense replay policy separately. Both
policies retain BF16 projected/permanent K/V storage under the tested mixed
profile. Arm C disables autocast and TF32; the protected-policy switch is
inactive during this full-FP32 computation.

One detail must be measured rather than inferred: the earlier
[R3 report](../r3-bf16/results.md) observed **legacy running maxima already in
FP32**. That observation does not establish every shape/compiler path, but it
rules out describing the earlier failure as a proven BF16-running-max defect.
The new tiny observer records projected tensors, initial/running state,
compiled-helper input/output boundaries, reconstructed attention and adjoints
for all three arms. An FP32 destination does not prove an FP32 incoming
contraction or accumulation, and tensor boundary dtypes do not establish the
precision of an internal fused exponential or GEMM accumulator.

## Classification of earlier changes

| Change | Classification and decision | Effect with all recurrent blocks and CUDA graphs |
| --- | --- | --- |
| Canonical ownership of modules shared by pre/post-attention helpers; strict loading of agreeing legacy checkpoint aliases | Retain: parameter/checkpoint correctness. | Prevents duplicate registration and ambiguous checkpoint ownership. CUDA capture depends on stable parameters and does not replace this repair. |
| Complete normalization/activation-state copying and semantic checks during sequential-to-recurrent conversion | Retain: conversion correctness. | Inactive when constructing the standard RT directly, but reverting it would weaken a separate supported path. |
| Causal masking for direct ordinary block calls with raw ALiBi; repaired benchmark bias construction | Retain: masking correctness. | The ordinary-attention fix is inactive in the all-recurrent model. Native recurrent execution is structurally causal; graph capture must use that same model path. |
| Guards for unsupported recurrent cache/packed inputs, rho and configuration combinations | Retain: explicit execution contract. | A faster execution path cannot make an unsupported semantic combination valid. The current fixture remains dense, fixed-length, rho 1. |
| Explicit outer autocast and preservation of forward dtype for dense backward recomputation | Retain: replay correctness. | Promoting arithmetic Q must not accidentally switch the dense recomputation to FP32. The protected path records the actual forward autocast state; the legacy branch keeps its upstream-style projected-dtype replay. Graphs capture whichever path was selected at construction. |
| `bf16_fp32_state` recurrent attention/adjoint promotions | Evaluate: optional precision protection. | This is the A/B treatment, including backward working tensors. Previous evidence justified scoped use, not a universal need for these promotions at all model sizes or recurrence placements. |
| FP32 ordinary-attention option and explicit math-SDPA diagnostic path | Retain but exclude from this experiment. | Earlier CDRM analysis identified ordinary-backbone attention as a major error source. There are no ordinary blocks in the present RT, and its recurrent precision flag is a different control. |
| Eager reference paths and diagnostic observers outside compiled helper bodies | Retain for diagnosis only. | They help localize differences but do not clear compiled/captured deployment by themselves. Remove all observers and hooks before graph validation or timing. |
| Compiled-helper limits, fail-on-limit behavior and fallback auditing | Retain: execution reproducibility. | Graph replay reduces Python overhead, but warmup and capture still select compiled kernels. Cache identity and no-fallback checks remain relevant. |
| Retention of the Inductor compilation/tuning cache | Retain for reproducible recovery within the recorded runtime. | A prior controlled repeat recovered bitwise only with the retained cache. This is different from autocast's cached BF16 weight views and is not a cross-hardware guarantee. |
| Disabling the old `make_graphed_callables` wrappers | Retain: graph/autograd contract. | Tiled backward performs nested backward calls that accumulate parameter gradients as side effects. The wrapper's explicit `autograd.grad` contract does not safely describe those hidden writes. |
| Explicit whole-forward/loss-backward CUDA capture, stable gradient buffers, replay guards and scoped autocast | Retain: validated graph implementation. | Captures nested gradient writes, zeroes persistent gradients once per replay, and replays weight casts after optimizer changes. Clip/Adam remain outside capture. A fresh graph is required for each precision policy; changing config after capture must fail. |
| Synchronization and empty-cache calls before side-stream warmup and before graph capture | Retain: setup memory management, no model-math change. | Avoids duplicated temporary reservation from an earlier uncaptured reference. The previously measured steady workspace remains part of the physical headroom assessment. |

The ordinary FP32 path, CDRM fabric precision/gates, sequential conversion,
naive recurrence and direct-block benchmark paths are not ablation variables
for this standard RT comparison. Their retained presence in the code does not
mean they execute in the current model.

## What the earlier numerical evidence does and does not justify

The [tiled R3 investigation](../r3-bf16-tiled-resolution/results.md) localized a
real naive/tiled gradient difference to repeated rounding at a shared BF16
weight-cast node. Tiled's batched dense backward was closer to the same-operand
FP64 contraction. Changing tiled to imitate naive BF16 would therefore be the
wrong target. There is no established reason to disable autocast's weight
cache globally for the tiled model.

The same investigation passed prospective tiled-BF16-versus-tiled-FP32 screens
on fresh B64 cases, while preserving separate strict FP32 rounding-scale flags.
Its model had D256 and only one recurrent block. It supports the chosen
diagnostic method and a scoped historical precision policy; it does not clear
all-12 recurrence at D1024/T512/B512.

The [CDRM numerical investigation](../cdrm-numerical-resolution/results.md)
showed that bypassing a fabric or setting its injection gate to zero did not
remove the main mismatch. Protecting ordinary attention helped materially;
ALiBi-only protection and disabling BF16 reduced-precision GEMM reduction did
not resolve that fixture. These observations support isolating the actual
arithmetic path instead of attributing a discrepancy to recurrence or a gate
from architectural appearance alone. They do not justify enabling ordinary
attention protection in a model that has no ordinary blocks.

The [CUDA-graph milestone](../rt-cuda-graphs/results.md) established exact
same-precision agreement for the protected path, including all initial B512
gradients and the ten-update loss/norm trajectory. That is strong evidence
for the capture implementation, not a BF16-versus-FP32 result. The legacy
candidate requires its own observer-free capture/optimizer checks.

Keep TF32 disabled, FP32 master/optimizer state, normalization, ALiBi, loss
scaling, clipping, optimizer settings, data and initial weights fixed across
A/B. Keep BF16 reduced-precision-reduction and autocast-cache settings recorded
and unchanged unless a later targeted diagnosis isolates one of them. These
are controlled comparison choices, not claims about undocumented author
acceptance criteria. Prior maximum-error budgets remain investigation triggers;
they must not be presented as tolerances published in the paper.

## Bounded dtype-observer contract

[`scripts/rt_precision_dtype.py`](../../../scripts/rt_precision_dtype.py) runs
one fresh process per arm: A protected BF16, B legacy BF16, or C strict FP32.
It constructs the same tiny D64/H4/FFN256, two-block all-recurrent model,
B3/T16, rho 1 and head microbatch 2. The final one-example head chunk exercises
the tail case. It uses the native shifted-CE objective with denominator B×T;
it performs no optimizer updates or CUDA capture.

After two compiled warmup backwards, the driver records an unobserved
reference backward, one metadata-observed backward and a second unobserved
backward after cleanup. It requires bitwise equality of loss and every raw
parameter gradient, finite FP32 gradients, unchanged model state and complete
observer removal. Metadata callbacks retain no tensors and do not inspect
values of uninitialized storage. Helper wrappers invoke the original compiled
functions unchanged. Compiler auditing, source snapshots, runtime identity,
initial-state/token hashes and required online W&B records accompany the
result. Each arm uses the same deterministic initialization and token seed;
the retained hashes allow the combined evidence audit to verify alignment.

This proves observer neutrality only if the run passes. It provides boundary
dtypes on a small fixture, not full-size kernel equivalence, accumulator
precision, a mathematical-gradient oracle or training-quality clearance. The
next gates remain observer-free legacy capture validation, identical-state
A/B/C numerical comparisons, and the separately bounded real-data comparison.
