# Numerical-only BF16 resolution: prospective reference contract

This milestone investigates the two retained pilot update-1000 numerical
failures. It authorizes numerical replay and isolated derivative/optimizer
experiments, not training, task-accuracy testing, a learning sweep, or a change
to the scientific architecture. Lambda remains the fixed configuration value
0.01; the two adapters are learned parameters. Diagnostic simplifications are
controls and do not become the validated production configuration.

## Immutable evidence and the existing screens

Use the FP32-trained checkpoint/slot 0 and BF16-trained checkpoint/slot 64 from
`.runtime/cdrm-tiled-pilot/20260907T212606Z`. Verify checkpoint, tensor packet,
fixture, source and criteria identities. The slots have different minibatches:
their error magnitudes cannot isolate the effect of training precision alone.
Preserve every original numerical failure and the original contract SHA256
`26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20`.
The global/per-tensor/maximum, unscaled-state and trained-Adam screens remain
unchanged. A diagnostic completing successfully is not numerical clearance.
Before checking a new candidate, freeze its source and execution policy.

## Three distinct references

1. A full FP32 forward/backward measures the distance of the mixed policy from
   FP32, including different forward operands. It is not a fixed-operand
   derivative oracle for a BF16 forward.
2. A local oracle uses the actual saved operands, saved forward auxiliary
   values and a common incoming cotangent. Independent FP64 VJP formulas test
   the intended chain-rule convention, with every intended cast and reduction
   recorded. The literal quantized forward is discontinuous or locally
   constant; its finite differences do not validate AMP's cast-backward rule.
   Smooth FP64 primitive checks can validate the oracle itself. Replayed
   normalization, softmax, ALiBi and activation semantics must match the saved
   operation; promoting operands and recomputing a different forward is a
   separate diagnostic.
3. An optimizer oracle starts from identical retained FP32 parameters and
   Adam state. It consumes the saved raw or clipped gradients without running
   a model. FP64 clipping/Adam isolates sensitivity to gradient perturbation;
   explicit FP32 parameter writes distinguish arithmetic error from the
   representability of a small update to an existing weight.

Use unnormalized cotangents. Keep direct proposed-memory and unscaled bridge
checks so lambda cannot hide a defect. Preserve canonical ownership, complete
gradient coverage and the original near-zero mask. Explain missing/duplicated
dependencies or systematic VJP discrepancies; no tolerance excuses semantics.

## Prospective CPU clipping/Adam local screens

The retained optimizer is one canonical parameter group, non-fused,
non-foreach AdamW, with FP32 master weights/moments, no AMSGrad/maximize,
weight decay zero and norm-2 clipping at 1. Use the actual recorded LR, betas,
epsilon and step, including bias correction. Unsupported settings fail closed.
All scalar and vector diagnostic reductions use CPU FP64. Let `u=2^-24` and
`s=2^-149` (FP32 unit roundoff and smallest positive subnormal).

* Compare the native global norm with the independent FP64 norm using relative
  error `<=16*u`. This is a prospective reduction engineering screen, not a
  universal theorem for arbitrary norm kernels. Reproduce the clipping scalar
  from the saved native FP32 norm and require the saved clipped gradients to
  equal a single FP32 multiply by that scalar exactly.
* With identical saved clipped gradients, form FP64 ideal moments `m*` and
  `v*`. Bound the native first-moment error by
  `8*u*(beta1*abs(m0)+(1-beta1)*abs(g))+8*s`, and the second-moment error by
  `8*u*(beta2*v0+(1-beta2)*g*g)+8*s`. These conservative operation-count
  screens accommodate lerp/multiply-add implementations and cancellation;
  retain every coordinate outside them.
* Condition separately on the native saved post-step moments. Compute the
  ideal Adam increment `d` in FP64 from those moments. The native final weight
  must differ from `w0+d` by at most
  `16*u*abs(d)+2*u*max(abs(w0),abs(w0+d))+8*s` per coordinate. The first term
  allows the finite sequence of FP32 denominator/update operations; the
  second explicitly accounts for writing the result to FP32. Record the
  tighter correctly rounded ideal-weight comparison too, without imposing
  bitwise equality across CPU/CUDA arithmetic orders.
* Require exact step advancement, correct shapes, finite values, nonnegative
  second moments and exact packet consistency. The archived delta is formed
  by subtracting promoted FP32 weights in FP64, so it must equal that exact
  subtraction. Do not attribute its error to a nonexistent FP32 subtraction.

Apply these local screens to naive FP32, tiled FP32 and tiled BF16 alike,
before interpreting the BF16-vs-FP32 update difference. Fixed synthetic CPU
tests cover zero/tiny gradients, cancellation, clipped/unclipped inputs,
nonempty moments, step/bias correction, FP32 write quantization and injected
errors. The screens are not replacement tolerances for the old BF16 guards.

Compare raw-gradient magnitude/direction, independently norm-clipped
gradients, and gradients under one common FP32-reference clipping coefficient.
Report native, ideal-FP64 and FP32-write-realized Adam deltas. The common
coefficient is a causal diagnostic only: it changes the clipping rule and is
not a candidate training policy. Contributions need not add as percentages;
report the vector decomposition and cross terms. Do not change optimizer
hyperparameters to make a discrepancy pass.

## Local VJP screens and decision limits

Operation-specific VJP screens must be declared before their probe results,
using their actual casts, accumulation dtype and reduction length. FP64
arithmetic plus expected output casts is the reference; universal bitwise
equality or one-ULP accuracy for every cancellation-prone reduction is not
required. Retain coordinates, error energy and scale-aware diagnostics.
The already explained raw-side FP32 cancellation flags stay visible.

For the planned SDPA probe, use independent analytic FP64 softmax-attention
forward and VJP formulas at the captured q/k/v, actual additive mask and
incoming output cotangent. Validate those formulas against FP64 autograd on
small smooth cases. Preserve the exact causal mask in computation; exclude
masked future positions only from bias-error summary statistics. The original
FP32 ALiBi is a separate operand-perturbation control, not a replacement for
the mask actually used by BF16 execution.

Before these local results, define `r` as the analytic FP64 coordinate,
`f=2e-6+2e-5*abs(r)`, and `Q_d` as rounding to the native output storage dtype.
The prospective local engineering screen requires the native coordinate to
lie inclusively in `[Q_d(r-f), Q_d(r+f)]`. This allows rounding-boundary
crossings compatible with the existing FP32 floor; it is not a universal
floating-point theorem. Record errors against both `r` and `Q_d(r)`, ULP
statistics, and the residual of local FP32 SDPA on the identical representable
operands. Require that local FP32 check against the same analytic oracle and
floor before attributing a failure specifically to BF16. If it fails, retain
the unresolved oracle/arithmetic-floor question instead of relaxing a budget.

Prefer localization of the earliest material forward/adjoint discrepancy to
a broad precision factorial. The existing lambda-zero and head-only controls
need not be repeated without new evidence. Permit at most two causally
motivated precision candidates before a review. Restore the complete CDRM
path for any candidate confirmation and preserve initialization regression
and unscaled branch semantics as applicable.

A change is resolved for the tested scope only when it addresses the
identified mechanism and passes the unchanged full-model/optimizer screens
on both retained failing cases without breaking semantic or local checks.
Correct local VJPs with excessive full-policy distance support an
"explained but unresolved" disposition; BF16 remains experimental. If a
selective fix is ineffective or effectively requires FP32 throughout, pause
and retain FP32 as the dependable execution policy. Numerical evidence alone
does not establish equivalent learning or task performance.
