# One NF-only fusion precision diagnostic

2026-09-29. Written before execution under the initial protocol's allowance
for one conditional precision boundary. The eight-case matrix is complete;
its runtime source `6ed920a` and protocol remain frozen.

## Evidence and hypothesis

The matrix found shared-backbone BF16/FP32 gradient differences of0.9804%
for ordinary computation,25.4583% for RT-only,60.8698% for FBT-only and95.8461%
for their combination. Shared states, inputs/noise and first-pass identities
passed, and the original NFR endpoints reproduced exactly. Reference report:
`.runtime/olmo-recurrence-precision/matrix-01/report.json`, SHA256
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`.

FBT-only is already sensitive without RT, so it provides a clean setting for
testing one inexpensive precision boundary: feedback fusion. The hypothesis
is that rounding injected at fusion materially affects subsequent feedback
states and gradients. This is a candidate test, not proof that fusion is the
cause; repeated ordinary-stack computation may instead amplify perturbations
originating elsewhere.

## Three CE cases, six physical backwards

Use NF only, with the same original two physical B2/T16 records, four FBT
passes, no RT layers, source checkpoint, fusion/predictor seeds, shared noise,
beta1, jitter0.02, masks, denominators and CE pass weights. Keep NextLat branches
active with zero auxiliary cotangents exactly as in the matrix.

1. Recompute the full FP32 math reference.
2. Recompute the existing production BF16/Flash reference.
3. **Candidate:** production BF16/Flash with autocast disabled only inside
   `FBTGateProduct.forward`. Run its existing projections, gate product and
   normalization on FP32 inputs and FP32 master weights. Keep its original
   FP32 return dtype, native output-scale buffer and formulas unchanged.

There are three aggregate CE gradient cases, six physical model backwards.
The first two repeat references because complete gradient vectors were not
saved. Compare their metrics, forward fingerprints, component gradient
summaries and within-pair precision geometry exactly against the matrix.

The candidate changes neither ordinary attention nor backbone projections,
MLPs, norms, loss arithmetic, jitter application, masking, beta or pass policy.
There is no global autocast change. Use a temporary instance-scoped wrapper,
restore the exact original callable on success or exception, and test that
ordinary backbone execution remains under the outer BF16 autocast. No edits to
core model code or existing diagnostic helpers are needed.

## Comparisons and guards

The candidate's first-pass hidden/embedding/input fingerprints must be byte
identical to production BF16: fusion has not yet executed. Record wrapper
call count, actual input/output dtypes, autocast state and parameter/buffer
integrity. Expected scope is three fusion calls per record, six in total;
ordinary checkpoint replay must not broaden that scope.

Compare full and shared-backbone raw-gradient relative L2, cosine, norm ratio
and absolute difference against FP32 and the original BF16 path. Measure
valid-token hidden drift by pass and total incoming cotangents. Look at both
physical records; an improvement in one scalar loss or one norm is not enough
to claim a resolution. Different gradient-space and hidden-space error ratios
must not be interpreted as identified amplification factors.

The existing strict numerical qualifications remain unchanged. Operational
pass/fail covers reference reproduction, finite execution, expected gradient
participation, first-pass identity, wrapper scope/restoration and source,
parameter/buffer/input/RNG integrity. No newly chosen error threshold converts
this experiment into numerical acceptance.

## End of this bounded follow-up

If the candidate does not improve forward and gradient agreement together,
retain that negative result and stop this milestone rather than promote more
modules. If it improves them, report the improvement and remaining discrepancy;
adoption still requires a separate combined-model and packed T1024 confirmation.
An NF-only result cannot clear NFR or the separate auxiliary loss-layout issue.
No quality training, Q/K normalization, architecture change, distributed
execution or throughput claim is added here.

Apply the same deterministic controls before CUDA, TF32 off, FP32 masters,
900second timeout, online W&B and GCS retention. Use fresh output directories,
capture source bytes and write per-case progress. Preserve work every20–30min.
