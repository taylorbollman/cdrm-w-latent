# F4 continuation after bounded roundoff diagnosis

Decision2026-09-23, after completed diagnostic `f4-rt-fbt-roundoff-02` at
runtime949731b. The original protocol and failed report remain unchanged.

The RT+FBT BF16 materialized-versus-recompute coordinate gate remains failed:
layer11.ff_proj max/reference-peak6.349206% versus6.25%; globalL2 1.020619%
and worst tensorL2 1.380478% pass their original limits. No acceptance threshold
or model/kernel math has changed.

Fixed-state BF16 materialized/recompute/eager-history repeats are bitwise.
Candidate initial/changed-token/overwrite/changed-weight graph checks and
three eager versus three graph complete Adam updates are bitwise. Changing
only recompute historical backward to eager yields globalL2 .857865% and
worst-coordinate ratio4.875283%, within the original limits. FullFP32 math
materialized versus recompute globalL2 is3.036839e-6, worst tensor4.431382e-6,
worst coordinate/reference-peak2.533063e-5. FP32 falls back to eager and does
not directly validate BF16 Triton. These observations support reduction and
rounding sensitivity rather than an identified semantic implementation bug.

Broader precision sensitivity is visible: BF16 materialized/fused-recompute/
eager-history differ from FP32 materialized by17.9721%/18.2783%/18.0080%
global gradientL2. This diagnostic changes forward precision and ordinary
attention backend too, so it does not isolate a single operation. All arms
are finite; the original BF16 control already shares the broad discrepancy.
Do not call this full-native BF16-versus-FP32 numerical clearance, infer
NextLat fixes precision, or conclude there is no possible learning impact.

Resume **resource-only** measurements under these explicit qualifications.
Their finite-update checks establish execution and timing at the measured
batch, not reference-gradient equivalence there. In the host queue only,
skip re-executing the completed failed correctness job but keep it among
final inputs. Continue combined short correctness/operator audit and the
common B64 matrix, then allowed B96 trials. Stop on additional numerical or
finite-state failures. All result tables must mark RT+FBT as qualified and
include its original failed reference screen. Separate six diagnostic Adam
updates from the main F4 matrix counts. No learning run, threshold adjustment,
Q/K change, optimization rewrite or all-combinations numerical-clearance claim.

Before substantive learning, review the broader BF16/FP32 gradient sensitivity
and decide whether a small transition-state/clipped-update check is useful.
The present evidence does not justify automatically adding Q/K normalization.
