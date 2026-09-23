# Integration execution notes

Runtime/protocol freeze:2c38d649, PR25, after isolated comparison PR24 merged9ce9523.
Native constructor behavior is unchanged. CPU236 integration/harness regressions
and40 retention tests pass. The small preparatory-batch hashing addition after
the236-test run is evidence-only; evidence tests cover the complete hash schema.

The first actual RT-only B8/T512 run fails cross-backend BF16 numeric comparison:
aggregate raw-gradient relativeL2.312458; final outputL2.008860,maxratio.062551;
CE relative change.00149439. All own Flash/graph/Adam/changed-weight checks pass,
with six physical optimizer updates. The failure is retained, not relabeled.
The combined K2+NextLat screen similarly fails at.162606. It also completed all
five operational checks successfully, including its six physical optimizer
updates. Both completed reports retain the compatibility failure. Final numbers
are in results.md and summary.json. No tolerance is relaxed.

Read-only analysis attributes92.69% of RT-only gradient-error energy to block0,
4.29% to the tied embedding and2.57% to block1. This localizes the effect without
proving whether forward perturbation, reconstructed nonlinear states or local
VJP arithmetic is responsible. Until resolved, isolated speedups are not a
backend replacement recommendation.

Localization first freeze29a2c4b produced one retained zero-update harness
failure before any model forward: its static layout was prepared outside the
forced Flash context, then correctly rejected when validated inside it. Commit
524fa89 prepares the layout inside that same context; the new attempt uses
`localize-block0-r2`. No model math, numerical budget or protocol changed.

The r2 diagnostic completed at524fa89 with eight localVJPs and zero optimizer
updates. All ten comparisons are finite/owned; one FP32 and four BF16 numerical
screens pass, while five other rows are descriptive only. Holding the actual
native incoming cotangent fixed, author/native mixed parameter-gradient L2 is
.003327; fullFP32 agreement is4.6523e-7. Native/author mixed versus nativeFP32
are .004743/.004964. Separate-self reconstruction does not improve aggregate
agreement (.003520 versus .003327), and legacy compiled token0 reconstruction
already equals its temporary value exactly. Native local forward matches the
captured forward bitwise; parameter-gradient norms match the earlier complete
native run to reduction precision. No conclusion about the end-to-end error's
cause is yet established.

The protocol's matched full-LM timings proceeded as exploratory measurements
after the successful own-execution and local checks; timing does not resolve
the retained compatibility misses or change the native default. Results and
explicit qualifications are in results.md.

The final diagnostic extension conflicted with the already-applied Flash-context
fix during cherry-pick. A shell sequence mistakenly proceeded to the launcher
after that conflict; Python rejected conflict markers at parse time, before any
imports, report/W&B setup or model work. The parse-error log is retained as
`integration-localize-launch-conflict.log`. The conflict was resolved by keeping
both the fixed context and the new version guard, and syntax checked. Final
freeze987bc46 precedes the real `localize-block0-r3` run. Zero optimizer updates
or numerical observations came from the parse failure.

The r3 extension completed at987bc46: two full-model backward captures, nine
local VJPs, zero optimizer updates. Inputs/positions are identical and model
weights unchanged. Both native and author local VJPs reproduce all four of
their own full-model block0 parameter gradients BITWISE, with their respective
incoming cotangents. Incoming-cotangent relativeL2 is.786611; fresh full-model
block0 parameter-gradient relativeL2 is.443495. This fresh capture is distinct
from the original integration's.437355. Shared-native-cotangent discrepancy
remains.003327; changing only the author's cotangent gives.443501. Thus the
dominant block0 discrepancy is carried by its incoming gradient in this
fixture. Its origin within the surrounding model and consequences for training
remain unresolved. No production math or screen was changed. Native remains
default; combined-case qualification remains open. See results.md for source,
W&B, exact comparisons and limitations.
