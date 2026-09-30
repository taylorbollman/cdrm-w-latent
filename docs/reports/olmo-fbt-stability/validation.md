# FBT-only stability validation

The new implementation keeps all 200 accepted asynchronous training sources
unchanged. The F-only startup uses the strict historical fusion128 loader, then
constructs a new F-only wrapper around the same imported backbone and fusion
objects. The NextLat predictor is absent before fresh Adam is constructed.
Native RT is disabled; CE uses the existing K4 pass weighting. No backbone,
fusion, attention, gradient or numerical precision equations were changed.

## CPU checks and independent review

The execution tests verify that the accepted save, prepare, update, boundary and
logging callbacks remain AST-identical. They also check actual optimizer
ownership, unchanged imported core tensors and parameter identities, preserved
RNG, retained tied readout, no predictor, and the authentic 128-update ordered
data/allocation/LR prefix under the new 192-update finite ceiling.

The probe tests compare streamed hidden states with the unchanged FBT forward,
exercise padding and actual document boundaries, verify beta-zero behavior,
confirm the causal settled-prefix index, and test preservation around the live
evaluation controller. Probe statistics are additive across positions and
ranks; they are not averages of rank-local ratios. Vocabulary projection is
position-chunked. No vocabulary-sized sequence tensor is retained.

Independent review also covered the separate saved-checkpoint exact-online
helper. It authenticates the F-only source lineage and model ownership, imports
weights without optimizer history, selects two single-document crops by metadata,
and compares finite passes against fresh-cache serial feedback on those same
tokens. Its per-position CE excludes the final input's nonexistent next target.
The omitted preceding context and reset RoPE positions are explicitly recorded;
these crops must not be presented as the packed T1024 development curve.

Independent audit tests currently pass **55 tests** in the CPU container. They
reject added RT/predictor/auxiliary terms, changed optimizer ownership, altered
native or fusion initialization, retained Adam history, changed data/order/LR
or keyed-noise seed, and corrupt pass statistics or state/RNG restoration.
They also require complete update parity when the probes are inserted. The
auditor imports neither PyTorch nor cloud clients and does not rewrite reports.

The report auditor independently reconstructs regional position and CE counts,
including the tail and the unsettled suffix, checks that quarter regions
partition the full row, recomputes RMS/relative-change/cosine/CE/entropy from
raw sufficient statistics, and checks source, schedule, checkpoint publication
and named-development-evaluation evidence. The first pass has no previous-pass
difference. Empty settled suffixes have undefined ratios, rather than false
zero-error observations.

The native F-only report is additionally compared against the retained NF
origin: every common model/buffer digest must match exactly, predictor tensors
must be absent, Adam must be fresh, and the complete first 128 logical updates
and token LR prefix must remain identical. This establishes the intended clean
ablation; it does not demand that the learned F and NF trajectories match.

## GPU acceptance and scope

GPU acceptance is pending. The intended bounded test compares the accepted
F-only tiny execution with the new probe-enabled execution, then restores a
committed new update-2 checkpoint in a fresh process and executes update 3.
Complete model/Adam/cursor/RNG/input/raw-gradient state must agree exactly. The
tiny probe schedule includes update 2 specifically to exercise a live captured
training graph before the subsequent optimizer update, and repeats scheduled
observations at the restored boundary.

The native run has not yet passed its report audit. This document will be
updated with actual evidence after execution. These checks address implementation
integrity and diagnostic accounting. Finite updates or decreasing state changes
do not establish useful refinement, global contraction, BF16 equivalence, or an
advantage over an ordinary model. Exact-online comparisons require their
separately bounded probe; the finite-pass curves do not imply them.
