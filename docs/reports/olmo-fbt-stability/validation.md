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

GPU acceptance completed successfully on the two H100s. The accepted F-only
tiny execution and the new probe-enabled execution both completed three updates.
The independent insertion audit passed **8,531 checks**, including exact
model/Adam/cursor/RNG/input/raw-gradient evidence at every update and equal
ordinary development evaluation. No training report was relabeled to obtain
this agreement.

The new update-2 checkpoint was restored from its committed GCS publication into
a fresh directory, then update 3 was executed in a new process. The independent
restart audit passed **10,150 checks**. Its restored origin, next update and
final state match uninterrupted execution exactly. The repeated development
evaluation and stability observations also match. The tiny probe schedule
includes update 2 specifically to exercise a live captured training graph
before the subsequent optimizer update, and observes update 3 as well.

All three executions finished their finite plans and synchronized W&B:
[accepted reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/b79u20pb),
[with probes](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rmjsgmmk),
[fresh-process restart](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/3fxf5pjh).
Independent audit reports are in
`.runtime/olmo-fbt-stability/tiny-insertion-audit-01/report.json` and
`.runtime/olmo-fbt-stability/tiny-restart-audit-01/report.json`. The audits pin
their exact input-report bytes, check the new source snapshots and verify
checkpoint publication metadata. This is a bounded tiny FP32 acceptance check;
it is not native BF16 trajectory equivalence.

The native run has not yet passed its report audit. This document will be
updated with actual evidence after execution. These checks address implementation
integrity and diagnostic accounting. Finite updates or decreasing state changes
do not establish useful refinement, global contraction, BF16 equivalence, or an
advantage over an ordinary model. Exact-online comparisons require their
separately bounded probe; the finite-pass curves do not imply them.
