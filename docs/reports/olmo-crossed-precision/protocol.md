# Crossed-state NF precision diagnostic

2026-09-29. User approved the two crossed-state comparisons after PR43. Base
`318948c`, branch `feat/olmo-crossed-precision`. Freeze new code/tests and this
protocol before GPU execution; keep all prior runtime sources unchanged.

## Four cases, no training

Construct current cold NF, verify its state/configuration against the retained
matrix, and retain independent CPU copies of native and complete fusion state.
Use the existing validated importer to load O5c mixed update512, verify against
the completed adapted report, and retain equivalent adapted component copies.
Keep the current fresh predictor, gradient participation and runtime settings.

Assemble two combinations using strict in-place copies, preserving parameter
identity and tying:

1. Original OLMo step60000 backbone with adapted O5c fusion.
2. Adapted O5c backbone with original deterministic NF fusion.

Fusion means both matrices **and output_scale**, not just trainable parameters.
Never recalibrate its buffer from the selected embedding. Verify all selected
state bytes and unchanged predictor/configuration/modes/RNG before execution.
Independent snapshots must not alias a model whose weights are later overwritten.
No optimizer, scheduler, data cursor or historical trainability is restored.

For each combination run FP32/math then production BF16/Flash: **four aggregate
CE cases/eight physical model backwards** total. No additional diagonal cases,
local VJPs, optimizer updates or training are part of this milestone.

## Fixed numerical contract

Reuse the two isolated B2/T16 records, valid lengths (16,5)/(6,2), four documents,
29 inputs, global CE25/latent25/KL21 targets and exact keyed noise. Recipe remains
K4, beta1, jitter0.02, CE pass weights (1/2,1/6,1/6,1/6); current predictor remains
present with zero latent/KL cotangents. No temporal RT executes. The campaign's
T1024 recipe is unchanged; this is its short numerical fixture.

FP32 masters, raw unclipped gradients, ordinary activation checkpointing,
pointwise eager and RoPE native; TF32 off, highest FP32 precision, autocast cache
off. Configure determinism before CUDA and restore production flags before each
path. Record runtime/reduction flags, sources, inputs/noise and complete state
provenance. Retain per-case CE, full/group gradient geometry and per-pass valid
hidden-state/total incoming-cotangent summaries. Compare precisions **within each
hybrid**. Cold/cold and adapted/adapted reports provide context, not a requirement
that the hybrids' full losses or gradients match either diagonal.

Because fusion is unused in pass0, require each hybrid's first-pass hidden states
and token embeddings to match the saved diagonal with the same backbone and
precision exactly. Later states and incoming cotangents need not match.

Reference matrix SHA256:
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`.
Adapted report SHA256:
`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.
O5c checkpoint SHA256:
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
Use the frozen importer and its explicit six-source mapping without weakening
historical guards. Rehash checkpoint bytes once during its real import; no
independent duplicate large checkpoint read is needed.

## Position-level observations

For each valid position and pass, record hidden-state differences, actual and
reference full incoming-cotangent norms, document/row/position identifiers and
the direct CE prediction-position mask. CE mask index t+1 supervises hidden
position t: use the existing loss mask and pad False on the right. Structural
feedback eligibility is recorded in both source and destination orientation;
the last pass has no outgoing feedback and pass0 has no incoming feedback.

Report aggregate all-valid, union-supported, both-zero-support, direct-CE and
no-direct-CE geometry. Union support is valid positions with any exactly nonzero
incoming-cotangent element in **either** precision, never precision-dependent
selection on only one side. Preserve each side's support counts and explicitly
mark empty groups. Zero observed cotangent is not proof of harmlessness or of
causal independence. A position without a direct CE target can still affect
later feedback losses. These masks do not change the backward computation.

Retain the position summaries with the report. They can diagnose a similar
discrepancy in these hybrids; they cannot retrospectively locate the previously
observed adapted/adapted pass1 spike, whose individual tensors were not retained.
No full parameter-gradient vectors or new model checkpoint are exported.

## Interpretation, persistence and stopping point

Require finite losses/states/gradients/cotangents, correct counts and participation,
zero predictor gradient, unchanged assembled state/snapshots/sources/fixture/RNG,
restored flags and exact first-pass controls. These are operational checks;
do not adopt or relax a BF16 acceptance budget from the measured result.

If adapted fusion transfers good agreement to the cold backbone, this supports
trying fusion-only warmup, not proof that such training can reach those weights.
Otherwise prioritize a controlled feedback-strength startup; a failed hybrid
does not prove fusion-only learning impossible because coadaptation can break.
The reverse hybrid tests whether adapted backbone alone suffices on this fixture.
Stop after the four cases and assess. No automatic training or extra dtype sweep.
RT/NFR, actual auxiliary gradients, packed T1024 and update/restart qualification
remain separate.

Root alone runs GPU0 inside the project container, under a900-second external
limit. GPU1 stays unused. Use online W&B, atomic per-case reports and immutable
GCS retention. Save/push every20–30minutes; preserve failed attempts. Each stage
is shorter than the interruption-recovery interval and contains no training.
