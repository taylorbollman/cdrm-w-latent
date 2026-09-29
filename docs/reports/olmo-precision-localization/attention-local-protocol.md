# Fixed-input ordinary-attention check

Recorded before execution, 2026-09-29. This is a bounded local check motivated
by the completed crossed-backend experiment. The initial and crossed-backend
protocols and source bytes remain unchanged.

## Evidence and question

For the original CE-only NFR T16 fixture, BF16 Flash/eager native RT and BF16
Flash/Triton native RT produce identical forward fingerprints and complete
parameter gradients. Both differ from BF16 math/eager by 79.8459% gradient L2.
This identifies ordinary attention backend selection as the changed operation
responsible for that paired result. It does not establish a Flash bug, exclude
amplification by RT or FBT, or explain the BF16 math/eager versus FP32 gap.

Hold actual ordinary-attention Q/K/V and its incoming gradient fixed to ask
whether a large local attention discrepancy remains, or whether the complete
model is amplifying much smaller changes. Do not infer global harmlessness
from a small local result.

## Capture and controls

Reproduce the production BF16 Flash/Triton CE anchor once: two physical input
records, same source checkpoint, recipe, masks, global loss denominators,
feedback noise, runtime flags and deterministic controls as the bridge/cross.
Verify exact prior scalar metrics, pass-output fingerprints and gradient-group
summaries; original full parameter gradients are not retained. No optimizer,
DDP, CUDA graphs or architecture changes.

Capture post-RoPE Q/K/V, ordinary SDPA output and its total incoming cotangent
at ordinary layers 1 and 14, passes 0 and 3, for each record: eight sites. Use
the existing full-model computation; keep activation checkpointing enabled.
Observation is restricted to the original outer forward, excluding checkpoint
replays. Hooks may copy detached values but must return no replacement output
or gradient. Interception must be restored on both success and failure and
must not replace global torch functional state. Record actual shapes, dtypes,
attention mask/causal settings and capture coverage. Any coverage or anchor
failure stops this phase.

## Local computation

At each site, run three local forward/vector-Jacobian products, always using
identical captured values and identical captured incoming cotangent:

1. FP32 math SDPA: exact FP32 promotion of the captured BF16 Q/K/V and cotangent.
2. BF16 math SDPA.
3. BF16 Flash SDPA.

This is 24 local VJPs plus the two model backwards needed for capture. Local
leaves are detached clones; there is no gradient propagation into the model.
Autocast is disabled for explicit-dtype local operations, TF32 remains off,
and deterministic settings precede CUDA initialization. Preserve the actual
mask, causal flag, scale and dropout settings; do not substitute random
cotangents or normalize them. BF16 math reduction settings remain recorded.

Report output and dQ/dK/dV norms, maximum absolute differences, relative L2,
cosines, and finite/zero status. Include bounded FP32 logit-range and
maximum-attention-weight summaries over valid query rows as scale context. Compare BF16 math and Flash both to promoted
FP32 and to each other. Verify local Flash output equals the captured output;
this protects against reconstructing a different attention invocation. These
are descriptive measurements, with no newly chosen numerical acceptance
threshold. Captured tensors are small and should be retained with byte hashes
and source provenance for a later reproducible local test.

## Boundaries

W&B tracking and a 900-second process bound apply. Retain reports, source
snapshots and the bounded captured fixture in GCS. Source/parameter/input/RNG
guards must remain intact. Prior BF16 qualifications remain open. This phase
does not establish training quality, packed T1024 numerical agreement, or a
need to change Q/K normalization. Review the result before choosing a precision
promotion, a full-model common-cotangent replay or any wider sweep.
