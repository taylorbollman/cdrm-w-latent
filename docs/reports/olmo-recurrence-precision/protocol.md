# Separating recurrence and precision sensitivity

2026-09-29. User authorized the next milestone after PR40. Branch
`feat/olmo-recurrence-precision`, starting at `4eff982`. Freeze this protocol
and the helper's source inventory before GPU execution. Later conditional
experiments require separate written scope before launch; do not rewrite
completed probes or their protocols.

## Question and existing evidence

The initial K4 FBT+RT model has a large BF16/full-FP32 CE-gradient difference.
In the prior crossed-backend test, ordinary Flash attention with eager RT
exactly matched Flash with Triton RT, while both differed from ordinary math
attention/eager RT by 79.85% in full-gradient relative L2. At eight actual
fixed-input ordinary-attention sites, Flash versus FP32 output errors were
0.160–0.186% and Q/K/V gradient errors were 0.174–1.501%.

These results motivate separating ordinary computation, temporal RT, repeated
FBT passes and their combination. They do not establish that small local
rounding differences are harmless, or that a particular kernel is incorrect.
Existing loss-layout and full-model precision qualifications remain open.

## Eight aggregate CE cases

| Existing arm | Backbone computation | FBT passes | RT layers | Diagnostic objective |
| --- | --- | ---: | --- | --- |
| N | Ordinary | 1 | None | CE only |
| NR | Temporal RT | 1 | 0 and 15 | CE only |
| NF | Feedback without temporal RT | 4 | None | Weighted CE only |
| NFR | Feedback with temporal RT | 4 | 0 and 15 on every pass | Weighted CE only |

For each arm, compare full FP32 with forced math SDPA/eager RT to the existing
production BF16-mixed path with forced Flash SDPA/Triton RT. Preserve FP32
parameter masters/raw gradients and the original mixed RT-attention policy.
Use the existing configuration helper to restore production flags between
conditions; do not accidentally retain FP32 attention in the BF16 case.

The N prefix keeps NextLat enabled. Its latent/KL branches execute with zero
cotangents, as in the previous CE diagnostic. This retains positive counts,
predictor participation and the existing helper contract. These are not new
tests of the usefulness of NextLat or models trained without its loss.

Each aggregate case contains two original physical B2/T16 records, so this
matrix has **eight aggregate gradient cases and sixteen model-backward calls**.
No optimizer, parameter update, DDP, CUDA graphs or training trajectory.

## Fixed state and matching controls

Use OLMo-1B step60000 (~252B tokens), revision
`81b71efbce6f4dada57c94860301af4298bcd351`, native weight SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
The original 16-layer/width2048 backbone, tied embeddings, RoPE, normalization
and parameter values are unchanged. Construct each arm using the existing
helper and seeds; compare hashes of shared backbone, predictor and fusion
weights, including the fusion output-scale buffer. Fusion is frozen/dormant
without FBT and trainable with FBT; report this difference explicitly.

Use the original isolated-document update-zero fixture: the same tokens,
validity/document/target masks and right-padded causal semantics in every arm.
Record actual counts (29 valid inputs, 25 CE targets, 25 latent pairs, 21 KL
triples). T16 is the diagnostic input override; the recipe still records the
campaign context length of1024. No packed-data/T1024 equivalence is claimed.

NF and NFR must receive byte-identical keyed feedback-noise tensors, jitter
amplitude0.02, beta1 and the same fusion initialization. No jitter executes in
N/NR. Record CE pass weights explicitly: `(1)` for N/NR and
`(1/2,1/6,1/6,1/6)` for NF/NFR. Preserve global denominators and stop-gradient
semantics. For each precision, compare first-pass forward fingerprints for
N↔NF and NR↔NFR: feedback has not yet acted, so those corresponding states
should match. Incoming cotangents may differ because later losses and feedback
paths are present; do not require those to match.

The NFR pair must reproduce the completed bridge's CE endpoints, using
`.runtime/olmo-precision-localization/bridge-01/report.json`, SHA256
`39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b`.
Check checkpoint/recipe/input/runtime/source pins, scalar metrics, forward
fingerprints, gradient-group summaries and the recomputed within-pair gradient
geometry. The original full gradient vectors were not retained; do not claim
an independent bytewise comparison against those unavailable vectors.

## Measurements and execution controls

Configure deterministic algorithms/CUBLAS before CUDA; disable TF32 and use
highest FP32 matmul precision. BF16 autocast keeps caching disabled. Record
actual runtime flags and reduction settings. All GPU work runs inside the
project container, on device0 only; two virtual records are not DDP ranks.

Measure full raw-gradient relative L2, cosine, norm ratio, absolute error,
finite/zero status and component groups. Report the shared backbone group
separately, since fusion participation differs across arms. Capture valid-token
hidden states and their total incoming cotangents by pass using the existing
observer, outside checkpointed block internals. Report actual dtypes and skip
dummy positions explicitly. Retain CPU gradient references only within each
pair, then release them; summaries cannot reconstruct gradient vectors later.

Compare errors **within each arm**. Different pass weights and active fusion
mean cross-arm gradient norms are not direct causal contributions, and error
norms must not be added. This matrix changes both precision and its associated
backends within each pair; it localizes a sensitive model configuration rather
than identifying one arithmetic operation by itself.

Pass/fail gates cover exact controls, finite execution, expected participation,
source/parameter/buffer/input/RNG integrity and prior NFR reproduction. Error
budgets from earlier diagnostics are not relaxed; these new cross-precision
measurements are descriptive, not numerical clearance.

## Conditional next step and persistence

If the matrix identifies a clear sensitive configuration, choose at most one
narrow precision boundary supported by the result (for example feedback fusion
or RT attention). Before running it, record the exact selected configuration,
reference/candidate cases and additional backward count in an addendum. Retain
unsuccessful corrections. If attribution is ambiguous, stop this milestone
with the measured limitation and a bounded next recommendation rather than
changing multiple precision policies together.

An actual correction needs a subsequent bounded packed-data T1024 confirmation
before adoption; graph/restart/performance checks are relevant if the runtime
policy changes. No Q/K-normalization transition, quality training, production
data preparation or H200 qualification is part of this matrix.

Each GPU stage has an external900s timeout, atomic per-case progress, online
W&B logging in `taylorbollman/pretrained-fbt-rt-nextlat`, and source snapshots.
Save/push progress and retain evidence in `gs://fast-chunks` at least every
20–30minutes. Use new stage directories and cloud prefixes; preserve failed
attempts. These probes create no trained checkpoint.
