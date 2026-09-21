# OLMo O3: language-model objectives and training platform

Recorded 2026-09-21 before GPU validation. This milestone adds language-model
NextLat and verifies the optimizer/checkpoint path. It does not start O4's
comparative learning experiment or implement FBT.

## Fixed model and objective

Reuse original OLMo-1B step60000 (~252B), revision
`81b71efbce6f4dada57c94860301af4298bcd351`, native checkpoint SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
The native backbone and O2 tiled implementation are unchanged. Actual RT
checks select layer 0 only. The 16-layer/D2048 backbone retains its native
tied readout, LayerNorm, RoPE and full MHA without Q/K normalization.

Pin NextLat source at `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`, using the
released **1B language-model, horizon-one** configuration rather than A5's
recipe. Projection factor 1.6 gives predictor hidden width 6528 at D2048.
Predictor: concatenate `[embedding(t+1); post-finalnorm hidden(t)]`, learned
RMSNorm (epsilon 1e-5), three bias-free linear layers with two GELUs, and a
residual addition of hidden(t). Match source initialization and record resolved
defaults. Its predictions are training auxiliaries, never free-running states.

The loss is next-token CE + latent SmoothL1(beta=1) + teacher-to-predictor KL,
with separate configurable coefficients (initial diagnostic values 1/1/1;
auxiliary predicted-token CE is absent, as in the pinned LM configuration).
The latent target is stopped `hidden(t+1)`. KL compares stopped readout
distributions from that target and predicted hidden(t+1), concerning token
t+2. The auxiliary readout weight is detached; ordinary CE and conditioning
embedding lookup still train the tied matrix. Do not apply finalnorm twice.

Pair masks cover valid, same-document t/t+1; KL requires t/t+1/t+2. CE response
masking, latent target-position masking and KL token-position masking are
independent explicit fields. Mean each objective by its actual valid positions;
latent coordinates are averaged too. Gradient accumulation uses global counts
across microbatches independently for all three terms. All-empty or disabled
terms are graph-safe zeros, and a disabled NextLat path bypasses its predictor.

For O3, **one document per batch row**, with padding. Reject packed multiple
documents in one row: masking losses alone would not prevent attention leakage.
Fixtures use the pinned native tokenizer without automatic BOS/EOS, then append
one explicit EOS per document. No comparative dataset/split is selected yet.

Vocabulary objectives operate in bounded position chunks with recomputation;
test values/gradients against an independent dense objective. Avoid saving all
vocabulary logits across chunks. Source/teacher detachments remain identical.

## Correctness gates

CPU tests independently check source predictor behavior, LM shift and teacher
direction, pair/triple/padding/response masks, normalization and gradient
ownership, disabled/empty terms, chunked versus dense objectives, document-row
isolation, ordinary/tiled RT integration and weight tying. Unequal valid-count
microbatches must reproduce the correctly reduced batch objective and update.

Checkpoint tests include native plus predictor parameters, AdamW moments,
scheduler, counters, data cursor, resolved config, source identity and RNG.
Save atomically at an optimizer boundary, reject incompatible resumes, and
compare interrupted/resumed execution with uninterrupted execution on tiny
models, including restored RNG and zero-LR operation.

Actual-checkpoint H100 checks use short literal text/code fixtures at B1/T16
and a padded two-row fixture. Compare chunked objective values and all active
parameter gradients with a dense reference for ordinary/RT, NextLat off/on.
FP32/TF32-off is the semantic reference. Loss tolerance is absolute+relative
1e-5; each gradient tensor must meet relative L2<=1e-4 (or whole error norm
<=2e-6) AND max error<=2e-6+1e-4*reference maximum. Preserve stricter coordinate
diagnostics. Missing gradients are allowed only for intentionally disabled
predictor parameters, with ownership explicitly recorded.

Add a bounded BF16 RT+NextLat finite/backward observation against the same
FP32 state, with O2's mixed and FP32 attention policies. Do not reopen O2's
gradient campaign or interpret finite gradients as long-run training clearance.

Exercise a few AdamW updates on a **disposable diagnostic state**, using
LR 1e-5, betas .9/.95, epsilon 1e-8, weight decay .1 and clipping norm 1.
Save after two updates and verify the next uninterrupted/resumed update with
nonempty moments, scheduler/RNG/counters and data cursor. Record all update
counts and hashes; this is recovery validation, not a candidate adapted model.
Native weights and predictor state used for precision comparisons remain at
the original initialization. Large diagnostic recovery files are disposable;
retain source identity, fixtures, checkpoint hash and recovery evidence.

## Operational profiling

One H100 80GB is available. Two-GPU validation is explicitly untested. Compile,
CUDA graphs and distributed training are outside this milestone.

Measure complete forward/objective/backward/clipping/AdamW steps, with moments
initialized, for ordinary/RT with NextLat off/on at B1/T128, then RT+NextLat
at B1/B4/B8 and T512 if comfortably below 60 GiB. Use three warmups and three
timed iterations per point. Profiling LR is zero: execute AdamW and update its
moments while keeping weights fixed. Report wall/device time, valid input and
supervised token counts, objective counts, peak allocated/reserved memory,
parameter/state residency and validation-scratch scope. No maximum-batch search.
The O2 backend's eager loops and quadratic backward scratch remain in scope;
do not claim an optimized kernel.

Each GPU run logs online to `taylorbollman/pretrained-fbt-rt-nextlat`, uses a new
output directory, and records code/source hashes, native manifest, exact token
fixtures, model/optimizer configuration and runtime. Retain source/evidence
under `gs://fast-chunks`, reusing the O1 checkpoint object. Failed attempts are
preserved. Update the usage guide, results and compaction handoff, then stop at
O3 for review before any O4 comparative learning run.
