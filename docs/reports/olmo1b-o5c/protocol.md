# O5c: fusion-only domain adaptation

Authorized 2026-09-22 after the completed O5b assessment. This bounded paired
experiment tests whether general-text training repairs the learned feedback
input while the native backbone stays fixed. It is not an interaction sweep.

## Starting point and fixed model

Both arms load the retained O5b FBT update 2634 endpoint, SHA256
`99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`.
All native parameters, including the tied embedding/readout, are frozen. Only
8,388,608 parameters in the two fusion matrices are trainable; the native
embedding-RMS output-scale buffer remains fixed. Verify every frozen tensor
hash before/after training. The existing OLMo/FBT model math is unchanged.

FBT beta 1, K2, gamma1; RT and NextLat off. The frozen pass 0 contributes its
ordinary CE reporting term but has no parameter gradient; pass 1 gradients
must propagate through the frozen backbone to fusion. No sampled prefixes,
hidden jitter, compilation, CUDA graphs, TF32 or distributed execution.
FP32 master parameters and AdamW moments, BF16 mixed compute, native SDPA.

## Data and matching

Use fresh CodeSearchNet Python windows after the 84,288 windows consumed by
O5b, preserving the pinned prepared source/order. Use the corresponding pinned
WikiText2 raw official TRAIN split for general text. Exclude training documents
matching any development/test document by exact reconstructed-text and token hashes. This is not semantic or near-duplicate deduplication.
Reserved tests are used only for contamination checks, never quality evaluation.

Each arm receives 512 optimizer updates with 8,192 supervised CE targets/update:
4,194,304 CE targets total. Code-only uses all code; mixed uses 4,096 code and
4,096 general-text targets every update. The mixed arm's code stream is a
prefix of the code-only stream. No cycling. Deterministic contiguous segments
carry one context token, without inventing EOS or cross-document targets.
Code is cut into 4,096-target chunks in both arms (two per code-only update,
one per mixed update), so shared code targets have identical context boundaries.
Each segment resets model/cache context and has length at most 512. The number
of segments/examples per update varies; physical chunks contain at most 16.
This matches CE exposure and updates, not exactly valid input-token exposure:
each segment contributes one extra input position. Report both and per-domain
counts; do not label this an unchanged B32 experiment. Code/general objectives
are balanced by supervised targets, not padded rows or characters.

The roughly 4.2M budget replaces the provisional 5M to stay within existing
unused code data. The smaller code exposure in the mixed arm is an explicit
tradeoff. WikiText2 is one small general-text domain; improved WikiText dev
quality alone cannot establish broad general-language transfer. Original OLMo
pretraining overlap is unknown. This is one-seed development evidence.

## Optimization, validation and recovery

Reset fusion AdamW in both arms: LR 1e-4, linear 50-update warmup, betas (0.9, 0.95),
epsilon 1e-8, matrix weight decay 0.1, global gradient clip 1. This is a new optimizer
lineage, not continuation of O5b moments. Seed 20260922. Code-only runs first,
then mixed. Profile zero-LR full steps and a bounded nonzero fusion step on the
actual H100; confirm frozen-state integrity and restore disposable changes.
Reuse CPU objective/gradient tests; add focused freeze/ownership/recovery/data
checks. Do not reopen the broad historical precision study.

Evaluate both passes on the unchanged 128-window code and retention development
sets initially, at 50 updates, then every 64 updates and at completion. Evaluate
512 windows with original-document records before training and at both final
endpoints. Report pass 1 NLL, token accuracy and perplexity; pass 0 is the fixed
backbone control. Paired bootstrap intervals resample original documents and
do not estimate training-seed variation. No reserved-test scores or online
long-context/generation claims.

Stop on nonfinite losses/gradients/state or frozen-tensor mutation. Pause if
feedback NLL exceeds its initial value by 1.5 nats in either domain for two
consecutive scheduled evaluations; this coarse gate is not a quality target.
Save a complete model/fusion-optimizer/scheduler/RNG/data-cursor checkpoint
every 128 updates, every 10 minutes if sooner, and at requested stops/final.
Retain to a new GCS prefix with verified generation/size/MD5/SHA metadata.
Keep the latest complete local checkpoint until its successor is retained.
W&B records every update, domain exposure and both development curves.

## Interpretation

A substantial retention recovery in the mixed arm with unchanged backbone and
without a large code penalty supports domain adaptation of the new input path
as one contributor. The code-only arm controls continuing recovery and update
budget. Failure to improve weakens that explanation at this budget only.
Do not attribute the comparison solely to adding general data: fixed total
CE exposure necessarily reduces code exposure in the mixed arm. No longer
extension or new RT/NextLat arm is automatically queued after this pair.
