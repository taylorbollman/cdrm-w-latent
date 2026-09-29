# Next bounded step toward usable BF16

2026-09-29. Recommendation after the [fixed-boundary results](results.md):
**compare FP32 and production BF16 at an already-adapted FBT checkpoint before
changing more arithmetic.** This is proposed, not launched. The present
diagnostic is complete; no additional GPU job is queued.

## Why this is the next useful test

The current 60.87% NF backbone gradient difference occurs with newly initialized,
full-strength feedback. At a common input and incoming gradient, fusion itself
differs by only about 0.39%; the full ordinary stack differs by 7.68–8.19%.
Changing only the stack's incoming state under FP32 arithmetic produces
11.43–29.14% gradient differences. This motivates checking the operating state,
as well as the arithmetic, before another precision promotion. It does not prove
that initialization caused the discrepancy or that training will resolve it.

We already have a useful adapted checkpoint: O5c mixed update 512. O5c trained
fusion only, but inherited an **already-adapted O5b backbone**. Both backbone and
fusion therefore differ from today's cold diagnostic. This is a practical
trained-state comparison, not a causal fusion-only ablation. It requires no new
training and should be a short diagnostic once import compatibility is checked.

## Checkpoint authority and import work

- Local file: `.runtime/olmo1b-step60000/o5c-pilot-01/mixed/update-000512.pt`.
- Size: 4,807,843,871 bytes; full local file SHA-256 verified on 2026-09-29:
  `7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
- Retained object:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/mixed/update-000512.pt`,
  generation `1790059437165208` (prior retention authority; no new cloud download here).
- Completed report: `.runtime/olmo1b-step60000/o5c-pilot-01/mixed/report.json`,
  SHA-256 `020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.

The old `olmo_o5d_common.endpoint_metadata()` correctly requires its historical
runtime source hashes. Six core files have changed since then. **Do not weaken
that validator or label an import into today's runtime an exact old-run
reproduction.** Add a separate, explicit weights-only import for this diagnostic.
Validate the immutable report/checkpoint, architecture, tensor names/shapes/dtypes,
tied embedding/readout ownership and complete state hashes. Preserve all saved
buffers, including fusion's output scale; do not recompute it from the adapted
embedding matrix. Record the historical-to-current mapping. Keep
`torch.load(weights_only=True)` and reject unexplained missing/extra tensors.
Test these import contracts using tiny CPU fixtures before any GPU execution.
File identity has been checked; actual model-schema compatibility has not yet
been exercised.

The old checkpoint has no NextLat predictor. Initialize and pin only that
intentionally absent module using the existing NF seed; its auxiliary cotangents
remain zero. Do not restore an optimizer, scheduler, data cursor or RNG from the
old training run. Do not freeze parameters for the diagnostic VJPs.

## Matched diagnostic

Use the unchanged current NF fixture: two B2/T16 records, K4, beta 1, jitter
0.02, the same keyed noise, CE weights/denominators and runtime settings. Load
identical adapted backbone/fusion state into the FP32 and BF16 cases. Retain
state/source/input/noise pins and pass-level hidden-state, CE and gradient
geometry. This is two aggregate cases/four physical backwards, no updates,
with the existing 900-second GPU-stage limit, W&B and GCS retention.

O5c was trained at **K2, without RT, NextLat or jitter**. Keeping today's K4/jitter
recipe tests transfer of its weights into the current numerical fixture; it
does not reproduce its training recipe. Its earlier O5d K4 evaluations establish
that this checkpoint has been used at K4, not FP32/BF16 gradient compatibility.
Compare precisions *within* the adapted state; do not expect the cold model's
loss or gradient values to match. Retain the cold result as context.

## Decision after that check

- If agreement is substantially better across forward and gradient measures,
  prioritize a controlled feedback-startup/transition experiment. It would test
  a proposed warmup, not assume that adaptation makes BF16 safe. The adapted
  backbone confound remains explicit.
- If disagreement persists, inspect selected ordinary layer boundaries around
  the first feedback pass, using the established common-input/common-cotangent
  method. Localize before selecting one precision change; do not expand into
  an automatic all-module sweep.

Neither outcome alone qualifies production training. Any candidate remedy must
retain the intended objective and be checked with native RT, both NextLat losses,
packed T1024 and representative update/restart behavior. Measure memory and
throughput only once there is a credible numerical policy to profile. No error
budget is relaxed, no Q/K normalization is added, and no training is launched
as part of this completed milestone.
