# Next step: learn a usable fusion startup

2026-09-29. Recommend one bounded **full-FP32 fusion-only warmup from the original
OLMo checkpoint**, followed by matched FP32/BF16 checks. This is proposed after
the completed crossed-state diagnostic; no training is queued or launched.

Adapted fusion on the original backbone lowers backbone gradient relative error
from60.87% to8.73%, cosine rises from0.7973 to0.99653, and the absolute discrepancy
falls from332.72 to8.85. The reverse swap retains large discrepancies. This is
enough evidence to prioritize learning the fusion startup over another dtype or
feedback-strength sweep. It is not numerical clearance: fusion error remains
13.25%, one incoming-cotangent error16.57%, and fully adapted weights agree more
closely. The old adapted/adapted12.44% intermediate hidden discrepancy is still
unlocalized. Swapping weights tests transfer, not training reachability.

## One short learning test

Start from original OLMo step60000, the deterministic fresh fusion and fresh
predictor used by the cold diagnostic. Do not transplant the historical adapted
fusion. Freeze the complete native backbone, tied embedding/readout, predictor
and fusion output-scale buffer; train only the two fusion matrices. Keep K4,
beta1, jitter0.02 and the same CE pass weighting. No temporal RT and no auxiliary
NextLat gradients yet. Holding feedback strength fixed makes this a test of
fusion learning, without simultaneously changing recurrence strength.

Use FP32/math throughout warmup, with the existing deterministic controls,
checkpointing and chunked CE. Freezing weights must preserve gradients through
the later backbone passes into fusion: do not wrap those passes in `no_grad`.
Verify frozen hashes, real fusion gradients/updates, keyed noise and exact
checkpoint continuation in a short preflight. Preserve all completed diagnostic
sources; use a new runner and protocol.

Suggested starting budget: **128 updates, 8,192 supervised CE targets/update**
(1,048,576 targets), with checkpoints at0,32,128 and every10minutes if sooner.
Use the existing pinned prepared training documents, disjoint from precision
probe records, under the already-tested isolated document policy. This is a
numerical startup test, not a production mixture or a quality comparison.
Set a modest context and physical batch after one bounded memory/step-time
preflight, record actual input/target counts and the data cursor, and freeze
these before training. Do not repeatedly train on the two numerical records.
Count supervised targets explicitly: 8,192 is not physical batch times length.
Reuse count-based accumulation and predeclared final masking, with the existing
within-row shift and global normalization; never invent a cross-chunk target
to reach a round budget. If the chosen runner uses a nearby budget instead,
freeze and report the actual target count before launch.

Use the existing fusion AdamW convention as the initial optimizer proposal:
LR1e-4, betas(0.9,0.95), epsilon1e-8, matrix weight decay0.1, clip1, with a short
16-update linear LR warmup. This LR schedule is separate from feedback beta,
which remains1. Inspect raw norms before clipping. Finalize this bounded recipe
in the next protocol before launch; do not tune it after observing its outcome.
Log W&B and retain complete optimizer/RNG/cursor state and checkpoints in GCS.
Profile first and tell the user if the estimate reaches multiple hours; avoid
any interval with more than20–30minutes of unretained progress.

## Assess saved states, then make one decision

At0,32,128, evaluate the original short fixture and a small predeclared fresh
training-disjoint fixture with matched FP32/BF16 pairs. Use the current full
trainability diagnostic contract when measuring backbone gradient sensitivity;
frozen-training backbone gradients would trivially be zero. Record this
transition explicitly and preserve the frozen optimizer/checkpoint state.
Restore `requires_grad`, module modes and empty `.grad` fields afterward; the
training optimizer must continue to own only fusion, and probes make no updates.
Retain relative and absolute group errors, direction/norm ratios, pass states,
incoming cotangents and supported-position summaries. Reuse existing state,
fixture, objective and source controls. No new kernel comparison grid is needed.

- If warmup produces a consistent material improvement on both fixtures without
  new forward-state pathology, test one short BF16 continuation from its retained
  endpoint against FP32. Decide whether that startup is practical before adding
  more model components. Improved errors alone do not establish update stability.
- If the single bounded warmup does not help, stop it rather than extending or
  sweeping indefinitely. The next alternative is controlled feedback-strength
  startup, beginning closer to the ordinary model. Failure at this small budget
  does not prove that fusion-only learning can never work.

No new precision acceptance threshold is inferred from the current8.73% result.
Native RT/NFR, both real NextLat losses, packed T1024, representative updates and
distributed graph/restart checks still need staged confirmation before the
production BF16 campaign. The goal remains reliable, efficient training rather
than exact FP32 equality or model-quality wins in this diagnostic phase.
