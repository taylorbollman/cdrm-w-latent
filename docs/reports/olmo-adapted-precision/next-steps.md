# Next step: identify a practical feedback startup

2026-09-29. The completed adapted-state check materially improves CE gradient
agreement: backbone relative L2 **60.87% → 0.91%**, fusion **65.21% → 1.41%**,
with smaller absolute errors too. The current architecture and production BF16
path can therefore have much better agreement at a different parameter state.
This supports investigating startup before making more kernel/precision changes.
It does not show how to reach that state safely from the original checkpoint.

## One bounded crossed-state diagnostic

The existing cold/cold and adapted/adapted states differ in both backbone and
fusion. Add only the two missing combinations:

| Case | Backbone | Complete fusion state |
| --- | --- | --- |
| Cold backbone / adapted fusion | Original OLMo step60000 | Saved O5c mixed update512 |
| Adapted backbone / cold fusion | Saved O5c mixed update512 | Original deterministic NF initialization |

For each combination, run the same FP32/BF16 pair: **four aggregate cases/eight
physical backwards**, no updates. The previous diagonal cases remain retained
context; no need to rerun them solely to fill a table. Keep K4, beta1, jitter0.02,
the same two B2/T16 records/noise/masks/objective/denominators, current trainability
and fresh zero-cotangent predictor. No RT executes. Swap the **entire** fusion
state, including output_scale, without recalibrating it from either embedding
matrix. Validate each assembled state's provenance and all unchanged fields.

Compare FP32 against BF16 *within each hybrid*. Treat the result as a test of
whether a component's adapted state transfers sufficiently to the other
backbone/fusion state. It is not an additive causal decomposition, a claim about
how training reached the checkpoint, or proof that a warmup can learn those
weights from the cold state. A failed hybrid can reflect broken coadaptation.

## Observe position-level effects in those same runs

Record per-valid-position hidden-state differences and full incoming-cotangent
norms, with direct CE prediction-position masks and feedback eligibility. Keep
these concepts distinct: a position without a direct CE target can still affect
a later loss through feedback. Report both all-valid and common supported-position
summaries, using the union of nonzero incoming-cotangent support across precisions
so positions cannot disappear from only one side's comparison.

This adds observations to the four planned cases, not another model run. The
current adapted/adapted aggregate report has a **12.44% pass-1 hidden difference**
in record0, worse than the cold counterpart, even though later states and CE
gradients agree much better. Its per-position tensors were not retained. The new
hybrid observations can diagnose similar effects if they recur; they cannot
retrospectively locate or clear that particular adapted/adapted discrepancy.
Do not infer harmlessness from its small cotangent error or assume it occurred
only at tokens without a target.

## How the result selects the next intervention

- If adapted fusion also gives good agreement on the cold backbone, prioritize
  a bounded **fusion-only warmup from the original checkpoint**, followed by
  matched numerical checks. Transfer would support this route, not prove its
  reachability or set its training duration.
- If it does not, prioritize a **controlled feedback-strength startup** before
  another dtype sweep. Start near the ordinary model and assess a gradual
  transition; the hybrid result would not prove fusion-only learning impossible.
- Use the reverse hybrid to distinguish whether adapted backbone alone is
  sufficient on this fixture, or whether the combination appears necessary.

No warmup schedule, training run or revised numerical budget is adopted yet.
Any usable policy still needs native RT/NFR, actual NextLat latent and KL
objectives, packed T1024, and representative optimizer/update/restart checks.
The goal is a practical BF16 training path, not exact equality to FP32.

This is proposed after the completed adapted-state diagnostic, **not launched**. Preserve
the existing importer/runner/tests/protocols; use a new bounded runner/protocol,
CPU state-assembly checks, online W&B and immutable GCS evidence. Root owns GPU
launches inside the project container, with 900s stage limits and 20–30 minute
progress persistence. No additional GPU experiment is queued at this closeout.
