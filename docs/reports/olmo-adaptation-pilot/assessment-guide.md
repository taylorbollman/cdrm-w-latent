# Assessing the first 32 updates

This guide supports the planned review after B, NF and NFR finish at update 32.
It does not authorize continuation, alter stop criteria or propose another
readiness campaign. At preparation time, only B is complete; NF/NFR outcomes
must be assessed from their finalized reports and common development panel.

## What the reported losses mean

The active K4 objective in NF/NFR is

\[
L = \tfrac12 C_1 + \tfrac16(C_2+C_3+C_4)
  + \tfrac14\sum_{p=1}^{4} A_p
  + \tfrac14\sum_{p=1}^{4} Q_p.
\]

Here each term is already normalized by its own global valid-position count:

- \(C_p\): ordinary next-token cross-entropy on pass \(p\). The reported
  aggregate training CE is the weighted combination above, not final-pass CE
  and not an unweighted four-pass average.
- \(A_p\): coordinate-mean SmoothL1 with beta 1 between the predicted next
  latent \(\hat h_{t+1}^{(p)}=G(h_t^{(p)},e_{t+1})\) and the detached actual
  next latent \(h_{t+1}^{(p)}\), averaged over eligible pairs.
- \(Q_p\): vocabulary-summed
  \(D_{KL}(q(h_{t+1}^{(p)})\,\|\,q(\hat h_{t+1}^{(p)}))\), averaged over
  eligible triples. The teacher state and readout weights are detached; the
  predicted-state branch receives gradients. This is each pass's next-state
  prediction objective, not distillation from pass 1 or from the previous FBT
  pass. There is no additional supervised token CE on the predicted latent.

All three objective weights are 1.0 when enabled. Positions are counted once
per logical update, not multiplied by four passes. Under continuous packing,
CE includes valid within-row adjacency across true document boundaries;
latent pairs and KL triples stay within the same actual document. Dummy rows
contribute no targets. The scalar loss magnitudes have different units and
normalizations: a larger KL or latent value does not by itself establish which
term dominates backbone gradients.

B uses one ordinary pass and CE alone. Its disabled latent/KL means are null,
not measured zeros. Shared recipe fields such as `rt_layers` or auxiliary
weights describe available settings; the active arm/mode controls whether they
run. The predictor is shared across FBT passes and is not used as a learned
latent rollout at inference. Detaching the KL readout does not detach the tied
embedding lookup path through the predictor input.

These statements were checked against the frozen source pins in
`summary-b-01/report.json`, specifically
[`CampaignObjective.forward`](../../../cdrm/pretrained/campaign_training.py),
[`compute_dynamic_nextlat_loss_sums`](../../../cdrm/pretrained/campaign_losses.py),
[`aggregate_pass_losses`](../../../cdrm/pretrained/fbt_training.py), and the
[NextLat masks and weights](../../../cdrm/pretrained/nextlat.py). No runtime
source was changed.

## Review checklist

1. Confirm completed, retained update-32 endpoints and the pinned cohort
   summary. Compare NF/NFR only after the summary verifies common ordered
   memberships, development membership, prior fusion ancestry and exact named
   initial parameters. Lean evaluation preservation is an ownership/version,
   runtime/RNG and boundary-metadata check, not a new full-byte model/Adam
   equivalence result.
2. Report absolute dev CE for every pass at 16 and 32, then each later-pass
   gap \(C_p-C_1\). A shrinking gap is insufficient if it shrinks because
   pass 1 gets worse. Useful refinement at a checkpoint requires a later pass
   to improve on that checkpoint's first pass; movement toward zero can still
   be encouraging adaptation without reaching that condition.
3. Compare NFR minus NF at equal new input exposure, per pass and per gap.
   NFR applies RT at layers 0/15 on the first pass as well as later passes, so
   its first-pass comparison already includes RT. Use B as an ordinary-model
   continuation reference, with its different prior exposure and loss mix
   stated. A lower weighted training objective across these arms is not a
   like-for-like CE comparison.
4. Examine separate CE, latent and KL trajectories together with raw gradient
   norms, clipping fractions and clipping coefficients in the fixed eight-
   update windows. The training windows see different examples; the fixed
   development panel supplies the cleaner 16-to-32 comparison. Training uses
   BF16 mixed precision and feedback jitter; development uses common FP32 with
   jitter disabled, so train/dev loss levels need not coincide.
5. Keep adaptation and execution findings separate. Persistent clipping can
   coexist with useful adaptation. Falling auxiliary losses can coexist with
   worsening language prediction and are not sufficient evidence of useful
   latent structure. Nor do large gradients alone identify a precision bug.
6. Report the practical cost: active/trainable versus registered parameters,
   scoped throughput, memory samples, setup/evaluation cost and checkpoint
   stalls. Equal input exposure is not equal compute: NF/NFR use four passes,
   and RT adds recurrent work without new parameter tensors. Background
   retention time overlaps other work and cannot be added as sequential time.

## How to interpret clipping and the short horizon

Clipping occurs once on the accumulated, globally normalized gradient, before
AdamW. The reported norm is the pre-clip norm and the estimated coefficient is
\(\min(1,1/(\|g\|+10^{-6}))\). It is not an effective learning-rate multiplier:
Adam's moment normalization, epsilon, prior gradient history and separate weight
decay determine the parameter update. Do not infer a proportionally smaller
parameter step merely from a small clipping coefficient.

Every arm starts fresh optimizer moments. NF/NFR additionally activate a fresh
predictor and train the shared fusion128 import with the backbone. Such startup
is materially different from uninterrupted steady-state pretraining, even
though the backbone is pretrained. It makes transient scale changes plausible;
it neither proves those changes harmless nor attributes them to BF16.

Update 32 is only 32% through the declared 100-update warmup. The first update
uses LR 0.0000200, update 16 uses 0.0000470 and update 32 uses 0.0000758; the next
scheduled rate is 0.0000776, versus the 0.0002 peak. The warmed-up regime has not
been tested. There is no update-zero evaluation, so this pilot observes changes
from 16 to 32, not immediate pristine-start retrofit damage.

B currently provides this reference: dev CE 2.631118 at update 16 and 2.631794
at 32, a difference of +0.000676 nats/target; zero of 32 updates clipped, with
norms 0.3797–0.4532. This is descriptively flat ordinary continuation on this
panel, not evidence that NF/NFR must have the same norms or that fresh-optimizer
effects are ruled out for their newly active objectives.

## Decision at the planned stop

| Observation at 32 | Interpretation and next discussion |
| --- | --- |
| Absolute later-pass CE improves, gaps narrow without sacrificing pass 1, and gradient scales settle | Encouraging adaptation. Discuss a bounded continuation within the declared ceiling; useful refinement remains a separate criterion. |
| A later pass beats pass 1 on the fixed panel | Preliminary refinement functionality. Compare NF/NFR and cost; two observations on one development panel are not a scientific win. |
| Gaps narrow primarily because pass 1 worsens | Do not call this refinement improvement. Assess absolute CE and compare B/NF before extending. |
| Auxiliary losses improve while CE worsens, or none of the key indicators improves | Pause extension as declared. Prefer a bounded saved-state per-loss/fusion diagnostic over an immediate architecture or precision change. |
| Norms remain large but absolute CE and pass gaps improve | Clipping alone is not a failure. Retain its scale/frequency and assess whether continued adaptation is worth the measured cost. |
| Nonfinite values, incoherent counters or failed integrity | Execution failure under the existing stop policy; preserve the last verified recovery authority. |

If a targeted gradient diagnostic is needed, hold checkpoint, examples, masks,
precision and initialization fixed. Measure the already implemented weighted
CE/latent/KL gradient contributions separately by backbone, fusion and predictor,
and their agreement or cancellation in the combined gradient. Loss scalars
alone cannot answer that question. Keep this as conditional follow-up; do not
launch it or change weights/LR during the fixed cohort.

The development prefix covers seven of nine source strata and omits books and
Wikipedia. These are early teacher-forced development results, not confirmation
performance, sampled-generation quality or a general verdict on RT. Do not
extend beyond update 32 automatically.
