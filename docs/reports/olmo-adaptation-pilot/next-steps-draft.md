# Provisional follow-up if NFR also retains the later-pass CE deficit

**Proposal only.** NFR is still running when this draft is prepared. Finish the
fixed cohort, retain its endpoints and assess the common summary before choosing
follow-up. No diagnostic or continuation is launched by this document.

NF already has the deficit without RT. Its dev CE improves from
3.254/7.775/7.798/7.816 at update 16 to 2.960/7.393/7.426/7.435 at 32.
Later-pass gaps narrow only about 0.08–0.09 nats/target and remain roughly 4.43–4.47.
All 32 updates clip. The eight-update median norm falls from 69.35 to about
12.98, while latent/KL losses fall substantially. This is some adaptation with
a large unresolved feedback deficit, not evidence of representation collapse.
If NFR is similar, diagnose the common NF path first, then confirm the relevant
finding with RT present. Avoid attributing a shared failure to RT.

## 1. Small forward-only localization

Predeclare eight complete T1024 rows from the existing fixed dev prefix, retaining
their membership, document identities and masks. These are a diagnostic subset,
not a new evaluation result or a substitute for the full panel. Use the existing
common-FP32/no-jitter path and immutable copied checkpoint weights.

Start with NF:

- Evaluate saved origin 0 and endpoint 32 at the actual beta 1.0, reporting every
  pass CE. This supplies a same-data origin comparison that the cohort's
  scheduled evaluations lack. An origin deficit that persists points toward
  incomplete adaptation of the imported fusion route; deterioration introduced
  during these updates raises a different concern. Neither alone identifies
  the responsible loss or rules out a longer startup transient.
- At endpoint 32, evaluate beta 0.0 and 0.5 as two fixed controls. Keep four
  passes and all other settings unchanged. Beta 0 exactly bypasses fusion;
  with deterministic evaluation and the same RT mode on all passes, each pass
  should reproduce the first-pass result within the established execution's
  numerical behavior. Beta 0.5 tests whether the learned full-strength
  feedback replacement is locally too disruptive. This is a diagnostic
  interpolation, not a proposed training schedule or a sweep to pick a winner.

In the actual implementation, beta 1 replaces eligible token inputs with the
normalized gate-product output; it is not a small additive residual. Beta 0.5
interpolates that fused output with the original embedding. Record fused-input
versus embedding RMS and cosine similarity. RMS alone is weak evidence because
fusion already normalizes and scales its output to the stored embedding scale.

Record only a few additional per-pass summaries using the same forwards:
teacher next-token entropy, predictor next-token entropy where available, and
across-position hidden-state variation on valid positions. Later-pass KL can
become small because student and teacher agree even when neither predicts the
real next token well. These summaries can identify an easier auxiliary target;
they cannot establish representation collapse or useful state tracking.

Mirror the decisive beta control at the NFR endpoint, retaining RT at 0/15.
Do not start a full RT-placement, beta, jitter, precision or architecture grid.

## 2. One bounded per-loss gradient decomposition

At the NF endpoint, use a small, separately pinned training diagnostic batch
(for example two T1024 rows from the next unused logical update), with the
actual masks and independently normalized denominators. Use one fixed
diagnostic precision and noise policy for every contribution. The existing
FP32/no-jitter diagnostic path is a reasonable simple choice; clearly label
it as local loss geometry, not the BF16 large-batch training update.

Compute four already implemented contributions, retaining the current weights:

1. First-pass CE: `0.5 * CE1`.
2. Later-pass CE: `(CE2 + CE3 + CE4) / 6`.
3. Latent regression: the four-pass mean.
4. NextLat KL: the four-pass mean.

Report norms and pairwise dot products/cosines by backbone, fusion and predictor,
plus whether adding auxiliary gradients aligns with or opposes the combined CE
gradient on shared parameters. Check that the decomposition reconstructs the
joint gradient to the diagnostic path's expected accumulation accuracy. Zero
gradient components have undefined cosines: CE does not train the predictor,
and first-pass CE does not directly use fusion.

This separates internal first-versus-later CE conflict from auxiliary conflict.
It also shows whether the large total norm is mostly predictor adaptation or
substantial pressure on the shared backbone/fusion. Preserve the exact detached
teacher/readout semantics: reimplementing a differentiable teacher would answer
a different question. Small-batch probe norms must not be numerically compared
to the cohort's 524,288-input gradient norms as if their scales were equivalent.

If a clear shared-parameter conflict appears, repeat only that decomposition on
one second fixed batch and the NFR endpoint. Do not expand to a broad numerical
campaign. Gradient opposition is local evidence, not proof that NextLat caused
the training deficit; Adam moments and subsequent updates matter.

## Decision after these probes

| Finding | Smallest sensible next discussion |
| --- | --- |
| The imported origin already has poor later passes, absolute CE improves during the cohort, and auxiliary gradients do not oppose CE strongly | Transient adaptation remains plausible. Discuss a bounded continuation with the existing optimizer/schedule and unchanged objective. |
| Beta attenuation restores CE substantially while full-strength feedback remains poor | Focus on fusion/feedback adaptation. A separately labeled short beta schedule or fusion-focused continuation becomes a candidate, not an automatic change. |
| Auxiliary gradients repeatedly oppose CE on backbone/fusion, while the predictor/later-pass auxiliary targets become much easier | Consider a short paired fork from the same endpoint and saved Adam state, changing only the implicated auxiliary treatment. Do not reset optimizer moments and call it the same comparison. |
| First- and later-pass CE gradients conflict even before adding NextLat | Focus on the feedback input distribution and multi-pass objective; removing NextLat alone may not resolve it. |
| Beta-zero passes fail their expected single-pass control | Localize the execution/control discrepancy before learning changes. This is a concrete functional check, not an architectural verdict. |

These probes should answer where to look with a few saved-state forwards and
backwards. They do not yet justify changing Q/K normalization, kernels, model
family, precision policy or all objective weights. Do not run a learning fork
or continue beyond 32 until the completed cohort and this conditional proposal
have been reviewed.
