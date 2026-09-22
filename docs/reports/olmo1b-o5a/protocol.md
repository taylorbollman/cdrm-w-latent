# O5a: bounded FBT correctness reference

Frozen scope before GPU execution, 2026-09-22. O4 is complete and assessed;
the user asked to continue. This milestone implements feedback mechanics and
their independent switches, without starting another learning comparison.

Use the unchanged original OLMo-1B step60000 checkpoint, native RoPE, no Q/K
normalization, tied embeddings, and layer0 when RT is selected. The FBT
reproduction source is `xidulu/Full-bandwidth-transformer` revision
`7037c60924870aca6e30fac95212b0c7caee052d`, retained as a nonexecuted source
snapshot. Its gate-product mechanism is the reference; this is not an exact
reproduction of the paper's pretrained model or training recipe.

For pass p>0, use previous-pass **post-finalnorm** state h at position t-1:

```
f_t = s * RMSNorm((W_U h_(t-1)) * sigmoid(W_G RMSNorm(e_t)))
u_t = (1-beta) * e_t + beta * f_t
```

Document starts keep e_t. Beta0 bypasses fusion exactly. Fixed new-branch
scale s is measured from the native embedding matrix; epsilon is explicitly
1e-5, and the new matrices use an isolated recorded RNG stream and source
uniform initialization. Native weights/norms are unchanged. One document per
row, with explicit valid masks; packed-document attention remains unsupported.

K counts complete stack passes. Pass0 is ordinary, extra passes use selected
RT or ordinary attention, all share parameters, and cross-pass gradients stay
attached. When FBT is off, standalone RT remains a single pass. Alpha and beta
are independent. The exact online API instead consumes the freshly completed
previous-token state and carries its own cache provenance; it has no K.
Finite-pass and online predictions may differ. Do not reuse previous-pass KVs
as current-pass KVs or silently accept an ordinary prefix as an online cache.

NextLat is optional, training-only, with the existing horizon-one predictor
and loss/mask/detachment contracts. Each pass has CE plus its independently
weighted auxiliary losses. Total loss is pass0 plus gamma times the mean of
extra-pass losses (gamma1 initially). K1 has no extra-pass term. Counts remain
data-position counts, not multiplied by passes. No autonomous predictor rollout.

Validation:

1. Tiny CPU tests compare finite passes, outputs and gradients against direct
   equations and exact online execution against independently reconstructed
   prefixes. Check causal shift, masks, all switches, K1/beta0, fractional
   alpha/beta, parameter ownership, initialization isolation, attached gradients,
   multiple outstanding forwards and cache rejection after mode/weight changes.
2. Actual checkpoint, H100, FP32/TF32 off: short B1/T8 finite outputs and all
   active parameter gradients against independently composed sequential RT and
   direct fusion. Cover ordinary endpoints, fractional transition, FBT-only,
   FBT+RT and NextLat objectives. Preserve explicit loss/gradient tolerances
   from the earlier OLMo semantic checks; report inactive branches separately.
3. Short exact online versus full-prefix replay, chunk continuation and finite
   convergence. Because pass0 is ordinary, allow T+1 complete passes when RT
   is enabled. Record finite-K discrepancies without treating them as errors.
4. One bounded actual BF16 mixed comparison against FP32 with descriptive
   per-tensor differences, not a new precision campaign or training clearance.
5. Small disposable optimizer/save-reload checks establish shared ownership and
   pass-loss compatibility. No research training checkpoint is produced.

Record source/config/checkpoint hashes, all cases and failures, W&B metrics,
test results, GCS evidence retention, usage and the compaction handoff. Review
this correctness milestone before choosing FBT training exposure, beta ramp,
prefix mixing or any auxiliary warm-start experiment. No multi-GPU claim.
