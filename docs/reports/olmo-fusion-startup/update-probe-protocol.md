# Matched counterfactual fusion Adam updates

2026-09-29. Conditional follow-up after root assesses the saved fusion startup
endpoint. This new helper does not modify warmup, checkpoint, model or previous
probe sources. Root alone schedules GPU execution after source/test review.

Use the independently SHA-pinned compact **update 128** checkpoint, including
the two fusion matrices, full Adam moments and scheduler. Original OLMo backbone,
tied readout, predictor and fusion scale remain frozen. Import through the
existing strict loader; preserve its source/configuration/RNG/ownership checks.
Reject another endpoint or mismatched startup data authority.

Use exactly the supplementary four-document held-out T128 fixture: two B2
records, 512 input tokens, 508 CE targets, existing fixed row-keyed jitter.
No selection by observed outcomes. K4, beta 1, jitter 0.02, isolated documents,
CE pass weights 1/2,1/6,1/6,1/6 and the global 508-target denominator are fixed.
As in fusion warmup, omit unused auxiliary arithmetic and freeze every parameter
except the two fusion matrices. This is an artificial held-out update, **not**
continuation on the 8,192-target training schedule or a quality evaluation.

Perform exactly three real optimizer calls, each independently restored to the
same complete checkpoint boundary:

1. Zero-gradient Adam control: explicit zero gradients on both matrices, not
   missing gradients. Adam's existing moments age and weight decay still applies.
2. FP32/math CE gradient, unchanged clipping and saved AdamW configuration.
3. Production BF16/Flash CE gradient with FP32 masters and identical optimizer.

Only the latter two execute model backward: two aggregate cases, four physical
backwards total. Keep forced SDPA active through backward. Match production's
backward outside forward autocast; checkpoints restore their saved forward
autocast context during recomputation. FP32 remains outside autocast throughout.
Restore the checkpoint after observations;
never change the saved training cursor, publish an updated training checkpoint,
or treat these held-out candidates as training progress. No full-backbone
optimizer, additional precision, LR sweep or follow-on update is authorized by
this protocol.

Report raw and clipped gradient norms/directions, pre-clip norm and scale, first
and second Adam moments, exact stored FP32-master weight changes observed in
FP64, and their differences between precisions. Also report the common rounded
decay-only delta, each update minus that delta, and each update minus the actual
zero-gradient Adam control. The last control includes historical momentum, not
just decay. These subtractions are descriptive; Adam is nonlinear and the
residual is not an additive causal attribution. Small full-update differences
may otherwise be dominated by common history or decay. Record absolute errors
and reference norms, particularly for near-zero residuals.

Acceptance covers only finite healthy execution, exact initial state/Adam/RNG
identity across candidates, one optimizer-step advance per matrix, unchanged
frozen state and fixtures, restored original checkpoint, source integrity and
the three-call limit. No new numerical tolerance or BF16 production clearance.
CPU tests use tiny models and explicit CPU math attention; production requires
the GPU container and never falls back. Root's GPU stage has a 900-second cap,
online W&B, incremental reports, exact source snapshots and GCS retention.
