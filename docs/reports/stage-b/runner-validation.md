# Stage B runner validation (NUM)

The actual `scripts/stage_b_train.py` runner passed a bounded FP32 recovery check:
four uninterrupted updates equal two updates followed by a separate-process
checkpoint resume, bit for bit, for both SEQ and compiled tiled R3. Every model
tensor (99 SEQ / 100 R3), every Adam tensor (297 / 300), Python/NumPy/Torch/CUDA RNG
state, data digest/offset, learning-rate schedule, scientific counters, and
training/development metric matched. Runtime, compiler counters, UUIDs, and
artifact paths are excluded from the numerical identity comparison.

This used private real MQAR fixtures, D32 / 12 blocks / 4 heads / MLP128 /
vocabulary 1024 / T128, four examples per update, FP32 weights and computation,
deterministic algorithms, TF32 disabled, math SDPA, and R3 tiled helpers with four
backward MLP chunks. Initialization was paired exhaustively: all 99 sequential
state tensors mapped exactly to all 100 recurrent tensors. This validates the
runner path on a small fixture; it is not a full D256/B64 reproducibility claim
or SYN learning evidence.

A separate R3 probe evaluated real MQAR batches at T128, T256, and T512, then
performed a T128 training update with backward, clipping, and AdamW. It passed
with finite gradients for every parameter, 17 observed compiled graphs, no graph
breaks or unsupported paths, and `fail_on_recompile_limit_hit=True`. The probe
uses the runner's actual `evaluate` and `train_update` functions.

Two extra checks retain failures:

- R3 resume from update zero into a fresh output directory is not bitwise equal
  to fresh execution. The largest final difference is 4.30e-5 in an embedding
  element (embedding relative L2 1.40e-6). SEQ update-zero resume passes exactly.
  No semantic RNG/data/state-loading discrepancy was found in review; compiler
  history or allocation/layout effects are unverified hypotheses. R3 update-zero
  resume must fail closed pending validation. The original initialization
  checkpoint remains available for inspection.
- For one global batch split into two unequal-target microbatches (8 and 16
  answers), all 100 R3 gradients and all 300 Adam tensors pass unchanged
  `atol=2e-6, rtol=2e-5`, but one embedding parameter element fails that bound:
  max absolute difference 1.16e-5, embedding relative L2 3.68e-7. Maximum gradient
  and optimizer-tensor differences are 1.86e-8 and 1.86e-9. Adam sensitivity to
  small gradient differences is consistent with these observations but is not a
  proven cause. No parameter-equivalence claim is made for R3 accumulation.
  SEQ accumulation passes all three comparisons. The actual pilot uses
  `global_batch=microbatch=64`, so this unvalidated path is not required.

Metadata correction: the frozen trainer's `gradient_accumulation.note` calls
the failed coordinate a "near-zero-gradient embedding element." That specific
coordinate's gradient magnitude was not measured in this diagnostic. The phrase
is an interpretation, not an established observation; the measured differences
and unproven causal explanation above are authoritative. The original trainer
and run manifests remain unchanged to preserve experiment identity.

The read-only runner review checked aligned answer-position loss, full-global-
batch target normalization, the unchanged full-horizon schedule across pauses,
exhaustive paired initialization, exact stream identity on resume, development
selection, final-test gating, canonical parameter ownership, and compiler
fallback rejection. Review fixes addressed initial development selection on
resume, no-op resume, evidence classification, and source identity before the
integration runs. The subsequent update-zero rejection guard does not alter the
validated fresh or midpoint paths.

Machine-readable results and limitations are in `runner-validation.json`.
Detailed diagnostics, initial failures, input fixtures, CLI manifests,
checkpoints, and probe scripts are retained under:

```
.runtime/stage-b/20260906T190223Z/runner-validation/
```

`seq-continuous` / `r3-continuous` ran the full four-update plan.
`seq-resumed` / `r3-resumed` ran `--stop 2` and then resumed their `u0002.pt`
checkpoint to the same four-update horizon. The `*-init-resumed` branches used
the corresponding retained `init.pt`. `check_results.py` performs exhaustive
checkpoint comparisons; `diagnose_runner.py` retains every accumulation tensor
comparison and the mixed-length transition. The original failing report is
`runner-validation-initial-failure.json`; bounds were not loosened.
