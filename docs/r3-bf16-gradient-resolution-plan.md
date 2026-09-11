**Plan: identify and resolve the remaining R3 BF16-mixed gradient discrepancy**

Originally prepared as a plan; execution was authorized on 2026-09-07. The
[completed investigation](reports/r3-bf16-tiled-resolution/results.md) records
the tiled-first numerical findings, reviewed screen failures, compiler-cache
recovery correction, and the proposed PR pause. The original criteria and
historical results remain preserved.

The deployment target is **compiled tiled R3**; the non-tiled implementation is temporary diagnostic infrastructure. For this next milestone, prioritize tiled BF16 versus the validated tiled FP32 execution path, cross-checked with naïve FP32 and independent local higher-precision calculations where needed. Strong reference agreement, correct temporal credit, and bounded optimizer/training/recovery evidence can justify tiled clearance even if temporary naïve BF16 still differs, provided that disagreement is explained. Improving naïve BF16 merely to make the two mixed backends match is not a deliverable. Preserve the old failed comparison screens as historical results and freeze a separately documented, reference-centered acceptance contract before fresh confirmation.

A gradient is the signal the optimizer uses to decide how a weight should change to reduce loss. Naïve R3 retains the recurrent computation graph and lets ordinary autograd differentiate it. Tiled R3 saves less intermediate state and reconstructs parts of the computation during a custom backward pass. They implement the same intended recurrence, but organize arithmetic differently. BF16 rounds much more coarsely than FP32, so arithmetic that sums the same contributions in different orders can produce different gradients. Storing the final gradient in FP32 cannot restore information already rounded away internally. PyTorch documents both [operation-specific autocast behavior](https://docs.pytorch.org/docs/2.14/amp.html) and [differences between batched and sliced arithmetic](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html). The installed runtime and observed dtypes remain authoritative for this project.

The relevant evidence concerns the earlier Stage B MQAR profile: twelve blocks, D256/H4/full MHA, MLP1024/GELU, T128/B64, recurrent block 3, rho 1, actual aligned answer-only mean CE, learned normalization and ALiBi. The recent six-block CDRM selective-copying pilot ran in FP32 and supplies no BF16 clearance.

There are two different errors to distinguish:

- **Precision-conversion error:** naïve mixed versus naïve FP32. This includes the consequences of rounding forward activations as well as backward arithmetic.
- **Additional backend disagreement:** tiled mixed versus naïve mixed. This can include differences in forward values, replay, internal gradient arithmetic and parameter-gradient accumulation. It is not automatically a tiled backward defect.

The current mixed policy keeps parameters, residual state, sensitive recurrent-attention calculations, temporal adjoints and final gradients in FP32, while dense projections/MLPs and projected record storage use BF16. That change improved backend agreement, but it has not passed the complete frozen numerical gate.

| Retained evidence | Meaning |
|---|---|
| All 100 parameter gradients and two retained input gradients present, finite and FP32 | No recorded missing-gradient or NaN/Inf failure in these candidate tests. |
| All per-tensor relative-L2 and added-L2 budgets pass; worst mixed-backend relative L2 about 1.11% | The overall discrepancy within each gradient tensor is bounded under those screens. |
| Initialization B64 fails seven added-maximum-error budgets | Seven weight tensors have worst-coordinate disagreement above the predeclared allowance. This is not seven failed training runs. |
| Exceedances are 0.2–15.9% over the permitted error | These percentages describe the budget exceedance, not the percentage error of the gradient itself. |
| Trained-checkpoint B64 passes the added-error budgets | One numerical comparison from the retained FP32-trained update-2000 checkpoint and moments; it does not demonstrate long BF16 training convergence. |
| Separate 100-update FP32/BF16 pair remains finite with similar losses; midpoint recovery is exact | Useful operational evidence, with insufficient duration/task accuracy to settle learning quality. |

The authoritative criterion is [confirmatory-criteria.md](reports/r3-bf16/confirmatory-criteria.md); the older `frozen-criteria.json` is explicitly a superseded proposal. The current criterion bounds tiled-versus-naïve disagreement by the measured naïve-mixed-versus-FP32 error plus the old FP32 tolerance, using both tensor L2 and maximum error. It is a conservative engineering screen, not a theorem that identifies which backend is more accurate.

For example, one flagged block-3 gradient coordinate is approximately 0.00984518 in FP32, 0.00958252 in naïve mixed, and 0.00994873 in tiled mixed. Tiled is closer to FP32 at that coordinate despite contributing to a failed backend-disagreement screen. At six of the seven worst coordinates the mixed values straddle FP32; tiled is closer at four. Six gaps are one BF16 representable step. However, all seven coordinates are substantial non-near-zero gradients, and another output projection moves farther from FP32. The correct conclusion is an unresolved numerical discrepancy that needs attribution, not proof that either backend is universally wrong or that every flag is harmless. Exact values and original flags remain in [numerical-analysis.md](reports/r3-bf16/numerical-analysis.md).

The strongest current hypothesis is **different summation and rounding of dense weight-gradient contributions**. The naïve path calls the output projection and MLP per token ([model.py](../recurrent-transformer/olmo/model.py)). Tiled backward computes local temporal/input gradients and subsequently uses a batched, full-sequence MLP pass for parameter gradients. The `bwd_mlp_chunks` setting partitions local activation recomputation; it does not partition that final parameter-gradient reduction. Autocast's shared weight casts, GEMM reduction precision and compiler scheduling are plausible contributors, not established causes.

The isolated recurrent test supports that priority: both mixed backends produced identical projected K/V values and gradients for all 15 earlier writes, with nonzero credit to every earlier write. Its residual flags concern MLP weight gradients. Replay inputs also essentially matched at sampled positions 0, 1, 64 and 127. These observations narrow the investigation but do not establish equivalence at every position or full width.

1. **Reproduce a fixed starting point and preserve the original evidence.**

   Create a separate investigation lineage and record current/archived source identities. Reproduce the small isolated discrepancy and the original initialization B64 failure under its saved runtime policy. Check for source drift since the BF16 milestone before attributing any changed result to a fix. Retain naïve FP32, naïve mixed and compiled tiled mixed; add tiled FP32 as a regression control. Keep the actual normalization, Q/K normalization, masks, ALiBi, rho, loss reduction, physical batch and optimizer state unchanged. Set precision policy before constructing blocks and verify each block's effective configuration. Use the original failures as diagnostic regression cases, not fresh confirmation data.

   Exit: the failure and its exact scope are reproducible, or a documented source/runtime difference explains why they are not.

2. **Separate different forward values from different backward arithmetic.**

   First record forward logits, the actual loss gradient at logits, recurrent-block output, and the gradient entering that output. Full-model gradients can differ because the forward activations or loss gradients already differ. Give an isolated recurrent block identical captured inputs and an identical incoming output gradient in both implementations. Observe the first divergence through attention adjoints, temporary/permanent Q/K/V gradients, dense inputs, dense output gradients, casts and shared weight accumulation. Capture all positions for a bounded small fixture; expand to full-width B2/T128, keeping B64 capture targeted to avoid instrumentation-driven memory changes.

   Extract a minimal dense operation with exactly the same saved operands: `dW = sum_t(dY_t.T @ X_t)`. Compare repeated per-token operations against one concatenated operation, native BF16 behavior against FP32 accumulation of the same BF16-representable operands, and a small FP64 contraction reference. Preserve the intended cast/adjoint semantics. Finite-differencing a perturb-and-round BF16 program is not an independent smooth-gradient oracle. A full FP32 model is also a different forward computation, so use it alongside, not in place of, this fixed-operand test.

   Exit: localize the first discrepancy to forward rounding, temporal propagation, replay, dense input gradients or weight-gradient reduction. Agreement of two mixed implementations alone is insufficient.

3. **Test the implicated mechanism with controlled interventions.**

   Change one factor at a time on the fixed diagnostic fixtures: autocast weight cache enabled/disabled; BF16 GEMM reduced-precision reduction enabled/disabled; eager/compiled helpers; and local recomputation chunks 1, 4 and T. Start with the factors supported by step 2, rather than running a large factorial sweep. A cache test must consistently cover the outer forward and every custom-backward replay context, including the batched MLP helper. The current custom function records autocast enabled/dtype but not an explicit cache policy; auditing that omission is not the same as proving it caused the current default-policy failures. Disabling GEMM reduced-precision reduction does not remove BF16 output casts. Record actual pre/post-cast adjoints and compare every intervention against the same higher-precision local reference. Repeat a small uninstrumented comparison to detect effects of hooks or altered compilation.

   Exit: a factor changes the predicted intermediate and error pattern reproducibly. A flag merely disappearing is not sufficient causal evidence.

4. **Implement the smallest justified correction or document a precision-contract limitation.**

   If a shared runtime/cache policy fixes the identified mechanism, make that policy explicit and preserve it across replay. If weight-gradient accumulation is responsible, prototype FP32 accumulation for the implicated dense operations while retaining their declared BF16 forward behavior; validate both input and parameter gradients, shared ownership and exactly-once accumulation. Casting an already rounded final gradient to FP32 would not fix it. If replay or temporal routing is wrong, correct the specific mismatch and add a reproducing regression case. Save additional attention state only if measured replay differences justify it. Do not force tiled arithmetic to imitate a less accurate naïve result simply to reduce their mutual distance.

   If the behavior instead reflects reasonable BF16 rounding that the original one-sided budget misclassifies, retain every original failure and propose a separately versioned, reference-centered criterion with a mathematical rationale. Freeze any new thresholds before evaluating fresh fixtures. Do not quietly widen the old budget or tune an allowance against each failed coordinate. An unresolved discrepancy keeps BF16 experimental.

5. **Confirm the selected candidate on untouched cases and actual optimizer updates.**

   After freezing one candidate and its criteria, use new full B64/T128 batches at both retained initialization and trained checkpoint, plus an independent small seed fixture. Check every parameter and both retained input gradients, all original L2/maximum/coordinate screens, reference-centered errors for both mixed arms, tail counts/directions, fixed-forward power-of-two scaling and earlier-write credit. B2 and stricter FP32/FP64 arithmetic are diagnostic aids; actual B64 mixed execution is required for B64 clearance. Then test identical-state clipping and AdamW deltas, including initialization and nonempty trained moments. Keep the existing near-zero mask and report both inside/outside it: earlier initialization B64 had 95.31% of update-disagreement energy inside the mask but 440 gradient sign flips outside. Do not change optimizer epsilon, clipping or loss scale simply to hide those discrepancies.

   Exit: a documented numerical disposition under the declared criteria, supported by fresh confirmation rather than only the reproducer.

6. **Check training behavior, recovery and practical cost separately.**

   Following a satisfactory numerical disposition, run a bounded paired FP32/BF16 training and midpoint-recovery check with identical starting state, data order and schedule. Log loss, answer accuracy/exact match, gradient norms, clipping, per-tensor numerical errors and timing/memory to a clearly named project under W&B entity `taylorbollman`. Label numerical probes and operational training separately and share the dashboard URLs. Retain tensor packets/checkpoints and exact source/config/data/runtime identities locally and at a new `gs://fast-chunks` prefix. Rebenchmark warmed-up real forward/backward/clipping/Adam updates with observers removed and verify intended compiled execution.

   The previous B64/T128 R3 result saved 27.9% allocated memory but was 20.9% slower. A mathematically defensible policy may still be unattractive for throughput. Report numerical status, operational status and cost independently. Keep the initial clearance limited to the actual R3 profile; other sequence lengths, accumulation, distributed execution and CDRM need their own applicable validation.

The first deliverable is a reproducible cause-localization report and minimal failing example. The next is either a focused patch with fresh numerical/operational evidence or a documented explanation of the residual precision limitation and a defensible prospective validation contract. The goal is trustworthy gradients under a specified execution policy, not bitwise equality between every lower-precision execution schedule.
