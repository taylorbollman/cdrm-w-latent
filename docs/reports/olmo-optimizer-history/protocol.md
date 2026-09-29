# Fixed-state Adam-history diagnostic

This bounded experiment asks whether resetting Adam changes the translation of
the same FP32/BF16 gradient difference into a parameter update. It does not
resume training, change epsilon, select a new backbone or establish a numerical
acceptance threshold. Source, tests and this protocol freeze before execution.

The only model is the retained BF16 full-NFR update20 endpoint, SHA256
`bbd00dc1b70e08cae516ab4bb0f491ff03884efcd6e2d8b92558bf45ddbd7dd6`.
Its original and paired continuation reports are independently byte-pinned in
the helper. The unchanged historical report and checkpoint validators check
complete serialized model/Adam/scheduler/RNG/cursor boundaries, tied ownership,
source/runtime identity and verified cloud retention. Current native construction
must match the historical recipe and model contract before import. These are
**20 adaptation steps of Adam history, not original OLMo pretraining moments**.

Use the unchanged two-record B2/T128 development fixture (four documents,
512 inputs, 508 CE/latent targets and 504 KL triples), SHA256
`830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9`.
Load it through its explicit NF data-only construction contract, then check the
exact retained tensors/noise for NFR execution. Native RT selects layers0/15
with alpha1; K4 feedback uses beta1, keyed jitter0.02 and isolated-document
semantics. Combined CE/latent/KL weights and denominators remain unchanged.

Run exactly **two aggregate backwards / four physical backwards** at these fixed
weights: FP32 math/eager and BF16 mixed Flash/native Triton, with FP32 masters,
TF32 disabled and deterministic settings. Forced SDPA context encloses backward
recomputation; BF16 autocast encloses forward only via the unchanged loss helper.
Report raw gradients separately. Apply the actual global norm clipping at1
independently to each precision before any counterfactual Adam arithmetic.

Six conceptual full-model updates cross inherited versus reset history with
FP32, BF16 and explicit-zero gradients. Inherited retains each saved first and
second moment and step20; reset clears both moments and step, giving step1 after
the candidate. Every condition uses the same saved current LR, betas(.9,.95),
epsilon1e-5 and per-group decay. Scheduler/counters never advance. In particular,
reset does not restart the learning-rate warmup. Zero gradients are explicit
tensors, so momentum/decay still operate; `grad=None` is not used as that control.

To avoid six full-state copies, stream canonical parameters. Each candidate runs
the actual CUDA FP32 `torch.optim.AdamW(foreach=False, fused=False)` on temporary
parameter/moment clones. The live model and its restored optimizer never step.
After global clipping, unfused Adam is independent per parameter; CPU tests
compare this factorization exactly with literal full multi-group Adam. Report
six conceptual candidates and `6 × canonical_parameter_tensors` actual temporary
optimizer calls separately. This is the original diagnostic's unfused arithmetic,
not an equivalence claim about the newer fused production optimizer.

Accumulate component/all vector norms, absolute and relative errors and cosines
in FP64 from actual stored FP32 parameter changes. Compare actual deltas, deltas
minus the common FP32 decay multiply, and deltas minus each history's common
zero-gradient Adam update. The last quantity is a counterfactual diagnostic,
not an additive causal decomposition. Also report reset-versus-inherited update
geometry within each precision. Two clipped CPU gradient maps are retained;
temporary candidate states and FP64 deltas are bounded to one parameter.

Epsilon observations use the gradients actually passed to Adam and each FP32
candidate's bias-corrected sqrt(second moment), before adding epsilon. Report
coordinate fractions below epsilon and their fraction of FP32 update energy.
Report strict sign flips separately from zero/nonzero sign mismatches, their
gradient-energy fraction, and their fraction of the reset precision-difference
energy. A large coordinate fraction alone cannot establish optimizer harm.
No epsilon or learning-rate sweep is authorized here.

Operational acceptance requires finite losses/gradients/candidates, complete
participation, exact budgets and unchanged model/Adam/scheduler/counters/RNG,
parameter identities, fixed data/noise, source inventories and authority bytes.
The mmap checkpoint is hashed before load and its file identity checked at the
end; the complete saved tensor boundary is rechecked. Clear gradients and restore
runtime flags on the gradient helper's exception path. No checkpoints are written
or uploaded by the probe. W&B receives diagnostic scalar comparisons. Root owns
the single-GPU launch and separate evidence retention. Soft budget is2700seconds;
an external bounded process timeout is required. Long-run quality, actual
pretraining-moment restoration and unrelated optimizer settings remain untested.
