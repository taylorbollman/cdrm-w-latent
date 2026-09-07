# Semantic decisions

## 2026-09-06 — Stage A

1. The recurrent-transformer personal fork is a top-level submodule. Its starting
   upstream pin is `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`; local changes and
   numerical reports are distinct from that pristine source. No upstream push is
   part of this work.
2. `ModelConfig.recurrent_layers=None` preserves legacy global block selection.
   Explicit `[]` means SEQ, `[3]` means R3, and `[8]`/`[3,8]` are constructible
   controls, not claims that their training/backend combinations were evaluated.
   Architecture identity is separate from `recurrent_backend=naive|tiled`.
3. Explicit mixed recurrence requires pre-norm, ALiBi, full MHA, no RoPE, no
   dropout, and ungrouped blocks. Cached decoding and document packing are
   rejected. Legacy all-recurrent post-norm remains available at rho=1 for
   upstream reproduction; it has no claimed sequential rho-zero endpoint.
4. R3 stores `(1-rho)*x_t + rho*z_t` after computing `z_t`, but passes `z_t`
   upward. Use `model.set_recurrent_write_rho(rho)` so saved model configuration
   and independent block configurations agree. Fractional rho is naïve only.
5. Pre/Post helper modules are non-owning views of canonical block modules.
   They do not register duplicate parameters or create Parameter wrappers over
   shared slices. Old agreeing helper state-dict aliases load; conflicting
   aliases fail. FSDP module replacement and distributed custom backward are
   outside the validated contract.
6. Ordinary attention enforces causal masking even when supplied raw positional
   bias. Direct block benchmark bias is combined causal+ALiBi and preserves that
   bias through compilation. CUDA graphs remain disabled until their runtime
   mask, gradient, and block-index semantics are independently tested.
7. `reference_eager=True` bypasses helper compilation and internal autocast.
   FP32 weights/inputs define the reference. BF16 tests and benchmarks use an
   explicit outer autocast context with FP32 weights and AdamW state. Tiled
   backward recreates the forward projection dtype during recomputation.
8. CPU reference tests, NUM comparisons, OPS overfit, SYN research, and LM
   research are distinct evidence. Stage A uses synthetic token IDs only.
   Its full model has a T5-shaped vocabulary but no verified T5 corpus/tokenizer
   pipeline or pretrained checkpoint. Smoke weights must not seed an LM parent.
9. Exact smoke resume restores weights, optimizer, schedule metadata, rho,
   random-generator states, and data offset. A fresh optimizer resets branch
   counters while preserving the common parent's next data position and stream
   RNG. The smoke schedule is deliberately small and is not the provisional
   250-update LM ramp.
10. Initial absolute BF16 gradient bounds failed a few final-head entries.
    The preserved same-cotangent FP32 diagnostic shows ordinary BF16 reference
    error is larger than the naïve/tiled difference for the failing final-head
    comparison and worst relative-L2 metric, though not every error metric.
    BF16 checks therefore bound
    every parameter separately: relative L2 <= 2*BF16 epsilon and maximum
    absolute error/reference RMS <= 8*epsilon, comparing both backends with
    FP32 and with each other. This tests a bias-free B2/T9/D32/H4, 12-block
    fixture at seed 937 with 1/4 backward chunks and both helper modes;
    it is not a full-size T512 precision certificate. FP32 tolerances remain
    unchanged. The pristine
    upstream FP32 failures remain failed records, not waived passes.
11. The provisional 5,000-update parent and 2,500-update continuations have not
    been launched. Data/checkpoint identity, primary metric, and a measured
    GPU-hour budget remain decisions before substantive experiments.
12. Later CDRM must retain an ordinary preview and the anchor-relative bridge.
    Replacing p8 by p3 in the difference adapter is a deep-source-removal
    control; it does not prove equal active capacity. An active same-depth
    adapter comparison is a later attribution decision. NL1 does not require an
    R3 win; NL2 target normalization must be fixed before its comparisons.

## 2026-09-06 — Stage B numerical and runner gates

1. The initial Stage B pilot uses FP32 parameters and computation, deterministic
   algorithms, explicit math SDPA, and disabled TF32. R3 uses tiled recurrence at
   rho=1 with compiled helpers and four backward MLP chunks. Compiler fallback
   is an error; observed graphs and rejection of unsupported paths are part of
   validation. BF16 remains blocked despite successful operational cost probes.
   Configuration and evidence are recorded in the
   [backend report](reports/stage-b/backend-summary.md).
2. Raw, unnormalized random-cotangent FP32 gradient comparisons failed some
   elementwise bounds. Those reports remain failures. A separately labeled
   diagnostic normalizes the same sampled output direction to unit L2 and keeps
   `atol=2e-6, rtol=2e-5` unchanged. It passes logits, the embedding-output input
   gradient, and all 100 named parameter gradients for naïve versus tiled R3 at
   D32/T128, T256, and T512 with vocabulary 256, and D256/T128 with vocabulary 1024;
   all use B2 and 12 blocks. This bounds initialization fixtures at the declared
   directional-derivative scale, not arbitrary cotangent magnitudes, trained
   weights, or full-width T512 gradients.
3. Stage B BF16 comparisons retain the per-tensor gradient limits
   relative L2 <= 0.015625 and maximum absolute error/reference RMS <= 0.0625.
   Naïve BF16 and tiled BF16 are each compared with a common naïve FP32 reference
   and with each other, including every parameter and the input gradient.
   Failures occur at D32/T128, T256, and T512 and D256/T16. These longer or wider
   fixtures extend the narrow Stage A test scope; its pass does not authorize
   BF16 here. Failed tensors are not omitted, and bounds are not loosened.
4. The actual runner passes bitwise midpoint recovery on private MQAR NUM
   fixtures: four uninterrupted updates equal two updates plus a separate-process
   resume for both SEQ and R3. All model/Adam tensors, Python/NumPy/Torch/CUDA RNG
   states, data identity/offset, schedule, scientific counters, and train/dev
   metrics match; runtime, compiler counters, UUIDs, and artifact paths are
   excluded. The fixture is D32, 12 blocks, H4, MLP128, vocabulary 1024, T128, B4.
   All corresponding initial SEQ/R3 tensors also match through exhaustive
   conversion. This is a bounded runner check, not a full D256/B64 exact-resume
   certificate. See the [runner report](reports/stage-b/runner-validation.md).
5. R3 resume from update zero into a fresh directory fails bitwise equivalence;
   the largest observed final embedding difference is 4.30e-5. The runner
   rejects R3 `--resume` at update zero pending validation. Fresh paired
   initialization and the tested midpoint recovery remain available, and the
   initial checkpoint is retained for inspection. SEQ update-zero resume passes.
   No semantic RNG/data/state-loading discrepancy was identified; compiler
   history and allocation/layout effects are hypotheses, not established causes.
6. Loss uses already aligned answer positions with no next-token shift; each
   microbatch contributes its summed answer loss divided by the full global
   batch's target count. An unequal-target accumulation check (8 and 16 answers)
   passes all SEQ comparisons. For R3, all 100 gradients and 300 Adam tensors
   pass `atol=2e-6, rtol=2e-5`, but one embedding parameter element fails
   (max absolute difference 1.16e-5). Adam sensitivity is plausible but unproven.
   Optional R3 accumulation has no parameter-equivalence claim; the pilot uses
   `global_batch=microbatch=64` and does not require that path. The failure is
   preserved without changing the bound.
7. A D32/B4 R3 probe using the runner's actual evaluation and update functions
   passes T128/T256/T512 development evaluation followed by a T128 training
   update. It observes 17 compiled graphs, finite gradients for every parameter,
   no graph breaks or unsupported paths, and an enabled recompile-limit failure
   guard. This checks the mixed-length evaluation-to-training transition without
   claiming full-size numerical or generalization coverage.
8. NUM runner fixtures and OPS random-ID timings remain separate from held-out
   SYN learning evidence. The short cost probes support an update-work estimate,
   not an end-to-end job budget or completed pilot result. The runner preserves
   the full planned schedule across pauses, uses development data for checkpoint
   selection, and gates final-test evaluation on completion of the frozen
   horizon. These decisions do not claim final Stage B task results.

## Post-Stage-B FP32 backward validation (2026-09-06)

The [bounded investigation](reports/r3-backward/results.md) clears the tested
FP32/no-accumulation R3 regime at rho=1: MQAR, D256/12 blocks/H4/T128/B2 and B64,
retained initialization and update-2000 weights, math SDPA and compiled helpers.
It reproduces the historical raw failure and attributes the evidence to ordinary
FP32 rounding and scale-sensitive coordinate tolerances. Actual masked-CE checks
include every parameter and the true block-3 input; an isolated probe confirms
gradients through all earlier permanent writes.

Original numerical flags remain failures. These include raw-cotangent coordinates,
a trained-B64 logit coordinate, max-error/RMS tails, and sparse first-step Adam
update screens. FP64 references and an exact first-step Adam decomposition support
shared floating-point/epsilon sensitivity rather than a backward defect. No
model/trainer change or Stage B research rerun was needed. The result does not
claim trajectory equivalence or clear update-zero resume, BF16, accumulation,
training at T512, distributed execution or CDRM. Twelve serial GPU commands took
241.22 seconds; no new research training was launched. New NUM artifacts use the
separate `r3-backward/20260906T210249Z` local/GCS lineage.
