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
