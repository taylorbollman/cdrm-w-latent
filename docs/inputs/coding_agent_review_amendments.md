# Coding-agent review: essential amendments

Proceed with the bounded R3 correctness/conversion patch. No blocking questions for that patch.

1. **Audit masking across every execution path.** Raw ALiBi is not a causal mask. Ordinary attention uses `is_causal = attention_bias is None`; supplying raw ALiBi disables its causal fallback. Both capture helpers and uncaptured block benchmarks bypass the normal model's combined mask construction; graph wrappers also ignore runtime bias. Enforce explicit masking contracts and test SEQ/mixed causality independently. Keep CUDA graphs disabled until their own tests pass. Eager whole-model work can proceed.

2. **Make conversion exhaustive.** Existing helpers omit learned normalization state, including final/embedding norms. Test nondefault norms/biases and map fused-QKV gradients to split parameters. Pre-norm is the initial supported contract; never silently change a pretrained checkpoint's normalization. Weight transfer alone does not establish functional equivalence.

3. **Use one canonical parameter owner.** Share fused QKV through differentiable slices/functional projections. Do not wrap slices in new `nn.Parameter` objects: separate optimizer entries can share storage. Verify unique optimizer ownership and summed preview/writer gradients. Keep the refactor local and preserve checkpoint names where practical.

4. **Preserve the same-depth CDRM control.** R3 reads block-3 inputs and sends its output through later blocks; CDRM reads post-block-3 previews and injects after block 8. Thus CDRM versus R3 does not isolate deep-source value. With `c = p3 + epsilon * A(N(p8 - p3))`, substituting `p8 = p3` zeros the adapter input. Label this deep-source removal; equal nominal parameters do not guarantee equal active capacity. Add an active same-depth adapter control if attribution requires it, not automatically in the first patch.

5. **Fix latent-objective semantics.** Preserve next-observed-token conditioning, stopped targets, and enabled source gradients. Freeze NL2's target and normalization: raw versus normalized `p8` changes SmoothL1's effective scale. NL1 may compare `SEQ`, `SEQ+NL`, `R3`, and `R3+NL` once R3 is correct/trainable; an R3 win is not a prerequisite.

6. **Separate correctness from performance evidence.** Keep upstream surrogate-loss reproduction tests; benchmark actual shifted CE, backward, and optimizer updates separately. For custom backward, compare random-output-cotangent gradients. At `T=1`, compare outputs/logits, since shifted LM loss has no valid target pair. Add a tiny deterministic training/gradient/save-resume smoke test. Give SEQ the same optimizer reset and LR re-warm as R3 continuation. Distinguish learning from scratch from checkpoint adaptation when interpreting failures.

7. **Keep staging bounded.** Separate the mechanical submodule move from model changes; verify the imported path afterward. CPU support means small eager reference tests. Use validated upstream tiled R3/R8 early; defer custom CDRM tiling until semantics and gradients are stable.

Before substantive experiments, resolve: actual checkpoint/tokenizer/normalization compatibility; bounded pilot budget and primary metric; exact NL2 target/normalization before latent comparisons.
