# Conditional integrated follow-up

This is a code-path assessment, not an implemented or scheduled integration.
Only pursue it if the isolated author backend has a repeatable useful advantage.

The smallest integration is an opt-in `rt_implementation` selector on
`OLMoTiledRTForCausalLM`, defaulting to native. Its `_forward_prepared` loop is
shared by public finite forwarding and `PreparedFBTLayout._stack`; dispatch the
selected RT blocks there to the author function. Keep ordinary blocks, final
normalization, tied embeddings/readout, parameter identities and checkpoints.
Lazy-import the author module only when requested. Explicit author options are
precision policy, compiled helpers, MLP chunks and autocast-cache policy.

Before capture require full-strength recurrence, all tokens valid, no prefix or
exported cache, supported precision and prepared native FP32 RoPE tables.
`PreparedFBTLayout.all_tokens_valid` is CPU-derived and avoids reading a CUDA
mask during capture. Reject unsupported requests instead of silently falling
back. Include all backend/options in static structure/execution signatures and
metadata; changing them requires rebuilding the layout and graph.

FBT's first pass remains ordinary; later passes use selected RT layers0/15.
The explicit returned parameter gradients should accumulate correctly across
ordinary and recurrent passes, but this requires validation. Ordinary layers
retain deterministic PyTorch Flash SDPA and checkpointing; author RT itself has
no Flash call. NextLat and CE2048/KL128 semantics remain unchanged.

The bounded follow-up would include tiny FP32 ordinary/RT/K2 combined checks and
one K3 shared-block check, then actual native B8/T512 RT-only and K2 combined
raw-gradient/graph/full-Adam checks. Only afterward consider paired native and
author B64 full-CE timings. Verify ordinary Flash dispatch explicitly and update
the integrated resource ledger to count author materialized backward, regardless
of the native recompute flag. Preserve any existing precision qualifications.
