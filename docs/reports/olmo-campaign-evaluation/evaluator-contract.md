# Local per-pass evaluation contract

This additive evaluator uses the unchanged campaign model and canonical
`FBTNextLatLM.loss_sums`. It supplies no gradients, optimizer operations,
collectives, training-cursor movement or CUDA graph replay. It does not select a
quality benchmark or clear BF16 training numerics.

```python
with evaluation_runtime(model, device=device, generators=rank_generators) as preservation:
    local = per_pass_sums(model, batch_on_device, recipe)
```

The caller must be at a completed optimizer/cursor boundary, with no pending
backward. Existing parameter gradients must be absent or zero-valued persistent
buffers. The model is the unwrapped native campaign model with FP32 masters.
TF32 must already be disabled on CUDA; this context does not change process-wide
training precision settings. Local and named RNG are restored, as are individual
module training modes and all snapshotted native runtime fields.

The context forces math SDPA, FP32 attention and eager native RT forward/backward,
native RoPE and eager ordinary pointwise operations. Autocast and its cache are
disabled, and every module uses eval mode under `no_grad`. This matches the seven
runtime overrides in the existing canonical FP32 diagnostic. Other native flags
retain their values and are restored explicitly. No weight casting, replacement,
optimizer-state copy or model-sized tensor snapshot occurs.

Native RoPE tables are invocation-owned; the implementation has no lazy model
RoPE cache. Evaluation never receives the training adapter's prepared tables.
The caller separately snapshots and checks graph-owned inputs/noise/RoPE tables,
gradient storage, graph owners, optimizer/scheduler/cursor state and, in acceptance
mode, complete byte hashes. The local context checks registered tensor ownership,
storage, shape, stride, version and trainability, cache-generation markers, modes,
RNG and zero-gradient identities. These metadata checks are not a claim to detect
arbitrary external `.data` mutation that bypasses tensor version accounting.

Restoration runs on errors as well. Unexpected gradient replacement or changed
zero buffers are restored to their original zero-valued objects, but the integrity
check still fails. Mutated weights are never rolled back from an unsafe snapshot.
The evidence distinguishes `restored` runtime/modes/RNG from `integrity_passed`;
a failed integrity check is an error, not permission to save potentially damaged
state. Previously committed checkpoints remain the recovery authority.

## Scalars and denominator semantics

`per_pass_sums` makes exactly one canonical forward using the recipe's selected
passes/RT layers and `feedback_jitter=0`, with `feedback_noise=None`. It retains
continuous-stream attention/CE and same-document auxiliary eligibility, explicit
term masks and right padding. Neither EOS tokens nor row boundaries are invented.
The input must already be on the local device at the declared sequence length.

The returned JSON contains `passes` (index, raw CE/latent/KL sums and per-term
counts), canonical weighted `aggregate_sums`, `counts`, `weights`,
`term_pass_coefficients`, `enabled`, `input_tokens`, and the exact no-jitter mode.
Counts are positions in one pass, never multiplied by the number of feedback
passes. Campaign K4 CE weights are `(1/2,1/6,1/6,1/6)`; auxiliary pass weights are
uniform. K1 has coefficient one. Disabled auxiliary terms explicitly have sum
zero, count zero and `enabled=false`. Entirely padded local batches are legal
zero contributions. Nonfinite loss scalars are rejected.

The caller reduces raw sums and counts globally before forming means and weighted
objectives. It must not average rank/batch means or divide by a zero local count.
Evaluation timing includes eager forwards, scalar device synchronization and
preservation checks; it is not training throughput.

CPU tests use literal token-coordinate masks and mathematical CE, SmoothL1 and
teacher-to-student KL oracles across all eight arms, including true document
boundaries, padding, a dummy row and independent term masks. They cover empty
local batches, runtime/RNG/mode/gradient preservation on success and exceptions,
visible integrity failures, and unchanged subsequent real prepared CPU updates.
They make no claim about CUDA graph or NCCL behavior; the shared engine's bounded
GPU insertion acceptance supplies that evidence separately.
