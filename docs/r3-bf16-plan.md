# Bounded R3 BF16 autocast milestone

The user authorized the [mixed-precision brief](inputs/r3_bf16_mixed_precision_agent_brief.md).
Use the unchanged Stage B resolved plan/checkpoints as the architecture/data
contract. Preserve old research and FP32-validation artifacts. New NUM/OPS work
uses `.runtime/r3-bf16/20260906T225438Z/`, archived under
`gs://fast-chunks/cdrm-w-latent/r3-bf16/20260906T225438Z/`.

One H100 is available in the required Docker environment. Run serial bounded
commands, at most 60 minutes of GPU command execution, with a 15-minute maximum
per command. Stop optional numerical expansion once the selected T128 profile
has enough evidence. No framework upgrade, architecture change, research sweep,
accumulation, distributed execution, or attention-backend optimization.

## Sequence and evidence

1. Reproduce the unchanged D256/T16/B2/V256 saved BF16 numerical fixture and
   preserve its original criteria, hashes and failure rows. Inspect actual
   masked-CE mixed precision at the original R3 initialization, B2/T128, before
   modifying the recurrent implementation. Save a pre-change source snapshot.
2. Diagnose the explicit FP32/BF16 boundaries and recomputation. Start from
   standard outer CUDA BF16 autocast, FP32 parameters/residuals/optimizer state,
   default autocast cache, disabled scaler/TF32, deterministic math SDPA, and
   compiled tiled helpers. Backward remains outside the outer autocast region;
   only recomputed dense forward operations re-enter it. Observe actual dtypes.
3. If justified, add an opt-in `bf16_fp32_state` recurrent precision policy,
   preserving the legacy default and the inactive FP32 arithmetic. Match naïve
   and tiled attention policy: BF16 dense projections, FP32 sensitive attention
   state/products and temporal adjoints. Projected BF16 records may have explicit
   FP32 working views. Additional forward state is conditional on measured
   mismatch at recomputed activation/quantization boundaries.
4. Compare naïve FP32, naïve mixed, and compiled tiled mixed using actual masked
   mean CE, all100 parameters, true block3 input and embedding gradients, and
   identical-state AdamW updates. Begin with B2, then B64 at initialization and
   update2000. Reuse saved FP32 packets only when fixture/checkpoint identities
   match. Include a separate unchanged-FP32 regression and isolated persistent-
   write credit/scaling checks. Confirmation uses new diagnostic examples
   (next batch counter or disjoint B2 subset) after any exploratory decisions.
5. Once numerical behavior is explained, run one paired100-update R3 comparison
   at B64/T128, consuming the first100 batches of the original2000-update
   schedule. Save a midpoint50 checkpoint and resume50→100 in a fresh process.
   Report exact equality separately from numerical closeness, keep precision
   metadata and all state identities, and preserve the update-zero restriction.
   Monitor fixed development data, loss, clipping, norms, finiteness and write
   parameter gradients. This is operational evidence, not architecture efficacy.
6. Benchmark SEQ/R3 FP32 and selected BF16 using actual masked-CE optimizer
   updates, five warmups and20 synchronized measured updates at B64/T128. Record
   observed dtypes, compilation separately, latency/tokens per second, allocated
   and reserved peaks, and optimizer bytes. Report measured benefit honestly.

## Numerical interpretation

Historical screens remain relative-L2≤0.015625 and max-error/reference-RMS≤0.0625
for BF16 gradients, and logit `atol=.002, rtol=.02`. The prior FP32 diagnostics
remain separately visible. Neither BF16 implementation is an exact oracle. All
intended gradients must be present, finite and FP32; all parameters and Adam
moments remain FP32. No model-wide BF16 cast or global backward autocast.

The initial passes are explicitly exploratory. Diagnose individual parameter
and activation differences before choosing any additional mixed-precision
engineering screens; record the rationale and freeze them before confirmatory
fixtures. Use per-tensor norms, maxima, cosine/direction, near-zero coordinates,
clipping, moments, actual deltas and their sparsity. Compare additional tiled
error with the measured precision error of the naïve implementation. A good
loss or global cosine alone cannot waive an affected parameter group. Do not
increase thresholds merely to make the observed fixture pass.

The FP32 Adam investigation already established near-epsilon first-step
sensitivity; preserve local update flags and trace their magnitude/direction,
rather than treating a rigid percentage-of-LR screen as a sole clearance rule.
Bounded trajectory/development and recovery evidence are required in addition
to pointwise numerical comparisons. Report separately: dense autocast active,
recurrent numerical clearance, bounded training/recovery, and useful performance.
