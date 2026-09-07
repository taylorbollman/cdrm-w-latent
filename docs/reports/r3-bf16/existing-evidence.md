# Existing BF16 evidence audit

Historical source audited 2026-09-06 before BF16 implementation changes;
extended with the completed legacy replay and actual-CE baseline. **The historical BF16
runs already used FP32 parameters with outer BF16 autocast. They were not
all-BF16 model conversions.** The existing limits pass on the small Stage A
fixture and fail on every Stage B BF16 numerical fixture. The subsequent
[FP32 backward clearance](../r3-backward/results.md) does not clear BF16.

This audit ran no GPU or model computation and changed no historical report or
source. The accompanying [JSON](existing-evidence.json) records exact file and
source hashes, resolved configurations, failure names, numerical summaries,
and provenance. A CPU-only project container inspected the retained fixture
and installed AMP API.

## What precision was actually requested

The [Stage B harness](../../../scripts/stage_b_backend_check.py) constructs
ordinary FP32 models, wraps forward in `torch.autocast("cuda", dtype=torch.bfloat16)`,
and calls backward after leaving that context. There is no `model.bfloat16()`
or gradient scaler. All 12 Stage A BF16 benchmark records explicitly record
FP32 parameter dtype and BF16 forward autocast. Stage B OPS metadata describes
FP32 parameters and AdamW moments; masked CE consumes FP32-cast logits.

The original numerical records cast returned logits and gradients to FP32 on CPU, so
they do **not** establish the original dtype of every intermediate, gradient
buffer, or optimizer tensor. The new legacy actual-CE baseline below fills
several of those observation gaps. `config.precision=None` leaves internal `block_autocast` inactive;
`reference_eager=True` bypasses helper compilation and internal autocast, but
does not disable the surrounding outer autocast.

The tiled backward captures the forward Q dtype and explicitly reenters
autocast for projection/MLP recomputation. Attention recomputation and temporal
gradient formulas execute outside those contexts. The code uses no
`custom_fwd`/`custom_bwd` decorators. Installed PyTorch is
`2.13.0a0+8145d630e8.nv26.06`, CUDA build 13.3. Its `custom_bwd` restores the
forward autocast state; this alone would not establish FP32 recurrent
reductions. In `custom_fwd`, non-None `cast_inputs` also disables autocast
inside forward, so it is not an interchangeable annotation.

## Numerical evidence and limits

Stage A uses B2/D32/T9, vocabulary 32, 12 layers, four full-MHA heads, MLP64,
pre-norm ALiBi, affine Q/K normalization, **embedding normalization enabled**,
normal initialization with standard deviation 0.02, no biases/dropout, and
seed 937. Its diagnostic uses eager helpers and one backward chunk; the final
GPU test also covers compiled helpers and chunks 1/4. This differs materially
from Stage B's Mitchell initialization, MLP128, and absent embedding norm.

The preserved [initial test failure](../stage-a/validation/bf16-initial-failure.txt)
used gradient `atol=0.003`, `rtol=0.03` and failed final-head coordinates.
The later [same-cotangent diagnostic](../stage-a/bf16_diagnostic.json) compares
all 101 parameter gradients, including the embedding norm. It has no separate
input-gradient check. Every tensor satisfies the later scale bounds: maximum
relative L2 is 0.01077 against FP32 and 0.007001 between BF16 backends. This
does not erase the initial failed rule. The final GPU suite records 55 passes,
but does not retain per-case BF16 metrics for each helper/chunk setting.

Stage B uses B2, vocabulary 256, 12 layers, four full-MHA heads, GELU,
MLP width 4D, pre-norm ALiBi, affine Q/K normalization, Mitchell initialization,
no embedding norm/bias/dropout, untied head, R3 at rho=1, and four backward
chunks. Each comparison includes all 100 mapped parameter gradients plus the
embedding-output gradient. No missing gradient is skipped.

The unchanged BF16 gradient limits require **both** relative L2 <= 0.015625
and maximum absolute error/reference RMS <= 0.0625 per tensor. Logits use
`atol=0.002`, `rtol=0.02`. Stage B additionally applies the logit rule against
FP32; Stage A checked logits only between BF16 backends.

| Fixture | Comparison | Failed gradients / 101 | Worst relative L2 | Worst max error / RMS |
|---|---|---:|---:|---:|
| D32/T128 | Naive BF16 vs FP32 | 47 | 0.03249 | 0.18283 |
| D32/T128 | Tiled BF16 vs FP32 | 47 | 0.02848 | 0.12235 |
| D32/T128 | Tiled BF16 vs naive BF16 | 10 | 0.01972 | 0.19272 |
| D32/T256 | Naive BF16 vs FP32 | 47 | 0.02396 | 0.24277 |
| D32/T256 | Tiled BF16 vs FP32 | 42 | 0.02840 | 0.16107 |
| D32/T256 | Tiled BF16 vs naive BF16 | 9 | 0.01980 | 0.21956 |
| D32/T512 | Naive BF16 vs FP32 | 24 | 0.02520 | 0.31810 |
| D32/T512 | Tiled BF16 vs FP32 | 22 | 0.02192 | 0.19198 |
| D32/T512 | Tiled BF16 vs naive BF16 | 5 | 0.02383 | 0.30547 |
| D256/T16 | Naive BF16 vs FP32 | 61 | 0.02063 | 0.16904 |
| D256/T16 | Tiled BF16 vs FP32 | 61 | 0.02222 | 0.18898 |
| D256/T16 | Tiled BF16 vs naive BF16 | 23 | 0.01835 | 0.15625 |

Every row also fails its logit rule; all recorded gradient values are finite.
The [original records](../stage-b/backend-summary.md) remain failed. These
scale-normalized failures cannot be cleared by the later FP32 raw-cotangent
cancellation analysis.

At D32/T128, the ten between-BF16 gradient failures comprise block 3's
`k_norm`, `q_norm`, `attn_out`, `ff_out`, `ff_proj`, and `kv_proj` weights,
the embedding weight, and blocks 1/2/5 fused QKV weights. Worst relative L2
is block 3 `q_norm`; worst max/RMS is block 3 `ff_proj`. At D32/T512, the
five failures are the embedding-output gradient and block 3's `attn_out`,
`ff_out`, `ff_proj`, and `kv_proj` weights. Common-FP32 failures extend across
the rest of the model. This identifies inspection targets, not a proven faulty
operation or proof that every failure is unavoidable BF16 roundoff.

## Completed exact reproduction

The completed replay is
`.runtime/r3-bf16/20260906T225438Z/legacy-d256-t16.json`:
**B2/D256/T16/V256 with explicit math SDPA**. Independent JSON comparison
confirms exact equality with the historical D256/T16 record for the harness
hash, recorded source hashes, resolved model configuration, token/cotangent
hashes, conversion map, bounds, every comparison metric, and all **187 failed
tensor comparisons**. These comprise 39 FP32 gradient failures, 145 BF16
gradient failures, and three BF16 logit failures. Timing/provenance timestamps
are new; the failures remain failures.

This is the shortest saved full-width failure and uses the exact surviving
historical harness, avoiding reconstruction of the earlier D32 automatic-SDPA
script. It also matches the target model width and MLP width. Repeating D32
is unnecessary to establish that the original BF16 failure is reproducible.
The cheaper D32 packet remains available for subsequent localization.

## Alternative smaller-width fixture

The cheapest documented failing model is **B2/D32/T128/V256**, seed 937.
The original command used automatic SDPA dispatch; D32/T256, D32/T512, and
D256/T16 explicitly used math SDPA. All Stage B numerical runs enabled
deterministic algorithms, disabled TF32, and requested compiled tiled helpers
with eager naive recurrence. Their records show captured graphs and empty
graph-break/unsupported counters. Only D256/T16's historical harness hash
matches the current harness; the earlier scripts evolved. Running today's
default math harness is not an exact replay of the first auto-dispatch record.

Use the later FP32 reproduction's retained packet:
`.runtime/r3-backward/20260906T210249Z/raw-d32-auto/fixture.pt`.
The math sibling is byte-identical. CPU inspection confirms 100 FP32 state
tensors and the exact historical token/cotangent hashes:

- Packet SHA256: `f9bfb2298feb4e52499d59b40a88599aee47b87ca5cac06b9571c12ca533c828`.
- Token SHA256: `2522ca43d1d5c1007aee52c19bf121018a243ae6219fee53a9fcb1826eda527d`.
- Cotangent SHA256: `5708dcc617860b51e0c335eeb03f4c68e9e0f27236f6c5392cdd200f3941e5ac`.

This packet was saved during the later reproduction, not the original BF16
run. That reproduction matched the historical FP32 metrics; historical BF16
tensors were not retained. Replay the original BF16 comparisons and limits
with automatic SDPA, then label explicit math separately. D256/T16 is the
shortest saved failing sequence and a useful secondary reproduction, but has
many more parameters. There is no established smaller failing T9 fixture
under the scale bounds.

Trace forward and recomputed Q/K/V, residuals, normalization, running maximum,
denominator, weighted sum, reconstructed attention, and temporal gradients.
Historical source already allocates `atts`, `sum_scores`, `k_grads`, and `v_grads`
using residual `x.dtype`; that does not recover earlier BF16 rounding.
Source inspection identifies projection-derived cached K/V, `gs`, and
`g_dot_atts`; it did not establish the actual running-maximum dtype.
The observed legacy `max_logit` is FP32, as detailed below.
`recompute_alphas` explicitly softmaxes in FP32 then casts to Q dtype;
`recompute_atts` and temporal formulas consume those rounded values. Confirm
actual dtypes before assigning a cause or proposing a local change.

The historical fixtures are random output-cotangent derivatives, not masked
task CE. Actual CE comparisons on retained MQAR init/final checkpoints at
B2/T128 and then B64/T128, Adam effects, bounded paired training, midpoint
recovery, and performance remain separate BF16 gates. T512 is not a required
next experiment.

## Actual-CE legacy baseline and observed dtypes

The completed baseline is
`.runtime/r3-bf16/20260906T225438Z/legacy-ce-init-b2/report.json`.
It uses the retained MQAR R3 initialization, D256/T128/V1024, physical batch 2,
16 scored answers, aligned mean CE, and no accumulation. Its checkpoint SHA256
is `f8aad1e01d795cb695efad690aeabf837d0a48694ab8c2449ad8a321154ca2ba`;
batch SHA256 is `ddd68f0800eb573e3aeb98c7aa6828593bbd53c82ac6cb29d97bd36a2459094c`.
It records no source changes against the saved pre-BF16 snapshot and reuses
the matching retained strict FP32 reference. The original BF16 limits remain
unchanged. Its 102 gradient tensors add block 3's input to the original 101.

| Comparison | CE absolute error | Failed gradients / 102 | Worst relative L2 | Worst max error / RMS |
|---|---:|---:|---:|---:|
| Naive BF16 vs FP32 | 0.00046444 | 64 | 0.01853 | 0.26117 |
| Tiled BF16 vs naive BF16 | 0.00044203 | 23 | 0.02180 | 0.22665 |
| Tiled BF16 vs FP32 | 0.00090647 | 61 | 0.02483 | 0.28372 |

All intended gradients are present and finite. All three logit comparisons
fail the original elementwise rule. This actual-loss evidence confirms that
the legacy failures are not restricted to unnormalized random cotangents;
small scalar CE differences alone do not clear the gradient contract.

The new module/helper trace observes the following under the legacy policy:

| Boundary | Observed dtype |
|---|---|
| Embedding/residual and normalization outputs | FP32 |
| Dense projection outputs and recurrent pre-attention Q/K/V | BF16 |
| `block_attention_add` incoming/outgoing weighted sum | FP32 |
| `block_attention_add` incoming/outgoing running `max_logit` | **FP32** |
| `block_attention_add` incoming/outgoing denominator | FP32 |
| `recompute_alphas` output | **BF16** |
| `recompute_atts` output | **BF16** |
| Parameters, final gradients, and Adam state | FP32, explicitly verified |

Running-state observations cover 127 helper calls. They supersede any
source-only assumption that `max_logit` must follow Q dtype. Helper-boundary
traces do not by themselves reveal every fused internal intermediate or
temporal buffer. FP32 running state therefore does not establish FP32
weighted-sum inputs or accurate BF16 attention reconstruction.

The baseline also retains one real clipped AdamW update with LR 1e-5 and
epsilon 1e-8. Maximum parameter-delta discrepancies reach about 2e-5, and
the gradient-scale screen flags many moment/delta tensors. Those screen
labels are descriptive when applied to optimizer tensors, not an established
Adam acceptance criterion. As with FP32, first-step near-zero/sign sensitivity
requires separate interpretation. This baseline establishes neither corrected
BF16 backward precision nor bounded training/recovery clearance.

## Runtime and operational caveats

Historical runs used one H100 80GB, Python 3.12.3, the PyTorch build above,
cuDNN 92300, and host driver 580.173.02 with CUDA compatibility driver
610.43.02. Model/config/converter/common-helper hashes still matched the four
Stage B reports when this audit was written. Exact hashes are in the JSON;
future BF16 patches must be identified separately.

Stage A's initial compiled B1 BF16 benchmark explicitly records cache-limit
fallback; its later limit-64 rerun records no such fallback. Completion is
OPS evidence, not numerical clearance. Stage B's D256/T128/B64/V1024 probes
use two warm-up plus three measured CE/backward/AdamW updates: SEQ averages
0.03122 seconds in BF16 versus 0.02912 in FP32; tiled R3 averages 0.26710
versus 0.23269. BF16 reduces allocated memory in those probes but is slower.
The short timing sample and random-ID loss neither establish useful training
nor predict the performance of a corrected precision policy.

## Candidate core review

Read-only review of the proposed `bf16_fp32_state` branch found no blocking
gradient-connectivity defect. The naive FP32 Q/K/V views remain connected to
their projections and shared across temporal uses. Tiled backward applies its
manual FP32 adjoints to the corresponding graph-connected views. It preserves
the saved forward projection/autocast dtype separately from promoted arithmetic
Q, and replays the attention-to-BF16 boundary in both MLP recomputation paths.
Sensitive helper arithmetic explicitly disables autocast; dense recomputation
reenters the saved context. The inactive legacy/FP32 branches retain their
original arithmetic. The JSON records the exact candidate source hashes.

This is a source review, not numerical clearance. Changed helper signatures
and compiled code still require the FP32 runtime regression, followed by the
declared mixed-policy gates. The runtime must continue to disable TF32.
One integration limit is deliberate or unresolved: `convert_model` currently
rejects differing `recurrent_precision_policy` values. The diagnostic harnesses
load authoritative state dictionaries directly, so this does not block them.
