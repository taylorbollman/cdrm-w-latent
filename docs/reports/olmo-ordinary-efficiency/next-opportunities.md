# Ordinary OLMo: next optimization opportunities

## Evidence and scope

This note interprets the initial `control-b64-t512-r1` measurement at runtime
commit `ed26653a5606546ab0581049be8a8dd8e52df5a5`:
[raw report](../../../.runtime/olmo-ordinary-efficiency/control-b64-t512-r1/report.json).
The experiment uses the actual pretrained 16-layer OLMo checkpoint, B64/T512,
BF16 mixed precision, FP32 parameters/residuals/gradients, ordinary PyTorch Flash
attention, all-layer ordinary checkpointing, RoPE table reuse, full CE with
2048-position chunks, and CUDA graphs. RT, FBT, and NextLat objectives are off.

Complete updates took **892.00 ms median**, or **36,735 input tokens/s**.
Forward/loss/backward took **849.23 ms median**. The separate untimed profile
sums to **848.31 ms** across 3,660 device events. It excludes optimizer work;
its close agreement with timed forward/loss/backward supports that scope but
is not an independent complete-update throughput measurement.

These categories aggregate matching kernel names from `profile.device_kernels`:

| Kernel-name grouping | Summed device time | Fraction |
| --- | ---: | ---: |
| GEMMs (`nvjet_`/`gemm`) | 369.64 ms | 43.57% |
| Copy/cast (`copy_kernel_cuda`) | 88.80 ms | 10.47% |
| Fill (`FillFunctor`) | 51.20 ms | 6.04% |
| Concatenation (`CatArray`) | 43.23 ms | 5.10% |
| Explicit SiLU forward/backward | 39.18 ms | 4.62% |
| Flash attention (`pytorch_flash`) | 31.82 ms | 3.75% |
| Softmax/NLL | 27.17 ms | 3.20% |
| LayerNorm | 26.70 ms | 3.15% |
| Negation | 10.74 ms | 1.27% |

The table is intentionally incomplete. Kernel names identify operation families,
not exact source phases: copy/cast, fill, and arithmetic kernels serve multiple
call sites. These fractions are neither additive optimization promises nor
measured upper bounds on a proposed implementation's gain. Attribute suspect
work with a targeted, untimed profile before estimating savings.

## Ranked follow-ups after the current candidates

1. **Fuse the existing FP32 split-half RoPE application.**
   [Application source](../../../cdrm/pretrained/olmo_rope.py) currently performs
   a shared FP32 cast, split, negation, concatenation, two products, addition,
   and restoration of the Q/K projection dtype. The profile has 96 FP32
   concatenations, 96 negations, and 192 FP32 multiplies, consistent with Q/K
   rotation in forward, recomputation, and backward. Those named kernels alone
   total about 9.94%; attributing all of them to RoPE remains an inference.
   A small compiled helper or specialized kernel could avoid materializing
   intermediates while retaining native positional coordinates, FP32 rotation,
   and BF16 output boundaries. Preserve reference derivatives and validate
   rounding; fusion or changed accumulation can affect numerical agreement.

2. **Fuse residual addition, nonaffine LayerNorm, and projection-input casting.**
   The [ordinary block](../../../cdrm/pretrained/olmo.py) separately creates the
   attention residual, normalizes it, and lets the next projection cast its
   input. A helper could return the required FP32 residual plus the normalized
   projection input with less intermediate traffic. Retain FP32 residual and
   normalization arithmetic, native epsilon, and the absence of affine weights;
   this is not a proposal to add Q/K normalization or lower residual precision.
   Measure attribution first: the aggregate copy/add/LayerNorm buckets do not
   establish the saving available to this particular fusion.

3. **Reduce CE/readout traffic, starting with a bounded fused-CE comparison.**
   [_ce_chunk and _chunked_sum](../../../cdrm/pretrained/nextlat.py) project the
   complete 50,304-row vocabulary, convert logits to FP32, and checkpoint each
   chunk. Fused post-logit CE could reduce loss-buffer traffic, but explicit
   softmax/NLL is only 3.20% here, limiting what eliminating that work alone
   could save. A later fused projection/loss approach might also reduce
   recomputation or storage; it needs its own parameter-gradient, tied-weight,
   denominator, and full-vocabulary checks. Do not promise its benefit from
   the current kernel-name profile.

## Precision and cast cautions

The current compiled SwiGLU helper now uses the local Inductor option
`emulate_precision_casts=True`. Its bounded pretrained comparison passed with
bitwise outputs/loss and gradient relative L2 about 0.00313, plus exact own
CUDA-graph/optimizer checks. The earlier unrounded compiled result remains a
separate retained numerical failure. This demonstrates that preserving native
BF16 intermediate rounding can matter even for a small pointwise optimization;
it does not validate any future RoPE, normalization, or CE change.

Do not simply enable global autocast caching or share a single differentiable
BF16 readout-weight cast across all CE chunks. The
[static training path](../../../cdrm/pretrained/static_training.py) deliberately
disables autocast caching to avoid freezing stale weight copies in CUDA graphs.
Sharing a cast can also move gradient accumulation from FP32 into BF16. Any
future reuse must refresh from current weights and preserve the intended FP32
parameter-gradient accumulation, including tied embedding/readout ownership.

No follow-up optimization in this note has been implemented or benchmarked.

## Follow-up profile: rounded compiled SwiGLU

[Compiled B64/T512](../../../.runtime/olmo-ordinary-efficiency/compiled-b64-t512-r1/report.json)
completed at `18351ef35a39e62d8405277b4e70319355d236d5` with **39,221 tokens/s**
(835.48 ms complete update; 800.21 ms forward/loss/backward). The same-commit
[control r2](../../../.runtime/olmo-ordinary-efficiency/control-b64-t512-r2/report.json)
measured **36,789 tokens/s**, reproducing the earlier control's approximately
36.7k throughput; this pair indicates a **6.6%** complete-update gain.

The two profiled executions are from different source commits. The relevant
model change adds helper-local precision emulation to compiled SwiGLU; the eager
control arithmetic is unchanged. Their kernel comparison below is descriptive,
not a same-commit paired profile or exact attribution of the step-time saving.

| Profile grouping | Initial control | Rounded compiled |
| --- | ---: | ---: |
| Explicit ATen SiLU forward/backward | 39.18 ms, 48 calls | Absent |
| Explicit BF16 multiply kernels | 46.03 ms, 64 calls | Absent |
| Fused Triton SwiGLU forward/backward | Absent | 35.20 ms, 48 calls |
| GEMMs | 369.64 ms, 43.57% | 371.69 ms, 46.96% |
| Copy/cast | 88.80 ms, 10.47% | 89.37 ms, 11.29% |
| Flash attention | 31.82 ms, 3.75% | 32.35 ms, 4.09% |
| Summed device events | 848.31 ms, 3,660 events | 791.56 ms, 3,580 events |

The fused kernels are `triton_poi_fused_mul_silu_split_0` (32 calls, 16.46 ms)
and `triton_poi_fused_cat_mul_silu_silu_backward_split_0` (16 calls, 18.74 ms).
BF16 concatenations also fall from 32 to 16 calls; FP32 concatenations remain
at 96. Together these observations support the intended removal of separate
SwiGLU intermediates. Do not classify the fused backward kernel as a remaining
standalone SiLU operation merely because its name contains `silu_backward`.

Absolute GEMM, copy/cast, and Flash time remain similar while their fractions
increase as total work shrinks. That keeps native FP32 RoPE and residual/cast
traffic plausible next targets; none is cleared or assigned a guaranteed gain.
Separately, the B64 alternating-checkpoint attempt exhausted memory during
capture after three preparation updates. Smaller-batch follow-up is a distinct
capacity check, not evidence against the compiled SwiGLU result above.
