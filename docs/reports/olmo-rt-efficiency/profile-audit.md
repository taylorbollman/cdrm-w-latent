# RT efficiency profile audit

The two retained profiles cover **one untimed captured forward/loss/backward**
at B64/T512 with two recurrent layers (0/15) in the 16-layer native model,
full next-token CE2048, and the fixed Stage A execution settings. The control
uses on-demand RoPE and permanent QKV; the candidate enables both optimizations.
Neither profile includes clipping, AdamW or the scheduler. Profiling occurred
after the separate complete-update timing samples.

Sources: [control report](../../../.runtime/olmo-rt-efficiency/rt-b64-control/report.json)
and [candidate report](../../../.runtime/olmo-rt-efficiency/rt-b64-both/report.json).
Each contains the trace filename, byte count and SHA256, plus runtime/source
provenance. The compressed Chrome traces are alongside their reports.

## Method

Sum only CUDA device-event self durations and call counts, grouped by kernel
name. The control Chrome trace confirms **179,615 kernel events plus 437 GPU
memset events**, totaling the report's 180,052 GPU events. These are leaf GPU
events from one captured graph/stream; CPU `cudaGraphLaunch` duration, flow
arrows and enclosing trace annotations are excluded. Do not add those enclosing
durations to kernel times. The candidate Chrome trace independently confirms
145,849 kernels plus 435 GPU memsets, totaling 146,284 events on one graph/stream.
Its per-name counts match the report after accounting for the generic Triton
name's display capitalization (`Kernel` in Chrome, `kernel` in the report).
The summed leaf durations agree with the report aggregates in both profiles.

The categories below are exclusive. GEMMs are `nvjet_*`/GEMM names; copies
include direct/dtype copies, concatenation kernels and copy helpers; fills are
`FillFunctor` kernels. Add/multiply groups use the corresponding functor names.
RoPE trigonometry uses only `cos_kernel_cuda`/`sin_kernel_cuda`. The frequency
family groups power, reciprocal and arange names; 134 non-RoPE arange calls
remain in the candidate. LayerNorm, SiLU, ordinary Flash and CE softmax/NLL
have explicit names. Unclassified work stays in `Other`.

The generic Triton name `kernel` is attributed to historical forward tiles
using its definition in `olmo_rt_kernels.py` and the matching 1,022-call count,
equal to `2 × (512−1)`. Historical backward is explicitly named
`historical_recomputed_backward_kernel`, also with 1,022 calls. This limited
source/count attribution is distinct from inferring projection shapes from
shared GEMM names.

## GPU-event comparison

Durations are summed device milliseconds from one profile per arm, **not a
decomposition of the separately measured complete-update wall time**.

| Exclusive event family | Control calls | Both calls | Control ms | Both ms |
| --- | ---: | ---: | ---: | ---: |
| GEMMs | 12,752 | 12,752 | 559.527 | 549.219 |
| Copies / conversions / concatenations | 41,526 | 37,304 | 241.518 | 233.807 |
| Multiplies | 21,301 | 14,955 | 144.558 | 137.301 |
| Fills | 56,140 | 43,518 | 119.023 | 106.876 |
| Adds | 14,930 | 14,930 | 118.588 | 118.821 |
| LayerNorm | 6,244 | 6,244 | 50.911 | 50.779 |
| Other | 10,673 | 10,673 | 48.645 | 48.680 |
| SiLU | 3,118 | 3,118 | 47.792 | 47.879 |
| Ordinary Flash | 70 | 70 | 27.751 | 27.817 |
| CE softmax / NLL | 96 | 96 | 27.125 | 27.178 |
| Historical RT Triton tiles | 2,044 | 2,044 | 26.304 | 26.381 |
| Power / reciprocal / arange | 6,482 | 134 | 8.803 | 0.123 |
| Cosine / sine | 4,228 | 0 | 8.055 | 0.000 |
| GPU memsets and memset helper kernels | 448 | 446 | 0.361 | 0.361 |
| **Total** | **180,052** | **146,284** | **1,428.963** | **1,375.222** |

The candidate removes 33,768 GPU events (18.75%) and 53.741 ms of summed
device time in these profiles. All 2,114 cosine and 2,114 sine launches disappear.
Power and reciprocal each lose 2,116 launches; one arange variant also loses
2,116. Their combined named-family reduction is 16.736 ms. Precomputation
occurred before capture, so this confirms repeated table construction has left
the replay; it does not imply trigonometric setup is free.

The remaining decrease appears mainly in fills, copies, multiplication and
GEMMs. Generic pointwise/copy kernels serve many operations, and GEMM names do
not identify a unique caller or tensor shape in a captured replay. Therefore
the trace does **not** establish that every removed fill/copy was RoPE-related
or that all 10.309 ms of GEMM reduction came from the permanent writer. The
independent executed-matrix accounting establishes the writer's arithmetic
reduction; this profile measures its combined execution context.

Historical forward tiles take 9.074 versus 9.075 ms; historical backward tiles
take 17.230 versus 17.306 ms. Ordinary Flash similarly changes very little.
Thus this trace does not implicate those attention kernels as the source of
the measured improvement. Dense projections and generic pointwise/memory work
still account for most device duration. This is an integrated two-RT-layer
observation, not an all-RT model or single-block bottleneck determination.

## Relation to the timing measurements

| First matched B64 measurement | Complete update ms | Forward/loss/backward replay ms | Input tokens/sec |
| --- | ---: | ---: | ---: |
| Control | 1,524.427 | 1,480.205 | 21,495.30 |
| RoPE reuse | 1,474.620 | 1,431.309 | 22,221.31 |
| Both | 1,460.733 | 1,416.604 | 22,432.57 |

These medians come from separate unprofiled samples. There is no RoPE-only
profile in this pair, so exact profile attribution between the two optimizations
is unavailable. A second timing round in reverse arm order produced control
21,479.26, RoPE-only 22,210.11 and both 22,417.47 input tokens/sec. Medians of the
two run medians are 21,487.28 / 22,215.71 / 22,425.02 respectively: approximately
3.39% for RoPE reuse, 4.36% for both versus control, and 0.94% incremental gain
from KV-only versus RoPE-only. The small KV benefit repeats directionally; two
bounded runs do not establish statistical significance or sustained training
throughput. These profiles explain observed work without replacing the repeated
complete-update measurements or numerical checks.
