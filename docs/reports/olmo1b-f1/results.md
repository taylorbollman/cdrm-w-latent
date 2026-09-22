# F1: bounded integration and early execution profile

**18/18 configured cases passed** within the explicitly requested scope (full_configured_matrix). This checks functionality and execution; it does not establish language-model quality.

Original OLMo-1B step60,000 (~252B tokens), native model revision `81b71efbce6f4dada57c94860301af4298bcd351`; checkpoint SHA256 `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`. 16 layers, width2048, 16 heads, SwiGLU8192, tied50304-row embedding/readout, native RoPE and non-affine LayerNorm; no Q/K normalization.

BF16 mixed compute / FP32 parameters, gradients and AdamW moments; LR1e-05. Runtime: 2.13.0a0+8145d630e8.nv26.06, CUDA13.3, NVIDIA H100 80GB HBM3. [W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8patkvpi).

44 original observed updates + 12 warmup + 12 timed + 4 separately profiled = **72 retained-counter updates**. 2 additional physical replays reproduce already counted updates, for **74 optimizer executions** overall.

| Case | B / T | RT layers | FBT passes | NextLat | Observed updates | Resume / online cache |
| --- | ---: | --- | ---: | --- | ---: | --- |
| ordinary | 2 / 32 | off | off | off | 3 | passed / untested |
| nextlat | 2 / 32 | off | off | on | 3 | untested / untested |
| rt | 2 / 32 | [0] | off | off | 3 | untested / untested |
| rt-nextlat | 2 / 32 | [0] | off | on | 3 | untested / untested |
| fbt | 2 / 32 | off | 2 | off | 3 | untested / untested |
| fbt-nextlat | 2 / 32 | off | 2 | on | 3 | untested / untested |
| rt-fbt | 2 / 32 | [0] | 2 | off | 3 | untested / untested |
| rt-fbt-nextlat | 2 / 32 | [0] | 2 | on | 3 | passed / passed |
| all-three-k3 | 2 / 32 | [0] | 3 | on | 3 | untested / untested |
| all-three-fractional | 2 / 32 | [0] | 2 | on | 3 | untested / untested |
| all-three-two-rt-layers | 2 / 32 | [0, 15] | 2 | on | 3 | untested / untested |
| all-three-transition | 2 / 32 | [0] | 2 | on | 3 | untested / untested |
| ordinary-t128 | 2 / 128 | off | off | off | 2 | untested / untested |
| all-three-t128 | 2 / 128 | [0] | 2 | on | 2 | untested / untested |
| ordinary-t512-profile | 1 / 512 | off | off | off | 1 | untested / untested |
| rt-t512-profile | 1 / 512 | [0] | off | off | 1 | untested / untested |
| fbt-t512-profile | 1 / 512 | off | 2 | off | 1 | untested / untested |
| all-three-t512-profile | 1 / 512 | [0] | 2 | on | 1 | untested / untested |

All listed cases passed expected gradient participation/finite checks, active-weight changes, inactive-state invariance, finite parameters/optimizer state and endpoint losses. Exact resume includes loaded state, cursor, next fixture, next update, metrics and CPU/CUDA RNG. The online cache check is B1/T8 whole-versus-split plus incompatible-mode rejection.

K includes the ordinary bootstrap; RT applies to extra FBT passes. Alpha/beta are1 unless the case is fractional (.37/.35) or transitioning through(0/0),(.37/.35),(1/1). Objective: pass0 + mean(extra-pass losses), with unit CE and, when enabled, latent/KL weights. CE and KL use response positions; latent regression uses valid pairs.

| Parameter scope (unique tensors, tying counted once) | Registered/resident | Requires grad | Observed active | Optimizer owned | Inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary, rt, ordinary-t128, ordinary-t512-profile, rt-t512-profile | 1,185,153,024 | 1,176,764,416 | 1,176,764,416 | 1,176,764,416 | 1,176,764,416 |
| nextlat, rt-nextlat | 1,267,879,936 | 1,259,491,328 | 1,259,491,328 | 1,259,491,328 | 1,176,764,416 |
| fbt, rt-fbt, fbt-t512-profile | 1,185,153,024 | 1,185,153,024 | 1,185,153,024 | 1,185,153,024 | 1,185,153,024 |
| fbt-nextlat, rt-fbt-nextlat, all-three-k3, all-three-fractional, all-three-two-rt-layers, all-three-transition, all-three-t128, all-three-t512-profile | 1,267,879,936 | 1,267,879,936 | 1,267,879,936 | 1,267,879,936 | 1,185,153,024 |

FBT-off models retain frozen fusion matrices in resident memory. NextLat-off models omit the predictor; NextLat inference excludes its training-only predictor. RT adds no parameters. Activity is the union of backward participation across observed updates, including intentionally zero contributions; transition-case fusion is inactive at beta0.

| Early complete-update profile | B / T | Wall median (s) | CUDA median (s) | Input tokens/s | CE targets/s | Peak allocated / reserved GiB | Actual ordinary dispatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ordinary-t512-profile | 1 / 512 | 0.0676 | 0.0676 | 7,571 | 3,786 | 18.40 / 20.41 | cuDNN SDPA |
| rt-t512-profile | 1 / 512 | 1.6266 | 1.6266 | 315 | 157 | 18.40 / 20.33 | cuDNN SDPA |
| fbt-t512-profile | 1 / 512 | 0.1043 | 0.1043 | 4,910 | 2,455 | 18.68 / 21.95 | cuDNN SDPA |
| all-three-t512-profile | 1 / 512 | 1.6803 | 1.6802 | 305 | 152 | 20.40 / 23.59 | cuDNN SDPA |

Largest recorded exclusive CPU/device operators in the separate profiler update:

- ordinary-t512-profile: CPU `cudaStreamSynchronize` 30.6ms; `cudaLaunchKernel` 8.9ms; device `Optimizer.step#AdamW.step` 32.0ms; `aten::mul_` 10.1ms.
- rt-t512-profile: CPU `_TiledRecurrenceBackward` 399.5ms; `cudaLaunchKernel` 302.9ms; device `aten::mm` 90.3ms; `aten::copy_` 44.2ms.
- fbt-t512-profile: CPU `cudaStreamSynchronize` 30.8ms; `cudaLaunchKernel` 15.7ms; device `Optimizer.step#AdamW.step` 32.2ms; `aten::mm` 12.6ms.
- all-three-t512-profile: CPU `_TiledRecurrenceBackward` 381.0ms; `cudaLaunchKernel` 304.6ms; device `aten::mm` 99.4ms; `aten::copy_` 49.9ms.

Profiler durations include instrumentation overhead and are not synchronized wall-time attribution. Device lists can mix CPU-attributed scopes and raw CUDA kernels representing the same work; do not add mixed rows or interpret their sum as a fraction of total GPU time. Generic SDPA is not a backend identity; cuDNN fused attention does not mean use of the flash-attn package/FA4. Native RT still uses eager tiles/custom backward; ordinary fused operators do not imply fused RT.

Scope and remaining limitations:

- Operational fixtures from two prompt strings with deterministic token rotations, EOS, response masks and right padding; no language-quality or learning-efficiency claim.
- BF16 mixed compute with FP32 parameters, gradients and moments; historical independent tiny math checks remain separate evidence. This is not a new broad FP32 comparison.
- Selected native RT uses eager dyadic tiling and explicit custom VJP. Fused ordinary SDPA events do not establish Flash/CuTE RT execution; current RT backward has quadratic attention intermediates.
- FBT K counts total shared-stack passes including ordinary pass0; RT applies to extra FBT passes. Standalone RT is a single recurrent pass. NextLat is training-only.
- Directional timings use complete nonzero-LR updates after warmup, excluding fixture construction, hooks, hashes, checkpoint/W&B I/O and the separate profiler update. They are not optimized capacity, final FLOP or scaling measurements.
- All-three changes pass count, recurrence and auxiliary training together. Throughput differences do not isolate any one feature.
- Profiler key averages can include both CPU-attributed device durations and raw CUDA kernels for the same work. Do not sum those mixed rows or convert them into fractions of total GPU time.
- No compile, CUDA graphs, Q/K normalization or distributed execution. All-16-layer RT and full-length online combined execution remain untested.
- Resume replays reproduce an already counted update after restoring an earlier boundary; count those physical executions separately from retained training counters.

Machine-readable case scopes, ownership, profiler evidence, exact replay details, counters and frozen source/version hashes are in [capability-ledger.json](capability-ledger.json).
