# F3c historical RT backward fusion

Status: **passed**.

Primary reference: F3b fused historical forward tiles and local BF16 weight-cast reuse, with the existing eager backward. The candidate changes only historical dK/dV tile execution. Native checkpoint, forward recurrence, parameters, RoPE, Q/K treatment and canonical FBT/NextLat losses stay fixed.

This is a bounded functionality/execution result, not a quality or all-layer RT result. Full-model checks select RT layer0 only. **Global probability/error matrices remain materialized: this milestone does not remove quadratic backward storage.** Multi-GPU, longer/padded graphs and graph resume remain separate.

## Evidence and numerical checks

| Run | Scope | Status | Declared passed / total |
| --- | --- | --- | ---: |
| [f3c-profile-reference-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ev8anieg) | profile | passed | 3 / 3 |
| [f3c-tile-probe-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4r1se0op) | tile | passed | 60 / 60 |
| [f3c-triton-combined-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/07awt17o) | native | passed | 5 / 5 |
| [f3c-triton-rt-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6t9h3z4x) | native | passed | 5 / 5 |
| [f3c-capacity-triton-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/3exnf0cz) | native | passed | 1 / 1 |
| [f3c-capacity-triton-rt-b128-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hubes8cx) | native | passed | 1 / 1 |
| [f3c-profile-triton-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/myh5cmwd) | profile | passed | 3 / 3 |

Completed successful F3c runs contain 36 physical optimizer updates: 18 eager and 18 graph. Backward-only warmup/capture/profiles and historical F3b updates are excluded. Profile observer neutrality is checked bitwise for losses and all active gradients.

Primary gradient screens are global relative L2<=1/64, per tensor<=1/32, maximum error/reference maximum<=1/16. An exact-zero reference requires exact zero. Same-candidate eager/graph uses stricter F3 limits; three full AdamW updates per arm must match exactly. Stricter coordinate flags and original-eager/full-FP32 controls remain visible diagnostics rather than silently replacing the declared primary gate.

| Native case | B/T | Initial losses bitwise | Global gradient relative L2 | Max tensor relative L2 | Max error/reference max | Graph comparisons bitwise | Full updates exact |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| f3c-triton-combined-b8-t512-01 | 8/512 | True | 0.00139246 | 0.00272988 | 0.00694444 | True | True |
| f3c-triton-rt-b8-t512-01 | 8/512 | True | 0 | 0 | 0 | True | True |

f3c-tile-probe-01: 48 frozen backward rectangles and 12 tiny native blocks passed their requested gates. Primary block forward/cache are bitwise equal, every expected fused call is observed, and raw hidden/exported-KV cotangents include masked attached prefixes and irregular lengths.

## Measured complete updates

Full-step wall timing includes validated input copy, graph forward/loss/backward, clipping, AdamW and scheduler. Input tokens count a physical batch once, not K times for FBT passes. Profile-run timings were collected before instrumentation. Peak allocated/reserved and current reserved memory have different meanings; none isolates graph-private memory or establishes a quadratic-storage improvement.

| Run | Backward variant | Case | B/T | Input tokens/s | Seconds/update | Peak allocated / reserved GiB | Current reserved GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f3c-profile-reference-combined-b64-01 | eager | combined | 64/512 | 10,858 | 3.0178 | 40.84 / 63.82 | 43.40 |
| f3c-capacity-triton-combined-b64-01 | fused | combined | 64/512 | 10,973 | 2.9862 | 40.84 / 63.82 | 43.40 |
| f3c-capacity-triton-rt-b128-01 | fused | rt | 128/512 | 26,480 | 2.4749 | 42.10 / 68.30 | 44.31 |
| f3c-profile-triton-combined-b64-01 | fused | combined | 64/512 | 10,977 | 2.9851 | 40.84 / 63.82 | 43.40 |
| f3b-capacity-triton-rt-b128-01 | historical eager | rt | 128/512 | 26,101 | 2.5109 | 42.10 / 68.30 | 44.31 |

## Backward attribution

Observer RT/* spans are inclusive and can overlap primals, VJPs, kernels and host gaps. CPU and CUDA annotation views cannot be added together. Actual graph-kernel self-time sums are separate from wall time and can include overlapping execution. Local writer/finish VJPs, batched parameter VJP, historical backward tiles, complete attention reconstruction and four final attention matmuls are labeled separately. CPU observer child-device time can undercount directly launched Triton kernels; use actual CUDA kernel events and full-update timing for performance comparisons, not ratios of those observer values.

f3c-profile-reference-combined-b64-01: 128,637 graph kernel invocations; summed kernel self-device time 2.9361s. Observer neutrality passed.

| Observer span | Attribution | Calls | Inclusive CPU ms | Attributed device ms |
| --- | --- | ---: | ---: | ---: |
| RT/primal_finish | DeviceType.CUDA | 1,025 | 0.000 | 359.950 |
| RT/reverse_historical_tile | DeviceType.CUDA | 511 | 0.000 | 295.385 |
| RT/local_writer_vjp | DeviceType.CUDA | 512 | 0.000 | 280.243 |
| RT/local_finish_vjp | DeviceType.CUDA | 512 | 0.000 | 232.490 |
| RT/primal_project | DeviceType.CUDA | 1,027 | 0.000 | 138.511 |
| RT/forward_historical_tile | DeviceType.CUDA | 511 | 0.000 | 58.491 |
| RT/batched_parameter_vjp | DeviceType.CUDA | 1 | 0.000 | 27.019 |
| RT/attention_reconstruction | DeviceType.CUDA | 1 | 0.000 | 13.831 |
| RT/final_attention_matmul | DeviceType.CUDA | 4 | 0.000 | 3.654 |
| RT/primal_project | DeviceType.CPU | 1,027 | 255.496 | 30.410 |
| RT/primal_finish | DeviceType.CPU | 1,025 | 454.503 | 89.678 |
| RT/forward_historical_tile | DeviceType.CPU | 511 | 97.402 | 1.768 |
| RT/attention_reconstruction | DeviceType.CPU | 1 | 12.819 | 13.701 |
| RT/local_writer_vjp | DeviceType.CPU | 512 | 377.424 | 27.871 |
| RT/local_finish_vjp | DeviceType.CPU | 512 | 321.482 | 40.905 |
| RT/reverse_historical_tile | DeviceType.CPU | 511 | 368.681 | 37.854 |
| RT/final_attention_matmul | DeviceType.CPU | 4 | 0.644 | 3.627 |
| RT/batched_parameter_vjp | DeviceType.CPU | 1 | 3.080 | 26.846 |

| Actual graph kernel, top 8 | Calls | Self-device ms |
| --- | ---: | ---: |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 10,404 | 393.732 |
| `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 132 | 256.472 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 4,576 | 217.951 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 387 | 197.335 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 6,297 | 166.574 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 22,574 | 164.783 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)` | 15,692 | 141.363 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | 98 | 132.857 |

f3c-profile-triton-combined-b64-01: 117,205 graph kernel invocations; summed kernel self-device time 2.9090s. Observer neutrality passed.

| Observer span | Attribution | Calls | Inclusive CPU ms | Attributed device ms |
| --- | --- | ---: | ---: | ---: |
| RT/primal_finish | DeviceType.CUDA | 1,025 | 0.000 | 361.761 |
| RT/local_writer_vjp | DeviceType.CUDA | 512 | 0.000 | 284.037 |
| RT/local_finish_vjp | DeviceType.CUDA | 512 | 0.000 | 225.155 |
| RT/primal_project | DeviceType.CUDA | 1,027 | 0.000 | 139.085 |
| RT/forward_historical_tile | DeviceType.CUDA | 511 | 0.000 | 60.987 |
| RT/reverse_historical_tile | DeviceType.CUDA | 511 | 0.000 | 54.524 |
| RT/batched_parameter_vjp | DeviceType.CUDA | 1 | 0.000 | 27.018 |
| RT/attention_reconstruction | DeviceType.CUDA | 1 | 0.000 | 13.980 |
| RT/final_attention_matmul | DeviceType.CUDA | 4 | 0.000 | 3.656 |
| RT/primal_project | DeviceType.CPU | 1,027 | 256.607 | 30.203 |
| RT/primal_finish | DeviceType.CPU | 1,025 | 460.053 | 89.444 |
| RT/forward_historical_tile | DeviceType.CPU | 511 | 101.009 | 1.769 |
| RT/attention_reconstruction | DeviceType.CPU | 1 | 12.845 | 13.849 |
| RT/local_writer_vjp | DeviceType.CPU | 512 | 389.197 | 27.868 |
| RT/local_finish_vjp | DeviceType.CPU | 512 | 330.724 | 40.793 |
| RT/reverse_historical_tile | DeviceType.CPU | 511 | 86.045 | 1.565 |
| RT/final_attention_matmul | DeviceType.CPU | 4 | 0.632 | 3.627 |
| RT/batched_parameter_vjp | DeviceType.CPU | 1 | 2.988 | 26.848 |

| Actual graph kernel, top 8 | Calls | Self-device ms |
| --- | ---: | ---: |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 9,893 | 392.861 |
| `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 132 | 256.565 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 4,576 | 218.016 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 387 | 197.202 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 22,063 | 164.066 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 4,764 | 161.350 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)` | 12,114 | 137.074 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | 98 | 132.889 |

Backward helper timings are uncaptured CUDA-event intervals, including host submission gaps. They are not isolated kernel latency, full backward time or model throughput.

| Rectangle | Source / target / head width | Eager / fused event microseconds |
| --- | ---: | ---: |
| backward_tile_w32_nonzero | 32 / 31 / 16 | 264.3 / 62.2 |
| backward_tile_w128_nonzero | 128 / 127 / 64 | 262.7 / 60.8 |
| backward_tile_w256_nonzero | 256 / 255 / 128 | 251.4 / 194.2 |

## Accounting, failures and provenance

Shared parameter counts and matrix FLOP formulas remain those in [resource accounting](../../olmo-resource-accounting.md). Fusion changes scheduling and intermediate storage, not the three complete historical dK/dV matrix products. The existing analytic range excludes pointwise work, optimizer arithmetic, host launch gaps and hardware padding.

No failed diagnostic was supplied.

summary.json retains declared checks, nested stricter diagnostic flags, exact per-run source/protocol lineage, timing samples and target counts. Failed variants are not promoted and historical F3b source versions remain distinct. Only explicitly supplied runtime directories are summarized.
