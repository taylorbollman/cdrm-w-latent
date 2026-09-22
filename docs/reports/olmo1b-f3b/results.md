# F3b native RT execution evidence

Status: **passed**.

Bounded functionality and execution measurements on original OLMo-1B step60000. Native-model rows use RT at layer index0, BF16 mixed compute with FP32 weights/gradients/Adam, ordinary Flash and activation checkpointing. No quality, all-layer RT, fused RT backward or multi-GPU result is implied.

## Evidence and correctness

| Run | Scope | Status | Declared checks passed / total |
| --- | --- | --- | ---: |
| [f3b-profile-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/0upabr5w) | profile | passed | 0 / 0 |
| [f3b-fa4-smoke-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hz3s0qpq) | fa4 | passed | 0 / 0 |
| [f3b-cast-combined-t32-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/c50071fp) | native | passed | 5 / 5 |
| [f3b-cast-combined-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ex8j7fjf) | native | passed | 5 / 5 |
| [f3b-capacity-cast-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/b51luda5) | native | passed | 1 / 1 |
| [f3b-tile-probe-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w6iejh4m) | tile | passed | 60 / 60 |
| [f3b-triton-combined-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4p5dww1c) | native | passed | 5 / 5 |
| [f3b-capacity-triton-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/gwnwd02r) | native | passed | 1 / 1 |
| [f3b-triton-rt-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/wowp6tah) | native | passed | 5 / 5 |
| [f3b-capacity-cast-rt-b128-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/g9mwb2u7) | native | passed | 1 / 1 |
| [f3b-capacity-triton-rt-b128-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/d533qiak) | native | passed | 1 / 1 |
| [f3b-profile-triton-combined-b64-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/d4fam7qe) | profile | passed | 0 / 0 |

Successful F3b reports record 60 physical optimizer updates. Backward-only warmup/capture/profile calls and historical F3 measurements are excluded from that count.

Declared top-level checks determine acceptance. Nested stricter or independent-oracle diagnostics remain visible in summary.json, including false flags; they are not silently promoted to passes or recursively substituted for the run's stated gate. Candidate/eager graph parity retains the stricter F3 tolerance. FP64 reduction can land on a different BF16 rounding tie, so independent-oracle state diagnostics are reported separately from the candidate/eager FP32-state gate.

- f3b-cast-combined-t32-01: cast_once, combined, B1/T32; reference comparison bitwise=True, candidate graph comparisons bitwise=True; three complete updates per arm match exactly.

- f3b-cast-combined-b8-t512-01: cast_once, combined, B8/T512; reference comparison bitwise=True, candidate graph comparisons bitwise=True; three complete updates per arm match exactly.

- f3b-triton-combined-b8-t512-01: triton, combined, B8/T512; reference comparison bitwise=False, candidate graph comparisons bitwise=True; three complete updates per arm match exactly.

- f3b-triton-rt-b8-t512-01: triton, rt, B8/T512; reference comparison bitwise=False, candidate graph comparisons bitwise=True; three complete updates per arm match exactly.

FA4 smoke f3b-fa4-smoke-01: forward/dQ/dK/dV recorded; maximum relative-L2 error 0.002395; capture output exact. This confirms standalone FA4 availability, not native RT kernel equivalence.

## Measured complete updates

Full-step input tokens/s includes input validation/copy, graph forward/loss/backward, clipping, AdamW and scheduler. It counts each input once; FBT passes do not multiply data exposure. Profile-run timing was collected before instrumentation. Capacity checks establish finite updates, not every-gradient parity at the larger batch. Reserved setup peaks and current reserved memory are distinct.

| Run / variant | Case | B/T | Input tokens/s | CE targets/s | Seconds/update | Peak allocated / reserved GiB | Current reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f3b-profile-combined-b64-01 / reference | combined | 64/512 | 10,409 | 5,204 | 3.1481 | 40.84 / — | — |
| f3b-capacity-cast-combined-b64-01 / cast_once | combined | 64/512 | 10,717 | 5,359 | 3.0575 | 40.84 / 63.84 | 43.19 |
| f3b-capacity-triton-combined-b64-01 / triton | combined | 64/512 | 10,871 | 5,436 | 3.0142 | 40.84 / 63.82 | 43.40 |
| f3b-capacity-cast-rt-b128-01 / cast_once | rt | 128/512 | 25,588 | 12,794 | 2.5612 | 42.10 / 68.30 | 44.31 |
| f3b-capacity-triton-rt-b128-01 / triton | rt | 128/512 | 26,101 | 13,051 | 2.5109 | 42.10 / 68.30 | 44.31 |
| f3b-profile-triton-combined-b64-01 / triton | combined | 64/512 | 10,866 | 5,433 | 3.0156 | 40.84 / — | — |
| f3-capacity-rt-b128-01 / F3 reference | rt | 128/512 | 24,684 | 12,342 | 2.6550 | 42.10 / 68.34 | 43.59 |

## Device attribution and helper timings

RT/* entries are observer annotation ranges, including CUDA-attributed ranges; they are not additional kernels. CPU and CUDA annotations must not be added together. Graph kernel self-time sums can overlap and are not wall-clock update times.

f3b-profile-combined-b64-01: 149,970 graph kernel invocations; aggregate recorded kernel self-device time 3.063s.

| Graph kernel (top 8 by recorded self-device time) | Count | Self-device ms |
| --- | ---: | ---: |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 12,959 | 466.844 |
| `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 132 | 256.508 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 5,599 | 219.150 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 387 | 197.323 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 7,319 | 170.678 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 25,640 | 167.794 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)` | 19,014 | 164.918 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | 98 | 132.924 |

| Observer annotation | Device attribution | Count | Inclusive CPU ms | Attributed device ms |
| --- | --- | ---: | ---: | ---: |
| RT/_finish | DeviceType.CUDA | 1,025 | 0.000 | 412.932 |
| RT/_add_tile | DeviceType.CUDA | 511 | 0.000 | 409.920 |
| RT/_project | DeviceType.CUDA | 1,027 | 0.000 | 148.803 |
| RT/_attention_from_completed | DeviceType.CUDA | 1 | 0.000 | 13.788 |
| RT/_project | DeviceType.CPU | 1,027 | 272.456 | 46.982 |
| RT/_finish | DeviceType.CPU | 1,025 | 502.533 | 165.383 |
| RT/_add_tile | DeviceType.CPU | 511 | 499.893 | 43.395 |
| RT/_attention_from_completed | DeviceType.CPU | 1 | 12.706 | 13.648 |

f3b-profile-triton-combined-b64-01: 128,637 graph kernel invocations; aggregate recorded kernel self-device time 2.935s.

| Graph kernel (top 8 by recorded self-device time) | Count | Self-device ms |
| --- | ---: | ---: |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 10,404 | 393.711 |
| `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 132 | 256.548 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 4,576 | 217.921 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 387 | 197.242 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 6,297 | 166.903 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 22,574 | 164.747 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)` | 15,692 | 141.399 |
| `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | 98 | 132.854 |

| Observer annotation | Device attribution | Count | Inclusive CPU ms | Attributed device ms |
| --- | --- | ---: | ---: | ---: |
| RT/_finish | DeviceType.CUDA | 1,025 | 0.000 | 381.656 |
| RT/_project | DeviceType.CUDA | 1,027 | 0.000 | 139.980 |
| RT/_add_tile | DeviceType.CUDA | 511 | 0.000 | 58.272 |
| RT/_attention_from_completed | DeviceType.CUDA | 1 | 0.000 | 14.014 |
| RT/_project | DeviceType.CPU | 1,027 | 264.109 | 30.375 |
| RT/_finish | DeviceType.CPU | 1,025 | 482.737 | 89.924 |
| RT/_add_tile | DeviceType.CPU | 511 | 98.078 | 1.766 |
| RT/_attention_from_completed | DeviceType.CPU | 1 | 12.905 | 13.873 |

Helper CUDA-event elapsed time covers an uncaptured historical-tile forward call. The interval includes host submission gaps while Python dispatches kernels, so it is not isolated kernel latency or device compute time. It excludes RT projection/MLP/backward, full loss and optimizer work, and cannot be presented as full-model throughput.

| Helper case | Source / target / head width | Eager / fused CUDA-event microseconds, including host gaps |
| --- | ---: | ---: |
| tile_w32_nonzero | 32 / 31 / 16 | 353.2 / 80.6 |
| tile_w128_nonzero | 128 / 127 / 64 | 370.9 / 82.6 |
| tile_w256_nonzero | 256 / 255 / 128 | 365.0 / 80.1 |

## Common analytic resource cards

All eight rows use B64/T512, ordinary checkpointing, one unpadded document per row, CE=16,384; enabled NextLat has 32,704 latent/predictor positions and 16,384 KL triples. These are analytic matrix FLOP estimates, not eight throughput experiments. Counts include permanent full-QKV writes, custom backward reconstruction, pass work, fusion, predictor and checkpointed full-vocabulary readouts. Multiply-add=2; optimizer, pointwise/RoPE/softmax, launch overhead and hardware padding are excluded. The range covers ordinary attention and checkpoint early-stop assumptions. Shared/tied weights count once; inactive resident modules require the actual ownership inventory. See [derivation](../../olmo-resource-accounting.md).

| Architecture | Training parameters | Deployable inference parameters | Matrix TFLOPs/update |
| --- | ---: | ---: | ---: |
| Ordinary | 1,176,764,416 | 1,176,764,416 | 281.8–304.9 |
| Ordinary + NextLat | 1,259,491,328 | 1,176,764,416 | 314.9–338.0 |
| RT layer0 | 1,176,764,416 | 1,176,764,416 | 294.9–316.5 |
| RT layer0 + NextLat | 1,259,491,328 | 1,176,764,416 | 328.0–349.6 |
| Ordinary + FBT K2 | 1,185,153,024 | 1,185,153,024 | 565.2–611.4 |
| Ordinary + FBT K2 + NextLat | 1,267,879,936 | 1,185,153,024 | 631.5–677.6 |
| RT layer0 + FBT K2 | 1,185,153,024 | 1,185,153,024 | 578.3–623.0 |
| RT layer0 + FBT K2 + NextLat | 1,267,879,936 | 1,185,153,024 | 644.5–689.3 |

## Failures, incomplete scope and provenance

No failed diagnostic was supplied.

summary.json pins each input report and verifies recorded source versions against exact snapshots or matching current files. Historical source versions remain distinct. FA4's external installed interface hash is recorded provenance; it is not revalidated against a different host installation. No run is discovered automatically and missing experiments remain untested.
