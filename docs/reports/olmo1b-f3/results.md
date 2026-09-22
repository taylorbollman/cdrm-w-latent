# F3 canonical static-layout graph training

Status: **passed**.

Original OLMo-1B step60000 (~252B tokens), RT at layer index0, native RoPE/LayerNorm and tied readout; no Q/K normalization change. BF16 mixed compute, FP32 parameters/gradients/Adam state, deterministic ordinary Flash SDPA. Native RT remains eager tiled math executed inside the captured graph.

These are bounded functionality and execution measurements, not quality or learning-speed results. Only the supplied configurations are covered; all-layer RT, cache continuation, distributed execution and graph serialization/checkpoint-resume are not cleared.

## Correctness

Graph versus prepared-eager uses the original relative-L2 <=1e-5 and maximum-error <=1e-6+1e-5×reference-max budgets. Bitwise equality is reported separately. Tiny FP32 canonical-versus-static tests are separate evidence. Each row includes changed tokens, repeated gradient overwrite, changed weights and complete AdamW/model/scheduler/counter comparisons.

| Run | Case | B/T | Ordinary checkpoint | Active gradient tensors | All gradients bitwise | Full update exact | Updates eager + graph |
|---|---|---:|---|---:|---|---|---:|
| [f3-combined-t32-cp-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xxg3frlo) | combined | 1/32 | on | 71 | True | True | 3 + 3 |
| [f3-rt-t32-plain-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/fdu3s697) | rt | 1/32 | off | 65 | True | True | 3 + 3 |
| [f3-rt-t32-cp-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yn535s8d) | rt | 1/32 | on | 65 | True | True | 3 + 3 |
| [f3-combined-t32-plain-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/90qxy38x) | combined | 1/32 | off | 71 | True | True | 3 + 3 |
| [f3-k3-t32-cp-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yvec05ye) | combined-k3 | 1/32 | on | 71 | True | True | 3 + 3 |
| [f3-rt-t512-b8-cp-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8z8qtl8y) | rt | 8/512 | on | 65 | True | True | 3 + 3 |
| [f3-combined-t512-b8-cp-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/iikfsf6u) | combined | 8/512 | on | 71 | True | True | 3 + 3 |

## Complete-update capacity

Capacity rows establish finite changing-input/changing-weight updates at their recorded shape; they do not substitute for all-gradient/full-state parity there. Three real preparation updates per arm initialize Adam; then ten or more backward warmups and three or more timed updates. Full-step wall times include validated input copies, forward/loss/backward or replay, clipping, AdamW and scheduler. Independently measured region timings exclude input copies and optimizer work. Input tokens/s counts one physical batch once, not internal FBT passes or CE-only targets.

| Case | B/T | CP | Eager / graph input tokens/s | Full-step speedup | Region speedup | Eager / graph peak allocated GiB | Eager / graph peak reserved GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| [rt](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/37mmlxmd) | 32/512 | on | 7,960 / 16,672 | 2.09× | 2.13× | 24.26 / 24.32 | 24.48 / 31.61 |
| [rt](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/pjl6z07b) | 64/512 | on | 12,914 / 21,553 | 1.67× | 1.68× | 30.18 / 30.25 | 30.62 / 43.67 |
| [rt](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rvprpiob) | 128/512 | on | 18,127 / 24,684 | 1.36× | 1.37× | 42.04 / 42.10 | 42.98 / 68.34 |
| [combined](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6s74unps) | 32/512 | on | 5,779 / 9,152 | 1.58× | 1.59× | 32.10 / 32.17 | 32.76 / 46.23 |
| [combined](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aao7lcwn) | 64/512 | on | 7,833 / 10,409 | 1.33× | 1.34× | 40.78 / 40.84 | 41.37 / 64.27 |
| [combined](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zfepyr1l) | 128/512 | on | 9,184 / 10,801 | 1.18× | 1.18× | 58.13 / 58.20 | 60.55 / 77.95 |

The raw capture_seconds field measures setup/warmup + capture, not just the CUDA graph-capture context. It wraps plan.capture, including backward warmup, synchronization and validation. Correctness setup also includes initial gradient discovery when not already initialized. This setup time is outside timed complete updates.

| Run | Arm | Setup/warmup + capture seconds | Current postcapture allocated / reserved GiB (eager: after warmup) | Setup/capture peak allocated / reserved GiB | Within comfort budget |
|---|---|---:|---:|---:|---|
| f3-capacity-rt-b32-01 | eager | — | 17.63 / 24.48 | 24.26 / 24.48 | True |
| f3-capacity-rt-b32-01 | graph | 22.164 | 17.69 / 24.53 | 24.32 / 31.61 | True |
| f3-capacity-rt-b64-01 | eager | — | 17.63 / 30.62 | 30.18 / 30.62 | True |
| f3-capacity-rt-b64-01 | graph | 27.268 | 17.69 / 31.12 | 30.25 / 43.67 | True |
| f3-capacity-rt-b128-01 | eager | — | 17.63 / 42.98 | 42.04 / 42.98 | True |
| f3-capacity-rt-b128-01 | graph | 38.366 | 17.69 / 43.59 | 42.10 / 68.34 | True |
| f3-capacity-combined-b32-01 | eager | — | 18.96 / 32.76 | 32.10 / 32.76 | True |
| f3-capacity-combined-b32-01 | graph | 30.276 | 19.02 / 33.01 | 32.17 / 46.23 | True |
| f3-capacity-combined-b64-01 | eager | — | 18.96 / 41.37 | 40.78 / 41.37 | True |
| f3-capacity-combined-b64-01 | graph | 44.446 | 19.02 / 43.33 | 40.84 / 64.27 | True |
| f3-capacity-combined-b128-01 | eager | — | 18.96 / 60.55 | 58.13 / 60.55 | True |
| f3-capacity-combined-b128-01 | graph | 74.477 | 19.02 / 64.85 | 58.20 / 77.95 | True |

Current postcapture memory and reserved-memory peaks are different measurements. Warmup cache can be released on graph entry, so peak reserved memory is not the continuing graph-pool footprint. The first table's peaks cover setup, complete updates and region timing; the second table separates current memory after setup from setup/capture high-water marks. None of these totals isolates graph-private memory from model/optimizer memory.

Memory-stop boundaries and incomplete pairs remain recorded; they do not become successful graph capacity points. Failed runs contribute no throughput values or plots. All timing samples, target counts, post-update health and memory fields remain in summary.json.

## Diagnostics and provenance

No failed diagnostic was supplied.

Reported successful-run physical optimizer executions: 114 (75 prepared-eager updates + 39 graph-replay updates). Grouping by comparison arm instead gives 57 eager-arm and 57 graph-arm updates. Each graph capacity arm begins with three eager preparation updates, followed by its timed graph updates; those preparation updates count as eager execution even though they belong to the graph comparison arm. Correctness arms replay the same short trajectory. Backward warmup, capture and region-only calls do not advance optimizer counters. Partial executions from failed runs are excluded from this successful-run total.

| Run | Status | Capture succeeded | Source version | W&B |
|---|---|---|---|---|
| f3-combined-t32-cp-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xxg3frlo) |
| f3-rt-t32-plain-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/fdu3s697) |
| f3-rt-t32-cp-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yn535s8d) |
| f3-combined-t32-plain-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/90qxy38x) |
| f3-k3-t32-cp-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yvec05ye) |
| f3-rt-t512-b8-cp-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8z8qtl8y) |
| f3-combined-t512-b8-cp-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/iikfsf6u) |
| f3-capacity-rt-b32-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/37mmlxmd) |
| f3-capacity-rt-b64-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/pjl6z07b) |
| f3-capacity-rt-b128-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rvprpiob) |
| f3-capacity-combined-b32-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6s74unps) |
| f3-capacity-combined-b64-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aao7lcwn) |
| f3-capacity-combined-b128-01 | passed | True | current hashes match | [run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zfepyr1l) |

Actual ordinary attention operator traces from correctness runs:

- f3-combined-t32-cp-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-rt-t32-plain-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-rt-t32-cp-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-combined-t32-plain-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-k3-t32-cp-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-rt-t512-b8-cp-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`
- f3-combined-t512-b8-cp-01: `aten::_flash_attention_backward`, `aten::_flash_attention_forward`, `aten::_scaled_dot_product_flash_attention`, `aten::_scaled_dot_product_flash_attention_backward`, `aten::scaled_dot_product_attention`

Native checkpoint SHA256: `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c` (4,707,065,440 bytes). Per-run source/report/protocol hashes and exact snapshot lineage are retained in summary.json. The results use no widened numerical budgets and make no fused RT-kernel claim.
