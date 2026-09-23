# F3e multiple selected RT layers

Status: **passed**; primary-layout status: **passed**.

Statuses describe the supplied completed runs, not automatic clearance of every planned case. These are functionality and execution measurements. Two/four-layer layouts are the primary integration scope; all16 is an optional stress case, not an assumed main architecture. No quality, placement-superiority, native FA4 or multi-GPU claim follows.

| Run | Case / layout | RT indices | Role | B/T | Stage | Passed / declared |
| --- | --- | --- | --- | ---: | --- | ---: |
| [f3e-recompute-combined-spread2-b1-t32-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/dgnuec4q) | combined / spread2 | [0, 15] | primary_multi_layer_integration | 1/32 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-adjacent2-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6ah9zxj4) | combined / adjacent2 | [0, 1] | primary_multi_layer_integration | 8/512 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-spread2-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w3ukvlle) | combined / spread2 | [0, 15] | primary_multi_layer_integration | 8/512 | correctness (passed) | 5/5 |
| [f3e-recompute-rt-spread2-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ku9zhzsl) | rt / spread2 | [0, 15] | primary_multi_layer_integration | 8/512 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-spread4-b4-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/srv816ia) | combined / spread4 | [0, 5, 10, 15] | primary_multi_layer_integration | 4/512 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-k3-spread2-b1-t32-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zz37c2sy) | combined-k3 / spread2 | [0, 15] | primary_multi_layer_integration | 1/32 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-spread2-b1-t2048-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/a8trttyh) | combined / spread2 | [0, 15] | primary_multi_layer_integration | 1/2048 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-single-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/dfpczp34) | combined / single | [0] | single_layer_reference | 64/512 | capacity (passed) | 1/1 |
| [f3e-recompute-combined-adjacent2-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/t6qvuwgc) | combined / adjacent2 | [0, 1] | primary_multi_layer_integration | 64/512 | capacity (passed) | 1/1 |
| [f3e-recompute-combined-spread2-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/lxs56eez) | combined / spread2 | [0, 15] | primary_multi_layer_integration | 64/512 | capacity (passed) | 1/1 |
| [f3e-recompute-combined-spread4-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mdnb07ac) | combined / spread4 | [0, 5, 10, 15] | primary_multi_layer_integration | 64/512 | capacity (passed) | 1/1 |
| [f3e-recompute-rt-spread2-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4e5cypqz) | rt / spread2 | [0, 15] | primary_multi_layer_integration | 64/512 | capacity (passed) | 1/1 |
| [f3e-recompute-rt-spread2-b128-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ty6gzibh) | rt / spread2 | [0, 15] | primary_multi_layer_integration | 128/512 | capacity (passed) | 1/1 |
| [f3e-recompute-combined-spread2-b8-t2048-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hmwy4z58) | combined / spread2 | [0, 15] | primary_multi_layer_integration | 8/2048 | capacity (passed) | 1/1 |
| [f3e-recompute-combined-all16-b1-t32-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/szexamab) | combined / all16 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | optional_all_layer_stress | 1/32 | correctness (passed) | 5/5 |
| [f3e-recompute-combined-all16-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kzuex5hy) | combined / all16 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | optional_all_layer_stress | 8/512 | capacity (passed) | 1/1 |

Successful runs contain 96 physical optimizer updates (48 eager + 48 graph). Warmup/backward-only work and failed attempts are excluded.

## Correctness

Initial materialized-versus-recompute gradients retain global relative L2 ≤1/64, per tensor ≤1/32 and maximum error/reference maximum ≤1/16. Zero references require exact zero. Same-candidate graph checks retain tighter budgets; full Adam/state comparisons require exact parity. Stricter diagnostic flags remain in summary.json.

| Run | Initial loss exact | Global gradient relative L2 | Recompute tiles | Graph exact | Full Adam exact |
| --- | --- | ---: | ---: | --- | --- |
| f3e-recompute-combined-spread2-b1-t32-01 | True | 0 | 62 | True | True |
| f3e-recompute-combined-adjacent2-b8-t512-01 | True | 0.00462952 | 1022 | True | True |
| f3e-recompute-combined-spread2-b8-t512-01 | True | 0.00729182 | 1022 | True | True |
| f3e-recompute-rt-spread2-b8-t512-01 | True | 0.00591965 | 1022 | True | True |
| f3e-recompute-combined-spread4-b4-t512-01 | True | 0.0064827 | 2044 | True | True |
| f3e-recompute-combined-k3-spread2-b1-t32-01 | True | 0 | 124 | True | True |
| f3e-recompute-combined-spread2-b1-t2048-01 | True | 0.00747728 | 4094 | True | True |
| f3e-recompute-combined-all16-b1-t32-01 | True | 0 | 496 | True | True |

## Complete-update resources

Wall time includes input copy/validation, graph forward/loss/backward, clipping, AdamW and scheduler. Three-update medians are directional. Peak allocated includes setup; reserved peak and current reserved are distinct. Matrix FLOPs are an analytic ledger excluding elementwise/optimizer/communication work and kernel padding, not hardware utilization.

| Run | Case / layout | B/T | Input tokens/s | CE targets/s | Seconds/update | Allocated / reserved peak / current GiB | Matrix TFLOPs/update |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f3e-recompute-combined-single-b64-t512-01 | combined / single | 64/512 | 10,933 | 5,467 | 2.9972 | 39.093 / 60.865 / 41.842 | 644.65–689.37 |
| f3e-recompute-combined-adjacent2-b64-t512-01 | combined / adjacent2 | 64/512 | 9,895 | 4,948 | 3.3115 | 39.091 / 60.330 / 41.432 | 657.84–701.12 |
| f3e-recompute-combined-spread2-b64-t512-01 | combined / spread2 | 64/512 | 9,889 | 4,945 | 3.3134 | 39.094 / 60.455 / 41.377 | 657.84–701.12 |
| f3e-recompute-combined-spread4-b64-t512-01 | combined / spread4 | 64/512 | 8,310 | 4,155 | 3.9433 | 39.094 / 60.848 / 41.664 | 684.23–724.62 |
| f3e-recompute-rt-spread2-b64-t512-01 | rt / spread2 | 64/512 | 19,445 | 9,723 | 1.6851 | 32.247 / 47.963 / 34.500 | 308.18–328.38 |
| f3e-recompute-rt-spread2-b128-t512-01 | rt / spread2 | 128/512 | 22,720 | 11,360 | 2.8844 | 46.106 / 76.797 / 49.039 | 616.36–656.75 |
| f3e-recompute-combined-spread2-b8-t2048-01 | combined / spread2 | 8/2048 | 4,708 | 2,354 | 3.4801 | 31.289 / 44.432 / 33.254 | 342.96–380.06 |
| f3e-recompute-combined-all16-b8-t512-01 | combined / all16 | 8/512 | 939 | 469 | 4.3626 | 25.440 / 32.543 / 26.838 | 105.32–108.20 |

Parameter ownership and actual objective counts are recorded per run in resource-ledger.json. RT adds no parameters; multiple FBT passes share weights. The NextLat predictor is training-only. Capacity health at a larger batch is not a full gradient-equivalence check at that batch.
