# F4 independent RT / FBT / NextLat training resources

Status: **completed_with_failed_diagnostics**; complete common B64/T512 matrix: **True**.

Statuses describe the supplied completed runs, not automatic clearance of every planned case. These are functionality and execution measurements with independently switched RT, FBT and NextLat. RT selects layers (0,15); FBT uses K2. Native Q/K math stays unchanged. No quality, placement-superiority, inference-throughput, native FA4 or multi-GPU claim follows.

| Run | Case / layout | RT indices | Role | B/T | Stage | Passed / declared |
| --- | --- | --- | --- | ---: | --- | ---: |
| [f4-ordinary-correctness-b1-t32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w32wnirg) | ordinary / none | [] | common_feature_comparison | 1/32 | correctness (passed) | 5/5 |
| [f4-nextlat-correctness-b1-t32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/atn8g700) | nextlat / none | [] | common_feature_comparison | 1/32 | correctness (passed) | 5/5 |
| [f4-fbt-correctness-b1-t32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/tzzzzgpt) | fbt / none | [] | common_feature_comparison | 1/32 | correctness (passed) | 5/5 |
| [f4-fbt-nextlat-correctness-b1-t32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vqemaeb2) | fbt-nextlat / none | [] | common_feature_comparison | 1/32 | correctness (passed) | 5/5 |
| [f4-rt-nextlat-correctness-b8-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7d55jn8f) | rt-nextlat / spread2 | [0, 15] | common_feature_comparison | 8/512 | correctness (passed) | 5/5 |
| [f4-rt-fbt-correctness-b8-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/03b7mkv4) | rt-fbt / spread2 | [0, 15] | common_feature_comparison | 8/512 | correctness (failed) | 0/1 |
| [f4-combined-correctness-b1-t32](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/l9kcrl2x) | combined / spread2 | [0, 15] | common_feature_comparison | 1/32 | correctness (passed) | 5/5 |
| [f4-ordinary-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9epoq971) | ordinary / none | [] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-rt-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vfqoplon) | rt / spread2 | [0, 15] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-nextlat-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yr4v9a86) | nextlat / none | [] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-rt-nextlat-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/os27mmz3) | rt-nextlat / spread2 | [0, 15] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-fbt-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/l0fpl5gs) | fbt / none | [] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-rt-fbt-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mpr30ssl) | rt-fbt / spread2 | [0, 15] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-fbt-nextlat-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4m1zv6p5) | fbt-nextlat / none | [] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-combined-capacity-b64-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/b84phlpg) | combined / spread2 | [0, 15] | common_feature_comparison | 64/512 | capacity (passed) | 1/1 |
| [f4-ordinary-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8s3rs6ez) | ordinary / none | [] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-rt-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vw8x32oq) | rt / spread2 | [0, 15] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-nextlat-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vbn50pne) | nextlat / none | [] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-rt-nextlat-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/i2cser73) | rt-nextlat / spread2 | [0, 15] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-fbt-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/epshd1f1) | fbt / none | [] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-rt-fbt-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/plwa2wz8) | rt-fbt / spread2 | [0, 15] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-fbt-nextlat-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ngg0ixdd) | fbt-nextlat / none | [] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |
| [f4-combined-capacity-b96-t512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vqpto6b1) | combined / spread2 | [0, 15] | common_feature_comparison | 96/512 | capacity (passed) | 1/1 |

Successful runs contain 132 physical optimizer updates (66 eager + 66 graph). Warmup/backward-only work, failed attempts and 12 historical F3e updates are excluded.

Common cells still missing: none.

## Correctness

Initial materialized-versus-recompute gradients retain global relative L2 ≤1/64, per tensor ≤1/32 and maximum error/reference maximum ≤1/16. Zero references require exact zero. Same-candidate graph and full Adam/state comparisons require exact parity. Stricter diagnostic flags remain in summary.json.

| Run | Initial loss exact | Global gradient relative L2 | Recompute tiles | Graph exact | Full Adam exact |
| --- | --- | ---: | ---: | --- | --- |
| f4-ordinary-correctness-b1-t32 | True | 0 | 0 | True | True |
| f4-nextlat-correctness-b1-t32 | True | 0 | 0 | True | True |
| f4-fbt-correctness-b1-t32 | True | 0 | 0 | True | True |
| f4-fbt-nextlat-correctness-b1-t32 | True | 0 | 0 | True | True |
| f4-rt-nextlat-correctness-b8-t512 | True | 0.00228343 | 1022 | True | True |
| f4-combined-correctness-b1-t32 | True | 0 | 62 | True | True |
| f3e-recompute-rt-spread2-b8-t512-01 (historical F3e, reused) | True | 0.00591965 | 1022 | True | True |
| f3e-recompute-combined-spread2-b8-t512-01 (historical F3e, reused) | True | 0.00729182 | 1022 | True | True |

## Complete-update resources

Wall time includes input copy/validation, graph forward/loss/backward, clipping, AdamW and scheduler. Three-update medians are directional. Setup and timed steady peaks are measured separately; combined allocated/reserved peaks are their maxima. Matrix FLOPs are an analytic ledger excluding elementwise/optimizer/communication work and kernel padding, not hardware utilization.

| Run | Case / layout | B/T | Input tokens/s | CE targets/s | Seconds/update | Allocated / reserved peak / current GiB | Matrix TFLOPs/update | Numerical scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f4-ordinary-capacity-b64-t512 | ordinary / none | 64/512 | 31,113 | 15,557 | 1.0532 | 26.736 / 37.168 / 28.666 | 281.79–304.87 | bounded_correctness_evidence_recorded |
| f4-rt-capacity-b64-t512 | rt / spread2 | 64/512 | 19,437 | 9,719 | 1.6859 | 32.247 / 47.963 / 34.500 | 308.18–328.38 | bounded_correctness_evidence_recorded |
| f4-nextlat-capacity-b64-t512 | nextlat / none | 64/512 | 24,026 | 12,013 | 1.3638 | 28.221 / 39.227 / 29.779 | 314.90–337.99 | bounded_correctness_evidence_recorded |
| f4-rt-nextlat-capacity-b64-t512 | rt-nextlat / spread2 | 64/512 | 16,444 | 8,222 | 1.9928 | 33.733 / 50.514 / 35.461 | 341.29–361.49 | bounded_correctness_evidence_recorded |
| f4-fbt-capacity-b64-t512 | fbt / none | 64/512 | 15,798 | 7,899 | 2.0742 | 32.161 / 48.535 / 34.166 | 565.23–611.39 | bounded_correctness_evidence_recorded |
| f4-rt-fbt-capacity-b64-t512 | rt-fbt / spread2 | 64/512 | 12,136 | 6,068 | 2.7001 | 37.611 / 58.369 / 39.678 | 591.62–634.90 | qualified_unresolved_initial_gradient_screen |
| f4-fbt-nextlat-capacity-b64-t512 | fbt-nextlat / none | 64/512 | 12,177 | 6,089 | 2.6909 | 36.010 / 55.477 / 38.398 | 631.45–677.62 | bounded_correctness_evidence_recorded |
| f4-combined-capacity-b64-t512 | combined / spread2 | 64/512 | 9,892 | 4,946 | 3.3125 | 39.094 / 60.455 / 41.377 | 657.84–701.12 | bounded_correctness_evidence_recorded |
| f4-ordinary-capacity-b96-t512 | ordinary / none | 96/512 | 31,259 | 15,630 | 1.5724 | 31.019 / 45.410 / 33.051 | 422.69–457.31 | bounded_correctness_evidence_recorded |
| f4-rt-capacity-b96-t512 | rt / spread2 | 96/512 | 21,618 | 10,809 | 2.2736 | 39.177 / 62.453 / 41.479 | 462.27–492.56 | bounded_correctness_evidence_recorded |
| f4-nextlat-capacity-b96-t512 | nextlat / none | 96/512 | 23,938 | 11,969 | 2.0533 | 32.630 / 49.145 / 36.133 | 472.36–506.98 | bounded_correctness_evidence_recorded |
| f4-rt-nextlat-capacity-b96-t512 | rt-nextlat / spread2 | 96/512 | 17,833 | 8,917 | 2.7562 | 40.788 / 64.967 / 43.236 | 511.94–542.23 | bounded_correctness_evidence_recorded |
| f4-fbt-capacity-b96-t512 | fbt / none | 96/512 | 15,764 | 7,882 | 3.1181 | 39.069 / 61.910 / 41.676 | 847.85–917.09 | bounded_correctness_evidence_recorded |
| f4-rt-fbt-capacity-b96-t512 | rt-fbt / spread2 | 96/512 | 12,893 | 6,446 | 3.8124 | 47.164 / 78.391 / 51.584 | 887.43–952.34 | qualified_unresolved_initial_gradient_screen |
| f4-fbt-nextlat-capacity-b96-t512 | fbt-nextlat / none | 96/512 | 12,049 | 6,024 | 4.0794 | 44.220 / 73.887 / 48.697 | 947.18–1016.43 | bounded_correctness_evidence_recorded |
| f4-combined-capacity-b96-t512 | combined / spread2 | 96/512 | 10,305 | 5,152 | 4.7698 | 48.774 / 78.223 / 53.068 | 986.76–1051.68 | bounded_correctness_evidence_recorded |

**rt-fbt remains numerically qualified.** Its original screen in `f4-rt-fbt-correctness-b8-t512` is not cleared. Successful resource measurements or separate repeat/graph/FP32 diagnostics do not change that outcome. Common-matrix completion means resource coverage, not numerical clearance.

Parameter ownership and actual objective counts are recorded per run in resource-ledger.json. RT adds no parameters; multiple FBT passes share weights. The NextLat predictor is training-only. Frozen fusion weights remain registered when FBT is disabled. Capacity health at a larger batch is not a full gradient-equivalence check at that batch.

| Case | Registered | Trainable / gradient / optimizer | Deployable | Setup allocated / reserved peak GiB | Steady allocated / reserved peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary B64 | 1,185,153,024 | 1,176,764,416 / 1,176,764,416 / 1,176,764,416 | 1,176,764,416 | 26.736 / 37.168 | 18.462 / 28.666 |
| rt B64 | 1,185,153,024 | 1,176,764,416 / 1,176,764,416 / 1,176,764,416 | 1,176,764,416 | 32.247 / 47.963 | 18.463 / 34.500 |
| nextlat B64 | 1,267,879,936 | 1,259,491,328 / 1,259,491,328 / 1,259,491,328 | 1,176,764,416 | 28.221 / 39.227 | 19.697 / 29.779 |
| rt-nextlat B64 | 1,267,879,936 | 1,259,491,328 / 1,259,491,328 / 1,259,491,328 | 1,176,764,416 | 33.733 / 50.514 | 19.696 / 35.461 |
| fbt B64 | 1,185,153,024 | 1,185,153,024 / 1,185,153,024 / 1,185,153,024 | 1,185,153,024 | 32.161 / 48.535 | 18.558 / 34.166 |
| rt-fbt B64 | 1,185,153,024 | 1,185,153,024 / 1,185,153,024 / 1,185,153,024 | 1,185,153,024 | 37.611 / 58.369 | 18.558 / 39.678 |
| fbt-nextlat B64 | 1,267,879,936 | 1,267,879,936 / 1,267,879,936 / 1,267,879,936 | 1,185,153,024 | 36.010 / 55.477 | 19.792 / 38.398 |
| combined B64 | 1,267,879,936 | 1,267,879,936 / 1,267,879,936 / 1,267,879,936 | 1,185,153,024 | 39.094 / 60.455 | 19.792 / 41.377 |
| ordinary B96 | 1,185,153,024 | 1,176,764,416 / 1,176,764,416 / 1,176,764,416 | 1,176,764,416 | 31.019 / 45.410 | 18.463 / 33.051 |
| rt B96 | 1,185,153,024 | 1,176,764,416 / 1,176,764,416 / 1,176,764,416 | 1,176,764,416 | 39.177 / 62.453 | 18.463 / 41.479 |
| nextlat B96 | 1,267,879,936 | 1,259,491,328 / 1,259,491,328 / 1,259,491,328 | 1,176,764,416 | 32.630 / 49.145 | 19.697 / 36.133 |
| rt-nextlat B96 | 1,267,879,936 | 1,259,491,328 / 1,259,491,328 / 1,259,491,328 | 1,176,764,416 | 40.788 / 64.967 | 19.697 / 43.236 |
| fbt B96 | 1,185,153,024 | 1,185,153,024 / 1,185,153,024 / 1,185,153,024 | 1,185,153,024 | 39.069 / 61.910 | 18.559 / 41.676 |
| rt-fbt B96 | 1,185,153,024 | 1,185,153,024 / 1,185,153,024 / 1,185,153,024 | 1,185,153,024 | 47.164 / 78.391 | 18.557 / 51.584 |
| fbt-nextlat B96 | 1,267,879,936 | 1,267,879,936 / 1,267,879,936 / 1,267,879,936 | 1,185,153,024 | 44.220 / 73.887 | 19.791 / 48.697 |
| combined B96 | 1,267,879,936 | 1,267,879,936 / 1,267,879,936 / 1,267,879,936 | 1,185,153,024 | 48.774 / 78.223 | 19.792 / 53.068 |

## Operator coverage audit

Untimed eager B1/T32 traces include shapes and selected PyTorch operation FLOPs. Fused Flash/Triton and custom/recomputed work can be undercounted. These figures are neither measured hardware FLOPs nor complete-update timing; the analytic resource ledger remains the comparison.
- `f4-ordinary-correctness-b1-t32`: 3,872 device events; 288,101,537,838 selected-operation FLOPs; retained `operator-trace.json.gz`.
- `f4-combined-correctness-b1-t32`: 19,223 device events; 664,668,857,723 selected-operation FLOPs; retained `operator-trace.json.gz`.

`f4-rt-fbt-capacity-b96-t512` exceeds 72 GiB peak reservation: successful but tight setup headroom.

`f4-fbt-nextlat-capacity-b96-t512` exceeds 72 GiB peak reservation: successful but tight setup headroom.

`f4-combined-capacity-b96-t512` exceeds 72 GiB peak reservation: successful but tight setup headroom.

Failed diagnostic `f4-rt-fbt-correctness-b8-t512` (common_feature_comparison): AssertionError: same_state_variant_vs_reference. Its partial timing is excluded; its original source/protocol/error record is retained.
