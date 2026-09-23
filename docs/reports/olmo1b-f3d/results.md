# F3d bounded RT backward workspace

Status: **passed**.

The primary control is F3c fused forward/backward with cast reuse. The candidate changes only backward memory policy. Native parameters, recurrence, RoPE, Q/K treatment and losses stay fixed. Full-model checks select RT layer0 only; quality, all-layer RT, multi-GPU and broader graph configurations remain untested here.

Primary gradient budgets: global relative L2≤1/64, per tensor≤1/32, maximum error/reference maximum≤1/16. Exact-zero references require exact zero. Same-candidate graph and full-Adam checks retain stricter requirements. Nested stricter/FP32 diagnostic failures remain in summary.json and original reports.

| Run | Scope | Status | Passed / declared |
| --- | --- | --- | ---: |
| [f3d-probe-02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/puvm7gux) | probe | passed | 69/69 |
| [f3d-recompute-rt-b8-t512-02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aycfe3gv) | native | passed | 5/5 |
| [f3d-recompute-combined-b8-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/j67nv8l3) | native | passed | 5/5 |
| [f3d-reference-rt-b128-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/um4fp8tm) | native | passed | 1/1 |
| [f3d-recompute-rt-b128-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hxovou71) | native | passed | 1/1 |
| [f3d-reference-combined-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ntu4mxwz) | native | passed | 1/1 |
| [f3d-recompute-combined-b64-t512-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/tv4vh7uz) | native | passed | 1/1 |
| [f3d-recompute-combined-b2-t1024-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7hw7ai6t) | native | passed | 5/5 |
| [f3d-reference-combined-b16-t1024-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/sjz0dqmy) | native | passed | 1/1 |
| [f3d-recompute-combined-b16-t1024-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ydffwbfr) | native | passed | 1/1 |

Successful native runs contain 54 physical optimizer updates (27 eager + 27 graph). Warmup/backward-only probes and failed attempts do not count as successful updates.

| Native check | B/T | Initial losses exact | Gradient relative L2 | Same-candidate graph exact | Full Adam exact |
| --- | ---: | --- | ---: | --- | --- |
| f3d-recompute-rt-b8-t512-02 | 8/512 | True | 0.0008922688525634918 | True | True |
| f3d-recompute-combined-b8-t512-01 | 8/512 | True | 0.0023761505835345996 | True | True |
| f3d-recompute-combined-b2-t1024-01 | 2/1024 | True | 0.0022486273360431807 | True | True |

## Complete-update measurements

Wall time includes input validation/copy, graph forward/loss/backward, clipping, AdamW and scheduler. Peak allocated includes setup; reserved peak and current reserved are distinct. These are full-model measurements, separate from reconstruction-only workspace probes.

| Run | Policy | Case | B/T | Input tokens/s | Seconds/update | Allocated / reserved peak GiB | Current reserved GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f3d-reference-rt-b128-t512-01 | reference | rt | 128/512 | 26,454 | 2.4774 | 42.100 / 68.297 | 44.309 |
| f3d-recompute-rt-b128-t512-01 | recompute | rt | 128/512 | 26,257 | 2.4959 | 38.605 / 61.297 | 40.924 |
| f3d-reference-combined-b64-t512-01 | reference | combined | 64/512 | 10,970 | 2.9871 | 40.840 / 63.816 | 43.396 |
| f3d-recompute-combined-b64-t512-01 | recompute | combined | 64/512 | 10,927 | 2.9989 | 39.093 / 60.865 | 41.842 |
| f3d-reference-combined-b16-t1024-01 | reference | combined | 16/1024 | 8,637 | 1.8969 | 33.163 / 48.186 | 34.881 |
| f3d-recompute-combined-b16-t1024-01 | recompute | combined | 16/1024 | 8,579 | 1.9097 | 31.289 / 44.309 | 33.016 |

## Isolated reconstruction workspace

The shape observer inspects Torch outputs/views, not device-private Triton scratch. Source review and allocated peaks complement it. These figures cover attention reconstruction, not total model memory.

| Probe | Length | Control incremental peak MiB | Candidate incremental peak MiB | Candidate/control |
| --- | ---: | ---: | ---: | ---: |
| f3d-probe-02 | 512 | 29.290 | 3.150 | 0.1076 |
| f3d-probe-02 | 1024 | 116.080 | 6.232 | 0.0537 |
| f3d-probe-02 | 2048 | 462.160 | 12.396 | 0.0268 |
