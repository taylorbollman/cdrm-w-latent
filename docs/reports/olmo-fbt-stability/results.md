# FBT-only stability study — in progress

Started 2026-09-30. The [protocol](protocol.md) fixes K4, full feedback strength,
the fusion128 starting weights, fresh Adam, and the existing data/LR schedule.
The [curve guide](reading-the-curves.md) explains what the measurements mean.
Native FBT-only results have not been produced yet.

The ordinary control has completed the requested 128 optimizer updates. Final
checkpoint publication and terminal report closeout are still pending. This
resumes the original ordinary update32 state and optimizer; it is not another
random initialization or a new data order.

| Optimizer update | Ordinary development CE, nats/target |
| ---: | ---: |
| 32 | 2.631794 |
| 48 | 2.635389 |
| 64 | 2.642590 |
| 80 | 2.656053 |
| 96 | 2.671181 |
| 112 | 2.676973 |
| 128 | 2.687618 |

The modest deterioration is relevant context for the same-schedule FBT-only
run; it is not evidence of feedback failure because this control has no feedback.
These are the fixed 64-row development measurements, not the smaller eight-row
pass-curve panel or a final quality evaluation.

[Ordinary control on W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/37uu86ip).

Implementation checks: 52 focused execution/probe CPU tests, 55 independent
auditor tests, and 23 isolated online-comparison tests pass. The small two-GPU
insertion/cloud-resume checks are next, before the native F-only launch.
