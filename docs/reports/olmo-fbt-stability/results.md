# FBT-only stability study — in progress

Started 2026-09-30. The [protocol](protocol.md) fixes K4, full feedback strength,
the fusion128 starting weights, fresh Adam, and the existing data/LR schedule.
The [curve guide](reading-the-curves.md) explains what the measurements mean.
The native FBT-only run is active; the first origin measurement is available.

The ordinary control has completed the requested 128 optimizer updates. Final
checkpoint publication and terminal report closeout are complete. This
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
insertion/cloud-resume checks also pass exactly (8,531 and10,150 independent
checks respectively). Native F-only training is underway.

## Before new F-only training: settling without useful prediction

On the fixed eight-row FP32/no-jitter panel, the inherited fusion128 state already
settles over repeated passes. This is empirical behavior on this panel, not a
proof of global contraction. Its poor later-pass CE remains the central issue.

| Total passes | CE, nats/target | Relative hidden change, final128 positions |
| ---: | ---: | ---: |
| 1 | 2.673866 | — |
| 2 | 7.364103 | 0.979872 |
| 4 | 7.188377 | 0.108499 |
| 8 | 7.234902 | 0.004238 |
| 16 | 7.239706 | 0.000406 |
| 32 | 7.239747 | approximately0.000002 |

Predictive entropy rises from2.653nats on the first pass to6.832 at32; the poor
predictions are much broader. Pre-final-normalization RMS remains finite,
1.603→1.323, and stack-input RMS0.0398→0.0371. This does not demonstrate a
representation-collapse mechanism. It separates the question of settling from
that of learning useful feedback.

Independent readback confirms this across all eight rows: the tail-relative
change at32 is1.45e-6–1.89e-6. The unsettled suffix still contains7,944/8,192
positions, so removal of the guaranteed settled prefix cannot explain the
pattern. The last several CE values change only around1e-7. Final hidden and
fusion-input scales are substantially imposed by normalization and are not
independent evidence of a healthy representation. The FP32 numerical floor may
account for the small residual differences.

The previous NF-origin diagnostic has identical membership/counts; its first
four CE values agree within1.62e-7 aggregated and5.11e-7 per row. This is close
numerical agreement, not bit-exact identity, because readout grouping/reductions
differ. The old entropy used a different eligible-position mask and must not be
compared directly with the new CE-position entropy.

[Origin pass curves](figures/update-000000-figure3-style.pdf) ·
[Scale and entropy](figures/update-000000-scale-entropy.pdf) ·
[Curve W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/socosavi) ·
[Training W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/32sqvp7e).
