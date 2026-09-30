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

## Early training, update16

The regular64-row panel improves from originCE
2.642404/7.399125/7.143659/7.171813 to
2.645652/6.909640/6.899153/6.993049 at16 (passes1–4). This is an early feedback
loss improvement with nearly unchanged first-pass CE, still far from useful
refinement relative to the ordinary pass. All updates remain finite/clipped.

The separate eight-row probe has CE2.6732/6.8466/6.9288/7.1965 at passes1/2/4/8.
Its tail relative change at8 is0.035253, versus0.000470 at update8. Learning is
therefore changing the iteration dynamics: improved trained-pass prediction
does not imply faster settling or better extrapolation beyond four passes.
There is no K32 measurement at update16 yet. The planned deep update32 probe,
and the retained update16 checkpoint if needed, can resolve that question.

## Update32: partial adaptation with settling retained

The deep probe again settles on the fixed panel: tail relative state change
falls to6.91e-7 at pass32. Small-panel CE at passes1/2/4/8/16/32 is
2.6783/6.1424/6.3226/6.3789/6.3887/6.3890. Later prediction improves during
training, while extra inference passes still do not beat the ordinary pass.
The temporary large K8 change at update16 did not persist at update32.

Matched regular64-row development measurements at optimizer update32:

| Condition | First-pass CE | Pass4 CE |
| --- | ---: | ---: |
| Ordinary B | 2.631794 | — |
| FBT only F | 2.651994 | 6.368114 |
| FBT + NextLat NF, originalKL1 | 2.960392 | 7.435141 |
| FBT + RT + NextLat NFR, originalKL1 | 3.043012 | 7.063815 |

F and NF start from the same core weights and use the same new data/LR prefix;
the active auxiliary objective differs. NFR also changes the core computation
through RT. This short comparison supports investigating loss balance, not a
general conclusion about NextLat or RT value. None shows useful later-pass
refinement here.

At pass32, entropy declines from6.832 at origin to6.021. Pre-final-normalization
RMS increases from1.323 to3.764; final normalization keeps hidden RMS near1.
Thus representation scale is adapting even while the iteration settles. These
bounded observations are not a blanket activation-health or precision clearance.

[Curves through32](figures/update-000032-figure3-style.pdf) ·
[Scale/entropy through32](figures/update-000032-scale-entropy.pdf) ·
[W&B curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/j2qipybb).
