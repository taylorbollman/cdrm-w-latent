# FBT-only stability study

Started 2026-09-30. The [protocol](protocol.md) fixes K4, full feedback strength,
the fusion128 starting weights, fresh Adam, and the existing data/LR schedule.
The [curve guide](reading-the-curves.md) explains what the measurements mean.
The native FBT-only training segment has finished 128 updates (67,108,864 new
input tokens). Final checkpoint publication and independent terminal auditing
are pending. Feedback learns substantially while first-pass performance remains
close to the ordinary control. The iteration settles, but feedback still does
not improve prediction over the ordinary pass on these panels.

| Optimizer update | First-pass CE | Second-pass CE | Fourth-pass CE |
| ---: | ---: | ---: | ---: |
| 0 | 2.642404 | 7.399125 | 7.171813 |
| 32 | 2.651994 | 6.241516 | 6.368114 |
| 64 | 2.675040 | 4.536458 | 4.765664 |
| 96 | 2.680980 | 3.279866 | 3.349725 |
| 100 | 2.680524 | 3.195912 | 3.259452 |
| 128 | 2.687038 | 2.919213 | 2.942000 |

These are fixed 64-row development CE values in nats per target. The separate
eight-row [final pass curves](figures/update-000128-figure3-style.pdf) show
settling to about 1.6e-6 relative hidden change by pass 16. The shared starting
point includes prior fusion-only training, so this study cannot identify any
earlier stabilization transition during that preparation.

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
checks respectively). Native F-only training has reached its requested
128-update boundary.

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

## Update 48: continued feedback learning

The regular 64-row development CE is now **2.657441 / 5.433011 / 5.562218 /
5.639437** for passes 1–4. Relative to the origin, pass 4 improves by 1.532
nats per target, while pass 1 worsens by 0.015. The separate eight-row probe
gives first/fourth/eighth-pass CE of 2.682534 / 5.461301 / 5.529436, with a
tail relative state change of 0.009648 at pass 8.

Updates remain finite and clipped. Clipping is an observed property of this
recipe, not a numerical failure by itself; the loss improvement does not prove
that its gradient balance or learning rate is optimal. Update 64 is the next
planned deep curve.

## Update 64: better predictions at a still-settling state

Regular development CE is **2.675040 / 4.536458 / 4.688323 / 4.765664** for
passes 1–4. On the small curve panel, CE at passes 1/2/4/8/16/32 is
2.700558 / 4.444502 / 4.650921 / 4.724816 / 4.750531 / 4.751938.
Thus later-pass predictions improve substantially during training, but the
ordinary pass remains best and additional passes beyond 2 modestly worsen CE.

The tail relative state change is 0.08314 at pass 4, 0.01255 at 8, 0.00253
at 16 and 2.12e-6 at 32. The unsettled-suffix curve also approaches 1e-6.
Settling takes somewhat more passes than at update 32, without persistent
oscillation or divergence on this panel. There has been no observed binary
transition from non-settling to settling: the origin already settled.
The origin includes the prior fusion-only 128-update adaptation. This does not
exclude a stabilization transition during that earlier preparation stage; it
describes the shared starting point used by the current F/NF/NFR comparisons.

Late predictive entropy is now 4.850 nats and pre-final-normalization RMS is
2.733, versus 6.832 and 1.323 at the origin. These complement the state-change
curves; normalized hidden RMS alone would conceal the changing scale.

[Curves through 64](figures/update-000064-figure3-style.pdf) ·
[Scale/entropy through 64](figures/update-000064-scale-entropy.pdf) ·
[W&B curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1tbe2qdx).

## Update 96: the prediction gap narrows, with faster settling

Regular development CE is **2.680980 / 3.279866 / 3.328161 / 3.349725**.
The gap between fourth and first passes is 0.669 nats per target, versus
4.529 at the origin. At update 80 the same four values were
2.685428 / 3.735140 / 3.826067 / 3.876882. First-pass CE remains close to the
ordinary control at the same update (2.671181).

On the small panel, first/fourth/32nd-pass CE is
2.711076 / 3.283633 / 3.294059. The tail relative state change is 0.05670
at pass 4, 0.00214 at 8, 6.87e-6 at 16 and 1.51e-6 at 32. This is faster
settling than at 64, alongside better feedback prediction. Late entropy is
3.432 nats and pre-final-normalization RMS is 1.968. Extra inference passes
still offer no CE benefit on this panel.

[Curves through 96](figures/update-000096-figure3-style.pdf) ·
[Scale/entropy through 96](figures/update-000096-scale-entropy.pdf) ·
[W&B curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/yg1f732m).

## Update 100: warmup boundary

Regular development CE is **2.680524 / 3.195912 / 3.240730 / 3.259452**.
The small panel gives CE 2.711169 / 3.192372 / 3.196360 at passes 1/4/32.
Tail relative state change reaches 1.61e-6 by pass 16 and 1.47e-6 by pass 32.
This adds no evidence of a sudden failure at the end of LR warmup. The model
continues with the declared constant LR plateau through update 128.

[Curves through 100](figures/update-000100-figure3-style.pdf) ·
[Scale/entropy through 100](figures/update-000100-scale-entropy.pdf) ·
[W&B curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/5ogbkzpu).

## Update 128 and the next diagnostic

Final regular development CE is **2.687038 / 2.919213 / 2.937069 / 2.942000**.
The first/fourth-pass gap has narrowed from 4.529 to 0.255 nats per target.
The ordinary control finishes at 2.687618 on the same panel; the 0.000580
difference is not a meaningful quality win. The F-only run preserves first-pass
performance while learning to handle its feedback input.

The small panel gives CE 2.714610 / 2.909644 / 2.909915 at passes 1/4/32.
Tail relative hidden change is 0.02679 at pass 4, 0.000329 at 8 and about
1.6e-6 at 16/32. A flat CE curve by pass 4 does not imply identical hidden
states. The bounded exact-online comparison is the next check of this point.
Late predictive entropy is 2.978 nats and pre-final-normalization RMS is 1.783.

All 128 updates are finite. Preclip norms range from 0.7066 to 51.8164, with
median 3.0007. The first 116 updates clip; the last 12 do not. This is an
observed optimization trajectory, not an assertion of an optimal LR or general
BF16 equivalence. No gate, pass-count, normalization or training-loss change was
needed to complete this study.

[Final pass curves](figures/update-000128-figure3-style.pdf) ·
[Final scale/entropy curves](figures/update-000128-scale-entropy.pdf) ·
[W&B curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ynyxw8ml).

The next work is the prepared exact-online check and matched saved NF/NFR
curves, followed—if those checks expose no blocker—by the independently declared
NFR KL1 versus KL0.1 continuation from its common saved update32 state. We are
not extending F-only automatically to192. F versus NF removes both NextLat
losses, so it does not by itself isolate the effect of KL; the paired test does.
