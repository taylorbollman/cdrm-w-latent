# NFR settling and continuation through update 128

The two saved NFR64 endpoints completed matched K1–32 probes with bounded,
decaying hidden-state changes and exact preservation checks. Their poor
later-pass CE is not repaired by additional iterations. This supports the
authorized unchanged KL0.1 continuation, which began at 15:49 UTC on
2026-09-30 and is **currently running**, stopping at update 128.

[Training on W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2zu5jloq)
and [paired endpoint curves](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/3p91owsx).
See the [PDF](figures/endpoint-curves.pdf), [protocol](protocol.md),
[validation](validation.md) and [progress/recovery record](progress.md).

## Saved NFR64 endpoint measurements

Both use the same eight packed T1024 development rows: 8,192 inputs and 8,184
CE targets, common FP32, no jitter, native RT at layers 0/15, no predictor
execution and no optimizer updates. This is the deep-pass panel, separate
from the regular 64-row development panel used for training progress.

| Metric | KL 1 | KL 0.1 |
| --- | ---: | ---: |
| CE at K1 | 2.935495 | 2.788031 |
| CE at K4 | 6.606082 | 5.412258 |
| CE at K8 | 6.622874 | 5.430928 |
| CE at K32 | 6.627102 | 5.431623 |
| CE K32 minus K4 | +0.021020 | +0.019366 |
| Tail consecutive change at K4 | 1.5173% | 3.7064% |
| Tail consecutive change at K8 | 0.1611% | 0.1540% |
| Tail consecutive change at K32 | 0.0006970% | 0.00004587% |
| Direct tail K4–K32 hidden residual | 1.5480% | 2.1792% |
| Direct tail K8–K32 hidden residual | 0.5902% | 0.1766% |
| Direct all-position K4–K32 residual | 1.7495% | 2.7539% |

Tail means the final 128 positions of each row. Consecutive change is pooled
RMS(hK − hK−1) / RMS(hK−1). Direct residual is pooled
RMS(hK − h32) / RMS(h32), measured from the same canonical forward, rather
than inferred from the consecutive differences. K32 is a finite reference,
not an exact-online target.

Both pooled tail curves decrease across passes. Reduced KL leaves more
remaining change at K4 but is closer to its K32 state at K8. Therefore, lower
KL does not simply slow settling at every depth. The control's slower deep
tail is primarily associated with one panel row; it continues decaying at
K32 rather than flattening at the earlier checkpoint's numerical floor.

The normalized hidden RMS is approximately one by construction, so it is not
sufficient evidence of bounded internal scales. Pre-normalization RMS remains
bounded too: at K4/K32 it is about 8.795/8.775 for control and 4.676/4.687 for
reduced. Input RMS remains approximately 0.037–0.040. K32 entropy is 6.545
versus 5.666. No runaway growth or non-decaying oscillation appears on this
panel. Every weights/RNG/gradient/runtime preservation check passed.

The practical conclusion is that K4 retains a measurable hidden-state
residual, but simply iterating these saved weights more does not recover the
large first-to-later-pass prediction gap. This does not rule out a role for
finite-pass mismatch during learning, and is not a generation or BF16
optimization-equivalence result.

## Continuation in progress

The selected KL0.1 branch restores its exact update64 model, populated Adam,
rank RNG, data cursor and original 128-update schedule. It adds 64 updates
(33,554,432 real input tokens), retaining four passes, both RT layers and all
other settings. Development measurements/checkpoints include 96, 100 and 128,
with update100 marking the existing warmup boundary. No additional training
beyond128 is authorized by this execution.

The saved128 state will receive the same bounded K1–32 probe after training.
Regular development and final deep-pass results are pending. The control64
endpoint will not be represented as a matched128 control, and no RT-benefit or
useful-refinement claim follows from finite updates alone.

Interim update80 regular-panel CE is2.787306/4.407264/4.511726/4.563218.
Relative to64, fourth-pass CE improves.956904nats while first-pass CE rises
.012336. The gap shrinks2.745152→1.775912nats. All16 resumed updates are
finite/clipped and evaluation preserves training state exactly. Later-pass
raw latent/KL losses rise while CE improves. Training remains in progress;
see the dated record above for the latest checkpoint status.

Interim update96 regular-panel CE is2.774441/3.630273/3.676090/3.700264;
the fourth-minus-first gap falls further to.925822nats. The first pass is
close to its64 CE, but the feedback passes remain worse. All32 resumed updates
are finite/clipped; dev96 preserves training state exactly. Later-pass raw
latent/KL continue rising while CE improves. These auxiliary losses predict
the next token position's hidden state/distribution within each pass, not the
next FBT iteration. Their detached teachers evolve with the trained model.
The opposite trends alone establish neither numerical failure nor a cause
such as gradient conflict or predictor lag.

At the end of warmup, update100 CE is2.769629/3.510893/3.552609/3.574547,
gap.804918nats. The36 resumed updates remain finite/clipped. Update101
will use the original schedule's peak LR2e-4; no schedule change was made.

Update112 CE is2.763501/3.231428/3.253204/3.263256, gap.499755nats.
Adaptation continues after warmup, but feedback still worsens prediction
relative to the first pass. The final128 results/probe are still pending.

Final regular-panel update128 CE is2.756071/3.045481/3.053890/3.058500,
gap.302429nats. The gap continues shrinking, more slowly than before;
first-pass CE also improves slightly. All64 resumed updates are finite,
including six consecutive unclipped final updates. Evaluation preserves
training state exactly. Final checkpoint publication and deep-pass probe
remain pending at this observation; no update129 has been launched.
