# FP32 mixed A5/Fuzzy: learning-rate acceleration probe

The probe complete at **5,000 optimizer updates**, with2,560 examples per task per update. LR rises linearly from1e-4 at update1 to3e-4 at update100, then remains constant.

Only the LR recipe changes; model, NextLat, initialization, data order, task weights, FP32 and clipping remain matched.

| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |
|---:|---|---:|---:|---:|
| 1,000 | constant_1e4 | 0.0000% | 0.0000% | 14.9497% / 14.6382% / 0.0000% |
| 1,000 | warmup_3e4 | 0.0000% | 0.0000% | 26.8017% / 15.3824% / 0.0000% |
| 2,500 | constant_1e4 | 0.0000% | 0.0000% | 46.9152% / 27.8223% / 0.0000% |
| 2,500 | warmup_3e4 | 44.9512% | 1.7734% | 84.6405% / 82.4307% / 2.0312% |
| 5,000 | constant_1e4 | 0.0000% | 0.0000% | 90.5379% / 82.5510% / 6.1719% |
| 5,000 | warmup_3e4 | 99.9375% | 83.4307% | 99.5960% / 99.6508% / 91.1719% |

At the probe's actual endpoint, full-development A5 length36 whole-word accuracy is **83.4307%**; Fuzzy answer accuracy is **99.5960%**. All endpoint tasks use the same saved checkpoint.

The original baseline eventually learned A5: its first observed positive length36 result was at10,400 updates on4,096 words, and its full15k endpoint reached **92.8535% whole-word accuracy**, with **99.9112% Fuzzy answer accuracy**. A disappointing5k probe means this acceleration recipe did not achieve the desired early learning; it does not establish architectural failure.

| Arm | Training updates shown | Cumulative training minutes |
|---|---:|---:|
| constant_1e4 | 5,000 | 163.02 |
| warmup_3e4 | 5,000 | 162.83 |

One matched initialization and reused development pools. Only the learning-rate recipe changes: FP32, D128, two-layer restricted-first RT, NextLat, batch2560 per task, task weights, data order and gradient clipping remain fixed. The new arm is a 5000-update acceleration probe. The original 1e-4 baseline succeeded with more training: length36 A5 whole-word accuracy was 92.853515625% at15000. A zero result at5000 does not establish architectural failure. Common-update comparisons match evaluation pool sizes; monitoring subsets and full evaluations are identified separately. First observed threshold crossings are not sustained convergence. Training time excludes evaluations/checkpointing/reporting and is not a repeated throughput benchmark. No new model inference, final confirmation or autonomous latent rollout is performed by this report.

## Figures

[matched-learning-updates](matched-learning-updates.pdf)
[matched-learning-training-time](matched-learning-training-time.pdf)
[baseline-extended-context](baseline-extended-context.pdf)
[matched-prefix](matched-prefix.pdf)
[training-diagnostics](training-diagnostics.pdf)
