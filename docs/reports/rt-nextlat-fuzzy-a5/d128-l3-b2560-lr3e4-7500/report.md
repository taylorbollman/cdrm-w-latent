# Three-layer mixed A5/Fuzzy continuation

The three-layer model complete at **7,500 total optimizer updates**. Its recipe remains restricted RT → full RT → full RT, NextLat, FP32, 2,560 examples per task and LR3e-4 following the original 100-update warmup. Model, Adam, RNG and data streams resume exactly.

**Matched architecture comparisons end at 5,000 updates.** The two-layer reference has no later observations. The continuation results below are additional-budget context.

| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |
|---:|---|---:|---:|---:|
| 1,000 | 2 layers: restricted + full RT | 0.0000% | 0.0000% | 26.8017% / 15.3824% / 0.0000% |
| 1,000 | 3 layers: restricted + two full RT | 0.0000% | 0.0000% | 79.6372% / 65.1362% / 0.3906% |
| 2,500 | 2 layers: restricted + full RT | 44.9512% | 1.7734% | 84.6405% / 82.4307% / 2.0312% |
| 2,500 | 3 layers: restricted + two full RT | 0.0000% | 0.0000% | 99.6361% / 99.6451% / 90.7031% |
| 5,000 | 2 layers: restricted + full RT | 99.9375% | 83.4307% | 99.5960% / 99.6508% / 91.1719% |
| 5,000 | 3 layers: restricted + two full RT | 1.5850% | 0.0000% | 97.3321% / 96.1988% / 49.2188% |

## Three-layer continuation only

| Update | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |
|---:|---:|---:|---:|
| 5,000 | 1.5850% | 0.0000% | 97.3321% / 96.1988% / 49.2188% |
| 6,000 | 21.7119% | 0.0000% | 98.4870% / 98.1738% / 67.5000% |
| 6,500 | 66.4717% | 0.0000% | 98.8108% / 98.4257% / 73.1250% |
| 7,000 | 74.0977% | 0.0000% | 95.1085% / 93.2104% / 29.2188% |
| 7,500 | 87.3857% | 0.0000% | 99.2635% / 99.3130% / 82.7344% |

Three-layer cumulative committed training time: **6.074 hours**; additional training after 5k: **122.41 minutes**. Evaluation, checkpointing, reporting, shutdown downtime and discarded work are excluded.

Matched training time through 5k: 2 layers: restricted + full RT 2.714 hours, 3 layers: restricted + two full RT 4.034 hours.

One seed per architecture and reused development pools; this is a directional depth comparison, not a replicated effect or a parameter-matched comparison. Both use native Mitchell initialization with the same seeds; changing depth changes the backbone draw and depth-dependent scaling, so backbone tensor identity is neither required nor claimed. The independently initialized NextLat predictor is identical. Data order, effective batch, task weights, objective, FP32 runtime and LR schedule remain fixed. Any physical microbatch difference is disclosed. A5 monitoring subsets and full evaluations have different sample sizes; matched observations require equal sample sizes. Threshold crossings are first observed, not sustained convergence. Training time excludes evaluation, checkpointing and reporting and is not a repeated throughput benchmark. No new model inference, final confirmation or autonomous latent rollout is performed. The three-layer run resumed from a verified full-state checkpoint after VM shutdown. Only checkpoint-committed ancestor updates are included; post-checkpoint work is excluded and replayed from restored model, Adam, RNG and data-stream state. Original files are preserved. Timing sums canonical committed updates once and excludes lost work and shutdown downtime. The same initialization, objective, data order and global LR schedule continue across recovery. Only the three-layer model continues beyond 5000 updates. The two-layer reference ends at its actual saved 5000-update checkpoint; all matched-update comparisons and matched training times stop there. Three-layer results from 5001 through its actual endpoint (at most 7500) are unequal-budget continuation context. There is no extrapolated reference curve or reference training time beyond 5000. The 5000-update result and any continuation result are disclosed separately, and no learning-rate, optimizer, model or data recipe changes at the extension.

## Figures

[matched-learning-updates](matched-learning-updates.pdf)
[matched-learning-training-time](matched-learning-training-time.pdf)
[matched-prefix](matched-prefix.pdf)
[training-diagnostics](training-diagnostics.pdf)
[continuation-context-updates](continuation-context-updates.pdf)
[continuation-context-training-time](continuation-context-training-time.pdf)
[continuation-prefix](continuation-prefix.pdf)
[continuation-diagnostics](continuation-diagnostics.pdf)
