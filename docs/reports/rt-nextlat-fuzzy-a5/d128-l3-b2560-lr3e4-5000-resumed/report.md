# Mixed A5/Fuzzy: two versus three RT layers

The three-layer run complete at **5,000 optimizer updates**. It uses a restricted first RT layer followed by two full RT layers, with NextLat.

Both arms use 2,560 examples per task per update, FP32, and LR rising from 1e-4 at update 1 to 3e-4 at update 100, then constant. Parameters increase from 479,616 to 676,736.

Physical microbatch: two layers **2560**, three layers **2560**. The effective batch and single global clipping/Adam step remain fixed.

| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |
|---:|---|---:|---:|---:|
| 1,000 | 2 layers: restricted + full RT | 0.0000% | 0.0000% | 26.8017% / 15.3824% / 0.0000% |
| 1,000 | 3 layers: restricted + two full RT | 0.0000% | 0.0000% | 79.6372% / 65.1362% / 0.3906% |
| 2,500 | 2 layers: restricted + full RT | 44.9512% | 1.7734% | 84.6405% / 82.4307% / 2.0312% |
| 2,500 | 3 layers: restricted + two full RT | 0.0000% | 0.0000% | 99.6361% / 99.6451% / 90.7031% |
| 5,000 | 2 layers: restricted + full RT | 99.9375% | 83.4307% | 99.5960% / 99.6508% / 91.1719% |
| 5,000 | 3 layers: restricted + two full RT | 1.5850% | 0.0000% | 97.3321% / 96.1988% / 49.2188% |

At the actual three-layer endpoint, full-development A5 length36 whole-word accuracy is **0.0000%**; Fuzzy answer accuracy is **97.3321%**. All endpoint tasks use the same saved checkpoint.

| Arm | Training updates shown | Cumulative training minutes |
|---|---:|---:|
| 2 layers: restricted + full RT | 5,000 | 162.83 |
| 3 layers: restricted + two full RT | 5,000 | 242.04 |

One seed per architecture and reused development pools; this is a directional depth comparison, not a replicated effect or a parameter-matched comparison. Both use native Mitchell initialization with the same seeds; changing depth changes the backbone draw and depth-dependent scaling, so backbone tensor identity is neither required nor claimed. The independently initialized NextLat predictor is identical. Data order, effective batch, task weights, objective, FP32 runtime and LR schedule remain fixed. Any physical microbatch difference is disclosed. A5 monitoring subsets and full evaluations have different sample sizes; matched observations require equal sample sizes. Threshold crossings are first observed, not sustained convergence. Training time excludes evaluation, checkpointing and reporting and is not a repeated throughput benchmark. No new model inference, final confirmation or autonomous latent rollout is performed.

## Figures

[matched-learning-updates](matched-learning-updates.pdf)
[matched-learning-training-time](matched-learning-training-time.pdf)
[matched-prefix](matched-prefix.pdf)
[training-diagnostics](training-diagnostics.pdf)

## Recovery lineage

The three-layer run resumed from a verified full-state checkpoint after VM shutdown. Only checkpoint-committed ancestor updates are included; post-checkpoint work is excluded and replayed from restored model, Adam, RNG and data-stream state. Original files are preserved. Timing sums canonical committed updates once and excludes lost work and shutdown downtime. The same initialization, objective, data order and global LR schedule continue across recovery.

{
  "stages": 2,
  "restored_updates": [
    1200
  ],
  "discarded_tail_bytes": 9087,
  "boundary_full_state_exact": true
}
