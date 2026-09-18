# Six-layer fixed input injection through 20,000 updates

Exact continuation from 10,000 to 20,000; requested endpoint 20,000. Fixed λ=0.01. Latest matched full checkpoint: 20,000.

The first layer uses window-2 attention and the next five use full recurrent attention. Only the second block input receives 0.01 × Pe(raw token embedding), before its existing normalization. NextLat is unchanged.

E(t): every state through t correct; A(t): state t alone correct; M(t): mean token accuracy through t. Full and boundary plots use identical36-prefix rows.

| Update | Arm | E(36) | A(36) | M(36) |
|---:|---|---:|---:|---:|
| 1,000 | baseline | 17.0352% | 50.6709% | 75.6267% |
| 1,000 | injection | 29.3359% | 61.4814% | 83.2379% |
| 5,000 | baseline | 67.9365% | 81.6270% | 93.9311% |
| 5,000 | injection | 71.2295% | 83.6250% | 94.7432% |
| 10,000 | baseline | 82.0771% | 89.7324% | 97.0899% |
| 10,000 | injection | 73.7578% | 84.8799% | 95.2499% |
| 15,000 | baseline | 79.0273% | 87.6680% | 96.3723% |
| 15,000 | injection | 82.7910% | 90.2002% | 97.2665% |
| 20,000 | baseline | 80.9629% | 88.4922% | 96.7088% |
| 20,000 | injection | 76.5264% | 85.9150% | 95.7257% |

One seed and repeatedly inspected development pools. The user extended the current six-layer input run while it was training. This continues the exact10k model/Adam/RNG/data order at fixedlambda0.01 through at most20k. All61 original baseline initial tensors are shared; Pe adds262144 parameters. The baseline also continues its own exact10k state to20k. Comparisons use shared full102400-word checkpoints; a user-selected earlier stop is retrospective and its unmatched terminal is shown separately. No new inference, confirmation or autonomous latent rollout is included.

[Matched full curve](length-full.pdf) · [Matched boundary](length-boundary.pdf) · [Length36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)

[Injection continuation W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/9hrfzx53) · [Baseline continuation W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/vmg2i60x)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/dqs2k3rl)
