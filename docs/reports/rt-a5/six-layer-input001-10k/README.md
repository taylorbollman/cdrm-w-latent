# Six-layer input injection through 10,000 updates

Terminal saved endpoint; λ is fixed at 0.01, including initialization. The requested pilot endpoint is 10,000 updates; training status when read: complete.

The first block uses window-2 attention and the remaining five use full recurrent attention. Only the input to block 1 receives 0.01 × Pe(raw token embedding), before its existing normalization. NextLat training and backbone-only evaluation are unchanged.

E(t) is the fraction of words with every state through t correct; A(t) is accuracy at state t alone; M(t) is mean token accuracy through t. Full and boundary figures use identical prefixes of the same 102,400 length-36 words.

| Injection update | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|
| 1,000 | 29.3359% | 61.4814% | 83.2379% |
| 5,000 | 71.2295% | 83.6250% | 94.7432% |
| 10,000 | 73.7578% | 84.8799% | 95.2499% |

Original six-layer baseline versus injection at common saved budgets:

| Update | Arm | λ | E(36) | A(36) | M(36) |
|---:|---|---:|---:|---:|---:|
| 1,000 | Original six-layer baseline | 0 | 17.0352% | 50.6709% | 75.6267% |
| 1,000 | Six-layer input injection | 0.01 | 29.3359% | 61.4814% | 83.2379% |
| 5,000 | Original six-layer baseline | 0 | 67.9365% | 81.6270% | 93.9311% |
| 5,000 | Six-layer input injection | 0.01 | 71.2295% | 83.6250% | 94.7432% |
| 10,000 | Original six-layer baseline | 0 | 82.0771% | 89.7324% | 97.0899% |
| 10,000 | Six-layer input injection | 0.01 | 73.7578% | 84.8799% | 95.2499% |

One seed and repeatedly inspected development pools. This fresh six-layer run shares all 61 baseline backbone/predictor initial tensors and the same data order; its only added parameter is Pe (262,144 weights), injected before block 1 with fixed coefficient 0.01. The six-layer baseline has 19,998,208 parameters; the injected model has 20,260,352. The experiment was chosen after previous development results. Matched comparisons use only common retained checkpoint budgets; an earlier stopped or partial endpoint is retrospective and its unequal terminal is shown separately. No confirmation or autonomous latent rollout was evaluated.

[Matched full length curve](length-full.pdf) · [Matched boundary](length-boundary.pdf) · [Length36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)

[Injection training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/redf7ybc) · [Baseline training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/fg4quqku)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/sagaral4)
