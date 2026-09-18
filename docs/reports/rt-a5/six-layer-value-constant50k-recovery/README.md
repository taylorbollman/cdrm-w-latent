# Six-layer constant value injection through 25,001 updates

Actual status: stopped; requested stage endpoint: 50,000; current λ=0.01.

Only block 1 permanent values receive λ Pe(raw token embedding). Block 0 uses window-2 and the five subsequent RT blocks retain full attention. Temporary self KV, contextual keys and the original NextLat objective are unchanged.

E(t): every state through t correct. A(t): state t alone correct. M(t): mean token accuracy through t. All full/boundary curves use identical prefixes from the same 102,400 length-36 words.

| Value update | λ | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 0.01 | 29.6709% | 61.1582% | 83.0754% |
| 5,000 | 0.01 | 71.5645% | 83.4111% | 94.7186% |
| 10,000 | 0.01 | 78.6465% | 87.4688% | 96.2640% |
| 15,000 | 0.01 | 84.1328% | 90.8184% | 97.4906% |
| 20,000 | 0.01 | 82.1289% | 89.3672% | 96.9867% |
| 25,000 | 0.01 | 79.8447% | 87.9473% | 96.5065% |
| 25,001 | 0.01 | 84.6934% | 90.8730% | 97.5301% |

Latest shared retained budget: 10,000 updates.

| Shared update | Baseline E(36) | Value E(36) | Baseline M(36) | Value M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 17.0352% | 29.6709% | 75.6267% | 83.0754% |
| 5,000 | 67.9365% | 71.5645% | 93.9311% | 94.7186% |
| 10,000 | 82.0771% | 78.6465% | 97.0899% | 96.2640% |

Actual endpoints differ: value 25,001, baseline 10,000. [Unequal terminal curves](length-unequal-terminals.pdf) are descriptive.

Single-seed reused-development diagnostic selected after earlier results. All 61 baseline initial tensors and the data order match; Pe adds262,144 parameters. Baseline 19,998,208 versus value 20,260,352 parameters. The positive full 10k E36 gate permits an exact 50k continuation; linear additionally requires constant full 5k E36>50%. Only common retained budgets are matched comparisons. Unequal actual endpoints are descriptive. No confirmation or autonomous latent rollout was evaluated.

[Full](length-full.pdf) · [Boundary](length-boundary.pdf) · [E/A/M36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/rci2vhfb)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/jp3vx496)
