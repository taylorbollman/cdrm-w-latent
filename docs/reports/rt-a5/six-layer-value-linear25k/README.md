# Six-layer linear value injection through 25,000 updates

Actual status: complete; requested stage endpoint: 25,000; current λ=0.01.

Only block 1 permanent values receive λ Pe(raw token embedding). Block 0 uses window-2 and the five subsequent RT blocks retain full attention. Temporary self KV, contextual keys and the original NextLat objective are unchanged.

E(t): every state through t correct. A(t): state t alone correct. M(t): mean token accuracy through t. All full/boundary curves use identical prefixes from the same 102,400 length-36 words.

| Value update | λ | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 0.0005 | 27.6572% | 59.9189% | 82.2518% |
| 5,000 | 0.0025 | 63.8252% | 79.4707% | 93.0074% |
| 10,000 | 0.005 | 75.4678% | 85.9863% | 95.6513% |
| 15,000 | 0.0075 | 81.4209% | 88.9609% | 96.8940% |
| 20,000 | 0.01 | 78.6396% | 86.9014% | 96.0643% |
| 25,000 | 0.01 | 83.8047% | 90.3691% | 97.3577% |

Latest shared retained budget: 10,000 updates.

| Shared update | Baseline E(36) | Value E(36) | Baseline M(36) | Value M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 17.0352% | 27.6572% | 75.6267% | 82.2518% |
| 5,000 | 67.9365% | 63.8252% | 93.9311% | 93.0074% |
| 10,000 | 82.0771% | 75.4678% | 97.0899% | 95.6513% |

Actual endpoints differ: value 25,000, baseline 10,000. [Unequal terminal curves](length-unequal-terminals.pdf) are descriptive.

Single-seed reused-development diagnostic selected after earlier results. All 61 baseline initial tensors and the data order match; Pe adds262,144 parameters. Baseline 19,998,208 versus value 20,260,352 parameters. The positive full 10k E36 gate permits an exact 25k continuation; linear additionally requires constant full 5k E36>50%. Only common retained budgets are matched comparisons. Unequal actual endpoints are descriptive. No confirmation or autonomous latent rollout was evaluated.

[Full](length-full.pdf) · [Boundary](length-boundary.pdf) · [E/A/M36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)
[Coefficient](coefficient.pdf)

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/z547dpz8)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/lr999w8p)
