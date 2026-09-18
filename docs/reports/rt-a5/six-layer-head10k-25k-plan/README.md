# Six-layer reassigned embedding head through 10,000 updates

Actual status: complete; requested stage endpoint: 10,000.

The original six-layer architecture is retained: block index 0 has a two-token attention window and blocks 1–5 use full recurrent attention. In block 1, the existing eighth head (index 7) replaces its permanent values with the existing value-projection rows applied to raw token embeddings. Contextual keys, the other seven heads and temporary self KV remain unchanged. There is no added parameter or scalar gate. Both models have 19,998,208 parameters.

E(t): every state through t correct. A(t): state t alone correct. M(t): mean token accuracy through t. All full/boundary curves use identical prefixes from the same 102,400 length-36 words.

| Head update | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|
| 1,000 | 32.3965% | 61.0957% | 83.0088% |
| 5,000 | 69.4609% | 82.3770% | 94.2698% |
| 10,000 | 78.2344% | 87.4844% | 96.3159% |

Latest shared retained budget: 10,000 updates.

| Shared update | Baseline E(36) | Head E(36) | Baseline M(36) | Head M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 17.0352% | 32.3965% | 75.6267% | 83.0088% |
| 5,000 | 67.9365% | 69.4609% | 93.9311% | 94.2698% |
| 10,000 | 82.0771% | 78.2344% | 97.0899% | 96.3159% |

Single-seed reused-development diagnostic selected after earlier results. All 61 baseline initial tensors and the data order match exactly; both models contain 19,998,208 parameters. One existing head (index 7 in block index 1) writes its permanent values from raw token embeddings using its existing value-projection rows. It replaces that head’s contextual permanent values; it does not add an attention head, projection or scalar gate. Permanent keys, other heads and provisional self KV remain contextual and unchanged. The positive full 10k E36 gate permits an exact 25k continuation. Only common retained budgets are matched comparisons; unequal actual endpoints are descriptive. No confirmation or autonomous latent rollout was evaluated.

[Full](length-full.pdf) · [Boundary](length-boundary.pdf) · [E/A/M36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/3uz2ot97)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/efalr448)
