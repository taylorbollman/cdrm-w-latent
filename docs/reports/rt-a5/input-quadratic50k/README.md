# Quadratic input injection through 13,181 updates

Terminal saved endpoint; λ=0.00347478. Planned endpoint: 50,000 updates. Training status when read: stopped.

E(t) is the fraction of words with every state through t correct; A(t) is accuracy at state t alone; M(t) is mean token accuracy through t. Full and boundary figures use identical prefixes of the same 102,400 length-36 words.

| Update | λ | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 2e-05 | 0.0000% | 1.6768% | 10.8601% |
| 5,000 | 0.0005 | 0.0000% | 1.6787% | 36.3281% |
| 10,000 | 0.002 | 0.0000% | 1.7158% | 37.0038% |
| 13,181 | 0.00347478 | 0.0000% | 1.7021% | 37.3613% |

One seed and repeatedly inspected development pools. This fresh run shares learned initialization with fixed input injection; only the nonlearned schedule changes. A stopped or partial endpoint is selected after reviewing development evidence. No unmodified four-layer control or confirmation result is supplied.

[Full length curve](length-full.pdf) · [Boundary](length-boundary.pdf) · [Length36 trajectory](length36-vs-updates.pdf) · [Training and schedule](training-and-schedule.pdf)

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/db40gut3)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/6pcz6c4n)
