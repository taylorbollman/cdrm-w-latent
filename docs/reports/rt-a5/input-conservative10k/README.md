# Conservative input injection through 9,665 updates

Terminal saved endpoint; λ=3.73649e-05. Pilot endpoint: 10,000 updates; warmup spans 50,000 updates and remains incomplete at this pause. Training status when read: stopped.

E(t) is the fraction of words with every state through t correct; A(t) is accuracy at state t alone; M(t) is mean token accuracy through t. Full and boundary figures use identical prefixes of the same 102,400 length-36 words.

| Update | λ | E(36) | A(36) | M(36) |
|---:|---:|---:|---:|---:|
| 1,000 | 4e-07 | 0.0000% | 1.6484% | 10.8268% |
| 5,000 | 1e-05 | 0.0000% | 1.7041% | 35.9727% |
| 9,665 | 3.73649e-05 | 0.0000% | 1.6064% | 35.7375% |

One seed and repeatedly inspected development pools. This fresh run shares learned initialization with fixed input injection; only the nonlearned schedule changes. A stopped or partial endpoint is selected after reviewing development evidence. No unmodified four-layer control or confirmation result is supplied.

[Full length curve](length-full.pdf) · [Boundary](length-boundary.pdf) · [Length36 trajectory](length36-vs-updates.pdf) · [Training and schedule](training-and-schedule.pdf)

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/h5m6ufcn)

[Report W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/yd98box9)
