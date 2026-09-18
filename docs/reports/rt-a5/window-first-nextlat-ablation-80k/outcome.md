# First-window RT: does this result require NextLat?

At **80,000 updates**, CE-only training reached **0.0000% length-36 whole-word accuracy** and **37.3472% token accuracy**. The matched NextLat reference reached **100.0000%** and **100.0000%**, respectively. Both fresh runs and the final report are complete.

The NextLat reference retained a higher whole-word score at the fixed endpoint in this paired development run. The result measures this recipe and seed; it does not establish a general necessity for NextLat.

| Training objective at 80k | L12 whole word | E(36): whole word | A(36): final state | M(36): token accuracy |
| --- | ---: | ---: | ---: | ---: |
| CE only | 99.9570% | 0.0000% | 1.6748% | 37.3472% |
| CE + NextLat | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

| Updates | CE-only E(36) | CE-only M(36) | NextLat E(36) | NextLat M(36) |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.0000% | 15.2682% | 46.7471% | 88.3541% |
| 5,000 | 0.0000% | 36.4385% | 74.0674% | 95.4556% |
| 10,000 | 0.0000% | 36.1609% | 85.7910% | 97.3167% |
| 20,000 | 0.0000% | 35.8888% | 96.3369% | 99.2835% |
| 25,000 | 0.0000% | 37.0040% | 96.2090% | 99.1667% |
| 30,000 | 0.0000% | 36.5195% | 100.0000% | 100.0000% |
| 40,000 | 0.0000% | 36.3958% | 100.0000% | 100.0000% |
| 50,000 | 0.0000% | 37.2384% | 100.0000% | 100.0000% |
| 60,000 | 0.0000% | 38.3196% | 100.0000% | 100.0000% |
| 70,000 | 0.0000% | 37.2552% | 100.0000% | 100.0000% |
| 80,000 | 0.0000% | 37.3472% | 100.0000% | 100.0000% |

First retained full-development checkpoint with zero length-36 whole-word errors — CE only: none through 80k; CE + NextLat: 30,000 updates. These are checkpoint observations, not exact learning thresholds or independent replications.

[Whole-word learning curves (PDF)](whole-word-vs-updates.pdf) · [Full length curves (PDF)](length-full.pdf) · [Boundary view (PDF)](length-boundary.pdf) · [Training state CE (PDF)](training-state-ce.pdf) · [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/8odwv1dz)

Both models start fresh from **exactly the same 21 backbone tensors**, with width 512, two RT layers, ALiBi, Mitchell initialization, full FP32, the same training data and minibatch order, and matching backbone AdamW settings. The first layer attends directly to itself and the preceding recurrent output; the second uses full recurrent attention. Recurrent states and gradients remain attached.

The control optimizes state CE only and has **6,357,504 parameters**. NextLat adds the original weight-one latent SmoothL1 objective and a **1,049,600-parameter predictor**, for **7,407,104 total parameters**. The predictor is not used for backbone inference. Removing it also changes the parameters contributing to global gradient clipping. This is the intended NextLat-training ablation, with matched backbone parameters and update budget, rather than matched total parameters or compute. Each arm sees 81,920,000 word presentations across 800,000 unique length-12 training words.

E(t) requires every state through t correct; A(t) checks only state t; M(t) averages token correctness through t. Prefix curves use the same 102,400 fixed length-36 development words; L12 uses a separate set of 102,400 words. The full and boundary figures use identical rows. This is one seed on repeatedly inspected development data. Both 80k endpoints were set before their own training, but this follow-up control was chosen after seeing the reference's development results. Zero observed errors does not imply perfect population accuracy, replication across seeds, or generalization beyond length 36. Final confirmation and autonomous latent predictor rollout remain unevaluated.

[Detailed report](report.md) · [Exact metric counts](metrics.csv) · [Checkpoint summaries](checkpoint-summary.csv) · [Provenance](report.json)

Source report SHA-256: `0ab04c76ee597051e7f41dec1237e9cffaa3b3dbde7d6aa9cbe68bdebd418695`.
