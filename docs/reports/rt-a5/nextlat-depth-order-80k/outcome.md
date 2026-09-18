# A5 + NextLat: depth and recurrent window-order outcome

**Putting window-2 attention in the first RT layer produced zero observed errors at the completed 80,000-update endpoint**, on both 102,400 length-12 development words and 102,400 length-36 development words. Every state in every word was correct. The other three models had zero completely correct length-36 words at 80k. Both authorized training runs and the audited report are complete; no new experiment is running.

All four models below train with NextLat and use their backbone for inference. Layer order is written first / second.

| Model at 80k | L12 whole word | E(14) | E(16) | M(36) | E(36) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RT: window 2 / full | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| SEQ4 | 95.1816% | 65.4463% | 10.6631% | 40.0371% | 0.0000% |
| RT: full / full | 99.5449% | 29.6934% | 0.2236% | 37.8208% | 0.0000% |
| RT: full / window 2 | 99.9189% | 36.3848% | 0.4375% | 38.1807% | 0.0000% |

SEQ4 achieved better length-14/16 prefix exactness than the two historical RT controls, while retaining more length-12 errors. The first-window RT's improvement extends across the full evaluated length range.

The first-window RT's complete retained checkpoint trajectory is:

| Training updates | E(36): all 36 states correct |
| --- | ---: |
| 1,000 | 46.7471% |
| 5,000 | 74.0674% |
| 10,000 | 85.7910% |
| 20,000 | 96.3369% |
| 25,000 | 96.2090% |
| 30,000 | 100.0000% |
| 40,000 | 100.0000% |
| 50,000 | 100.0000% |
| 60,000 | 100.0000% |
| 70,000 | 100.0000% |
| 80,000 | 100.0000% |

The other three arms had E(36) = 0 at every listed checkpoint. The first-window model had zero observed errors in both development roles at each retained checkpoint from 30k through 80k. These are repeated measurements on the same words, not independent replications. The slight dip between 20k and 25k also means the full learning trajectory was not monotonic.

E(t) requires every state through t correct; A(t) checks only state t; M(t) averages token correctness through t. At first-window 80k, E/A/M all equal 1 across both evaluation roles. Prefix curves come from the same fixed length-36 words; L12 uses a separate development set. The full and boundary plots show identical underlying rows.

[Full length curves (PDF)](length-full.pdf) · [Readable boundary (PDF)](length-boundary-readable.pdf) · [Readable boundary (PNG)](length-boundary-readable.png) · [Exactness across updates (PDF)](exactness-vs-updates.pdf) · [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/tb8ey88n)

The RT window-order comparison preserves two layers, **7,407,104 parameters / 25 learned tensors**, and exact initial learned tensors. The first-layer window reads provisional self K/V plus the immediately preceding recurrent output K/V. Recurrent states and gradients remain attached, so the direct read window does not limit represented history to two tokens. The result supports window location as a consequential design choice **in RT trained with NextLat**.

SEQ4 has **13,702,656 parameters / 39 learned tensors** and its own canonical four-layer Mitchell initialization. Its comparison is matched on width and update budget, not parameters, FLOPs or backbone tensors. All arms share width 512, ALiBi, full FP32, the same A5 training data and minibatch order, original state-CE plus weight-one next-latent objective, predictor initialization, and optimizer recipe.

These are one-seed development results. The new 80k endpoints were fixed before training; historical 80k endpoints and the choice of these experiments followed development inspection. Historical full/full and second-window unsaved tails of 1,607 and 5,376 updates remain excluded, and neither historical arm has a completed 100k outcome. Zero observed errors here do not establish perfect population accuracy, replication across seeds, or generalization beyond length 36. Final confirmation and autonomous latent predictor rollout remain unevaluated.

The current comparison does not isolate whether NextLat is necessary for the first-window improvement. A useful possible next control is **the same first-window RT without NextLat**, with the backbone, data/order and training budget held fixed. That control has not been launched.

[Detailed report](report.md) · [Exact metric counts](metrics.csv) · [Checkpoint summaries](checkpoint-summary.csv) · [Validated provenance](report.json)
