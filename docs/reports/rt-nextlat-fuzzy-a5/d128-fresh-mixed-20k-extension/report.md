**Authoritative extended-budget interpretation.** The historical 10k gate is unchanged; this report supersedes the earlier gate-based conclusion about the extended budget.

# D128 A5 and Fuzzy development pilot

**mixed, actual endpoint 19,810 updates.** Training status: stopped.

Two RT layers; first window two, second full recurrent. Width 128, 16 heads, FFN 512; ALiBi, Mitchell, FP32 eager, NextLat retained. 479,616 parameters; no embedding bypass or autonomous latent rollout.

128 examples per active task per optimizer update; 2,535,680 examples per task at the endpoint. Losses are means within each task; mixed training weights the two task objectives equally.

| Same-checkpoint endpoint metric | Accuracy |
| --- | ---: |
| A5 L12 mean token | 99.7935% |
| A5 L12 final state | 98.9834% |
| A5 L12 whole word | 98.6279% (100,995 / 102,400) |
| A5 L36 mean token | 91.3196% |
| A5 L36 final state | 73.4775% |
| A5 L36 whole word | 57.6582% (59,042 / 102,400) |
| Fuzzy answer tokens | 98.8509% |
| Fuzzy answer-motif exact | 97.7673% |
| Fuzzy all-answers sequence exact | 74.2188% |
| Fuzzy first value token | 99.8855% |
| Fuzzy terminal probe | 97.4783% |

## Historical 10k gate and extended-budget observations



The original continuation gate applies only through update 10,000 and is preserved unchanged.

That historical gate did not pass: no positive full L36 observation was recorded by 10k. This does not determine what happened during the extension.

Across the actual extended budget, the first positive full L36 observation was at **update 15,000**: 19/102,400 words (0.018555%).

At the actual endpoint, L36 whole-word accuracy is **57.658203%** (59,042/102,400). Earlier positive observations and endpoint performance are reported separately.

A positive finite-sample development count establishes observed correct length-36 words; it does not establish reliable or general state tracking. The first observed full positive is not the exact onset of learning. Final confirmation remains unused. The unchanged same-checkpoint endpoint table reports Fuzzy performance alongside A5.

## Controls and qualifications

A5 control actual endpoint: **10,000 updates**. matched single-task control.
Shared retained comparison updates: 500, 1,000, 1,500, 2,000, 2,500, 3,000, 3,500, 4,000, 4,500, 5,000, 5,500, 6,000, 6,500, 6,521, 7,000, 7,500, 8,000, 8,500, 9,000, 9,500, 10,000. Matching per-task data order verified through 10,000 updates.

| Update | Task metric | Joint/current | Single-task control |
| ---: | --- | ---: | ---: |
| 500 | A5 dev whole word | 0.0000% | 0.0000% |
| 500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 1,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 1,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 1,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 1,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 2,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 2,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 2,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 2,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 3,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 3,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 3,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 3,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 4,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 4,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 4,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 4,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 5,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 5,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 5,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 5,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 6,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 6,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 6,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 6,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 6,521 | A5 dev whole word | 0.0000% | 0.0000% |
| 6,521 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 7,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 7,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 7,500 | A5 dev whole word | 0.0000% | 0.0000% |
| 7,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 8,000 | A5 dev whole word | 0.0000% | 0.0000% |
| 8,000 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 8,500 | A5 dev whole word | 0.0000% | 7.4463% |
| 8,500 | A5 ood_dev whole word | 0.0000% | 0.0000% |
| 9,000 | A5 dev whole word | 0.0000% | 54.7852% |
| 9,500 | A5 dev whole word | 0.0000% | 78.4912% |
| 9,500 | A5 ood_dev whole word | 0.0000% | 13.8916% |
| 10,000 | A5 dev whole word | 0.0000% | 88.4736% |
| 10,000 | A5 ood_dev whole word | 0.0000% | 23.6865% |

FUZZY control actual endpoint: **6,521 updates**. matched single-task control.
Shared retained comparison updates: 1,000, 5,000, 6,521. Matching per-task data order verified through 6,521 updates.

| Update | Task metric | Joint/current | Single-task control |
| ---: | --- | ---: | ---: |
| 1,000 | Fuzzy answer tokens | 13.3163% | 14.6632% |
| 5,000 | Fuzzy answer tokens | 44.2731% | 99.7794% |
| 6,521 | Fuzzy answer tokens | 94.4924% | 99.8854% |

Fuzzy query-ignoring answer-prefix shortcut: 40.6367%. Dense Fuzzy training CE and masked answer evaluation CE have different scopes. Answer exactness is teacher-forced. Causal lookup coverage is not a universal accuracy ceiling.

Single initialization and reused development data; no final confirmation. The historical 10k continuation gate is preserved separately from observed performance through the actual endpoint. Only common retained checkpoints match control training exposure; controls are not extended or extrapolated.

Checkpoint: `step-019810.pt`; SHA256 `82c29baee78e4562f010750bc87d70b659d4b42f1e49680ea40076e4e001ec3d`.

[Training W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/atuxgi4p)

## Exact checkpoint recovery

History, evaluations and control comparisons use the committed parent prefix followed by the resumed updates. Post-boundary parent observations are excluded; their original bytes remain preserved. Matching state at the restored boundary does not establish bitwise equivalence of later updates across GPU instances.

| Stage | Updates used | Raw history SHA256 | Excluded tail bytes |
| --- | --- | --- | ---: |
| train-mixed | 1–8,000 | `f9fe68bac517b209e1400ff7a800ccc08d73574d70c76ff12ba9c794d1590fd1` | 307,831 |
| train-mixed-resume-8000 | 8,001–10,000 | `b75bb7b7569c04e84b38ca454b7cb4b93dfac7da27696cfbb4378683eecbded7` | 0 |
| train-mixed-continue20k | 10,001–19,810 | `591f4c433386340b895cc5e722a4e65757fd87ccff1fb8d17add2ada722c505a` | 0 |

![a5 learning](a5-learning.png)

![a5 prefix full](a5-prefix-full.png)

![a5 prefix boundary](a5-prefix-boundary.png)

![task losses](task-losses.png)

![fuzzy learning](fuzzy-learning.png)

