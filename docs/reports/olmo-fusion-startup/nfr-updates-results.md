# Four-update full-NFR precision and optimizer diagnostic

2026-09-29. Both trajectories completed four finite full-model updates, and both improved on the same small held-out fixture when evaluated in FP32. BF16 still produced materially different gradients and Adam updates. This is a bounded functionality pass, with a numerical qualification; it is not production BF16 clearance or evidence of equivalent learning.

Execution finished with exit 0 in **3,512.8 seconds** (58.5 minutes). [W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rmz59xy0). The [frozen protocol](nfr-updates-protocol.md) and [full report](../../../.runtime/olmo-fusion-startup/nfr-updates-01/report.json) retain exact settings and full-precision geometry. Report SHA256: `f24c6035f9035189bcb9fefd8ee12ceb42d3f249aca7c7221beb175d3f0da9dd`.

The common starting state was the original OLMo step 60000 backbone and seeded predictor, plus the strictly imported FP32 fusion-only update 128 weights. NFR selected native RT layers 0 and 15, `alpha=1` on all K4 passes, `beta=1`, jitter 0.02 and combined CE/latent/KL with the original pass weights. All 1,267,879,936 parameters participated: 1,176,764,416 backbone, 8,388,608 fusion and 82,726,912 predictor. Both paths started with identical **fresh** campaign AdamW state; the earlier fusion-only Adam history was deliberately discarded.

FP32 used math attention/eager RT. BF16 used ordinary Flash SDPA/native Triton RT with FP32 masters. Both used unfused AdamW, gradient clipping at 1, TF32 off and deterministic controls. The learning rate followed actual valid-input exposure, from 0.0000200000 on update 1 to 0.0000200865 on update 4; the next LR was 0.0000201155. The 52,428,800-token warmup was not accelerated.

Each path consumed source selections 144–147: isolated B8/T128, 8,192 CE targets per optimizer update, **39 physical backwards**, 32,768 CE targets, 32,768 latent pairs, 32,472 KL triples and 33,638 valid input tokens in total. These are tiny diagnostic updates, not the planned 524,288-input-token campaign updates. The legacy documents count is 300 window presentations, not 300 unique documents.

**Only update 1 compares the same model and optimizer state.** Updates 2–4 use the same data/noise and schedule but independently evolved states. Their differences combine arithmetic effects with trajectory divergence; they cannot be read as same-state backward errors.

Relative L2 differences below use FP32 as reference. Master delta is the actual before/after FP32 parameter change, including clipping, Adam and weight decay. Values are percentages, rounded; moment columns compare the stored Adam first and second moments.

| Component | Update | Raw gradient | Clipped gradient | Master delta | First moment | Second moment |
|---|---:|---:|---:|---:|---:|---:|
| backbone | 1 | 13.447% | 13.416% | 17.469% | 13.416% | 25.970% |
| backbone | 2 | 17.259% | 17.298% | 19.058% | 15.302% | 7.286% |
| backbone | 3 | 34.608% | 34.873% | 23.561% | 20.979% | 9.859% |
| backbone | 4 | 133.276% | 70.116% | 30.309% | 30.575% | 18.788% |
| fusion | 1 | 11.938% | 11.952% | 26.581% | 11.952% | 11.274% |
| fusion | 2 | 28.311% | 28.352% | 32.349% | 21.363% | 17.898% |
| fusion | 3 | 44.214% | 44.341% | 40.161% | 27.287% | 22.933% |
| fusion | 4 | 88.028% | 55.884% | 37.674% | 29.865% | 23.205% |
| predictor | 1 | 0.785% | 0.863% | 1.505% | 0.863% | 0.785% |
| predictor | 2 | 7.838% | 8.150% | 8.030% | 5.612% | 1.839% |
| predictor | 3 | 22.273% | 22.663% | 15.844% | 14.108% | 10.912% |
| predictor | 4 | 115.583% | 60.632% | 30.040% | 27.840% | 34.580% |

At the identical initial state, the backbone update differed by **17.469%** (cosine 0.984763), and the fusion update by **26.581%** (cosine 0.964696). Gradient clipping therefore did not remove the directional discrepancy. First-step first-moment differences are the scaled clipped-gradient differences, not independent evidence.

Absolute backbone geometry prevents mistaking a shrinking reference norm for increasing absolute error:

| Update | FP32 raw norm | Raw difference norm | Raw cosine | FP32 delta norm | Delta difference norm | Delta cosine |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 121.475310 | 16.334222 | 0.991006 | 0.273607 | 0.047796 | 0.984763 |
| 2 | 85.100979 | 14.687948 | 0.985008 | 0.182608 | 0.034802 | 0.981994 |
| 3 | 33.136347 | 11.467710 | 0.938792 | 0.147408 | 0.034730 | 0.972482 |
| 4 | 24.868213 | 33.143281 | 0.758752 | 0.132405 | 0.040130 | 0.953219 |

From updates 1–3 the relative backbone gradient difference grew while its absolute difference shrank. **Update 4 is different:** its absolute difference rose to 33.1433, with FP32/BF16 backbone norms 24.8682/47.7839 and cosine 0.758752. Total raw-gradient norms were 29.5526/55.7683; both were clipped to 1. The resulting backbone delta difference was 0.0401304 against a 0.132405 reference, or 30.309%. This is a real trajectory discrepancy, while the run still remained finite. Fusion delta differences were 0.00708173 at update 1 and 0.00462797 at update 4; percentages alone do not describe their absolute scale.

Training means below compare matching batches at each step. Different steps contain different data, so the downward trend is not itself held-out learning evidence. In particular, opposing CE and KL changes can hide behind a small combined-objective gap.

| Update | CE FP32 | CE BF16 | KL FP32 | KL BF16 | Latent FP32 | Latent BF16 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 6.376691 | 6.377260 | 5.747547 | 5.747770 | 0.864184 | 0.864047 |
| 2 | 6.178183 | 6.251622 | 4.132408 | 4.107127 | 0.797540 | 0.797087 |
| 3 | 5.761438 | 5.728259 | 3.127920 | 3.201423 | 0.734919 | 0.736726 |
| 4 | 5.445023 | 5.458650 | 2.908844 | 2.911607 | 0.669461 | 0.666651 |

The same four long dev documents were evaluated using **FP32 for every entry** below: 508 CE/latent targets and 504 KL triples. The initial evaluation was measured once and shared because starting states were identical. Endpoint values consequently measure the trained states without adding a different evaluation precision.

| Training state | CE | KL | Latent | Combined |
|---|---:|---:|---:|---:|
| Common initial state | 6.309803 | 6.221116 | 0.861473 | 13.392392 |
| FP32 after 4 updates | 5.675359 | 2.748084 | 0.598274 | 9.021716 |
| BF16 after 4 updates | 5.719054 | 2.881924 | 0.616070 | 9.217047 |

Both endpoints improved on this fixed fixture. BF16's endpoint was higher by **0.043695 CE nats/target**, 0.133840 KL and 0.017796 latent loss; combined difference 0.195332. Four updates and four repeatedly inspected dev documents cannot establish long-run stability, task quality, or an acceptable general BF16 error budget. The gradient discrepancy has an optimizer effect; these results also do not establish that it causes a failed training run.

Independent closeout checked all eight row contracts, matching inputs/LRs/RNG, exact saved-state chains, global loss normalization, clipping arithmetic, all 20 cross-comparison geometry sets, the three new source/snapshot pins, all held-out immutability checks and final boundary identities. All eight final integrity checks passed; the report pins 136 sources. The focused implementation tests passed 17/17, including a literal canonical-update oracle and strict CPU checkpoint recovery. No additional full GPU restart was performed in this diagnostic.

Full model/Adam/scheduler/RNG/cursor checkpoints at updates 2 and 4 were retained for both paths: **60,858,875,412 bytes** total. Each producer receipt verifies the complete cloud download by SHA256; this audit matched generations, hashes, byte lengths and local file sizes without duplicating the large downloads. Final endpoints share this prefix:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/nfr-updates-01/`

| Endpoint object | Generation | SHA256 |
|---|---|---|
| `fp32_math_eager-update-000004.pt` | `1790677290079286` | `5744b6b5704aac595e8d826dcaf96d72fb7fc78236dfe1bbb52e78991e631937` |
| `bf16_flash_triton-update-000004.pt` | `1790677664012201` | `6030c92f1c09d2c561ca173eee816316ae51de4bc66b368a34996577b8c1dd0a` |

This run spent substantial time copying/checking full CPU states, computing FP64 vector geometry and verifying four 15.2 GB checkpoints. Each paired step performed ten self-summary geometries, five cross-comparisons and a parameter-separation comparison. Its 58.5-minute wall time is **not a training-throughput measurement**. The next execution bridge and any conditional continuation retain their separately declared gates and budgets.
