# A5 L1R RT + NextLat: BF16 precision repeat

BF16 completed at **80,000 updates**; FP32 reference completed 80,000. Latest common full checkpoint: **80000**.

Both models use D512, two RT layers (first window2), B1024 and identical initial tensors/data order.

| Update | Arm | Length12 whole word | Length36 whole word | Length36 token |
|---:|---|---:|---:|---:|
| 1,000 | FP32 | 95.6348% | 46.7471% | 88.3541% |
| 1,000 | BF16 | 95.1260% | 45.6699% | 87.8207% |
| 5,000 | FP32 | 99.3350% | 74.0674% | 95.4556% |
| 5,000 | BF16 | 99.2861% | 72.8818% | 95.0901% |
| 10,000 | FP32 | 99.4775% | 85.7910% | 97.3167% |
| 10,000 | BF16 | 99.4766% | 86.2666% | 97.4885% |
| 20,000 | FP32 | 99.8066% | 96.3369% | 99.2835% |
| 20,000 | BF16 | 99.8389% | 97.4062% | 99.4980% |
| 25,000 | FP32 | 99.6807% | 96.2090% | 99.1667% |
| 25,000 | BF16 | 99.8369% | 97.6084% | 99.5118% |
| 30,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 30,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |
| 40,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 40,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |
| 50,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 50,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |
| 60,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 60,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |
| 70,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 70,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |
| 80,000 | FP32 | 100.0000% | 100.0000% | 100.0000% |
| 80,000 | BF16 | 100.0000% | 100.0000% | 100.0000% |

At the actual BF16 endpoint, length-12 whole-word accuracy is **100.0000%**, and length-36 whole-word accuracy is **100.0000%** (102,400 words each).

| Arm | Actual endpoint | Training-loop minutes | Total minutes |
|---|---:|---:|---:|
| FP32 | 80,000 | 74.48 | 76.74 |
| BF16 | 80,000 | 79.63 | 82.33 |

Training-loop time includes per-update diagnostics and excludes evaluations/checkpointing/logging. Unequal terminal budgets are not throughput comparisons. Matched-exposure timing is retained in report.json.

One paired development seed; precision is the intended change, not a replicated equivalence result. Both arms use the same D512 two-layer restricted-first RT + NextLat, initial tensors, optimizer, A5 corpus and word order. BF16 uses protected bf16_fp32_state with FP32 parameters/gradients/Adam. Evaluation uses each arm's native execution precision. Routine evaluations use 4096 words; full checkpoints use 102400. Compare full measurements at common optimizer updates. An unmatched early-stop endpoint is descriptive. Final confirmation and latent rollout remain unused.

[Learning curves](whole-word-vs-updates.pdf) · [Length36 prefix](length36-prefix.pdf) · [Training losses](training-losses.pdf) · [Metrics](metrics.csv)
