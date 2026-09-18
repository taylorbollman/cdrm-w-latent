# Mixed throughput at 3,072 examples per task

Fresh user-requested test on the same H100 80GB, with actual A5 length 12 and Fuzzy length 400. Model remains the two-layer D128/H16/FFN512 restricted-first RT + NextLat, FP32 eager. Physical and logical batch sizes are equal per task.

Ten warmup updates followed by thirty synchronized timed full training updates. Smaller-batch results come from the preceding short trials. Source, model/data hashes, runtime, GPU identity, optimizer, initialization and initial ordered row prefixes match exactly; candidate batch size is the only contract difference.

| Examples per task | Total examples/update | Tokens/update | Tokens/second | Seconds/update | Allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 52,736 | 39,135 | 1.3475 | 3.11 / 4.39 |
| 1,024 | 2,048 | 421,888 | 271,043 | 1.5565 | 23.39 / 33.00 |
| 2,048 | 4,096 | 843,776 | 454,152 | 1.8579 | 46.58 / 64.94 |
| 3,072 | 6,144 | 1,265,664 | 593,574 | 2.1323 | 69.77 / 71.98 |

The 3,072-per-task batch completed all forty updates without OOM and with finite FP32 parameters, gradients and Adam state. Throughput is **1.307x versus 2,048/task** (+30.7%), **2.190x versus 1,024/task**, and **15.17x versus 128/task**.

An update takes 14.8% longer than 2,048/task and processes 50% more tokens. Ten thousand updates imply approximately **5.92 training hours**, excluding evaluation and retention, with 30.72 million example presentations per task.

This is the fastest tested batch for the current shape. Allocated memory is substantially higher than at 2,048/task, so the smaller batch leaves more room for future architecture or sequence-length changes. Reserved memory includes allocator cache and is not additional to allocated memory.

Timing includes selection/order hashes, transfers, encoding, both task CE and NextLat forward/backward passes, diagnostics, clipping and Adam. It excludes initialization/data verification, logging, evaluation and checkpointing. Tokens count processed input positions including native padding. Benchmark weights were discarded; no larger-batch mixed learning run has started. These results do not establish the number of optimizer steps required for learning.

[W&B: 3,072/task](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/opdx9b15)
