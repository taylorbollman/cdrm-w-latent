# Mixed throughput at 2,048 examples per task

Fresh brief test on the same H100 80GB: actual A5 length 12 and Fuzzy length 400; two-layer D128/H16/FFN512 restricted-first RT + NextLat, FP32 eager. Physical batch equals logical batch per task. No optimization or architecture change.

Ten warmup updates and thirty timed updates. The 128/1,024 results are the immediately preceding separate benchmark; 2,048 is the new trial. Source, model, data, runtime, optimizer, GPU identity and initialization agree.

| Examples per task | Total examples/update | Tokens/update | Tokens/second | Seconds/update | Allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 52,736 | 39,135 | 1.3475 | 3.11 / 4.39 |
| 1,024 | 2,048 | 421,888 | 271,043 | 1.5565 | 23.39 / 33.00 |
| 2,048 | 4,096 | 843,776 | 454,152 | 1.8579 | 46.58 / 64.94 |

The 2,048-per-task batch completed without OOM: **1.68x throughput versus 1,024/task** (67.6% more), and **11.60x versus 128/task**. An update takes 19.4% longer than 1,024/task while processing twice the tokens.

Ten thousand 2,048-per-task updates imply about **5.16 training hours**, excluding evaluation and retention. That budget presents 20,480,000 examples per task. Short execution timings do not predict the number of optimizer updates needed to learn.

Counts are processed input positions, including native Fuzzy padding. Timing includes synchronized row selection/order hashes, transfers, encoding, CE and NextLat forward/backward on both tasks, clipping and Adam. Startup, dataset verification, logging, evaluations and checkpoints are excluded. All model parameters, gradients and Adam state were finite FP32 after the trial. Benchmark weights were discarded; no larger-batch learning experiment was started.

[W&B: 2,048/task](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/3uyhzypr) · [W&B: previous comparison](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/v1nr4whu)
