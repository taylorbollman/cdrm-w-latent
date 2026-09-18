# D128 mixed A5/Fuzzy batch throughput

Actual A5 length12 and Fuzzy length400; two-layer restricted-first RT + NextLat, FP32 eager. Each candidate starts from the same initialization. These are execution measurements, not a learning comparison.

10 discarded warmup updates followed by30 timed updates per candidate. Physical batch equals logical batch per task; total mixed batch is twice the listed count.

| Examples per task | Status | Mean seconds/update | Total examples/second | Allocated GiB | Reserved GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | complete | 1.4153 | 180.89 | 3.11 | 4.39 |
| 256 | complete | 1.4549 | 351.92 | 6.00 | 8.48 |
| 512 | complete | 1.5262 | 670.96 | 11.80 | 16.68 |
| 1024 | complete | 1.6311 | 1255.56 | 23.39 | 33.00 |

Timing includes selection/transfers, both forward/backward passes, clipping and Adam. W&B logging, initialization, data validation and final finite-state audits are excluded. Sequential short trials on one GPU do not establish time to convergence.

[W&B timing and memory charts](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/bkg4gs26)
