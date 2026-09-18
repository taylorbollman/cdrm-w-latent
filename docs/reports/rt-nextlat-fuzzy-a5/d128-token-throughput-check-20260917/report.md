# Mixed A5/Fuzzy token throughput: brief recheck

Completed on the H100 80GB with the unchanged two-layer D128/H16/FFN512 restricted-first RT + NextLat, FP32 eager. Both real task streams were used: A5 length12 and Fuzzy length400. Physical batch equals the listed per-task logical batch.

Ten discarded warmup updates and thirty synchronized timed full updates per batch. Each candidate starts from the same initialization; weights are discarded.

| Examples per task | Total examples/update | Tokens/update | Tokens/second | Seconds/update | Allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 52,736 | 39,135 | 1.3475 | 3.11 / 4.39 |
| 1,024 | 2,048 | 421,888 | 271,043 | 1.5565 | 23.39 / 33.00 |

**Larger-batch throughput: 6.93x.** Each update processes eight times as many tokens and takes 15.5% longer. Ten thousand larger-batch updates imply about 4.32 training hours, excluding evaluation and retention.

Timing includes selection/order hashing, host-to-device transfer, encoding, both task forward/backward passes with NextLat, diagnostics, gradient clipping and Adam. It excludes initialization/data verification, W&B/local logging, evaluation and checkpoints. Tokens count input positions, including native Fuzzy padding; answer-only tokens and the auxiliary NextLat work are not separate additions to the count.

This is a short directional execution comparison, not evidence of the number of updates needed to learn. No model, precision, compiler or optimizer optimization was performed. Both candidates completed with finite FP32 model/Adam state.

[W&B timing, memory and throughput records](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/v1nr4whu)

The preceding mixed run was saved, fully evaluated and retained at update19,810 before this benchmark. Its full A5 L36 whole-word accuracy was57.6582% and Fuzzy answer accuracy98.8509%. The proposed larger-batch learning baseline has not started.

[Preceding mixed results](../d128-fresh-mixed-20k-extension/report.md)
