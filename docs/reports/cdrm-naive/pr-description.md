Add an opt-in ordinary-autograd FP32 CDRM reference and complete a six-block synthetic pilot against matched SEQ baselines. CDRM reads earlier permanent memory plus current temporary K/V, writes after the read, and bridges its proposed state into the ordinary suffix. It shares canonical early-block weights and adds only two nonzero adapters. Lambda-zero bypass matches SEQ; the active same-depth control is implemented for later research. Existing defaults and legacy HF output tuples remain compatible.

The harness preserves native MAD supervision, exact initialization/data-order pairing, checkpoint/source identities, exact optimizer/RNG/scheduler resume, numerical monitoring and separate development/final data. Following SEQ-first screening, selective copying V16/T256/K96 was chosen because SEQ learned substantially while retaining errors. Both architectures completed 45 epochs for seeds 0/1. A runtime-only policy selected that common endpoint; final checkpoint roles were frozen before generating the fresh 1,280-example final split.

| Held-out common epoch 45 | SEQ token accuracy | CDRM token accuracy | SEQ / CDRM exact matches |
|---|---:|---:|---:|
| Seed 0 | 81.497% | 77.097% | 0 / 0 |
| Seed 1 | 87.877% | 89.104% | 2 / 6 |

There is **no consistent CDRM advantage** across seeds or checkpoint-selection policies. Six unique final evaluations cover all eight endpoint/best-dev roles; complete metrics and checkpoint epochs are in [results.md](results.md). This is a two-seed, one-LR, partial-horizon pilot, not a tuned paper reproduction. The naïve implementation costs 18.82× SEQ update time and 1.75× peak allocated memory at the selected B128 shape, with 2.76% more parameters.

Validation: 60 CDRM reference CPU tests, 60 existing CPU regressions (20 CUDA-only skips), 29 MAD-data tests and four HF compatibility tests passed. GPU numerical/operational checks cover the actual six-block B128 FP32 path, exact resume and repeated-batch fitting. All four trajectories and six final evaluations passed independent pairing, payload/hash and count audits. BF16 and optimized CDRM execution remain deferred.

All 613 closed session files are retained at `gs://fast-chunks/cdrm-w-latent/cdrm-naive/20260907T123830Z/`, with checksum comparisons and an independently verified inventory. The [storage record](storage.json) identifies the separate final-document snapshot and its external verification receipt.
