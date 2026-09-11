**CDRM naïve FP32 reference and six-block synthetic pilot — PR summary**

This change implements a trainable Cross-Depth Read-Conditioned Recurrent Memory (CDRM) reference and executes a bounded, paired synthetic-learning pilot. The implementation passes its declared numerical and recovery checks. The pilot shows substantial learning, but **does not establish a consistent CDRM advantage over the matched ordinary Transformer**.

CDRM adds a causal side-memory scan to an ordinary Transformer. In the selected six-block model, ordinary blocks 0–3 produce early and late previews after blocks 1 and 3. The side scan combines the current deep preview with a read of earlier memory, writes the resulting record for subsequent tokens, and adds a small correction before ordinary suffix blocks 4–5. Each current read sees earlier permanent records and its own temporary K/V pair; its new permanent write becomes available only to later tokens.

The side computation shares the early block's canonical attention, normalization and MLP weights. It adds only two nonzero, bias-free adapters. History remains connected to ordinary PyTorch autograd and resets on each independent forward; there is no custom backward or detached temporal history. The fixed pilot gates are epsilon 0.1, rho 1 and lambda 0.01. Lambda zero gives the matched SEQ bypass; rho zero still leaves an active side branch. An active same-depth writer and current-only read are implemented as controls, with the broader research arms deferred. CDRM is opt-in, and a compatibility fix preserves the existing HF output tuple layout.

The accompanying workflow adds pinned official MAD data preparation and audits, resolved model profiles, deterministic training, checkpoint-role ledgers, exact continuation, container execution receipts, and JSON/CSV/PNG/SVG reporting. It preserves native supervision alignment, learned normalization, ALiBi and the declared FP32 execution policy. Existing R3/Stage B/BF16 artifacts remain separate.

Following the updated guidance, **SEQ6 screened harder settings first**. Neither of the two-seed, 25-epoch screens satisfied the early-ace rule: at least 99.9% token accuracy and 99% whole-example exact match for three consecutive epochs by epoch 10. Recall at vocabulary 128 remained below its simple modal baseline and was deferred. Selective copying at length 256, vocabulary 16 and 96 copied tokens showed substantial learning with remaining errors, so it received the CDRM comparison. The compatible SEQ baselines were reused. The earlier twelve-block SEQ recall calibration remains historical; no twelve-block CDRM research run launched.

The selected comparison uses six blocks, width 128, 16 attention heads, MLP width 512 and physical batch 128 on one H100. Both architectures use corresponding fresh backbone initialization, identical ordered data, AdamW at learning rate 5e-4, clipping at norm 1, and FP32 with autocast/TF32 disabled. Both seeds completed **45 epochs / 4,500 updates per architecture**, continuing exactly from epoch 25. The cosine schedule retains its original 200-epoch horizon. The common endpoint was chosen using measured runtime rather than accuracy.

There are 12,800 fixed training examples and 1,280 development examples. Only after the setting, endpoint and checkpoint roles were frozen did we generate the independent 1,280-example final split. All final examples passed the task oracle; no exact input overlaps were found across splits, and all previously frozen files retained their hashes. Each example requires all 96 copied tokens to be correct for an exact match. Final results therefore use 122,880 scored tokens and 1,280 directly counted sequences.

| Held-out result at epoch 45 | Answer CE | Token accuracy | Exact sequences |
|---|---:|---:|---:|
| SEQ6, seed 0 | 0.397963 | 81.50% | 0 / 1,280 |
| CDRM6, seed 0 | 0.435744 | 77.10% | 0 / 1,280 |
| SEQ6, seed 1 | 0.304838 | 87.88% | 2 / 1,280 |
| CDRM6, seed 1 | 0.259957 | 89.10% | 6 / 1,280 |

CDRM is behind on seed 0 and ahead on seed 1 at the common endpoint. Checkpoints selected by minimum development answer CE also give mixed results: seed-0 CDRM selects epoch 44 and reaches 81.74% final token accuracy, versus SEQ's 81.50% at epoch 45, but has worse CE; seed-1 SEQ selects epoch 41 and reaches 90.91% with 8 exact sequences, versus CDRM's 89.10% and 6 at epoch 45. Those roles were fixed before final evaluation. Six unique checkpoint evaluations cover all eight endpoint/best-development roles. The final order-ignoring modal baseline is 12.13% token accuracy and zero exact matches.

Validation includes 60 CDRM reference CPU tests, 60 existing foundation/conversion CPU tests, four HF compatibility tests and 29 MAD data tests. Twenty CUDA-only regression cases were skipped in the explicit CPU run. Targeted GPU checks separately passed at the declared shapes, including the selected six-block B128 configuration. Evidence covers an independent FP64 forward/gradient oracle, causality, temporal credit, shared parameter ownership and gradient accumulation, exact lambda-zero equivalence, finite FP32 training, checkpoint round trip and exact midpoint recovery. A tiny repeated-batch operational check reaches 100% training accuracy; it is a fitting diagnostic. The completed pilot audit verifies 51 matching initial backbone tensors per seed, all 4,500 paired update records, exact inherited epoch-25 history, and finite parameters/gradients/optimizer moments at the recorded checks.

| Selected-shape update benchmark | SEQ6 | CDRM6 |
|---|---:|---:|
| Parameters | 1,186,944 | 1,219,712 |
| Mean update time | 0.0458 s | 0.8621 s |
| Peak allocated GPU memory | 6.23 GiB | 10.91 GiB |

The naïve reference costs **18.82× the update time** and **1.75× the allocated memory**, with 32,768 extra parameters. These measurements include loss, backward, clipping and Adam after warmup; they exclude evaluation, startup and checkpointing. Equal-update results do not establish equal-time efficiency.

This is a two-seed, one-learning-rate, partial-horizon pilot on an adapted six-block configuration, not a reproduction of the paper's tuned one-block results. Whole-sequence performance remains low, development curves are noisy and clipping is frequent. No broad positive or negative architectural conclusion follows. The active same-depth writer is a prepared next attribution comparison; it has not been trained as a research arm. BF16 clearance, optimized CDRM kernels, distributed training, accumulation, cached decoding, packing and latent objectives remain outside this milestone. Further experiment choices informed by these final results require fresh confirmatory data.

The complete evidence is linked from [results.md](results.md), with [architecture](../../cdrm-architecture.md), [usage and exact recovery commands](../../cdrm-naive-usage.md), [configuration profiles](../../../configs/cdrm/README.md), [paired learning curves](development-learning-curves.png), and [final checkpoint-role plan](final-evaluation-plan.json). Local data/checkpoints/logs live under `.runtime/cdrm-naive/20260907T123830Z/`. All 613 closed session files (3.98 GB) are retained at `gs://fast-chunks/cdrm-w-latent/cdrm-naive/20260907T123830Z/`. Checksum comparisons found no differences and the downloaded inventory matched SHA256; large artifacts were not individually downloaded. The [storage record](storage.json) identifies the final-document snapshot and separate verification receipts.

This document summarizes the implemented changes and executed evidence in the workspace. No GitHub PR has been opened or merged as part of this milestone.
