# Packed campaign readiness

2026-09-29. Operational checks and numerical qualification are separate.
Runtime implementation: `f0be95d`; model policy and precision probe: `b9985bc`.
This is readiness work on the retained Dolma coverage fixture, not a production
mixture or learning-quality comparison.

## Completed implementation and acceptance

- Opt-in `continuous-stream-v1` packs actual documents with stride T. CE crosses
  document boundaries; NextLat latent/KL targets do not. Attention, RT and FBT
  carry history within the row. State and targets never cross chunk boundaries.
  Legacy isolated-document behavior remains the default.
- Disk-backed chunks preserve true provenance, verified source files and a
  committed cursor. Prefetch and rank partitioning do not consume the stream.
- All 6,947,277 train tokens / 6,785 chunks match an independent document-stream
  oracle. The cloud-restored index is byte-verified and has identical chunk
  keys, counts and cursor behavior at a different path.
- Two-H100 tiny eager and graph probes pass all eight combinations (88 gates
  each). Pretrained B/NFR graph probes pass 22 operational gates against the
  same prepared local arithmetic. Packed NFR independently retains a 1.6953%
  BF16 sparse/dense gradient qualification failure.
- Broad CPU regression: 971 passed; final focused data hardening: 20 passed
  (overlapping scope). See [test ledger](test-ledger.md) for exact evidence.

## Material numerical finding

The bounded [component diagnostic](precision-assessment.md) reproduces the prior
3.40224% isolated BF16 layout difference: CE-only gradients are exactly equal,
while latent-only and KL-only gradients differ 2.173% and 2.284%. This points
toward auxiliary-loss cotangents and their propagation, not distributed
communication or CUDA graph replay. It does not establish the responsible
operation or harmlessness.

More significantly, both BF16 paths differ about **86% relative L2** from the
common full-FP32 combined gradient, with cosine about **0.51** and roughly half
the FP32 norm. CE alone also shows a large gap. This comparison changes Flash
to math SDPA and Triton to eager RT alongside precision, so it cannot yet
attribute the result to BF16 rounding or to a specific backend. Scalar losses
change far less; that does not establish gradient-direction agreement.

Recommend bounded fixed-hidden-state auxiliary cotangent checks and a BF16
eager/math bridge before longer training or architectural changes such as Q/K
normalization. No numerical budget was relaxed, no model architecture changed,
and no training-quality claim follows from operational acceptance.

## Full-length actual-data recovery

NFR, K4 FBT, native RT at layers 0/15 on every pass, both NextLat losses, T1024,
B12/rank on two H100s. Each logical update contains 524,288 valid inputs in 512
real rows, spread over 22 slots/rank. The final synchronization slot has eight
real rows on rank 0 and none on rank 1.

The write phase passed all 13 gates and two updates in 835.55 seconds, including
setup, hashes and checkpoint I/O. Model/Adam replicas, real loss counts, cursor,
RNG, graph preparation and source pins agree as required. Checkpoint after
update one is retained in GCS; its bytes are being downloaded and verified for
fresh-process recovery. The second update ran on the original live graph.

| Update | Global valid input tokens/s | Loader and jitter | Captured backward | Adam and cursor commit | Peak reserved/GPU | Sampled free/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 3,968.23 | 7.823 s | 123.702 s | 0.597 s | 58.90 GiB | 14.40 GiB |
| Original live-graph continuation | 3,957.59 | 7.953 s | 123.947 s | 0.576 s | 58.90 GiB | 14.40 GiB |

Rates count input tokens once across ranks, not K4 pass tokens, loss targets or
padding. Timed segments include loader/jitter, refill/preflight, graph backward,
NCCL, clipping, Adam/scheduler and cursor commit. Diagnostic hashing, state scans,
reporting, warmup/capture and checkpoint I/O are excluded. This is a directional
complete-update rate over a small sample, not a long-run throughput estimate.

The first fresh-process attempt passed configuration, restored-state and cold
Adam-resident DDP/capture checks, but **failed exact next-update continuation**.
Inputs, jitter, RNG, counts and all scalar losses agree. All four predictor
gradients match; the 65 backbone and two fusion gradients do not. Preclip norm
is 196.2971954 versus 196.2953186, which does not bound vector error. Cold capture
left 14.95 GiB free/GPU. The failed attempt is retained separately.

The runner omitted deterministic setup used by our earlier distributed harness.
A bounded fixed-input Flash-backward test and a fresh deterministic write/resume
pair are now planned under the protocol addendum. **Restart remains unqualified**
at this entry; neither scalar-loss agreement nor a small norm change clears it.
The recovery checkpoint is a readiness fixture, not an approved production
quality-training starting point.

See [usage](usage.md), [progress](progress.md), [frozen protocol](protocol.md)
and the [W&B project](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat).
