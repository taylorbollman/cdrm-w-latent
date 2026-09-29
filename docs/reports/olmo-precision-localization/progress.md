# Numerical localization progress

2026-09-29. Completed milestone in [PR40](https://github.com/taylorbollman/cdrm-w-latent/pull/40),
branch `feat/olmo-precision-localization`, from PR39/main `536458d`.
**All four GPU stages, documentation and evidence retention are complete.**
No GPU work or quality training is queued. Read [results](results.md),
[test ledger](test-ledger.md), [storage receipt](storage-receipt.md) and
[next steps](next-steps.md) before resuming.

## Authorization and scope

The user approved a bounded numerical milestone after packed campaign/restart
readiness. This milestone uses the original isolated T16 NFR fixture, K4 FBT,
native RT at layers 0/15, existing NextLat branches, and unchanged pretrained
OLMo-1B step60000 weights. Two virtual B2 records run on GPU 0; there is no DDP,
optimizer update, graph capture, training run, Q/K normalization or core model
change. Deterministic controls precede CUDA, TF32 is off, and each stage has an
external 900-second timeout. W&B is online under `taylorbollman`.

## Completed stages

| Stage | Scope | Result | W&B |
| --- | --- | --- | --- |
| `bridge-01` | 6 aggregate CE/combined gradient cases; 12 model backwards | Operational guards pass; large BF16/FP32 differences remain | `p6oooxmt` |
| `auxiliary-01` | 8 fixed-hidden latent/KL cases; 16 loss-only backwards | Operational guards pass; small local BF16 layout differences measured | `mdo63etu` |
| `backend-cross-01` | 3 CE aggregate cases; 6 model backwards, including repeated anchors | Ordinary Flash/eager RT equals Flash/Triton RT exactly | `mrgqe7di` |
| `attention-local-01` | 1 repeated CE anchor / 2 model backwards; 8 sites × 3 local VJPs | Anchor exact; all 24 local VJPs and integrity guards pass | `s3jwgeks` |

Initial source freeze `c8575d7`; crossed-backend source `5f1cb79`; local-attention
source `7082225`. Each report's per-file hashes are authoritative. Original
protocols and runtime helpers are frozen and must not be silently edited for
follow-up experiments. New helpers can import them while recording new scope.
Final focused CPU suite: 44 passed in 3.84s, with the known CPU RMSNorm dispatch
warning. Earlier suites overlap; do not sum them as unique tests.

## Interpretation and next decision

- Combined BF16 math/eager versus FP32 gradient L2: 81.50%; current BF16
  Flash/Triton versus FP32: 85.96%. Same-state original endpoints reproduce exactly.
- CE Flash/eager versus Flash/Triton:zero full-gradient difference. Both differ
  79.85% from math/eager. This separates the ordinary attention backend change
  from the RT tile change on this fixture; it does not clear longer contexts.
- Fixed-hidden auxiliary BF16 prepared/sparse hidden-gradient errors: 0.1416%
  latent and 0.2016% KL. FP32 layout errors below 8e-7. These are local quantities,
  not directly comparable amplification factors for parameter gradients.
- At identical actual Q/K/V and incoming cotangents, Flash versus FP32 math has
  0.160–0.186% output error and 0.174–1.501% Q/K/V gradient error over eight sites.
  All local Flash outputs reproduce captured output bytes. 112 original ordinary
  calls were observed; 112 checkpoint replays were excluded. This supports
  investigating sensitivity through the assembled model, not a large local
  attention-backward error. It does not prove harmlessness in training.

Recommend a separately frozen eight-case within-arm BF16/FP32 CE comparison
for ordinary / RT / K4 FBT / K4 FBT+RT computation (existing N/NR/NF/NFR arms,
zero auxiliary cotangents). Preserve data/noise and compare shared backbone
as well as total gradients. Only then select one precision boundary. This
follow-up is **not launched**. The original 3.40224% isolated/1.6953% packed
layout qualifications and broader BF16/FP32 qualification remain open.

## Persistence

Runtime evidence: `.runtime/olmo-precision-localization/`.
Cloud namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T043105Z/`.
The namespace is historical; these probes used one device/process. All four
diagnostic stages are retained and independently read back: 8 objects,
318 inventory members and 308 source pairs. Both independent audit records and
final code/test/report evidence are also retained in verified
`precision-closeout-01`; see the storage receipt for generation/hash pins.
No new trained checkpoint is created or required.
The retained anchors are 3.52 MB auxiliary tensors and 7.02 MB attention tensors.
Runtime code is committed and pushed through `7082225`; final reports and
cross-compaction records are included in PR40. The next proposed experiment
has not been launched, and all prior numerical qualifications remain visible.
