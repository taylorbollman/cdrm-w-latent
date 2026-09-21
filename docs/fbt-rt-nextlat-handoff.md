# Pretrained OpenELM / RT / FBT / NextLat implementation handoff

Updated 2026-09-21. Read this first after compaction or interruption.

## Current authorization and scope

The user approved the staged research plan and asked us to implement the first
planned PR. The delivered milestone is **A: native OpenELM-1.1B checkpoint import
and ordinary-model fidelity**. The implementation and validation are complete.
The subsequent RT, FBT, NextLat and learning milestones remain staged; do not
start a long learning experiment simply because the platform is being built.

**Stage A implementation and validation are complete.** See
[first-PR results](reports/openelm-import/results.md) and
[usage](openelm-import-usage.md). Final GPU evidence is
`.runtime/openelm-import/validation-final/report.json`, W&B
[`z4jzbq02`](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/z4jzbq02).
The actual checkpoint matches the independent native reference exactly in
FP32 and BF16 math-backend outputs and all 226 parameter gradients. Default
cuDNN fused BF16 source parity also passes (global gradient relative L2
2.07e-9). There is no training job to resume. The next research code milestone
is the native-block RT reference after review of this first PR.

Branch: `feat/openelm-checkpoint-import`, based on `330a310` (`Pre-OpenELM dev`).
The working tree was clean at the start. Preserve subsequent user changes.
Model source for this new lineage belongs in `cdrm/pretrained/`; the historical
Recurrent OLMo/synthetic implementations remain their own reference lineage.

## Read in order

1. [Current research plan](fbt-rt-nextlat-research-plan-v2.md): authoritative
   model choice, architecture, comparisons and budget review points.
2. [Attention and geometry audit](fbt-openelm-attention-compatibility.md):
   causal attention, native shapes, current RT limitations and possible FA4 use.
3. [Numerical handoff](rt-numerical-handoff.md): reusable checks and historical
   results, not an instruction to reopen the old precision study.
4. [Architecture/budget/auxiliary review](fbt-architecture-and-auxiliary-loss-review.md)
   and [Nanochat budget audit](fbt-nanochat-budget-audit.md) when interpreting
   training budgets or proposing later experiments.

The original [OLMo proposal](fbt-rt-nextlat-pretrained-plan.md) is historical;
its model selection and layer indices have been superseded.

## Frozen choices

- Primary model: **OpenELM-1.1B**, 28 layers, residual width 2048 throughout,
  native layer-varying attention/FFN dimensions, head dimension 64, 4:1 GQA,
  SwiGLU, learned headwise Q/K RMSNorm before RoPE, tied input/readout.
- Initial artifact: native individual 300k checkpoint, not HF final/average.
  Preserve all **32,128 vocabulary rows**, native padding ID 32,000, logits
  denominator and the authorized native Llama SentencePiece tokenizer.
- CoreNet source pin: `f9f83e616a34d02c422733a06a3fe5bde63ae575`.
- HF code/config reference pin:
  `apple/OpenELM-1_1B@ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f`.
- Model-only checkpoint URL:
  `https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_299999.pt`.
- OpenELM-450M is an optional fallback. Nanochat experiments are deprioritized.
- Feature matching is retained as a **late diagnostic**, not an initial
  implementation requirement or training objective. Semantic Tube is future work.

## First PR acceptance and status

The selected validation uses physical batch 1 and 44/64-token fixtures; all
parameter gradients are compared on the 44-token fixture. This is bounded
ordinary-model coverage, not a large-batch or long-context clearance.

- [x] Dedicated branch and clean starting state recorded.
- [x] GPU environment verified **inside the project container**: H100 80GB;
  PyTorch `2.13.0a0+8145d630e8.nv26.06`; CUDA available; GPU initially idle.
- [x] Pinned source/config/tokenizer artifacts with hashes and provenance.
- [x] Native checkpoint downloaded, hashed and strictly loaded; exact tensor
  keys/shapes/count recorded; no cropping, retie conversion or reinitialization.
- [x] Ordinary adapter preserving native attention/norm/RoPE/MLP and tying.
- [x] Independent pinned-source reference and small causal/cache/masking tests.
- [x] Actual-checkpoint bounded output/loss/gradient and cache comparison.
- [x] Tied ownership preserved across optimizer construction and save/reload.
- [x] Actual mixed precision and attention dispatch scope explicitly recorded.
- [x] W&B diagnostic record and durable local results.
- [x] Verified GCS retention, including the native checkpoint, evidence archive
  and [storage receipt](reports/openelm-import/storage-receipt.json).
- [x] Usage guide and first-PR summary; review before recurrence implementation.

The final scoped CPU-container suite passed **60 tests**, including source
fidelity, cached training gradients, a tied AdamW save/resume check and artifact
retention checks. Actual GPU validation is separate. The small `ftfy==6.3.1`
dependency was added and the project image rebuilt.

One important porting correction must survive future refactors: native RMSNorm
shares one `x.float()` node across its two derivative branches. Separate casts
have equal forward values but round BF16 backward contributions before they
combine. Keep the shared cast in RMSNorm and RoPE; the native gradient regression
tests cover this. This is native parity, not an extra precision policy.

This PR does not claim RT, FBT or NextLat compatibility has already been tested
in OpenELM. It establishes the unchanged pretrained function they will extend.

## Architectural contracts for subsequent PRs

- Keep independent FBT, RT and NextLat switches, supporting all eight modes.
  Standalone RT is one pass; it does not execute an unused ordinary pass.
- In the FBT multipass API, K counts complete stack passes and K1 is ordinary.
  Finite passes consume shifted previous-pass states; exact online decoding
  consumes the freshly completed previous-token state. Preserve prefix/cache
  boundaries, attached cross-pass gradients and explicit pass-loss reductions.
- Initial selected RT set is `{0}`; top layer is index 27. Use full causal
  history. The restricted-first synthetic architecture is not this default.
- Persistent RT writes use the layer output before final model normalization.
  The memory-source bridge is `m_t=(1-alpha)*x_t+alpha*z_t` before native input
  normalization. Alpha0 must recover ordinary outputs and gradients; fractional
  alpha backward must return both source branches. Alpha is immutable per call.
- FBT feedback and NextLat use post-finalnorm model states. Shared residual
  width is 2048; variable internal head/FFN dimensions are preserved.
- Preserve one embedding/readout parameter and count it once in the optimizer.
  NextLat auxiliary KL detaches only its readout use, not the embedding globally.
  No autonomous learned-MLP latent recurrence is planned.
- RoPE cache convention must be explicit. Native keys are normalized and
  unrotated in cache; avoid accidental double rotation or cached causal shifts.
- Ordinary OpenELM/FBT attention can use SDPA fused kernels. Exact RT remains
  a distinct schedule and backward. The FA4 dependency is not evidence of use.
- Current RT autograd accumulates hidden block-parameter gradients internally;
  DDP/FSDP compatibility must be established separately. Start with the planned
  explicit gradient synchronization baseline before more elaborate sharding.

## Experiment order after the platform work

1. Native import/fidelity (current PR).
2. Native-block RT reference, then tiled backward and cached decoding, including
   shared-pass gradient ownership and independently checked NextLat losses.
3. Bounded physical-batch/throughput checks; two-GPU checks when available.
4. Paired ordinary / ordinary+NextLat / RT / RT+NextLat with FBT off. RT alone
   succeeding is not a prerequisite for testing RT+NextLat.
5. Independent FBT-only control, then motivated FBT interactions. Its reference
   development can begin after native import without waiting for tiled RT.
6. Late feature diagnostic and selected follow-ups, not an automatic full matrix.

Keep all learning comparisons matched on original checkpoint, data exposure,
optimizer reset/warmup and evaluation protocol. The plan's budget ladder is a
proposal with review points, not authorization for billion-token runs now.

## Execution and retention

Host project: `/home/taylorbollman/cdrm-w-latent`; container project:
`/workspace/cdrm-w-latent`. Use
`bash scripts/docker_shell.sh bash -lc '<command>'`. Never run GPU workloads
directly on the host and never silently substitute CPU. Intentional CPU unit
tests can use `CDRM_DOCKER_GPUS=none` with the same launcher.

Persistent artifact root for import: `.runtime/openelm-import/artifacts/`.
Use new timestamped validation output directories under `.runtime/openelm-import/`.
Checkpoints/manifests/results retained past the session belong in persistent
project paths and `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/` with checksums.
Local SSD contents are disposable. Do not print `.env`, tokens or credentials.

Selected retention prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T182701Z/`.
Local retention directory: `.runtime/openelm-import/retention-verified/`; log:
`.runtime/openelm-import/retention-verified.log`. Remote sizes, server MD5 and
SHA256 metadata were verified; `upload-result.json` also records the receipt
object's generation. The report's `storage-receipt.json` preserves these records.
The archive includes the tested source overlay and restore instructions; the
4.3GB checkpoint is a separate object. Earlier failed diagnostics remain local
and in W&B but are not the acceptance record.

The first retention attempt stopped before upload because an inherited
`GOOGLE_APPLICATION_CREDENTIALS` pointed to a nonexistent old path. Mounted
standard ADC credentials were verified, and the successful command used
`env -u GOOGLE_APPLICATION_CREDENTIALS`. No credential contents were exposed
and no environment file was changed. There is no upload or GPU job still running.

Graphable diagnostics use W&B entity `taylorbollman`, project
`pretrained-fbt-rt-nextlat`. Record the real run URL in results. Keep the
independent native-reference code/config hashes alongside adapter source hashes
so later modifications cannot silently replace the baseline.
