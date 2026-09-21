# Pretrained OLMo / RT / FBT / NextLat implementation handoff

Updated 2026-09-21. **Read this first after compaction or interruption.**

## Current decision, authorization and next action

The user selected **original OLMo-1B at approximately 200–300B pretraining tokens**
as the new primary model, replacing OpenELM because of uncertainty about its
layer-wise capacity scaling. We selected **step 60,000, approximately 251–252B**.
This is a model choice, not a finding that OpenELM's scaling is defective.

The user approved the [v3 plan](fbt-rt-nextlat-research-plan-v3.md) and authorized
**O1 development**. Both gates are now complete: native ordinary fidelity and
selected-layer sequential RT. See [results](reports/olmo1b-o1/results.md),
[usage](olmo1b-native-rt-usage.md) and [protocol](reports/olmo1b-o1/protocol.md).
The scoped CPU suite passes **148 tests**. The actual 1.177B checkpoint passes
ordinary source equivalence in FP32/BF16 and sequential RT output/all-parameter/
input-gradient checks in FP32 at alpha 0/0.37/1, physical B1/T16, layer 0 only.

**Stop at the O1 review point.** O2 (native RoPE tiled forward/backward), FBT,
NextLat and learning remain staged; do not start them without the next user
direction. No GPU, training or evaluation job is running or awaiting resumption.
There was no optimizer update to the pretrained checkpoint. The user authorizes
direct PR closure/merges; this does not expand research/training scope.

Implementation branch: `feat/olmo1b-native-rt-reference`, based on `e894fe0`
(planning PR #3). The O1 source hashes are in the selected validation reports;
source/evidence, not an unrecorded working tree, defines tested behavior.

## O1 selected results and recovery

- Artifact root: `.runtime/olmo1b-step60000/artifacts/`;
  `artifact-manifest.json` and `checkpoint-inspection.json` record full integrity.
- Ordinary: `.runtime/olmo1b-step60000/ordinary-validation-01/report.json`, W&B
  [td5cce3w](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/td5cce3w).
  B1/T42 text backward/all 65 parameters and input gradients; T64 code forward.
  Adapter/native differences are exactly zero in checked outputs/gradients for
  FP32 math, BF16 math and BF16 default SDPA. Cache/causality checks pass;
  ordinary BF16 profiler observes cuDNN fused/Flash-style SDPA.
- RT: `.runtime/olmo1b-step60000/rt-validation-01/report.json`, W&B
  [ujs94viz](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ujs94viz).
  Actual B1/T16 bottom-layer alpha 0 versus ordinary, plus alpha 0/.37/1 versus
  independent history oracle. Largest FP32 individual-tensor relative L2 error
  is 4.880e-6; all declared joint tensor-norm/maximum and input/output checks pass.
  Two stricter elementwise coordinates flag and are retained as diagnostics;
  no acceptance tolerance was changed after results.
- BF16 RT: finite/complete-gradient smoke scope only. Alpha 1 global gradient
  difference versus same-path FP32 is 1.695% math/1.451% default; worst tensor
  is 3.751%/3.461%, `transformer.blocks.5.ff_proj.weight`. This is not tiled or
  training clearance. Ordinary mixed precision is descriptively similar on a
  different fixture, so it is not a controlled recurrence-sensitivity ablation.
- Combined CPU record: `.runtime/olmo1b-step60000/cpu-suite.log`, also retained
  as [test-results.txt](reports/olmo1b-o1/test-results.txt): 148 passed.
- Current implementations: `cdrm/pretrained/olmo.py`, `olmo_recurrent.py`,
  `olmo_artifacts.py`, isolated `olmo_reference.py` and
  `olmo_recurrent_oracle.py`; pristine source in `_olmo_reference/`.
- Retention is verified at
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/20260921T202457Z/`;
  [storage receipt](reports/olmo1b-o1/storage-receipt.json) records generations,
  sizes, server MD5 and SHA256. Local receipt/evidence:
  `.runtime/olmo1b-step60000/retention-20260921T202457Z/`.
  The native source checkpoint is retained, not a newly trained model.

## Read in order

1. [Current v3 research plan](fbt-rt-nextlat-research-plan-v3.md).
2. [Checkpoint selection audit](olmo-1b-250b-checkpoint-selection.json).
3. [RT numerical handoff](rt-numerical-handoff.md): reusable methods, not an
   instruction to reopen the historical numerical study.
4. [Archived OpenELM implementation handoff](fbt-openelm-implementation-handoff.md)
   when reusing artifact, cache, immutable-mode or gradient-validation machinery.
5. [Nanochat budget audit](fbt-nanochat-budget-audit.md) and
   [architecture/auxiliary review](fbt-architecture-and-auxiliary-loss-review.md)
   for retained evidence; their historical OpenELM choices are superseded.

Both [v2](fbt-rt-nextlat-research-plan-v2.md) and the
[initial OLMo proposal](fbt-rt-nextlat-pretrained-plan.md) are historical.
Returning to original OLMo does not reactivate the initial proposal's final
approximately 3T checkpoint or obsolete milestone details.

## Frozen checkpoint and source choices

- Primary: `allenai/OLMo-1B`, branch **`step60000-tokens252B`**, revision
  **`81b71efbce6f4dada57c94860301af4298bcd351`**.
- Native file: `model.safetensors`, 4,707,065,440 bytes; advertised SHA256
  `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
  **Complete bytes and published SHA256 verified in O1.**
- Secondary official conversion: `allenai/OLMo-1B-hf`,
  `step60000-tokens251B`, revision
  `6e6042e824831c7b42223f75cf60fe3a5d92eb79`.
  Same-step naming does not prove actual tensor parity; verify before use as a
  secondary oracle. Native artifact is authoritative.
- Validated original source: `allenai/OLMo` v0.2.4 at
  `b3741bc21f1dd504838b7dbd9878ee077ded63bd`. Pristine source execution and native
  checkpoint fidelity validated in O1. Checkpoint remote-code stubs import `hf_olmo`;
  checkpoint SHA alone does not pin model math.
- Use native tokenizer files at the primary revision. HF conversion tokenizer
  differs, including postprocessor. Do not silently substitute or copy OpenELM
  BOS/EOS defaults. Freeze tokenizer versus dataset EOS insertion explicitly.
- Approximate age is 251–252B; exact original token counter was not verified.
  Header confirms 1,176,764,416 unique parameters. At rough 20 tokens/parameter,
  23.54B is the reference budget, so this is approximately 10.7x. This heuristic
  is neither a saturation threshold nor an exact FBT reproduction match.
- This is original OLMo, **not** July OLMo-1B-0724 or OLMo 2. No automatic
  alternate model is queued.

## Native architecture and porting pitfalls

- 16 uniform layers, residual width 2048, 16 query/16 KV heads, head dimension 128.
- SwiGLU intermediate 8192; fused 16384 output splits **value/up, gate** and
  computes `silu(gate) * value`, not OpenELM's gate-first convention.
- Non-affine LayerNorm, epsilon 1e-5, including final normalization; **no Q/K
  normalization**, no biases, no ALiBi or QKV clipping.
- RoPE base 10,000, native FP32 split-half computation, context 2048; cache keys
  are unrotated. Adding Q/K norm would be a separate adaptation experiment.
- One tied input/readout parameter, all **50,304 rows** retained; tokenizer
  vocabulary 50,280, EOS 50279, configured pad 1.
- Native embedding sets **no `padding_idx`**. Do not introduce pad-row lookup
  gradient suppression. Masks/labels are explicit; preserve full logits width.
- 65 native stored tensors, FP32. HF has 113 split tensors with the same element
  count. Map native/HF weights and gradients rather than comparing raw names.
- Freeze an independent original-source reference. Our modified local `olmo`
  namespace is not an independent upstream oracle. Isolate imports.

## Research and gradient contracts

- Keep independent RT, FBT and NextLat switches. Standalone RT is one pass;
  FBT K counts complete passes and K1 is ordinary pass 0.
- Initial selected RT set is `{0}`. Top index is 15; derive from configuration.
  Use full history, not the restricted-window synthetic default.
- RT writes from `m_t=(1-alpha)*x_t+alpha*z_t`, then native input LayerNorm and
  K/V projections, with RoPE in attention coordinates. There is no Q/K norm.
  Temporary self KV comes from input; persistent write follows completed
  block output. Top-block z is before model finalnorm.
- Alpha 0 must exercise the scan and recover ordinary outputs/gradients.
  Fractional alpha returns both source branches. Modes/alpha/masks/positions
  are immutable per call, including backward and concurrent shared forwards.
- Retain cache-provenance checks for model/mode/weight versions, conversion,
  autocast/grad context and backend. Preserve attached cache training paths.
- FBT uses shifted previous-pass **post-finalnorm** states through the shared
  stack; exact online decoding uses the freshly completed previous-token state.
  Cross-pass gradients remain attached. New feedback branch scaling must not
  modify the native backbone at the ordinary endpoint.
- NextLat is auxiliary training only; no autonomous MLP rollout. Source state
  and conditioning embedding stay attached, target state/distribution stop.
  Detach only auxiliary readout use, not the shared embedding globally.
  Latent pair and KL triple masks respect documents/padding; pass loss
  reductions are explicit. Do not silently reuse synthetic-only loss code.
- Early learning comparison remains ordinary / ordinary+NextLat / RT /
  RT+NextLat with FBT off. RT-only success is not a prerequisite for RT+NextLat.
  In a separately authorized milestone, develop FBT-only independently after
  ordinary fidelity, then examine combined modes with matched controls and
  exposure; this does not expand O1's scope.
- Feature matching remains a late diagnostic. Semantic Tube, EBFT, alternative
  backbones and new auxiliary objectives are not queued initial experiments.

## Preserved implementation/evidence; do not overwrite

OpenELM import and sequential RT are complete, now historical references:

- [PR #1](https://github.com/taylorbollman/cdrm-w-latent/pull/1), merge `959c225`:
  native import, actual 1.1B ordinary source fidelity in FP32/BF16; implementation
  `b24abd8` and documentation `ad86d71`. [Results](reports/openelm-import/results.md).
- [PR #2](https://github.com/taylorbollman/cdrm-w-latent/pull/2), merge `2169520`:
  native sequential RT, implementation/evidence `6603521`, docs `7752534`/`be08149`.
  [Results](reports/openelm-rt-reference/results.md).
- Stage B: 136 scoped CPU tests; actual B1/T16, layer 0, alpha 0/.37/1, outputs and
  all 226 parameter/input gradients pass FP32. BF16 finite smoke only: global
  gradient difference 3.859% math / 4.842% default; worst tensor 25.83% / 29.87%.
  These are neither OLMo results nor BF16 training clearance.
- Local sources: `cdrm/pretrained/`; ordinary/RT usage guides and reports remain
  unchanged. Preserve pinned CoreNet reference snapshots and all source hashes.
- W&B: [Stage A z4jzbq02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/z4jzbq02),
  [Stage B uowl622p](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/uowl622p).
- Import artifacts: `.runtime/openelm-import/artifacts/`; selected evidence
  `.runtime/openelm-import/validation-final/report.json` and
  `.runtime/openelm-rt-reference/validation-final/report.json`.
- GCS import prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T182701Z/`.
- GCS RT prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-rt-reference/20260921T190151Z/`.
  Reports retain verified storage receipts. Reuse existing checkpoint objects;
  do not upload another OpenELM copy.

The completed planning revision changed documentation only; O1 subsequently
implemented and validated OLMo. OpenELM's proposed tiled milestone is superseded
by the OLMo lineage and is not still queued in parallel.

## Execution, resources and retention

Host project: `/home/taylorbollman/cdrm-w-latent`; container project:
`/workspace/cdrm-w-latent`. Launch project commands with
`bash scripts/docker_shell.sh bash -lc '<command>'`. Verify container location
and successful `nvidia-smi` **inside it** before GPU work. Never execute CUDA,
training/evaluation/profiling on the host or silently use CPU. Explicit CPU unit
tests use `CDRM_DOCKER_GPUS=none` with the same launcher.

The OLMo runtime lineage is `.runtime/olmo1b-step60000/`; cloud retention is
recorded by a verified receipt under the O1 results. Project files persist;
local SSD is disposable. Retain checkpoints,
source/config/tokenizer hashes, evidence and receipts. Graphable runs go online
to W&B `taylorbollman/pretrained-fbt-rt-nextlat`; record actual URLs.

The previous GCS attempt found stale `GOOGLE_APPLICATION_CREDENTIALS`. Valid
mounted standard ADC worked with `env -u GOOGLE_APPLICATION_CREDENTIALS`.
Do not print credentials or modify `.env` just to work around that stale path.

Re-profile OLMo: bottom-layer KV width is 8x OpenELM bottom KV width, but that is
not an 8x whole-model memory/time estimate. Different depth/MLP/MHA geometry
prevents transplanting earlier batch-size claims. FP32 is the semantic reference;
actual-runtime BF16 and tiled behavior need bounded fresh checks.

Multi-GPU work remains staged. Historical tiled autograd internally accumulates
parameter gradients; DDP/FSDP is not proven by existing launcher code. Start
with explicit post-backward gradient reduction when hardware is available,
then consider optimizer-state sharding. Do not let any old training launcher
reset pretrained weights after wrapping.
