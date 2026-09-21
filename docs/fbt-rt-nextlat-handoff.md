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

The user reviewed O1 and authorized **O2: native RoPE tiled forward/backward**.
O2 is complete on `feat/olmo1b-tiled-rt`, based on O1 merge `a806835`.
Implementation/evidence commit: `c6a41c23568a4ef4e561b7b44a05f47fe38ab59b`,
[PR #5](https://github.com/taylorbollman/cdrm-w-latent/pull/5).
See [O2 results](reports/olmo1b-o2/results.md),
[usage](olmo1b-tiled-rt-usage.md) and [protocol/amendment](reports/olmo1b-o2/protocol.md).
The combined scoped CPU suite passes **230 tests**. Actual-checkpoint tiled
FP32 checks pass at B1/T16 alpha 0/.37/1 and B1/T128 alpha 1, all 65 parameter
and input gradients. A raw-cotangent B2/T17 strict input-coordinate failure is
retained and adjudicated with an independent FP64 oracle; see details below.
BF16 differences and eager H100 performance are measured, not training clearance.
The user reviewed O2 and authorized **O3 language-model objectives/platform**.
O3 is complete on `feat/olmo1b-nextlat-platform`, based on O2 merge
`ef3380f1beb41642150c693c33ca2558365e4f84`. Implementation/evidence commit:
`40290357aff82f0caa40f9deff79c90412758faa`,
[PR #6](https://github.com/taylorbollman/cdrm-w-latent/pull/6). See
[O3 results](reports/olmo1b-o3/results.md), [usage](olmo1b-nextlat-platform-usage.md)
and [protocol](reports/olmo1b-o3/protocol.md). The scoped CPU suite passes
**326 tests**; actual-checkpoint FP32 objective/gradient checks and exact full
optimizer recovery pass. Bounded BF16 complete-step profiles are recorded below.
The user reviewed O3 and authorized continuing to O4. One H100 is available;
two-GPU correctness requires later hardware. Native checkpoint files and research
starting weights are unchanged. Four physical optimizer updates were executed
on disposable diagnostic state (three logical updates, including one replay);
profiling executed 42 zero-LR updates. No adapted research checkpoint or learning
run was awaiting resumption at the O3 close. The user authorizes direct PR closure/merges; this
does not expand research/training scope.

**O4 is now active** on `feat/olmo1b-o4-learning-pilot`, based on O3 merge
`29fee0dae0f551ef75a29de1d1269e3a69ec8055`. Implementation commit
`e294ce6735dd7f1a4d8be08e464d4b8336892db6`,
[draft PR #7](https://github.com/taylorbollman/cdrm-w-latent/pull/7); keep draft
until the learning comparison has completed and been reviewed. Read the
[O4 protocol](reports/olmo1b-o4/protocol.md). Scope: four matched code-continuation
arms, ordinary / ordinary+NextLat / RT / RT+NextLat; original checkpoint, RT
layer0 only, FBT off. Approximately20–22M valid input tokens/arm:100-update
alpha-zero LR warmup, >=10M-token/200-update alpha ramp, then at least equal
actual alpha-one exposure. Source-pinned CodeSearchNet Python and WikiText
development retention; test splits reserved. No100M+ or FBT run is automatically
queued. The optional domain question offered code or math; preparation proceeds
with the stated Python-code default in the absence of contrary steering.

New modules: `lm_data.py`, `lm_evaluation.py`, `lm_schedule.py`; drivers
`scripts/olmo_lm_prepare_data.py`, `olmo_o4_preflight.py`, `olmo_o4_train.py`.
Prepared data: `.runtime/olmo1b-step60000/o4-data-01/prepared/manifest.json`.
25,000,031 unique train CE targets,101,591 windows; no cycling. Native EOS only
at document ends; stride511/T512 preserves CE targets and omits one KL triple
per internal window cut. Rows/windows have independent attention context.
Full preparation/loader source and raw/token/prepared byte hashes are recorded.

Runtime destinations (update with actual status before resuming):
`.runtime/olmo1b-step60000/o4-preflight-01/` for baseline/capacity/configuration;
`.runtime/olmo1b-step60000/o4-pilot-01/` for arm reports/checkpoints. All arm
training must use the same frozen preflight configuration/source hashes and
prepared data; do not silently change code mid-comparison. Each saved optimizer
checkpoint is uploaded and verified before older local copies are removed.
Use `env -u GOOGLE_APPLICATION_CREDENTIALS` inside the container for valid GCS
ADC. Never treat W&B success or a local log alone as checkpoint retention.

O4 preflight passed, W&B
[mnpa95sr](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mnpa95sr).
B32 full-length real-data RT+NextLat profile:2.249s/update,45.99GiB allocated.
The exact frozen schedule is2,634 updates/20,855,799 input tokens per arm;
warmup ends100, ramp ends1367, final2634. Ordinary initial512-window code
NLL1.787902/accuracy62.579%; WikiText development NLL3.040993/accuracy42.141%.
RT-alpha0 agrees closely; immediatealpha1 codeNLL3.545859/accuracy37.147%,
retentionNLL5.032289/accuracy23.576%, motivating the gradual conversion.

The queue is running in container via unified-exec session74807; inspect
`.runtime/olmo1b-step60000/o4-pilot-01/queue.json` and arm reports for **current**
status. Order:ordinary,ordinary-nextlat,rt,rt-nextlat. Initial host `nohup`
attempt did not start a job (empty log/no arm); do not trust its stale PID file.
Current queue log:`.runtime/olmo1b-step60000/o4-pilot-01-queue.log`.
Cloud prefix:`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z/`.
Configuration SHA256:`6b7e9168f2aa1afc495d28cfdd84006e5fa3409f66fcc70c17c0fa83c8ff2e20`.
It records a pre-training recovery-only source amendment; see protocol.
No core training/model/data source changes are permitted between matched arms
without explicitly stopping/revising this lineage. Analysis/report-only new
files can be developed independently. New implementation tests:397 combined
O1–O4 core tests plus22 runner helper,12 retention and35 report tests:466 distinct
passing tests, recorded in [test-results](reports/olmo1b-o4/test-results.txt).

Initial source/data/preflight evidence has been verified in GCS under the above
prefix's `initial/` directory, including68,381,709-byte evidence archive. Receipt:
[initial-storage-receipt.json](reports/olmo1b-o4/initial-storage-receipt.json).
The ordinary control's update100 full checkpoint is also verified remotely
(`ordinary/update-000100.pt`, SHA256
`0cc6f230144dd9c1eb0d8accd076fa08179f7f9fc4886d8023912702807a23b4`).
Its live W&B run is
[4rhi7s7i](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4rhi7s7i).
As of22:17UTC it had passed7.5M tokens and was healthy; use actual reports for
newer status. Other arms are queued, not claimed complete.

Automatic CPU finisher is running in unified-exec session3634, log
`.runtime/olmo1b-step60000/o4-finish-01.log`; state is
`o4-pilot-01/finish-status.json`. It waits for all four completed arms, then runs
the strict analysis reporter to create `docs/reports/olmo1b-o4/results.md`,
`final-comparison.json`, `learning-curves.pdf/png`, and verifies a final GCS
evidence upload (`o4-final-retention-01/`, cloud `final/`). It stops without
claiming completion if the queue stops/fails. If the VM shuts down, resume the
GPU queue and relaunch this CPU finisher; no external scheduler was created.
The finisher does not commit reports or merge PRs: after completion, inspect
results/curves/receipts, update this handoff and PR description, commit evidence
and close the draft PR. See [O4 usage](olmo1b-o4-usage.md) for stop/resume/report.

O1 implementation branch: `feat/olmo1b-native-rt-reference`, based on `e894fe0`
(planning PR #3). The O1 source hashes are in the selected validation reports;
source/evidence, not an unrecorded working tree, defines tested behavior.

## O3 selected results and recovery

- Code: `cdrm/pretrained/nextlat.py` and `lm_training.py`; GPU drivers
  `scripts/olmo_lm_validate.py` / `olmo_lm_profile.py`; shared literal fixtures
  and dense objective in `scripts/olmo_lm_common.py`. O1/O2 model math unchanged.
- Source fidelity: NextLat revision `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`,
  retained byte snapshot in `_nextlat_reference/`. Select its **1B LM horizon-one**
  recipe, not A5: predictor factor 1.6 (hidden6528), latent/KL coefficients 1/1,
  no auxiliary predicted-token CE. Predictor adds 82,726,912 parameters; total
  1,259,491,328 versus native 1,176,764,416. Predictor is training-only, with
  isolated initialization; inference remains `model.backbone(...)`.
- Masks: CE targets token t+1; latent targets stopped post-finalnorm state t+1;
  KL compares stopped teacher and predicted-state distributions for token t+2.
  Source/conditioning embeddings remain attached; only auxiliary readout use
  is detached. Separate target-position masks and global valid counts per
  objective across microbatches. One document per row; reject multi-document
  packing because loss masks cannot prevent attention leakage.
- Validation: `.runtime/olmo1b-step60000/lm-validation-01/report.json`, W&B
  [r0zqx75g](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/r0zqx75g).
  Full-checkpoint FP32 math B1/T16 ordinary/RT with NextLat off/on and padded
  B2/T16 RT+NextLat pass independent dense-objective/all-active-parameter checks.
  Global gradient relative L2 1.579e-6–4.811e-6; worst tensor 6.226e-6.
  BF16 RT+NextLat versus FP32 global differences: mixed attention 1.932%, FP32
  attention 1.911%; worst tensors 2.839%/2.590%. Finite descriptive diagnostics,
  not long-sequence fused-backend gradient or learning-equivalence clearance.
- Exact actual-model recovery: save after update2, rebuild, reload, repeat update3.
  Model, AdamW moments, scheduler, counters, CPU/CUDA next RNG draws and update
  metrics match bit-for-bit. LR1e-5 with two-update warmup; FP32, layer0 RT.
  Disposable 15,114,028,075-byte checkpoint SHA256
  `8d615b49730f8c8e1292fac008596977e1fdc754e0f78366293418a05b6fba22`
  was deleted after passing; full hashes/fixtures/evidence retained. No need to
  recover this diagnostic model. CPU tests also cover nonzero predictor dropout
  and Python/NumPy/explicit data-generator RNG.
- Profile: `.runtime/olmo1b-step60000/lm-profile-01/report.json`, W&B
  [zbrcek8g](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zbrcek8g).
  BF16 default SDPA + mixed tiled attention; layer0 RT only; FP32 parameters,
  real initialized AdamW moments, clipping, LR0. Three warmups/three timed steps.
  RT+NextLat B1/B4/B8,T512: 1.496/1.587/1.626 seconds, 342/1,290/2,519 valid
  input tokens/s, 19.61/20.37/23.99 GiB allocated peaks. B8 reserved25.79 GiB.
  All weights unchanged and moments finite. Not a maximum-batch search or a
  learning run; literal repeated text with half-length CE masks, no data loader.
- Memory: vocabulary loss chunks recompute full-vocabulary logits rather than
  retaining all such activations. Validation chunk8; profile chunk128 positions.
  O2 backward still reconstructs quadratic attention and uses per-position VJPs.
  No compile/graphs, FBT, distributed, packed-document attention or cached LM
  objective training clearance. Only layer0 is recurrent in actual-model checks.
- CPU record: `.runtime/olmo1b-step60000/lm-cpu-suite.log`, copied to
  [test-results.txt](reports/olmo1b-o3/test-results.txt): 326 passed, 41 warnings.
  Compact machine-readable record: [summary](reports/olmo1b-o3/validation-summary.json).
- Evidence prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-nextlat-platform/20260921T213500Z/`.
  [Storage receipt](reports/olmo1b-o3/storage-receipt.json) records verified
  objects. Local archive/receipts:
  `.runtime/olmo1b-step60000/lm-retention-20260921T213500Z/`.
  Reuse O1's existing immutable checkpoint object; do not upload another copy.
- Next review: choose O4 domain/splits and masking, practical batch/exposure,
  ordinary/ordinary+NextLat/RT/RT+NextLat controls and matched alpha ramp. Inspect
  untrained predictor scale before selecting adaptation schedule; initial short
  RT fixture means CE5.095/latent0.916/KL8.349 are not model-quality measurements.

## O2 selected results and recovery

- Code: `cdrm/pretrained/olmo_tiled.py`, `OLMoTiledRTForCausalLM` and low-level
  `tiled_recurrent_layer`. Parameters/layout unchanged; O1 references preserved.
  Dyadic attention, native RoPE, fractional alpha, explicit returned parameter
  gradients, attached prefix/exported-cache gradients. Separate tiled cache
  provenance. Saved x/z enable recomputation without sequential forward replay.
- Final validation: `.runtime/olmo1b-step60000/tiled-validation-02/report.json`,
  W&B [nfys79l3](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nfys79l3).
  FP32 full-model global gradient relative L2 1.543e-6–2.319e-6; worst tensor
  3.434e-6. All original full-model acceptance budgets pass. Some stricter
  parameter-coordinate diagnostics remain flagged, as recorded in the report.
- Initial stress-test failure remains in `tiled-validation-01/`, W&B
  [q4gu72t4](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/q4gu72t4),
  and [retained report](reports/olmo1b-o2/initial-validation-01.json). Final
  raw-input adjudication retains 65/69,632 coordinate flags but finds FP32
  input-gradient norm error versus FP64 only 6.143e-7 tiled / 5.439e-7 scan.
  A documented **post-failure calibration** applies the existing joint tensor
  norm/maximum budget to raw input gradients, requiring both FP32 paths to
  agree with FP64. Do not call this an unchanged original coordinate screen.
  Original source hashes are recoverable with the retained reverse source patch.
- BF16 alpha1 versus tiled FP32: T16 global 1.489–1.660%, worst tensor
  3.499–3.723%; T128 mixed attention global 2.714%, worst 3.197%; T128 FP32
  attention global 1.675%, worst 2.219%. All finite and complete. Both attention
  policies remain available; this is bounded observation, not learning clearance.
- Profile: `.runtime/olmo1b-step60000/tiled-profile-01/report.json`, W&B
  [em18z6bm](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/em18z6bm).
  Full model (RT layer0 only), BF16 B1/T512: 1.464 s forward/backward,
  350 tokens/s, 9.509 GiB operational peak; no optimizer. B4/T512 block-only
  1,388 tokens/s. Eager tiled B1/T128 is slower than scan (1.6x FP32/1.4x BF16).
  Entire backbone remains resident even for block-only measurements. Memory
  includes finiteness-check scratch; timings exclude it.
- Limits: quadratic attention reconstruction in backward, per-position local
  autograd, first-order only; no compiler/graphs, multi-GPU, context2048,
  all-16-layer actual-checkpoint recurrence or training-quality clearance.
- CPU record: `.runtime/olmo1b-step60000/tiled-cpu-suite.log` and
  [230-test record](reports/olmo1b-o2/test-results.txt).
- O2 evidence retention reuses the existing O1 checkpoint object without
  another model upload. [O2 storage receipt](reports/olmo1b-o2/storage-receipt.json)
  records the verified timestamp prefix, object generations and hashes.
  Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-tiled-rt/20260921T205831Z/`;
  local receipts/archive in `.runtime/olmo1b-step60000/tiled-retention-20260921T205831Z/`.
- Its subsequent O3 single-GPU milestone is now complete (above); two-GPU
  correctness remains untested. No adaptation run is implicitly authorized by
  platform work.

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
