# Pretrained OLMo / RT / FBT / NextLat implementation handoff

Updated 2026-09-21. **Read this first after compaction or interruption.**

## Current decision, authorization and next action

The user selected **original OLMo-1B at approximately 200–300B pretraining tokens**
as the new primary model, replacing OpenELM because of uncertainty about its
layer-wise capacity scaling. We selected **step 60,000, approximately 251–252B**.
This is a model choice, not a finding that OpenELM's scaling is defective.

The current task is **plan revision and durable documentation only**. Share the
[v3 plan](fbt-rt-nextlat-research-plan-v3.md), then wait for the user's direction
before implementing the next milestone. No full OLMo checkpoint has been
downloaded, no OLMo implementation or GPU validation has been performed, and
there is no pretrained-model GPU or learning job to resume.

**Next review milestone: O1**, one bounded migration PR with two internal gates:

1. Native original-OLMo import, tokenizer/source pinning and ordinary function
   fidelity against an independent pristine reference.
2. Native sequential RT reference and independent history oracle, alpha
   0 / 0.37 / 1, initially layer 0, bounded actual-checkpoint gradient/cache checks.

Ordinary fidelity comes first. If that uncovers a substantial compatibility
problem, pause at an ordinary-only reviewable result. O1 does not implement
FBT, NextLat, the tiler or learning. Subsequent staged work is described in v3;
no long training budget is authorized by a platform milestone. The user has
previously authorized direct PR closure/merges; that does not expand scientific
or training scope.

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
  **Whole-file bytes/hash not yet downloaded/verified.**
- Secondary official conversion: `allenai/OLMo-1B-hf`,
  `step60000-tokens251B`, revision
  `6e6042e824831c7b42223f75cf60fe3a5d92eb79`.
  Same-step naming does not prove actual tensor parity; verify before use as a
  secondary oracle. Native artifact is authoritative.
- Prospective original source: `allenai/OLMo` v0.2.4 at
  `b3741bc21f1dd504838b7dbd9878ee077ded63bd`. Source inspected but execution and
  checkpoint fidelity untested. Checkpoint remote-code stubs import `hf_olmo`;
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
- Initial selected RT set is `{0}`. Top index is15; derive from configuration.
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

The current planning revision changes documentation only. OpenELM's proposed
next tiled milestone is superseded by O1, not still queued in parallel.

## Execution, resources and retention

Host project: `/home/taylorbollman/cdrm-w-latent`; container project:
`/workspace/cdrm-w-latent`. Launch project commands with
`bash scripts/docker_shell.sh bash -lc '<command>'`. Verify container location
and successful `nvidia-smi` **inside it** before GPU work. Never execute CUDA,
training/evaluation/profiling on the host or silently use CPU. Explicit CPU unit
tests use `CDRM_DOCKER_GPUS=none` with the same launcher.

Next OLMo runtime lineage is proposed as `.runtime/olmo1b-step60000/`, with a
new prefix under `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/`. No OLMo upload
exists yet. Project files persist; local SSD is disposable. Retain checkpoints,
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
