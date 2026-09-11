# Handoff: numerical testing of future Recurrent Transformer changes

Written 2026-09-11, after completing the standard-RT precision milestone.
This is a persistent project handoff for a future session, including after
context compaction. It records reusable methods, evidence and implementation
constraints; it does not authorize or start another experiment.

## Resume here

The active project is `/home/taylorbollman/cdrm-w-latent`, mounted inside the
container at `/workspace/cdrm-w-latent`. The implementation is in
`recurrent-transformer/`, directly under the project, rather than the old
`vendors/` location. The current foundation is **standard RT with every block
recurrent**, not CDRM. The user's next architectural modification has not yet
been specified or implemented in this milestone.

The completed decision is to keep protected `bf16_fp32_state` as the default
and retain released-style `legacy` mixed precision as experimental. All
intended GPU jobs finished. There is no unfinished training run to resume.
The historical execution lineage and its compiler cache are closed and
retained. Use a new lineage for future work; preserve the historical records.

Read these first, in order:

1. [Final results and interpretation](reports/rt-precision-alignment/final-results.md).
2. [Implementation/PR summary](reports/rt-precision-alignment/pr-summary.md).
3. [Driver usage](rt-precision-alignment-usage.md) and
   [retained correctness fixes versus precision choices](reports/rt-precision-alignment/change-audit.md).
4. [Frozen protocol](../.runtime/rt-precision-alignment/20260910T191100Z/reference/protocol.json)
   and [endpoint scope](../.runtime/rt-precision-alignment/20260910T191100Z/reference/endpoint-check-scope.json).

The project had substantial existing uncommitted/untracked work from earlier
milestones. No commit, push or external PR was created for the precision
milestone. Inspect the current diff before editing; do not discard unrelated
work. Exact source snapshots and hashes in run reports identify tested code
more reliably than a vendor commit alone.

## Exact baseline and precision contract

| Item | Tested value |
| --- | --- |
| Architecture | Standard RT, all 12 blocks tiled recurrent, rho 1 |
| Width / heads / FFN | 1024 / 16 / 4096 |
| Parameters | 151,045,120 backbone; 216,843,264 total; 111 parameter tensors |
| Vocabulary | Untied 32,128-row embedding/head; 32,100 valid token IDs |
| Attention / normalization | Causal ALiBi; learned normalization including Q/K normalization; no dropout, dense-layer biases or RoPE |
| Operational shape | Physical B512, T512; one backbone pass, no backbone gradient accumulation |
| Loss | Shifted CE over 511 targets per row, summed and divided by B×512 |
| Memory strategy | Head-only microbatch 2; tiled internal recomputation with four backward MLP chunks; no outer checkpoint wrapper |
| Capture | Whole forward/loss/backward CUDA graph; gradient clipping and Adam outside capture |

This is the user-selected **width-1024 150M backbone**, not the width-1408
300M backbone in the paper's D.1 configuration. Do not silently switch sizes
because an earlier prompt cited D.1.

| Arm | Meaning |
| --- | --- |
| A | BF16 autocast plus `bf16_fp32_state`: our extra attention working/reconstruction/adjoint protection |
| B | BF16 autocast plus `legacy`: released-style recurrent arithmetic, retaining correctness repairs |
| C | Tiled FP32 reference: autocast off and TF32 off |

Parameters, residual storage, parameter gradients and Adam moments are FP32
in all arms. B is **not all BF16**: upstream already retains FP32 running
weighted-value sums, denominators, observed max-logit state and K/V gradient
buffers. A protects additional attention operands and backward intermediates.
Tensor boundary dtypes do not prove the accumulation precision of an internal
GEMM or fused operation. The old ordinary-attention FP32 option from CDRM is
inactive when every block is recurrent.

AdamW used peak LR 0.001, betas (0.9, 0.95), epsilon 1e-8, weight decay 0,
gradient clipping at norm 1, `foreach=False`, `fused=False`. The released
`CosWithWarmup` schedule retained 5,000 warmup updates, a 12,500-update horizon,
and alpha0/alpha_f 0.1. LR at updates 1/100/500 was
0.00010018 / 0.000118 / 0.00019.

The runtime was NVIDIA container 26.06, PyTorch
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3 and driver 580.173.02 on one H100 80GB
HBM3. TF32 was disabled; BF16 reduced-precision reduction, autocast's weight
cache and deterministic algorithms were enabled. Record the actual future
runtime; this snapshot does not guarantee equality on another stack.

## What the completed evidence establishes

Candidate capture matched uncaptured loss and gradients exactly at the tested
tiny/full-B2 cases and at initialization at physical B512. Changed-input Adam
updates, causality, stable gradient ownership, observer neutrality, and bounded
fresh-process resume checks passed. All four policy/seed training runs completed
500 updates with finite FP32 parameters, gradients and moments.

Initial roundoff flags were real and remain in the record. At B512, three
block-9 query/key-related tensors in B exceeded the 3.125% tensor-L2 screen
slightly, at about 3.31–3.34%; all maximum-error screens passed. Do not replace
this result with the older seven-tensor B64 failure count from a different
R3/CDRM configuration.

At both first-seed policies' trained 100- and 500-update checkpoints, A and B
passed the gradient and actual Adam-update screens against the same-state
physical-B512 FP32 reference. Across these trained B/C checks, global gradient
relative L2 was approximately 0.51–0.80%, and actual Adam-delta relative L2 was
0.33–0.51%. This is representative **first-seed** numerical coverage; there
was no second-seed trained FP32-gradient comparison.

Nevertheless, the learning comparison did not clear the prospective allowance:

| Seed, update 500 | B−A development CE / supervised token | One-sided 95% upper bound | Decision under 0.005 margin |
| --- | ---: | ---: | --- |
| 20260910 | +0.00892543 | 0.01033497 | Not cleared; interval lies above the margin |
| 20260911 | +0.00416939 | 0.00610802 | Not cleared; point is below the margin, uncertainty crosses it |

Native evaluation closely agreed. Candidate throughput was about 12.5% higher,
with about 9.8 GiB less GPU reservation. These are sequential matched-device
measurements, with verification overhead, rather than randomized benchmarks.

The conclusion is a small development learning cost in this bounded pilot,
alongside reassuring trained-state numerics. It is not catastrophic gradient
failure, a proof about final convergence, or evidence that one identified
attention operation explains the entire learning difference. See the
[frozen disposition](../.runtime/rt-precision-alignment/20260910T191100Z/reference/development-disposition-500.json).

Only the first 500 of 5,000 warmup updates were trained. Peak-LR behavior,
full-warmup behavior and final convergence remain untested. The confirmation
set was **not evaluated**, because B was not nominated after development.
Preserve that status if considering a future selected comparison.

## Reuse the method, not an inapplicable gradient baseline

For a modified architecture, compare its BF16 implementation with **the same
modified architecture in FP32**, using identical parameters, tokens, loss,
normalization and optimizer state. Old RT gradient tensors are not an oracle
for a model whose function, parameterization or objective intentionally changed.
Architecture-quality comparisons against the old RT are a separate experiment.

If a modification has an exact baseline limit, test that limit explicitly:
for example, a disabled new branch should recover the intended base behavior
and shared-parameter gradients. An intended zero-gradient or detached branch
must be recorded as such, rather than mistaken for missing recurrent credit.
For new auxiliary objectives, record their coefficients and individual loss
and gradient contributions before interpreting the combined gradient.

A proportionate sequence for a future modification is:

1. **Specify the mathematical and execution changes.** List new state, writes,
   reads, masks, normalization, loss terms, parameter sharing and optimizer
   mappings. Freeze a new configuration and choose the smallest diagnostic
   fixture that can exercise the change. Keep A as the initial operational
   control unless there is a concrete reason to change precision too.
2. **Establish a small semantic reference.** Use ordinary differentiable
   operations/FP32 or a local FP64 oracle where practical. Test causality,
   boundary tokens, intended gradient paths, and recomputation against the
   modified mathematical function. A naïve BF16 scan is not automatically a
   better reference than tiled BF16.
3. **Compare identical-state precision arms.** Start small; inspect raw and
   clipped gradients, losses, layer/position samples and actual optimizer
   deltas. Include a representative trained state with nonempty moments when
   available. Preserve initial small-gradient/sign sensitivity rather than
   judging only a global cosine.
4. **Localize consequential differences.** Freeze actual rounded operands and
   incoming cotangents, then compare the local derivative against independent
   FP32/FP64 arithmetic. Change one suspected precision component at a time.
   Check observer neutrality. Escalate only when the finding warrants it.
5. **Validate the real captured path.** Remove diagnostic hooks, capture a
   fresh graph for each configuration/policy, compare uncaptured/captured
   gradients, and perform repeated changed-input/changed-weight optimizer
   steps. Check that gradients are cleared once and that replay sees updated
   weights. Revalidate fresh-process resume if using checkpoints.
6. **Check the intended physical shape.** After the small cases work, include
   a bounded forward/backward/optimizer check at the actual batch, sequence
   length and precision. A microbatch diagnosis or strict-FP32-only run cannot
   clear the BF16 deployment shape. Reprofile memory after architecture changes.
7. **Evaluate learning only when needed for the question.** Freeze data,
   matching initialization rules, token order, schedule, acceptance criteria
   and development/confirmation roles prospectively. Preserve meaningful
   warmup; a compressed schedule changes the experiment. A good numerical
   result alone does not prove equal learning.

Do not automatically rerun the entire historical matrix or launch another
500-update/two-seed pilot. Reuse validated infrastructure, choose checks that
exercise the actual modification, and stop at a reviewable result with clear
remaining scope. These methods are intended to help us use and extend RT,
not to demand arbitrary extra decimal places beyond a meaningful concern.

## Error metrics and diagnostic traps

For candidate gradient g and same-state reference r, report
`||g-r||₂ / ||r||₂` globally and per tensor, and
`max(|g-r|) / max(|r|)` per tensor. The historical review triggers were:

- Global gradient L2: 0.015625 (1.5625%).
- Per-tensor gradient L2: 0.03125 (3.125%).
- Per-tensor maximum error/reference maximum: 0.0625 (6.25%).
- Trained actual Adam-delta L2: 0.015625; initial Adam cosine: at least 0.99.

Use FP64 metric reductions over all coordinates, with no absolute acceptance
floor. Flag undefined zero-reference ratios explicitly. These are engineering
review triggers, not author-published tolerances or automatic learning-failure
criteria. A future protocol may choose justified criteria before its results;
do not relax them after seeing a failure.

Compare the **actual applied FP32 parameter delta**, `theta_after-theta_before`,
after the real clipping and Adam step, starting from cloned identical moments.
Initial Adam updates can be sensitive to signs of tiny gradients despite a
high cosine. The saved trained probes use the checkpoint's saved LR
(0.000118 or 0.00019), not the next scheduled update's LR; keep that convention
explicit if reproducing them.

Useful earlier findings that prevent repeated dead ends:

- Tiny fixed-forward cotangent scaling by 1/32 and 32 produced bitwise equal
  rescaled parameter gradients and write adjoints. Earlier permanent K/V
  writes received credit; the unused terminal permanent write did not.
- At zero-based block 9, B2/T512, frozen actual rounded Q/K/V and incoming
  cotangent gave a 4.3098% legacy query-adjoint error against local FP64;
  independent FP32 on those operands gave 7.82e-7 relative error. This
  separates local attention arithmetic from upstream forward drift.
- A final BF16 cast alone explained only about 0.1668% in that local query
  check. The first-token cancellation hypothesis was not observed. Neither
  finding isolates a single primitive to patch. See
  [attention findings and equations](reports/rt-precision-alignment/attention-findings.md).
- Earlier naïve-versus-tiled BF16 differences included repeated rounding at a
  shared weight-cast node; tiled was closer to a same-operand FP64 contraction.
  Do not make tiled imitate naïve BF16 or disable autocast's weight cache
  globally without evidence. See the [historical change audit](reports/rt-precision-alignment/change-audit.md).
- CUDA graphs preserve the captured arithmetic; they do not eliminate
  roundoff. Compiled uncaptured diagnostic packets need separate captured-path
  validation. Dtype observers must be removed before operational capture.

## Source map and adaptation points

| File / entry point | Reusable role or assumption to revisit |
| --- | --- |
| [model.py](../recurrent-transformer/olmo/model.py), `OLMoRecurrentBlockTiledFunction`, `OLMoRecurrentBlockTiled` | Forward scan, custom backward and internal recomputation; provisional self versus permanent earlier K/V semantics |
| [model.py](../recurrent-transformer/olmo/model.py), `recompute_alphas`, `recompute_atts`, `mlp_batched_body` | Attention reconstruction and dense backward; preserve the actual forward autocast policy during recomputation |
| [config.py](../recurrent-transformer/olmo/config.py) and [full.json](../configs/stage_a/full.json) | Explicit model/precision configuration; historical full fixture is not a generic future config |
| [rt_cuda_graph_validate.py](../scripts/rt_cuda_graph_validate.py), `validation_config` | Central tiny/full constructor used by comparison/training; tiny is D64/H4/L2/T16/vocab128, full forces standard all-recurrent semantics |
| [rt_batch_profile.py](../scripts/rt_batch_profile.py), `head_chunked_backward`, `main` | Native shifted-CE head chunking and physical profile; independently constructs its full model |
| [rt_cuda_graph.py](../scripts/rt_cuda_graph.py), `CapturedRTBackward` | Stable gradient buffers, capture/replay guards and nested parameter-gradient writes |
| [rt_precision_compare.py](../scripts/rt_precision_compare.py), `run_arm`, `validate_starting_state`, `compare_arm_packets` | Same-state A/B/C packets, checkpoint checks, actual Adam deltas and all-coordinate reductions |
| [rt_precision_credit.py](../scripts/rt_precision_credit.py), [rt_precision_attention_probe.py](../scripts/rt_precision_attention_probe.py) | Architecture-specific credit hooks and local attention oracle; adapt operands, diagonal semantics, cotangents and masks if the recurrence changes |
| [rt_precision_train.py](../scripts/rt_precision_train.py), [rt_precision_resume_probe.py](../scripts/rt_precision_resume_probe.py) | Frozen bounded protocol, training/checkpoint contracts and resume proof |
| [rt_precision_eval.py](../scripts/rt_precision_eval.py), [rt_precision_eval_contract.py](../scripts/rt_precision_eval_contract.py) | Evaluation bound to a completed training endpoint, finite canonical FP32 saved state and source/data identity |
| [rt_precision_pair_eval.py](../scripts/rt_precision_pair_eval.py), [rt_precision_outcome_report.py](../scripts/rt_precision_outcome_report.py) | Document pairing/bootstrap and reporting from explicit compatible closed reports |

The existing drivers are **bounded experiment harnesses**, not a generic
architecture CLI. Training/profiling enforce the tested counts and B512/T512;
the tiny/full factories encode model assumptions. Introduce a consistent new
configuration/fixture path and adapt every affected caller, contract, label
and test rather than changing only `full.json`. Preserve the old source
snapshots and original experiment protocol.
The evaluator has independent restrictions too: `rt_precision_eval.main`
expects 12 tiled blocks and T512/511 targets, while
`validate_evaluation_contract` restricts endpoints to 100/500 and requires
`next_data_row == updates * 512`. Changing the factory, trainer and profiler
alone does not update those evaluation assumptions.

Especially important when adding latent state or auxiliary objectives:

- **The head-loss driver owns the training objective.**
  `head_chunked_backward` requests `return_pre_logits=True`, detaches those
  hidden states, runs `transformer.ff_out` in head chunks, then propagates the
  accumulated hidden cotangent through the backbone once. Adding a loss only
  to ordinary `model.forward()` may leave it out of this path. Explicitly
  adapt loss construction, weighting, normalization and gradient accounting
  for new heads, masked targets, tied embeddings or latent losses.
- **Parameter tensors and persistent buffers are different state.** Current
  comparison/evaluation contracts expect saved model keys to equal named
  parameters. New persistent `state_dict()` buffers require a deliberate
  serialization/validation extension. Gradient/Adam checks also expect every
  included trainable parameter to participate; handle intentionally dormant,
  detached or frozen parameters through an explicit coverage contract.
- **The graph class is shape-generic but structurally restricted.** It does
  not hardcode 12 layers or 111 tensors, but currently requires tiled RT
  blocks, all parameters trainable FP32, an untied head, ALiBi/no RoPE,
  ungrouped blocks, no CDRM/outer checkpoint/dropout/hooks. Extend the validated
  contract for a new design rather than merely deleting a rejection.
- **Partitioned FP32 assumes independent examples.** Its helper preserves
  the global B×T denominator and performs clipping/Adam once. Batch-coupled
  objectives may invalidate that substitution. Repeated ordinary head-loss
  calls normalize each chunk separately, and captured replay clears gradients;
  neither is a drop-in gradient-accumulation reference.
- **Adapt observers and oracles to actual semantics.** Layer-output observers
  expect tuple outputs with `[B,T,D]` first. The credit test excludes the
  unused vocabulary head and assumes the final permanent write is never
  consumed; an auxiliary memory loss can legitimately change that. The local
  attention oracle assumes equal `[L,B,H,D]` Q/K/V, already-scaled queries,
  provisional self K/V and permanent earlier K/V. New GQA, layouts, scaling,
  masks or write/read rules require corresponding oracle changes.
- **Extend provenance for new code.** Source snapshots enumerate explicit
  harness files and `recurrent-transformer/olmo/**/*.py`. New implementation
  files outside that tree must be added to coverage so run hashes include
  the code that actually executed.

Changed model structure may require deliberate checkpoint conversion, shared
weight mapping, initialization of new parameters, and optimizer-slot migration
or reset. Document that as a new lineage. Do not bypass source/runtime/name/
shape guards to present a changed model as an exact historical resume. The
resume-proof packet is also distinct from a normal training checkpoint.
Its current transition is fixed at 100→101→102. The existing
`checkpoint_conversion.py::convert_model` is a checked weights-only
sequential/recurrent conversion, not an optimizer-state or generic
architecture migration tool.

Retain existing ownership, masking, unsupported-input, autocast-restoration
and graph repairs unless the actual new design replaces their assumptions.
The old `make_graphed_callables` wrappers are not a drop-in replacement:
nested backward accumulates parameter gradients as side effects, which their
explicit `autograd.grad` interface did not safely represent here.

## Artifacts, checkpoint selection and recovery

All paths below are relative to the historical lineage:
`.runtime/rt-precision-alignment/20260910T191100Z`.

| Path | Meaning |
| --- | --- |
| `reference/protocol.json` | Original prospective numerical/learning criteria and schedule |
| `reference/development-disposition-500.json` | Completed keep-A/default, B/experimental, no-confirmation decision |
| `reference/gpu-closure-gate.json` | All intended GPU work closed; historical compiler cache frozen |
| `train/A-seed0-100`, `train/B-seed0-100` | Original first-seed steps 0/100 and complete updates 1–100 |
| `train/A-seed0-500-restart1`, `train/B-seed0-500` | Accepted first-seed endpoints; reports contain updates 101–500 |
| `train/A-seed1-500`, `train/B-seed1-500` | Direct second-seed runs with steps 0/100/500 and complete histories |
| `numerics/trained-{A,B}-seed0-{100,500}-b512` | Same-state C/A/B packets, starting checkpoint, diagnostic IDs and detailed comparisons |
| `numerics/full-b2-seed0`, `numerics/full-b2-seed1`, `numerics/full-b512-seed0`, `numerics/tiny-seed0` | Initial numerical fixtures |
| `attention/block9-seed0`, `credit/tiny` | Local derivative and recurrent-credit evidence |
| `data/train/manifest.json`, `data/heldout/manifest.json` | Pinned token matrices, splits, documents and hashes |
| `data/heldout/diagnostic.npy` | Fixed physical-B512 numerical input fixture |
| `eval/{A,B}-seed{0,1}-500-dev-{fp32,native}` | Completed endpoint token losses/document aggregates |
| `analysis/pair-seed{0,1}-500-dev-{fp32,native}` | Exact paired evaluation and bootstrap reports |
| `analysis/outcome-two-seed-500` | Final figures, explicit input/source snapshots and coverage report |
| `verification/` | Independent packet, training-lineage, resume, source, scientific and figure audits |
| `analysis/storage/` | Per-bundle verified GCS receipts and member manifests |

Choose `checkpoints/step-000500.pt` inside the accepted training directory for
a trained endpoint. `report.json` records its full SHA256, size, model digest
and optimizer/data position. For a fresh matched experiment, steps 0 are in
the original seed0-100 directories or direct seed1-500 directories. Within
each seed, A/B initial tensors are bitwise equal; seed0 and seed1 differ.

**Do not use `train/A-seed0-500` as the accepted continuation.** It was
interrupted around update 169, with no checkpoint after 100; its JSONL tail
was damaged. The directory is preserved as historical evidence. The accepted
`A-seed0-500-restart1` resumed at 100 and exactly reproduced all 69 saved
overlapping scalar updates. Combine its report with A-seed0-100 for a full
curve; do not concatenate the interrupted history and double-count updates.
Fresh-process proofs separately checked actual parameter/gradient/Adam/RNG
bytes at the bounded proof step, not nonexistent saved tensors at every
intermediate update.

The physical H100 UUID changed across interruption, while model, capacity,
capability and software/driver identity matched. Each accepted paired phase
used the same physical device. Preserve the distinction between an exact
scalar overlap, bounded tensor resume proof and cross-hardware portability.

Data is a pinned, freshly tokenized bounded C4/T5 slice, not the authors'
unavailable processed corpus. The training matrix contains exactly 256,000
rows of length 512: enough for these 500 physical-B512 updates without repeats.
It is **not a ready-made full-warmup dataset**. Development and confirmation
each have 1,024 rows; document boundaries and cross-document context are
retained. Final CE uses 523,264 supervised targets per role. Document-bootstrap
intervals are conditional on a trained seed, not estimates of training-seed
uncertainty; do not pool documents across seeds to claim seed robustness.

## Environment, storage and verification

Follow the current project [AGENTS.md](../AGENTS.md). Never run CUDA, training,
GPU evaluation or profiling in the host shell, and never silently fall back
to CPU. On a fresh instance, `/home/taylorbollman/start.sh` bootstraps the GPU
container; thereafter use this project's launcher. A GPU command has this
outer form (substitute the actual adapted driver command):

```bash
bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv &&
  test "$PWD" = /workspace/cdrm-w-latent &&
  nvidia-smi -L &&
  <adapted GPU driver command with a fresh output directory and cache>
'
```

Use `CDRM_DOCKER_GPUS=none` for CPU container analyses. The host Python did not
have Torch/NumPy in the tested session; use the project container rather than
assuming host dependencies. Do not enumerate unrelated archived projects.

Use a new explicitly named `TORCHINDUCTOR_CACHE_DIR` for a new lineage. The
old cache helped preserve exact replay but is now frozen and archived; do not
reuse it as a writable scratch directory. A copied cache, if useful, still
needs source/runtime validation and new captured-versus-uncaptured checks.

Graphable runs must log online under W&B entity `taylorbollman`; project
`rt-precision-alignment` contains this milestone. New projects/runs are
authorized when useful. Credentials come from `.env` through the launcher;
never print them. [Final W&B figures](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/lbo30ro4).

The retained prefix is
`gs://fast-chunks/cdrm-w-latent/rt-precision-alignment/20260910T191100Z/`.
It contains 28 verified evidence archives (107,192,682,023 compressed bytes),
plus 27 data objects totaling 349,229,421 bytes. Do not download everything by
default: a full numerical case has about 15.6 GB of raw packets. Start with
reports/manifests, then fetch the particular checkpoint or local oracle packet
needed. Useful anchors are:

- [Final storage receipt](../.runtime/rt-precision-alignment/20260910T191100Z/analysis/storage/final-closure/storage.json).
- [Verified inventory](../.runtime/rt-precision-alignment/20260910T191100Z/analysis/final-closure-inputs/verified-storage-inventory.json)
  for the 27 preceding bundles; the final receipt is the terminal 28th anchor.
- [Compiler-cache receipt](../.runtime/rt-precision-alignment/20260910T191100Z/analysis/storage/compiler-cache-final/storage.json):
  1,040 actual cache files, 127,900,862 uncompressed bytes, preserved unchanged.

Each bundle has `evidence.tar.gz`, a member manifest and a verified receipt.
Validate the recorded object generation, archive SHA256/bytes and member
hashes when restoring; extract into a separate location rather than replacing
current work. Home/project files persist, while SSD-only storage does not.
Keep future reusable checkpoints/data in `gs://fast-chunks` too, under a new
prefix. The tiny metadata/storage helpers under the old lineage are tailored
to that experiment and are not universal acceptance protocols.

Historical verification included **152 passing integrated CPU tests** with
CUDA initialization forbidden, **16 separate outcome-contract checks**, and
independent GPU/packet/lineage/scientific audits. The 12 storage-helper metadata
checks are separate again; do not inflate the model-test count. See the
[final source check](../.runtime/rt-precision-alignment/20260910T191100Z/verification/final-source-check.json)
and [final scientific review](../.runtime/rt-precision-alignment/20260910T191100Z/verification/final-scientific-review.json).
For future code changes, run the relevant tests and add meaningful tests for
new semantics; historical passing tests do not validate a modified model.

This handoff was created after the historical archive closed and is not
claimed as a member of that archive. Keep this project note and its AGENTS.md
pointer discoverable; add dated follow-up notes when future work changes the
configuration, findings or remaining scope.
