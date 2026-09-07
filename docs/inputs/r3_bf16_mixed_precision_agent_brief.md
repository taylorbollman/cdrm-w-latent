# R3: implement and validate BF16 mixed-precision training

Implement a usable BF16 **autocast/mixed-precision** profile in the existing OLMo-based project. Keep FP32 trainable parameters and optimizer state. Preserve the R3 architecture, and retain the cleared FP32 implementation as the reference. This task includes diagnosis, focused fixes if necessary, bounded training validation, and performance measurement.

The user explicitly wants standard mixed-precision casting, not an all-BF16 model. Do not call `model.bfloat16()` or convert checkpoint parameters, optimizer moments, or all intermediate buffers to BF16.

## Starting evidence and scope

Read the existing `docs/reports/r3-backward/results.md` and its relevant diagnostics first. The completed investigation cleared FP32/no-accumulation R3 for the tested MQAR regime. It found no backward defect and required no model, kernel, optimizer, or Stage B source patch. Coordinate-level flags remained, but scaling experiments, actual masked CE, all-parameter gradients, persistent-write probes, AdamW updates, and naïve FP64 references supported ordinary FP32 rounding and scale-sensitive thresholds. Do not reopen that entire investigation or rerun Stage B without new evidence of an affected computation.

Use the saved Stage B resolved configuration and retained checkpoints as authoritative, rather than reconstructing settings from this brief. The tested regime was MQAR, vocabulary 1024, 12 blocks, D256, H4/full MHA, MLP1024/GELU, pre-norm ALiBi, affine Q/K normalization, recurrent block index 3, `rho=1`, four backward MLP chunks, T128, physical/global B64, and no accumulation. The FP32 runtime used autocast/TF32/dropout off, deterministic math SDPA, compiled tiled helpers, and no whole-model compilation or CUDA graphs.

Retained R3 checkpoints are under:

```text
.runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/
  SYN-mqar-R3-seed0-init.pt
  SYN-mqar-R3-seed0-final.pt
```

The existing tools include `r3_backward_validate.py`, `r3_write_path_probe.py`, `r3_ce_fp64_reference.py`, `r3_validation_metrics.py`, and `r3_first_adam_analysis.py`. Reuse their identity checks, parameter mapping, compiler audits, metrics and retention mechanisms. Preserve original research artifacts; new optimizer steps must operate on diagnostic copies with separate lineage.

The immediate target is BF16 mixed precision at the existing T128 regime. CDRM, new architecture changes, full research sweeps, distributed execution, and accumulation repair are not prerequisites. No change to normalization, ALiBi, residual scaling, rho, optimizer hyperparameters, or task semantics is implied. In particular, do not switch to the authors' `norm_after=true` debug case to obtain a pass.

## 1. Establish the actual mixed-precision contract

Inspect the existing BF16 failures before changing code. Identify whether those runs used FP32 parameters with autocast or an actual BF16 model, and whether their acceptance criteria distinguished expected reduced-precision error from implementation error.

Begin with ordinary CUDA BF16 autocast and a documented conservative precision policy:

| Component | Initial policy |
|---|---|
| Trainable parameters, parameter-gradient accumulation, optimizer moments | FP32 |
| Eligible projection and MLP matrix multiplications | BF16 autocast |
| Residual stream | Preserve FP32 initially |
| Normalization statistics and masked-CE reductions | FP32 |
| Running attention/softmax statistics and sensitive recurrent-gradient accumulators | Inspect explicitly; retain/promote to FP32 where required |
| Projected Q/K/V and their storage | Follow the declared autocast policy; validate BF16 storage and every precision boundary |

Do not prescribe gratuitous casts where the existing operator already handles precision correctly. Record observed tensor and operation dtypes; a configuration label is not evidence that the intended mixed-precision computation occurred. Autocast does not automatically make every custom running buffer safe. Conversely, an FP32 destination buffer cannot recover precision already lost in a BF16 intermediate.

Start without a GradScaler, as is customary for BF16 mixed precision; loss scaling is not the default repair for rounding/accumulation error. Preserve the existing optimizer implementation initially. Check actual parameter, gradient and moment dtypes before and after an update.

Inspect nested autocast and `config.precision` handling. An outer autocast context may not control helpers that establish their own context. Coordinate the model and custom-function behavior explicitly. Do not globally wrap the ordinary training-loop `backward()` in autocast. Recomputed forward operations inside the custom backward must nevertheless reproduce the intended forward precision policy. Use `torch.amp.custom_fwd`/`custom_bwd` or equivalent explicit handling when appropriate for the existing implementation; decorators alone do not validate recomputation or buffer precision.

Use the installed environment's APIs. Retain the current container/PyTorch version initially; do not introduce a framework upgrade as an unrelated variable.

## 2. Reproduce, localize and triangulate the BF16 discrepancy

Reproduce the most relevant saved BF16 failure with its original fixture and acceptance rules. Preserve those results. First determine whether the leading discrepancy is in the forward output, recomputed intermediates, temporal backward accumulation, or optimizer update.

Use this comparison matrix with identical weights, examples, labels, answer masks and mean-CE reduction:

| Comparison | Purpose |
|---|---|
| Naïve BF16 mixed precision versus cleared naïve FP32 | Characterize error introduced by the precision policy |
| Compiled tiled BF16 mixed precision versus naïve BF16 mixed precision | Identify additional discrepancies associated with the tiled implementation |
| Compiled tiled BF16 mixed precision versus cleared naïve FP32 | Assess overall approximation and update effects |

Match the declared precision policy between naïve and tiled paths as closely as possible and document differences. The naïve path may use SDPA that internally upcasts intermediates, whereas the custom tiled path manages its own arithmetic. Do not mistake different precision policies for identical implementations in different schedules. Neither BF16 backend is automatically a numerical oracle. Use a tiny FP64 reference only when needed to distinguish explanations.

Start with the smallest informative fixture and B2/T128, then cover actual B64/T128 at both retained initialization and the trained checkpoint. Reuse existing FP32 packets when identities and settings match. Cover actual loss, all intended parameter gradients, block input/embedding gradients, persistent-write credit, and one actual optimizer update from identical optimizer state. Use `loss.backward()` and inspect all registered `.grad` tensors, since the custom backward accumulates parameter gradients internally.

Record TF32, reduced-precision GEMM-reduction options, SDPA backend and compiler behavior. Preserve the reference's strict settings. If a setting is changed for the candidate, identify it as part of the precision profile and compare accordingly. Keep attention-backend optimization separate from the first precision diagnosis. Observe compiled execution and reject silent fallback when claiming a compiled tiled result.

## 3. Make focused precision fixes only if the evidence calls for them

In the upstream design, relevant inspection points include `block_attention_add`, `recompute_alphas`, `recompute_atts`, and the custom backward's gradient buffers. Confirm their current implementation in this fork; these are investigation targets, not diagnosed bugs.

Inspect running softmax maxima/denominators and weighted-value sums; attention weights and cancellation-sensitive products used in the backward; per-token gradient accumulation; and buffers whose dtype follows `x.dtype` or `q.dtype`. A softmax evaluated in FP32 and immediately cast back to BF16 may still make subsequent backward arithmetic sensitive.

Also compare selected recomputed activations against the forward values that they are intended to reconstruct. Tiling/recomputation changes evaluation order. At BF16 this can change the activation at which a downstream derivative is evaluated, even if the real-arithmetic equations agree. If necessary, retain a small amount of additional forward state or promote a localized operation, and measure its memory/time cost. Do not redesign the kernel before locating the discrepancy.

The preferred outcome keeps substantial projection/MLP work in BF16 while retaining sensitive arithmetic in FP32. Preserve architecture and gradient connectivity. Add targeted regression tests for any actual fix; keep the existing FP32 checks passing without recreating their entire validation matrix.

An optional interim profile is BF16 ordinary blocks with block 3 executed in FP32. Label and benchmark it separately. It can help the 12-block model, but it does not establish BF16 support for the recurrent block or for a one-block RT reproduction. Do not silently report this fallback as the primary target achieved.

## 4. Use justified acceptance criteria

Do not demand FP32 coordinate tolerances or bitwise equality from BF16. Also do not declare a blanket percentage error acceptable without investigation. Keep the historical criteria as historical diagnostics and define precision-appropriate engineering screens before the confirmatory pass. If exploratory evidence motivates revised screens, record the revision and confirm on held-out diagnostic fixtures; do not overwrite earlier outcomes.

Retain per-parameter relative L2, maximum-error/reference-RMS, absolute-error and near-zero diagnostics with documented denominator floors. Supplement with gradient direction where useful, but do not let a good global norm or cosine conceal a badly affected parameter group. Inspect missing/zero/nonfinite gradients separately.

Compare actual AdamW deltas, moments and clipping behavior. Interpret initial near-epsilon coordinate sensitivity using the prior FP32 analysis, rather than requiring every small coordinate to pass a rigid percentage-of-LR screen. Assess magnitude, sparsity, direction and downstream effect together. Scaling checks apply to the fixed-forward backward map before clipping/Adam; approximate rather than bitwise scaling may be appropriate in BF16. A wrong linear backward can pass a scaling check, so it is supporting evidence only.

A passing loss comparison alone is insufficient. Clearance requires explained numerical behavior across forward, gradients, memory credit and updates, followed by bounded operational validation. If acceptable behavior cannot be established, report unresolved scope rather than widening tolerances until everything passes.

## 5. Validate short training and recovery

After numerical checks, run one bounded paired 100–300-update R3 comparison: selected BF16 mixed precision versus the cleared compiled tiled FP32 profile, from corresponding weights, on the same ordered training batches. The naïve implementation remains the numerical reference rather than the required backend for this short trajectory. Use the saved initial training schedule where applicable. If starting from the final checkpoint instead, declare a new diagnostic continuation schedule and preserve both originals. Keep this an operational precision comparison, not a new architecture efficacy study.

Monitor training and fixed development CE, gradient/update norms, clipping frequency, finite state and persistent-path gradients. Inspect for systematic deterioration or instability relative to FP32. Exact trajectories need not agree; one short pair cannot establish equal final quality or task generalization. Do not adjust data or hyperparameters independently to make the BF16 arm look better.

Demonstrate checkpoint save/load and bounded midpoint resume for the selected profile, including precision metadata. Preserve the documented R3 update-zero restriction unless independently fixed; do not make its repair a hidden prerequisite. Reuse existing supported recovery machinery.

## 6. Measure the usable result and state its boundary

Benchmark actual masked CE, backward and optimizer updates for SEQ and R3 in FP32 and the selected BF16 mixed profile. Verify SEQ's selected profile with a small actual-loss/finite-update check as needed before using it as the timing baseline. Start at the same B64/T128 shape and batch size. Use warm-up and synchronized timings, and separate compilation/startup from steady-state updates.

Report update latency, tokens/s, peak allocated/reserved memory, optimizer footprint, effective dtypes, compiler/backend settings and comparison identities. Measure improvements within each architecture as well as R3/SEQ ratios. BF16 may accelerate the standard stack more than the recurrent bottleneck. Do not infer speedup from Tensor Core capability or promise that total memory halves while parameters and moments remain FP32.

Use one H100 80 GB and bounded serial jobs. Reuse the existing harness; no multi-seed research sweep is needed for this milestone. Once sufficiently validated, stop optional numerical expansion and produce the result.

The first deliverable is an honest T128 clearance or an actionable unresolved diagnosis. Before research with the paper's one-block D128/H16/MLP512 configuration or training at T512, validate those specific shapes and profile separately. Gradient accumulation is the next distinct operational milestone if needed; compare target-count-weighted accumulated gradients/updates with the equivalent physical batch and explicitly address the existing failure before use. Do not bundle distributed training, CUDA graphs, CDRM or broad attention-kernel optimization into this task.

## Deliverables

Provide the minimal implementation/configuration changes, runnable diagnostic/training/benchmark commands, one recommended resolved mixed-precision profile, and a concise evidence report. Include exact versions, identities, retained fixtures/checkpoints, an observed dtype map, numerical metrics, remaining flags, short training/recovery results, and timing/memory comparisons.

State separately whether (a) BF16 autocast is active in the intended dense operations, (b) the recurrent block is numerically cleared for the tested scope, (c) bounded training/recovery works, and (d) the measured performance benefit is useful. An interim FP32 recurrent block, an uncompiled fallback, or a numerically acceptable but slower profile must be labeled accurately.

Keep the original Stage B and FP32 validation artifacts immutable. Use existing retention/upload conventions for new artifacts and record actual verification. Scope clearance to the tested task, shapes and execution settings; it is not a claim of trajectory-wide equivalence or BF16 clearance for every Stage B task.

## Primary implementation references

- [PyTorch AMP and autocast](https://docs.pytorch.org/docs/2.14/amp.html): operation-specific casting and custom-autograd integration. Consult the installed version when applying APIs.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html): reduced-precision GEMM/SDPA behavior and accumulation concerns.
- [Pinned upstream recurrent implementation](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/model.py): design reference; the current fork and saved project configurations are authoritative for this task.
