# Standard RT precision alignment: experiment plan

Status: the bounded 500-update comparison across two initialization seeds
completed on 2026-09-11. Keep protected `bf16_fp32_state` as the operational
default; `legacy` remains experimental. Trained-state numerical screens passed,
but neither seed cleared the prospective development-loss margin. Confirmation
was not run. See the [completed results](reports/rt-precision-alignment/final-results.md)
and [PR summary](reports/rt-precision-alignment/pr-summary.md). Peak learning
rate and full convergence remain untested: the original 5,000-update warmup
was preserved. The sections below retain the authorized experiment plan;
the original pre-execution copy is saved in the execution lineage.

The question is whether the released-style mixed-precision arithmetic gives
adequate numerical and learning behavior in our standard, all-recurrent model,
and whether our additional FP32 attention protections provide enough benefit
to justify their cost. CUDA graph correctness and cross-precision accuracy are
separate questions; the preceding milestone established the former for the
current policy.

## What we are comparing

The model remains 12 tiled recurrent layers, D1024, FFN4096, 16 heads, T512,
151,045,120 backbone parameters and 216,843,264 total parameters. Keep causal
ALiBi, learned normalization including Q/K normalization, rho 1, no dropout,
untied 32128-row tables / 32100 valid IDs, internal checkpointing with four
MLP backward chunks, head-only microbatch 2 and native shifted CE divided by
B×512. Primary operational batch size is physical B512. Keep clipping and
AdamW settings fixed across precision arms.

| Arm | Computation | Purpose |
| --- | --- | --- |
| A: current | BF16 autocast plus `bf16_fp32_state` | Established operational control |
| B: released-style | BF16 autocast plus `legacy` recurrent precision | Candidate for reducing our extra FP32 work |
| C: reference | Tiled FP32 with autocast and TF32 disabled | Evaluate both A and B against a higher-precision computation |

`legacy` is not an entirely BF16 implementation. The upstream source already
keeps some state and reductions in higher precision. The candidate should
preserve that mixture. Upstream revision checked during planning:
[`a21b42d2bc292edb86ed1b62cee4bcab809a9d21`](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/model.py).
Its [base configuration](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/configs/kempner/base-c4-t5.yaml)
selects BF16 AMP.

The expected differences from our source inspection are below. Verify actual
dtypes at runtime before treating this as the executed precision contract.

| Quantity | A | B |
| --- | --- | --- |
| Parameters, residual stream, loss, optimizer moments | FP32 | FP32 |
| Q/K/V projections, MLP and output head | BF16 | BF16 |
| Persistent forward K/V storage | BF16 | BF16 |
| Forward weighted-value running numerator and denominator | FP32 | FP32 |
| Q/K attention working operands | FP32 | BF16 |
| Max-logit state at observed helper boundaries | FP32 | FP32 |
| Backward softmax | FP32 result retained | FP32 computation followed by BF16 result cast |
| Recomputed attention and several attention-adjoint intermediates | FP32 | BF16 |
| K/V gradient accumulation buffers | FP32 | FP32 |

Autocast may promote particular operations, including exponentials. Input,
output, accumulation and storage precision must be distinguished. Our current
policy also affects backward intermediates; changing it is more than replacing
one attention matrix multiplication.

Execution update (2026-09-10): the tiny observed A/B runs confirmed FP32
max-logit state in both policies. The original planning table grouped this
with Q/K products too broadly; the rows above correct that description.
All loss and raw gradients were bitwise unchanged by observation. Internal
fused-operation and GEMM accumulator precision is not inferred from these
boundary dtypes.

The ordinary-attention precision option used in the previous CDRM work is
irrelevant here: all twelve blocks are recurrent. Do not change residuals,
normalization, optimizer precision or existing running sums to make the
candidate artificially uniform in dtype. Preserve the current ownership,
masking, autocast-restoration and CUDA-capture fixes.

## What the paper establishes

The paper acknowledges floating-point reordering in its tiling equivalence
claim, uses a stable online-softmax construction, and emphasizes normalization
for training stability. I found no reported BF16-versus-FP32 gradient tolerance
or precision-ablation acceptance budget. Its stability analysis is not a
finite-precision error guarantee. [Sections 4–5 and Appendix C](https://arxiv.org/html/2604.21215v1#S5).

For scale, its 150M/B512 table reports validation CE 3.046 for RT versus 3.067
for the standard Transformer, a 0.021-nat gap. This is not a precision
tolerance or a target for a short pilot. [Appendix E.3](https://arxiv.org/html/2604.21215v1#A5.SS3).

## 1. Freeze the experiment and verify the precision contract

Pin source, runtime, compiler cache, RNG, initialization, loss and optimizer
configuration. Audit the released arithmetic against the local legacy branch;
do not revert unrelated correctness fixes for historical fidelity. Add an
explicit precision-policy argument to the bounded profiler/validator and
construct every layer with that policy.

The graph helper currently accepts only `bf16_fp32_state`. Extend that explicit
contract for the candidate and capture a fresh graph per policy. Changing a
configuration after capture does not change its kernels and should continue
to fail. Keep stable gradient buffers, exactly-once accumulation, scoped
autocast and clipping/Adam outside capture.

Use an instrumented tiny fixture to verify forward and backward dtypes. Then
repeat without observers before using numerical or performance results.
Record graph/compiler behavior and any kernel selection differences.

Prepare deterministic random IDs for execution tests and a bounded frozen
C4 training/development/confirmation corpus using the existing T5 tokenizer
contract. The authors' configured data paths are specific to their machine;
data availability and preprocessing need verification, not assumption. Pin
tokenizer revision, document boundaries, token order, padding and supervision.
Development data guides diagnosis; fresh held-out data confirms the selected
policy. Do not use random-token loss to infer language-model learning quality.

Exit: a verified dtype map and a frozen, reproducible A/B/C comparison.

## 2. Validate capture within the candidate precision

Repeat the existing capture checks for B: a tiny model and the actual 12-layer
model at B2/T512. Compare captured and uncaptured executions at identical
precision, including two unchanged-input replays, changed inputs, at least
three real Adam steps, every gradient, parameter and optimizer-state tensor,
state-dict preservation and causality. Keep the declared replay tolerance
separate from the much looser cross-precision screens; require every residual
to be explained rather than silently relaxing it.

Carry the same checks to actual B512, including all initial raw gradients and
a bounded update sequence. Preserve small-batch diagnostics and actual-batch
validation separately. Confirm no new compilation during measured replay.

Exit: the candidate graph performs the intended mixed-precision computation,
uses changed weights/inputs and does not drop or accumulate stale gradients.

## 3. Measure precision effects at identical states

Compare A–C, B–C and B–A at the exact same weights, tokens and optimizer state.
Start with two initializations at B2/T512. Extend the primary check to physical
B512 for both mixed policies. Add nonempty-Adam checkpoints from the short
text runs in step 5, including a checkpoint from each policy; evaluating only
an A-trained checkpoint could favor A's trajectory.

For each state record native CE, logits, layer-output differences versus depth
and token position, all raw parameter gradients, clipping behavior and one
Adam update from cloned moments. Report gradient direction, relative L2,
maximum errors scaled to the reference tensor, error tails, sign changes and
near-zero behavior. Verify nonzero earlier-write credit on a diagnostic loss
designed to exercise it. Avoid retaining every full attention tensor at B512;
save aggregate metrics and targeted replay packets.

Evaluate the full FP32 reference at B512 if feasible. If it does not fit, use
a qualified reference that accumulates the same B512 objective over smaller
FP32 microbatches, with the original B×512 denominator and clipping/Adam applied
once after the full gradient sum. This needs a dedicated accumulation path:
the current loss helper normalizes each invocation independently, and captured
replay clears its gradient buffers. Reusing those calls unchanged would not
produce the intended reference. This is mathematically valid because our
sequences have no cross-example computation; GEMM grouping and rounding still
change. Validate that reference procedure against a physical FP32 batch at a
size that fits, and report the residual. Use higher-precision accumulation if
needed. Label it explicitly as a reference to the full B512 objective, not as
physical-B512 FP32 execution or a throughput result. If that reference cannot
be validated, retain the direct A/B B512 evidence and state the missing C scope.

Exit: determine whether B adds material error, whether A itself has substantial
error against C, and where differences first become significant.

## 4. Investigate only consequential discrepancies

On flagged layers, freeze actual forward operands and the incoming output
gradient. This separates a local backward problem from the effect of changed
forward activations or gradients arriving from later layers. Sample early,
middle and late layers/positions, then expand only where the pattern warrants.

Use small independent FP64 attention calculations and dense contractions if
needed. Compare each policy using its own BF16-representable operands and
declared casts. An entirely FP32 forward is a useful model reference, but is
not by itself a derivative oracle for the different BF16 forward. Test
fixed-forward cotangent scaling at 1/32 and 32; distinguish it from scaling
the loss and recomputing a different low-precision forward.

If a discrepancy is attributable to one precision boundary, test one narrow
intervention at a time: retain FP32 logits/online-softmax calculations, retain
FP32 attention-backward intermediates, or retain a particular accumulation.
Verify the intervention actually changes the intended dtype under autocast.
Select the smallest sufficient protection based on the evidence; do not begin
with a broad factorial search.

Do not use naive BF16 as the accuracy target. The completed
[R3 investigation](reports/r3-bf16-tiled-resolution/results.md) showed that its
repeated weight-gradient rounding can be less accurate than tiled arithmetic.
The later [CDRM investigation](reports/cdrm-numerical-resolution/results.md)
also showed that early-layer gradient error can arrive from surrounding
computation rather than originate in that layer's backward.

## 5. Check learning and practical cost

After startup numerical and replay checks, run A and B for 100 matched updates
on real text. Preserve checkpoints and Adam state for step 3. Treat this as a
screen for instability and gross differences; it is not the final quality
comparison. If the evidence is satisfactory, continue those runs to 500
updates and use a second paired initialization for confirmation, rather than
restarting the first pair or choosing seeds that happen to agree.

Keep the same data order, physical B512, T512, learning-rate schedule and
evaluation data. Evaluate both trained models under one common FP32 evaluation
policy to distinguish learned-weight quality from evaluation-time rounding;
also report each model under its intended deployment precision. Track token
CE, validation CE, gradient norms, clipping frequency, update magnitudes,
nonfinite states and outlier frequency. Parameter trajectory equality is not
required once the two training runs diverge.

Freeze the schedule before running. A 500-step prefix of the released
5000-step warmup does not test peak-learning-rate behavior. Do not compress
the schedule and call it a paper replication. Where no compatible later
checkpoint exists, add a separately labeled, bounded target-learning-rate
stress check from a shared trained checkpoint, or leave that operating regime
explicitly untested. Continued training should monitor through the actual
warmup before broadening the claim.

Benchmark both accepted mixed policies with CUDA graphs at B512/T512, without
diagnostic observers: complete updates, setup cost, peak reservation and
sampled free GPU memory. Keep head chunking, optimizer, internal checkpointing
and token count fixed. Select by measured benefit, not by dtype labels. The
current measured reference is approximately 6.14 seconds/update. At that rate,
one 500-step pair costs about 1.7 GPU-hours, and two paired seeds about 3.4
GPU-hours, excluding setup, evaluation, numerical references and data work.
The candidate's actual speed is not yet known.

## Decision rules and stopping points

Hard failures include nonfinite states, missing or duplicate gradients,
incorrect causal/temporal credit, stale graph replay, wrong dtype restoration
or an unexplained fixed-forward backward inconsistency. Preserve the failure,
localize it and keep B experimental until resolved.

For cross-precision gradients, carry the previous approximately 1.56% global
L2, 3.13% per-tensor L2 and 6.25% maximum/reference-maximum screens as initial
engineering review triggers, not author tolerances or universal failure laws.
Predeclare their exact formulas and FP32 noise floors on calibration controls
before fresh confirmation. Do not transplant a coordinate floor whose
sqrt(parameter-count) contribution would dominate this 217M-parameter model's
gradient. Show how much each floor contributes. Maximum-coordinate flags
remain visible even if local references explain them as ordinary rounding.

For Adam, distinguish the first step with zero moments from trained moments.
The earlier investigations found sizable first-step distances concentrated in
near-zero gradients without an optimizer arithmetic defect. Report full update
distance, direction and material sign flips; a cosine near one alone is not
proof that every update is close. The prior initial cosine >=0.99 and trained
update-distance <=1.56% can be reused as review triggers, with the same
qualification and independently frozen floor treatment.

Proposed practical quality margin: no more than **0.005 nats per supervised
token** degradation for B versus A on the frozen confirmation set, evaluated
with the same precision. This is our proposed engineering margin, not a paper
criterion. Report paired uncertainty across held-out documents and agreement
across the two training seeds. Document resampling does not establish seed
uncertainty. If uncertainty is too wide to resolve the margin, the result is
inconclusive; a non-significant difference is not evidence of equivalence.
Development versus confirmation roles and this margin must be frozen before
seeing confirmation results. Short-pilot success only supports monitored use
at the tested training stage, not equal final convergence.

| Outcome | Disposition |
| --- | --- |
| B has explainable rounding, acceptable update/learning behavior and useful speed or memory benefit | Prefer B for a monitored next training milestone |
| B needs one small FP32 protection | Validate and use that narrower mixed policy |
| B degrades learning, leaves a material unexplained defect, or offers little practical benefit | Keep A; exact precision matching to upstream is not worth a worse result |
| Both A and B disagree materially with C | Investigate the shared computation rather than assuming A is an oracle |

The first useful PR pause is after the dtype/capture audit and initial numerical
report: either B is ready for the bounded text pilot, or a specific discrepancy
has been localized. The complete milestone ends with a scoped precision
recommendation supported by numerical, optimizer, learning and cost evidence.

All future graphable runs go to W&B under `taylorbollman`, with source/config
identities and separate diagnostic/confirmation labels. Retain useful token
fixtures, checkpoints, moments, replay packets and compiler caches under a new
`gs://fast-chunks/cdrm-w-latent/rt-precision-alignment/` lineage, alongside local
records. Preserve prior evidence and defaults until the candidate disposition
is recorded. Run CUDA only inside the required project container on the H100;
never substitute CPU execution for a GPU validation.
