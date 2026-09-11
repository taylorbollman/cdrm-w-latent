**Plan: resolve the five-block CDRM numerical questions independently of task performance**

Implemented after the first 1000-update pilot. The [numerical results and
qualifications](reports/cdrm-numerical-resolution/results.md) record the
diagnosis, selected ordinary-attention-FP32 candidate and three fresh numerical
confirmation roles. The original proposed plan is retained in the new artifact
lineage alongside the unchanged pilot and its failed screens. The numbered
procedure below preserves the plan that guided this work.

The objective is to distinguish an incorrect derivative or replay from the
expected effects of the declared BF16 arithmetic, then either make a small,
validated correction or document the precision limitation. Task accuracy,
learning curves and longer training are outside this investigation. The saved
task examples and aligned CE remain useful numerical test inputs; using their
gradients does not require a learning experiment.

The [pilot report](reports/cdrm-tiled-pilot/results.md) supplies two retained
same-state failures at update1000. At the FP32-trained weights, BF16 gradient
and Adam-update relative L2 errors are 4.4423% and 4.6329%; at the BF16-trained
weights they are 2.5737% and 2.2058%. Each comparison uses identical weights and
moments within its three precision/reference arms. The two checkpoint roles
use different fixed B64 batches, so their error sizes are not a controlled
comparison of training precision. Both isolated BF16 memory-gradient probes
pass their applicable screens, at approximately 0.44–0.45% relative L2.

On the FP32-trained reproducer, lambda-zero retains 4.3984% gradient and 4.4175%
Adam error. An FP32 output head retains 4.4387% and 4.6151%. These findings make
ordinary-backbone precision the first investigation target. They do not prove
that the fabric is correct for every input and incoming derivative.

**What lambda means, and useful simplifications**

For the current zero-based sites1/3, the input to ordinary block4 is

\[
\widetilde h^{(3)}
=h^{(3)}+\lambda W_B N(\widehat m-h^{(1)}).
\]

Here lambda is a fixed configuration scalar, 0.01. It is not learned. The
bridge matrix W_B is learned; the other extra parameter matrix is the deep
adapter W_D. Setting lambda to1 removes the multiplier but increases the
branch contribution100 times. That changes the operating point and is not
the first diagnostic. Making lambda smaller can conceal whole-model branch
errors; the existing direct, unscaled proposed-memory probe already avoids
that problem.

The clean initial simplification is to bypass the fabric entirely while
preserving ordinary-block precision, attention backend, masks, weights and
all runtime settings. Compare that bypass with the existing lambda-zero
control. Keep the CDRM configuration flag active where necessary so its
attention-backend selection does not change incidentally. Shared-backbone
outputs and gradients should agree. For an optimizer control, distinguish
zero adapter gradients from absent gradients: trained Adam updates moments
for zero gradients but skips parameters whose gradients are absent. Preserve
the zero-gradient convention explicitly if comparing complete optimizer
packets; otherwise label the control as backbone-only.

Further simplifications are conditional. Replay isolated blocks with fixed
inputs and incoming gradients before changing the architecture. If the fabric
becomes implicated, detach its two preview inputs while preserving the direct
late-residual path, and differentiate those preview values as independent
leaves. If shared ownership becomes implicated, create identical untied copies
and compare their summed gradients with the canonical shared gradient.
Replacing adapters by identity maps or removing normalization changes the
function and should follow a specific hypothesis. Freezing a parameter does
not eliminate its forward arithmetic and must not count as clearing its
gradient by omitting it from measurement.

1. **Freeze a small diagnostic set and verify the precision actually used.**

   Reuse both saved u1000 checkpoints and their failing B64/T256 fixtures,
   including weights, moments, labels, CE reduction and raw tensor packets.
   Include initialization or the earlier u100 checkpoint as a secondary
   reference if checkpoint dependence helps localization. Existing inspected
   fixtures are diagnostic cases, not fresh confirmation.

   Record every relevant effective dtype and backend setting: TF32, math SDPA,
   ALiBi/mask dtype, dense inputs/outputs, normalized Q/K storage, residuals,
   saved forward values, adjoints, parameter gradients, autocast cache and BF16
   reduction policy. Reproduce the uninstrumented failure before relying on
   hooks; verify that instrumented execution preserves the observation.

   The current common setup explicitly enables BF16 reduced-precision GEMM
   reductions. Turning that setting off is a cheap diagnostic control, applied
   after setup and recorded as an intentional deviation. It cannot recover
   precision already lost in BF16 operands or final output casts. PyTorch
   documents both reduction truncation and operation-dependent autocast
   behavior; the installed container and captured tensors remain authoritative
   over newer documentation defaults. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
   [AMP](https://docs.pytorch.org/docs/2.14/amp.html).

   Deliverable: a reproducible numerical fixture and an observed dtype map.

2. **Locate where error is introduced and where it is amplified.**

   Capture activations and incoming gradients at all five ordinary-block
   boundaries, final norm/head, and the fabric input/output. Retain relative
   L2, absolute and normalized maxima, error direction, signed bias and error
   energy, rather than only a pass/fail count. Large embedding-gradient error
   can be propagated from later operations; it does not imply that embedding
   lookup is the source.

   Use the clean bypass to simplify the backbone investigation. Replay each
   implicated block with identical captured inputs and identical incoming
   gradients. This removes changes in surrounding blocks from that local
   comparison. A shared saved loss cotangent additionally isolates changes in
   the loss derivative, but is labeled a derivative probe, not the original
   end-to-end CE gradient.

   On the full model, first test FP32 ordinary blocks/head with the memory
   path retaining its BF16 policy. Then selectively promote ordinary block
   calls in the mixed backbone, or reintroduce BF16 in the implicated region.
   There are only five blocks. Use the intermediate measurements to interpret
   the intervention; error cancellation means an aggregate pass alone is not
   a causal explanation. Investigate interactions only if the measurements
   show that a single-region intervention is insufficient.

   Current concrete leads are the ALiBi cast in model.py:_cast_attn_bias,
   the recast of normalized Q/K to the projection dtype, attention output
   storage, and QKV/MLP forward and backward contractions. Test FP32 ALiBi with
   identical Q/K/V first where attention is implicated, preserving causality;
   then expand to the smallest attention or projection region needed.

   Wrap an ordinary block call rather than casually wrapping shared child
   modules: the fabric reuses the early owner's MLP modules and projects QKV
   through functional weight slices that bypass Linear.forward hooks. Record
   execution phase and preserve a single canonical optimizer owner.

   Deliverable: a small implicated subgraph and interventions with predicted,
   reproducible effects on its intermediate error, not merely lower totals.

3. **Establish a derivative oracle for the same numerical operands.**

   Full FP32 and BF16 executions evaluate derivatives at different rounded
   activations. Their distance does not isolate a backward implementation bug.
   At the implicated primitive, capture the actual forward operands and
   incoming derivative, and independently compute its local derivative in
   FP64. For a dense projection this includes both dW=dY^T X and dX=dY W,
   followed separately by the explicitly expected storage/cast operations.
   Extend to normalization and attention formulas if they are implicated;
   a correct dense contraction alone does not validate a whole block.

   Compare native arithmetic against the same-operand reference, then compare
   the references evaluated on the FP32 and BF16 operands. If necessary,
   reconstruct the reverse pass of the extracted subgraph from saved primal
   values with high-precision adjoints. Separate forward-operand perturbation,
   permitted backward rounding/casts, and residual implementation disagreement.
   Report vector differences and their interaction; their norms are not
   independent percentages that necessarily add up.

   This reference checks the backward rules used by AMP. Finite differences
   through literal BF16 rounding do not supply a smooth derivative oracle.
   Directional finite differences can validate the corresponding smooth FP64
   primitive after removing discrete casts. Declare local-oracle tolerances
   from the specified operations, casts and reduction behavior before judging
   a candidate; do not demand one universal ULP bound for every network tensor.

   Deliverable: an explanation of whether the native backward implements the
   declared arithmetic correctly, and where forward/backward precision is lost.

4. **Analyze clipping and Adam as fixed numerical functions.**

   Feed captured gradient packets into identical saved optimizer state. Use an
   independent FP64 calculation of the actual clipping and Adam equations to
   distinguish optimizer implementation error from sensitivity to its inputs.
   Compare raw gradients, clipped gradients, new moments and proposed parameter
   updates. A shared clipping coefficient is a diagnostic to isolate the norm
   effect; final checks retain the real clipping rule. No weights are carried
   into a training trajectory.

   Keep LR, betas, epsilon, clipping and loss reduction unchanged. Retain
   relative and absolute update errors, angles, per-tensor maxima, cancellation
   and the existing near-zero mask. Most observed Adam error energy lies
   outside that mask, so small-gradient sign flips are not a sufficient
   explanation. Correct, finite optimizer arithmetic alone does not establish
   that its two proposed updates are equivalent.

   Deliverable: quantified conversion of the identified gradient perturbation
   into the observed update discrepancy.

5. **Apply one or two mechanism-supported corrections.**

   If a replay, derivative or accumulation is wrong, repair that operation and
   retain its reproducer as a regression test. If forward rounding dominates,
   preserve FP32 only at the implicated inputs, operation or stored intermediate.
   If backward arithmetic dominates, promote the implicated reduction or
   adjoint before information is rounded away. Casting an already rounded
   gradient to FP32 cannot fix it. If optimizer sensitivity dominates, test the
   smallest upstream precision change that reduces the problematic perturbation.

   Runtime/compiler/cache variants are tested only if local evidence points
   there. Keep a passing FP32 ordinary-backbone plus BF16-memory configuration,
   if found, as a diagnostic upper bound on the required precision. It is not
   automatically the preferred final mixed policy. Do not rewrite the tiled
   recurrence, replace learned normalization, sweep gates or tune optimizer
   parameters without a demonstrated reason.

   Deliverable: a small patch with a causal account, or a localized limitation
   showing why a selective patch is insufficient. Pause after one or two
   justified candidates if the remaining remedy effectively requires FP32
   throughout; avoid an open-ended search for an all-pass result.

6. **Confirm the selected result on untouched numerical cases.**

   Freeze the candidate, precision policy and applicable criteria before new
   confirmation. Generate a small separately designated set of fresh B64/T256
   numerical batches, preassign them to both retained trained checkpoints and
   a distinct initialization, and then run the comparisons. Do not repurpose
   the old pilot's unused2500/second-seed slots after seeing their role schedule.
   Small-B/local-FP64 results aid diagnosis but do not replace physicalB64.

   Retain actual aligned CE gradients, all canonical parameters and relevant
   inputs, unscaled memory/bridge states, independent side probes, intended
   gradient presence/finiteness, and identical-state clipping/Adam deltas.
   Use validated tiled FP32 as the primary full-model reference, with naive
   FP32 cross-checks; temporary naive BF16 is not an acceptance target. Include
   the fixed-forward powers-of-two cotangent-scaling check at1/32 and32, with
   normalized resulting derivatives, to detect unexplained backward behavior.
   Run semantic regressions affected by the patch, including future-reader
   credit and shared ownership if any memory path changes. Verify the intended
   compiled path with observers removed. Graph numerical errors and attribution
   in W&B and retain fixtures, tensors, source and decisions in a new GCS
   lineage. No task-performance study is included.

**What counts as resolution**

The existing approximately1.56% global-gradient/Adam screens are reasonable
prospective alarms, not mathematical error bounds for a multi-layer nonlinear
BF16 network. They remain visible and unchanged in every comparison. Expected
BF16 rounding can amplify through a composition even when each local operation
is implemented correctly; high cosine or matching a generic AMP execution is
not sufficient by itself to dismiss a failure.

The strongest outcome is a causally supported precision correction that passes
the existing applicable screens on the reproducers and fresh confirmation,
with semantic and independent memory checks intact. That resolves the issue
within the tested configuration and execution policy.

Another legitimate outcome is to establish that the implementation correctly
realizes its declared BF16 arithmetic, with observed differences explained by
measured conditioning and expected rounding. This resolves a suspected
derivative bug but does not claim FP32 equivalence. If the old global screen is
shown to be inappropriate for the intended precision contract, propose a
separately versioned, operation/conditioning-based contract and freeze it before
fresh confirmation. Do not fit a larger threshold to the failed observations.
Any accepted scope must name the remaining end-to-end gradient/update distance.

If no selective correction or defensible revised numerical requirement is
supported, keep BF16 experimental and FP32 as the dependable reference. A
numerical investigation can establish implementation correctness and a bounded
precision specification; it cannot prove task-level harmlessness without
evidence outside this numerical-only scope.

The first reviewable milestone is the reproducer, dtype/adjoint map and causal
localization report. The next is a focused correction with fresh numerical
confirmation, or an explicit, justified precision limitation. Further training
is not required to reach either milestone.
