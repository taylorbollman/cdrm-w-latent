# Bounded five-block tiled CDRM learning pilot

Authorized on 2026-09-07. Lineage `20260907T212606Z`; parent implementation and
100-update evidence: `cdrm-tiled-bf16/20260907T191403Z`. This protocol is frozen
before generating fresh numerical-confirmation examples or inspecting new
training outcomes. It preserves the [prior numerical qualification](../cdrm-tiled-bf16/results.md).

## Questions and unchanged model

Compare ordinary SEQ-5 FP32, tiled CDRM-5 FP32 and tiled CDRM-5 BF16 from matched
backbone initializations. The architecture comparison is FP32 SEQ versus FP32
CDRM; the precision comparison is FP32 versus BF16 CDRM. No accuracy, speed or
numerical-clearance claim is assumed.

Keep five ordinary blocks, one fabric at zero-based sites 1/3, D128/H16, full
MHA, MLP512/GELU, learned normalization/QK normalization, ALiBi and zero dropout.
CDRM gates remain epsilon 0.1, rho 1, lambda 0.01. Use the existing BF16 dense /
FP32 state policy including its BF16 output head. The two CDRM adapters are the
only extra parameter owners. SEQ actually disables the fabric and shares all
ordinary starting tensors; a lambda-zero CDRM evaluation is a separate control.

Use the parent's fixed native MAD selective-copying V16/T256/K96 train/dev
datasets, with 12,800/256 examples and physical B64. Native labels are already
aligned with logits; apply no additional shift. Use the same ordered examples,
shuffle seed 925704, AdamW LR 5e-4, betas (0.9,0.98), epsilon 1e-8, zero weight
decay and clip norm 1. Preserve the 200-epoch cosine schedule and its 1e-6 minimum,
stepping only after complete 200-update epochs. No accumulation or precision
retuning is introduced.

## Cohorts, endpoints and development decisions

The first model seed is 7500. Its two CDRM arms inherit their exact update-100
model/Adam/scheduler/RNG/data-position state from the parent milestone. SEQ
starts from the corresponding original common backbone, not trained CDRM
weights. The optional second seed is 7501, with separate deterministic adapter
and training RNG seeds under the existing initialization rule. It shares the
fixed corpus and shuffle policy; it is a second initialization, not a new task.

Run SEQ first. Inspect development metrics at updates 200,400,600,800,1000 and
the later common endpoint; retain checkpoints including 1000 and 2500. The
first review point is **1000 total updates (5 epochs)**, with a maximum of
**2500 total updates (12.5 epochs)** per arm. For continuation to2500 keep
evaluation every200 updates and add2500. The common endpoint answers an
equal-update question; also report time and scored tokens, because CDRM adds
computation and parameters.

Early-ace criterion: SEQ token accuracy >=99.9% and direct whole-sequence exact
match >=99% at three consecutive evaluation points through1000. If reached,
deprioritize this task and stop new CDRM investment here. Do not infer exact
match by exponentiating token accuracy.

At1000, substantial learning is evidenced by any arm exceeding the fixed
order-ignoring modal development baseline by >=5 percentage points, or reducing
development CE by >=20% below the uniform-source-target reference `log(14)`
(i.e. CE <=2.11125 nats). Improvement only from a randomly initialized model's
loss is insufficient: the parent100-update smoke already achieves that while
remaining near a simple shortcut. Continue to2500 if at least
one arm demonstrates such learning, the task is not early-aced, and numerical /
operational concerns below are resolved sufficiently for this bounded use.
If neither criterion is met, inspect curves and supervision/optimization rather
than conclude the architecture is ineffective. Any justified departure from
this allocation rule is recorded explicitly; do not silently redefine success.

At2500, an informative pilot means substantial learning in at least one arm,
remaining task errors, and no unresolved semantic or operational blocker. In
that case repeat the matched cohort with seed7501, regardless of which
architecture won seed7500. Use the same1000/2500 rules. Cap this milestone at
two initializations and2500 updates per arm; no architecture/hyperparameter
sweep or automatic long continuation follows.

Development metrics are native CE, answer-token accuracy and direct
whole-sequence exact match. Record updates, learning rates, clipping, intended
gradient finiteness, owner/adapter signals and correction magnitudes. Review
paired FP32/BF16 CE over matched update windows; a >0.02-nat BF16 degradation
over the last200 matched updates or at two consecutive development evaluations
triggers investigation, not an immediate declaration of numerical failure.
No final research test set is generated or used in this development pilot.

## Fresh numerical checks and branch use

Generate a new confirmation dataset before training: seed925801,768 examples,
native V16/T256/K96. Freeze eight disjoint B64 roles in order:
seed7500 / update1000 / FP32-trained and BF16-trained checkpoints;
seed7500 / update2500 / FP32-trained and BF16-trained checkpoints;
then the same four roles for seed7501. These consume offsets0,64,...,448;
the final256 examples are reserved and unused. Audit exact input overlap with
the inherited train/dev/confirmation datasets. Previously inspected numerical
examples remain diagnostic evidence, not fresh confirmation.

For each used role, compare naive FP32, tiled FP32 and tiled BF16 at the same
saved CDRM weights and optimizer moments using the frozen prior numerical
contract. Include actual CE gradients, unscaled proposed-memory/bridge states,
the independent unnormalized side-gradient packet and Adam deltas. Comparing
gradients at two independently diverged trained models is not a precision test.

Known FP32 raw-side coordinate rounding flags and the explained BF16 aggregate
gradient flag stay visible; their recurrence alone does not automatically block
this reviewed opt-in use. Missing/nonfinite intended gradients, a new temporal
or ownership defect, new isolated-side/maximum/optimizer failures, or persistent
loss degradation require investigation. Do not raise thresholds, replace
confirmation examples, or quietly substitute the FP32-head diagnostic policy.
Additional FP64 or head controls are reserved for new discrepancies requiring
localization, not repeated solely to obtain an all-pass summary.

At the1000 and2500 checkpoints, evaluate each CDRM on development examples with
its normal bridge and with lambda zero, without training or changing weights,
optimizer state or RNG. This measures dependence on the learned branch. The
separately trained SEQ arm measures the benefit of training with the fabric;
these interpretations must not be conflated.

## Continuation, recovery and retention

Add a new continuation entry point instead of editing the frozen parent
harness. Record explicit parent checkpoint/source/config/data identities and
the child runner; permit only declared arm differences and longer stopping /
evaluation/checkpoint controls. Restore all training state exactly. Save the
next-batch hash and verify the complete paired data order and LR sequence.

Add one BF16 recovery check across an epoch boundary, e.g. resume190 to210
against the uninterrupted210 checkpoint, checking model, optimizer, scheduler,
RNG, position and non-timing metrics. Reuse the retained parent compiler cache;
its raw mutable directory is outside the parent's immutable artifact manifest.
Keep the original archived cache and all parent scientific files unchanged,
and archive the resulting cache with this child lineage.

All GPU work runs inside the project container after its GPU check. Training,
evaluation and numerical curves log online to W&B entity`taylorbollman`, project
`cdrm-tiled-learning-pilot`, group`20260907T212606Z`. Retain local source snapshots,
data identities, checkpoints, numerical tensors, plots, decisions and failed
attempts; archive and checksum-verify useful artifacts under
`gs://fast-chunks/cdrm-w-latent/cdrm-tiled-pilot/20260907T212606Z/`.

The deliverable is a bounded learning/precision comparison with explicit
qualifications, branch-ablation evidence and an epoch-boundary recovery check.
It does not establish long-run convergence, CDRM benefit across tasks, general
rho/multiple fabrics, linear backward memory or production-wide BF16 clearance.
