# Stage B: first synthetic mechanism pilot

The user authorized this milestone after Stage A and requested durable artifacts
in `gs://fast-chunks`. The frozen configuration is
[configs/stage_b/pilot.json](../configs/stage_b/pilot.json). This is a single-seed
pilot for three task families, paired SEQ/R3 initialization and training examples,
with separate task-native vocabularies. There is no LM checkpoint or latent loss.

## Task and measurement contract

- MQAR: pinned Zoology semantics, eight training associations; evaluate fixed
  association counts at longer delays and a separate 16-association condition.
- Noisy recall: pinned MAD generator with terminal-only, answer-only supervision.
  Train an equal mixture of low/moderate noise; report conditions separately.
  Noise probability also changes record counts. Fixed-record delay extensions
  isolate a different axis and are explicitly adaptations.
- Ordered state updates: three entities, four relevant and four distracting
  updates. Test unseen sequences, an ordered-operation-pair holdout, a queried
  entity-role holdout, longer delays, and eight relevant updates. All symbols
  appear during training. Compare against measured last-record/limited-operation
  heuristics, which can exceed uniform chance because some permutations cancel.

All labels align directly to model logit positions; the trainer must not shift
them again. Answer CE, answer accuracy, and all-answers-correct sequence accuracy
are primary. Context prediction is excluded. Development sets contain 1,024
examples per condition; final test sets contain 4,096. Exact oracle/label and
split/duplicate audits precede training. Every planned training example is
audited and materialized, then reused by both models. Controlled length variants
reuse base semantic examples within a split; train/dev/test remain distinct.

## Model, precision, and execution

The model has 12 blocks, width 256, four heads, MLP width 1024, pre-norm ALiBi,
full MHA, Q/K norms, GELU, and no dropout or biases. R3 replaces index 3, rho=1,
using compiled tiled helpers with four backward chunks. Initial weights are
paired by exhaustive conversion from ordinary SEQ; each run starts a fresh
optimizer. Task embeddings and heads are untied.

Use FP32 parameters and computation, deterministic math SDPA, and TF32 disabled.
The [backend gate](reports/stage-b/backend-summary.md) preserves failed BF16
checks; they are not waived. FP32 random directional derivatives pass at long
lengths and actual training dimensions with unit-L2 cotangents and unchanged
elementwise tolerances. Raw unnormalized-cotangent failures remain recorded.

Before the substantive run, verify fixed-batch learning and the actual runner's
interrupted/resumed trajectory. The small calibration is labeled OPS and has a
separate schedule and weights. Each SYN run uses 2,000 optimizer updates, global
batch 64, length 128, AdamW at 1e-3, betas .9/.95, epsilon 1e-8, no weight decay,
gradient clipping at 1, 100 warmup updates, then cosine decay to .1 of peak LR.
Stopping early never shortens the declared scheduler horizon. Evaluate every
200 updates and choose best-development weights by the declared in-distribution
mean answer CE. Final checkpoints are the primary endpoints; best-development
weights are retained separately. No learned final-test results guide this setup.

The measured FP32 update work for all six runs projects to 26.2 minutes on one
H100, excluding data/evaluation/checkpoint overhead. Use a two-hour total
execution budget for this initial pilot, with progress/checkpoints throughout;
record interrupted or incomplete runs honestly if the budget is reached.
No automatic multi-seed expansion or large LM run is included in this first
pilot. Follow-up replication depends on the observed development/results map.

## Durable artifacts

The session prefix is:

`gs://fast-chunks/cdrm-w-latent/stage-b/20260906T190223Z/`

Retain the plan, exact source snapshot and source revisions/hashes, fixed
train/dev/test arrays, metadata/oracle/baseline audits, all run initialization,
final and best-development checkpoints, recovery state, learning curves, and
per-example final predictions. Local working files live under
`.runtime/stage-b/20260906T190223Z/`; completed artifacts are copied to the
specified bucket and recorded in a storage manifest. Upload only these project
artifacts, excluding credentials and private environment files.
