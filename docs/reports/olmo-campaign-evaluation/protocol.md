# Declared per-pass evaluation during shared campaign execution

This milestone adds held-out evaluation at completed optimizer boundaries. It
tests integration and preservation of the next training update, not model
quality, BF16 accuracy against FP32, a new training recipe or a production run.
Existing accepted helpers, model code, tests and protocols remain unchanged.
The new evaluator, controller, engine, CLI, tests and this protocol freeze before
GPU execution; declarations, resolved plans and source snapshots are retained.

## Training and evaluation identities

`olmo_campaign_eval_execute.py` constructs the same campaign models and optimizer
ownership as the accepted launcher. Its versioned engine explicitly calls
`EvaluationController` after optimizer, counters and training cursor have all
committed. It does not patch the old engine or change the captured loss/update.
Native execution retains BF16 mixed training with FP32 masters, fused AdamW,
deterministic Flash SDPA, native Triton RT and the existing checkpoint/graph
settings. The new evaluation declaration and source inventory intentionally
produce a different execution identity and checkpoint configuration.

Resume accepts only this exact declared execution lineage. The previous PR46
checkpoint is not migrated into the new identity. Comparing completed report
evidence across the two versions is a separate offline acceptance control:
permit only the explicitly declared evaluation, source/configuration schema,
derived identity and output/tracking/retention-location differences. Require the
training recipe, packed membership, startup weights, fresh optimizer policy,
ownership, scheduler, process/named RNG policy, precision/runtime and DDP settings
to match. Inherited training source files must retain their exact prior hashes.

## Held-out membership and loss semantics

The declaration supplies a pinned, separately indexed `dev` split from the same
prepared token corpus, vocabulary, EOS ID, padding ID and packing policy as
training. Evaluation uses the same sequence length, including T1024 for native
acceptance. Its finite valid-token budget and deterministic first-prefix plan
are resolved before CUDA; each scheduled evaluation reuses that prefix. There
is no training-cursor advance, corpus cycling, retokenization, outcome-dependent
selection or random dev sampling. The exact dev index, membership, counts and
rank allocation belong to the retained declaration/resolution. This small
prepared-corpus prefix is a readiness fixture, not a representative Dolma score.

Evaluate all trained passes with the existing canonical `loss_sums`, setting
only evaluation feedback jitter to zero. Use FP32 math attention, eager native
RT and ordinary pointwise operations, native RoPE, disabled autocast and TF32,
`eval()` and `no_grad()`. No generation or online recurrence evaluation is added.
Training retains its declared keyed jitter and BF16 backend after restoration.

For every pass record CE, latent and KL sums plus their separate integer counts.
Continuous packed attention/CE/feedback carry across real document boundaries;
latent pairs and KL triples retain same-document exclusions. Row-end truncation,
true EOS metadata, explicit masks, right padding and local dummy rows are
unchanged. Disabled auxiliary terms have zero sums/counts and no mean. Reduce
raw sums and counts across batches and ranks before division; do not average
local means. Counts are eligible positions once, not multiplied by pass count.
For K4, canonical CE pass coefficients are `(1/2, 1/6, 1/6, 1/6)` and auxiliary
coefficients are uniform. Report each pass and the canonical weighted aggregate.

## Live state, scheduling and failures

Evaluation starts only with absent or zero-valued completed-boundary gradients.
Restore every module's training mode, all native runtime flags, local/named RNG
and autocast state on success or failure. Preserve parameter/buffer ownership,
storage and versions, cache generations and zero-gradient storage/values.
The controller also checks training cursor, prepared input/noise/loss/forward
tables, pointer layout and graph owners before/after evaluation. Acceptance mode
adds complete model/Adam/scheduler/RNG/counter/cursor boundary hashes. Lean
metadata checks do not claim to detect arbitrary external `.data` corruption.

An evaluation publishes only after local restoration, global accounting and
boundary checks pass. A failure stops the segment; it does not checkpoint an
unverified live state. The earlier committed checkpoint remains authoritative.
Training-update logging and normal checkpoint retention follow successful
evaluation. Checkpoint cadence is checked at completed boundaries and is not a
hard upper bound on one evaluation or storage operation.

On same-lineage resume, an eligible restored boundary is evaluated once in that
new segment, even if a prior segment evaluated it. Publication is therefore
segment-scoped and explicitly recorded. An eligible terminal resume may evaluate
but must not prepare a training graph, advance the cursor or perform an update.

## Bounded acceptance sequence

CPU checks cover all eight component selections with literal per-position loss
oracles, same-document masks, disabled/empty terms, global reductions, schedule
selection, declarations/identities, restoration on errors and unchanged next
prepared updates. Native constructors remain covered by existing ownership and
startup contracts. New tests must complete before source freeze/GPU execution.

Tiny two-rank GPU acceptance runs a three-update reference and matching insertion
at completed update2, then checks update3 exactly. A stop at update2 followed by
fresh-process same-lineage resume repeats that eligible evaluation once and
reproduces the next update. Compare rank inputs, losses, raw gradients and full
model/Adam/schedule/RNG/cursor boundaries, including accumulation/dummy slots.
Check graph preparation preserves the restored boundary. Any terminal-resume
check must record zero new updates and no capture.

Native acceptance inserts evaluation after update2 of the same three-update NFR
setup already retained by PR46. Its reference report is
`.runtime/olmo-campaign-execution/native-reference-01/report.json`, SHA256
`362d0b1fb058e1287980a088272275f3daf51ba71ed025b538e7a7d743c518c7`.
Reuse this completed no-evaluation reference instead of another roughly17-minute
native reference run. Startup is original OLMo backbone/predictor plus strict
fusion128 weights, followed by fresh all-active Adam and reset training clocks;
it is not the later full-NFR20 endpoint or original pretraining optimizer history.
Native RT selects layers0/15, alpha1; FBT is K4/beta1; both NextLat losses are on.
At T1024/B1 per rank, each update has three real rows, two physical slots per
rank and one dummy row on rank1. Three updates total9,216 valid input tokens,
9,207 CE targets,9,207 latent pairs,9,198 KL triples and12 physical microbatches.
Training has no internal document boundary in this particular small prefix.

Require exact pre-evaluation training controls and exact update3/final state
against PR46, with the declared identity differences above. A mismatch triggers
configuration/source localization before attributing it to evaluation. Tiny
same-identity replay and native cross-version training parity are distinct tests.
Root alone launches bounded GPU jobs and retains checkpoints/evidence in GCS;
graphable evaluation metrics also go to W&B. Report evaluation, integrity hashing,
graph preparation, checkpoint and transfer time separately. These timings do not
establish optimized throughput, capacity, a quality win or BF16 clearance.
