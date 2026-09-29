# Next readiness milestone

After this execution/recovery acceptance, connect the already-declared finite
held-out evaluation policy to the common launcher. The manifest resolver and
execution identity support a pinned evaluation plan, but the native CLI currently
rejects every evaluation choice except `deferred`. Do not silently launch a
quality campaign or treat the readiness corpus as the selected training mixture.

## Per-pass evaluation with preserved live training state

Use `FBTNextLatLM.loss_sums`, `pass_losses` and `term_pass_coefficients` to report
raw sums, separate global target counts and means for CE/latent/KL at every trained
pass, plus the unchanged weighted campaign aggregates. For B/R/F/FR, explicitly
label auxiliary terms disabled with count0. Reduce global sums/counts before
normalizing; do not average rank means. Preserve continuous-stream CE and the
same-document auxiliary masks across padding and dummy/empty local slots.

Read a separately pinned finite dev plan through `PackedCampaignData`, with an
independent cursor and declared physical allocation. Consume the existing
no-jitter/common-FP32 evaluation policy explicitly. Training mode, graph inputs,
gradient-buffer ownership, RNG and native runtime flags must survive evaluation
and errors unchanged. `evaluation_scope` in `olmo_campaign_eval_insertion` is a
useful preservation reference, but it does not switch RT precision/backend flags.
The FP32 evaluator in `olmo_fusion_startup_nfr_updates` provides a runtime-switch
pattern, but is not a drop-in live-graph evaluator: it requires cleared gradients
and retains its fixed training jitter. Build a new explicit adapter around the
canonical unwrapped model; do not patch historical observer globals.

Insert evaluation only at a completed optimizer/cursor boundary. Specify when a
resumed run repeats or skips an already-recorded evaluation, and distinguish the
last trained boundary from the last published evaluation. Coordinate rank-local
errors and restore runtime/cache/modes/RNG/zero-valued graph gradient buffers in
`finally`. A failed evaluation must not invalidate an earlier committed checkpoint.

Bound acceptance: literal CPU per-pass/count oracles across the eight arms,
partition and dummy cases, preservation/errors and resume scheduling; then a tiny
two-rank evaluation insertion that leaves the next training update exact. Add one
bounded native insertion for RT runtime restoration and memory, if the new path
introduces behavior not covered by the tiny case. This is evaluation plumbing,
not another broad BF16/FP32 training sweep. Keep new versioned sources beside the
frozen execution lineage.

## Pilot decisions after readiness

Review the actual source mixture/order, representative disjoint held-out coverage
and evaluation frequency, original versus adapted startup/exposure accounting,
finite training budget, selected arms/RT layers, and target hardware. Choose
per-arm physical batches and accumulation from actual capacity/steady-state
measurements while keeping logical token/data exposure explicit. The native
B1/rank acceptance allocation exists to bound a recovery test and exercise a local
empty slot; it is not a throughput or training-batch recommendation.

Measure on the intended H100/H200 topology after warmup, with evaluation and
checkpoint costs separate. Same-topology exact recovery does not establish
checkpoint portability to a changed runtime/device count or guarantee H200 batch
capacity. Schedule extension, arbitrary adapted-Adam migration, resharding and
local pruning remain separate work. Retain the prior numerical qualifications;
no new Q/K normalization change or extra numerical sweep is implied.
