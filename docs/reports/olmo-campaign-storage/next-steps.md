# Next readiness milestone

Complete storage acceptance first, then prepare the bounded data pool described
in [data-plan.md](data-plan.md). That proposal deliberately separates corpus
capacity from a training budget. Its source weights and Common Crawl stratum
proxy are choices for a transparent pilot, not recovered historical OLMo order.
Resolve those in the data manifest before acquisition; retain complete documents,
exact duplicate/split/exclusion authority and deterministic selected chunk order.
A new versioned ordered reader is necessary: the old packed reader authenticates
canonical shard/record order and cannot silently be repurposed.

Use a small source-balanced membership check first, then prepare the bounded
pool with incremental durable raw/token/index commits. Check reconstruction,
per-loss boundaries/counts, exact held-out disjointness, finite exhaustion and
relocation recovery on CPU. Freeze online tuning membership after estimating
common-FP32 evaluation cost. Keep confirmation outcomes unopened until the later
comparison decision. Stronger near-duplicate guarantees remain unimplemented.

Then measure the intended two-H100 packed T1024 allocation with B and NFR anchors.
Use native BF16, graphs, activation checkpointing and the existing optimizer,
without a new numerical or LR sweep. Start near known sensible physical batches;
[prior K4/T1024 evidence](../olmo-campaign-two-gpu/results.md) favored B12/rank (about 4,311 tokens/s and 14.23 GiB sampled
free), while B16 left only 3.30 GiB. These historical numbers are orientation, not
an acceptance result for the new data/evaluation/storage path. Do not replace
physical batching by accumulation and assume equivalent RT utilization.
Measure valid-token throughput and memory separately from graph preparation,
common-FP32 dev evaluation, checkpoint writes/hashes/transfers and pruning. The
current evaluator uses the training physical batch; introduce a declared smaller
evaluation allocation if necessary, with a bounded reduction/state-preservation
check. H100 allocation does not establish H200 capacity or a different world
size's restart behavior.

Startup must stay explicit. The currently accepted adapted NFR route imports
only fusion weights from 128 FP32 fusion-only updates, then starts fresh Adam for
all active parameters. Its prior exposure is 1,073,565 valid inputs and 1,048,576 CE
targets. It does not import the full-NFR20 backbone/predictor/Adam state. This
adapted NFR and original B are useful capacity fixtures, but are not a matched
startup/exposure quality comparison. Before a scientific cohort, define common
original/fresh versus a separately accounted fusion-adaptation treatment across
all relevant arms; adapted import currently accepts NF/NFR only. Existing
numerical qualifications remain unchanged.

After representative data and allocation are ready, freeze a small operational
pilot's arms, startup, finite schedule, dev cadence and storage capacity. Rehearse
one retained completed-boundary restart on that exact configuration before
larger quality budgets. The proposed 500M screen, cooldown, SFT, compute-matched
extensions and stronger evaluation remain later decisions. There is no queued
large campaign from this storage milestone.

Local keep-two is per new segment. The implementation protects historical data
and restore sources; it is not a global garbage collector. Account for a temporary
third checkpoint plus restore sources. Historical boot-disk cleanup, cross-segment
SSD cleanup and arbitrary topology/checkpoint migration remain separate work.
