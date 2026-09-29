# Readiness and precision disposition

Continue with the selected original OLMo-1B step 60000 checkpoint and native RT.
The next work should complete evaluation/recovery acceptance, then choose and
validate an actual data/compute plan. It does not require another broad precision
sweep, an epsilon change, Q/K normalization, a new architecture or a quality run.
This updates the [PR46 next step](../olmo-campaign-execution/next-steps.md); the
[broader experiment plan](../../olmo-nextlat-fbt-rt-experiment-plan-v1.md) remains
a proposal whose older implementation-status and isolated-data sections are
superseded by the completed packed execution work.

## What the optimizer result supports

The [fixed-state update 20 diagnostic](../olmo-optimizer-history/results.md)
finds 1.289% backbone raw-gradient disagreement and global actual-update
disagreement 0.969% with inherited Adam versus 1.942% with reset Adam. The inherited
absolute update difference is also 7.29 times smaller. Removing the shared
zero-gradient Adam contribution narrows the relative comparison to 1.640% versus
1.942%; shared momentum is part of the explanation, not the whole result.

Keep Adam moments, steps, schedule and RNG when resuming the same accepted
trajectory. Do not silently reset them during interruption recovery. This does
not establish a superior startup optimizer for a new objective: these are our
20 adaptation steps, not original pretraining moments. Original raw optimizer
files are historically documented but public retrieval remains unverified;
weights-only initialization with explicitly fresh Adam remains a valid design
choice. Do not assign a historical step count to zero moments.

The smaller current gradient discrepancy uses a different state and batch from
the earlier 13.447% initial backbone result. The controlled optimizer comparison
does not isolate why raw gradients changed, validate a cold BF16 start, erase
prior trajectory differences or supply a universal numerical threshold. Prior
shared-origin short continuations and packed checks support a qualified adapted
functionality pilot; they are not independent cold-start or long-run evidence.
An independently verified same-fixture pre-adaptation reference is more direct:
backbone error 12.059% → 1.289% and absolute difference 21.4491 → 0.119592 after 20
full-model updates. This supports lower sensitivity at that trained state on
the same data, while predictor relative error increases; it still does not
attribute the raw-gradient change to optimizer history or establish generality.

## Close evaluation integration before expanding the experiment

The new [protocol](protocol.md) gives evaluation its own declared source and
execution identity. Finish the bounded tiny reference/insertion and same-lineage
stop/resume checks, then native insertion against the retained PR46 training
reference. Report their terminal results before declaring this milestone ready;
GPU status belongs in [progress](progress.md), not inferred from code completion.

Require unchanged next-update inputs, losses, gradients and full committed
boundaries. Compare native training across versions only with the explicitly
allowed source/evaluation identity differences; never resume a checkpoint across
those identities. Eligible restored boundaries may evaluate once in each new
segment. Evaluation uses common FP32, no jitter, all trained passes and global
per-term sums/counts. Failed integrity leaves the earlier committed checkpoint
authoritative. Preserve these rules in any later data or hardware adapter.

The prepared T1024 dev index contains 104 documents and 49,427 tokens: 49 chunks,
with 275 valid tokens in the final chunk. Its manifest SHA256 is
`af5ff22203ee9b00fc16b213965f9d134ce209632759e29c14580287584cba2f`.
The planned 3,072-token acceptance prefix covers only the first two C4 documents;
their true boundary is at stream position 1663. It can exercise real boundary
exclusions but is not a representative dev mixture or a quality benchmark.
Original pretraining exposure to these documents also remains unknown.

## Next concrete preparation

1. **Select data and freeze exposure.** Agree on a bounded Dolma-v1.5 source
   mixture/order and document-disjoint tuning-dev/confirmation sets; retain the
   pinned tokenizer, true boundaries and continuous packed T1024 semantics.
   Resolve train/evaluation membership, per-source token shares, exact targets,
   padding and finite exhaustion before any model run. Do not reuse an ordered
   readiness prefix as the production mixture by default.
2. **Choose startup explicitly.** A common original-weights/fresh-Adam cohort is
   different from a cohort with preliminary fusion adaptation. Fusion-only FP32
   warmup through update 128 has bounded supporting evidence for NF and the subsequent NFR
   functionality checks, not a universally selected startup. If adopted, account
   for its extra data/compute and define comparable treatment across F arms;
   current adapted import support is NF/NFR. The full-NFR update 20 endpoint additionally
   changes backbone/predictor and optimizer history and must not masquerade as
   original OLMo plus fusion warmup. Arbitrary inherited-Adam migration stays
   outside the accepted same-lineage resume contract.
3. **Measure the intended hardware/allocation.** Use representative packed masks
   and the actual enabled components to choose physical batch and accumulation
   with memory headroom. The B1/rank native recovery fixture is not a throughput
   recommendation. Large physical batches may matter for RT; accumulation alone
   does not improve its per-call utilization. Separate steady training tokens/s,
   valid-token utilization, evaluation, capture and retention costs. Declare
   evaluation allocation explicitly if it must differ for FP32 memory capacity.
4. **Freeze a small operational pilot before quality budgets.** Bind the chosen
   arms, startup, data, finite token/update budget, schedule, dev cadence and
   storage policy in the manifest. On the intended topology, rehearse a retained
   completed-boundary restart and evaluation insertion before committing the
   larger experiment. H100 results do not establish H200 capacity or changed-rank
   exact restart; new device/world-size behavior needs its own bounded check.

The current boot-disk check shows approximately 110 GiB free on the 1 TB disk,
before the new native checkpoint states of approximately 20 GB. Local pruning
is not implemented. Before a long campaign, explicitly choose SSD checkpoint
staging, verified GCS retention and a local checkpoint retention policy. The
current CLI confines evidence and checkpoint roots to the persistent project
directory, so SSD staging may also require an explicit path-authority change.
This is a readiness follow-up, not approval to start a production run.

The earlier 500M-token screen, cooldown, SFT and compute-matched extensions remain
later decisions. No quality campaign, LR grid, SFT/generation implementation,
schedule extension or cross-topology checkpoint migration is launched by this
readiness plan. Preserve checkpoints in GCS and source/declaration evidence so
the eventual comparisons have an explicit starting point and accounting.
