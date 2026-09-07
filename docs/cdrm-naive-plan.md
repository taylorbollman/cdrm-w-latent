> **Updated during execution:** the user replaced the twelve-block pilot below
> with six-block SEQ-first development screening. See
> [the update](inputs/cdrm_six_block_screening_update.md). The completed twelve-block
> NUM/OPS checks and SEQ-only recall trajectory remain historical evidence. No
> twelve-block CDRM pilot was launched. The six-block comparison completed both
> seeds through epoch 45; final checkpoint roles were frozen before generating
> and evaluating the selected copying holdout. See the [completed results](reports/cdrm-naive/results.md).

# Naive FP32 CDRM milestone and bounded pilot

The current [brief](inputs/cdrm_naive_fp32_agent_brief.md) authorizes implementation,
NUM/OPS validation, and a preliminary fixed-data MAD pilot in this milestone.
The four-hour window began 2026-09-07 12:38:30 UTC; training stops by 16:18:30,
reserving twenty minutes for final evaluation, checkpoints, retention and report.
New lineage: `.runtime/cdrm-naive/20260907T123830Z/`, retained under the matching
`gs://fast-chunks/cdrm-w-latent/cdrm-naive/20260907T123830Z/` prefix.

1. Add a narrowly scoped FP32 ordinary-autograd scan after ordinary block 8,
   using post-block-3 and post-block-8 previews and the exact anchor-relative
   equations. Share canonical fused block-3 parameters; keep adapters separately
   owned and nonzero. Preserve ordinary hidden-state indexing and all defaults.
2. Independently test the new composition, causal indexing, lambda-zero bypass,
   rho endpoints, deep temporal credit, read-conditioned writing, parameter
   ownership and summed shared gradients. Do not use the new helper as its own
   sole oracle. Run CPU tiny tests before substantive GPU checks.
3. Validate actual FP32 loss/backward/update at D256/H4/T128, increasing measured
   batch only after fit. Separately check pilot D128/H16/MLP512 and task-native
   lengths/vocabularies. Benchmark matched SEQ/R3/CDRM shapes that fit. No
   accumulation, BF16 repair, custom CDRM backward or detached history.
4. Preserve pinned official MAD generation and train/evaluation supervision.
   Audit alignment, effective length/vocabulary, task oracles/counterfactuals,
   overlap, answer-token metrics and direct example exact match. Resolve any
   native dense training targets explicitly rather than silently adopting the
   previous Stage B answer masks. Fixed train/dev/test splits have 12800/1280/1280
   examples and separate seeds; epoch permutations are shared by the pair.
5. Demonstrate tiny-batch fitting, short fresh-batch training, checkpoint round
   trip and midpoint recovery. Verify adapters/gates, optimizer, RNG, sampler,
   config/source/data identity. Side memory resets per forward.
6. Measure 10–20 real pilot updates and evaluation cost. Before reading pilot
   development results, freeze a feasible first common endpoint near 10/25/50
   epochs for SEQ-12/CDRM-12 recall. Use a 200-epoch cosine horizon, native MAD
   stepping convention once verified, LR5e-4, betas(.9,.98), eps1e-8, WD0,
   clip1, dropout0, FP32, and the largest common tested physical batch up to128.
   Gates remain epsilon=.1/rho1/lambda=.01 after initialization-scale checks.
7. Complete one common pair first. Continuation and a second task compete for
   the remaining budget; choose from measured cost before outcome comparisons.
   Preserve milestones, latest and best-dev roles. Final test is evaluated only
   at the declared endpoint and best-dev checkpoint after choices are fixed.
8. Report NUM, OPS and exploratory SYN separately, including paper alignment,
   actual common epochs, token/sequence scores, baselines, time/memory, source
   identities, retention verification, usage and subsequent joint manifests.

Research comparisons are fresh initializations with corresponding SEQ/CDRM
backbone weights. Existing trained R3 checkpoints are not CDRM preview warm
starts. Previous Stage B and FP32/BF16 evidence remains immutable. The active
same-depth writer uses normalized p3, while the main writer uses normalized
p8-p3; both preserve the ordinary p8 bridge destination.

## Active six-block procedure

Use D128/H16/MLP512 with six blocks. Provisional early/late sites1/3 leave two
ordinary suffix blocks; this architectural choice precedes harder-task outcomes.
Retain the same FP32, normalization, adapter, optimizer and200-epoch schedule
contracts. Validate both six-block topology and actual native-loss GPU execution.

Screen two official one-factor difficulty settings: recall vocabulary128 at
configuredT128(actual127), and selective copying vocabulary16/T256 with96 copy
targets. Generate independent new train/dev streams and leave final generation
until comparison selection. Keep fixed12800/1280 examples and shared shuffles.

SEQ-6 screening has a25-epoch cap, with an early-ace criterion requiring both
answer-token accuracy>=99.9% and whole-example exact match>=99% for three
consecutive completed epochs by epoch10. Use two training seeds to distinguish
repeatable easy settings from a single favorable initialization. Retain all actual
endpoints, initial weights and common milestones; a stopped baseline remains
usable at its compatible checkpoint. Final tests never decide screening.

Prioritize substantial learning with persistent errors. If both models stay near
shortcuts, audit tiny-batch fitting and the native objective before allocating
CDRM; this remains eligible rather than an automatic rejection. Select one
setting and a cost-feasible common endpoint, preserve SEQ's exact initialization
and data order, then run CDRM-6 and compare common epochs. Extend both arms only
within the remaining original session budget, preserving the200-epoch schedule.
Freeze final endpoint/best-development roles before generating/evaluating the
selected fresh final split. Unused easier settings get no CDRM training.
