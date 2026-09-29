# Optimizer-history and evaluation work

2026-09-29 15:55 UTC start; user authorizes roughly90minutes without review.
Branch feat/olmo-optimizer-history-and-eval from PR46 main e3ae7e2.
Root owns every GPU launch, runtime declaration and GCS retention. Prior
execution sources/tests/protocols and cdrm/pretrained remain frozen.

Two independent tracks: (1) fixed saved NFR BF16-update20 weights/data, two
precision gradient evaluations followed by inherited/reset Adam counterfactuals
without advancing live training; (2) common-FP32/no-jitter per-pass held-out
evaluation in a new versioned shared engine/CLI. No model switch, epsilon change
or production quality campaign. Source freeze and CPU review precede GPU use.

precision_assessment owns optimizer-history probe/tests/protocol.
packed_runner_review owns evaluator runtime/per-pass functions/tests/contract.
packed_data_review owns publiccheckpoint/loss notes then independent probe review.
Root owns eval scheduling/aggregation/data/engine/CLI/tests/protocol and GPU jobs.

New engine explicitly invokes evaluation at completed boundaries; no global
patches. Separate pinned dev cursor never advances training cursor. Aggregate
per-pass sums and integer counts globally before division. A scheduled restored
boundary repeats once per segment, recorded separately from trained checkpoints.
Failed evaluation leaves previous retained checkpoint authoritative; no saving
of a potentially corrupted live boundary. Terminal resume may evaluate a
scheduled terminal boundary but never prepares another training graph/update.

Planned bounded GPU acceptance: tiny two-rank reference vs insertion at update2
then exact update3; tiny checkpoint2resume repeats eligibleeval once. Native
NFR T1024 insertion uses retained PR46 reference for training-state comparison,
with explicit changed evaluation/source identities and unchanged recipe/data/
startup/runtime. Native fullstate snapshots guarded, graph-owned inputs checked.
Checkpoint/GCS timing separated; no throughput claim. Need retain all evidence.

No GPU job started yet. Last verified both H10080GB idle; bootfree110GiB.
Save/push20–30minutes. Use progress here and latest runtime reports before any
launch after interruption.
