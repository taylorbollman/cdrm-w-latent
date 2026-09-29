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

16:13UTC: fixed-state GPU probe COMPLETE, all11integritychecks true, no training
updates. Runtime~164s; W&B yezbu1wv synced; GCS receipt exists at
.runtime/olmo-optimizer-history/receipts/probe-01.json. Global rawgradient error
1.369%, actualAdam delta error inherited0.969% versusreset1.942%; complete
interpretation forthcoming in optimizer-history/results.md. Probe/evaluator
core committed/pushed3bbef58. Their sources remain frozen.

Dev indexes built at .runtime/olmo-campaign-evaluation/index-dev-t16-01/index
and index-dev-t1024-01/index. New rootengine/controller/CLI are still prefreeze;
CPUintegrationtests+independentaudit inprogress. W&Bsame-step duplicate avoided
by merging fresh-evaluation and trainingmetrics into one log; eligible resumed
boundary publishes separately before nextupdate. No evaluationGPUjob yet.
Bootfree~110GiB before newstates. Save/push20–30minutes; use latest runtime
reports before any launch after interruption.

16:20UTC: evaluation source frozen ecac9b9 (164files), corrected declarations-02
pass realCLI CPUauthoritypreflight. Declarations-01 had unsupportedretainerprefix
and was never GPU-launched; retained assupersededpreflight. Tinyreference and
liveinsertion complete (~19s each); exactnextupdate/finalboundary accepted by
independentaudit1,787checks. Tinyupdate2stop/cloudrestore/resumecomplete; repeat
scheduledrestored evaluation succeeds. Terminal-boundaryresume from2withstop2
currentlyrunning; thennativeNFRinsertion againstretainedPR46reference. Rootowns
GPU schedule. All priorGPUstagesclosedandretained; sourceaudit helperfrozenafter
71CPUtests. TotalcurrentCPUcoverage196distincttests. NativeGPUstillpending.

16:25UTC: PR47 draft opened. Nativeinsertion active(twoH100s), initializedfrom
samefusion128freshAdamstartup; initialcheckpoint retainedinGCS. Native authority
setup matches193comparisonchecks; graphpreparation underway. Tinycloudresume
passes1,558checks; evaluation-only resumedupdate2withstop2 passes1,362checks,
zeroupdates/nocapture. Auditor-only absentemptyupdatesfield bug fixed withone
regression; failed audit-terminal-boundary-01 preserved, corrected02passes.
All164executionsources unchanged. FinalCPU197distincttests (72auditor scope),
cpu-02 retains finalscope; cpu-01 immutableearlier196-testrecord. Audit/runtime
evidence retained. Latestcommit e45dcc8; no launchotherGPUjobuntilnativecloses.
