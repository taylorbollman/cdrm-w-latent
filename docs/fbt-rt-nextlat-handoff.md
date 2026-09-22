# Pretrained OLMo / RT / FBT / NextLat implementation handoff

Updated 2026-09-22. **Read this first after compaction or interruption.**

## Current decision, authorization and next action

**Priority reset, 2026-09-22: functionality and execution before quality.**
The user now wants confidence that RT, FBT and NextLat work separately and
together, with bounded numerical/gradient checks where concerns exist, reasonable
efficiency, explicit parameter/throughput/FLOP accounting, a Q/K-normalization
decision, native tiled-RT/Flash integration and genuine multi-GPU checks.
Quality wins and substantial baseline/variant training come after that review.

**F3 CUDA-graph integration is complete and assessed (2026-09-22).** Read
[assessment](reports/olmo1b-f3/assessment.md), [results](reports/olmo1b-f3/results.md),
[protocol](reports/olmo1b-f3/protocol.md) and [usage](olmo1b-f3-usage.md).
The static-layout path reuses native RT, ordinary checkpointing, fusion and
canonical selected-position CE/NextLat math. It validates fixed masks/documents/
positions outside capture and accepts new tokens and in-place weight updates.
Graphs capture forward/loss/backward; clipping/AdamW/scheduler stay outside.
**No GPU job or learning run is queued.** This closes the graph slice, not the
broader F3 native RT fused-kernel goal.

F3 durable execution record:

- Branch `feat/olmo1b-f3-combined-cuda-graphs`, base F2 merge `ffe7df3`, frozen
  runtime/protocol `b402d91`; [PR15](https://github.com/taylorbollman/cdrm-w-latent/pull/15).
  267 distinct scoped CPU tests pass. All 13 GPU reports have identical source
  inventories and exact run-local snapshots matching the frozen runtime.
- Seven correctness cases pass: RT/combined K2 B1/T32 checkpointing off/on,
  combined K3 B1/T32 on, and RT/combined B8/T512 on. Original/changed tokens,
  gradient overwrite and changed-weight losses/gradients match bitwise.
  Three eager versus three graph AdamW updates in each case give exact model,
  moments, scheduler, counters and metrics: 42 physical updates in total.
- Six paired capacity cases pass: RT and combined K2 B32/64/128 at T512 with
  ordinary checkpointing. Three eager preparation and three timed updates per
  arm add 72 physical updates. Across correctness/capacity: 114 total, with
  75 executed eagerly and 39 by graph replay; 57 belong to each designated arm.
  Warmup/capture/region-only backwards do not advance optimizer counters.
- Graph RT B128: 24,684 input tokens/s, 42.10 GiB peak allocated, 1.36x matched
  eager speedup. Combined B64: 10,409/s, 40.84 GiB, 1.33x; combined B128:
  10,801/s, 58.20 GiB, 1.18x. B64 retains 96.4% of combined B128 throughput;
  use B64 as the common development default, B128 as a measured capacity option.
- Combined B128 reserved setup peak is 77.95 GiB, current postcapture reserved
  64.85 GiB; B64 values 64.27/43.33 GiB. Do not confuse these with graph-private
  allocation. Installed PyTorch already clears unused cache before capture.
- Runtime: one H100 80GB; BF16 autocast, FP32 params/grads/Adam, TF32 off,
  deterministic Flash in ordinary layers, autocast weight cache disabled.
  F2 timings used different execution settings. RT selects only layer0; its
  native dyadic/custom-VJP math is captured, not replaced with Flash/CuTE.
  No Q/K normalization change. Large-batch finite checks are not all-gradient
  parity evidence at those shapes. All-layer RT/distributed/graph resume remain
  untested. The standard checkpoint boundary needs plan disposal/cleared grads.
- Outputs: `.runtime/olmo1b-step60000/f3-*-01/` and sibling logs; explicit list
  in `f3-final-inputs.json`. Capacity queue session54878 exited0; last run
  finished 2026-09-22T17:41:46UTC. W&B links are in the results. No weights from
  disposable updates were archived; use the retained original native checkpoint.
- Small evidence retained under
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3-graph-training/`;
  [receipt](reports/olmo1b-f3/storage-receipt.json) pins objects/hashes/generations.

Next review milestone: brief device profiles at useful batches plus the common
parameter/throughput/memory/FLOP ledger. Choose the bounded native RT fused-tile
prototype from measured bottlenecks. Continue genuine two-GPU checks when a
second GPU is available; there is only one here. Do not launch long learning
comparisons or resume completed numerical campaigns by default.

Read the new **[v4 plan](fbt-rt-nextlat-research-plan-v4.md)**. It supersedes
the O5e recommendation below to run another joint-backbone FBT learning comparison.
**F2 is complete and assessed (2026-09-22).** Read its
[assessment](reports/olmo1b-f2/assessment.md), [results](reports/olmo1b-f2/results.md)
and [usage](olmo1b-f2-usage.md). B128/T512 with ordinary-block checkpointing is
a practical common starting point: RT20.2kinputtokens/s at41.4GiB, combined
K2+NextLat10.5k/s at51.9GiB. Keep native Q/K math. Deterministic Flash SDPA
provides an exact B8/T512 native-stack graph reference. F3 subsequently extended
capture to combined canonical training as recorded above.

**F1 is complete and assessed (2026-09-22).** The user approved v4 including
brief early profiling, and authorized this bounded integration milestone.
All18 cases passed in one actual-checkpoint run: all eight feature combinations,
K3/fractional/transition/two-RT-layer cases, representative T128/T512 updates,
two exact BF16 checkpoint replays, and a short exact online cache check.
Read [assessment](reports/olmo1b-f1/assessment.md),
[results](reports/olmo1b-f1/results.md),
[capability ledger](reports/olmo1b-f1/capability-ledger.json),
[protocol](reports/olmo1b-f1/protocol.md) and [usage](olmo1b-f1-usage.md).

Operational record:

- Branch `feat/olmo1b-f1-integration`, base O5e merge `9139783`;
  frozen runtime/protocol commit `c9524ca`;
  [PR13](https://github.com/taylorbollman/cdrm-w-latent/pull/13).
- Run .runtime/olmo1b-step60000/f1-integration-01, adjacent .log;
  exec55172 completed0, 2026-09-22T15:31:17–15:41:34UTC, 615.86seconds.
  [W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8patkvpi).
- 72 retained-counter updates, 86 microbatches, 19,512 valid input tokens;
  two duplicate recovery replays give 74 physical optimizer executions.
  These are operational fixtures, not corpus/quality evidence.
- 244 distinct scoped CPU tests pass. Independent final source/objective/
  ownership/counters/recovery/cache audit passes. Runtime sources unchanged.
- Both disposable full recovery checkpoints were deleted after exact successful
  replay. Model/optimizer/scheduler/counters/fixture and subsequent CPU/CUDA
  RNG checks are retained in the report. Reuse the original retained weights.
- Small evidence retained and verified at
  gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f1-integration/20260922T154616Z/.
  [Receipt](reports/olmo1b-f1/storage-receipt.json); archive588,413bytes,
  generation1790092036330262, SHA256
  d5eded73f55ad403855657ce89b14268200a0581bd2e57d900b63c75be7c1bfa.
  The native checkpoint is referenced at its existing verified object.
- B1/T512 median full-step seconds / inputtokens-per-second / peakallocatedGiB:
  ordinary .0676 /7571 /18.40; RT1.6266 /315 /18.40;
  FBT .1043 /4910 /18.68; all-three1.6803 /305 /20.40.
  RT selects layer0, FBT K2; half the input positions carry CE targets.
- All four profiles show cuDNN fused SDPA in ordinary blocks; RT remains eager.
  Large custom-backward/forward CPU and launch durations make batch scaling
  and eager scheduling/replay the leading efficiency investigation.
- All56 retained preclip-norm observations clipped at1. Random fusion/NextLat
  produce large norms, especially K3 (initial2805.8 versus K2combined887.7);
  these are F2 scale/startup leads, not evidence that derivatives are wrong
  or that absence of Q/K normalization is the cause.

F2 operational record:

- Branch `feat/olmo1b-f2-health-capacity`; runtime/protocol commit `77ee7bc`,
  final diagnostic/report implementation `c705cf4`;
  [PR14](https://github.com/taylorbollman/cdrm-w-latent/pull/14).
  Independent final review found no material numerical or scope issues.
  Small evidence retention is recorded in the
  [storage receipt](reports/olmo1b-f2/storage-receipt.json), with original native
  weights referenced at their existing verified object.
  All outputs are under .runtime/olmo1b-step60000/ with matching sibling .log.
- f2-health-01: six B2/T32 cases; f2-health-t128-01: ordinary/combinedK2/K3
  atB2/T128. Both pass; combined observer loss/gradients are bitwise neutral.
  Sampled Q/K scales remain bounded; retain native absence of Q/K normalization.
  Large startup gradients are associated with feedback/KL. K3's large T32
  gradient is not universal: atT128 K3 norm1171.5 versus K2 1205.1.
- f2-checkpoint-01: exact actual BF16 complete-update parity, checkpoint off/on,
  including model/Adam/scheduler/counters. Ordinary non-reentrant checkpoints
  never replay selected RT forward. Execution flag is default off.
- f2-capacity-01:23cells,22measured and1OOM,363.1s,132recorded complete updates.
  RT and combinedK2, T512, physical B1/8/16/32/64/128/256; each arm stopped before
  attempting512. Three warmup plus three timed steps per measured cell.
  All measured endpoint states finite. No trained weights retained from fixtures.
  B128RT=20,176inputtokens/s,41.4GiB allocated; combined=10,490/s,51.9GiB.
  RTB256=24,196/s,65.116GiB (just above frozen65GiB cutoff); combinedB256OOM.
  W&B https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vzn186td.
- f2-graph-t32-01: historical CPU scalar-copy capture blocker, repaired by two
  equivalent diagonal-view zero operations. Its exact old source snapshot is
  retained inside that runtime directory. f2-graph-t32-02 passes all original
  changed-token/weight/gradient-overwrite checks bitwise;2.91xstack/VJP speedup.
- f2-graph-t512-b8-01: auto/cuDNN forward exact, gradient budget failed.
  f2-localize-auto-rt-01 and f2-localize-auto-ordinary-01 demonstrate comparable
  eager/eager and graph/graph variability (~.3-.5%) even without RT. Traces show
  ordinary cuDNN fused SDPA. This is not a capture-only discrepancy.
- f2-graph-flash-t512-b8-01: nondeterministic Flash also fails original budget;
  f2-graph-cudnn-det-t512-b8-01: strict deterministic cuDNN unavailable.
  f2-graph-flash-det-t512-b8-01 PASSES every original check bitwise: PyTorch
  Flash SDPA + deterministic algorithms + configured cuBLAS workspace,
  B8/T512/RTlayer0/BF16, changed tokens and updated weights included.
  Stack/VJP eager1.583s versus replay.481s (3.29x), timing allocated17.72GiB,
  reserved aftercapture28.63GiB. W&B
  https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/38m1ghdp.
- Graph probes exclude CE/FBT/NextLat/Adam/padding/cache continuation and disable
  autocast weight caching in both compared executions. They do not clear whole
  combined training, checkpointing within capture, B128 capture or all16RT layers.
  Keep the original failed reports and thresholds; do not relabel them passing.
- 211 scoped CPU tests pass. Final results, summary, plot and assessment record
  source hashes and every completed/failed diagnostic. Retention receipt is
  docs/reports/olmo1b-f2/storage-receipt.json (generated separately from archive).

Next review milestone: static-layout canonical CE/NextLat/FBT graph integration,
changed-token/weight/full-update checks, and graph+ordinary-checkpoint memory/
throughput near a practical physical batch (B128 eager reference). Preserve
objective/mask semantics; use the deterministic Flash stack reference for
precision localization instead of imposing bitwise checks on nondeterministic
cuDNN. Measure bottlenecks before a major RT Flash/CuTE rewrite. Full FLOP and
multi-GPU work remains planned; a second GPU is still unavailable. No quality
contest or long learning run is queued.

Important context for resumption:

- All eight RT/FBT/NextLat combinations already have tiny independent
  objective/gradient coverage; the gap is broader actual-runtime integration.
- FBT K counts total shared-stack passes including ordinary pass 0. RT applies
  to extra FBT passes; FBT K1 does not exercise RT. Standalone RT remains possible.
- Native selected RT blocks use eager PyTorch tiling/custom backward, not a
  Flash/CuTE RT kernel. Ordinary layers use SDPA; actual fused backend dispatch
  depends on shape/masks and must be traced. Current RT backward has quadratic
  probability intermediates despite forward input/output reconstruction.
- Native custom backward returns parameter gradients normally. The historical
  vendor warning about hidden parameter-gradient writes does not apply here.
- Preserve native absence of Q/K normalization initially; diagnose scale health
  before making a separate progressive-normalization model change.
- Preserve the current pass0 + gamma * mean(extra-pass losses) objective during
  functionality work. Changing comparison loss weighting is a later decision.
- Read-only container inventory exposed one H100 80GB. Two-GPU checks need an
  actual second GPU and can start on the existing validated backend.
- F1 actual-checkpoint execution is complete. No learning comparison is queued.
  O1–O5e reports and checkpoints remain historical evidence.

## Latest completed milestone: O5e

**O5e is complete and assessed (2026-09-22).** The ordinary additional-training
control completed512updates /4,194,304CE targets on the exact O5c mixed plan,
from the same O5b native state. It trains65native tensors with one ordinary CE,
LR1e-5/warmup50; FBT/RT/NextLat off, three fusion tensors fixed and unused.
Read [assessment](reports/olmo1b-o5e/assessment.md), [results/plots](reports/olmo1b-o5e/results.md),
[protocol](reports/olmo1b-o5e/protocol.md) and [usage](olmo1b-o5e-usage.md).
**No GPU job or further training is queued. Do not resume this completed run.**

Full512-window code / WikiText NLL:
- Shared starting ordinary:1.698216 /3.183361.
- Trained ordinary:1.696348 /2.720260; WikiText accuracy45.762%.
- O5c mixed K2:1.720900 /3.061443; O5d mixed online:1.723307 /3.140935.

Ordinary beats both fusion paths in both domains, with paired document intervals
excluding zero. Code versus its own start is essentially maintained (NLL delta
-0.001869, interval[-0.004935,+0.001339]); WikiText perplexity is28.9% below K2
and34.3% below online. The frozen-ordinary advantage in O5c/O5d does not establish
an advantage over ordinary adaptation. Qualification:1.177B trainable native
parameters versus8.39M fusion, different LR/compute and initial forward functions;
this is equal data/exposure, not a pure architecture ablation or general FBT verdict.
Historical recommendation, now deferred by the v4 priority reset:
shared-source/data full-backbone+fusion FBT training
against this reusable ordinary control, with finite-pass CE weighting explicitly
specified, before a quality comparison of RT/NextLat interactions.
**That next learning comparison is not launched or queued.**

Operational record:
- Branch `feat/olmo1b-o5e-ordinary-control`, base O5d merge
  `1364373cb38714da13bdc4e3c88f05c0ac60dcb9`; [PR12](https://github.com/taylorbollman/cdrm-w-latent/pull/12).
  Runtime/protocol commit `3d3f865`; retention implementation `07774f7`.
- Preflight `.runtime/olmo1b-step60000/o5e-preflight-01/` passed, exec29100
  completed0, W&B `62kr0zjb`. Config SHA
  `497f2dc70fcd0fefbf907a51306e7c8bb76208c8e44c928cbe289d9b214221cd`.
  Actual native updates, fusion freeze and exact restoration/source scores passed.
- Run `.runtime/olmo1b-step60000/o5e-pilot-01/ordinary/`, adjacent
  `o5e-pilot-01.log`, exec72097 completed0;2026-09-22T13:27:22–13:51:09UTC,
  report wall23.7827min. W&B `ri76qycw`, project `taylorbollman/pretrained-fbt-rt-nextlat`.
- Exact4,208,250inputtokens /13,946segments /1,056microbatches, no cycling,
  no auxiliary counts/resumptions. All512updates finite and clipped atnorm1;
  medianpreclip2.8524,max29.8041at84(isolated), peak34.7593GiB.
  Medianstep0.3681s,totalstep191.22s; checkpoint retention dominates elapsed time.
-94distinct scoped CPUtests pass. Independent source/exposure/objective/state/
  stability/paired-report/plot audits pass. All42frozen sourcefiles unchanged;
  all65native tensors changed, all3fusion tensors fixed. No broad precision
  campaign or full-GPU optimizer replay was added.
- GCS prefix
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5e-ordinary-control/20260922T132415Z/`.
  All128/256/384/512complete checkpoints retained; final local file remains.
  Final `ordinary/update-000512.pt`,14,154,906,281bytes, SHA
  `62b288c1e0e7e725ca9b56f65b52d018080b97eaaf28af3215c1613dba3eba45`,
  generation `1790085054588072`. Initial/final evidence live under corresponding
  prefix subdirectories; receipts in the O5e report directory. Parent data and
  checkpoint objects reused without duplicate uploads.

Historical O5d statements describing this control as unlaunched are superseded
by the completed record above. Further training remains a separate decision.


**O5d is complete and assessed (2026-09-22).** The user authorized the fixed-weight
finite K2/K3/K4 versus exact sequential feedback diagnostic after O5c. All16 cases
passed in30.4minutes, one attempt, no training or checkpoint mutation. Read
[assessment](reports/olmo1b-o5d/assessment.md), [results](reports/olmo1b-o5d/results.md),
[protocol](reports/olmo1b-o5d/protocol.md) and [usage](olmo1b-o5d-usage.md).
No GPU job or further learning is queued. The completed evaluation must not resume.

Main full512-window/max512 results, code / WikiText NLL:
- Source K2:1.737004 /4.969719; source online:1.742469 /5.101028.
- Repaired mixed K2:1.720900 /3.061443; mixed online:1.723307 /3.140935.
- Shared frozen ordinary:1.698216 /3.183361.

The repair survives online: mixed versus source online improves WikiText by
1.960093 nats, with a remaining mixed online-minus-K2 cost0.079491 (about8.27%
PPL). Mixed online beats its own frozen ordinary WikiText NLL by0.042427 but
has worse code NLL and slightly lower token accuracy in both domains. This is
teacher-forced transfer evidence, not free-running generation or FBT efficacy
versus an equally additionally trained ordinary model. No numerical instability
was identified; preserve the current implementation/BF16 settings. Recommended
next decision: a matched ordinary additional-training control, with adaptation
capacity/LR/compute differences explicit. **That control is not launched.**

Operational record:
- Branch `feat/olmo1b-o5d-online-diagnostic`, base O5c merge
  `ec618060b51e0a7ef472903a50d4009c8dc45ced`; [PR11](https://github.com/taylorbollman/cdrm-w-latent/pull/11).
  Runtime sources/protocol commit `245e221`; reporting/retention `54daac2`.
- Run `.runtime/olmo1b-step60000/o5d-diagnostic-01/`, adjacent `.log`,
  exec19959 completed0,2026-09-22T10:50:24–11:20:48UTC.
  W&B `jhhx390b`, project `taylorbollman/pretrained-fbt-rt-nextlat`.
- Source O5b checkpoint and repaired O5c mixed checkpoint use the same SHA/
  generation pins recorded below. New mixed model-only loader also pins its
  completed report SHA `020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.
- Beta1, RT/NextLat off, BF16 mixed/FP32 parameters/nativeSDPA/B8/TF32off,
  no graphs/compile. First32 max64 plus first512 max512 development windows.
  Full selection422code docs/127850targets,59Wiki docs/246910targets.
  No training/test scoring, optimizer creation, state updates or new model copy.
- All68 state hashes unchanged perendpoint;66native/fixed-scale identical across
  endpoints. Prior fullK2 and source-short metrics reproduced exactly.109distinct
  CPUtests pass; independent data/report/state/plot review found no blocker.
- Evidence prefix `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5d-online-diagnostic/20260922T105024Z/`;
  final receipt in the report directory. Parents remain at their existing retained
  object generations; no model upload is duplicated. Local retention directory
  `.runtime/olmo1b-step60000/o5d-retention-01/`.

Historical O5c statements describing O5d as not launched are superseded by the
completed O5d record above. RT/NextLat interaction and further training remain
separate decisions.

**O5c is complete and assessed (2026-09-22).** The user authorized the paired
fusion-only code versus code/general-text experiment with periodic checkpoints.
Both arms completed 512 updates / 4,194,304 additional CE targets. Every native
parameter and fixed fusion scale stayed byte-identical; only the two fusion
matrices trained. Read [assessment](reports/olmo1b-o5c/assessment.md),
[results](reports/olmo1b-o5c/results.md), [protocol](reports/olmo1b-o5c/protocol.md)
and [usage](olmo1b-o5c-usage.md). No GPU job or further training remains queued.

Final 512-window code / WikiText NLL:
- Shared starting feedback: 1.737004 / 4.969719.
- Code-only fusion: 1.715954 / 4.623289.
- Mixed fusion: 1.720900 / 3.061443.
- Fixed ordinary pass throughout: 1.698216 / 3.183361.

Mixed minus code-only: +0.004947 code NLL, −1.561846 WikiText NLL. Mixed
repairs the measured retention loss with a small code tradeoff. Its WikiText
NLL beats its own ordinary pass by 0.121918, but code remains 0.022684 worse.
This is domain-adaptation evidence, not FBT efficacy versus an equally trained
ordinary model. WikiText is a narrow development proxy; exact online inference
at these new weights remains untested. Recommended next: unchanged-checkpoint
K2/K3/K4 versus exact online on identical bounded prefixes, then an ordinary
additional-training control. **This recommendation is not launched.**

Operational record:
- Branch `feat/olmo1b-o5c-fusion-adaptation`, base O5b merge `4a53e48a2c56f71f4b9b4cc899e6c2c7ecd4e417`;
  [PR10](https://github.com/taylorbollman/cdrm-w-latent/pull/10).
- Frozen implementation/preflight commit `9a1fd3a`; preflight `o5c-preflight-01`
  passed, W&B `b1yqi0ug`. Config SHA
  `5fafdc28167c117f5c681ff60f784085125186a833dec050837ccc0265b5aa08`.
- Authoritative data `o5c-data-02/prepared`, manifest SHA
  `f837f7f412dee17a304df5f78f4b65571f15163af440adecfc77b88cca8b1490`.
  Data01 was superseded before training to align shared code context boundaries.
- Both arms use 8192 CE targets/update; mixed splits 4096/4096. Shared code
  segments match exactly; mixed has half the code exposure. Variable rows,
  physical chunks at most16; code 4,212,498 input tokens, mixed 4,208,250.
- Fresh fusion AdamW LR1e-4, warmup50, clip1; native frozen, beta1/K2/gamma1,
  BF16 mixed/FP32 master, RT/NextLat off. No prefix mixing or hidden jitter.
- Queue `.runtime/olmo1b-step60000/o5c-pilot-01/queue.json` is **completed**;
  code W&B `obv0yofk`, mixed `28cum2gv`. Queue elapsed27.6min. No restart
  of training state; one mixed pre-training NVML query failure was resolved
  by retrying the unchanged queue in a fresh container. Cause unconfirmed;
  `startup-incident.json` preserves evidence. Exec3961 ended1; retry78761
  completed0. Do not relaunch either process or the completed queue.
- 138 distinct scoped CPU tests pass. Actual H100 nonzero step changed only
  fusion, then exact restoration; source evaluation matches O5b. All updates
  finite; code0 clipped, mixed63/512 clipped, settling after startup.
- GCS prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/`.
  Each arm retained update128/256/384/512 with verified receipts; latest local
  checkpoint remains. Initial/final evidence receipts are in the report directory.
- Final code checkpoint `code/update-000512.pt`, SHA
  `263dcfd6b0553aa2ee6be26483bbe20ad04486dcc8a76e9f3ab6b9a727270f33`,
  generation1790058561034880; mixed `mixed/update-000512.pt`, SHA
  `7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`,
  generation1790059437165208. Each full file4,807,843,871bytes.
- New O5c checkpoint metadata uses built-in strings; its wrapper narrowly
  handles legacy TorchVersion only when needed. CPU exact future-update replay
  passes; actual full GPU optimizer replay remains unclaimed.

Historical O5b proposal wording below is superseded by this completed O5c record.

The user selected **original OLMo-1B at approximately 200–300B pretraining tokens**
as the new primary model, replacing OpenELM because of uncertainty about its
layer-wise capacity scaling. We selected **step 60,000, approximately 251–252B**.
This is a model choice, not a finding that OpenELM's scaling is defective.

The user approved the [v3 plan](fbt-rt-nextlat-research-plan-v3.md) and authorized
**O1 development**. Both gates are now complete: native ordinary fidelity and
selected-layer sequential RT. See [results](reports/olmo1b-o1/results.md),
[usage](olmo1b-native-rt-usage.md) and [protocol](reports/olmo1b-o1/protocol.md).
The scoped CPU suite passes **148 tests**. The actual 1.177B checkpoint passes
ordinary source equivalence in FP32/BF16 and sequential RT output/all-parameter/
input-gradient checks in FP32 at alpha 0/0.37/1, physical B1/T16, layer 0 only.

The user reviewed O1 and authorized **O2: native RoPE tiled forward/backward**.
O2 is complete on `feat/olmo1b-tiled-rt`, based on O1 merge `a806835`.
Implementation/evidence commit: `c6a41c23568a4ef4e561b7b44a05f47fe38ab59b`,
[PR #5](https://github.com/taylorbollman/cdrm-w-latent/pull/5).
See [O2 results](reports/olmo1b-o2/results.md),
[usage](olmo1b-tiled-rt-usage.md) and [protocol/amendment](reports/olmo1b-o2/protocol.md).
The combined scoped CPU suite passes **230 tests**. Actual-checkpoint tiled
FP32 checks pass at B1/T16 alpha 0/.37/1 and B1/T128 alpha 1, all 65 parameter
and input gradients. A raw-cotangent B2/T17 strict input-coordinate failure is
retained and adjudicated with an independent FP64 oracle; see details below.
BF16 differences and eager H100 performance are measured, not training clearance.
The user reviewed O2 and authorized **O3 language-model objectives/platform**.
O3 is complete on `feat/olmo1b-nextlat-platform`, based on O2 merge
`ef3380f1beb41642150c693c33ca2558365e4f84`. Implementation/evidence commit:
`40290357aff82f0caa40f9deff79c90412758faa`,
[PR #6](https://github.com/taylorbollman/cdrm-w-latent/pull/6). See
[O3 results](reports/olmo1b-o3/results.md), [usage](olmo1b-nextlat-platform-usage.md)
and [protocol](reports/olmo1b-o3/protocol.md). The scoped CPU suite passes
**326 tests**; actual-checkpoint FP32 objective/gradient checks and exact full
optimizer recovery pass. Bounded BF16 complete-step profiles are recorded below.
The user reviewed O3 and authorized continuing to O4. One H100 is available;
two-GPU correctness requires later hardware. Native checkpoint files and research
starting weights are unchanged. Four physical optimizer updates were executed
on disposable diagnostic state (three logical updates, including one replay);
profiling executed 42 zero-LR updates. No adapted research checkpoint or learning
run was awaiting resumption at the O3 close. The user authorizes direct PR closure/merges; this
does not expand research/training scope.

**O4 is complete and reviewed** on `feat/olmo1b-o4-learning-pilot`, based on
O3 merge `29fee0dae0f551ef75a29de1d1269e3a69ec8055`; [PR #7](https://github.com/taylorbollman/cdrm-w-latent/pull/7).
Read [results](reports/olmo1b-o4/results.md), [assessment](reports/olmo1b-o4/assessment.md)
and [protocol](reports/olmo1b-o4/protocol.md). All four arms completed 2,634
updates / 20,855,799 valid input tokens (20,771,511 CE targets), original
checkpoint, batch32, T512, LR1e-5, BF16 mixed, layer 0 RT only. Alpha warmup
ends100, ramp ends1367, final2634. NextLat coefficients fixed1/1; FBT off.

Final 512-window code/retention NLL:
- Original ordinary: 1.787902 / 3.040993.
- Ordinary: 1.699360 / 3.173815.
- Ordinary+NextLat: 1.792776 / 3.390668.
- RT: 1.706351 / 3.248923.
- RT+NextLat: 1.799070 / 3.485577.

All losses/gradients finite, all updates clipped at1; no restarts. NextLat
hurts code before RT turns on (update50 alpha0); initial KL~11.49 versus
CE~1.738 suggests auxiliary adaptation pressure, not an identified numerical
bug. RT mostly recovers ordinary code quality but has worse retention.
Single-seed, short recovery evidence; no architecture efficacy conclusion.

Queue `.runtime/olmo1b-step60000/o4-pilot-01/queue.json` and `finish-status.json`
are **completed**, finished 2026-09-22T03:12Z. No O4 job remains to resume.
W&B IDs: ordinary `4rhi7s7i`, ordinary-nextlat `tsgun4ax`, RT `uvfoj6id`,
RT-nextlat `4cpi8hz0`; project `taylorbollman/pretrained-fbt-rt-nextlat`.
All final `update-002634.pt` optimizer checkpoints retained at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z/<arm>/`.
See result report for hashes/generations. Initial prepared-data/source archive
and final evidence archive have verified receipts in the report directory.
Do not trust the historical nohup PID file or relaunch the completed queue.

Prepared data remains `.runtime/olmo1b-step60000/o4-data-01/prepared/`;
25,000,031 unique train CE targets, 101,591 windows, one document/window,
stride511/T512, EOS only at true ends. Official tests prepared but untouched.
Preflight/config `.runtime/olmo1b-step60000/o4-preflight-01/`;
configuration SHA256 `6b7e9168f2aa1afc495d28cfdd84006e5fa3409f66fcc70c17c0fa83c8ff2e20`.
All arms used identical frozen sources; a documented pre-training amendment
only added failed-upload resume retry. Never silently change this lineage.
466 distinct passing implementation tests; source inventory and retention
receipts, not current mutable code, define the completed experiment.
For GCS use `env -u GOOGLE_APPLICATION_CREDENTIALS` in container.

The user asked to assess and continue on 2026-09-22. O4 was assessed and PR7
merged as `9bdeddd55514f95e7799bb49b1c37dfba1fbeaa0`. **O5a bounded FBT
reference/correctness is now implemented and passes** on
`feat/olmo1b-fbt-reference`, based on that merge. Implementation/evidence commit
`4a473bfc296f0c0d6b65a8a6d5bb434ab7002c04`,
[PR #8](https://github.com/taylorbollman/cdrm-w-latent/pull/8). See
[O5a results](reports/olmo1b-o5a/results.md), [usage](olmo1b-fbt-usage.md), and
[protocol](reports/olmo1b-o5a/protocol.md). At the O5a close no GPU job remained running and no adapted FBT weights
existed. O5b is now authorized below; do not treat that historical status as live.

New modules `cdrm/pretrained/olmo_fbt.py` and `fbt_training.py`; no existing
backbone/training math modified. Pinned author-reproduction source snapshot
`_fbt_reference/` at `7037c60924870aca6e30fac95212b0c7caee052d` is static evidence,
not executed. Core API:
- `OLMoFBT(loaded_rt_capable_backbone, FBTConfig(norm_eps=1e-5,seed=20260922))`;
  two new D-by-D matrices uniform ±sqrt(3/D), independent RNG; native embedding
  RMS output scale fixed buffer 0.03707655891776085, explicit FP32 RMS reductions.
- `FBTMode(enabled=True,num_passes=2,beta=1,rt_mode=RTMode(()))`. Pass0 ordinary,
  fresh extra-pass KVs, attached previous-pass post-finalnorm feedback at t-1.
  K1 ordinary; disabled FBT runs standalone RT; beta0 does not disable RT writes.
  **`RTMode(())` is ordinary; `RTMode()` historically selects layer 0.**
- `forward_online(...,FBTOnlineMode(beta,rt_mode),past_key_values=FBTOnlineCache)`
  consumes the fresh previous-token state; no K. Requires RT-capable backbone,
  uses an empty selected-layer set for FBT-only. Full-prefix mask/document IDs, current
  positions. Cache rejects changed modes, weights/fusion/buffers, conversion,
  autocast/gradient contexts; strong storage refs catch child dtype roundtrips.
  No ordinary-prefix injection/switch, packing, jitter or sampled prefix mixin.
- `FBTNextLatLM(core,NextLatConfig,enabled,gamma=1)` wraps per-pass objectives.
  One shared predictor; pass0 + gamma*mean(extra losses). Counts remain data
  positions once. Aggregate CE is not final-pass NLL. Save gamma/config/modes
  explicitly with existing training checkpoint API.

549 distinct CPU tests pass: 526 model/platform plus 23 retention, including
60 new FBT math/platform checks (28 core + 9 adversarial + 23 objectives/platform). Tiny full optimizer replay
is bitwise exact across third update, optimizer/scheduler/counters/RNG/cursor.
Actual H100 B1/T8 FP32 validates finiteK1/K2/K3, beta0/.37/1, selectedlayer 0
alpha.37/1, all active 65/67/71 parameter gradients, exact online prefix oracle and
chunk continuation, NextLat off/on objectives. Largest FP32 tensorgradient
relativeL2 is1.50e-5. K9 finite convergence matches exact online to~1e-6.
One BF16 mixed combined check is finite with1.51% global gradient difference vsFP32,
2.23% worst tensor L2: descriptive, not long-context, batch or training clearance.
Every model/buffer byte remains unchanged. Actual full-model input-gradient checks
are not added; tiny raw-cotangent input/cache/cross-pass coverage is retained.

Actual report `.runtime/olmo1b-step60000/o5a-validation-01/report.json` and log
`.runtime/olmo1b-step60000/o5a-validation-01.log`; selected full report copied to
`docs/reports/olmo1b-o5a/validation-summary.json`. W&B
[ii69nrrg](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ii69nrrg).
Run took 57.54 s once fixture/model validation started, peak 20.89 GiB for paired
models/gradients; that is correctness scratch, not training capacity evidence.
Verified [retention receipt](reports/olmo1b-o5a/storage-receipt.json) identifies
the source/report archive and read-only reused O1 checkpoint. Prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-fbt-reference/20260922T033600Z/`.
Local retention `.runtime/olmo1b-step60000/o5a-retention-01/`; schema
`olmo-fbt-reference-v1`. No research checkpoint was generated.

**O5b is complete and assessed.** Both matched arms finished2,634 updates /
20,855,799 valid input tokens each, with no restart/health-gate stop. User asked
to proceed; we supervised completion and added one bounded **evaluation-only**
endpoint diagnostic to understand the observed feedback-specific retention loss.
No longer training budget or new learning arm was added. Read
[results](reports/olmo1b-o5b/results.md), [assessment](reports/olmo1b-o5b/assessment.md),
[diagnostic](reports/olmo1b-o5b/diagnostic/results.md),
[protocol](reports/olmo1b-o5b/protocol.md) and [usage](olmo1b-o5b-usage.md).

Branch `feat/olmo1b-o5b-fbt-pilot`, base O5a merge
`4a3e19630b8c5b0ad87eb228d5d471f5c382da6e`; [PR9](https://github.com/taylorbollman/cdrm-w-latent/pull/9).
Learning source/evidence commit `4751d508b26887b8873cf43b06acc2b47994067c`;
follow-up helpers/protocol commit `7b6b199`.

Frozen recipe:
- Original step60000 weights and O4 prepared data/order; RT and NextLat **off**.
- Ordinary K2/beta0; FBT K2/gradual beta0→1; gamma1 and two CE losses in both.
  O4's single-CE control is contextual, not substituted for the matched control.
- Effective batch32,T512; physical16 accumulated twice. FP32 parameters/AdamW,
  BF16 mixed, native SDPA, no TF32/compile/graphs/distributed.
- Native LR1e-5, fusion LR1e-4; native warmup100; first nonzero beta update102
  starts fusion LR1e-6, reaching1e-4 at201. Ramp ends1367, final2634.
- No prefix sampling or hidden jitter; deliberate source-recipe departures.
- Same84,288 windows /20,771,511 CE targets,41,711,598 stack-input positions
  per arm. Native fusion scale0.0370765589, unchanged O5a fusion semantics.

Final512-window code/retention NLL:
- Original:1.787902 /3.040993.
- Matched ordinary:1.699385 /3.177141.
- FBT pass0:1.698216 /3.183361.
- FBT feedback pass1:1.737004 /4.969719.

FBT versus ordinary costs+0.037619 code NLL (paired document95% interval
[+.033360,+.041884]) and+1.792577 retention ([+1.631192,+1.932436]). Ordinary
path is largely preserved; no feedback benefit at this budget. Still improving
at beta1, so not an asymptotic or general architectural verdict. All recorded
scalars finite; EVERY update clipped at1. Norm medians/maxima4.871/10.402
ordinary,5.043/21.144 FBT. All full checkpoint parameters/moments finite.
Do not equate numerical execution health with satisfactory retention.

Post-hoc diagnostic `.runtime/olmo1b-step60000/o5b-diagnostic-01/`, W&B
`nw0l3ol8`, **passed** with model/buffer hashes unchanged,13 fixed cases.
128-window K2 beta0/.25/.5/.75/1 codeNLL:
1.599637/1.609754/1.622601/1.620166/1.634389;
retention:3.249787/3.284040/3.367670/3.608247/5.014280.
Every tested positive beta remains worse than this checkpoint's ordinary path.
Short32-window/max64 beta1 K2/K3/K4/online retention:
5.083958/5.013408/5.033363/5.032388; code~2.200–2.201 throughout.
More refinement does not rescue quality on this short subset (only6 original
retention documents). Intermediate beta mitigates input-path damage but does not
show a benefit. The original final512 evaluation and these128/32 subsets must
remain separate; no reserved tests used or best-beta confirmation claimed.

Operational close:
- Queue `.runtime/olmo1b-step60000/o5b-pilot-01/queue.json` and `finish-status.json`
  are **completed**. Ordinary ended04:49UTC; FBT and finisher ended05:39UTC
  on2026-09-22. Endpoint diagnostic ended05:40UTC. No GPU job remains to resume.
- Historical exec sessions42624(queue),69027(finisher),2507(diagnostic) are
  completed; do not relaunch them. Reports/events remain authoritative.
- W&B: ordinary`ichekj67`, FBT`gtv1hp98`, preflight`upirj0yb`, diagnostic`nw0l3ol8`,
  under`taylorbollman/pretrained-fbt-rt-nextlat`.
- Config SHA256:`b26d4f7af7e6888bf0f8720724d2aadc3f4c1ed842dd988ebbedc89545bce626`.
  Preflight `.runtime/olmo1b-step60000/o5b-preflight-01/` passed; physical16accum2
  used49.60GiB and0.919s/full-length step versus74.53GiB/0.909s physical32.
- Scope tests:693 passing OLMo CPU tests from the implementation, plus37 new
  diagnostic/health tests. No frozen model/training/evaluation source changed.
- Verified storage prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/20260922T040000Z/`.
  Initial and final paired receipts are in the report directory. Follow-up
  interpretation/diagnostic evidence is separately retained under`review/`.
- Final ordinary checkpoint`ordinary/update-002634.pt`,SHA256
  `738ed2ef2a31a4a52cb28276be3ce4494170c40658025c614f0d3614c8323b6a`,
  generation1790052541850455.
- Final FBT checkpoint`fbt/update-002634.pt`,SHA256
  `99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`,
  generation1790055503754266. Latest local full checkpoints remain in arm dirs;
  older checkpoints were removed locally only after cloud verification.

**Recovery caveat:** actual O5b checkpoints include `TorchVersion` runtime metadata.
Weights-only loading needs the narrow scoped allowance implemented by
`scripts/olmo_o5b_diagnose.py:load_endpoint_payload`. Apply the same context
around frozen `load_training_checkpoint` for any future optimizer resume; the
old queue's bare resume command does not install it. Do not switch to a broad
unsafe pickle loader or silently mutate frozen sources. This was found and
regression-tested during actual endpoint loading; checkpoint bytes are unchanged.
No actual full-optimizer O5b replay was performed. Beta-zero warmups also show
small numerical trajectory differences despite matching data/LRs/counts; see
optimization-summary.json rather than claiming bitwise cross-run replay.

**Next review:** recommended small fusion-only code-versus-code/general-text
adaptation comparison, from this completed FBT endpoint with native backbone
frozen. Candidate~5M valid input tokens per arm; freeze data/mixing/exposure and
profile before launching. Use fresh general training documents excluding all
retention dev/test; equal total exposure means unequal code exposure, report both.
This is a proposal, **not launched**. No blind50–100M extension, new RT/NextLat
interaction sweep or multi-GPU test is queued. O4's NextLat startup shock remains
a separately staged warm-start/component-gradient question. These results do not
veto later interaction hypotheses; they identify an adaptation confound to resolve.

O1 implementation branch: `feat/olmo1b-native-rt-reference`, based on `e894fe0`
(planning PR #3). The O1 source hashes are in the selected validation reports;
source/evidence, not an unrecorded working tree, defines tested behavior.

## O3 selected results and recovery

- Code: `cdrm/pretrained/nextlat.py` and `lm_training.py`; GPU drivers
  `scripts/olmo_lm_validate.py` / `olmo_lm_profile.py`; shared literal fixtures
  and dense objective in `scripts/olmo_lm_common.py`. O1/O2 model math unchanged.
- Source fidelity: NextLat revision `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`,
  retained byte snapshot in `_nextlat_reference/`. Select its **1B LM horizon-one**
  recipe, not A5: predictor factor 1.6 (hidden6528), latent/KL coefficients 1/1,
  no auxiliary predicted-token CE. Predictor adds 82,726,912 parameters; total
  1,259,491,328 versus native 1,176,764,416. Predictor is training-only, with
  isolated initialization; inference remains `model.backbone(...)`.
- Masks: CE targets token t+1; latent targets stopped post-finalnorm state t+1;
  KL compares stopped teacher and predicted-state distributions for token t+2.
  Source/conditioning embeddings remain attached; only auxiliary readout use
  is detached. Separate target-position masks and global valid counts per
  objective across microbatches. One document per row; reject multi-document
  packing because loss masks cannot prevent attention leakage.
- Validation: `.runtime/olmo1b-step60000/lm-validation-01/report.json`, W&B
  [r0zqx75g](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/r0zqx75g).
  Full-checkpoint FP32 math B1/T16 ordinary/RT with NextLat off/on and padded
  B2/T16 RT+NextLat pass independent dense-objective/all-active-parameter checks.
  Global gradient relative L2 1.579e-6–4.811e-6; worst tensor 6.226e-6.
  BF16 RT+NextLat versus FP32 global differences: mixed attention 1.932%, FP32
  attention 1.911%; worst tensors 2.839%/2.590%. Finite descriptive diagnostics,
  not long-sequence fused-backend gradient or learning-equivalence clearance.
- Exact actual-model recovery: save after update2, rebuild, reload, repeat update3.
  Model, AdamW moments, scheduler, counters, CPU/CUDA next RNG draws and update
  metrics match bit-for-bit. LR1e-5 with two-update warmup; FP32, layer 0 RT.
  Disposable 15,114,028,075-byte checkpoint SHA256
  `8d615b49730f8c8e1292fac008596977e1fdc754e0f78366293418a05b6fba22`
  was deleted after passing; full hashes/fixtures/evidence retained. No need to
  recover this diagnostic model. CPU tests also cover nonzero predictor dropout
  and Python/NumPy/explicit data-generator RNG.
- Profile: `.runtime/olmo1b-step60000/lm-profile-01/report.json`, W&B
  [zbrcek8g](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zbrcek8g).
  BF16 default SDPA + mixed tiled attention; layer 0 RT only; FP32 parameters,
  real initialized AdamW moments, clipping, LR0. Three warmups/three timed steps.
  RT+NextLat B1/B4/B8,T512: 1.496/1.587/1.626 seconds, 342/1,290/2,519 valid
  input tokens/s, 19.61/20.37/23.99 GiB allocated peaks. B8 reserved25.79 GiB.
  All weights unchanged and moments finite. Not a maximum-batch search or a
  learning run; literal repeated text with half-length CE masks, no data loader.
- Memory: vocabulary loss chunks recompute full-vocabulary logits rather than
  retaining all such activations. Validation chunk8; profile chunk128 positions.
  O2 backward still reconstructs quadratic attention and uses per-position VJPs.
  No compile/graphs, FBT, distributed, packed-document attention or cached LM
  objective training clearance. Only layer 0 is recurrent in actual-model checks.
- CPU record: `.runtime/olmo1b-step60000/lm-cpu-suite.log`, copied to
  [test-results.txt](reports/olmo1b-o3/test-results.txt): 326 passed, 41 warnings.
  Compact machine-readable record: [summary](reports/olmo1b-o3/validation-summary.json).
- Evidence prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-nextlat-platform/20260921T213500Z/`.
  [Storage receipt](reports/olmo1b-o3/storage-receipt.json) records verified
  objects. Local archive/receipts:
  `.runtime/olmo1b-step60000/lm-retention-20260921T213500Z/`.
  Reuse O1's existing immutable checkpoint object; do not upload another copy.
- Next review: choose O4 domain/splits and masking, practical batch/exposure,
  ordinary/ordinary+NextLat/RT/RT+NextLat controls and matched alpha ramp. Inspect
  untrained predictor scale before selecting adaptation schedule; initial short
  RT fixture means CE5.095/latent0.916/KL8.349 are not model-quality measurements.

## O2 selected results and recovery

- Code: `cdrm/pretrained/olmo_tiled.py`, `OLMoTiledRTForCausalLM` and low-level
  `tiled_recurrent_layer`. Parameters/layout unchanged; O1 references preserved.
  Dyadic attention, native RoPE, fractional alpha, explicit returned parameter
  gradients, attached prefix/exported-cache gradients. Separate tiled cache
  provenance. Saved x/z enable recomputation without sequential forward replay.
- Final validation: `.runtime/olmo1b-step60000/tiled-validation-02/report.json`,
  W&B [nfys79l3](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nfys79l3).
  FP32 full-model global gradient relative L2 1.543e-6–2.319e-6; worst tensor
  3.434e-6. All original full-model acceptance budgets pass. Some stricter
  parameter-coordinate diagnostics remain flagged, as recorded in the report.
- Initial stress-test failure remains in `tiled-validation-01/`, W&B
  [q4gu72t4](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/q4gu72t4),
  and [retained report](reports/olmo1b-o2/initial-validation-01.json). Final
  raw-input adjudication retains 65/69,632 coordinate flags but finds FP32
  input-gradient norm error versus FP64 only 6.143e-7 tiled / 5.439e-7 scan.
  A documented **post-failure calibration** applies the existing joint tensor
  norm/maximum budget to raw input gradients, requiring both FP32 paths to
  agree with FP64. Do not call this an unchanged original coordinate screen.
  Original source hashes are recoverable with the retained reverse source patch.
- BF16 alpha1 versus tiled FP32: T16 global 1.489–1.660%, worst tensor
  3.499–3.723%; T128 mixed attention global 2.714%, worst 3.197%; T128 FP32
  attention global 1.675%, worst 2.219%. All finite and complete. Both attention
  policies remain available; this is bounded observation, not learning clearance.
- Profile: `.runtime/olmo1b-step60000/tiled-profile-01/report.json`, W&B
  [em18z6bm](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/em18z6bm).
  Full model (RT layer 0 only), BF16 B1/T512: 1.464 s forward/backward,
  350 tokens/s, 9.509 GiB operational peak; no optimizer. B4/T512 block-only
  1,388 tokens/s. Eager tiled B1/T128 is slower than scan (1.6x FP32/1.4x BF16).
  Entire backbone remains resident even for block-only measurements. Memory
  includes finiteness-check scratch; timings exclude it.
- Limits: quadratic attention reconstruction in backward, per-position local
  autograd, first-order only; no compiler/graphs, multi-GPU, context2048,
  all-16-layer actual-checkpoint recurrence or training-quality clearance.
- CPU record: `.runtime/olmo1b-step60000/tiled-cpu-suite.log` and
  [230-test record](reports/olmo1b-o2/test-results.txt).
- O2 evidence retention reuses the existing O1 checkpoint object without
  another model upload. [O2 storage receipt](reports/olmo1b-o2/storage-receipt.json)
  records the verified timestamp prefix, object generations and hashes.
  Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-tiled-rt/20260921T205831Z/`;
  local receipts/archive in `.runtime/olmo1b-step60000/tiled-retention-20260921T205831Z/`.
- Its subsequent O3 single-GPU milestone is now complete (above); two-GPU
  correctness remains untested. No adaptation run is implicitly authorized by
  platform work.

## O1 selected results and recovery

- Artifact root: `.runtime/olmo1b-step60000/artifacts/`;
  `artifact-manifest.json` and `checkpoint-inspection.json` record full integrity.
- Ordinary: `.runtime/olmo1b-step60000/ordinary-validation-01/report.json`, W&B
  [td5cce3w](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/td5cce3w).
  B1/T42 text backward/all 65 parameters and input gradients; T64 code forward.
  Adapter/native differences are exactly zero in checked outputs/gradients for
  FP32 math, BF16 math and BF16 default SDPA. Cache/causality checks pass;
  ordinary BF16 profiler observes cuDNN fused/Flash-style SDPA.
- RT: `.runtime/olmo1b-step60000/rt-validation-01/report.json`, W&B
  [ujs94viz](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ujs94viz).
  Actual B1/T16 bottom-layer alpha 0 versus ordinary, plus alpha 0/.37/1 versus
  independent history oracle. Largest FP32 individual-tensor relative L2 error
  is 4.880e-6; all declared joint tensor-norm/maximum and input/output checks pass.
  Two stricter elementwise coordinates flag and are retained as diagnostics;
  no acceptance tolerance was changed after results.
- BF16 RT: finite/complete-gradient smoke scope only. Alpha 1 global gradient
  difference versus same-path FP32 is 1.695% math/1.451% default; worst tensor
  is 3.751%/3.461%, `transformer.blocks.5.ff_proj.weight`. This is not tiled or
  training clearance. Ordinary mixed precision is descriptively similar on a
  different fixture, so it is not a controlled recurrence-sensitivity ablation.
- Combined CPU record: `.runtime/olmo1b-step60000/cpu-suite.log`, also retained
  as [test-results.txt](reports/olmo1b-o1/test-results.txt): 148 passed.
- Current implementations: `cdrm/pretrained/olmo.py`, `olmo_recurrent.py`,
  `olmo_artifacts.py`, isolated `olmo_reference.py` and
  `olmo_recurrent_oracle.py`; pristine source in `_olmo_reference/`.
- Retention is verified at
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/20260921T202457Z/`;
  [storage receipt](reports/olmo1b-o1/storage-receipt.json) records generations,
  sizes, server MD5 and SHA256. Local receipt/evidence:
  `.runtime/olmo1b-step60000/retention-20260921T202457Z/`.
  The native source checkpoint is retained, not a newly trained model.

## Read in order

1. [Current v3 research plan](fbt-rt-nextlat-research-plan-v3.md).
2. [Checkpoint selection audit](olmo-1b-250b-checkpoint-selection.json).
3. [RT numerical handoff](rt-numerical-handoff.md): reusable methods, not an
   instruction to reopen the historical numerical study.
4. [Archived OpenELM implementation handoff](fbt-openelm-implementation-handoff.md)
   when reusing artifact, cache, immutable-mode or gradient-validation machinery.
5. [Nanochat budget audit](fbt-nanochat-budget-audit.md) and
   [architecture/auxiliary review](fbt-architecture-and-auxiliary-loss-review.md)
   for retained evidence; their historical OpenELM choices are superseded.

Both [v2](fbt-rt-nextlat-research-plan-v2.md) and the
[initial OLMo proposal](fbt-rt-nextlat-pretrained-plan.md) are historical.
Returning to original OLMo does not reactivate the initial proposal's final
approximately 3T checkpoint or obsolete milestone details.

## Frozen checkpoint and source choices

- Primary: `allenai/OLMo-1B`, branch **`step60000-tokens252B`**, revision
  **`81b71efbce6f4dada57c94860301af4298bcd351`**.
- Native file: `model.safetensors`, 4,707,065,440 bytes; advertised SHA256
  `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
  **Complete bytes and published SHA256 verified in O1.**
- Secondary official conversion: `allenai/OLMo-1B-hf`,
  `step60000-tokens251B`, revision
  `6e6042e824831c7b42223f75cf60fe3a5d92eb79`.
  Same-step naming does not prove actual tensor parity; verify before use as a
  secondary oracle. Native artifact is authoritative.
- Validated original source: `allenai/OLMo` v0.2.4 at
  `b3741bc21f1dd504838b7dbd9878ee077ded63bd`. Pristine source execution and native
  checkpoint fidelity validated in O1. Checkpoint remote-code stubs import `hf_olmo`;
  checkpoint SHA alone does not pin model math.
- Use native tokenizer files at the primary revision. HF conversion tokenizer
  differs, including postprocessor. Do not silently substitute or copy OpenELM
  BOS/EOS defaults. Freeze tokenizer versus dataset EOS insertion explicitly.
- Approximate age is 251–252B; exact original token counter was not verified.
  Header confirms 1,176,764,416 unique parameters. At rough 20 tokens/parameter,
  23.54B is the reference budget, so this is approximately 10.7x. This heuristic
  is neither a saturation threshold nor an exact FBT reproduction match.
- This is original OLMo, **not** July OLMo-1B-0724 or OLMo 2. No automatic
  alternate model is queued.

## Native architecture and porting pitfalls

- 16 uniform layers, residual width 2048, 16 query/16 KV heads, head dimension 128.
- SwiGLU intermediate 8192; fused 16384 output splits **value/up, gate** and
  computes `silu(gate) * value`, not OpenELM's gate-first convention.
- Non-affine LayerNorm, epsilon 1e-5, including final normalization; **no Q/K
  normalization**, no biases, no ALiBi or QKV clipping.
- RoPE base 10,000, native FP32 split-half computation, context 2048; cache keys
  are unrotated. Adding Q/K norm would be a separate adaptation experiment.
- One tied input/readout parameter, all **50,304 rows** retained; tokenizer
  vocabulary 50,280, EOS 50279, configured pad 1.
- Native embedding sets **no `padding_idx`**. Do not introduce pad-row lookup
  gradient suppression. Masks/labels are explicit; preserve full logits width.
- 65 native stored tensors, FP32. HF has 113 split tensors with the same element
  count. Map native/HF weights and gradients rather than comparing raw names.
- Freeze an independent original-source reference. Our modified local `olmo`
  namespace is not an independent upstream oracle. Isolate imports.

## Research and gradient contracts

- Keep independent RT, FBT and NextLat switches. Standalone RT is one pass;
  FBT K counts complete passes and K1 is ordinary pass 0.
- Initial selected RT set is `{0}`. Top index is 15; derive from configuration.
  Use full history, not the restricted-window synthetic default.
- RT writes from `m_t=(1-alpha)*x_t+alpha*z_t`, then native input LayerNorm and
  K/V projections, with RoPE in attention coordinates. There is no Q/K norm.
  Temporary self KV comes from input; persistent write follows completed
  block output. Top-block z is before model finalnorm.
- Alpha 0 must exercise the scan and recover ordinary outputs/gradients.
  Fractional alpha returns both source branches. Modes/alpha/masks/positions
  are immutable per call, including backward and concurrent shared forwards.
- Retain cache-provenance checks for model/mode/weight versions, conversion,
  autocast/grad context and backend. Preserve attached cache training paths.
- FBT uses shifted previous-pass **post-finalnorm** states through the shared
  stack; exact online decoding uses the freshly completed previous-token state.
  Cross-pass gradients remain attached. New feedback branch scaling must not
  modify the native backbone at the ordinary endpoint.
- NextLat is auxiliary training only; no autonomous MLP rollout. Source state
  and conditioning embedding stay attached, target state/distribution stop.
  Detach only auxiliary readout use, not the shared embedding globally.
  Latent pair and KL triple masks respect documents/padding; pass loss
  reductions are explicit. Do not silently reuse synthetic-only loss code.
- Early learning comparison remains ordinary / ordinary+NextLat / RT /
  RT+NextLat with FBT off. RT-only success is not a prerequisite for RT+NextLat.
  In a separately authorized milestone, develop FBT-only independently after
  ordinary fidelity, then examine combined modes with matched controls and
  exposure; this does not expand O1's scope.
- Feature matching remains a late diagnostic. Semantic Tube, EBFT, alternative
  backbones and new auxiliary objectives are not queued initial experiments.

## Preserved implementation/evidence; do not overwrite

OpenELM import and sequential RT are complete, now historical references:

- [PR #1](https://github.com/taylorbollman/cdrm-w-latent/pull/1), merge `959c225`:
  native import, actual 1.1B ordinary source fidelity in FP32/BF16; implementation
  `b24abd8` and documentation `ad86d71`. [Results](reports/openelm-import/results.md).
- [PR #2](https://github.com/taylorbollman/cdrm-w-latent/pull/2), merge `2169520`:
  native sequential RT, implementation/evidence `6603521`, docs `7752534`/`be08149`.
  [Results](reports/openelm-rt-reference/results.md).
- Stage B: 136 scoped CPU tests; actual B1/T16, layer 0, alpha 0/.37/1, outputs and
  all 226 parameter/input gradients pass FP32. BF16 finite smoke only: global
  gradient difference 3.859% math / 4.842% default; worst tensor 25.83% / 29.87%.
  These are neither OLMo results nor BF16 training clearance.
- Local sources: `cdrm/pretrained/`; ordinary/RT usage guides and reports remain
  unchanged. Preserve pinned CoreNet reference snapshots and all source hashes.
- W&B: [Stage A z4jzbq02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/z4jzbq02),
  [Stage B uowl622p](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/uowl622p).
- Import artifacts: `.runtime/openelm-import/artifacts/`; selected evidence
  `.runtime/openelm-import/validation-final/report.json` and
  `.runtime/openelm-rt-reference/validation-final/report.json`.
- GCS import prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T182701Z/`.
- GCS RT prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-rt-reference/20260921T190151Z/`.
  Reports retain verified storage receipts. Reuse existing checkpoint objects;
  do not upload another OpenELM copy.

The completed planning revision changed documentation only; O1 subsequently
implemented and validated OLMo. OpenELM's proposed tiled milestone is superseded
by the OLMo lineage and is not still queued in parallel.

## Execution, resources and retention

Host project: `/home/taylorbollman/cdrm-w-latent`; container project:
`/workspace/cdrm-w-latent`. Launch project commands with
`bash scripts/docker_shell.sh bash -lc '<command>'`. Verify container location
and successful `nvidia-smi` **inside it** before GPU work. Never execute CUDA,
training/evaluation/profiling on the host or silently use CPU. Explicit CPU unit
tests use `CDRM_DOCKER_GPUS=none` with the same launcher.

The OLMo runtime lineage is `.runtime/olmo1b-step60000/`; cloud retention is
recorded by a verified receipt under the O1 results. Project files persist;
local SSD is disposable. Retain checkpoints,
source/config/tokenizer hashes, evidence and receipts. Graphable runs go online
to W&B `taylorbollman/pretrained-fbt-rt-nextlat`; record actual URLs.

The previous GCS attempt found stale `GOOGLE_APPLICATION_CREDENTIALS`. Valid
mounted standard ADC worked with `env -u GOOGLE_APPLICATION_CREDENTIALS`.
Do not print credentials or modify `.env` just to work around that stale path.

Re-profile OLMo: bottom-layer KV width is 8x OpenELM bottom KV width, but that is
not an 8x whole-model memory/time estimate. Different depth/MLP/MHA geometry
prevents transplanting earlier batch-size claims. FP32 is the semantic reference;
actual-runtime BF16 and tiled behavior need bounded fresh checks.

Multi-GPU work remains staged. Historical tiled autograd internally accumulates
parameter gradients; DDP/FSDP is not proven by existing launcher code. Start
with explicit post-backward gradient reduction when hardware is available,
then consider optimizer-state sharding. Do not let any old training launcher
reset pretrained weights after wrapping.
