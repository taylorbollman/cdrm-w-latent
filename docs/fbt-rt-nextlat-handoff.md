# Pretrained OLMo / RT / FBT / NextLat implementation handoff

Updated 2026-09-23. **Read this first after compaction or interruption.**

## Current decision, authorization and next action

**Priority reset, 2026-09-22: functionality and execution before quality.**
The user now wants confidence that RT, FBT and NextLat work separately and
together, with bounded numerical/gradient checks where concerns exist, reasonable
efficiency, explicit parameter/throughput/FLOP accounting, a Q/K-normalization
decision, native tiled-RT/Flash integration and genuine multi-GPU checks.
Quality wins and substantial baseline/variant training come after that review.

**Active milestone, 2026-09-23:** the user approved the plan to optimize
our current RT before comparing with an author-derived RoPE backend. Read
[RT efficiency and comparison plan](olmo-rt-efficiency-and-author-comparison-plan.md).
First review point: separately measure immutable RoPE-table reuse and K/V-only
permanent writes, preserving native parameters, math and graph contracts. Then
propose a restricted author-derived native-RoPE reference/tiled implementation
and matched large-batch block/stack comparison. This would precede the older
graph recovery/accumulation queue. **Stage A is complete on
`feat/olmo-rt-native-efficiency`. The user subsequently authorized proceeding
directly to Stages B/C after Stage A is finished and retained, lifting the
original review stop. Do not wait for another approval to start the comparison.**
Prospective protocol: `docs/reports/olmo-rt-efficiency/protocol.md`. Existing numerical
qualifications remain open; performance parity with the authors is unestablished.

Stage A evidence: [results](reports/olmo-rt-efficiency/results.md),
[summary](reports/olmo-rt-efficiency/summary.json),
[profile audit](reports/olmo-rt-efficiency/profile-audit.md),
[usage](reports/olmo-rt-efficiency/usage.md). Runtime/protocol `3fd27e0`, helpers
`072155e`; 406 runtime/accounting +14 retention CPU tests pass. All21 GPU reports
pass41 gates with158 actual updates;882 source pairs match the freeze. RoPE reuse
is bitwise exact; K/V-only RT/combined global gradient L2 is0.00302582/0.00654273,
with exact same-candidate graph/full-Adam parity. Native B64/T512 full-CE,
two runs per arm in reverse order: ordinary36.66k→36.77k, RT21.49k→22.43k
(+4.36%), combined10.95k→11.20k (+2.26%). Peak setup allocated/reserved for
both: RT32.08/47.36GiB; combined38.92/60.31GiB. No added parameters or defaults.
Use explicit `reuse_rope=True, kv_only_writes=True` for the improved native arm,
CE2048/KL128, cast reuse/Triton/recompute as before. Existing qualifications stay.
GPU queue ended idle; no learning run. Raw evidence is under
`.runtime/olmo-rt-efficiency/`, W&B group`olmo-rt-efficiency`; see its storage receipt.
Reports`429857a`; [PR23](https://github.com/taylorbollman/cdrm-w-latent/pull/23).
All977 evidence members and the native checkpoint reference are verified at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-rt-efficiency/20260923T185855Z/`.

Stage B/C isolated comparison is complete at runtime6eea309. Initialf483646
had a retained zero-update checkpoint-key mapping failure; all25 subsequent GPU
reports pass, totaling192 physical updates and1196 frozen source pairs. There
are122 runtime/accounting and20 evidence-helper CPU tests. Author FP32 tiled
raw-gradient L2 versus native scan is5.70e-7; BF16 cross-backend B8/T512 is0.004133.
All own-backend graph/full-Adam checks are exact. See
[results](reports/olmo-rt-author-comparison/results.md), its summary and audit.

AtB32 author is6.8–7.3% slower for1/2/6blocks. AtB128 author is7.5/8.5/8.6% faster;
one-block reverse repeats confirm the crossover. One-block author/native setup
peak allocated12.24/22.79GiB, reserved28.67/44.75GiB. These are isolated Gaussian
block/MSE rates, not LM throughput. B128 is a comfortable shared tested point,
not maximum capacity. Native defaults remain unchanged. PR24 contains this stage.

The conditional full-model integration is complete on
`feat/olmo-rt-author-integration`, PR25. Runtime2c38d649; capacity runs use
524fa89 with identical integration sources. Read
[results](reports/olmo-rt-author-integration/results.md), its summary, protocol,
usage and localization notes. The opt-in author route preserves native default,
checkpoint keys and ownership; padding, caches and fractional recurrence are
unsupported. CPU suites include236 integration/regression,47 evidence-helper
and25 localization tests, with overlapping harness reruns recorded separately.

**Numerical qualification:** actual16-layer RT0/15 B8/T512 full-CE BF16
cross-backend gradient L2 is0.312458 for RT-only and0.162606 for K2+NextLat.
Both fail unchanged screens. Each passes all five own operational checks,
including Flash dispatch, exact eager/graph gradients and complete Adam parity.
Four B64 capacity runs also pass operational checks. There are44 actual optimizer
updates and282 frozen source/report pairs across these six reports. Numerical
failure is not relabeled by operational or timing success.

**Performance:** at B64/T512, native/author rates are22430.8/22425.7 input tokens/s
for RT-only and11195.8/11186.5 for combined. Treat both pairs as tied in this
bounded five-update measurement; no small speed advantage is claimed, so reverse
repeats were not added. Author/native setup allocated peaks are26.78/32.08GiB
for RT and36.53/38.92GiB combined. Combined reserved peaks are66.39/65.64GiB,
so there is no demonstrated combined reservation/capacity advantage. This does
not invalidate the isolated B128 advantage; geometry and workload differ.

**Localization:** `localize-block0-r2` at524fa89 uses fixed actual native input
and incoming cotangent. Native/author full-FP32 local gradients agree at4.65e-7;
BF16 cross-backend error is0.003327. Both mixed paths differ by about0.5% from
local FP32. Separating the reconstructed self diagonal does not improve aggregate
agreement; compiled legacy token0 reconstruction is already exact. No production
arithmetic change follows from that probe.

`localize-block0-r3` at987bc46 captures both full-model incoming gradients. They
differ by0.786611 relative L2; each local backend exactly reproduces its own
full-model block0 raw parameter gradients when given its own cotangent. The
fresh capture's block0 difference is0.443495 (the original verification was
0.437355); shared-cotangent error stays0.003327. This supports a changed incoming
trajectory as the main source of that block's large discrepancy, not a gross
local VJP defect. It does not identify the downstream sensitive layer or choose
which full-model mixed trajectory is closer to FP32. No broad precision fix is
established. All three diagnostic attempts are retained; the first failed before
forward on a fixed Flash-context setup guard. A separate parse-time cherry-pick
launch error has its log retained. Diagnostics performed zero optimizer updates.

Recommendation: keep optimized native as the supported path and keep author
as an explicitly experimental option. No full-model speed benefit justifies a
replacement, and native supports the broader recurrence/cache contract. Before
adopting author mixed precision, separately assess full-stack FP32/precision
sensitivity. The older functional roadmap (graph recovery/accumulation, padded
online execution, real multi-GPU) remains the next broader work; no quality run,
Q/K-normalization change or further GPU job is queued. Existing F4 qualifications
remain open. Finalize verified GCS retention and PR25, then pause for review.


**CE integration and original 16-layer baseline complete (2026-09-23).** Read
[results](reports/olmo-ce-integration/results.md),
[usage](reports/olmo-ce-integration/usage.md), and
[Dao CE audit](reports/olmo-ce-integration/dao-ce-audit.md).
Runtime/protocol `0d39a22`, base PR21 merge `4528b37`. Six GPU reports pass all
18 gates, with 36 physical updates and 152 scoped CPU tests (30 new). All 246
run/source pairs match the frozen revision and snapshots. The selected raw
reports/W&B links are in [summary](reports/olmo-ce-integration/summary.json).
Reports/retainer commit `49c8c54`; review:
[PR22](https://github.com/taylorbollman/cdrm-w-latent/pull/22).

- New opt-in `NextLatConfig.ce_chunk_size=2048` groups CE selected positions
  independently from `vocab_chunk_size=128` for KL. All vocabulary rows remain.
  `None` and `to_dict()` preserve old defaults and exact config dictionaries;
  explicit overrides serialize. Rebuild static layouts/graphs after changes.
  Historical exact resumes retain saved settings. Evaluation chunking is separate.
- Native step60000, 16 layers, B64/T512, BF16 mixed, ordinary checkpointing,
  deterministic PyTorch Flash and CUDA graphs: matched half-CE control 31,114
  versus candidate 39,191 input tokens/s (+25.96%); full CE 36,633/s. Allocated
  peak 26.74 GiB and setup reserved peak 37.17 GiB in all arms. Each has three
  preparation plus three timed updates; no learning or sustained-rate claim.
- Native B8/T512 ordinary/NextLat/combined CE128-versus2048 global gradient L2
  is 0.4079%/0.1828%/0.6552%; worst tensor L2/max ratio 1.1543%/2.5641%.
  Maximum CE relative loss difference 9.65e-8; auxiliary losses are bitwise.
  Same-candidate graph loss/gradients and three-update Adam/model/moments/
  schedule/counters are exact in all three. Combined uses RT layers0/15, K2,
  unchanged optimized RT/recompute and KL128. All eight switches tested on CPU.
- Dao optimized CE imports without new dependencies, but receives materialized
  logits rather than fusing the readout. Prior CE2048 profile puts named CE
  kernels at 4.09% of device time. Audit only: no Dao GPU test or adoption.
  Consider a bounded loss/checkpoint/graph probe later; RoPE/SwiGLU stay separate.
- The [storage receipt](reports/olmo-ce-integration/storage-receipt.json) records
  verified retained sources, reports, logs and checkpoint reference. Disposable
  few-update weights are omitted. GPU idle, no training or further run queued.

Use explicit CE2048/KL128 for new bounded native development checks; production
defaults remain unchanged. Remeasure matched CE configurations before comparing
new ordinary throughput to historical RT/FBT/NextLat rates. The broader next
milestone is graph recovery/save-resume and accumulation, padding/online
readiness, plus the existing bounded precision follow-up before learning.
These CE results do not clear F4's RT+FBT coordinate miss or broader BF16/FP32
sensitivity. Q/K stays native. Genuine multi-GPU still needs a second GPU.

**Ordinary throughput diagnostic complete (2026-09-23).** The user questioned
F4's31.1k ordinary throughput against the paper's153k and requested six layers
with physicalB512. Read [results](reports/olmo-ordinary-throughput/results.md),
[paper comparison](reports/olmo-ordinary-throughput/paper-comparison.md) and
[profile audit](reports/olmo-ordinary-throughput/profile-audit.md).
Review: [PR21](https://github.com/taylorbollman/cdrm-w-latent/pull/21).
Runtime71fbccd; final reports6b61bf6, retentioncd98afd;
seven runs pass,42 physical updates,9 scoped CPU tests. Native random six-layer
SwiGLU8192-per-branch model has505,675,776 active parameters. No RT/FBT/NextLat,
kernel, model math, production default, or existing numerical gate changed.
At B64/H16/T512/halfCE, changing CE position chunk128→2048 raises57.38k→92.81k
inputtokens/s; at B512/H32,43.86k→93.10k. Copy/add/fill savings dominate the
profile improvement; ordinary Flash time stays unchanged. FullCE B512/H32
chunk2048 reaches73.52k, setupreserved76.78GiB (tight). FullCE B32/H32 gives
75.00k with ordinary checkpointing and88.86k without; forward/loss/backward
alone is82.36k/99.25k. The paper uses GELU, smaller vocabulary, ALiBi, compilation,
no ordinary checkpointing, and global512 from physical32 microbatches; its
ordinary recipe does not establish physical512. We have not reproduced153k.
All raw reports/sources/traces are retained under
`olmo-ordinary-throughput/20260923T172332Z/`; see the storage receipt in the
report directory. The1,556,831-byte archive SHA256 is
`e23637d8768550b571a7a0ca4f766e3e4ff847c841ff6908b995a2e2b90f3f45`.
Server metadata and downloaded bytes were checked. GPU idle; no further experiment queued. This user-requested
ordinary investigation motivated the now-completed CE integration above.
Do not silently change all feature defaults or clear the F4 precision qualification.

**F4 training resource cards are complete, with a retained numerical qualification
(2026-09-23).** Read the [assessment](reports/olmo1b-f4/assessment.md),
[results](reports/olmo1b-f4/results.md), [roundoff diagnosis](reports/olmo1b-f4/roundoff-assessment.md),
[operator audit](reports/olmo1b-f4/operator-audit.md) and [usage](olmo1b-f4-usage.md).
No model/kernel, native checkpoint, Q/K, RoPE or loss-math change. RT uses
`(0,15)` as an execution reference; main architecture placement remains open.

F4 durable execution record:

- Review: [PR20](https://github.com/taylorbollman/cdrm-w-latent/pull/20).
  Final reports `7be359c`; verified retention `a9222ab`.
- Branch `feat/olmo1b-f4-resource-cards`, base `deecfe6`. Main runtime/protocol
  `397885b`; diagnostic `c8f2311` then harness-only correction `949731b`;
  reporting/retention `e4dda74` plus final presentation. Frozen main sources
  did not change during runs. Explicit selection in
  `docs/reports/olmo1b-f4/final-inputs.json`; original plan and continuation
  decision are retained. Queues77851/12166 and diagnostic86130 are finished.
- 23 main GPU reports:22 pass, one numerical failure;46/47 named gates pass.
  Six successful new correctness runs,16 finite B64/B96 capacity cards;
  historical F3e RT/all-three T512 evidence is revalidated but not recounted.
  Successful main reports contain132 physical updates (66 eager+66 graph).
  Separate diagnosis contains six more (3+3). 235 scoped CPU tests pass.
- Common B64/T512 input tokens/s: ordinary31,113; RT19,437;
  NextLat24,026; RT+NextLat16,444; FBT15,798; RT+FBT12,136;
  FBT+NextLat12,177; all-three9,892. Allocated peaks26.74–39.09GiB;
  setup reserved37.17–60.46GiB. BF16 mixed, ordinary Flash/checkpointing,
  CUDA graphs and Triton RT/recompute. Three-update directional medians.
- All B96 attempts succeed. RT21,618tokens/s (+11.2%) and RT+NextLat17,833
  (+8.5%) have useful gains with reserved peaks62.45/64.97GiB. No-RT modes
  gain little or nothing. RT+FBT/all-three reserve78.39/78.22GiB for only
  +6.2%/+4.2%; FBT+NextLat reserves73.89GiB with no speed gain. B64 remains
  the conservative common default; do not call these near-capacity B96 cases
  comfortable. No maximum search, OOM or next training job.
- **Unresolved numerical qualification:** RT+FBT without NextLat at B8/T512
  exceeds the unchanged6.25% coordinate screen in one layer11 MLP tensor
  (6.3492%); globalL2 1.0206%, worst tensorL2 1.3805% pass. Forward/dispatch
  exact. The failed gate stays in all tables; resource coverage is not full
  numerical clearance. No thresholds were changed.
- Fixed-state BF16 repeats and candidate graph/full-Adam comparisons are
  bitwise. Eager historical backward removes the coordinate crossing.
  Full FP32 materialized/recompute globalL2 3.04e-6 supports reconstruction
  semantics; FP32 uses eager/math kernels, not BF16 Triton.
  Both BF16 control and candidate differ from fullFP32 by about18% gradientL2
  at this initialization (cosine .983–.984; CE difference ~.125%). Preserve
  this broader sensitivity; it does not establish a learning problem or its
  absence. No basis here to automatically add Q/K normalization.
- Roundoff01 failed before backward because the probe prepared its static
  layout outside the fixed backend context; corrected only that harness line.
  It has zero optimizer updates and exact historical snapshots. Roundoff02
  is a completed diagnostic, not a cleared precision gate. Both are separately
  retained from the main23 reports.
- Operator traces reconcile dense arithmetic exactly and show actual ordinary
  Flash plus RT Triton. Matrix FLOPs, not hardware utilization: ordinary
  B64/T512281.79–304.87TFLOPs/update; combined657.84–701.12. Registered,
  active, optimizer and deployable counts are explicit. RT adds no weights;
  NextLat82,726,912 is training-only; fusion8,388,608 is shared/deployable.
- All sources/protocols/raw reports, W&B runs, failed attempts and compressed
  traces are verified in GCS under `olmo1b-f4-features/20260923T160539Z/`;
  the final [receipt](reports/olmo1b-f4/storage-receipt.json) is authoritative.
  Pinned native weights are reused, disposable few-update weights omitted.
  Evidence: 9,119,232 bytes, SHA256
  `9addc555df2b0f64ee0eacb6df828c7d68657acd402eb79f6128baf255aa51dc`.
  Manifest: 793,161 bytes, SHA256
  `3e523e11b47b95187964265427d40da2b87aa9a79f9e5ad8784f14a8f816b539`.
  The archive contains 1,206 members and 1,002 checked run/source pairs;
  remote native-checkpoint identity was verified, without reuploading weights.
  Final in-container GPU check:0MiB used, no compute processes.

**Next review milestone:** graph recovery/save-resume and accumulation,
padding, finite-prefill/exact-online resource cards; keep the numerical
qualification visible. Before substantive learning, consider a bounded
transition-state/clipped-update comparison for the broader BF16 sensitivity.
Do not automatically launch a long quality run or add Q/K normalization.
True multi-GPU remains blocked on a second GPU; one H100 is exposed. The
existing AccumulateGrad stream warning stays visible despite exact single-GPU
parity. No inference-throughput, all-eight long-context or multi-GPU clearance.

**F3e multiple-selected-layer integration is complete (2026-09-23).** Read the
[assessment](reports/olmo1b-f3e/assessment.md), [results](reports/olmo1b-f3e/results.md),
[protocol](reports/olmo1b-f3e/protocol.md) and [usage](olmo1b-f3e-usage.md).
The user expects more than one RT layer, but has not chosen all-layer RT or a
placement. Adjacent `(0,1)`, separated `(0,15)` and four `(0,5,10,15)` are the
primary execution checks; all16 is separately labeled stress. No quality run is
queued. Main architecture selection remains open.

F3e durable execution record:

- Branch `feat/olmo1b-f3e-multilayer-rt`, base PR18 merge `9af736e`;
  runtime/protocol `543d243`, reporting/retention `de3e6b9`. No runtime changes
  during GPU work. Sixteen reports pass48/48 gates;96 physical updates comprise
  48 eager+48 graph. All474 scoped CPU tests pass. No failed attempts.
  Review: [PR19](https://github.com/taylorbollman/cdrm-w-latent/pull/19).
- No model/kernel/numerical-policy change. Reuse F3d recompute, cast reuse,
  Triton historical tiles, ordinary deterministic Flash/checkpointing and
  canonical BF16 CUDA graphs. Native checkpoint/RoPE/QK/parameters stay fixed.
  Added reusable per-layer validation and recompute-aware resource accounting.
- Eight correctness runs: combined spread2 B1/T32; combined adjacent2/spread2
  and RT-only spread2 B8/T512; combined spread4 B4/T512; combined K3 spread2
  B1/T32; combined spread2 B1/T2048; separate all16 B1/T32 stress. All initial
  losses bitwise; all same-candidate graph checks and full Adam/state exact.
  Global gradient relative L2 versus materialized is respectively
  0/.00462952/.00729182/.00591965/.00648270/0/.00747728/0. Unchanged budgets pass.
  Worst tensor L2 .014221 and max/reference-peak .045455. This is not a new
  full-native BF16-versus-FP32 campaign; independent tiny FP32 oracles also pass.
- Combined B64/T512 fresh input tokens/s: single10,933; adjacent2 9,895;
  spread2 9,889; spread4 8,310. Allocated peaks all39.09GiB, setup reserved
  60.3–60.9GiB and postcapture current41.4–41.8GiB. Three-update medians are
  directional; equal overall peaks do not imply zero incremental RT memory.
  Combined two-layer matrix work is657.84–701.12TFLOPs/update, four-layer
  684.23–724.62, excluding nonmatrix/optimizer/launch/communication work.
- RT-only spread2 B128/T512 passes22,720tokens/s, allocated46.106GiB but
  reserved peak76.797GiB (current49.039). Protocol-authorized half-batch
  headroom follow-up B64 passes19,445tokens/s, allocated32.247GiB, reserved
  peak47.963GiB/current34.5GiB. Prefer B64 for conservative development;
  retain B128 as successful higher-throughput evidence with tighter setup headroom.
- Combined spread2 B8/T2048 passes4,708tokens/s, allocated31.289GiB/reserved
  peak44.432GiB. All16 B8/T512 passes939tokens/s at25.440GiB/32.543GiB;
  this small-batch stress is not optimized all-layer throughput or a full
  T512 all-gradient comparison. Actual all16 equivalence is only B1/T32.
- Combined counts:1,267,879,936 active training and1,185,153,024 deployable;
  NextLat82,726,912 is training-only. RT-only active1,176,764,416; the harness
  also retains8,388,608 frozen fusion weights, explicitly counted resident.
  RT selection and FBT pass sharing add no parameters.
- T512 forward tiles are fused. T2048 two-layer calls:4088 fused+6 fallback;
  those six cover75.0366% of historical attention pair area, not full-step
  arithmetic/time. Backward historical recompute tiles stay fused. Native RT
  remains Triton, not FA4. Profile long context before extending its kernels.
- Queue66490 and headroom81397 completed0. Final exact selection is
  `.runtime/olmo1b-step60000/f3e-final-inputs.json`; fixed/adaptive matrix is
  `docs/reports/olmo1b-f3e/execution-plan.json`. Every run has W&B, original
  source/protocol snapshots and checked raw JSON. Small GCS evidence lives under
  `olmo1b-f3e-multi-rt/`; see the [receipt](reports/olmo1b-f3e/storage-receipt.json).
  Reuse pinned native weights; disposable few-update states are not retained.
- GCS verification completed under `olmo1b-f3e-multi-rt/20260923T145449Z/`:
  evidence3,404,632bytes SHA256 `212185ceb933fd18b2658f61b33e5759a749125f3c175d9f645b109b2840128a`;
  manifest344,034bytes SHA256 `2cd11bbec19cdab1fa7791f3d36bb1b83f8ef193ad8c555a951a146bcbf69e6e`.
  All624 run/source pairs (39 unique files) and16 protocols match current/run-local
  bytes. Final in-container nvidia-smi reports0MiB used and no compute processes.

**Next review milestone:** complete F4 feature-combination runtime cards with a
representative multi-layer selection, then graph recovery/accumulation, padding
and online readiness. Genuine F5 needs a second GPU; only one H100 is exposed.
Keep native Q/K math. The existing AccumulateGrad stream warning remains visible;
single-GPU parity passes, DDP ownership is still untested. No learning experiment
or architecture/placement choice is implied by this functionality milestone.

**F3d bounded backward workspace is complete (2026-09-23).** Read the
[assessment](reports/olmo1b-f3d/assessment.md), [results](reports/olmo1b-f3d/results.md),
[protocol](reports/olmo1b-f3d/protocol.md) and [usage](olmo1b-f3d-usage.md).
The user explicitly authorized this milestone. No GPU or quality run is queued.

F3d durable execution record:

- Branch `feat/olmo1b-f3d-rt-memory`, base 70e86e2, initial runtime b85337c,
  final runtime 40305d8; reporting/retention 62ab5de;
  [PR18](https://github.com/taylorbollman/cdrm-w-latent/pull/18). Ten final GPU reports pass
  90/90 gates; 54 actual updates comprise 27 eager+27 graph. 428 scoped CPU tests
  pass. The two historical diagnostics are retained separately from final counts.
- Opt-in `backward_memory="recompute"` retains row normalizers and reconstructs
  probabilities in bounded attention/gradient tiles. BF16 full-product rounding,
  temporary self, query/prefix gradients and normal autograd ownership remain.
  Cache/static signatures freeze the option. Materialized F3c stays default.
  Forward, parameters, checkpoint, RoPE, normalization and losses do not change.
- Final probe02: 69 gates, including 48 frozen cases,4 long history rectangles,
  14 raw-cotangent blocks and 3 reconstruction memory sizes. All block forward/
  cache outputs are exact;13/14 block gradients are exact, worst global L2
  7.92e-7. Worst block versus FP32 is .003489. 392 recompute calls verified.
  Preserve 22 stricter diagnostic flags; primary budgets all pass. W&B puvm7gux.
- Native RT B8/T512, combined K2+NextLat B8/T512 and combined B2/T1024 initial
  gradient global L2 versus F3c is .00089227/.00237615/.00224863. Losses are exact.
  Same-candidate changed-input/weight/overwrite graph and three-versus-three
  full Adam/model/moment/scheduler/counter comparisons are exact in all three.
  W&B aycfe3gv/j67nv8l3/7hw7ai6t. Actual backward calls511/511/1023.
- Fresh paired full-update capacity: RT B128/T512 allocated42.10→38.60 GiB,
  tokens/s 26,454→26,257; combined B64/T51240.84→39.09 GiB,
  10,970→10,927; combined B16/T1024 33.16→31.29 GiB,8,637→8,579.
  Memory savings1.75–3.50 GiB cost0.39–0.74% measured throughput. Three-update
  medians are directional. Reserved peak/current are separately reported.
- Isolated B2/H4/head dimension 64 reconstruction peaks at T512/1024/2048 fall from
  29.29/116.08/462.16 MiB to 3.15/6.23/12.40 MiB. The linear claim covers RT backward
  attention scratch, not full-model memory or long-context forward fallback.
  Forward rectangles above 256 remain eager; recompute backward supports2048.
- Native/checkpoint shape remains16 layers, selecting only RT layer 0 here.
  Native Q/K math stays. All-layer/multi-layer optimized runtime, native T2048
  complete updates, padded graphs, graph recovery/accumulation and multi-GPU
  remain open. Local writer/finish VJPs/permanent discarded-Q remain separate.
- Driver/library mismatch was repaired by reloading idle NVIDIA modules after
  confirming no GPU jobs:580.173.02→580.178.04. Telemetry/persistence restored;
  fabric manager reports no NVSwitch on this single H100. All GPU execution
  used the container. Fresh controls avoid relying on old-driver timing.
- Probe01 passed before native RT attempt01 exposed graph-unsafe scalar indexing.
  Runtime40305d8 uses an equivalent diagonal-view zero. Final probe02 has identical
  numerical results/fixture hashes. Failed attempt01 did zero optimizer updates;
  retain its source/error/W&B record. Existing AccumulateGrad stream warning
  remains visible, with measured parity passing; revisit ownership for DDP.
- Queue session35102 completed0; `.runtime/olmo1b-step60000/f3d-next-queue.log`.
  Explicit final selection is `f3d-final-inputs.json`; raw runs and sibling logs
  persist. [Receipt](reports/olmo1b-f3d/storage-receipt.json) records small GCS
  evidence under `olmo1b-f3d-rt-memory/`, including both historical attempts.
  Original native weights are referenced, no disposable trained weights retained.

**Historical F3d follow-up:** multi-layer/long-context integration is now
complete as F3e above. The [resource ledger](olmo-resource-accounting.md) now
accepts explicit recompute work while keeping the materialized default. Current
next steps are F4 and subsequent readiness work described above.

**F3c historical backward fusion is complete and assessed (2026-09-22).**
Read the [assessment](reports/olmo1b-f3c/assessment.md),
[results](reports/olmo1b-f3c/results.md), [protocol](reports/olmo1b-f3c/protocol.md)
and [usage](olmo1b-f3c-usage.md). The user authorized this after F3b. No GPU or
quality run is queued. Its quadratic-intermediate follow-up is now complete as F3d above.

F3c durable execution record:

- Branch `feat/olmo1b-f3c-rt-backward`, base PR16 merge29408ca, core7e93ece;
  [PR17](https://github.com/taylorbollman/cdrm-w-latent/pull/17).
  Seven GPU reports pass78/78 gates;36 physical updates comprise18 eager+18 graph.
  The historical F3b RT reference and warmup/profile backwards are excluded.
- Independent opt-in `backward_tile_backend="triton"` fuses historical dK/dV,
  preserving BF16 whole-product rounding, FP32 error/adjoint arithmetic and
  original gradient ownership. Default eager stays. Both primary arms enable
  F3b fused forward/cast reuse. Cache/static signatures include the new option.
  Model/checkpoint/RoPE/normalization/loss/parameter count are unchanged.
- `f3c-tile-probe-01`:48 frozen tiles+12 tiny blocks pass bitwise versus primary
  BF16 control;48 standalone+144 block backward calls verified. Maximum tiny
  FP32 global gradient relative L2 is0.003464. W&B `4r1se0op`.
- Native combined B8/T512 `f3c-triton-combined-b8-t512-01` passes allfive checks:
  initial losses exact, gradient globalL2 .00139246, max tensor .00272989, max
  coordinate/reference-max .00694445, actual511 fused calls. RT-only counterpart
  `f3c-triton-rt-b8-t512-01` has bitwise initial reference/candidate gradients.
  Both have exact same-candidate changed-input/weight/overwrite graph checks
  and three eager versus three graph AdamW/model/moment/scheduler/counter parity.
  W&B `07awt17o`/`6t9h3z4x`. No numerical gate was widened.
- Large-batch capacity: combined B64/T51210,973 inputtokens/s at40.84GiB,
  RT B12826,480/s at42.10GiB, about1.06%/1.45% above F3b forward-optimized
  reference. Reserved peak/current:63.82/43.40GiB and68.30/44.31GiB respectively.
  W&B `3exnf0cz`/`hubes8cx`. Three-update medians; initial all-gradient checks
  were B8, not the capacity batches. Allocated peaks are unchanged.
- Reference/final combined profiles reproduce10,858/10,977 inputtokens/s;
  observer-neutrality/all-gradient equality and expected counts pass.
  Actual CUDA calls128,637→117,205; summed kernel self2.93613→2.90897s.
  Exactly511 historical backward kernels take7.448ms; forward kernels4.527ms.
  Direct Triton kernels are incompletely attributed under CPU annotations;
  do not claim37.85→1.565ms as an isolated-kernel gain. W&B `ev8anieg`/`myh5cmwd`.
- CPU tests432:315 core/kernel/dispatch/accounting,12profiler,15tileprobe,
  18nativevalidator,29reporter and43retainer. Independent final audits pass.
  Native call counts exclude gradient-buffer initialization; zero-reference
  norms require exact zero; source/protocol hashes are rechecked.
- Runtime outputs `.runtime/olmo1b-step60000/f3c-*-01/`, sibling logs and explicit
  `f3c-final-inputs.json`. Queueexec4671 completed0; no job remains. Each run
  retains its exact source version, including the baseline profile's earlier
  validator. Preserve stricter oracle/coordinate diagnostics as recorded.
- Small evidence retained under
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3c-rt-backward/`;
  [receipt](reports/olmo1b-f3c/storage-receipt.json) pins objects/hashes/generations.
  Existing native weights are referenced; no disposable trained weights archived.

**Historical F3c follow-up, completed as F3d above: quadratic-intermediate removal.**
Retain row normalizers and recompute probabilities inside attention/gradient
tiles, preserving BF16 whole-product rounding, temporary-self separation, final
dQ and attached-prefix gradients. Keep the current VJP as reference. One FP32
B64/H16 attention matrix is1GiB atT512 and16GiB atT2048. Local writer/finish VJPs
and discarded permanent Q projections remain separate opportunities. The next
change needs its own bounded protocol and review; do not infer a quality run.
All actual F3c full-model checks select RT layer0 only. Longer/more-RT-layer,
padded graphs, graph resume, accumulation and multi-GPU still need validation.
Keep native Q/K math and finish broader resource cards after relevant changes.

**F3b forward-tile optimization is complete and assessed (2026-09-22).**
Read the [assessment](reports/olmo1b-f3b/assessment.md),
[results](reports/olmo1b-f3b/results.md), [protocol](reports/olmo1b-f3b/protocol.md)
and [usage](olmo1b-f3b-usage.md). This closes the bounded forward prototype,
not the broader RT backward/memory work. No GPU job or learning run is queued.

F3b durable execution record:

- Branch `feat/olmo1b-f3b-rt-kernel`, base `e86a818`, core runtime `6c5c81c`;
  [PR16](https://github.com/taylorbollman/cdrm-w-latent/pull/16).
  Twelve completed reports pass; 60 physical optimizer updates comprise
  30 eager and 30 graph updates. Warmup/profile backwards do not advance Adam.
  The historical F3 RT B128 baseline is separate from these counts.
- Two opt-in flags: `cast_weights_once=True` reuses BF16 projection-weight
  copies within an RT forward while rereading current weights on every replay;
  `tile_backend="triton"` fuses historical QK/state/PV. Original defaults stay.
  Native checkpoint, recurrence, RoPE, losses and parameter counts are unchanged.
  Raw cache/graph execution signatures include both options.
- Cast-only actual combined B1/T32 and B8/T512 losses/gradients equal the original
  path bitwise. Fused actual RT and combined B8/T512 pass predeclared engineering
  screens: global gradient relative L2 versus original BF16 is 0.003538 and
  0.010067. All four candidates have exact same-candidate eager/graph losses,
  gradients and three-update AdamW/model/moment/scheduler/counter parity.
  Fused versus original is not bitwise; stricter diagnostic flags are retained.
- Forty-eight frozen tiles and 12 tiny native blocks pass. Tiny-block raw hidden/
  exported-KV cotangents, masks, odd lengths, attached prefixes and FP32 reference
  are covered; max FP32 global gradient relative L2 is 0.003464. Candidate/eager
  BF16 block outputs/cache/gradients are bitwise. All expected fused calls occur.
  Total scoped CPU test count is 307 (259 integration/kernel/accounting/launcher,
  23 reporter, 25 retainer). See retained test-results.txt.
- Complete graph updates at T512: combined B64 rises from 10,409 to 10,871 input
  tokens/s (+4.4%), RT B128 from 24,684 to 26,101 (+5.7%). Cast reuse alone gives
  10,717/25,588 respectively. Peak allocated memory remains 40.84/42.10 GiB.
  Peak reserved and current reserved are recorded separately. Three-step medians
  after warmup provide direction, not a randomized speed estimate.
- The final combined profile independently reproduces 10,866 input tokens/s.
  Ordinary layers use deterministic PyTorch Flash; historical RT tiles use the
  new Triton kernel, not FA4. CUDA annotation ranges overlap kernels/include gaps;
  helper CUDA-event timings also include uncaptured host submission overhead.
- `CDRM_FLASH_ATTENTION_SOURCE=installed` selects compatible FA4 4.0.0b20 and
  CuTE 4.6.0.dev0 without reinstall; the old vendor remains default. Standalone
  BF16 forward/QKV gradients versus FP32 and fixed-input forward capture pass.
  Read [environment details](olmo-fa4-environment.md). FA4 LSE values, captured
  backward and RT integration are not established by this smoke.
- [Resource accounting](olmo-resource-accounting.md) now covers all eight feature
  combinations analytically. At B64/T512 with the actual loss fixture, RT uses
  an estimated 294.9–316.5 TFLOPs/update, combined 644.5–689.3. This includes
  recomputation and excludes pointwise/optimizer/launch work. It is not a runtime
  measurement for all eight combinations or a wall-time prediction.
- Evidence: `.runtime/olmo1b-step60000/f3b-*-01/`, adjacent logs and explicit
  `f3b-final-inputs.json`. Every run has its own exact source snapshots, including
  earlier diagnostic versions; do not require all historical hashes to coincide.
  The final GPU job was `f3b-profile-triton-combined-b64-01`, W&B `d4fam7qe`.
  Original retained native weights are reused; no disposable trained weights
  are retained. Small evidence is stored under
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3b-rt-kernel/`;
  [receipt](reports/olmo1b-f3b/storage-receipt.json) pins objects/hashes/generations.

**Historical F3b follow-up: backward tile fusion (now completed as F3c) and memory.**
Retain the current custom VJP as a reference. Fuse historical dK/dV updates and
replace full probability/error arrays with row normalizers plus recomputed tiles
in bounded, separately checked steps. Incoming adjoints arrive progressively;
ordinary Flash backward cannot replace the entire recurrent reverse calculation.
One FP32 B64/H16 matrix is 1 GiB at T512 and 16 GiB at T2048. Also inspect local
writer/finish VJPs and the discarded Q in permanent full-QKV projection. F3b is
an intentional review boundary before a substantial backward rewrite.

Full-model F3b coverage selects RT layer0 only; the other 15 layers are ordinary.
Longer contexts, more RT layers, padded graphs, graph resume, accumulation and
multi-GPU remain untested with these optimizations. Keep native Q/K math. Finish
resource cards after relevant optimizations, and perform two-GPU checks when a
second GPU is available. Do not start a long quality run by inference.

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

F3's initial profile/prototype follow-up is now completed as F3b above.
The broader backward and resource-coverage work remains open under V4.

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

Historical F2 follow-up (now completed by F3): static-layout canonical CE/NextLat/FBT graph integration,
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
