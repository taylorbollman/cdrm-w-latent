# Workspace

Use `/home/taylorbollman/cdrm-w-latent` as the active project. The old projects
are unrelated archives and must not be searched unless explicitly requested.
Do not print credentials or environment-file values.

# Container execution

Never run CUDA, training, GPU evaluation or profiling in the host shell and
never silently fall back to CPU. Bootstrap GPU instances with
`bash /home/taylorbollman/start.sh`; CPU instances with Local SSD use
`bash /home/taylorbollman/start_cpu.sh`.

After bootstrap, use `bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh`.
For a single command, append `bash -lc '<command>'`. The container working
directory must be `/workspace/cdrm-w-latent`. Before GPU work, verify that the
command is inside the container and that `nvidia-smi` succeeds there.
Use `CDRM_DOCKER_GPUS=none` explicitly for CPU container work.

Add dependencies to `docker/requirements-docker.txt` and rebuild with
`scripts/docker_build.sh`. Keep `.env` and `.docker-home` out of Git.

# Experiment tracking

For future training and evaluation runs with graphable metrics, log online to
Weights & Biases under the `taylorbollman` entity. The user authorizes creating
appropriately named projects and runs there. Use the credentials in `.env`
without printing them, and include the project or run URLs in progress updates
and results. Keep the existing local records and GCS artifact retention alongside
W&B tracking.

# Pretrained model handoff

2026-09-25 two-H100 work is now authorized and active on feat/olmo-two-gpu.
Read docs/reports/olmo-two-gpu/progress.md before using the older single-GPU
status below. Hardware is two H10080GB with NV18/NCCL2.30.5. Eager tiny eight
modes and actual ordinary/RT pass; combined independent update2 has a retained
parameter/moment tolerance failure despite passing raw-gradient budgets and
exact replicas. Fixed-state follow-up is running. Tiny eager distributed recovery
passes24gates and its checkpoint is verified in GCS. Graph integration and later
capacity/ZeRO1 remain in progress; do not claim them complete from CPU tests.
Retain checkpoints progressively; only~19GiB free on the persistent main disk,
so verify GCS before cleaning any actual checkpoint. No quality run is queued.

2026-09-25 single-GPU preparation is complete. Read current handoff and
 docs/reports/olmo-single-gpu-readiness/results.md. Four actual B2/T512 cases
pass70gates/20physicalupdates: RT+combined eager accumulation/complete updates
are exact; RT+combined checkpoint/graph reconstruction continuations are exact.
FirsteagerRTattemptfailed0updates due redundant full-valid mask; retained.
Opt-in full_valid_causal fixes Flash dispatch and preserves defaults/padding/cache
restrictions. Corrected adapter runtimef8be057; recovery/firstfailureed4333d.
CPU scopes256/247/166overlap. No realDDP/NCCL/distributedgraphs/sharding validated.
GPU ended idle. Next actualhardware milestone needs twoH100s; use approved
 docs/native-rt-single-to-two-gpu-plan.md. Do not add B512, broadnumerics or
optionalRTfusion as prerequisites. Noqualitytrainingqueued. Inspectliveprocesses
before resuming. CapacityPR28merged16a08d7, evidenceverified:22reports/153updates,
RTB19229.746k/s and combinedB12812.413k/s atT512; FA4notadopted, B256captureOOM
retained. NativeRTselected at0/15; priorprecisionqualificationsstayopen.

2026-09-24 authorized milestone: user reaffirms native RT and the paper's
physical B512/T512 emphasis. Read docs/native-rt-large-batch-plan.md. Integrate
accepted ordinary fusions in RT/combined with bounded checks, then scale physical
batch from64/128 toward512 if feasible. B64 is a conservative development point,
not a measured optimum. Diagnose setup/capture versus steady memory before
declaring capacity; accumulation does not supply large-batch RT utilization.
Profile RT leaf only after measuring useful batches. User authorized execution
and a conditional ordinary-FA4 memory comparison near capacity. Current work
is on feat/native-rt-large-batch; frozen protocol under docs/reports/olmo-rt-large-batch/.

2026-09-24 user decision: adopt optimized native RT and move on; author-derived
stays an experimental reference. Native was already default. Full-model B64/T512
speed was tied; isolated author B128 advantages and31%/16% BF16 qualifications
remain recorded, not resolved by this choice. No further backend-adjudication
run is queued. User also accepted PR27 Dao RoPE/fused Adam for upcoming ordinary
T512 runs with rounded compiled SwiGLU. Read the current handoff and
`docs/rt-backend-numerical-clarification.md`. Prefer a bounded RT finish/writer
compile experiment over a broad activation-library survey when returning to
RT optimization. Other combinations still need their integration checks.

Latest ordinary fusion milestone is complete (2026-09-24), runtime b70b3ec.
Read `docs/reports/olmo-ordinary-fusions/results.md`, numerical/profile audits,
usage and current handoff. Opt-in Dao native-FP32 RoPE plus fused AdamW improves
repeated B64/T512 full-CE39.18k→43.61k (+11.30%), reserved45.68GiB; one matched
B16/T2048 pair36.99k→41.03k (+10.92%). T512 numerical screen passes; T2048 retains
a CE-only relative-loss failure (absolute7.39e-6 nats/target), while output/gradient
and all own graph/full-update checks pass. Fixed-gradient scalar/fused Adam
update difference4.57e-5 relative passes.12reports:11pass/1numericfail,64/65gates,
all61operational,96physical updates,672source pairs. CPU410distinct scoped plus
75evidence tests. Full-step profiler CUDA inventories include overlapping GPU
annotations; derived report excludes these and recovers CPU phases from trace.
Defaults/native RT/QK unchanged; no quality run. See storage receipt for retained
sources/logs/traces/plots/checkpoint reference. GPU idle, no further queue.
V4 now explicitly considers DDP then ZeRO1/2, and bounded contiguous RT finish/
writer compilation; neither is implemented/validated by this milestone. One GPU
available. RT/FBT combination, online/cache and distributed scopes need their own
checks; all prior RT qualifications remain open. Runtime flags belong in future
checkpoint configurations, not model state tensors; fused Adam affects resume
optimizer identity. Current Dao option rejects ordinary prefix/exported caches.

Previous ordinary efficiency milestone is complete (2026-09-24), final runtime
18351ef. Read `docs/reports/olmo-ordinary-efficiency/results.md`, its usage,
summary and current handoff. Rounded compiled ordinary SwiGLU passes the unchanged
numerical screen (bitwise outputs/loss, gradient L2 .003133) and exact own
graph/Adam checks. Repeated B64/T512 full-CE throughput36.75k→39.16k (+6.55%),
reserved46.17GiB; B32 compiled+alternating checkpoints34.43k→40.37k (+17.25%),
reserved54.75GiB. Defaults and RT math stay unchanged. FA4 retains tiny loss-only
screen failures; output/gradient and own operational checks pass. Directional
FA4 gains are0.65% T512 and3.55% T2048. Alternating B64/no-checkpoint B32 OOM
during graph capture; failures are retained.23reports:17passed,4numericfailed,
2OOM;152updates;1242source pairs. CPU274initial runtime,100final focused
(overlapping) and56evidence tests. Evidence/plots/checkpoint reference retained;
read storage receipt. GPU idle, no quality or further GPU queue. Review before
new ordinary RoPE fusion or resuming graph recovery/accumulation/online work;
prior RT qualifications stay open. New ordinary options require combination
checks before use in RT/FBT training.

Latest authorized queue: native RT efficiency Stage A is complete. Read
`docs/reports/olmo-rt-efficiency/results.md` and the current handoff. Runtime3fd27e0;
420 scoped CPU tests and21 GPU reports/41 gates/158 actual updates pass. Repeated
B64/T512 full-CE RT21.49k→22.43k (+4.36%), combined10.95k→11.20k (+2.26%).
RoPE reuse is exact; KV-only gradient changes pass existing budgets; graph/full
Adam parity is exact. Both switches default off; use explicit reuse_rope=True,
kv_only_writes=True as the improved native comparison arm. Parameters unchanged.
The user explicitly lifted the review stop and authorized proceeding directly
to author-derived RoPE Stages B/C after finishing/retaining Stage A. Read
`docs/olmo-rt-efficiency-and-author-comparison-plan.md` and
`docs/reports/olmo-rt-author-comparison/author-port-audit.md`. Preserve author
writer-VJP scheduling, compilation/caching and recorded precision differences.
Stage B/C isolated comparison is complete: runtime6eea309,122 runtime/accounting
and20 retention CPU tests,25 passing GPU reports plus one retained zero-update
loader failure. Exact own graph/Adam checks; BF16 cross-backend gradientL2.004133.
B128 author is7.5–8.6% faster across1/2/6blocks and uses substantially less setup
memory; B32 native is faster. These are block/MSE rates, not LM rates. Read its
results. The conditional full-model integration is complete (PR25): core2c38d649,
capacity524fa89 with identical integration source. CPU236 integration,47 evidence
and25 localization tests pass. Both full-model numerical screens FAIL
(raw-gradient L2 .312458 RT / .162606 combined), despite exact own Flash/graph/
Adam checks. Four B64 timing runs pass; rates are essentially tied at22.43k RT
and11.19–11.20k combined tokens/s. Author RT setup allocated memory is lower;
combined reserved memory is not lower. Native remains default.
Fixed-real-input/shared-cotangent block0 gradients agree at.003327 mixed and
4.65e-7 FP32. Final diagnostic987bc46 finds incoming gradients differ by.786611;
each own local VJP exactly reproduces its full-model block0 raw gradients. No
local backward arithmetic fix is supported by the tested separate-self probe.
Read the handoff/results for limitations, retained failures, source pins and
next functional milestone. Evidence retention is verified (493 members); pause for review. GPU is
idle and no quality training is queued. Prior F4 qualifications remain open.

Prior user-directed investigation: CE integration and the refreshed original
16-layer ordinary baseline are complete. Read
`docs/reports/olmo-ce-integration/results.md` and the current handoff. Optional
`NextLatConfig.ce_chunk_size=2048` separates CE grouping from KL128; None preserves
historical defaults/config dictionaries. Six GPU reports pass 18 gates and 36
physical updates; 152 scoped CPU tests pass. B64/T512 half-CE throughput improves
31.11k to 39.19k input tokens/s (+26%); full CE reaches 36.63k. Peak allocated/setup
reserved stay 26.74/37.17 GiB. Native ordinary/NextLat/combined gradients pass the
chunk comparison; same-candidate graph and Adam checks are exact. Runtime0d39a22.
Dao CE was audited, not adopted or GPU-tested; no projection fusion. No defaults,
Q/K or model math changed. Next: graph recovery/accumulation, padding/online
readiness; use matched CE settings for future performance comparisons. GPU idle,
no learning run queued, and prior RT+FBT precision qualifications remain open.

For pretrained OLMo / RT / FBT / NextLat work, first read
`docs/fbt-rt-nextlat-handoff.md`, then
`docs/fbt-rt-nextlat-research-plan-v4.md`. Original OLMo-1B at step 60,000
(approximately 252B tokens) is the selected primary model; its native checkpoint,
source candidate and tokenizer pins are in the handoff and selection audit.
O1 native ordinary fidelity/sequential RT and O2 native tiled execution/backward
are complete, with bounded GPU evidence and a documented raw-input roundoff
qualification. O3 language-model NextLat, optimizer/save-resume and bounded
single-H100 profiling are complete. O4
matched Python continuation is complete; read its results and assessment. The
user asked to assess and continue: O5a bounded FBT correctness now passes;
O5b matched ordinary-versus-FBT-only learning and its endpoint diagnostic are
complete and assessed. O5c fusion-only code versus code/general-text adaptation
is also complete: mixed training repairs the measured retention deficit while
the native backbone remains unchanged. O5d is complete:
fixed-weight finite K2/K3/K4 versus exact sequential feedback confirms that the
repair survives teacher-forced online execution through512-token contexts.
Read the current handoff and O5d assessment for retained evidence and the
ordinary additional-training control. O5e is complete: shared-source ordinary
continuation on the exact O5c mixed plan beats both fusion-only endpoints in
code/WikiText NLL, with a140-fold trainable-capacity qualification. Read its
assessment/results and current handoff. All four full checkpoints are retained;
no GPU job or further learning is queued. The user has reset the next priority
to functionality, bounded numerical health, integration, parameter/throughput/
FLOP accounting, Q/K-normalization assessment, native tiled-RT/Flash efficiency
and multi-GPU execution before quality comparisons. The user approved V4,
including early profiling. F1 is complete:18actual-checkpoint cases, two exact
BF16 recovery checks, short online cache parity and244scoped CPU tests pass.
Read its assessment/results and handoff. Early B1/T512 profiling identifies eager
RT scheduling/replay/launch overhead as a leading bottleneck; large clipped
startup gradients motivated F2, now complete and assessed. Ordinary-only
checkpointing passes exact BF16 complete-update parity and allows B128/T512:
RT20.2kinputtokens/s at41.4GiB, combined10.5k/s at51.9GiB. Native Q/K math stays.
Deterministic Flash SDPA gives exact B8/T512 native-stack graph checks; default
cuDNN has separately measured eager-repeat variability. F3's canonical combined
CUDA-graph integration is now complete: seven actual-checkpoint cases have exact
loss/gradient/full-Adam parity; six paired B32/64/128 capacity checks and 267 scoped
CPU tests pass. Read F3 assessment/results/usage and the current handoff.
Graphs capture forward/loss/backward with ordinary checkpointing; clipping,
AdamW and scheduler remain outside. At T512, graph RT B128 reaches24.7kinputtokens/s
at42.1GiB; combinedK2+NextLat B64 reaches10.4k/s at40.8GiB, B12810.8k/s at58.2GiB.
Use B64 for common development checks, preserving memory headroom with96.4% of
combined B128 throughput. Peak reserved setup and current postcapture memory
are separate. Deterministic ordinary Flash/no-autocast-cache settings differ
from F2 timings. Only RT layer0 is selected.
F3b's bounded forward prototype is now complete: per-invocation weight-cast
reuse plus a Triton historical attention tile, both opt-in. Read F3b assessment,
results, usage, FA4 environment note and resource accounting. Twelve GPU reports
pass, including 48 frozen tiles, 12 tiny blocks, four native correctness cases
and four capacity cases; 307 scoped CPU tests pass. Fused actual B8/T512 gradient
relative L2 versus original BF16 is 0.00354 for RT and 0.01007 for combined;
same-candidate graph and complete AdamW update comparisons are exact. T512 graph
RT B128 now reaches26.1kinputtokens/s at42.1GiB and combined B64 10.9k/s at40.8GiB,
about5.7%/4.4% above reference. Only layer0 is RT. The installed FA4/CuTE wheel
works with the explicit installed-source launcher selector; RT uses Triton, not
FA4. Analytic parameter/FLOP cards cover all eight combinations, with broader
runtime coverage still pending. F3c historical backward fusion is also complete:
independent opt-in backward_tile_backend, seven GPU reports/78gates and432 scoped
CPU tests pass. RT B8 initial losses/gradients equal F3b control bitwise; combined
initial losses equal bitwise and global gradient relative L2 is0.00139246. Both
have exact same-candidate graph/full-Adam parity. T512 RT B128 reaches26.48k
inputtokens/s; combined B64 10.97k/s, about1.45%/1.06% additional gains with
unchanged allocated peaks. Read F3c assessment/results/usage and current handoff.
F3d bounded backward workspace is complete: opt-in backward_memory="recompute"
retains row statistics and recomputes attention/adjoint tiles, preserving BF16
whole-product rounding, temporary self and query/prefix gradients. Materialized
stays default/reference. Ten final reports pass 90 gates;428 scoped CPU tests pass.
RT B8/T512 and combined B8/T512/B2T1024 initial gradient global L2 versus F3c is
.00089227/.00237615/.00224863; same-candidate graph/full-Adam parity is exact.
Fresh capacity saves 1.75–3.50 GiB with 0.39–0.74% lower throughput: RT B128/T512
26.26k/s at 38.60 GiB; combined B64/T512 10.93k/s at 39.09 GiB; B16/T1024 8.58k/s
at 31.29 GiB. Isolated reconstruction memory is approximately linear throughT2048;
full-model memory is not. Read F3d assessment/results/usage and handoff. One
failed capture attempt (fixed scalar indexing) and an earlier probe are retained
separately from the final10-run selection. Native Q/K math remains unchanged.
F3e multi-layer/context integration is now complete. Read its assessment/results/
protocol/usage and current handoff. Runtime543d243, helpersde3e6b9, base9af736e;
16 GPU reports pass48 gates,96 physical updates,474 scoped CPU tests pass. User
expects more than one RT layer, not necessarily all16. Adjacent(0,1), spread(0,15)
and four(0,5,10,15) pass bounded native gradient/graph/full-Adam checks; K3 andT2048
also pass. Maximum global gradient L2 versus materialized .007477; same-candidate
graphs/full-Adam exact. Combined B64/T512 single/two/four RT:10.93k/9.89k/8.31k
inputtokens/s at39.09GiB allocated,60–61GiB reserved peak. RT-only spread2 B64
19.45k/s at32.25GiB/47.96GiB; B12822.72k/s but reserved peak76.80GiB, so use B64
for conservative development. Combined spread2 B8/T2048 4.71k/s at31.29GiB.
All16 is separate stress: B1/T32 full equivalence and B8/T512 finite capacity
939tokens/s; no larger-context all16 gradient or optimized throughput claim.
RT adds no parameters; combined training/deployable counts1,267,879,936/
1,185,153,024. All source/protocol snapshots and W&B/GCS evidence are retained.
No GPU or quality run is queued. Next: complete F4 feature runtime cards and
remaining graph/online readiness. Forward rectangles above256 remain eager;
atT2048 six such two-layer calls cover75.04% of historical attention pair area,
not full-step arithmetic/time. Recompute backward is fused through2048. Padded
graphs, graph recovery/accumulation and genuine multi-GPU remain untested.
Direct Triton CPU observer attribution undercounts kernels; use device
traces/full-step timings. Two-GPU checks need a second GPU; one H100 is exposed.
F4 training resource cards are complete with a retained numerical qualification.
Read F4 assessment/results/roundoff-assessment/operator-audit and current handoff.
All eight B64/B96 capacity cases have finite full updates. At B64/T512,
ordinary/RT/NextLat/RT+NextLat/FBT/RT+FBT/FBT+NextLat/combined reach
31.11k/19.44k/24.03k/16.44k/15.80k/12.14k/12.18k/9.89k input tokens/s.
B64 is the common default; B96 helps RT/RT+NextLat, but larger FBT combinations
reserve 73.9–78.4 GiB. The 23 main reports pass 46/47 gates, with 132 updates
in successful reports; separate roundoff has six more. 235 scoped CPU tests pass.
RT+FBT B8/T512 has one MLP coordinate ratio of 6.3492%, missing the unchanged
6.25% limit; the failed gate stays. Candidate graph/full-Adam comparisons are
exact; FP32 materialized/recompute global L2 is 3.04e-6. Both BF16 control and
candidate differ about 18% from full-FP32 gradients at initialization; retain
this qualification, not numerical clearance. No core math/QK change or
quality run. Main runtime397885b, diagnostic949731b; failed probe01 context
construction is retained separately. Next: recovery/accumulation, padding/online
readiness, and a bounded precision follow-up as needed before learning.
Read the handoff
for current authorization and evidence. Do not infer long-run
authorization from platform work. Completed OpenELM code/results are historical
reference evidence; do not resume its superseded next milestone by default.

# RT numerical handoff

For future changes to the base Recurrent Transformer or its numerical tests,
read `docs/rt-numerical-handoff.md` first. It records the completed precision
baseline, reusable validation methods, harness adaptation constraints and
retained artifacts. The historical precision lineage is closed; future
experiments should have new output directories and an explicit scope.

# A5 experiment handoff

For the two-layer A5 experiment, read `docs/rt-a5-handoff.md` and
`docs/rt-a5-usage.md`. The first paired 10,000-update development pilot is
complete. The user chose full FP32 and minimal bounded correctness checks;
do not automatically restart mixed-precision or compiler studies. Final
confirmation remains unevaluated.
