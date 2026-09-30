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

2026-09-30 15:39 UTC: NEXT MILESTONE ACTIVE, authorized by latest user.
Branch feat/olmo-nfr-stability-128. Read docs/reports/olmo-nfr-stability-128/
protocol.md and progress.md. Prepare matched no-update K1-32 NFR64 KL1/.1
endpoint curves on common eight-row FP32/no-jitter panel. If bounded/settling,
continue only saved reduced-KL NFR64 to128, preserving Adam/RNG/data/LR/model;
inspect96/100/128 across existing warmup100. Stop128; no extra training beyond
that. Two endpoint probes RUNNING, physicalGPU0control/GPU1reduced, root
sessions2942/66539; monitor25914. Runtime olmo-nfr-endpoint-curves reports
result-control-01/result-reduced-01; W&B y8xlyhtw/sf5dwyaz. Do not launch
training until both complete and GPUs released. Root owns activation receipt.
Continuation helper prepared with36CPUtests/222pins;44probeCPUtests passed.
Scope/launch commands and retention receipts in new report progress.md.
Historical200/208/210/215 bytes immutable; new protocol also now pinned.
Root coordinates all GPU launches; checkpoint/cloud/W&B policy unchanged.
Earlier notes calling this proposal unapproved are superseded by authorization.

2026-09-30 FBT stability and paired NFR KL continuation COMPLETE. PR55 merged
871d23c31f8e07337e5064f9c4d18fbb3ae4e7e3; main is current. Final closeout
metadata is in docs/reports/olmo-fbt-stability/progress.md.
Read that directory's results.md, post-diagnostics.md, resource-ledger.md and
next-steps.md, plus docs/reports/olmo-nfr-kl-continuation/results.md/validation.md.
Both NFR branches stopped64/cloud64/W&Bsynced, queue completed_pair/exit0.
Both H100s verified idle inside container at14:32UTC. No GPU work is queued.
Control KL1 CE2.930108/6.473414/6.604760/6.646855; reducedKL.1
2.774970/5.375003/5.481358/5.520121. Raw auxiliary losses higher for reduced
on every pass. Both32 resumed updates finite/clipped; median norm7.714→3.778.
Same original NFR32+populatedAdam, data/RNG/schedule; only KL coefficient
changed. RT0/15,K4,latent1,T1024,B12/rank,524288inputs/update, frozen215pins.
Not an F128 warmstart. Both added16,777,216 inputs; original128 plan preserved.
Independent pair audit16,483 passes; all branch/audit/summary evidence retained.
ControlW&Bujz924fj; reduced1xu07xdf; paired summaryni8f0ch6, SHA
b521e7546a3e4afef6fc81addb8dbf78b2e86585cb8f28e1d9c1bbd53907378a.
Runtime .runtime/olmo-nfr-kl-continuation; figures/receipts in report directory.
Next proposal: matched saved NFR64 K1–32 curves, then consider unchanged
selectedKL.1 continuation64→128. Neither is launched. NFR64 deep curves have
NOT yet been measured; existing component helper admits NFR32 only.
No useful-refinement, RT-benefit or general BF16-equivalence claim.

F128 COMPLETE/cloud128/synced32sqvp7e; terminalSHA8a07a7fc2a5ecafc4523a1f5adb6a9b2073ebd1f46a586c9034e3815977cbdfe.
Regular CE2.687038/2.919213/2.937069/2.942000; B1282.687618. All128finite,
116clipped then12unclipped. F nativeaudit160485/fullBprefix24617pass.
All17publications+evidence retainedGCS. F curves0/32/64/96/100/128 complete,
settle~1.6e-6 by16passes at128; origin inherits fusion128 preparation.
All five post-F GPU diagnostics completed: exactonlineF128 on2x128 crops,
NF32/NFR32 and NF64KL1/.1 K32 curves. All saved conditions settle tightly;
poor prediction is not explained by failure to settle. K4vs exactonline hidden
error~1.08%, K32~1e-6; no generation/T1024online equivalence claim.
Online helper complete/synced but pinned host launcher expected completed;
terminal-adoption.json preserves reporting-only mismatch, no rerun. F208 and
historical200/210 sources unchanged. Further GPU diagnostics must not run
without checking the idle state first. Post-diagnostics notes/retention complete.

2026-09-30 paired NF KL continuation COMPLETE, PR54 closeout metadata in
`docs/reports/olmo-kl-continuation/progress.md`. Read results.md, next-steps.md,
validation.md and storage-receipt.md there. Both sharedNF32 savedAdam branches
completed64 with128plan/LR/data preserved. KL1 vs.1 CE64:
2.845510/6.805770/6.923254/6.967279 vs2.704137/5.752041/5.883908/5.956001.
Median rawgradnorm7.791→2.928, all32updates/arm finite/clipped; rawKL/latent
higher under.1 in everypass. Laterpasses still~3.05–3.25nats behind first.
Directional loss-balance result; no useful-refinement/RT benefit or BF16clearance.
Nativepairedaudit16,139 and tiny pair/restart2,901/2,289 pass;125focusedCPUtests.
Frozen200unchanged,210executionpins. Summary W&B72jc2qi3, controlw5eekmse,
reducedx8f16eqv. Both64cloudverified/synced; no training queued. Retain parent
files on childresume. Runtime `.runtime/olmo-kl-continuation`, retention
`.runtime/olmo-kl-retention`, SSDkl-continuation. Recommend reviewed pairedNFR
saved32→64 KL1/.1 replication (~3–4h); not launched, needs explicit nativeNFR
scope. Full inventory/PRmerge details in progress; keep prior qualifications.

2026-09-30 saved-state feedback diagnostic COMPLETE; PR/merge closeout in
docs/reports/olmo-feedback-diagnostic/progress.md. Read results.md, forward-notes.md,
gradient-notes.md and next-steps.md there. Six FP32/no-jitter saved-state probes,
no training: NF0/NF32/NFR32 forwards8devrows, NF32 gradients2x2trainrows,
NFR32 gradients first2rows. Exact beta0 and crossbeta firstpass controls;
reconstruction<=2.47e-6; states/RNG/grads unchanged. NF0 already has bad laterCE;
beta.5 fails to repair. Broader laterreadouts and reduced position variation
accompany lowerauxloss; no collapse/quality/causality claim. NF totalCEaux
backbonecos-.058/+.231; NFR-.536, fusion-.646; firstCEauxopposition repeats,
but all jointdotCE values remain positive locally. No BF16 Adam inference.
Original200runtime/checkpoints unchanged; diagnostic helpers under scripts
avoid changing oldtraininginventoryglob. ExactGPU source snapshots retained.
Evidence/fixture GCS olmo-two-gpu/20260930T031600Z/feedback-*; receipts
.runtime/olmo-feedback-retention; W&Bsummaryp5xs1bod. No GPU/training queued.
Propose reviewed pairedNF savedAdam continuation32→64 controlvsKL0.1 only;
not launched, requires new branch identity, retainallpriorqualifications.

2026-09-30 first32 adaptation pilot COMPLETE, PR52 mergedfb866533
(final closeout metadata in its progress.md). Read docs/reports/olmo-adaptation-pilot/results.md, next-steps.md,
assessment-guide.md, validation.md, storage-receipt.md and progress.md. Runtime
85e5f78/200pins unchanged; new execution/analysis only. All B/NF/NFR32 finite
updates, dev16/32 and cloud32/synced W&B complete. No active/queued GPU work.
Same16,777,216new inputs/arm,T1024,effective524288/update; B32/GPU8slots,
NF/NFR12/GPU22slots. B original weights, NF/NFR paired fusion128 import/fresh
predictor/Adam, exact71named parameter tensors. Prior fusion exposure separate.

Dev32: B2.63179; NF2.96039/7.39349/7.42586/7.43514;
NFR3.04301/7.02627/7.05529/7.06382. Both improve16→32 but later passes remain
~4nats worse than own first. NFR laterpasses beat NF~.37, firstpass worse.083;
not a useful-refinement/RT quality win. All32 NF/NFR updates clipped, Bnone.
Selectedcompute+materialization71.0k/7.78k/3.57kinputs/s; executor12.6/51.9/
113.8min. Reserved42.50/58.70/59.08GiB. Keep timing/memory scope qualifications.
Summary ffb6e4e, W&Bp7vk0qz0. Sixteen checkpoints/21small receipts in inventory
43cad4ec, retainedGCS; inventory/closeout/admin later receipts separate.

Daemonrestart killed hostqueue01 only; NF GPUcontainer finished uninterrupted.
NF launcher exit unknown; final report/W&B/cloud authorities verified. Detached
queue02 adopted B/NF and ran only untouched NFR. queue01 preserved/stale;
queue02completed, no repeatedtraining. .runtime/olmo-adaptation-pilot,
SSDadaptation-pilot, GCSsmall olmo-two-gpu/20260929T231346Z. Reviewstop32 inside
unchanged128ceiling. Recommend bounded saved-state NF-first feedback sensitivity
and per-loss/fusion gradient probes, decisiveNFR confirmation, no automatic
continuation/newnumericalgrid. No BF16 clearance or Q/K/core change. Save/retain
progress every20–30min for future work; inspect reports/processes after interruption.

2026-09-29 asynchronous checkpoint milestone COMPLETE; PR51 merged039fa96b.
Closeout metadata is in docs/reports/olmo-pilot-async/progress.md. Read results.md,
pilot-plan.md, open-issues.md, test-ledger.md and storage-receipt.md there.
Runtime85e5f78/200 pins frozen; historical192 pins unchanged. New versioned
engine/loop/worker/executor resumes after immutable SSD save; one worker owns
storage and uses a CPU-only cloud child to avoid training-RNG races. Full
verification retained, drain before next save and terminal. VM loss while
uploading rolls back to the previous verified cloud checkpoint. A600s save
trigger is not a maximum rollback interval. No GPU job is active or queued.

CPU136 distinct tests pass. Tiny blocking/async passes2847 exact checks;
cloudrestore2/resume3 passes2427. Native NFR4 updates at T1024,B12/rank,
524288 inputs/update,22 slots/rank complete:2,097,152 inputs. Compute plus
materialization3568 inputs/s, reserved59.06GiB/GPU, sampled free12.78GiB.
Populated local-save regions76s; checkpoint2 background346s overlaps updates/
evaluation; terminal background365s drains. Checkpoints0/2/4 cloud-verified.
Native source/budget/ownership/evaluation assertions pass; exact new-runtime
native cloud continuation was not separately rerun. No BF16 clearance change.

Heavy clipping persists (norm211→52); final FP32 dev CE3.18058/7.87717/7.71814/
7.74559. Useful refinement remains unestablished. Keep explicit32-update review
of per-pass gaps, first-pass trajectory and separate-loss scales before any
extension; no update-zero dev measurement. CPU declarations B/NF/NFR128 ceiling,
firststop32 are prepared but UNLAUNCHED. NFR first segment budget roughly2h;
NF accumulated cost unmeasured. Four-update fixture is not cohort initialization.
W&B nativeosqidkgt, summary1ofs0x3r synced. Optional immediate metadata readback
failed; later independent confirmation passed, without rerunning training/charts.
Evidence/checkpoints retainedGCS; inventory12 checkpoints/16small receipts;
later inventory/closeout receipts separate. .runtime/olmo-pilot-async; SSDpilot-async;
cloud small olmo-two-gpu/20260929T204500Z. Save/push/retain every20–30min.
Check reports/processes before launching after interruption.


2026-09-29 ordered pilot execution COMPLETE, PR50 merged806c1be. Read
docs/reports/olmo-pilot-execution/results.md, next-steps.md, checkpoint-cost.md,
readiness-map.md and progress.md. Frozen runtime be74dde/192pins unchanged;
149 new CPU tests plus27 separate auditorv2 tests pass. Tiny insertion/cloud
resume/terminal acceptance passes2063/2136/1914 checks exactly; original v1
auditor false failures retained. Native B32/B64/NFR12 each completed8updates,
final FP32 evaluation and verified checkpoint0/4/8. No active/queued GPUrun.
Recorded compute+materialization rates67.3k/69.0k/3.58k inputs/s; these exclude
health/logging/eval/checkpoint gaps. Reserved42.40/60.75/59.06GiB perGPU, sampled
free35.21/16.87/12.78GiB. RecommendB32ordinary/B12combined. W&Bsummary48jhxju3.
Native fixtures differ in effective batch/ancestry, not a learning cohort. NFR
finite but heavily clipped; later dev passes worse than first. No BF16 clearance
change. Full checkpoint selected regions cost6.1–6.6min; later600sec cadence
means roughly16–17+min between durable publications, not every10min.
Evidence .runtime/olmo-pilot-execution; GCSsmallprefix olmo-two-gpu/20260929T190649Z.
Individual checkpoint receipt bytes use neutral-name inventory snapshots due
to generic archive filter; see storage-receipt. Historical model/vendor/engine/
tests unchanged. Next: costed short matched pilot declaration (startup, fixed
dev coverage, common logical batch, finite budget/stop), not automatic campaign.
Proposal128updates/firststop32 remains unlaunched; preserve prior qualifications.
Save/push/retain every20–30min. FinalPR/merge/retention state in progress.md.

2026-09-29 pilot data COMPLETE, PR49 merged c433406; final retention metadata is in
docs/reports/olmo-pilot-data/progress.md. Read results.md, coverage-assessment.md,
protocol.md, storage-receipt.md, test-ledger.md and next-steps.md there.
All 37 sources acquired: 584,851 unique documents / 310,669,141 stored tokens.
Ordered T1024 train panel: 134,217,728 inputs; 21 panels total. CPU 161 distinct
tests pass; independent 592-document retokenization and recovered ordered audit
(134,272 panel row entries / 336 literal samples) pass. Raw 111-object recovery,
first4 restored + next4 regenerated exact25files, remaining29 token recovery,
full corpus verification and all44 ordered-index objects (792,408,198 bytes)
pass. Books coverage is small; main/source overlap is explicit. See coverage.
W&B qyp83axd synced. Data/evidence retained in gs://fast-chunks; persistent
receipts under .runtime/olmo-pilot-data/. Large files are under
/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929 and -recovery.
No GPU/model training or confirmation outcomes; no active or queued data jobs.
Old model/vendor/helper/test sources remain frozen. No BF16 clearance change.
Next: new ordered execution/evaluation adapters to unchanged SSD engine; tiny
two-GPU graph/eval/cloud-restart acceptance; then native B/NFR T1024 capacity
and evaluation allocation. Data capacity is not a training budget. Keep prior
startup/exposure qualifications and save/retain progress every20–30minutes.

2026-09-29 SSD checkpoint readiness COMPLETE, PR48 merged5d901bab; read docs/reports/olmo-campaign-storage/
results.md, next-steps.md, operator-notes.md, storage-receipt.md and progress.md.
Runtime54ae688/172pins frozen; new versioned SSD executor/engine/storage/restore,
independent tiny auditor. CPU215distinct pass. Tiny two-H100 storage transition,
cloud resume and terminal evaluation-only pass2121/1649/1425auditchecks exactly.
Full native15.2GB asset restore passes70.63s; no native GPU or cross-version resume
claim. Keep newest2 local per new ownedsegment only after verifiedGCS + durable
receipt/journal/latest. Historical files and resume sources untouched. Boot~91GiB
free; SSD~1.4TiB. Evidence/checkpoints retainedGCS; bothGPUsidle, no queuedrun.
Next: bounded representative data recipe/orderedreader, then actualT1024 capacity
and declared evaluation allocation. See provisionaldata-plan; no newdata download
or qualitycampaign launched. Startup/exposure and BF16 qualifications unchanged.
Save/push every20–30min. FinalPR/merge/closeout authorities in progress.md.


2026-09-29 optimizer-history + per-pass evaluation COMPLETE; PR47 final merge
metadata is in docs/reports/olmo-campaign-evaluation/progress.md. Read results.md,
next-steps.md, operator-notes.md, test-ledger.md and storage-receipt.md there, plus
docs/reports/olmo-optimizer-history/results.md and original-checkpoint-notes.md.
Same-data raw backbone gradient difference falls12.059% to1.289% at adapted
NFR20 state. Fixed-state global actual Adam-update difference is0.969% with our
20-step history versus1.942% reset; history-control adjusted1.640% versus1.942%.
No epsilon/QK/core change, original pretraining Adam or universal BF16 clearance.
Probe unchanged complete state; all11integritychecks pass, W&B yezbu1wv synced.

New versioned shared runner executes pinned common-FP32/no-jitter per-pass dev
evaluation. CPU197distinct tests pass. Tiny live insertion/cloud-resume and
evaluation-only restored boundary pass; native NFR T1024 insertion preserves
every update and final state exactly against PR46 (1742checks,319sources).
Native W&B i0jnqdyy synced; checkpoints0/3 and all evidence retained in GCS.
Runtime sources frozen ecac9b9/164files; old helpers/tests/protocols/model core
unchanged. Latest numerical helper/audit source authorities are in reports.
Both GPUs idle, no queued run. Boot free~91GiB; no local pruning performed.
Next: actual representative mixture/dev membership, explicit startup exposure,
target-hardware physical batch/accumulation, SSD staging and retention policy.
No quality campaign or finite-plan extension launched. Save/push every20–30min.

2026-09-29 manifest component execution COMPLETE, PR46; final merge/retention
metadata in docs/reports/olmo-campaign-execution/progress.md. Read results.md,
test-ledger.md, storage-receipt.md, operator-notes.md and next-steps.md. New
shared B/N/F/R/NF/NR/FR/NFR launcher; all-eight original/fresh startup, adapted
fusion128 + fresh all-active Adam allowlist NF/NFR. Generic same-lineage resume
requires committed checkpoint, not a successful reference report. Lean updates
avoid full per-update hashes; startup/checkpoint/final integrity remains.
Training sources frozen febc312/155files; historical scripts/tests/protocols and
cdrm/pretrained remain unchanged. CPU242distinct tests pass. Tiny two-H100
reference/lean/stop/cloud-resume/terminal and native adapted NFR T1024 stop/
reference/cloud-resume pass exactly; final native audit1236checks. Adam resident
before DDP/capture; restored gradients/model/Adam/schedule/RNG/cursor exact.
B1/rank/K4/nativeRT0,15/latent+KL/3rows per update is bounded operational
acceptance, not throughput or production batch advice. Numerical qualifications
remain; no BF16-vs-FP32 clearance or quality campaign. Both GPUs idle; no queued
run. Checkpoints/evidence retained in gs://fast-chunks. Next: declared per-pass
held-out evaluation with live-state preservation; then review actual mixture,
budget/startup exposure and target-hardware capacity. Keep accepted sources
frozen; version next implementation separately. Save/push every20–30minutes.

2026-09-29 overnight numerical/readiness work COMPLETE, PR45; closeout/PR metadata
is recorded in docs/reports/olmo-fusion-startup/progress.md. Branch was
feat/olmo-fusion-startup. Read results.md, next-steps.md, implementation-map.md,
readiness-map.md, test-ledger.md and storage-receipt.md in that report directory.
No GPU job is active or queued; do not restart completed diagnostics.

FP32 fusion-only warmup128 improves NF precision. Full NFR4 plus16 conditional
updates end with common-FP32 CE5.669885/5.667641 for FP32/BF16 training, but
cumulative backbone updates differ16.27%. Both continuations inherit the same
BF16update4 model/Adam; these are not independent cold starts. Saved-state packed
T1024 BF16 prepared/captured losses and71gradients are exact. Crossprecision
prepared gradients differ1.0955% globally; separate sparse/prepared0.08792%global
still has50/71tensors failing unchanged elementwise limits. No universal BF16
clearance, quality/refinement win or QK/core change. No extra sweep warranted.

Actual ordinary-B host-loop/cloud-restore/replay passes460auditchecks. The
B-only manifest adapter (frozen c11cb05) also passes all stages,566auditchecks,
200sourcepairs and direct first-update parity. Its resume CLI deliberately
requires checkpoint1 from a fully completed reference; generic interrupted-run
recovery is next work. All-eight CPU manifest metadata validated;49,152-token
readiness prefix contains two books documents, not a production mixture.

CPU checkpoint readback verifies identical14.15GB bytes using streaming hashes
at~110MiBpeakRSS versus13.25GiB, both~92s. No retainer change or speed claim.
Storage snapshot:79verified small-stage receipts,65checkpoint payload objects,
208,579,046,712bytes. Independent selected13receipt/26object readback passes,
including1,212sourcepairs. Later audit/closeout receipts excluded from snapshot.
Checkpoints/evidence in gs://fast-chunks; exact authorities in storage-receipt.
No local files deleted. Bootfree~178GiB after these retained diagnostic states.

Next implementation: generalize accepted manifest execution to component arms,
explicit selected adapted startup and same-lineage recovery; separate lean
observations from heavy acceptance hashing. Follow implementation-map.md before
coding. Do not label adapted weights/inheritedAdam as original fresh startup.
Actual pilot mixture/budget/target hardware remain decisions. No quality campaign
launched. Completed helpers/tests/protocols and cdrm/pretrained/*.py remain frozen;
new versioned entrypoints must preserve historical source guards. Checkpoint10min
is a completed-boundary target, not a hard wall-clock guarantee. Save/push20–30min.

2026-09-29 crossed-state NF precision COMPLETE, PR44, feat/olmo-crossed-precision.
Read docs/reports/olmo-crossed-precision/results.md, next-steps.md, test-ledger.md,
storage-receipt.md and progress.md. Runtime3364376; four aggregate/eight physical
backwards,177.394s, no training. Cold backbone/adapted complete fusion gives
backbone8.7328%/fusion13.2529% BF16-vs-FP32 gradient error; adapted backbone/cold
fusion53.8216%/54.4150%. Reverse hybrid absolute errors worsen versus cold/cold,
despite smaller relative errors. Adapted fusion transfers much but not all of
AA benefit. CA incoming cotangent error still reaches16.57%; AC supported hidden
error9.323%; old AA12.44% spike remains unlocalized. No BF16 clearance.
CPU58passed finalscope; four first-pass controls and109source pairs verified;
W&Bcwxjdnwe synced, stage evidence independently read back from GCS. Both GPUs
idle. No extra GPU job queued. Recommend bounded FP32 fusion-only warmup from
original backbone/fresh fusion, fixedK4/beta1/jitter.02, then saved-state checks;
next-steps has provisional budget/optimizer, not adopted training protocol.
Prior runtime/test/protocol sources stay frozen. No core/QK/precision changes.
Save/push every20–30minutes; final PR/closeout state in progress.md.

2026-09-29 adapted-state precision COMPLETE, PR43, feat/olmo-adapted-precision.
Read docs/reports/olmo-adapted-precision/results.md,next-steps.md,test-ledger.md,
progress.md for finalPR/retention. Runtime9f4693e: saved O5c mixed update512
backbone+fusion imported exactly (68state entries), fresh predictor4entries
unchanged; old source guards untouched, six explicit source migrations.
Two NF precision cases/fourphysicalbackwards,63.210s. Backbone gradienterror
60.8698%cold→0.9085%adapted,cos0.9999588;fusion65.2122%→1.4051%. Absoluteerrors
alsofall; no denominator-only explanation. Record0/pass1hidden12.4366% worsens
versuscold4.0913%, laterfinal1.9277/2.2201%; do notclaimallforwardimproved or
spikeharmless. CEonly/noRT/T16, noBF16productionclearance. No optimizerupdates.
CPU43pass(one test-onlyscalarwarning);104sourcepairsverified,W&B8ymbic44 synced.
The proposed two hybrids are now complete; read the current crossed-state entry
above rather than relaunching them. BothGPUsidle. No core/precision/QK change. Preserveoldruntime
sources; save/push every20–30min andretainGCS.

2026-09-29 fixed-boundary precision diagnostic COMPLETE, PR42, feat/olmo-boundary-precision.
Read docs/reports/olmo-boundary-precision/results.md, next-steps.md, test-ledger.md
and progress.md for final PR/retention state. Runtime1023d7e; both NF CE anchors
reproduce exactly, including 60.8698% backbone discrepancy. Record0/pass1,3:
common-input fusion parameter-VJP errors0.387/0.390%, ordinary stack7.678/8.188%;
changing only stack inputs under FP32 yields11.428/29.136%. Whole-stack VJPs
include internal forward rounding; these norms are not additive causal fractions.
Eight exact local endpoints,104 local health checks,10 final integrity checks pass;
CPU31pass, GPU147.87s, no updates. W&Bd43pjmvw synced. Evidence/source snapshots
and44small boundary tensors retained with independent cloud readback.
That proposed adapted O5c matched-precision step is now complete; see above.
Local4.81GB checkpoint hash and actual schema import were verified, with old
source guards preserved. Do not relaunch from this historical entry.

2026-09-29 recurrence/precision separation COMPLETE, PR41.
Source branch feat/olmo-recurrence-precision. Read docs/reports/olmo-recurrence-precision/
results.md, test-ledger.md, next-steps.md and progress.md for final PR/retention.
Eight CE cases N/NR/NF/NFR x FP32/BF16 on original isolated T16/two-B2 fixture:
shared-backbone gradient errors 0.98%/25.46%/60.87%/95.85%. Shared state/noise,
first-pass identities and old NFR endpoints reproduce exactly. NextLat branches
execute with zero auxiliary cotangents. Neither recurrence is an exclusive cause.
One conditional NF-only FP32-fusion probe completed three cases: backbone error
60.87% to 56.16%, but final hidden-state errors worsen on both records; do not
adopt this as a fix or broaden precision changes automatically. No core/default,
Q/K-normalization or architecture change, no optimizer updates. Matrix runtime
6ed920a, fusion a5e9540. CPU scopes 66 and 27 pass (overlap). W&B lt54objk and
wih59gy7. Numerical qualifications remain; T16 is not packed T1024 clearance.
All completed helpers/tests/protocols stay frozen. The proposed bounded
shared-input/shared-cotangent diagnostic is now complete; see the entry above.
GPU runs finished and idle. All stages and closeout retained with independent
readback: 4 receipts, 8 listed objects, 218 inventory members. No next GPU run
is queued. Preserve work every 20–30 minutes and retain in GCS.

2026-09-29 bounded numerical localization complete, PR40, feat/olmo-precision-localization.
Read docs/reports/olmo-precision-localization/results.md, test-ledger.md,
next-steps.md, storage-receipt.md and progress.md first. Four single-process
GPU stages: six precision/backend cases, eight fixed-hidden auxiliary cases,
three crossed-backend CE cases, and eight local-attention sites x three VJPs.
No core/model changes or training. Final focused CPU suite: 44 pass; earlier scopes
are overlapping. Combined BF16/math versus FP32 still 81.50%, production 85.96%.
CE ordinary Flash with eager RT exactly equals Flash with Triton RT, both 79.85%
from math/eager. Fixed-hidden BF16 auxiliary layout errors 0.1416%/0.2016%.
Fixed actual attention QKV/cotangent: Flash vsFP32 local gradient errors
0.174–1.501%, output0.160–0.186%; local Flash outputs exact to captured values.
This supports full-model sensitivity investigation, not a large local Flash
backward defect or numerical clearance. Prior 3.40224%/1.6953% qualifications
remain. The proposed eight within-arm CE precision cases separating
ordinary/RT/FBT/FBT+RT are now complete in the current entry above. Source 7082225,
old protocols/helpers frozen. GPU idle. Stage evidence and small tensor anchors
retained with independent audits and verified closeout; see progress.md.
Save work every 20–30 min.

2026-09-29 packed campaign readiness complete on `feat/olmo-packed-campaign`,
PR 39. Read docs/reports/olmo-packed-campaign/results.md, test-ledger.md,
storage-receipt.md, precision-assessment.md and next-steps.md first.
Opt-in continuous-stream-v1 carries attention/RT/FBT within chunks across true
boundaries; CE crosses, NextLat latent/KL exclude boundaries. Isolated default
unchanged. All 6,947,277 train tokens match independent packed-stream oracle;
verified disk index and committed cursor restore at a new path. Do not retokenize.
Tiny all-eight-arm two-H100 eager/graph and actual pretrained B/NFR operational
checks pass. Deterministic T1024/B12/rank/K4/nativeRT0,15/both NextLat losses:
524,288 valid tokens/update,22 slots/rank; fresh-process cloud-restored next update
is bitwise exact on both ranks, actual Adam resident before DDP/capture. Write
13 gates/resume 10 gates pass. About 3.50k input tokens/s,59.03 GiB peak reserved and
12.82 GiB sampled free/GPU on resumed update. Both checkpoints/evidence retained.
Original resume01 failed; isolated T1024 Flash backward nondeterminism confirmed.
Runner e5a593b enables and pins deterministic controls before CUDA; new matched
pair 02 passes. This resolves repeatability for the pinned execution contract.
It does NOT resolve BF16 numerical qualification: isolated loss-layout 3.40224%,
packed fixture 1.6953%; BF16/full-FP32 combined gradients differ ~86%,cosine ~0.51
on initial isolated NFR T16. Kernels change too; cause unlocalized. CE-only
layouts agree; auxiliary paths carry layout gap. No architecture/QKnorm change,
quality training, production mixture or H200 qualification. Next recommendation:
six-backward precision/backend bridge, then fixed-hidden auxiliary cotangents.
CPU regression 971 passed plus overlapping 20 data/18 determinism checks. GPU tests
finished. See progress.md for final retention/PR state; preserve prior failures.

2026-09-29 campaign two-H100 execution, cloud restart and capacity complete. Read
docs/reports/olmo-campaign-two-gpu/results.md, test-ledger.md, usage.md and
storage-receipt.md. PR 38; capacity source cac5c1c. Tiny all-eight-arm
NCCL eager/graph pass; actual pretrained B/NFR prepared-reference graph passes.
Full 15.21 GB NFR checkpoint restored from GCS; new-process next update matches
original live-graph continuation bitwise on both ranks. Safe scalar metadata
fix retains weights_only=True. Independent NFR BF16 sparse/dense gradient
difference 3.40224% remains FAILED; reproduces before DDP, full-FP32 loss-layout
comparison passes (7.38e-7). Do not mislabel operational passes as numerical
clearance. CPU regression 445 pass, then 21 focused capacity checks (overlap).
T1024 K4/RT0,15/all NextLat losses: recommend B12/rank/M2 starting point,
4,311 input tok/s, 59.06 GiB peak reserved and 14.23 GiB free/GPU. B8 fallback;
B16 passes at 4,970/s but only 3.30 GiB free; B32 skipped. GPU tests stopped.
Read next-steps.md for proposed packed stream semantics, actual-data cursor
recovery, cold T1024 restart/accumulation and bounded BF16 follow-up. No quality
run or production packing. All 20 receipts/46 cloud objects verified; 1,120
source pairs checked. Save progress every 20–30 min; retain in GCS.

2026-09-28 reusable Dolma document preparation complete. Read
docs/reports/olmo-document-shards/results.md, usage.md and storage-receipt.md.
CPU-only seven-source v1_5 coverage fixture:28shards/12,512docs/7,054,230tokens.
124tests pass; GCS-restored partial resume matches all86 files of uninterrupted
prep;12,514rawrows independentlyretokenizeexact. W&Babscmr9h. Raw/token/evidence
artifacts retained under gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/.
Not productionmixture or packed-modelqualification. No contexttruncation; true
docoffsets preserved. User's cited OLMo shift is withinchunk, not EOSawareoffset.
Actual twoGPUupdate/restart checks remain next; final concatstream boundary
policy needs explicitmodeltests. Largerbootdisk notneeded forthis boundedprep.

2026-09-28 campaign graph integration complete. Read
docs/reports/olmo-campaign-graph/results.md, usage.md and progress.md first.
Runtimebe11ac1: changing right-padding/masks/jitter/counts with local CUDA graph
accumulation; eager DDP per-microbatch jitter. CPU867pass plus28focused(overlap).
Actual pretrained NFR K4/RT0,15 B2/T16: both probes pass11stages; raw gradients
and complete two-update Adam trajectory exact versus same-BF16 eager. Final
repeat removes diagnostic autograd stream warning. W&B7fumon57 synced; artifacts
retained inGCS. GPUidle; ready for two GPUs now. Next realNCCL accumulation,
captured local/final-sync microbatches, fresh-process checkpoint restart, then
K4/T1024 resource calibration. Local graph runner is explicitlyworld_size1;
CPU/Gloo is not GPUdistributedacceptance. Noqualityrun or productiondata;
priorBF16qualificationsremain. Saveprogress every20–30min.

2026-09-28 portable campaign readiness complete. Read
docs/reports/olmo-campaign-readiness/results.md, usage.md and progress.md first.
Runtime714f31c: opt-in K4/RTonallpasses, separate CE/auxpassweights, keyedjitter,
explicitAdam/tokenclock, bounded pinnedraw/tokenizeddata contracts. CPU706pass;
finalingest28pass(overlap), freshprocessCPUresumeexact. PretrainedNFR B1/T16
4GPUstagespass, independent loss/grad assemblyexact,2finiteupdates,60sourcepins.
W&Bkb0lbu1k synced; evidenceGCSretained. Largeinitialclipping is diagnostic only.
Portable slice complete; next paddedFlash, perrankjitter, graphmasks/accumulation
and actualdistributed freshprocessrestart. Noqualityrun/productiondata/calibration
orhardwarelaunch. PriorBF16qualificationsremain. Saveworkevery20–30min.

2026-09-28 combined T1024 benchmark complete. Read
 docs/reports/olmo-combined-t1024/results.md and progress.md first.
Same K2/nativeRT0,15/NextLat/nativeRoPE/SDPA. B64 repeated11,092.06tok/s,
10.27% below savedT512 vs25.70%deficit atT2048;68.445GiBreserved/9.195GiBfree.
B32:9,823.08/s with36.545GiBfree. 3stages/15gates/24updates pass;160CPUtests;
489newsourcepairs; evidence retained. GPUidle. Runtime dcbde27/audit46c03f0.
Recommend1024 for directional screen, pending data/control/budget review.
The proposed500M-token+SFT experiment is not launched. PriorBF16qualsremain.

2026-09-28 combined T2048 benchmark complete: read
 docs/reports/olmo-combined-long-context/results.md and progress.md first.
SDPA-only, K2 FBT, nativeRT0/15, NextLat bothlosses, nativeRoPE. RepeatedB32
9.185k/s versus savedT512B12812.362k/s:25.70%less throughput;69.014GiBreserved,
7.914GiBsampledfree. B16:7.502k/s,40.986GiBreserved,35.977GiBfree.
4stages/20checks/32updates pass;137CPUtests;652sourcepairs; all4stages retained.
No FA4 or newT512run. GPUidle; pauseforreview. Runtime79e850a, audit5f6c6ac.
PriorBF16qualificationsremain; largerRTforwardtilesstill eager.

2026-09-28 ordinary T2048 benchmark is complete on one H100. Read
 docs/reports/olmo-ordinary-long-context/results.md and progress.md first.
Recommend B32/T2048: SDPA41.48k/s, FA443.13k/s (+3.99%); both37.52GiBreserved,
40.90GiBsampledfree. Saved T512B12844.04k/s was reused; no T512rerun.
Runtimeedf3d0b/111CPUtests;6capacitypasses+1retainedloss-onlynumericfailure;
all35operationalchecks pass. All7stages/closeout retained inGCS. GPUidle.
No RT/FBT/NextLat/qualitytraining; defaultattention unchanged. T2048
rawgradient/output budgets pass, but strictCEqualification remains visible.

2026-09-25 ordinary-only two-GPU throughput follow-up is complete. Read
docs/reports/olmo-ordinary-two-gpu/results.md and progress.md before the RT
milestone below. Recommend Dao RoPE DDP B64/rank at T512:85.2k tokens/s,
44.9GiB sampled free/rank; B128/B192 add only about1%. Matched singleB128
44.0k/s yields1.93x scaling. Eight stages pass;93CPUtests; all evidence retained.
No RT/FBT/NextLat executes in this sweep; no quality run or GPUjob is queued.
ZeRO2 deferred, not ruled out for throughput; graph accumulation not validated.
Read zero2-nextlat-clarifications.md for reduction tradeoffs and confirmation
that current OLMo NextLat uses active SmoothL1+KL (both coefficients1.0).


2026-09-25 two-H100 milestone is complete. Read docs/reports/olmo-two-gpu/
results.md, summary.md, progress.md, usage.md, test-ledger.md and storage-receipt.md.
31attempts:26passed/5retainedfailures, all retained in GCS with8checkpoint stages.
Native RT at0/15 within16layerOLMo; combined=K2FBT+RT+NextLat. RealDDP/NCCL
CUDAgraphs and scoped reconstruction recovery pass; freshprocess restart untested.
Independent combinedBF16 update qualification and older native/author findings
remain. Development:DDPRT B128/rank55.8k/s,combinedB64/rank23.4k/s.
Fixedlargerbatch:ZeRO1RT B192/rank58.2k/s,combinedB128/rank24.4k/s.
Noqualityjobqueued; GPUqueuefinished. Persistentdisk~19GiBfree; cloudverify
fullcheckpointbeforeduplicatecleanup. Reviewresults beforechoosingnextworkload.

## Historical milestones (hardware availability below is historical)

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
