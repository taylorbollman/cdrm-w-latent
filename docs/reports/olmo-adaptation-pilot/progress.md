# Pilot progress

2026-09-29: user authorized the proposed first32 comparison. Branch
`exp/olmo-adaptation-pilot32`, starting main `358ab65`. Container preflight
confirmed two idle H10080GB GPUs, about90GiB boot free and1.3TiB SSD free.
The prepared declarations were copied byte-for-byte into
`.runtime/olmo-adaptation-pilot/declarations-01`. CPU-only async preflight
passed for B/NF/NFR; all200 runtime pins unchanged. Adoption report SHA256:
`6e9fc86876a582ebefa2cd1ed62ce36408ce5ea73a6c86dbb567c3a3d7d2ce1a`.
All128 ordered memberships/counts remain equal. First32 counts per arm are
16,777,216 inputs and16,760,832 CE targets; enabled auxiliary counts for NF/NFR
are16,730,817 latent pairs and16,684,487 KL triples. Disabled B losses count0.

Queue helper `.runtime/olmo-adaptation-pilot/run_queue.py` runs B, NF, NFR
sequentially, with explicit lean/async/stop32 and14400s external bound per arm.
It stops scheduling after a failed/incomplete stage. Stage names:
`native-b32-first32-01`, `native-nf12-first32-01`, `native-nfr12-first32-01`.
Expected report status at32 is `stopped_at_boundary`, segment_completed=true,
plan_completed=false. Keep the128-update finite plan unchanged. New SSD
namespace is `adaptation-pilot`; old cloud declaration roots gain these new
unique segment suffixes. All GPU work remains inside the required container.

Root owns launch/monitoring. The `native_async_declaration` agent independently
audited launch and is preparing a JSON-only result summarizer; no other GPU
jobs are authorized. Inspect `queue-01/report.json`, individual reports and
process state before recovery. Do not repeat completed work after interruption.

23:31 UTC: B completed32 with exit0, final cloud checkpoint32 verified and
W&B `nxm2prv9` synced. Queue automatically started NF. B reportSHA
`203b8fd63448cd4da7a90f424d374d37417758a73c85ca9825d26129bb5035bb`;
independent summary-b-01 passed, SHA
`4712d55fe8963b934904705be9ec549cce4fe8188cbdc36c16be3d81fa04d504`.
All B updates finite/unclipped, norm0.3797–0.4532. Dev16/32 CE2.631118/2.631794.
Selected training+materialization71,006inputs/s; max sampled reservation42.50GiB,
min sampled free35.11GiB. Executor757.34s includes terminal retention; not pure
training throughput. B stage and summary retention have been started/completed
as recorded in persistent receipt files. PR52 is draft. Runtime unchanged.

23:34 UTC: NF W&B `uf1ojrgl` is active and graph warmup has begun. Its local
checkpoint0 is saved; cloud retention runs in the background. B and summary-b-01
evidence retention both completed successfully. Declarations, launch-time helper
snapshots and standalone analysis helpers are also retained. Queue remains
running in the original host process, with NFR pending; do not start another
GPU run. Host session identifier7388 is a convenience, not recovery authority.

23:54 UTC: NF reached update16; first fixed FP32 development evaluation is
running. Update8 checkpoint is cloud-verified; next cadence save follows the
current update/evaluation boundary. All updates remain finite; gradients vary
(raw norm425 at1,24 at8,40 at13,22 at16), all clipped. Auxiliary losses fall;
per-pass CE assessment is pending. NFR remains queued. New assessment-guide.md
records exact loss semantics and planned review criteria; no runtime changes.
Posthoc inventory helper is prepared and retained in analysis-helpers-02.
Immutable progress-2354 snapshots contain current queue/native report metadata
and completed B retention authorities, not a coherent new recovery checkpoint.

2026-09-30 00:16 UTC: NF reached32 with both development evaluations complete;
final local checkpoint/retention still in progress, so do not claim the arm is
closed. Verified cloud checkpoint26 currently provides recovery. Dev CE16:
3.253614/7.774963/7.798390/7.815834; dev CE32:
2.960392/7.393494/7.425865/7.435141. Final gradient norm6.60185; all32 updates
clipped. Absolute CE improved, but later passes remain far worse than pass1.
Auxiliary improvement is not sufficient refinement. NFR remains queued;
originalqueue/session7388 still owns execution. progress-0016 preserves bounded
metadata snapshots. Prior progress-2354 retention succeeded.

00:25 UTC: Chat server restart killed host queue-01 parent, but NF's container
survived uninterrupted and finished all32, terminal cloud verification and W&B
sync. NF original launcher exit is unknown because its parent was lost; no
exit0 is fabricated. Final reportSHA8e6fe4d8aa933650943320c4f277fd92f4d59e530e08ef456d23634e79cbd5a0.
Independent B+NF summary passed, SHA45de106de400358cf9cb318338da6d567a0712b80236766b467f34943d4dbda3.
GPU-container preflight confirmed both GPUs empty. New detached resume_queue.py
validates/adopts B/NF final authorities and launches only untouched NFR with the
exact original command. queue-01 is preserved; queue-02 is current scheduling
authority, hostpid1282711. NFR stop32/timeout4h/settings unchanged; no repeated
training, checkpoint-resume or GPU-runtime modification. status.py now selects
queue-02 when present. B/NF evidence remains immutable. progress-0016 retention
succeeded; completed NF and summary-b-nf-01 retention started.

00:48 UTC: NFR W&B5byv5pkq active; graph preparation finished and5 counted
updates complete. Cloud checkpoint2 verified; norm trajectory211.12/76.50/
73.80/52.42/39.38, all clipped and finite. First update137.92s callback, timing
remains provisional. Origin comparison validates all71named parameter tensors
exact between NF/NFR and both replicas (values/shapes/dtypes; excludes buffers/
Adam/RNG). origin-pair-01 retained; completed NF, B+NF summary and recovery
launch snapshots all retained. queue-02 detachedpid1282711 owns NFR; original
queue-01 stale report stays preserved. Next dev at16; stop32 unchanged.
progress-0048 captures immutable metadata while native state retention proceeds.

01:10 UTC: NFR completed12 finite updates; current norm31.89, following
17.03/15.98/29.96 at9/10/11. Full update callbacks settle near147s (first138s),
excluding inter-update checkpoint operations. Local checkpoint12 saved; cloud7
verified. Queue remains detached/active. next-steps-draft.md is proposal-only
for saved-state localization if later-pass deficit persists, with no diagnostic
implementation or launch. progress-0048 retained successfully; progress-0110
snapshot preserves current metadata and draft. Next dev16, fixedstop32.

01:33 UTC: NFR at19 finite/clipped updates, checkpoint16 cloud-verified.
Common FP32 dev16 completed:3.233708/7.408049/7.420141/7.435445. Relative to
NF at16, firstpass is0.01991 lower and later passes0.36691/0.37825/0.38039 lower.
This is a modest directional advantage at equal exposure, with a large later-
pass deficit still present. Do not infer scientific efficacy. Evaluation16
added about225s to update callback; selected training timings exclude it.
No new runs/changes; stop32 remains. progress-0110 retained successfully;
progress-0133 records current metadata. Current queue-02 and active NFR remain
sole execution authorities, not original stale queue-01.

01:53 UTC: NFR completed27 finite/clipped updates; latest norm10.02, with
cloud checkpoint21 verified and local26 saved/background retention underway.
No new evaluation since16. Five updates remain before finaldev/stop32/terminal
retention. progress-0133 retained and45c0224pushed. progress-0153 preserves
current metadata. No additional GPU work or change in scope.

02:12 UTC: NFR reached32 and final development evaluation completed. CE32:
3.043012/7.026269/7.055292/7.063815. Later passes are0.3672/0.3706/0.3713
lower than NF32, while firstpass is0.08262 higher. All32 finite/clipped,
finalnorm10.5631. No pass refines firstpass; large deficit remains. Training
stopped; terminal checkpoint sequence draining prior31 before32save/retention.
Do not claim fully closed yet; wait finalreport/W&B/cloud32. progress-0153
retained;8d28794pushed. progress-0212 preserves endpoint observations.

02:25 UTC: All arms fullyclosed and queue-02 completed_all_three_first32.
Both GPUs confirmed0MiB/no compute processes in requiredcontainer. Fullsummary
passed with exact paired origin parameters, SHA
ffb6e4eae228e4ae8e57394bc117a90ce2918eab3d3adb7820f64084d3e1178c.
NFRfinalreportSHA01bceb2a1191adb513bea974d8dcb0a5b52e8b5c3b384dbd0e69d21cfea665f7.
Joint W&Bp7vk0qz0 synced (trackingreportf8853c62); nativeNFR, summary,tracking
andqueue02 retention verified. Inventory43cad4ec passes:16checkpoints,
21smallreceipts,74distinctobjects/211,682,402,282bytes; its own upload and later
closeout/admin excluded by design. Independent reviewer confirms96updates,
allreported arithmetic/scopes; mask-deriveddenominator and wall-clockoverlap
wording clarified. Finalresults/validation/storage and next-steps.md replace
draftassessment. No more GPUwork; stop32 retained. PR52 closeout next.
