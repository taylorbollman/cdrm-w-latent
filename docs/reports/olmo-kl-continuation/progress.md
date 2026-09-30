# Progress and recovery

2026-09-30: user approved paired NF update32→64 KL1/.1 continuation. Branch
`feat/olmo-kl-paired-continuation`, based on main a2ea5c4 / merged PR53.
Original 200-source production runtime remains unchanged. New helpers explicitly
fork only objective metadata after exact saved-Adam restore. Protocol in this
folder is the execution authority; no automatic extension beyond64.

Implementation review covered independent parent boundary equality, original
strict configuration loading, explicit child identity, scheduler/data continuity
and graph construction after the transition. The launcher retains a dependency
on the original parent files even for child resume; metadata records it.

Initial focused CPU validation: 40 tests pass (7.50s). Independent auditor tests
and tiny two-GPU unchanged-control/cloud-restart checks are pending. No native
continuation launched yet. Runtime/evidence: `.runtime/olmo-kl-continuation`.
SSDs: `/mnt/localssd/cdrm-checkpoints/kl-continuation`; parent NF32 stays under
`adaptation-pilot/native-nf12-first32-01/update-000032`.

Host launchers must be detached with a new session. Inspect live container work,
report.json, *-launch.json and publication receipts before recovery/relaunch.
A running/stale host report alone is not evidence the GPU run stopped. Commands
and source snapshots are retained per attempt; never overwrite failed attempts.

Native pair launched after acceptance. Tiny control02 reproduced accepted
updates2–3 exactly; reduced01 resumed from a verified cloud2 restore in
reduced-resume01 exactly. Independent v2 audits passed2,901 pair and2,289
restart checks, including source snapshot bytes. V2 fixes an audit-only schedule
shape assumption: tiny binds configuration.schedule, native has payload.schedule.
Frozen v1/runtime210 remain unchanged; 18 v2 regression tests pass.
Tinycontrol01 failed before parent restore/training because the launch prefix
was outside the existing allowed cloud namespace; launcher corrected only.
Failed evidence retained. Production code commit29e3948 pushed; draft PR54.

Detached host queue: `.runtime/olmo-kl-continuation/native-pair-01-host.json`;
queue producer and launcher snapshots: `native-pair-01/`. It runs
`native-nf-control-32to64-01`, then `native-nf-reduced-32to64-01`, each stopping64.
Checkpoint prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/kl-continuation/20260930-pair01/`.
Small evidence prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T043200Z/`.
No automatic extension beyond64 or NFR training.

05:01 UTC: control reached48 with finite updates. Repeated32 evaluation exactly
matched prior NF32:2.960392/7.393494/7.425865/7.435141. At48 it is
2.908634/7.090835/7.124708/7.127553: ordinary continuation improves all passes,
but later-vs-first deficit remains large. Continue authorizedstop64. Cloud40
fully published while training continued. Reduced branch remains queued and
unstarted. No model/source change since launch.

05:22 UTC: control completed all32 additional updates and final64 dev eval:
CE2.845510/6.805770/6.923254/6.967279. All32 finite and clipped; raw norm
min/median/max4.62280/7.79101/14.49341. Later passes improve versus32 but remain
much worse than first. Cloud58 verified; final64 save/retention in progress.
Reduced branch still queued. This is a control endpoint, not yet the paired
intervention result. Exact runtime210 remains frozen.

05:34 UTC: control fully closed (3171.07s executor wall, W&Bsynced,
cloud64verified), independent control audit8,009 passed. Controlreport SHA
`d0b9c32c0f3465d8a9950915cf8a5582c47ee16c6df16347e8c2ed50879708b2`.
Small evidence retained as kl-native-control. Reduced branch launched05:30UTC,
W&B x8f16eqv. Its full origin and raw repeated32 evaluation exactly equal
control; declaredKL.1. Preparing graphs, no numerical failure. Both branches
still use runtime210unchanged, data/LRplan128unchanged, savedAdam inherited.

05:54 UTC: reducedKL.1 reached48, devCE2.714837/6.659169/6.755933/6.782391,
versus matchedcontrol48 2.908634/7.090835/7.124708/7.127553. All four raw CEs
improve directionally; laterpass deficit remains. First update33 raw loss sums,
counts and LR exactly match control, with raw gradientnorm4.587565 vs11.802785.
Cloud40verified; continue to64 as authorized, no extra intervention. Reported
midpoint improvement is not a precision clearance or long-run quality claim.

06:16 UTC: both branches reached64, no further updates queued. ReducedKL.1
finaldevCE2.704137/5.752041/5.883908/5.956001 vscontrol
2.845510/6.805770/6.923254/6.967279. Reduced raw normmin/median/max
1.79871/2.92773/6.65571; all32finite/clipped. Larger laterpass gains remain
~3.05–3.25nats behind its own firstpass. Reducedcloud58verified; final64
retention pending. Pairedaudit, summaryplots and fullinventory run only after
report/worker/W&B close. No nativeRT or newBF16clearance. Strong directional
loss-balance finding, not useful-refinement/quality-campaign completion.

FINAL EXECUTION: native queue completed_pair; both32→64 continuations stopped,
W&Bsynced and cloud64verified. BothH100s independently observed idle (0MiB,
0%utilization, no compute processes). Nativepairedaudit16,139 passed, reportSHA
`d6642b86d4a93f4c5fd729ece4cf3a852abba4dbec28c05eb91be3b77b265d6d`.
Finalreducedreport SHA`bd7fbc420ad5472d3f032b006a4df3bb0c87a4bdbceab14c61bed3761bc67e72`.
125distinctfocusedCPUtests pass; no broad numerical/quality clearance.
Summaryplots/CSVs retained, W&Bsummary72jc2qi3 synced. SummaryJSON SHA
`ff84e19a6f056463b7a43a8a4797b82e85a0e0788a2438f8e0e8b18eb1fb2d72`.
Fullinventoryreport SHA`be70e73b248b001fd5d2bb5c58208f126270b4eb06fbc7c58e5bb63ec37bce0c`:
13checkpointstates+13manifests+11smallarchive/manifestpairs,48countedobjects,
121,856,167,737bytes. Inventoryitself and latercloseout/adminretention are separate;
see storage-receipt.md for self-publication exclusions. All13 publication receipt
bytes snapshotted under neutral names. Runtime210 and original200 files unchanged.
No NFR replication or further training is queued. Next-steps.md proposes reviewed
pairedNFR32→64 KL1/.1 (~3–4h). PR/administrative closeout follows below.

PR54 merged as `da239d0e32ccc747e853dad1a94fa87c97ed2611`; reviewed branch
head `b138f2009a6bdf2ebc460d027407b5d62b103dc7`. No hosted CI checks are
configured/reported for that head; the scoped local/container validation above
is the test authority. Main is synchronized. Small closeout archive113authorities
(code/docs/launcher/recovery records and receipts) verified at `kl-closeout01`;
receipt SHA`26424e2edcb1e2fd343534a6d5f31525a41e60832e947bf9356769fbf30926f3`. Later administrative retention at
`kl-admin01` records this closeout paragraph and merge response; its receipt
will live at `.runtime/olmo-kl-retention/admin-01.json`. No more training queued.
