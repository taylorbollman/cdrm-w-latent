# Recurrence/precision milestone progress

2026-09-29. Active branch `feat/olmo-recurrence-precision`, from PR40/main
`4eff982`. The user authorized the next bounded numerical milestone.

Read [protocol](protocol.md) first. Initial scope is eight aggregate CE cases:
N/NR/NF/NFR × full FP32 versus production BF16, original T16/two-B2 fixture.
No training or core-model change. The protocol is drafted before execution;
there is no new GPU result yet.

Root owns GPU execution and retention. Runner agent implements the new helper
and tests; precision reviewer independently checks the arm/observer contracts;
data reviewer verifies prior references and matching controls. Old PR40 helpers
and protocols stay frozen. A conditional precision-boundary follow-up needs a
separate written scope after the matrix identifies a useful target.

Runtime root: `.runtime/olmo-recurrence-precision/`. GPU phases are limited to
900seconds and will use device0. Save/push and retain every20–30minutes.
Prior numerical qualifications remain unresolved until evidence supports a
specific resolution; operational completion alone is not clearance.

## Baseline audit and environment

Both H10080GB devices were verified idle inside the project container. Boot
disk has about400GiB free. NVIDIA catalog was checked; no strong specialized
skill match for this custom numerical probe was found. Existing harnesses
provide its execution and validation controls.

Independent CPU baseline audit passed: four-arm token/mask identities, NF/NFR
noise equality, global counts25/25/21 and29valid inputs; all308 prior source
snapshots and10 locally retained PR40 objects matched. Read
[baseline controls](baseline-and-controls.md). Audit report SHA
`917c5b59376f87ca409da3de7044969c27322d3dcda7ddc5592733d9234beb51`.
New retention namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T053000Z/`.
Baseline audit retention is being verified. Root owns GPU launch.
Initial helper review has no blocker; focused tests are being completed.

## Matrix launched

Runtime source `6ed920a` is committed/pushed; helper and test bytes are frozen.
Independent review found no blocker. Final CPU scope66 passed in7.14seconds
(one known CPU RMSNorm fixture dispatch warning), recorded in
`cpu-diagnostics-01.log`. Stage `matrix-01` is now running on device0 with
900second timeout. Baseline audit cloud receipt is verified. Root owns launch
and retention; independent agents will assess the completed matrix.

## Matrix complete; one conditional boundary selected

`matrix-01` completed8cases/16physical backwards in119.94s; W&Blt54objk.
All79sources,68healthchecks, matchedstates/noise/firstpasses and oldNFRanchors
independently verified. ReportSHA
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`.
Backbone errors: N0.9804%,NR25.4583%,NF60.8698%,NFR95.8461%. Neither recurrence
is an exclusive explanation, and differences are not additive causal effects.

The single conditional follow-up is NF-only fusion FP32, recorded before
execution in [fusion protocol](fusion-protocol.md): two exact matched anchors
plus one candidate,3aggregate CEcases/6physicalbackwards. Only fusion autocast
is disabled; ordinaryFlash/BF16 unchanged. Candidate firstpass must match;
compare later forwardstates andbackbone gradients jointly. If unsuccessful,
retainnegative andstop rather than expandprecisionchanges. Newhelper/tests
ownedby runneragent; root alone launchesafterreview/sourcefreeze.

## Fusion diagnostic launched

Runtime source `a5e9540` is committed and pushed. The new helper and tests are
frozen; independent review found no blocker. Focused CPU checks passed 27 tests
in 4.41 seconds (15 existing recurrence tests and 12 new fusion tests; these
overlap the earlier 66-test scope). No warnings. Evidence is in
`cpu-fusion-01.log`, SHA256
`c364006ec3531723e9997e2c9ac7628409f81c94a4d9a1704f0e0474a0872065`.

Stage `fusion-01` is running on device 0 with a 900-second timeout. W&B run
`wih59gy7`. Baseline and matrix retention receipts are verified. Root owns the
GPU process and retention; the data and precision reviewers are independently
checking the final outcome. No production model code has changed.

## Fusion complete; precision changes stopped

`fusion-01` exited successfully in 38.67 seconds. Report SHA256
`4f98763c43d5e94fc6c5f783712151a2f648f74ea4147b1cf2f6424bba6446a6`.
All three cases, exact NF anchors, first-pass identity, six-call override scope,
15 contract checks and final integrity checks pass. W&B `wih59gy7` is synced.
All 82 source/snapshot pairs were independently checked.

Backbone gradient error improves from 60.87% to 56.16%, but final hidden-state
errors worsen on both records: 12.66% to 14.52%, and 3.228% to 3.273%. Fusion's
own gradient error rises from 65.21% to 67.24%. This is not a joint correction;
retain the negative result and stop the precision changes under the frozen
protocol. No production policy or core model code is changed.

Both GPU stages are complete: 11 aggregate cases / 22 physical backwards,
zero optimizer updates. Baseline, matrix and fusion retention receipts are
verified; independent cloud readback and final documentation are in progress.
The next proposed diagnostic is recorded in [next steps](next-steps.md), not
launched. Root owns final handoff/PR; data reviewer owns results/ledger/storage
audit. Preserve all completed sources, tests, protocols and reports unchanged.

## Independent closeout review

Matrix and fusion source/control audits pass. Independent cloud readback of
three stage receipts verifies six exact-generation objects, 168 inventory
members and 162 archived source snapshots (including the baseline audit's own
source). See [storage receipt](storage-receipt.md). Both GPUs are idle with
0 MiB reported in `postflight-01.log`. Final documentation review found no
blocker; the proposed fixed-boundary VJP diagnostic remains unlaunched.

Final closeout will retain the CPU logs, audit scripts/reports, source files,
documentation and prior receipts. That separate archive is not yet counted in
the three verified stage receipts. Root is preparing it and the PR now.

## Milestone complete

[PR41](https://github.com/taylorbollman/cdrm-w-latent/pull/41) contains the
completed diagnostic and compaction handoff. Final source inventories were
rechecked after documentation work: all 79 matrix and 82 fusion pins match.
The PR has no production-model changes. Focused tests, independent scientific
review and all measured execution/control guards are complete; numerical
qualifications remain open as recorded in the results.

The separate `recurrence-closeout-01` receipt is verified. Independent readback
checks both exact cloud generations and all 50 inventory members. It retains
the audit evidence, CPU logs, source files, documentation and prior receipts.
Its staging inventory also lists three duplicate downloaded archives that
the standard retention suffix filter excludes; each already exists in its
separately verified stage object. No required evidence is missing.

Totals are four receipts, eight listed cloud objects and 218 inventory members.
The closeout archive freezes documentation at `aca880c`, before its own receipt;
the final receipt summary and PR/control state live in Git to avoid a circular
self-retention dependency. No completed source or archive was rewritten.
Both GPUs are idle. The next recommended fixed-boundary VJP diagnostic has not
been launched, and no quality training is queued.
