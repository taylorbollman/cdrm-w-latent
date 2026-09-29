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
