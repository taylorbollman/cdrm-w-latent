# Component execution test ledger

2026-09-29. Candidate training sources frozen at `febc312` after independent
construction and lifecycle reviews. No historical execution, model, kernel,
checkpoint implementation, test or protocol was changed.

| CPU scope | Result | Local log |
| --- | --- | --- |
| Startup/recovery contract and unchanged manifest | 70 passed (43 new + 27 existing) | `.runtime/olmo-campaign-execution/cpu-contract-01/pytest-final.log` |
| Shared engine ownership/packing/transitions | 29 passed | `.runtime/olmo-campaign-execution-cpu-engine-01.log` |
| Observer with actual all-eight CPU/Gloo updates | 20 passed | `.runtime/olmo-campaign-execution-cpu-observer-final-03.log` |
| Launcher and engine after declared constructor fixes | 49 passed (20 launcher + same 29 engine) | `.runtime/olmo-campaign-execution/cpu-cli-engine-04.log` |
| Actual Adam/token-scheduler clocks | 18 passed | `.runtime/olmo-campaign-execution/cpu-clocks-01.log` |
| Generic cloud restore helper | 41 passed | `.runtime/olmo-campaign-execution/cpu-restore-01/pytest-raw-sdk.log` |

Counts across repeated engine runs must not be added as distinct tests. The
training candidate has 130 distinct new CPU tests plus 27 unchanged manifest
tests. Restore helper tests belong to its separate operational source set.

The initial observer fixture failed because its data did not match its declared
packed policy; the fixture was corrected before acceptance. Initial launcher
collection failed on a test-module import, also corrected before acceptance.
Review caught the required generic checkpoint fingerprint field, two omitted
tiny constructor flags, and missing explicit scheduler/Adam-to-counter checks.
The corresponding fixes and regression checks precede all GPU launches.

`contract-authority-01` authenticates the historical fusion128 checkpoint/report
and all116 historical source pins on CPU, explicitly forbidding tensor loading.
`declarations-01` is superseded preflight-only evidence, never launched.
`declarations-02` resolves the final155-source training candidate and actual
NF/NFR startup plan. It plans9,216 valid inputs,9,207 CE/latent positions and9,198
KL triples across three updates, with no source-document boundary in this short
native prefix. Independent CPU fixture tests cover real document boundaries.

The actual tiny cloud restore used the unchanged helper/protocol with the earlier
40-test snapshot. One later test-only refinement adds the production SDK
RawDownload retry oracle; it does not change helper behavior or training identity.
Both byte-download modes pass. Earlier logs/snapshots remain retained.

GPU acceptance stages and independent audits are complete, as recorded below.

Independent stdlib JSON auditor:44 CPU tests pass in
`.runtime/olmo-campaign-execution-cpu-audit-03.log`. Frozen auditor checks both
training source snapshots, enabled loss/global/physical accounting, all raw
active gradients and rank equality where observed, complete state and per-rank
RNG boundaries, data cursors, startup/final clocks and exact overlapping updates.
It is evidence consistency, not a fresh tensor reload or FP32/BF16 comparison.

| Tiny GPU pair against reference | Independent audit |
| --- | --- |
| Lean stop after update1 | 1,101 checks pass |
| Lean full three updates | 1,249 checks pass |
| Cloud-restored update1, acceptance resume2/3 | 1,236 checks pass |
| Completed update3 resume with no graph preparation | 1,030 checks pass |

Each comparison verified155 source pins for each report. Counts include routine
metadata/identity checks; they are not independent numerical experiments. The
same-precision scientific evidence is exact, without a tolerance or omitted
comparison field. Native acceptance is recorded below.

Final integrated CPU scope: **242 distinct tests passed in30.06s** in one run
across the seven new test modules and unchanged manifest tests. This is215 new
focused tests plus27 existing manifest tests. Stage`cpu-integrated-01` verifies
15 scoped live/source-snapshot pairs before and after the run. Only two existing
Google/grpc future-compatibility warnings occurred; no GPU or cloud operation.

| Native adapted NFR T1024 pair | Independent audit |
| --- | --- |
| Reference versus lean stop after update 1 | 1,101 checks pass |
| Reference versus cloud-restored updates 2/3 | 1,236 checks pass |

Native stages `native-stop-01`, `native-reference-01` and `native-resume-01` all
exit successfully and sync W&B. The streamed `native-restored-01` checkpoint
loads model/Adam/scheduler/RNG/cursor before DDP/capture. Preparation preserves
the entire boundary; every resumed input, all 71 raw active parameter gradients,
loss, optimizer state, RNG and cursor match exactly on both ranks. Final counters
are three updates, 9,216 inputs, 9,207 CE/latent targets and 9,198 KL triples.

Final native comparison report SHA256:
`4e3d5fc9309ef8aa6ff376af246fab672ec4224a76278e4ca5b0508f0c263221`.
Reference report SHA256:
`362d0b1fb058e1287980a088272275f3daf51ba71ed025b538e7a7d743c518c7`.
Resumed report SHA256:
`350f7a501097cf09faa00bfb78304ce2d5f70391ad59cbce66539042c96d044d`.
Native source snapshots each verify all 155 frozen execution files. No further
GPU numerical sweep, full-model all-eight GPU campaign or quality test was run.
