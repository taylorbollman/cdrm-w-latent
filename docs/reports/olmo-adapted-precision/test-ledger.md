# Adapted-state precision test ledger

2026-09-29. Sources/protocol frozen at **`9f4693e`** before the GPU probe.
Prior helpers, validators and protocols remain unchanged.

| Check | Result |
| --- | --- |
| Focused CPU tests | **43 passed in 4.69 s**, one test-only scalar-conversion warning; GPU-disabled container |
| GPU diagnostic | `adapted-01`, `passed_operational_diagnostic`, exit 0, 63.210 s; W&B `8ymbic44` synced |
| Precision pair | 2 aggregate CE cases, 4 physical backwards, 0 updates |
| Cold construction | All 14 reference controls pass before import; no extra cold backward |
| Adapted import | All 6 importer checks and 11 matched-fixture/import controls pass |
| Per-case health | All 8 checks pass in each case: finite values, counts, pass coverage, gradient participation, zero predictor gradient and RNG preservation |
| Final integrity | All 7 checks pass: state/configuration/input/RNG/source/reference preservation and restored production flags |
| Independent review | Source snapshots, controls, weight authority and numerical summaries reviewed; 104 source pins verified |

The CPU log is `.runtime/olmo-adapted-precision/cpu-final-01.log`.
The warning comes from comparing a requires-grad test tensor to a Python scalar;
it is not a training or runtime failure. No extra GPU work or large checkpoint
rehash was performed by independent reviewers.

The final report is `.runtime/olmo-adapted-precision/adapted-01/report.json`,
SHA-256 **`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`**.
The cold matrix authority remains
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`;
the historical O5c report remains
`020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.

Actual imported weights are pinned by checkpoint SHA
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
The report records 68 imported state tensors and complete current-state hashes,
preserved predictor hashes, explicit historical/current source mapping and saved
buffer identity. Construction-only `source_checkpoint` metadata must not be
mistaken for actual post-import weight provenance.

The earlier CPU baseline audit is under `baseline-audit-01/`; report SHA
`7070e459561e08fd253cd721001acfc7c316e38b7c6d340d35d0da3a8469e517`.
It checked the 37 historical sources, six explicit differences and 89 cold
source/snapshot pairs without rereading the large checkpoint. The new diagnostic
and independent cloud readback are covered by the [storage receipt](storage-receipt.md).

“Passed” is operational. No new BF16 tolerance was adopted; the descriptive
elementwise comparison still reports `all_parameters_close=false`. Aggregate
gradient agreement improves substantially, but record-0/pass-1 hidden-state
relative L2 remains 12.437%. No full gradient vectors, new trained checkpoint or
training-quality conclusion were produced.
