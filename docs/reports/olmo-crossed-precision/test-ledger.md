# Crossed-state precision test ledger

2026-09-29. Sources and protocol frozen at **`3364376`** before GPU execution.
Previous import/diagnostic helpers and protocols remain unchanged.

| Check | Result |
| --- | --- |
| Final focused CPU suite | **58 passed in 3.63 s**, one existing test-only scalar-conversion warning; GPU-disabled container |
| GPU stage | `crossed-01`, `passed_operational_diagnostic`, exit 0, 177.394 s; W&B `cwxjdnwe` synced |
| New runs | 2 hybrids × 2 precisions = 4 aggregate cases / 8 physical backwards; 0 updates |
| Reference controls | 19 cold/reference and 13 adapted/import checks pass |
| Assembly, each hybrid | 8 exact component/ownership checks, 11 fixed-contract checks and 5 post-run integrity checks pass |
| Per-case health | All 8 checks pass for all 4 cases, including finite gradients, counts, pass coverage, participation and zero predictor gradients |
| First-pass control | All 4 cases exactly reproduce both physical records' matching-backbone diagonal first-pass fingerprints |
| Final integrity | All 7 recorded checks pass |
| Independent readback/review | 109 source pins/snapshots, selected component-state pins, first-pass identities and 464 valid-position mask/support records verified |

Final CPU log: `.runtime/olmo-crossed-precision/cpu-final-02.log`, SHA-256
`db19130ceafc6a1f02809573764c9cc308eefc7e0effeda9370b3e3ce17aa37d`.
The earlier `cpu-final-01.log` (58 passed in 3.73 s) overlaps this scope; the final
rerun includes the frozen graphable position summaries. Do not sum these tests.
The warning is from
an existing import test converting a requires-grad tensor to a scalar for an
assertion, not a GPU runtime failure. Assembly review confirms independent CPU
component snapshots, validation of both components before mutation, `assign=False`
copying, complete fusion-buffer handling and preserved parameter ownership.

Final report: `.runtime/olmo-crossed-precision/crossed-01/report.json`, SHA-256
`ba7989f586e19937ff6c64d7f3c65306eae9f61440b4fedb3942bdd3ff18e298`.
Cold/adapted reference SHAs remain
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` and
`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.
The [baseline note](baseline-and-controls.md) records component weight authority.

The reused CPU-only exact-generation archive audit also verifies all 111
inventory members and both archive controls. It checks the report's actual
hybrid group pins and first-pass fingerprints against the immutable references,
and verifies direct-CE/feedback masks and union-support flags against the known
record lengths. No model execution, large weight read or new historical-source
sweep was added. Audit/object pins are in the [storage receipt](storage-receipt.md).

Operational success is not BF16 numerical clearance. Descriptive elementwise
comparisons are not acceptance gates; aggregate relative/absolute results and
the unresolved prior AA hidden-state spike remain explicitly reported.
