# Fixed-boundary diagnostic test ledger

2026-09-29. GPU helper, tests and protocol were frozen at `1023d7e` before
execution. Earlier milestone sources and reports remain unchanged.

| Check | Result and scope |
| --- | --- |
| Final focused CPU suite | **31 passed in 3.85 s**: 10 boundary tests, 12 fusion tests and 9 prior attention tests; CPU container only |
| GPU stage | `boundary-01`: `passed_operational_diagnostic`, process exit 0, 147.869 s, W&B `d43pjmvw` synced |
| Full-model anchors | 2 aggregate cases, 4 physical backwards; both original NF anchors reproduce saved metrics, fingerprints, gradient summaries and geometry |
| Local replay | 4 sites × 3 cases = 12 VJPs; no optimizer updates; 104 local health checks pass |
| Endpoint identity | 8/8 A/C outputs exactly reproduce captured FP32/BF16 outputs |
| Observer accounting, per anchor | 2 outer forwards, 8 stack calls, 6 fusion calls, 128 original layer entries and 128 excluded checkpoint-recompute entries |
| Final integrity | All 10 recorded checks pass, including unchanged state, sources, generators and cleaned-up observers |
| Independent CPU audit | 89 source/snapshot pairs, 44 decoded tensor payloads, masks/layouts, common BF16-origin cotangents, reference contracts and all local replay pins verified |
| Independent cloud readback | 1 verified stage receipt, 2 exact-generation objects, 92 inventory members and both archive control files verified; report, fixture and all 89 snapshots match |

The final CPU log is
`.runtime/olmo-boundary-precision/cpu-boundary-final-01.log`. The earlier
preparation log overlaps this suite; it is not an additional independent test
count. No GPU operation ran during the independent audit.

## Frozen evidence

Paths below are relative to `.runtime/olmo-boundary-precision/`.

| Artifact | SHA-256 |
| --- | --- |
| `boundary-01/report.json` | `60432b04554d1eeadc069b9b63f562049be38f23b65317ca4ff60775df559ff1` |
| `boundary-01/boundary-fixture.json` | `a86a1a667ac07c0f2ae6d236f023ff9a86ebbdfbd53691b1209a616e0be584aa` |
| `boundary-audit-01/audit.py` | `c5269bab09ac5d925dadf60da0525bad4300177216c8401a413bef2c6b5437bb` |
| `boundary-audit-01/report.json` | `aefea01a780fbf0c8487f0c5b904fac8ea646e293889cc7f6a2907afab360cb8` |

The fixture is **9,474,929 bytes**, with 44 JSON/base64 tensors covering both
origins at four sites. Its masks, position tensors, actual outputs and full
cotangents are retained along with inputs and layout metadata. The audit stores
raw tensor-byte pins, verifies the original reference and checkpoint identity,
and checks that final report and fixture bytes did not change during inspection.
Its own source snapshot is retained. It does not rerun a model or re-read the
large pretrained weight file.

The prior NF matrix report remains
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`;
the prior fusion report remains
`4f98763c43d5e94fc6c5f783712151a2f648f74ea4147b1cf2f6424bba6446a6`.
Those report summaries, not unavailable full old gradient vectors, are the
historical comparison authority.

## Qualification

“Passed” means the frozen diagnostic executed with its controls intact. It
does not mean BF16 and FP32 gradients agree within an adopted training budget.
The 60.8698% NF backbone discrepancy remains reproduced and unresolved. Local
errors are not additive causal fractions, and whole-stack B/C comparisons
include internal forward precision differences. Only record 0, passes 1 and 3,
were replayed; no RT, optimizer, DDP or graphs were exercised. Storage evidence
and exact cloud-object pins are recorded separately in the storage receipt.
