# Validation ledger

CPU work ran inside the project container with GPU passthrough disabled.

| Scope | Result | Evidence |
| --- | --- | --- |
| Optimizer-history helper plus frozen endpoint comparison | 37 passed / 4.57 s | `.runtime/olmo-optimizer-history/cpu-01/pytest.log` |
| Per-pass evaluator and preservation | 28 passed / 3.18 s | `.runtime/olmo-campaign-evaluation/cpu-evaluator-02.log` |
| Global reduction, dev authority, CLI and all-eight identities | 60 passed / 8.12 s | `.runtime/olmo-campaign-evaluation/cpu-control-cli-01.log` |
| Independent evaluator audit plus frozen execution auditor | 71 passed / 0.37 s | `.runtime/olmo-campaign-eval-audit-cpu-02.log` |

The evaluator tests include literal CE/SmoothL1/KL eligibility and arithmetic,
dummy-only local batches, disabled terms, exceptional cleanup and exact next
prepared CPU updates. Controller tests cover uneven denominators, invalid
metadata, immutable dev membership, declared schedules and source/startup
identity separation. The optimizer tests compare streamed candidate updates
with literal full multi-group Adam using actual clipping.

GPU optimizer-history evidence is closed and retained: two aggregate/four
physical backwards, six conceptual candidate updates, zero live training
updates. All 11 final integrity checks pass. Evaluation-enabled GPU acceptance
is tracked in progress.md until closed. Tiny insertion independently passes
1,787 checks, including exact subsequent training updates. These four CPU
scopes total 196 distinct tests; no repeated runs are added to that total.
