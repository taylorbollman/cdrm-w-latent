# Overnight validation ledger

2026-09-29, work still active. These are focused final scopes, not disjoint test
counts. **Do not sum this table:** several suites include the same supporting
tests. Earlier failing attempts remain in the runtime directories; a final
passing scope does not erase the separate GPU qualifications.

| Scope | Final CPU result | Evidence under `.runtime/` |
|---|---:|---|
| Warmup data, compact checkpoint and corrected backward context | 40 passed | `olmo-fusion-startup/cpu-context-fix-01.log` |
| Longer fixture and numerical probe | 36 passed | `olmo-fusion-startup/cpu-long-01.log` |
| Same-history counterfactual Adam step | 55 passed | `olmo-fusion-startup/cpu-update-probe-final-01.log` |
| Packed fixture and initial probe | 25 passed | `olmo-fusion-startup/cpu-packed-final-01.log` |
| Combined-objective component probe | 62 passed | `olmo-fusion-startup/cpu-component-probe-final-01.log` |
| Fusion-only continuation and comparison | 73 passed | `olmo-fusion-startup/cpu-continuation-final-02.log` |
| Ordinary packed baseline | 33 passed | `olmo-fusion-startup/cpu-packed-baseline-01.log` |
| Saved full-NFR sparse/prepared/captured bridge | 63 passed | `olmo-fusion-startup/cpu-packed-bridge-01.log` |
| Full-NFR continuation and checkpoint contract | 54 passed | `olmo-fusion-startup/cpu-nfr-continuation-03.log` |
| Final per-pass evaluation observation | 19 passed | `olmo-fusion-startup/cpu-nfr-per-pass-01.log` |
| Guarded common lifecycle loop | 28 passed | `olmo-campaign-lifecycle/cpu-guarded-01.log` |
| Abrupt-rank failure driver | 11 passed | `olmo-campaign-lifecycle/cpu-rank-failure-01.log` |
| Cloud recovery bundle | 14 passed | `olmo-campaign-lifecycle/cpu-recovery-bundle-03.log` |
| Live evaluation insertion | 38 passed | `olmo-campaign-lifecycle/cpu-eval-insertion-final-03.log` |
| Ordinary pretrained host-loop adapter | 24 passed | `olmo-campaign-lifecycle/cpu-base-loop-final-03.log` |

The four-update NFR helper separately passed 17 focused tests, documented in
[its results](nfr-updates-results.md). The recurrence-strength, adapted-position
and analytical resource scopes retain their own protocols and test evidence;
they are not implied by the table above.

GPU acceptance is separate from CPU tests:

- Warmup checkpoint fresh-process continuation and the fusion-only midpoint
  continuation reproduce their respective reference boundaries exactly.
- The full-NFR paired optimizer run completed eight finite optimizer calls,
  with all four saved endpoints verified in cloud storage. Its numerical
  differences remain reported rather than treated as passing equivalence.
- The saved full-NFR packed bridge completed five configurations and 23
  physical backwards including setup, with no optimizer calls. FP32 semantics
  pass; BF16 prepared eager versus captured losses and all 71 gradient hashes
  are exact. BF16 sparse/prepared retains its elementwise compatibility miss.
- Tiny two-GPU captured training survives a graceful stop, a coordinated
  logging failure, and a deliberate abrupt rank loss by restarting from a
  complete retained checkpoint. Both ranks' following updates match exactly.
- The live-evaluation pair completed successfully; an independent audit is
  being finalized. Full-NFR continuations and ordinary pretrained host-loop
  acceptance are still pending at the time of this entry.

Historical failures worth retaining include the initial warmup runner exiting
its forced attention context before checkpoint recomputation, retention work
advancing rank0 Python RNG, and exception-held graph owners delaying teardown.
The corrections live in new or explicitly superseding diagnostic drivers;
completed sources and the original failure reports stay available.

No new core model equations, precision defaults, Q/K normalization, optimizer
hyperparameters or acceptance budgets were adopted by these checks. An
operational pass, a numerical qualification and a deliberately failed fault
stage are distinct outcomes throughout the reports.
