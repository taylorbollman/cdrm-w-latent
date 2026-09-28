# Combined OLMo T1024 evidence summary

Historical T512 and optional T2048 references are explicitly selected; neither is a new or interleaved measurement.
SDPA only. No FA4 comparison/adoption or numerical equivalence claim.
K2 ordinary bootstrap + feedback with native RT0/15; NextLat on both passes. Tokens count input once.
Pooled rates are total timed tokens / total timed wall seconds, not averages of rates.
B2 diagnostic and every failed/incomplete report stay in the ledger but not performance figures.
Saved receipt verification is not a fresh cloud request; pending retention does not invalidate measured performance.
Free memory is sampled, not continuous; steady allocated omits reserved graph pools.
Equal input tokens does not equal physical batch, recurrence depth or model quality.
Context selection precedes a future RT screen: compare combined against matched FBT + NextLat without RT.
A possible 500M-token continuation plus SFT is planning context, not launched or evaluated by this audit.

| Origin | T / B | Runs | Input tokens/s | Setup allocated / reserved GiB | Steady reserved / sampled free GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical_T512_reference | 512 / 128 | 1 | 12,361.82 | 58.095 / 65.113 | 65.113 / 12.891 |
| new_T1024_experiment | 1024 / 32 | 1 | 9,823.08 | 38.921 / 41.123 | 41.123 / 36.545 |
| new_T1024_experiment | 1024 / 64 | 2 | 11,092.06 | 58.091 / 68.445 | 68.445 / 9.195 |
| historical_T2048_reference | 2048 / 32 | 2 | 9,184.55 | 58.087 / 69.014 | 69.014 / 7.914 |

New-run totals: `{"new_dependency_pairs": 0, "new_final_reports": 3, "new_operational_checks": 15, "new_operational_checks_passed": 15, "new_physical_updates": 24, "new_plot_eligible": 3, "new_reports": 3, "new_source_pairs": 489, "new_statuses": {"passed": 3}, "new_verified_retention": 3}`
