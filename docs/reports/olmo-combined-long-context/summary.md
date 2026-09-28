# Combined OLMo T2048 evidence summary

Historical T512 is explicitly selected; it is not a new or interleaved measurement.
SDPA only. No FA4 comparison/adoption or numerical equivalence claim.
K2 ordinary bootstrap + feedback with native RT0/15; NextLat on both passes. Tokens count input once.
Pooled rates are total timed tokens / total timed wall seconds, not averages of rates.
B2 diagnostic and every failed/incomplete report stay in the ledger but not performance figures.
Saved receipt verification is not a fresh cloud request; pending retention does not invalidate measured performance.
Free memory is sampled, not continuous; steady allocated omits reserved graph pools.
Equal input tokens does not equal physical batch, recurrence depth or model quality.

| Origin | T / B | Runs | Input tokens/s | Setup allocated / reserved GiB | Steady reserved / sampled free GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical_T512_reference | 512 / 128 | 1 | 12,361.82 | 58.095 / 65.113 | 65.113 / 12.891 |
| new_T2048_experiment | 2048 / 16 | 1 | 7,501.52 | 38.918 / 41.227 | 40.986 / 35.977 |
| new_T2048_experiment | 2048 / 32 | 2 | 9,184.55 | 58.087 / 69.014 | 69.014 / 7.914 |

New-run totals: `{"new_dependency_pairs": 0, "new_final_reports": 4, "new_operational_checks": 20, "new_operational_checks_passed": 20, "new_physical_updates": 32, "new_plot_eligible": 3, "new_reports": 4, "new_source_pairs": 652, "new_statuses": {"passed": 4}, "new_verified_retention": 4}`
