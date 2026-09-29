# Adapted-state hidden discrepancy localized

The replay reproduces the retained adapted O5c result exactly within the saved
evidence scope. Its **12.4366%** hidden-state relative L2 discrepancy at record 0,
pass index 1 is concentrated in the two terminal outputs. Those outputs have
exactly zero total incoming loss gradient in both precisions. On the other 19
valid positions, the hidden discrepancy is **0.659623%**.

This explains why the large aggregate hidden difference can coexist with the
much smaller adapted-state parameter-gradient discrepancies. It narrows this
particular concern; it neither identifies the numerical cause of the terminal
output difference nor establishes general BF16 safety.

## Replay verification

`adapted-position-01` completed two aggregate cases/four physical backwards in
86.97 seconds, with no optimizer updates. Independent CPU audit passed all ten
anchor comparisons across the two precision rows: exact losses/counts, all-pass
forward fingerprints, gradient group summaries, forward/cotangent geometry, and
BF16 gradient/per-parameter comparison summaries. State, fixture, recipe,
trainability contract, runtime and determinism match the previous adapted
diagnostic. All 116 source snapshots and current source pins matched; import
and final integrity checks passed.

This is the same retained discrepancy, not a newly substituted gradient
comparison. Full historical gradient vectors were not saved, so the replay
claim is exact retained summaries and forward hashes, not cross-run
gradient-vector byte identity. The backbone/fusion gradient relative L2 values
remain **0.908519% / 1.405077%**.

## Record 0, pass index 1

All indices below are zero-based. This record contains valid lengths 16 and 5.
Both terminal outputs receive previous-pass feedback but have no direct CE
target and cannot supply feedback to a later valid token.

| Position subset | Positions | Hidden difference norm | FP32 hidden norm | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| All valid | 21 | 25.7915 | 207.3833 | 12.4366% |
| Nonzero incoming cotangent in either precision | 19 | 1.30118 | 197.2610 | 0.659623% |
| Zero incoming cotangent in both | 2 | 25.7586 | 63.9996 | 40.2481% |
| Row 1, position 4: terminal output | 1 | 25.6649 | 45.2547 | 56.7121% |
| Row 0, position 15: terminal output | 1 | 2.19485 | 45.2543 | 4.85003% |

The two terminal positions account for **99.7455% of squared hidden difference**;
row 1, position 4 alone accounts for **99.0213%**. Their reference norms are
ordinary-sized, so the effect is not a tiny-denominator artifact. The largest
supported position is row 0, position 13: difference norm 0.919746 against
reference norm 45.2547, or 2.03238% relative L2. The supported aggregate incoming
cotangent difference is 0.527149%.

Across all eight record/pass combinations, supported hidden relative L2 ranges
from 0.543350% to 1.602659%. These support measurements describe the already
executed backward; nothing was removed from its objective or gradient. Zero
observed cotangent at a terminal output in this short isolated CE fixture does
not justify ignoring similar activation differences at positions used by a
longer sequence, a different objective or generation.

## Separate cold-state context

The initial startup probe also includes four real held-out document prefixes.
At cold weights, their backbone/fusion gradient discrepancies are
**24.3337% / 21.9682%**, compared with 60.8698% / 65.2122% on the original cold
fixture. These are different data observations, not improvements caused by
training.

The fresh fixture's record 0 has supported hidden errors of 8.35378% at pass 2
and 8.15698% at pass 3, close to its all-valid errors of 8.11176% and 8.14785%.
Thus the cold-state sensitivity remains on positions carrying loss gradients;
the terminal-output explanation for the adapted spike does not dismiss the
cold-start issue. This supports completing the planned cold-backbone fusion
warmup and matched original/fresh probes without changing their schedule.

## Evidence

- [W&B adapted replay](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mw8cssi9).
- Replay report: `.runtime/olmo-fusion-startup/adapted-position-01/report.json`,
  SHA256 `7346def87b40605af37583a86bbd4c1ade1c835d7fbadae7f9a1ff45325e9132`.
- Prior adapted report SHA256:
  `6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.
- Initial probe report SHA256:
  `a3e4ee76aefa6db1819c42627000bf1d1459e0164a2a410113765004d4770626`.
- Independent GPU-disabled stdlib audit:
  `.runtime/olmo-fusion-startup/adapted-position-audit-01/`; report SHA256
  `7a980a42707ebee9210ae64cef59f953d39ab31dfa782c64c39d0950ab744358`,
  reusable script SHA256
  `d71d72dcef1720620a2de0eca2c5d0875ac55e65ee9b2d30b9cef1e99e5bb326`.

No model code, original protocol, checkpoint or acceptance threshold changed.
