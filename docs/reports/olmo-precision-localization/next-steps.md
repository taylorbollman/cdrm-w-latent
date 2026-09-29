# Decision after numerical localization

The bounded local attention check is complete. It found small local output
and gradient differences relative to the much larger full-model differences.
The purpose of the next step is to identify which model combination is most
sensitive, then select a specific remedy if needed.

## What the existing result does and does not support

The crossed CE experiment gives no reason to change the native RT tile
implementation: eager and Triton tiles agree exactly under ordinary Flash
attention in the tested T16 computation. Replacing Flash with math SDPA changes
the full gradient substantially, but BF16 math also differs substantially from
full FP32. Replacing Flash alone therefore is not an established remedy.

The auxiliary-loss check shows small local BF16 layout differences before
backbone propagation. Their practical effect is not determined by comparing
their error norm with a different full-parameter error norm. Neither small
scalar-loss differences nor exact same-backend restart clears the larger
cross-precision qualification.

## Smallest useful continuation

The completed [fixed-input check](attention-local-protocol.md) found Flash
output errors of 0.160–0.186% and Q/K/V gradient errors of 0.174–1.501% versus
FP32 math on eight sampled sites. It does not establish a large local Flash
backward defect or clear end-to-end precision agreement. The next step should
therefore investigate sensitivity of the assembled computation.

1. Separate the two recurrence mechanisms on the same initial state and
   the original T16 fixture first: ordinary, temporal RT only, K4 FBT only,
   and K4 FBT+RT. Keep the NextLat branches present with zero auxiliary
   cotangents (existing arms N/NR/NF/NFR), preserving the CE helper contract.
   CE is sufficient for this first question because it already exhibits
   the large difference without auxiliary-loss cotangents. Compare BF16 and
   FP32 within each arm with fixed data/noise and explicit objective weights.
   Require the NFR pair to reproduce the saved endpoint metrics and forward
   fingerprints. Record the exact per-arm CE pass weights and target counts.
   Eight aggregate gradient cases suffice for the four precision pairs. Report
   the shared backbone group as well as the full gradient; cross-arm raw norms
   are not direct causal effect estimates because pass weights and active
   fusion differ. Additional inputs are conditional on that result.
   Record a bounded protocol before execution; this is a proposed follow-up,
   not a completed test or an automatically launched sweep.
2. Use that separation to select one suspect precision boundary. Candidates
   depend on the result: feedback fusion, RT attention/projection, or ordinary
   attention. Change one boundary, preserve model equations/weights, and check
   whether it improves both forward and raw-gradient agreement. Keep a failed
   correction in the record. Avoid promoting the entire model without learning
   which operation requires it.
3. Confirm an actual correction on a bounded packed-data T1024 fixture before
   changing a training policy. The current isolated T16 checks do not stand in
   for the campaign's packed data and context length. Recheck graph/restart and
   throughput only if the chosen implementation or precision policy changes.

Q/K normalization remains a possible later intervention, not a correction
justified by the present measurements. The sampled attention logit ranges do
not identify a normalization defect; a maximum attention weight of one is
expected at the first causal position and is not itself evidence of trouble. It changes the checkpoint's function
and needs a separate transition plan. There is no new acceptance threshold,
quality-training authorization or claim of H200 readiness in this milestone.

The implementation, fixture exports and storage receipts should allow these
decisions to survive interruption without repeating the completed bridge and
auxiliary matrix. Resume from this report's final decision and progress entry,
not an older plan that still calls those completed probes pending.
