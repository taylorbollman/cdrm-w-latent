# O5b endpoint diagnostic: evaluation only

Specified on 2026-09-22 while the frozen FBT learning arm was still running,
after observing its code recovery and feedback-specific retention deficit.
This is **post-hoc development exploration**, separate from the original paired
training comparison. It adds no training updates and changes no checkpoint.

Use the completed FBT endpoint only, after both arms finish and their strict
source/data/exposure/retention checks pass. Verify the full checkpoint checksum,
configuration and model-state metadata. Retain native OLMo, fusion state, BF16
mixed precision, masks and data ordering. RT and NextLat remain off. Confirm
all model/buffer bytes are unchanged after evaluation.

Fixed cases, chosen before their results are observed:

1. First 128 code and retention development windows, maximum length 512:
   two total passes, feedback beta in **0, 0.25, 0.5, 0.75, 1**.
2. First 32 windows of those same splits, truncated to maximum length 64:
   beta in **0.5 and 1**, scored with **2, 3 and 4 total passes**, and exact
   sequential feedback. Beta 0.5 is prespecified; do not replace it with the
   best beta from the larger grid.

Record every finite pass, target counts, original-document identities, exact
modes, current script/core hashes, parent checkpoint receipt, and W&B identity.
Report the full grid, not only the most favorable setting. Full-window and
short-prefix metrics remain separate. The short retention selection contains
only six original documents and cannot establish broad inference quality.

Interpretation:

- Improvement at an intermediate beta would suggest that retaining some native
  embedding input helps the partially adapted feedback model. Beta zero alone
  restoring its ordinary pass is expected and does not establish useful feedback.
- Improvement with more passes or exact online execution would suggest a role
  for finite-pass approximation. Persistence across these modes weakens that
  explanation as the dominant cause of the retention loss.
- These tests can guide a subsequent training experiment, but choosing a beta
  on this development set does not give an unbiased performance estimate.
- Do not add prefix sampling, noise, RT, NextLat, new normalization or additional
  training as part of this diagnostic. Those require a separately stated next
  experiment and comparison.
