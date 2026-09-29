# Crossed-state baseline and controls

2026-09-29. The two retained diagonal reports were independently rehashed:

| Reference | SHA-256 |
| --- | --- |
| Cold NF: `.runtime/olmo-recurrence-precision/matrix-01/report.json` | `bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` |
| Adapted NF: `.runtime/olmo-adapted-precision/adapted-01/report.json` | `6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4` |

Both report `passed_operational_diagnostic`. Every shared field in the adapted
report's `cold_construction` matches the matrix's NF state. Its `adapted_state`
preserves the contract, fixture/noise pins, counts, architecture, predictor
configuration, recipe and runtime flags exactly; only backbone/fusion state
changes. Predictor state remains identical. This is a bounded report/pin audit,
not a new checkpoint read, source-history audit, cloud download or model run.

## State assembly

The new cases fill the two missing cells; the diagonals are retained context:

| Combination | Backbone pins | Entire fusion pins | Predictor pins |
| --- | --- | --- | --- |
| Cold / cold, existing | Cold NF | Cold NF | Common fresh predictor |
| **Cold / adapted, new** | Cold NF | Adapted NF | Same common predictor |
| **Adapted / cold, new** | Adapted NF | Cold NF | Same common predictor |
| Adapted / adapted, existing | Adapted NF | Adapted NF | Same common predictor |

For each group, validate exact tensor names, dtypes, shapes and byte hashes
against its designated report. The backbone has 65 parameters; fusion has two
parameters and its scalar `output_scale` buffer; the predictor has four
parameters. Copy values into existing parameter/buffer objects and preserve
trainability, training modes and native tied embedding/readout ownership.
No optimizer, scheduler, cursor or RNG history is restored.

The saved cold/adapted `output_scale` hashes happen to be identical:
`6787dfdb46b09941d9b1d81f0b3f989901aa3365f04269c271b4b45ca805ea52`.
Still swap and verify the **complete** designated fusion state. Never recompute
its scale from the recipient backbone's embedding matrix.

The following compact digests hash each report's complete group pin dictionary,
using JSON `sort_keys=True, separators=(',', ':')`. They summarize recorded
per-tensor pins; they are not direct hashes of concatenated weight bytes.

| Group | Pin-dictionary SHA-256 |
| --- | --- |
| Cold backbone | `8e5e3e67dc20756e527afd19a9448e8693ea0dd1740b400a697b8b36f1eed6f9` |
| Adapted backbone | `df3de064349c036fcc472aec03af08b7a282c322349150bb6979676f5847901f` |
| Cold fusion | `19ad42ff5f28aa76c96b2c4c72db6c1bad987b6a0a15ee90938db2f2220e1043` |
| Adapted fusion | `641377ebf943a74d3bda94c576976c91210d1ef1e97d960dab7d3409ff3ad2f2` |
| Common predictor | `8ac958ebe78524f95244b47824f3179613d39281658a4892f70e0d77ee516f80` |

Actual adapted weight authority remains O5c mixed update512, checkpoint SHA
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
The cold native source SHA is
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
The adapted report's construction-only `source_checkpoint` field is not its
loaded-weight authority; use its explicit `adapted_import` record. The existing
importer and historical validators remain unchanged.

## Fixed comparison and interpretation

Compare FP32/math against production BF16/Flash **within each hybrid**. Preserve
the two physical B2/T16 records, valid lengths `(16,5)` / `(6,2)`, 29 valid
inputs, four documents, global counts CE25/latent25/KL21, exact masks and keyed
noise. NF means K4, beta1, jitter0.02, no selected temporal RT layers, NextLat
branches present with zero auxiliary cotangents. CE weights remain
`(1/2,1/6,1/6,1/6)`; the common predictor seed is `20260921`.

Retain absolute gradient norms/difference norms as well as relative L2, cosine
and norm ratios. Cold versus adapted backbone gradient error was 60.8698%
versus 0.9085%, but adapted record0/pass1 hidden error remained 12.4366%.
Its old per-position tensors were not retained: new hybrid observations cannot
retroactively locate or clear that discrepancy. Distinguish direct CE prediction
positions, feedback eligibility and incoming-gradient support; use the union
of nonzero support across precisions for a common supported-position comparison.
Positions without direct CE can still influence later losses through feedback.

O5c inherited an adapted O5b backbone. These crosses test whether a component's
adapted state transfers to the other state; they are not an additive causal
decomposition or evidence about how training reached those weights. A failed
cross can reflect lost coadaptation. Neither outcome automatically selects a
warmup schedule or clears BF16 for RT, auxiliary objectives, packed T1024,
updates/restarts, CUDA graphs, DDP or training quality.
