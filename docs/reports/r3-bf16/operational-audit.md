# Independent operational JSON audit

All recorded pairing, source/runtime, data, finite-state and dtype checks pass. The retained cold-resume comparison reports exact model/Adam/RNG/data equality; this audit independently confirms identical numerical JSON histories and development metrics. Raw checkpoint tensors are not re-compared here.

| Profile | Train CE: update 1 → 100 | Dev CE: 0 / 50 / 100 | Clipped updates | Final gradient norm | Final update norm |
| --- | --- | --- | ---: | ---: | ---: |
| FP32 | 7.383334 → 6.280227 | 7.424483 / 6.337363 / 6.236422 | 37/100 | 0.885647 | 0.726473 |
| BF16 | 7.382625 → 6.285615 | 7.424376 / 6.337330 / 6.232715 | 37/100 | 0.926009 | 0.720443 |

Mean paired training CE difference (BF16−FP32): -0.000938749; maximum absolute difference: 0.043478; final dev difference: -0.00370729 nats.

Every update retains 100 finite FP32 parameter tensors, 100 finite FP32 gradients and 200 finite FP32 moment tensors. BF16 logits and FP32 CE are observed throughout the BF16 run. Both block 3 Q/KV projection gradient signals remain nonzero in both arms; that supports connectivity but is not an isolated persistent-write proof.

**Norm comparisons concern scalar magnitudes along diverging trajectories.** They are not gradient-vector error norms or direction measurements. This 100-update pair covers only the initial warmup and supports bounded operation, not quality equivalence or unresolved-criterion clearance.

| Benchmark | Mean latency (ms) | Tokens/s | Peak allocated / reserved (MiB) | Adam state (MiB) |
| --- | ---: | ---: | --- | ---: |
| seq-fp32 | 28.975 | 282,722 | 2291.33 / 2488.00 | 76.10 |
| seq-bf16 | 32.576 | 251,470 | 1645.33 / 1808.00 | 76.10 |
| r3-fp32 | 242.757 | 33,746 | 2135.08 / 2386.00 | 76.10 |
| r3-bf16 | 293.393 | 27,922 | 1539.58 / 1716.00 | 76.10 |

Each benchmark uses the same input fixture, matching source/runtime policies, five warmups and 20 measured updates without new compiled graphs. Within each architecture, initial model/optimizer/RNG hashes match. Timing uncertainty is limited to the observed update range; no repeated-job confidence interval is claimed.

SEQ: BF16/FP32 latency ratio 1.1243; allocated-memory reduction 28.19%.
R3: BF16/FP32 latency ratio 1.2086; allocated-memory reduction 27.89%.

The figure and audit preserve the experimental label and make no speed or final-quality promise. Numerical confirmation results remain separate.

[Figure](operational-summary.svg) · [Full audit and source hashes](operational-audit.json)
