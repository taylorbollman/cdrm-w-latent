# Activating NextLat and native RT after fusion startup

The update-128 fusion state gives close FP32/BF16 gradients when the actual
NextLat losses are enabled without RT. Enabling full-strength native RT in
layers 0 and 15 brings back substantially larger backbone and fusion gradient
differences. The checkpoint was trained with feedback and CE only, so this is
an abrupt activation of RT and auxiliary losses, not a trained RT endpoint.

The bounded [component protocol](component-probe-protocol.md) completed all six
cases: NF combined loss, NFR CE only, and NFR combined loss, each under full
FP32/math and production BF16/Flash/native Triton RT. Here **N** means NextLat
branches exist, **F** means four feedback passes, and **R** selects RT layers
0 and 15 at strength 1 in every pass. Combined loss means CE + latent prediction
+ KL, with the current pass weights and global denominators. In CE-only cases,
the auxiliary branches still execute but receive zero loss cotangents.

Every case used the same saved fusion checkpoint, original pretrained backbone,
freshly seeded predictor, two physical B2/T128 records, fixed keyed feedback
noise, and isolated-document masks. There were 512 input tokens, 508 CE/latent
positions, and 504 KL triples. All model parameters were trainable for these
gradient measurements. There were **six aggregate backwards, twelve physical
backwards, and zero optimizer updates**; the job took 117.42 seconds, including
observation and hashing. This is not a throughput measurement.

The table reports the L2 norm of BF16 minus FP32 divided by the FP32 gradient
norm, separately within each arm and objective. It does not compare different
objectives against each other as if their raw gradient scales were equal.

| Arm/objective | Backbone difference | Fusion difference | Predictor difference |
| --- | ---: | ---: | ---: |
| NF, CE + latent + KL | 0.735% | 0.888% | 0.763% |
| NFR, CE only | 32.442% | 47.421% | Both zero |
| NFR, CE + latent + KL | 12.059% | 20.389% | 0.820% |

| Arm/objective | Backbone absolute difference / cosine | Fusion absolute difference / cosine |
| --- | ---: | ---: |
| NF, combined | 3.10272 / 0.999977 | 0.191080 / 0.999962 |
| NFR, CE only | 26.9416 / 0.949973 | 5.11692 / 0.888333 |
| NFR, combined | 21.4491 / 0.992726 | 3.72141 / 0.979445 |

The NFR combined result has a lower relative difference than NFR CE-only, but
also a different gradient mixture and larger reference norm. Auxiliary losses
therefore cannot be credited with resolving RT precision behavior from this
comparison. The absolute backbone difference remains 21.45. The predictor's
0.82% difference also does not clear the backbone/fusion computation that feeds
it.

For each physical record and pass, 254 of its 256 positions have nonzero
incoming hidden cotangents in at least one precision. The final position in
each sequence has zero cotangent. Across these supported record/pass aggregates,
the largest hidden/cotangent relative differences were respectively 1.41%/2.98%
for NF combined, 3.75%/83.95% for NFR CE-only, and 3.75%/48.66% for NFR combined.
These summaries exclude zero-support positions only when describing geometry;
the actual objective and backward computation retain the full masks and shapes.

All source, input/noise, checkpoint, state, RNG, mode, trainability and final
runtime-flag checks passed. The 130 retained source snapshots were independently
rehashed. NF combined reproduced the earlier NF CE forward states and raw loss
sums exactly in each precision; NFR combined likewise reproduced the new NFR CE
forward states and raw loss sums. The objective changes only the backward
cotangents. The RT transition preserved every learned tensor and buffer.

This supports continuing bounded BF16 readiness work for adapted feedback plus
NextLat. Full-strength RT still needs investigation. The subsequent
[RT-strength check and ordinary packed calibration](rt-strength-and-baseline-results.md)
are now complete: zero strength retains the native scan backend while removing
recurrence, and 0.25 tests a less abrupt functional change. Neither this result
nor that strength check alone establishes multi-step optimizer
stability, production BF16 clearance, or task quality. The separate
[saved-Adam and packed-context analysis](update-and-packed-results.md) covers
different objectives and data scopes and should retain its own qualifications.

Evidence: `.runtime/olmo-fusion-startup/components-128-01/report.json`, SHA256
`59e850755c25d3e585b1787b3b7a70ec67a43a8a7edfc97ceacd0d6f75a0653d`.
The full online record is [W&B run dyvzbehl](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/dyvzbehl).
Checkpoint SHA256:
`892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc`.
Held-out fixture SHA256:
`830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9`.
