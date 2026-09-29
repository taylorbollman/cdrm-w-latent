# Held-out packed T1024 NF precision bridge

2026-09-29. Conditional follow-up within the authorized overnight diagnostic
window. Freeze the new data helper/tests, GPU bridge/tests and this protocol
after focused CPU validation. Historical sources and startup protocols remain
unchanged. No training or optimizer is added by this stage.

## Fixed data selection

Reuse the verified prepared Dolma readiness corpus, manifest SHA256
`f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76`,
and startup training manifest
`2316978559b8db357c8d4adf706e41c94f809922171e8fb0ba50dc17ad66cbc0`.
Exclude all four short-fixture and all four long-fixture development documents
before concatenation. Their artifact SHA256 pins are
`5e55ee7bab67bcffb9fb01de48b9a971bae6ecbee59b8b7ebdf189c52600918f`
and `830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9`.
All retained documents must be dev; training and confirmation are excluded.

For each source separately, concatenate the remaining complete dev documents in
canonical corpus order. Divide with stride 1024 and choose that source's first
full chunk containing at least one actual document boundary. Select the first
two eligible source names alphabetically. This policy is fixed before model
execution and never uses loss, gradients or outcomes. It gives two different
sources while keeping the fixture small. Failure to find two eligible sources
is an explicit preparation failure, not permission to choose by outcomes.

The fixture contains two physical B1/T1024 records: 2,048 valid inputs and
**2,046 CE targets**. No padding or dummy rows are needed. Preserve stored EOS,
including embedded EOS as content, and derive boundaries from actual document
identity. CE predicts across within-row document boundaries, including EOS to
the next document's first token. Latent pairs and KL triples require the same
true document. Attention and shifted FBT feedback/noise continue across true
document boundaries within a row. Positions reset per row; targets, feedback
and model state never cross between chunks. No synthetic terminal EOS is added.

Record exact latent/KL counts, excluded boundary pairs/triples, true document
segments/completions, source exposure and slice offsets. The exporter verifies
the complete source inventory against the pinned training manifest, checks
disjointness, and emits a bounded JSON artifact with metadata-based selection
replay, token/mask/document hashes and keyed-noise hashes. Noise logical-update
key stays zero. The loader reconstructs CPU fixtures from this pinned artifact;
no training cursor is advanced. Its whole-file SHA authenticates export-time
corpus/provenance checks; it does not re-open the entire corpus during probes.

The corpus is a seven-source coverage fixture, not a production mixture. These
two deliberately boundary-containing rows are numerical observations, not a
quality evaluation or representative estimate of long-context behavior.

## Saved-state comparison and policy transition

Compare the cold-original backbone/fresh fusion with the saved update-128
fusion startup endpoint. Each state receives the exact same two physical rows,
masks and keyed noise. Each state runs one full-FP32 math/eager case and one
BF16 Flash/Triton case: **four aggregate cases/eight physical backwards total**.
NF K4, beta1, jitter0.02 and CE pass weights `(1/2,1/6,1/6,1/6)` remain fixed.
Global denominators count the two rows' targets once, not once per pass.
Keep all diagnostic parameters trainable and all NextLat loss branches present;
latent/KL cotangents are zero. There is no temporal RT in this bridge.

Load the compact update-128 checkpoint under its original isolated-document
contract with the existing strict loader. Only afterward make an explicit,
recorded policy-only transition of the recipe and NextLat configuration to
`continuous-stream-v1`, with tensor state, parameter ownership, trainability,
modes and RNG preserved. Do not weaken the checkpoint validator or silently
rewrite saved metadata. The cold state undergoes the same policy transition.
No other model, objective, recurrence-mode or configuration migration is allowed.

Measure each state's BF16 gradients against its own matched FP32 reference;
compare those error measurements across states without conflating different
gradient denominators. Include absolute norms, direction/norm ratios,
all-pass forward/cotangent geometry, supported-position aggregates and true
boundary-adjacent positions. Neither zero auxiliary cotangents nor CE-only
success clears actual NextLat auxiliary gradients or NFR.

## Execution and acceptance scope

Use deterministic setup before CUDA, TF32 off, highest FP32 precision, FP32
master weights and autocast cache off. Preserve ordinary activation
checkpointing and existing runtime flags. No DDP, CUDA graphs, optimizer or
throughput sweep. Root alone schedules GPU execution in the project container;
bound each saved-state phase to 900 seconds and preserve a partial failure if
the bound or memory limit is reached. Do not shorten the fixture silently.

CPU tests must cover independent source-stream selection, exclusion of eight
dev documents, literal EOS versus true boundaries, CE/auxiliary/feedback masks,
exact counts, deterministic noise, artifact relocation and corrupted-contract
rejection. Record the source hashes and test results before GPU execution.
Save immutable fixture/report/source evidence and online W&B; retain completed
stages to GCS within the normal 20–30-minute persistence cadence.

Operational finiteness/state integrity is separate from numerical agreement.
No new BF16 acceptance threshold, architecture change or production clearance
follows automatically. Only after inspecting this bridge should a separately
frozen stage add auxiliary losses or multiple native RT layers.
