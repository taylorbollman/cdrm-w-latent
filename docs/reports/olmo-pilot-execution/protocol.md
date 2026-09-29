# Ordered pilot execution acceptance

Connect the accepted PR49 ordered data to the unchanged shared SSD update engine.
New declarations, resolutions and execution identities authenticate corpus,
suite/catalog, named train/dev panels, source inventory, acquisition mapping,
recipe and exclusions. Do not relabel old canonical-order checkpoints or migrate
their cursors. Historical model, kernel, optimizer, storage and graph code remains
frozen; new adapters supply readers, finite plans and evaluation controllers.

Continuous-stream-v1 packing is unchanged: model state continues across true
document boundaries within a row and resets between rows. CE crosses internal
document boundaries; NextLat pairs/triples respect true document identity. No
cross-row prediction, cycling or implicit data-plan extension is allowed.

Development evaluation uses explicit named dev panels, a declared finite prefix
of each panel and separate physical batch sizes. Preserve common-FP32/no-jitter
per-pass evaluation, separate CE/latent/KL denominators, all live model/Adam/RNG/
cursor/gradient state and captured graph buffers. Never pool main/source panel
results as independent samples. Confirmation panels are excluded from routine
development invocation. Data correctness does not establish model quality.

First validate CPU authorities, finite exhaustion, actual masks, independent
evaluation allocation and resume rejection. Then use a separately named tiny
T16 NFR fixture on two H100s: three updates of five rows each, physical B2/rank,
including cross-document boundaries and uneven rank/microbatch allocation.
Compare no-evaluation reference with evaluation inserted after update two.
Require exact update inputs, losses, gradients and complete boundaries despite
the declared evaluation-identity difference. Stop at update two, restore its
verified cloud boundary in a fresh process, and require exact same-lineage
continuation to update three. Evaluate a terminal restored boundary without an
optimizer update to cover lazy graph preparation and independent evaluation.

Every GPU command runs in the required project container with both NCCL async
flags zero and a bounded external timeout. Persistent evidence and online W&B
accompany new SSD checkpoint roots. Retain verified GCS generations before local
pruning; keep two completed checkpoints per owned segment. Preserve all old
checkpoints. Source/input pins remain unchanged through each run and resume.

After integration acceptance, measure native B and combined NFR at T1024 using
the real pilot stream, BF16, CUDA graphs, accepted pointwise/RoPE optimizations
and activation checkpointing. Start combined physical batch near 12/rank with
8/rank fallback. Select a comfortable base batch from bounded measurements.
Separate useful update throughput from setup, data materialization, FP32 dev
evaluation and checkpoint/transfer costs. Count valid inputs, loss targets,
physical microbatches and useful exposure explicitly. Save at completed
boundaries every 20 minutes or sooner; report any indivisible longer operation.

These are operational capacity fixtures, not a matched learning cohort. Original
weights with fresh Adam and the accepted fusion128 weights-only NF/NFR startup
remain distinct. Historical adaptation exposure must remain visible. No new LR
sweep, numerical-clearance claim, Q/K normalization change, H200 acceptance or
quality campaign is implied. Freeze startup, token budget, monitoring membership,
batch/accumulation and cadence before conducting a learning comparison.
