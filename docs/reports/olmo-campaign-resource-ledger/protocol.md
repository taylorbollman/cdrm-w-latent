# CPU campaign resource ledger

2026-09-29. Independent readiness work while numerical startup is investigated.
Only a new script, focused tests and these documents change; previous runtime,
model, objective and checkpoint sources remain unchanged. No GPU work is part
of this task.

Use the exact first update from packed `pretrained-write-02`, SHA256
`a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754`.
Reconcile each rank's physical rows, slots, dummy rows, valid/padded tokens and
CE/latent/KL counts against the global update. This is T1024, B12 per rank, two
ranks, 22 slots each. No checkpoint or corpus download, tokenization or model
training is needed.

At this fixed physical footprint produce cards for all eight B/N/F/R/NF/NR/FR/NFR
arms, K4 where feedback is enabled, selected RT layers0/15 on every applicable
pass. These are accounting counterfactuals, not recommended physical batch sizes
or measurements of the other seven arms.

Construct real CPU modules and fresh scalar AdamW ownership without executing
the model or optimizer. One random native backbone is reused; no pretrained
weights are read. Count unique registered/resident, trainable, optimizer-owned,
branch-active and deployment parameters, including dormant fusion residency and
tied readout identity. Nominal parameter bytes are not a device-memory estimate.

Separate useful selected targets from executed matrix positions. Dynamic losses
execute CE/predictor at every allocated pair and KL at every allocated triple,
including padded/dummy rows. Sparse loss counts use selected CE, same-document
latent/KL masks and their predictor-source union. In this pinned packed workload
every latent pair is supervised, so all KL sources are already in that union.
Tests also cover independently masked cases where adding counts or taking their
maximum would be wrong.

Reuse the existing fully trainable matrix accounting, including ordinary
checkpoint ranges, native RT recomputation and KV-only writers. Extend only the
loss-position accounting in the new tool; no old estimator behavior changes.
Sum ranks and heterogeneous slots once. Dense backbone arithmetic uses allocated
tokens, never just valid tokens. Positive loss coefficients do not divide work.
Explicitly exclude pointwise operations, optimizer/communication, hardware tile
padding, graph preparation and launch overhead. This ledger does not describe
frozen-backbone warmup, issued hardware instructions, throughput, MFU or quality.

Acceptance: tiny actual ownership and all-pass invocation checks, independently
observed sparse/dense loss projection/predictor shapes, exact equality with old
full-row estimates, masked/dummy/heterogeneous accounting tests, exact input and
source hashes, all-eight-arm complete JSON. Run only in the CPU container with
GPU passthrough disabled. Retain sources and report under the persistent project.
