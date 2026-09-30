# Validation and execution scope

The final four-file CPU suite passes **55 distinct tests**, with no skips, in
11.52 seconds inside the GPU-disabled project container. It covers fixed row
membership/masks/counts, payload ownership/alias/counter validation, unchanged
weights-only import, canonical weighted-loss decomposition and structural zeros,
independent joint reconstruction, exact beta-zero controls, entropy masks and
bounds, and source authorities. Actual NF/NFR diagnostic specifications pass.
The ledger is `.runtime/olmo-feedback-diagnostic/final-cpu-tests-01/report.json`,
SHA256 `5faa0d21a1931f7f1cc284fa7456867084fad9110e444f49a9971f533d4002d6`.

Six native saved-state GPU probes complete on H10080GB in the required container:

| Probe | Scope | Main checks |
| --- | --- | --- |
| NF0 forward02 | Eight dev rows, beta1 | Finite losses/entropy, state preservation |
| NF32 forward01 | Same rows, beta1/0/0.5 | Exact bypass and beta-invariant first pass |
| NFR32 forward01 | Same rows, beta1/0, RT0/15 | Exact bypass and beta-invariant first pass |
| NF32 primary-gradient01 | Two unused training rows | Joint reconstruction1.611e-6 |
| NF32 conditional-gradient01 | Two predeclared additional rows | Joint reconstruction9.872e-7 |
| NFR32 primary-gradient01 | Same first two rows | Joint reconstruction2.463e-6 |

All six preserve weights/buffers, RNG and absent gradient buffers. Every gradient
case passes the global1e-4 decomposition bound and exact structural-zero checks.
Each gradient case computes four existing weighted loss VJPs plus a fifth
independent joint VJP. It makes no optimizer step and does not claim independent
verification of every underlying backward kernel. Existing numerical lineage
remains the authority for those kernels; BF16 qualifications are unchanged.

CPU gradients are transient working data; scalar reports, full fixture bytes and
source snapshots are durable. No new training checkpoint is needed because no
parameter changed. The original NF0/NF32/NFR32 checkpoint authorities remain
immutable and cloud retained.

Two preflight failures are preserved: the older restore utility rejected the
newer async identity, and the initial diagnostic launcher rediscovered two new
model-package files when checking the historical training inventory. Neither
attempt performed a GPU forward or training update. The matching existing pilot
restore succeeded; the diagnostic now explicitly authenticates the accepted
pinned plan and separately pins its observer code.

After GPU completion the two observer modules were mechanically relocated from
`cdrm/pretrained/` to `scripts/olmo_feedback_*` to avoid changing training source
discovery. Relative imports became absolute imports; function bodies are
unchanged. The GPU reports preserve their original source snapshots. The final
packaging receives CPU and original training-loader compatibility checks, rather
than repeating identical GPU mathematics solely for a file move. See progress
for the independent AST/inventory/loader result.

This remains a small FP32/no-jitter saved-weight diagnostic. It does not validate
large-batch BF16 Adam directions, establish useful refinement, prove collapse,
or authorize a changed-objective training continuation.
