# Add experimental BF16 autocast policy for recurrent attention

The legacy autocast path rounded reconstructed attention and temporal-gradient
intermediates to BF16, producing additional naive/tiled gradient discrepancies.
Add an opt-in `bf16_fp32_state` policy with BF16 dense projections and FP32
recurrent attention/adjoint arithmetic. Preserve FP32 trainable parameters,
final gradients, Adam moments, residuals, architecture, loss semantics and the
legacy default. Explicit autocast metadata controls dense forward replay in
the custom backward; standard backward stays outside outer autocast.

Add actual-loss numerical triangulation, dtype/write-credit probes, paired
training/recovery and synchronized benchmark harnesses, a resolved experimental
profile, and retained evidence. The supporting Stage B and FP32-validation code
already present in the workspace is prerequisite infrastructure; do not present
it as a new BF16 architecture change when preparing repository commits.

Validation: 25 targeted CPU tests passed with one GPU-only skip; five targeted
CUDA tests passed. Actual B2/B64 initialization/trained checks cover all 100
parameter and two retained input gradients. A paired 100-update R3 comparison
maintains finite FP32 state; fresh-process midpoint recovery is bitwise exact.
Original artifacts remain unchanged and the new lineage is archived separately.

Full numerical clearance is **incomplete**: seven initialization B64 tensors
fail the frozen maximum-error budget, despite passing relative-L2 and added-L2
budgets. BF16 reduces peak allocated R3 memory by 27.9% but increases update
latency by 20.9% at B64/T128. Keep FP32 as the default. See [the evidence
report](results.md) for exact identities, remaining flags and untested scope.
