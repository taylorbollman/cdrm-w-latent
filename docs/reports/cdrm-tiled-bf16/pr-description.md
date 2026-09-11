# Add an opt-in tiled CDRM scan with explicit BF16 precision and validation

CDRM previously supported only a strict FP32 naive scan. Add a tiled side scan
for the five-block profile with one fabric attached after ordinary blocks 1 and
3. Each ordinary block still executes once; the scan shares block 1's canonical
parameters and bridges into block 4. The tiled backend initially supports rho
one. Unsupported execution settings fail explicitly.

The custom first-order backward propagates future-reader credit through each
permanent write, combines it with direct proposed-memory credit, and returns
input and canonical parameter gradients to outer autograd. Candidate/QKV
preprocessing and the bridge remain ordinary autograd. Dense parameter VJPs are
batched; no nested parameter `.backward()` side effects or duplicate parameter
owners are introduced.

An explicit `bf16_fp32_state` policy uses BF16 dense operations/projected storage
with FP32 residuals, normalization arithmetic, attention/temporal accumulators,
master parameters, gradients and Adam state. The naive backend remains strict
FP32. Add the five-block configuration, numerical and operational harnesses,
W&B reporting, standalone plots, source/data/checkpoint identities and usage
documentation.

Validation includes 23 new tiled tests in the H100 container, existing CDRM/HF
compatibility tests, CPU foundation/checkpoint tests and five operational tests.
The local tests cover independent FP64 composition, irregular tile boundaries,
causality, read-conditioned writes, earlier-deep temporal credit, lambda-zero
SEQ equivalence, shared gradient summation, frozen inputs/owners and functional
autograd interfaces. B2 and physical B64 comparisons use native selective-copy
V16/T256/K96 labels, actual CE and an isolated unnormalized proposed-memory
probe. Paired 100-update FP32/BF16 runs remain stable; midpoint BF16 recovery is
bitwise exact in the retained runtime/cache.

**This is a scoped opt-in recommendation, not an all-screens-pass claim.** Strict
FP32 coordinate flags on the raw side probe persist at rounding scale in both
FP32 implementations versus independent FP64. One trained BF16 global gradient
screen also fails: 2.306% observed versus 2.148% allowed including the floor.
All individual-tensor, maximum, isolated-memory, loss and optimizer guards pass.
The lambda-zero control gives a similar 2.298% error, and an FP32-head diagnostic
reduces it to 1.203%. The production policy and frozen thresholds remain
unchanged; this evidence supports ordinary mixed-precision sensitivity rather
than a CDRM backward defect. The optional FP32-head policy is deferred for a
separate explicit validation decision.

See [results and numerical disposition](results.md), the
[frozen contract](validation-contract.md), [usage](../../cdrm-tiled-bf16-usage.md)
and [W&B project](https://wandb.ai/taylorbollman/cdrm-tiled-bf16-validation).
The short training runs have low accuracy and establish operation only.
Backward still materializes quadratic attention storage; warmed precision
benchmarks do not measure tiling speedup against naive or CDRM cost versus SEQ.
General rho, multiple fabrics, packing, cached decoding, higher derivatives,
distributed execution and long research comparisons are outside this PR.
