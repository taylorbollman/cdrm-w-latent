# Five-block tiled CDRM: prospective numerical contract

Frozen candidate-independent engineering screens, 2026-09-07. This contract
precedes fresh five-block B64 confirmation. It adapts the resolved R3 validation
to CDRM and keeps local derivative correctness separate from whole-model
precision suitability. It does not claim to reproduce the original authors'
unknown numerical test tolerances.

The model has five ordinary blocks, early site 1 and late site 3, D128/H16,
full MHA, MLP512/GELU, learned pre-normalization and Q/K normalization, ALiBi,
no dropout, epsilon 0.1, rho 1 and bridge lambda 0.01. Only the side scan is
tiled. Owner parameters retain canonical names and a single optimizer owner.
The intended mixed policy uses BF16 dense operations and projected storage
with FP32 residuals, adapter differences and normalizations, attention state,
temporal adjoints, master parameters, final gradients and Adam state.

## Inputs and three comparisons

Use the saved new five-block configuration and starting weights as authoritative.
Six-block checkpoints are not interchangeable with this model. Three arms
share weights, actual data and optimizer moments: naive CDRM FP32, tiled CDRM
FP32, and tiled CDRM BF16. Do not tune the optimization or architecture per arm.

Use native selective-copying V16/T256 with 96 copied symbols. Apply the native
MAD training labels without another shift; retain separate answer-label metrics.
The manifest and saved labels decide which tokens participate in loss. Never
substitute an all-token or independently invented answer-only loss.

Development diagnosis uses B2 from regular development data. Fresh B64
confirmation uses a separately generated 128-example set, seed 925703,
stored as the `dev` split under `data/confirmation`. Initialization uses examples
0–63; the retained five-block trained checkpoint uses examples 64–127. The
training generator seed is 925701 and development generator seed is 925702.
These confirmation examples must not be used by training or candidate tuning.
Record actual manifest, array, configuration, source, checkpoint and criteria
hashes. Both confirmations use physical B64, without accumulation.

## Implementation correctness before mixed precision

At B1–2, widths 16–32 and T1,5,16,17, compare tiled FP32 with naive FP32 and
the existing independent FP64 composition. Check candidate, proposed state
hat_m, rho-one memory, bridge, both independent preview inputs, every canonical
shared parameter and both adapters. Require every mathematically used gradient
to be present and finite; an unused terminal permanent write need not receive
a consumer gradient. Debug tensors returned by a custom function are not
evidence of graph-connected per-token memory records.

The required local FP32 comparison is elementwise
`abs(actual-reference) <= 2e-6 + 2e-5*abs(reference)`.
Record relative L2 and maximum/reference-RMS diagnostics as well. A fixed
maximum/RMS diagnostic can flag harmless FP32 reordering when the reference
RMS is small even though every elementwise check passes; such a flag is
retained and attributed with FP64 when needed, rather than turned into an
additional universal hard gate or silently deleted. Any elementwise failure,
missing dependency or systematic discrepancy requires investigation.

Strict semantic tests require causality, independent-example/call memory,
read-conditioned writes, and later-token credit into earlier **independent**
deep-preview leaves. The latter must not be inferred from an ordinary model
hidden state, where the preview itself supplies another temporal path. Test
lambda-zero equivalence to SEQ, shared-owner gradients equal to preview plus
untied side gradients, `torch.autograd.grad` without parameter `.grad` side
effects, and frozen-owner/input combinations. No tolerance excuses wrong
ownership, a duplicated contribution, or omitted future-reader credit.

## Unscaled side-branch checks

Lambda 0.01 must not hide an incorrect side scan. Save and compare `hat_m`
directly, and compare `bridge_correction/lambda` when lambda is nonzero.
Use identical detached early/deep preview inputs and a saved, unnormalized
incoming hat_m cotangent to form an independent side-gradient packet. Include
both input leaves, the owner and deep adapter. The bridge adapter is correctly
unused by a direct hat_m probe; its gradient is checked by the actual bridge/loss
comparison. Record the actual branch gradients separately from full-model
gradients, whose direct preview path can dominate.

For BF16, require finite FP32 gradients and fixed-forward power-of-two
homogeneity at 1/32 and 32. Keep the same graph and normalize the resulting
gradients; this tests backward scaling without changing BF16 forward operands.
Matched-operand FP32/FP64 dense contractions localize accumulation or cast
differences if necessary. A full FP32 forward pass is not a fixed-operand
derivative oracle for a differently rounded BF16 forward pass.

## Prospective B64 BF16 engineering screens

Let eps=2^-7, reference g be tiled FP32, error e be tiled BF16 minus tiled FP32,
and f_i=2e-6+2e-5*abs(g_i). Reductions use detached CPU FP64 values; no tensor
or zero-reference coordinates are omitted.

| Quantity | Required screen |
|---|---|
| Every intended gradient and optimizer state | Correct shape, present where mathematically used, finite FP32 |
| All canonical parameter gradients concatenated once | norm(e) <= 2*eps*norm(g)+norm(f), approximately 1.56% |
| Each parameter and each independent input gradient | norm(e) <= 4*eps*norm(g)+norm(f), approximately 3.13% |
| Maximum error of each gradient tensor | maxabs(e) <= 8*eps*maxabs(g)+max(f), approximately 6.25% of that tensor's maximum |
| Logits | norm(e) <= 2*eps*norm(g)+norm(f) |
| Same-state native mean CE | Absolute difference <=0.01 nats |
| Unscaled hat_m and bridge correction | Per-tensor 4*eps L2 and 8*eps maximum screens above |
| Independent direct-hat_m side-gradient packet | The same global and per-tensor gradient screens, excluding the mathematically unused bridge adapter |

These are pragmatic engineering guards with known precision scale, not
floating-point error theorems or paper-derived acceptance bounds. They are
chosen before these new B64 results. Retain maxima, top coordinates, reference
RMS/active support, signs, mean signed error, and near-zero error energy even
when the guards pass. Existing R3 backend-distance budgets do not apply to
this different CDRM comparison. Naive BF16 is not an accuracy target.

If a guard fails, preserve the failed result and investigate the cause. Do not
increase thresholds or replace confirmation examples in response to failures.
A numerical engineering recommendation may be qualified by explained
floating-point behavior, but must remain distinct from a claim that every
frozen screen passed. Local correctness, branch evidence and operational
behavior must support the recommendation together.

## Optimizer, bounded operation and scope

Use identical AdamW states and clipping in each same-state comparison. At
initialization, require global parameter-update cosine >=0.99; with retained
nonempty moments require global update relative L2 <=2*eps. Preserve per-tensor
and coordinate diagnostics. The fixed near-zero mask is
`abs(g_fp32) <= 2*eps*RMS(g_fp32_parameter)+2e-6`; report gradient sign flips and
update-error energy inside and outside it. Initial Adam can turn very small
gradient sign differences into changes approaching 2*LR at a coordinate, so
no smaller universal coordinate update maximum is imposed.

After local and initialization numerical checks, a bounded paired 100-update
tiled FP32/BF16 smoke run supplies a five-block midpoint checkpoint, training
curves, adapter/owner signals and exact BF16 midpoint recovery. Training and
evaluation use W&B entity `taylorbollman` plus persistent local/GCS records.
Require finite states and exact intended data/model/optimizer/RNG restoration.
Review a final-window paired mean training loss or development CE degradation
above 0.02 nats. Low accuracy in a short run establishes operational behavior,
not learning equivalence. A separate tiny repeated-batch check can verify
active optimization without becoming a research sweep.

Benchmark warmed complete updates and memory with diagnostic observers removed.
Report numerical disposition, operational evidence and performance separately.
Clearance is limited to the tested five-block, one-fabric rho-one configuration
and declared execution policy. General rho, multiple fabrics, packing/cached
generation, accumulation, distributed execution and long CDRM research
comparisons remain outside this milestone.
