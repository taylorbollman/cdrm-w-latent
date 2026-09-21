# OLMo O2: bounded tiled RT protocol

Recorded before actual-checkpoint execution on 2026-09-21. O2 implements native
OLMo exact tiled temporal recurrence and its checkpointed backward. It does not
train, implement FBT, or add NextLat.

## Model and mathematics

Reuse O1's original OLMo-1B step60000 checkpoint (~252B tokens), revision
`81b71efbce6f4dada57c94860301af4298bcd351`, SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Preserve all 65 native tensors, 1,176,764,416 parameters, tied 50,304-row
embedding/readout, non-affine LayerNorm, native SwiGLU, full MHA and FP32 RoPE.
Actual full-model checks select layer 0 for RT; the remaining 15 are ordinary.

Temporary self K/V comes from the current block input. Historical persistent
K/V comes from normalized `m_t = (1-alpha) x_t + alpha z_t`, where `z_t` is the
complete block output. Cache keys are unrotated. The tiled forward must merge
exact historical attention contributions, and backward reconstructs parallel
intermediates from saved inputs and outputs without replaying the sequential
attention forward. Reverse recurrent cotangent propagation remains sequential.

All four native block weight tensors are explicit autograd-function inputs.
Backward returns gradients, including prefix-cache cotangents, instead of
writing hidden parameter `.grad` fields. Current cache outputs participate in
autograd, including a terminal write used only by a later chunk or cache loss.

## Correctness and precision

Tiny CPU tests compare O2 with O1's sequential implementation and the independent
pristine-source history oracle. Cover alpha 0, 0.37 and 1; lower/upper/both-layer
placement; irregular lengths; mask and RoPE offsets; unnormalized random
upstream gradients; frozen inputs/weights; three shared-weight calls; attached
cached chunks; and native ownership. Cache metadata/provenance must reject
incompatible reuse. Higher-order derivatives and distributed training are not
cleared by these tests.

Actual checkpoint: B1/T16, all 65 parameter tensors and embedding-input
gradients, logits, hidden state and shifted cross entropy. Compare tiled alpha
0/0.37/1 to the validated O1 scan in FP32/math attention, with TF32 disabled.
An additional alpha-0 comparison uses ordinary native adapter execution. Use
O1's existing semantic budgets: logits/hidden absolute+relative 1e-4; loss
1e-5; input gradients absolute 2e-6 plus relative 3e-4; every parameter tensor
relative L2 <=1e-4 (or whole error norm <=2e-6) AND maximum absolute error
<=2e-6+1e-4*reference maximum magnitude. Retain stricter elementwise diagnostics.
Do not relax these budgets after observing results.

At alpha 1, compare BF16 autocast with mixed attention and FP32 attention against
the same tiled FP32 run; also record differences from the O1 scan. Require
finite outputs/loss/input/parameter gradients and complete gradient ownership;
cross-precision differences are descriptive, not a training-quality clearance.
Batching recomputation changes BF16 rounding relative to tokenwise forward, so
identical BF16 scan/tiled gradients are not presumed. Ordinary-layer math SDPA
and default SDPA are separate labeled cases.

Additional actual-size coverage: bottom-block B2/T17, with unnormalized random
output and persistent-cache cotangents, compared to O1; full-model B1/T128
FP32 baseline and BF16 comparisons using default ordinary-layer SDPA. The latter
checks a less tiny runtime shape without claiming context-2048 clearance.

## Performance and memory

Profile forward plus backward at native size, preserving checkpoint weights,
without optimizer updates. Start with isolated block B1/T128 (scan and tiled,
FP32 and BF16); tiled BF16 B1/T256, B1/T512, B4/T512; and full-model tiled B1/T128
and B1/T512. Use 3 warmups and 5 timed iterations, report synchronized wall and
CUDA-event times separately, and peak allocated/reserved memory. No compile,
CUDA graphs, FlashAttention kernel claim, or batch-capacity optimization.
Stop expansion if allocation approaches 60 GiB or a correctness failure occurs.
Backward's initial parallel attention reconstruction uses quadratic attention
storage: this is checkpointed exact recurrence, not an optimized FlashAttention
backward kernel. Measurements describe this implementation only.

## Evidence

New W&B runs in `taylorbollman/pretrained-fbt-rt-nextlat`; new immutable report
directories, source hashes, exact token fixtures, configuration and runtime.
Keep failed attempts distinct. Retain source and evidence on `gs://fast-chunks`,
referencing the verified O1 checkpoint object instead of uploading it again.
Document remaining limits and stop for review at the O2 milestone.

## Explicit amendment after the initial stress-test failure

Validation-01 passed all four full-model FP32 comparisons and all BF16 finite
checks, but its unnormalized-cotangent B2/T17 input gradient failed the strict
coordinate screen. The input-gradient relative L2 difference was 5.548e-7;
all four block parameter tensors passed the original joint tensor budgets.
The failing report remains in `initial-validation-01.json`; it is not relabeled.

For validation-02, preserve that original input-coordinate screen verbatim,
and additionally compare both tiled FP32 and scan FP32 against an independent
FP64 block oracle using exactly the same inputs and raw output/cache
cotangents. The oracle keeps native FP32 positional sine/cosine coefficients,
but applies those constants and all other block operations in FP64. This is
roundoff adjudication, not a change to production RoPE.

The raw-cotangent input gradient now uses the same joint tensor criterion
already applied to parameter gradients: relative L2<=1e-4 (or whole error
norm<=2e-6) AND maximum error<=2e-6+1e-4*reference maximum magnitude. Require
this for tiled-versus-scan AND each FP32 implementation versus the FP64 oracle.
This is an explicit post-failure calibration for cancellation in an
unnormalized input gradient, not an unchanged preregistered acceptance claim.
The full-model input-gradient checks, all parameter budgets and all other
semantic checks remain unchanged. Report the original coordinate failures and
the FP64 comparison, regardless of this additional criterion's outcome.

Validation-02 also includes two reviewed implementation corrections: separate
the temporary self diagonal before mixed-precision reconstruction matmuls
(avoiding rounded permanent-diagonal cancellation), and cast frozen local-VJP
weights once instead of at every position. These preserve the recurrence math.
The profiling run uses this corrected implementation. Profiling is operational
evidence only and does not clear the unresolved numerical screen by itself.
