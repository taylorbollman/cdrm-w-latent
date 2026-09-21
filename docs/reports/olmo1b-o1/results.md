# OLMo O1: native checkpoint and sequential RT reference

Completed 2026-09-21. **Both review gates pass.** The native OLMo-1B checkpoint
imports faithfully, and selected-layer sequential RT matches an independent
reference in FP32, including gradients. No backbone adaptation, tiled execution,
FBT or NextLat training was performed. The next milestone is the native RoPE
tiled backend, subject to the user's review of this result.

Implementation: [PR #4](https://github.com/taylorbollman/cdrm-w-latent/pull/4),
source/evidence commit `9a5bc78`. Subsequent documentation and whitespace-only
test formatting do not change model/validator source hashes in the reports.

- [Usage and reproduction](../../olmo1b-native-rt-usage.md)
- [Frozen protocol](protocol.md)
- [Compact machine-readable results](validation-summary.json)
- [148-test CPU record](test-results.txt)
- [Current compaction handoff](../../fbt-rt-nextlat-handoff.md)

## Exact model and implementation

Original `allenai/OLMo-1B@81b71efbce6f4dada57c94860301af4298bcd351`,
`step60000-tokens252B`, approximately 251–252B pretraining tokens. The complete
4,707,065,440-byte native safetensors file matches published SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Strict inspection finds 65 finite FP32 tensors totaling **1,176,764,416 parameters**.

The model preserves 16 uniform blocks, width 2048, 16-head full MHA with head dimension 128,
SwiGLU intermediate 8192, non-affine LayerNorm, native FP32 RoPE, no Q/K
normalization, and one tied 50,304-row input/readout matrix. The native tokenizer
has 50,280 entries, adds no BOS/EOS automatically, and comes from the same pinned
revision. Configured pad ID 1 does not suppress embedding lookup gradients.

The independent ordinary reference executes byte-identical original OLMo
v0.2.4 sources, isolated from our modified research fork. Native primitives and
model forward are unchanged; source hashes and limited import adaptations are
documented. The sequential RT adapter stores input/output-derived persistent
K/V and uses ordinary autograd. Its independent oracle reconstructs history
anew per query and performs explicit attention using native block primitives.

## Gate A: ordinary fidelity

Physical B1, text fixture T42 and code fixture T64. Outputs/loss/hidden states
were checked on both; all 65 parameter gradients and input gradients were checked
on T42. **Adapter-versus-native differences were exactly zero** in every checked
output and gradient for all three runtime combinations:

| Runtime | Source output parity | Source gradient parity |
| --- | --- | --- |
| FP32, math SDPA, TF32 off | Exact | Exact |
| BF16 autocast, math SDPA | Exact | Exact |
| BF16 autocast, default SDPA | Exact | Exact |

Weights and parameter gradients remain FP32. Source parity in BF16 means both
implementations execute the same learned function with the same runtime
rounding; it does not imply equality with FP32.

Chunked adapter execution, native cached decoding, native KV shapes and future
token isolation pass. Largest cached-logit absolute difference is 3.94e-5;
the future-token isolation difference is zero. The bounded profiler observed
cuDNN fused/Flash-style SDPA for ordinary BF16 attention. Its separate fused
versus math comparison is descriptive, not a deterministic cross-invocation
or throughput claim.

W&B: [ordinary import, td5cce3w](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/td5cce3w).
Full report: `.runtime/olmo1b-step60000/ordinary-validation-01/report.json`.

## Gate B: sequential RT semantics

Actual checkpoint, physical **B1/T16, layer 0 recurrent**, all remaining 15 blocks
ordinary. Alpha 0 executes the actual scan. All four FP32 comparisons pass the
predeclared per-tensor L2 and maximum-error criteria, as well as input-gradient,
output and loss checks:

| Alpha | Reference | Global parameter-gradient relative L2 | Worst tensor relative L2 |
| --- | --- | ---: | ---: |
| 0 | Native ordinary | 2.286e-6 | 4.880e-6 |
| 0 | Independent RT oracle | 1.660e-6 | 2.253e-6 |
| 0.37 | Independent RT oracle | 1.649e-6 | 2.504e-6 |
| 1 | Independent RT oracle | 1.839e-6 | 2.304e-6 |

The largest input-gradient relative L2 error is 2.534e-6. Cached/chunked scans,
causality, first-token temporary self-attention, native KV heads and wrong-mode
cache rejection pass at all three alpha values. Largest cached-logit absolute
difference is 4.46e-5; changing future tokens causes exactly zero prefix-logit
change.

Two stricter elementwise FP32 screens flag coordinates: block 1 `ff_out.weight`
for alpha 0 versus ordinary, and block 1 `ff_proj.weight` for alpha 0.37 versus the
oracle. These observations remain in the full report. Both tensors pass the
**predeclared** joint tensor-L2 and tensor-scaled maximum criteria; no tolerance
was changed after these results. Different scan/reconstruction GEMM shapes and
reduction orders need not preserve small cancellation coordinates elementwise.
No adapter or backward correction was required to pass the reference checks.

W&B: [sequential RT, ujs94viz](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ujs94viz).
Full report: `.runtime/olmo1b-step60000/rt-validation-01/report.json`.

## BF16 observations and interpretation

All bounded BF16 forwards/backwards are finite, with complete gradient
ownership. Full recurrence at alpha 1 in selected layer 0, B1/T16, gives:

| Execution | Global gradient relative L2 versus same-path FP32 | Worst tensor relative L2 |
| --- | ---: | ---: |
| Math SDPA | 1.695% | 3.751% |
| Default SDPA | 1.451% | 3.461% |

Both worst tensors are `transformer.blocks.5.ff_proj.weight`. The top predicted
token is unchanged at all 16 fixture positions in both cases. CE is 4.68350 in
FP32, 4.68648 with math BF16, and 4.69004 with default BF16. These fixture losses
are numerical diagnostics, not downstream model-performance estimates.

For context, ordinary BF16 at the **different T42 fixture** has global gradient
differences 1.546%/1.625% and worst tensor differences 3.471%/3.673% versus FP32.
This places the observed RT discrepancies in a similar numerical range, but
the differing fixtures prevent a controlled claim that RT adds no sensitivity.
The alpha 0 BF16 scan also differs from native parallel ordinary execution
(global gradient relative L2 1.084%), illustrating that changing execution
shapes alone changes mixed-precision rounding.

These observations do not justify adding Q/K normalization or a new precision
policy now. They also do not clear BF16 tiled training. The next backend should
be compared with this reference at its intended runtime settings. BF16 report
rows retain some stricter elementwise diagnostic labels; their outer `passed`
means only the documented finite/complete-gradient smoke check.

## CPU coverage, runtime and limits

**148 tests pass** in the explicit CPU container. Coverage includes immutable
artifact/tokenizer provenance; source integrity; geometry/tying; native pad-row
lookup gradients; ordinary/source parity; cached attached gradients; alpha 0,
fractional and full recurrence at bottom/top/both placements; explicit positions
and masks; finite-difference/branch checks; composed shared-weight forwards;
stale/mismatched cache rejection; tiny optimizer/save-resume; and retention
whitelists/no-overwrite verification. The 31 warnings concern deliberately tiny
fixture vocabularies, deprecated upstream autocast queries and a Google client
dependency notice; none is a failed check. Pristine source was not patched.

Actual-checkpoint GPU validation used the verified project container on one
H100 80GB, Torch 2.13.0a0+8145d630e8.nv26.06, CUDA 13.3, TF32 disabled. Ordinary
validation took 35.1s and RT validation 36.8s, excluding container launch. Peak
allocated GPU memory was about 19.54GiB for each paired validation process. This is
not single-model training capacity or a useful throughput benchmark.

The full-size RT coverage is **only layer 0, B1/T16**, with no optimizer updates
to the pretrained checkpoint. Wider placement coverage is on tiny fixtures.
Long contexts, large batches, tiled recomputation, CUDA graphs/compile, multi-GPU,
FBT, NextLat and adaptation quality remain untested here. The earlier OpenELM
implementation and its results were preserved without source changes.

## Retention and next review point

Native checkpoint, manifests/tokenizer, pristine reference, tested source
overlay and both complete validation reports are retained with verified object
checksums and generations; see [storage receipt](storage-receipt.json). The
checkpoint is a separate object, not compressed into the evidence archive.
No newly trained checkpoint was created.

O1 is a suitable review pause. The proposed O2 work is native-RoPE exact tiled
forward/backward with fractional-alpha gradients against this reference,
followed by bounded actual-runtime precision and memory/throughput checks.
FBT, NextLat and learning remain later milestones.
