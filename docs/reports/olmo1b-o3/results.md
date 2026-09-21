# OLMo O3: language-model NextLat and optimizer platform

Completed GPU validation and profiling on 2026-09-21. The language-model
objective, gradient ownership, complete AdamW update and exact checkpoint-resume
paths passed their bounded checks. **326 scoped CPU tests passed.** At B8/T512,
RT+NextLat executed a complete BF16 mixed optimizer step in **1.626 seconds**,
approximately **2,519 valid input tokens/s**, with **23.99 GiB** peak allocated
memory on one H100 80GB. These are correctness and operational results;
**O4 comparative learning has not started**, and FBT is not implemented.

See the [protocol](protocol.md), [usage/API guide](../../olmo1b-nextlat-platform-usage.md),
[machine-readable summary](validation-summary.json), and [CPU test record](test-results.txt).
Online records are [validation](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/r0zqx75g)
and [profiling](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zbrcek8g).
Implementation/evidence commit: `40290357aff82f0caa40f9deff79c90412758faa`,
[PR #6](https://github.com/taylorbollman/cdrm-w-latent/pull/6). Subsequent
documentation records this link; tested source hashes are in the raw reports.

## Fixed model and implementation

The starting checkpoint is unchanged original OLMo-1B at step60000 (~252B
pretraining tokens), revision `81b71efbce6f4dada57c94860301af4298bcd351`, SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Its 16-layer/D2048 backbone has 16 full-MHA heads, SwiGLU intermediate 8192,
non-affine LayerNorm, native FP32 split-half RoPE, no Q/K normalization, and
one tied 50,304-row embedding/readout. All native output rows participate in
the losses, including the rows beyond the tokenizer's 50,280 IDs.

The backbone has **1,176,764,416 parameters** in 65 tensors. NextLat adds
**82,726,912 parameters** in four tensors, for **1,259,491,328 total**. Its
predictor follows the pinned NextLat **1B LM, horizon-one** configuration at
revision `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`: projection factor 1.6 gives
hidden width 6528; a learned-scale RMSNorm precedes three bias-free linear
layers with two GELUs and a residual addition. Linear initialization std is
0.02, norm epsilon 1e-5, dropout 0, and predictor seed 20260921. This differs from
the historical A5 recipe's factor 0.5 and disabled KL. The
[pinned source audit](../../../cdrm/pretrained/_nextlat_reference/README.md)
records these differences and the upstream license.

The objective is ordinary next-token CE plus separately weighted latent
SmoothL1(beta=1) and teacher-to-student KL, initially weighted 1/1/1. The predictor
maps `(h[t], e[t+1])` to `stopgrad(h[t+1])`; its KL distribution and detached
teacher both concern token t+2. Source states and conditioning embeddings stay
attached. Only the auxiliary readout use is detached, preserving native tied
embedding gradients from CE and input/conditioning lookup. Final normalization
is applied once by the backbone.

Explicit same-document pairs and triples, plus independent CE/latent/KL target
masks, determine each term's denominator. The platform normalizes accumulated
microbatch sums by the whole update's separate valid counts. O3 accepts one
document per padded row and rejects packed multiple-document rows. Auxiliary
loss masks are not a substitute for document-isolated attention.

Actual RT checks select **layer 0 only**, at alpha 1; the other 15 layers remain
ordinary. Ordinary execution selects no RT layers. RT and NextLat are
independent switches. There is no predictor rollout during inference.

## Correctness and precision

The 326-test CPU suite includes the retained native/O1/O2 checks and new
NextLat, accumulation, checkpoint and retention coverage. Independent tests
execute unchanged pinned predictor class definitions and compare all predictor
derivatives; a separate dense objective checks the LM shift, KL direction,
pair/triple masks, prompt/response policy, coordinate/position reductions,
detached target/teacher/readout use, and attached conditioning path. Integration
tests cover ordinary and tiled RT at alpha 0/.37/1, all parameter gradients,
shared calls, empty/disabled terms, document-row causality, and no repeated
final normalization. Unequal valid-count accumulation and exact tiny-model
resume are also covered.

Actual-checkpoint validation used FP32 parameters, TF32 disabled, math SDPA,
and an independently coded dense objective on the same backbone. It checks
the new loss implementation; it does not repeat O1's native-source import
validation. Vocabulary projections were checkpointed in chunks of **eight
valid positions**, each retaining the full vocabulary normalization.

| Fixture | RT | NextLat | CE / latent / KL valid positions | CE / latent / KL mean losses | Global parameter-gradient relative L2 | Worst tensor relative L2 |
| --- | --- | --- | --- | --- | ---: | ---: |
| B1/T16 | Off | Off | 8 / 0 / 0 | 4.661606 / 0 / 0 | 2.403e-6 | 2.759e-6 |
| B1/T16 | Layer 0 | Off | 8 / 0 / 0 | 5.094544 / 0 / 0 | 1.579e-6 | 2.033e-6 |
| B1/T16 | Off | On | 8 / 15 / 8 | 4.661606 / 0.932018 / 8.471173 | 3.825e-6 | 4.333e-6 |
| B1/T16 | Layer 0 | On | 8 / 15 / 8 | 5.094544 / 0.916210 / 8.348677 | 4.811e-6 | 6.226e-6 |
| Padded B2/T16 | Layer 0 | On | 14 / 25 / 14 | 4.692825 / 0.931285 / 8.360292 | 3.132e-6 | 3.923e-6 |

All five cases passed the predeclared loss budget `atol=rtol=1e-5` and every
active parameter tensor's joint gradient budget: relative L2<=1e-4 (or error
norm<=2e-6), together with maximum absolute error<=`2e-6 + 1e-4 * max(abs(reference))`.
There were no missing gradients: 65 tensors without NextLat and 69 with it.
The largest component-loss discrepancy was 3.815e-6; the largest parameter
coordinate discrepancy across the five cases was 2.384e-5.

The stricter elementwise diagnostic flagged **2, 0, 11, 19, and 6 parameter
tensors**, respectively, in the table's row order. Those diagnostics and tensor
names are retained in the raw report. They do not disappear under a claim of
bitwise equivalence: acceptance used the protocol's unchanged joint per-tensor
criteria, which all passed. The short fixture losses are diagnostic values,
not held-out perplexity or evidence that recurrence improves language modeling.

### Bounded BF16 observations

At B1/T16 with layer 0 RT and NextLat enabled, BF16 autocast was compared with
the same initialized FP32 state. Ordinary layers used math SDPA in both
comparisons. Every objective and all 69 parameter gradients were finite.

| Selected RT attention policy | Global gradient relative L2 difference | Worst tensor difference | CE / latent / KL means |
| --- | ---: | ---: | --- |
| Mixed | 1.932% | 2.839% | 5.090590 / 0.916504 / 8.349944 |
| FP32 | 1.911% | 2.590% | 5.098221 / 0.916371 / 8.342242 |

These differences are **descriptive**, not gradient-equivalence or long-run
training clearance. The FP32 attention option affects the selected tiled
attention computation; it does not make the ordinary layers, projections,
MLPs or predictor fully FP32 under autocast. This bounded result does not
justify a new precision campaign or establish a preferred training policy.

## Complete optimizer recovery

The actual RT+NextLat model performed two disposable FP32 AdamW updates, saved
at the boundary, then compared the next uninterrupted update with a freshly
constructed and restored execution. AdamW used LR 1e-5, betas (.9,.95), epsilon1e-8,
weight decay .1, gradient clipping 1, and a two-update linear warmup.

The resumed third update matched **exactly** in every model tensor, optimizer
moment/state, scheduler state, counter, next CPU/CUDA RNG draw, and returned
update metric. The data cursor and update 2 counter were checked at restore.
The driver executed four optimizer updates in total: updates 1 and 2, then
update 3 twice for the uninterrupted/resumed comparison.

The complete update 2 checkpoint was 15,114,028,075 bytes (14.08 GiB), SHA256
`8d615b49730f8c8e1292fac008596977e1fdc754e0f78366293418a05b6fba22`.
It was **deleted after successful recovery**, as planned; its hash and full
recovery evidence remain in the validation report. This disposable state is
not an adaptation candidate, and the original checkpoint was not overwritten.
Validation peaked at 20.76 GiB allocated / 21.70 GiB reserved, including comparison,
recovery and digest scratch; that number is not a training batch-capacity estimate.

## Full-step profile

All seven cases used one H100 80GB, BF16 autocast, default SDPA for ordinary
layers, mixed precision for the selected tiled attention, and full-vocabulary
projection chunks of **128 valid positions**. Each had three warmup steps and
three timed steps, with AdamW moments initialized before timing. Timing includes
forward, all enabled objectives, backward, clipping and AdamW, with one
microbatch per update. There was no compile, CUDA graph or distributed wrapper.

LR was zero. All model/predictor tensor hashes remained unchanged, and all
optimizer moments stayed finite. There were **42 zero-LR step executions**
across the seven cases. This exercises optimizer computation and state without
measuring learning.

| Model | B × T | Median wall seconds | Median CUDA seconds | Valid input tokens/s | CE target tokens/s | Peak allocated GiB | Peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 1 × 128 | 0.0684 | 0.0684 | 1,871 | 935 | 18.37 | 19.79 |
| RT | 1 × 128 | 0.4285 | 0.4284 | 299 | 149 | 18.37 | 20.03 |
| Ordinary + NextLat | 1 × 128 | 0.0759 | 0.0758 | 1,687 | 844 | 19.61 | 21.07 |
| RT + NextLat | 1 × 128 | 0.4336 | 0.4336 | 295 | 148 | 19.61 | 21.00 |
| RT + NextLat | 1 × 512 | 1.4959 | 1.4959 | 342 | 171 | 19.61 | 21.54 |
| RT + NextLat | 4 × 512 | 1.5871 | 1.5871 | 1,290 | 645 | 20.37 | 23.65 |
| RT + NextLat | 8 × 512 | 1.6262 | 1.6262 | 2,519 | 1,259 | 23.99 | 25.79 |

The fixtures repeat literal text/code prompts to the chosen length and apply
response masks to half the positions. At B1/T128 there are 64 CE targets, with
127 latent pairs and 64 KL targets when enabled. At T512 these counts are
`256*B`, `511*B`, and `256*B`. Therefore valid-input throughput is twice the
reported CE-target throughput; neither is an estimate of evaluation accuracy.

Initialized GPU AdamW state occupies 8.77 GiB without NextLat and 9.38 GiB with
it. Resident allocations before timed steps were 13.22/14.14 GiB, respectively.
The table's peaks cover the complete optimizer step and exclude subsequent
hashing and finite-moment checks. The default-SDPA BF16 profile demonstrates
execution and capacity in that setting; quantitative gradient comparisons above
used the short math-backend fixtures instead.

The useful directional finding is that this eager RT implementation benefits
substantially from batching: at T512, increasing B1 to B8 changes median wall time
from 1.496 to 1.626 seconds while increasing valid input throughput from 342 to 2,519
tokens/s (**7.36×**). B8 leaves substantial memory on this GPU for the measured setup.
This was not a maximum-batch search, and three timed iterations per point do
not establish an optimized or universal throughput result. The O2 backward's
quadratic attention reconstruction and sequential recurrent adjoints remain.

## Evidence and limits

Raw local reports are
`.runtime/olmo1b-step60000/lm-validation-01/report.json` and
`.runtime/olmo1b-step60000/lm-profile-01/report.json`. They include exact token
fixtures/masks, source hashes, configurations, per-tensor comparisons and
recovery digests. Runtime was PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3,
H100 80GB HBM3, driver 580.173.02, in the project container.

Retention destination:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-nextlat-platform/20260921T213500Z/`.
The [storage receipt](storage-receipt.json) is the authority for uploaded-object
verification and the reused O1 checkpoint reference. The large disposable
optimizer checkpoint is not part of retention.

The actual-checkpoint scope is one GPU and one selected RT layer. Tiny tests
cover fractional alpha; this report does not extend O3 GPU clearance to all 16
recurrent layers. Packed-document attention, cached LM training, multi-GPU
accumulation/resume, compiler/graph execution and double backward remain outside
this milestone. No FBT or comparative task adaptation was run. O4 still needs
a selected dataset/split, explicit masking policy, matched training budgets and
a bounded ordinary/RT/NextLat learning comparison before conclusions about
language-model benefit.
