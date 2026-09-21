# OLMo-1B / RT / FBT / NextLat research plan

Updated 2026-09-21. **Current authoritative plan.**

**Implementation update:** O1 native import/sequential RT and O2 native RoPE
tiled forward/backward now pass bounded validation. See
[O1 results](reports/olmo1b-o1/results.md), [O2 results](reports/olmo1b-o2/results.md)
and the [handoff](fbt-rt-nextlat-handoff.md). O2 includes an explicit raw-input
coordinate-screen calibration backed by FP64, and initial eager performance
measurements; it does not clear training quality or optimized throughput.
O3 language-model NextLat and the single-GPU training platform are also complete:
326 scoped CPU tests, actual-checkpoint objective/gradient parity, exact optimizer
recovery, and bounded complete-step profiling. See [O3 results](reports/olmo1b-o3/results.md)
and [usage](olmo1b-nextlat-platform-usage.md). Two-GPU correctness remains untested.
The user subsequently authorized O4; the first matched Python-code pilot is
active, with its recipe/exposure fixed in the [O4 protocol](reports/olmo1b-o4/protocol.md).
O5 and later milestones remain staged. Historical planning-status statements
below describe checkpoint selection, before the implementation evidence.

This replaces the model
selection and forward milestones in [v2](fbt-rt-nextlat-research-plan-v2.md)
and the [original proposal](fbt-rt-nextlat-pretrained-plan.md). The user requested
this revision before authorizing the next implementation milestone. This change
contains planning and checkpoint metadata only; no OLMo model implementation,
full weight download, GPU validation or training has been performed.

Read the [current handoff](fbt-rt-nextlat-handoff.md) first after compaction.
The completed OpenELM import and sequential RT reference remain historical
working implementations, with their original evidence preserved in the
[OpenELM handoff](fbt-openelm-implementation-handoff.md). They are not evidence
that OLMo import, recurrence or precision has already passed.

## 1. Decision and research question

Use **original, first-release OLMo-1B at step 60,000**, approximately
**251–252B pretraining tokens**, as the primary backbone. This is not
OLMo-1B-0724, OLMo 2, or the final approximately 3T-token original checkpoint.
It gives us uniform layer dimensions, native tied input/readout weights, RoPE,
and a public intermediate checkpoint at the age the user requested.

The switch reflects the user's preference to remove uncertainty about
OpenELM's layer-wise capacity scaling, not evidence that scaling breaks FBT.
There is no need to add tying: preserve OLMo's existing shared parameter.
Original OLMo has **no Q/K normalization**; preserve that too for the first
implementation and controls. Native Q/K normalization is desirable in some
designs, but its absence is not an implementation error or a demonstrated
barrier to feedback. Do not bundle a normalization conversion with RT/FBT.

The research question remains whether output-derived temporal memory (RT),
top-to-bottom feedback (FBT), and NextLat auxiliary training improve adaptation
and sequential generation, separately and in combination. Keep independent
switches for all eight combinations. This is a capability requirement, not a
queue of eight long training runs. RT alone need not succeed before RT+NextLat
or feedback-conditioned RT is tested.

## 2. Frozen checkpoint and provenance

| Role | Repository / branch | Immutable revision |
| --- | --- | --- |
| **Primary native artifact** | `allenai/OLMo-1B`, `step60000-tokens252B` | `81b71efbce6f4dada57c94860301af4298bcd351` |
| Secondary official HF conversion | `allenai/OLMo-1B-hf`, `step60000-tokens251B` | `6e6042e824831c7b42223f75cf60fe3a5d92eb79` |
| Prospective independent native source | `allenai/OLMo`, tag `v0.2.4` | `b3741bc21f1dd504838b7dbd9878ee077ded63bd` |

Sources: [native checkpoint](https://huggingface.co/allenai/OLMo-1B/tree/81b71efbce6f4dada57c94860301af4298bcd351),
[official conversion](https://huggingface.co/allenai/OLMo-1B-hf/tree/6e6042e824831c7b42223f75cf60fe3a5d92eb79),
[historical native source](https://github.com/allenai/OLMo/tree/b3741bc21f1dd504838b7dbd9878ee077ded63bd).
The machine-readable [selection audit](olmo-1b-250b-checkpoint-selection.json)
records URLs, source/config/tokenizer hashes, safetensors header evidence,
published full-file hashes and untested scope.

The native file is `model.safetensors`, advertised size **4,707,065,440 bytes**,
advertised SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Only metadata, small source/tokenizer files and bounded header ranges have
been inspected. The whole-file hash has **not** been locally verified.

The two branches agree on optimizer step but use different rounded token-age
labels. Report approximately 252B, or 251–252B, rather than inventing an exact
training counter. Same-step labels and matching geometry do not prove tensor
equality: verify the actual conversion before treating it as a second oracle.
These are model-only artifacts; adaptation uses a new optimizer and warmup,
not a claimed exact resumption of original pretraining.

The checkpoint's remote-code stubs import `hf_olmo`, with an unbounded package
requirement. Pinning the weights repository alone therefore does not pin the
model math. The source revision above has been inspected, but execution and
checkpoint parity are still prospective. Freeze its primitives in an isolated
reference namespace and validate them; do not accidentally import our modified
`recurrent-transformer/olmo` as an independent upstream reference.

**Tokenization authority:** use files at the native checkpoint revision.
Native and converted HF tokenizer JSONs are not byte-identical and have
different postprocessors. Before substituting any tokenizer, verify text,
special-token, encode/decode and document-boundary fixtures. Freeze where EOS
is inserted, and whether any BOS is inserted, in the dataset contract; do not
inherit OpenELM or synthetic preprocessing defaults.

### Why this age, and the Chinchilla comparison

The native header contains **1,176,764,416 unique parameters**. Using the rough
20-training-tokens-per-parameter convention gives about **23.54B tokens**;
251–252B is therefore approximately **10.7 times that budget**. The user's
“around 10x” description is reasonable using total unique parameters.
Counting conventions and data quality matter; this is a useful heuristic, not
an exact compute-optimal estimate or a saturation threshold.
[Chinchilla paper](https://arxiv.org/abs/2203.15556).

This checkpoint is in the broad exposure range of the FBT paper's 200B/400B
experiments. Starting adaptation at 252B is nevertheless different from the
paper's feedback switch-on schedule, starting model, data and final exposure.
We are testing transfer of the mechanism, not reproducing its training run.
[FBT paper](https://arxiv.org/html/2608.08888v1).

## 3. Native architecture to preserve

| Component | Selected OLMo-1B |
| --- | --- |
| Depth / residual width | 16 uniform blocks / 2048 |
| Attention | 16 query and 16 KV heads; head dimension 128; full causal MHA |
| Positions / context | RoPE base 10,000; native FP32 split-half rotation; context 2048 |
| Norms | Non-affine LayerNorm, including final norm; epsilon 1e-5; no Q/K norm |
| MLP | SwiGLU, intermediate 8192; native fused projection width 16384 |
| Projections | No learned biases; no QKV clipping; no ALiBi |
| Input/readout | One tied parameter with 50,304 rows; preserve all rows |
| Tokenizer vocabulary | 50,280; EOS 50279; configured pad ID 1 |
| Stored parameters | FP32; 65 native tensor keys, totaling 1,176,764,416 elements |

These details are checkpoint/source contracts, not new hyperparameters:

- Native fused SwiGLU splits **value/up, then gate**, and computes
  `silu(gate) * value`. OpenELM's ordering must not be copied blindly.
- Native `nn.Embedding` has **no `padding_idx`**, despite the configured pad ID.
  Do not suppress that row's lookup gradient. Explicit masks and labels govern
  excluded positions. Preserve the full 50,304-way logits denominator.
- Keep the embedding/readout as one owned parameter, once in the optimizer,
  across materialization, save/reload and distributed wrapping.
- Preserve native RoPE dtype and operation ordering. Native caches hold
  **unrotated** K/V; do not double-rotate keys during chunked decoding.
- Prefer the native fused parameter layout for the first adapter. The official
  HF conversion has 113 stored tensors because projections are split; map
  weights and gradients explicitly rather than comparing raw key lists.

## 4. Model semantics retained through the switch

### RT: output-derived temporal memory

Initially select **layer 0 only**, with full causal history. The top layer is
now **15**, derived from the loaded configuration. Do not inherit OpenELM's
index 27 or the restricted-window first layer from the synthetic experiments.

Let x_t be a selected block's input, z_t its completed residual block output,
A its native input LayerNorm, and R_t the rotary transformation. Temporary
query/key/value projections come from A(x_t). At token t, attention uses
persistent entries from earlier positions plus its own temporary entry. After
completing z_t, publish the persistent write from

\[
m_t=(1-\alpha)x_t+\alpha z_t,\qquad
k_t=R_t\operatorname{heads}(W_K A(m_t)),\qquad
v_t=\operatorname{heads}(W_V A(m_t)).
\]

There is **no added key normalization** in this OLMo equation. The displayed
key is in attention coordinates; cache storage remains unrotated by convention.
At the top block, z_t is before the model's final LayerNorm. Alpha 0 must recover
ordinary outputs **and gradients through the actual scan**; alpha 1 is full
RT. Fractional alpha must propagate both input and output source branches.
Store alpha, positions, masks and mode immutably per forward, not in mutable
configuration that an outstanding backward will reread.

Reuse the established attached-autograd and cache-provenance design, adapting
the native block math. Cache metadata must reject incompatible model, mode,
weights, dtype/autocast, gradient context or backend. Keep tensors attached
where training requires it. Unsupported mutation of cached tensors or `.data`
writes is outside the contract.

### FBT: shifted top-state feedback through shared stack passes

Let h_t be the post-final-LayerNorm model state and e_t the token embedding.
For extra pass p, feedback at t comes from h_(t-1) of pass p-1, never from the
current or future token. Preserve the asymmetric state-value/token-gate fusion,
schematically

\[
f_t^{(p)}=W_U h_{t-1}^{(p-1)}\odot\sigma(W_G e_t).
\]

Normalize/scale the **new fusion branch** according to the pinned FBT design,
calibrated to native embedding RMS; do not replace OLMo's existing norms or
residual paths. Preserve the planned beta transition from ordinary embeddings
to the fusion endpoint. Beta 0 must bypass fusion normalization exactly.
Document starts use ordinary input and clear all cross-document feedback.
Alpha controls RT memory, beta feedback; they are distinct controls.

| FBT | RT | Execution; NextLat independently on/off |
| --- | --- | --- |
| Off | Off | Ordinary single pass |
| Off | On | One pass with selected RT blocks |
| On | Off | Ordinary pass 0, then feedback passes using ordinary attention |
| On | On | Ordinary pass 0, then feedback passes with selected RT blocks |

K counts complete passes in the FBT API; **K1 is ordinary**. Standalone RT does
not execute an unused ordinary pass. All passes share the same backbone
parameters and retain cross-pass gradients. Finite-pass prefill consumes
previous-pass states; exact online generation consumes the freshly completed
previous-token state. Their discrepancy is an evaluation target, not
automatically a cache bug. Cache provenance distinguishes ordinary-prefilled
and recurrent-prefilled histories, including any explicit prefix switch.

### NextLat: auxiliary learning, without autonomous predictor rollout

Use the post-finalnorm state and next token's embedding to predict the next
state: `predicted_h[t+1] = F(h[t], e[t+1])`. Preserve the pinned source's latent
SmoothL1 regression and separately weighted detached-readout KL. Keep source
states and conditioning embeddings attached; stop the target state and target
distribution. For auxiliary logits, detach only the readout use:
`F.linear(predicted_h, embedding.weight.detach())`. Ordinary CE and embedding
paths still train the tied matrix. Do not apply final normalization twice.
[Pinned NextLat source](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_nextlat.py).

Latent regression requires a valid same-document pair (t,t+1); the associated
KL concerns token t+2 and requires a valid same-document triple. Specify prompt
and response masking separately from CE. Reduce each objective by its actual
valid positions, with latent coordinates averaged too. Reuse the explicit
pass-loss contract: pass-0 CE plus its auxiliary loss; extra-pass CE/auxiliary
losses averaged over K-1, with independent aggregate weight gamma (initially
1). Log that K1 lacks this extra CE term. Do not silently sum increasingly
many equally weighted losses as K grows. No autonomous learned-MLP latent
recurrence is planned.

## 5. Next review milestone: O1, native OLMo and sequential RT reference

One bounded migration PR, with **two sequential internal gates**. If ordinary
import reveals a substantial compatibility problem, make that a reviewable
ordinary-only milestone before expanding scope. Model code will remain under
`cdrm/pretrained/`; preserve historical synthetic and OpenELM implementations.
Reuse artifact, validation and immutable-mode infrastructure where applicable,
without building an unnecessary general model-family framework.

### Gate A: establish the unmodified checkpoint function

1. Download the full native artifact, verify its hash, resolve every source
   config default, and pin tokenizer/source behavior. Retain checkpoint and
   provenance under a new OLMo artifact lineage.
2. Strictly load all native tensors with no backbone initialization, cropping,
   vocabulary changes or tying conversion. Check the count, shapes, finiteness,
   ownership and save/reload behavior. Construct optimizers only after final
   parameter materialization.
3. Run an independent pristine-source ordinary reference, isolated from the
   modified local `olmo` namespace. Use the official HF conversion as a
   secondary cross-check only after verifying its actual tensor mapping and
   tokenizer behavior; resolve discrepancies against native artifacts.
4. Check short text/code fixtures, positions, causality, masking, full versus
   cached/chunked execution, logits/loss and **all parameter/input gradients**.
   Tiny tests can use synthetic configurations; actual-checkpoint comparisons
   should remain bounded (B1, roughly 32–64 tokens for ordinary execution).
5. FP32 with TF32 disabled is the semantic reference. Add one bounded native
   mixed-precision/source-parity check and describe differences against FP32.
   Do not reopen the historical precision campaign by default.

### Gate B: adapt the native sequential RT reference

1. Preserve the same loaded weights and native LayerNorm/SwiGLU/RoPE behavior.
   Implement selected-layer scans and a separately constructed history oracle.
2. Compare alpha **0, 0.37 and 1**, including the direct input branch at
   fractional alpha, temporary self-attention and delayed persistent writes.
   Tiny cases cover bottom/top/both; actual checkpoint starts at B1/T16,
   bottom layer only.
3. Check outputs/loss, all parameter/input gradients, causality, cached
   continuation and attached chunked gradients. Exercise multiple outstanding
   forwards with different immutable modes sharing the same weights.
4. Record per-tensor errors as well as aggregate errors. Distinguish source
   parity, floating-point reduction differences and actual derivative defects.
   Explain tolerances rather than importing unrelated synthetic error budgets.

Deliverables: pinned import manifest, native adapter, independent reference,
bounded RT validation, usage/results, W&B evidence, GCS retention receipt and
updated handoff. **No learning run, FBT, NextLat or tiled implementation in O1.**
A source-fidelity pass does not by itself clear large-batch BF16 RT training.

## 6. Subsequent milestones and review order

| Milestone | Scope and review outcome |
| --- | --- |
| **O2: practical RT backend** | Native OLMo RoPE and fractional-alpha exact tiled forward/backward against O1; recomputation/checkpointing, cached decoding and shared-weight gradient ownership. Bounded actual-runtime precision and throughput checks. |
| **O3: language-model objectives/platform** | Independently check NextLat alignment, detachment and masks; optimizer/save/resume; actual batch and memory profiling. Two-GPU correctness when available. |
| **O4: first learning comparison** | Ordinary / ordinary+NextLat / RT / RT+NextLat, FBT off; same original checkpoint, data exposure, optimizer reset/warmup and held-out evaluation. |
| **O5: feedback and interaction** | Establish ordinary-versus-FBT control; then motivated FBT+NextLat, FBT+RT and combined comparisons. Reuse compatible controls and report earlier adaptation if a later experiment starts from adapted weights. |
| **O6: interpretation/follow-ups** | Finite-pass versus exact online behavior, placement/checkpoint-age studies and late frozen-feature diagnostics if main results justify them. |

In a separately authorized milestone, FBT-only reference development can proceed
after ordinary import is accepted, independently of RT backend progress. It
does not expand O1's scope. A negative RT-only pilot does not veto the separate FBT
hypothesis. The first no-FBT learning comparison uses bottom-layer recurrence;
placement/depth extensions follow interpretable controls, rather than silently
making every layer recurrent.

For RT adaptation, retain a gradual alpha transition with matched token and
update budgets in RT and RT+NextLat. The previous starting proposal remains
at least 10M valid input tokens and 200 optimizer updates in the ramp, followed
by at least comparable alpha-1 exposure before judging recovery. Freeze the
actual schedule after throughput and initial scale measurements. Do not jointly
ramp alpha, beta and new loss weights in the first diagnosis. If Q/K scale
proves problematic, consider a separate identity-to-normalized branch with a
matched ordinary control; it is not a prerequisite now.

Select one initial domain, plausibly math or code, on which this **252B-age**
checkpoint has measurable ordinary capability, plus general-language retention
evaluation. Freeze dataset revisions, held-out splits, packing/document masks,
response masks and decoding before comparative runs. A model at task floor
cannot establish that feedback is ineffective. Conversely, maturity beyond a
compute-optimal budget does not make adaptation pointless.

Provisional budget ladder, to be refined after profiling and review:

- **1–10M valid input tokens:** loss/gradient/optimizer health and rough cost;
  not a conclusion about research efficacy.
- **50–100M:** recovery review, particularly after introducing feedback; first
  opportunity to repair a clearly maladapted transition.
- **100M, then 250M:** initial learning comparisons and explicit review points.
- **500M–1B:** conditional extension if direction/cost warrant it, not an
  automatically authorized run.

Record examples, valid input tokens, supervised response tokens, optimizer
updates and total pass compute separately. Compare equal data exposure first;
add a compute-matched control before claiming efficiency. The author's Nanochat
result followed approximately 10.5B feedback-training token positions before
short SFT; it does not establish that SFT alone or our shorter screens must
recover its gains. See the retained [budget audit](fbt-nanochat-budget-audit.md)
and [architecture review](fbt-architecture-and-auxiliary-loss-review.md).

## 7. Precision, attention backends and resources

The research fork is closer to OLMo than to OpenELM, but its existing recurrent
paths are not already a validated importer or native-RoPE tiler. Preserve the
completed numerical work as methodology in the
[numerical handoff](rt-numerical-handoff.md), not a blanket precision guarantee.
OpenELM's Stage B BF16 observations neither clear nor condemn this checkpoint.
Use bounded tests of the actual new execution path; expand investigation only
when a material discrepancy warrants it.

Ordinary OLMo and ordinary attention within FBT can use a suitable SDPA fused
backend. Exact RT still needs its specialized causal schedule and backward.
FA4/CuTE remains an optional optimization after correctness and profiling,
not a dependency imposed by choosing OLMo. Record actual dispatch, dtype,
checkpointing and graph settings rather than assuming a dependency means use.

At the selected bottom layer, OLMo stores 16x128 KV elements per token versus
OpenELM's 4x64: **8 times the KV width in that layer**. This is not an 8x
whole-model memory or runtime estimate: OLMo also has fewer layers and different
MLP/attention geometry. Re-profile physical batch size, context, head-chunked
logits, optimizer storage, persistent recurrent state and activation
checkpointing on the H100; do not carry old batch-size promises forward.

For multiple GPUs, replicated data parallelism improves global batch and may
improve throughput, but does not combine per-GPU activation memory. The
historical tiled autograd accumulates block-parameter gradients internally;
existing launch code does not prove DDP/FSDP safety. Begin with a bounded
two-rank explicit post-backward gradient-reduction baseline, clip after global
reduction, and verify against one-device accumulated gradients. Then consider
optimizer-state sharding; deeper sharding may require a custom-autograd redesign.
Check tying and unchanged imported weight hashes after wrapping. Never reuse a
launcher that reinitializes pretrained weights. There is no multi-GPU hardware
or long-run assumption in the next milestone.

## 8. Retained and deferred decisions

- Preserve the completed OpenELM source, tests, checkpoint, results and retention
  receipts as a working reference. Do not run its next tiled milestone now.
- Original OLMo lacks Q/K normalization; monitor attention/state scales during
  adaptation, but keep native math until evidence motivates a distinct ablation.
- No OpenELM, Qwen, OLMo 2 or Nanochat fallback is automatically selected. Model
  changes require revisiting this recorded choice rather than silently swapping.
- Feature matching remains a **late diagnostic**: use a frozen baseline encoder
  with pinned provenance, preferably this same OLMo checkpoint, and distinguish
  feature deviation from task quality. Do not add its training loss initially.
- Semantic Tube Prediction, full EBFT, autonomous NextLat MLP rollout and new
  cross-pass objectives remain deferred.
- Cross-family changes in results cannot be attributed solely to layer scaling;
  OpenELM versus OLMo also changes norms, attention, vocabulary, data and age.

## 9. Durable execution record

Future model code belongs under `cdrm/pretrained/`; use a new runtime lineage
such as `.runtime/olmo1b-step60000/` and a corresponding prefix below
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/`. These are proposed locations,
not claims of an existing OLMo upload. Retain full checkpoint hashes, source
snapshots, configs, tokenizer fixtures, results, run URLs and storage receipts.
Keep graphable work online in W&B `taylorbollman/pretrained-fbt-rt-nextlat`.

All CUDA, training, evaluation and profiling must run inside the project
container; verify `nvidia-smi` there. Never silently substitute CPU. Persistent
project files survive sessions; local SSD contents do not. Keep credentials out
of logs. Read the handoff for container and retention recovery details.

**Current review boundary:** O4's first bounded four-arm recovery pilot is now
authorized/running on one H100. Each arm uses20.856M valid input tokens and2,634
updates, with matched data/order and a gradual bottom-layer RT transition.
Review its learning/retention results before more exposure or O5. O3's earlier
disposable optimizer recovery and zero-LR profiling are separate evidence.
The released NextLat 1B LM
recipe adds 82.7M training-only predictor parameters (factor 1.6, horizon one,
latent/KL coefficients 1/1); it differs from the historical A5 recipe. The
untrained predictor's initial auxiliary loss scales are recorded in O3 and should
inform the adaptation review. Read the handoff/queue state for current jobs and
recovery paths. Multi-GPU validation requires hardware not currently available.
