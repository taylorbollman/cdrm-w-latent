# Pretrained feedback and RT research: checkpoint choice, intermediate experiments, and multi-GPU

**Implementation status:** the user approved Stage A on2026-09-21. Its ordinary
OpenELM import is implemented and validated; see the
[implementation handoff](fbt-rt-nextlat-handoff.md) and
[first-PR results](reports/openelm-import/results.md). Subsequent milestones
below remain staged research plans.

Revised 2026-09-21, including the follow-up on weight tying, early RT+NextLat,
the author's Nanochat reproduction, public checkpoints, upstream feature
provenance, OpenELM attention/layer geometry, and feature-matching/STP follow-ups. **Planning only; no model changes or
training are authorized by this document itself.** This supersedes the checkpoint recommendation and milestone
order in the [first plan](fbt-rt-nextlat-pretrained-plan.md). Its causal feedback,
cache, NextLat and shared-gradient contracts remain applicable with the changes
below. Historical synthetic runs and their source/checkpoint records stay intact.

**Recommendation and research judgment**

Pursue the direction as a staged experiment. Start with **OpenELM-1.1B** at the
intended approximately 1B scale, subject to checkpoint/tokenizer import checks.
The user's final model choice supersedes the briefly proposed 450M-first pilot;
450M remains an optional resource fallback, not a scheduled preliminary run.
Its native tied embeddings, Q/K normalization and actual intermediate checkpoints
make it a good fit for this question. The first RT comparison is now the full
four-arm **ordinary / ordinary+NextLat / RT / RT+NextLat** comparison, with FBT
off. Do not postpone RT+NextLat until RT alone succeeds. Build FBT independently
and establish an ordinary-versus-FBT control before interpreting its interaction
with RT. All three switches remain independent.

The author's Nanochat reproduction is useful evidence. Its exact newer tied
d20 checkpoint was not found publicly, but **public pretrained Nanochat weights
do exist**, including an older untied 561M base-d20 and 2.22B d34. Nanochat does
not inherently require pretraining from scratch. The
[public-checkpoint comparison](fbt-pretrained-nanochat-options.md) corrects that
earlier inference and records the precise compatibility differences.

Keep OpenELM-1.1B as the main path. Public Nanochat base-d20 remains an audited
alternative, but both pretrained and scratch Nanochat experiments are
deprioritized. The [author-reproduction audit](fbt-nanochat-budget-audit.md)
still supplies useful training-time and adaptation-budget evidence.

The [architecture and auxiliary-objective review](fbt-architecture-and-auxiliary-loss-review.md)
records the latest refinements: the newer Nanochat backbone additions were
inherited upstream, not demonstrated FBT-specific requirements; the successful
reproduction is approximately 863M total parameters, not the public 561M model.
Retain frozen-feature rollout evaluation as a late diagnostic after the main
architecture comparisons. Defer Semantic Tube Prediction (STP), full EBFT
training and new cross-pass consistency losses to future work; none is an
initial requirement or an automatically queued training ablation.

The [attention and layer-scaling audit](fbt-openelm-attention-compatibility.md)
records native causal SDPA, the distinction between FlashAttention and exact
RT tiling, and the required per-layer geometry. FA4/CuTE is an optional
acceleration path, not a prerequisite for the research experiment.

The evidence justifies a pilot, not a prediction of success. The RT paper's
12-layer 300M C4 CE improves from 2.896 to 2.867, while its reported throughput
is 42k versus 132k tokens/s. This is a positive loss result with a substantial
training-cost tradeoff. It does not establish mature-checkpoint conversion,
selective recurrence, or programming/math SFT gains. Downstream accuracy is
mixed. Our A 5 evidence motivates testing interactions, but does not establish
natural-language reasoning gains. [RT paper](https://arxiv.org/html/2604.21215v1#S7.SS2).

RT-only is cheaper to train and easier to interpret than the full combination.
It still needs the difficult RoPE/GQA recurrence implementation. FBT-only is
actually cheaper to implement because it can retain ordinary pretrained
attention; it is a useful early control while the recurrent backend is developed.
A weak RT-only pilot will not automatically veto FBT+RT or RT+NextLat.

**Source-provenance correction**

The earlier plan treated `xidulu/Full-bandwidth-transformer` as the released FBT
implementation, and the next revision did not sufficiently emphasize its
authorship. **It is maintained by Xi Wang, FBT's first author.** Its README
describes a Nanochat reproduction using no Microsoft assets. That makes it a
valuable author-provided implementation and experimental reference, while still
distinct from the original backbone/training. Its ReLU-squared MLP differs from
the paper's Appendix A description. No public original-paper FBT checkpoint was
verified in a bounded check. [Author identity](https://huggingface.co/papers/2608.08888),
[repository's own description](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/README.md),
[paper publication](https://www.microsoft.com/en-us/research/publication/full-bandwidth-transformer/).

Use the paper as algorithmic authority. The reproduction may supply inspectable
implementation examples, explicitly labeled as such. Carrying post-finalnorm h,
normalizing the token-gate input, and calibrating a feedback output scale are
our recorded adaptation choices where the original publication does not settle
the precise implementation. Do not import unrelated reproduction-backbone features.
In particular, upstream already supplied token-value tables, residual/input
mixing, smear/backout, Q/K scaling and attention windows. These may interact
with feedback, but their presence does not establish that the FBT author added
them because feedback required them. See the linked commit audit.

**1. Which FBT ingredients matter?**

| Ingredient | Recommendation and reason |
| --- | --- |
| Causal top-state feedback with a one-token shift | Defining mechanism; preserve exactly. Previous-pass states in training, freshly completed previous-token states in exact online decoding. |
| Asymmetric state-value/token-gate fusion | Preserve the proposed cross-gate endpoint. A permanent additive token bypass would alter this experiment. |
| Controlled feedback magnitude and compatible input scale | High priority. Add normalization/scale control on the new branch while preserving pretrained ordinary inputs. |
| Attached gradients and explicit per-pass loss reductions | Required to train the intended model; do not detach cross-pass state to save memory. |
| Ordinary-mode supervision, deeper-pass exposure, correct prompt boundary | Retain; the exact K1/K2/K3 percentages are configurable, not a mathematical requirement. |
| Native Q/K normalization | Desirable for attention-logit scale when memory sources and input distributions change. It does not replace normalization of the persistent value source or guarantee stability. |
| Native embedding/readout tying | Desirable shared basis and lower parameter cost. Useful, but not mathematically necessary: the feedback projection can learn a basis mapping. |
| RoPE, RMSNorm, SwiGLU | Preserve the chosen checkpoint's versions, placement and dimensions. Do not retrofit them simply to approximate the paper's feature list. |
| Exact residual-depth scaling, headwise attention gates, sliding-window layout, optimizer | Potentially consequential, but not prerequisites for the defining feedback experiment. Preserve the pretrained backbone initially and report differences. |

The paper describes a tied, QK-normalized, RoPE/GQA model and discusses feedback
stabilization, but does not isolate every ingredient sufficiently to establish
which architectural choices are indispensable. These priorities are engineering
judgments, not paper-established ablation conclusions.
[FBT architecture and method](https://arxiv.org/html/2608.08888v1#A 1).

The previous alpha/beta notation remains distinct: alpha changes the RT memory
source, beta changes feedback fusion, and NextLat weights change only auxiliary
supervision. Avoid turning all three ramps on during the first diagnostic.

**2. Model choice and checkpoint authority**

| Candidate | Relevant fit | Main tradeoff |
| --- | --- | --- |
| **OpenELM-1.1B — primary** | Approximately 1.080B; tied readout, per-head Q/K RMSNorm,32k vocabulary, real intermediate checkpoints | GQA and different attention/FFN dimensions by layer; requires native block adapters. |
| **OpenELM-450M — optional fallback** | Approximately 457M; native tying/QK normalization and public checkpoint ages | Cheaper experiments, but retains generalized backend requirements; not evidence at 1B scale. |
| **Public Nanochat base-d20 — simpler backend alternative** | Approximately 561M; pretrained base, Q/K RMSNorm, uniform full MHA and relatively simple residual blocks | Older untied 65536-vocabulary backbone; preserve it and port FBT rather than load into the author's newer model. |
| **Public Nanochat d34 — larger fallback** | Approximately 2.217B; public pretrained base with uniform full MHA and Q/K RMSNorm | Older untied backbone and more compute/storage; not a smaller first experiment. |
| **Qwen3-1.7B-Base — alternate** | Approximately 1.721B; tied readout, per-head Q/K RMSNorm, uniform layer geometry, native Transformers model | Larger model and 151,936-row vocabulary; no official age-labeled intermediates verified. |
| **Original OLMo-1B — simplest engineering fallback** | Approximately 1.177B; tied readout, uniform full MHA, closest existing code, extensive checkpoint history | No Q/K normalization; inserting it creates a separate adaptation question. |

These are within-family research options, not parameter-matched baselines.
Do not interpret cross-family raw benchmark differences as evidence for recurrence.
OLMo2 is not preferred merely to obtain Q/K norm: it also changes residual norm
placement and has an untied readout. Qwen3-0.6B-Base is a possible cheaper fallback,
but still needs GQA and decoupled attention/residual widths.
[OpenELM config](https://huggingface.co/apple/OpenELM-1_1B/blob/ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f/config.json),
[Qwen base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json),
[OLMo config](https://huggingface.co/allenai/OLMo-1B-hf/blob/aee7752d9c08ee4775e9b0091426d8410e8f6a89/config.json).
Pins, public artifact checks and source links for the 450M/Nanochat additions are
in the [checkpoint comparison](fbt-pretrained-nanochat-options.md).

OpenELM target details:

| Property |450M optional fallback|1.1B primary|
| --- | --- | --- |
|HF artifact/code pin|`apple/OpenELM-450M@b53a9c5a731d154b71f8d311ef702f327e0cfa3a`|`apple/OpenELM-1_1B@ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f`|
|Layers / residual width|20 /1536|28 /2048|
|Query heads / KV heads|12–24 /3–6|16–32 /4–8|
|Head dimension / GQA ratio|64 /4:1|64 /4:1|
|Attention width before output projection|768–1536|1024–2048|
|Initial selected RT layer / optional top layer|0 /19|0 /27|

Both have layer-varying SwiGLU widths, learned per-head Q/K RMSNorm before RoPE,
theta 10000, trained context 2048 and tied input/readout. Preserve each native
layer's exact dimensions; a larger rotary cache allocation is not evidence of
longer-context training. Verify the native Llama1/2 tokenizer's legitimate
access, hash, BOS insertion, EOS and padding behavior before proceeding.

Apple publishes individual checkpoints including 300k and 350k, an average of
the last five, and optimizer checkpoints. The final family was trained on about
1.5T tokens; 1.8T describes the available dataset mixture, not training exposure.
The proposed conversion pilot starts from the **individual 300k checkpoint**,
after importer validation, rather than silently identifying the HF final/averaged
artifact with an individual step. [Official checkpoint table](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/README-pretraining.md),
[OpenELM paper](https://arxiv.org/html/2404.14619v1).

There is a concrete native/HF conversion difference: the native pretraining
config has **32,128 embedding/readout rows and padding ID32,000**, whereas the
HF release has 32,000 rows. Native forward uses the full tied readout. For the
300k checkpoint preserve native rows, softmax denominator and padding semantics
for equivalence; do not silently crop using the HF config. Verify the actual
downloaded tensors and report their exact parameter count. The approximate
457M/1.080B family sizes remain useful, but native/HF byte-level identity does
not follow from those rounded counts.
[450M native config](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/pretraining_configs/openelm_450M.yaml),
[1.1B native config](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/pretraining_configs/openelm_1_1B.yaml).

The pilot's exact model-only URL is
`https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_299999.pt`.
Its full-state counterpart is `training_checkpoint_epoch_0_iter_299999.pt` in
the same directory. The optional 450M fallback has corresponding files in the
`pretrained/450M/` directory. Availability was checked with metadata-only requests; weights
were not downloaded. At implementation, record object metadata and SHA256 after
download, pin the CoreNet config/code, and audit conversion against the native
model. Choose one starting weight artifact for all paired pilot arms.

Keep 300k for the question about modifying a mature model. Its nominal pretraining
exposure is approximately 1.258T positions. If starting age becomes the suspected
obstacle, the available50k checkpoint (approximately 209.7B positions) provides a
strong within-family contrast;100k is a less extreme alternative. Evaluate each
against its own ordinary continuation. Do not equate Nanochat's 40k/60k fraction
with OpenELM's 300k/350k fraction or infer a sufficient adaptation budget from it.

For Qwen, the verified **base**, not chat/thinking, alternate is
`Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
It has 28 layers, D2048, Q16/KV8, head 128, FFN6144, and RoPE theta1,000,000.
Its stored weights are BF16; casting them to FP32 creates an arithmetic reference,
not recovery of unrounded pretraining weights. Keep its own tokenizer/RoPE.
[Official base-model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base).

**3. Features that can be learned after a late checkpoint**

New fusion matrices, feedback normalization/scale, an auxiliary predictor and
RT memory-source interpolation can be trained during continued training with
a fresh warmup. A fresh schedule is justified because the model's computation
and possibly data distribution change. Use the same optimizer reset/warmup in
matched ordinary controls; having original Adam state available does not imply
it must be restored for SFT.

Some changes are not initially function-preserving:

- Adding Q/K normalization with unit gain generally changes attention logits.
  If needed later, an identity-to-normalized interpolation can start at zero,
  with an explicit ramp and matched control; it is a separate experiment.
- Forcing an untied embedding/head to share weights generally changes outputs.
  Prefer a natively tied checkpoint or learn the new branch's basis mapping.
- Replacing LayerNorm with RMSNorm, changing RoPE, or rescaling every pretrained
  residual path changes its learned computation. Avoid bundling those changes
  with RT/FBT adaptation.

Warmup helps optimization; it is not a guarantee that any architectural surgery
will preserve learned capabilities. Native Q/K normalization and tying are good
reasons to consider OpenELM, not reasons to add an extra normalization study first.

**Weight tying: concrete implementation**

There is no contradiction between feasible tying and the warning about untied
checkpoints. OpenELM was trained with a shared matrix already. With
E in R^(V x d), embedding lookup and readout are

\[
e_t=E[\mathrm{token}_t],\qquad \mathrm{logits}_t=h_tE^\top.
\]

Preserve that single owned parameter. The native HF implementation has no
separate `lm_head` module in tied mode and calls `F.linear(h, embedding.weight)`.
Our adapter must resolve the native readout weight rather than assuming
`get_output_embeddings().weight` exists. Do not clone it into a second trainable
matrix, change vocabulary rows, or count it twice in the optimizer. Check tying
after import, device materialization, save/reload and distributed wrapping.
[Pinned OpenELM implementation](https://huggingface.co/apple/OpenELM-1_1B/blob/ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f/modeling_openelm.py).

NextLat auxiliary KL uses the same matrix with only that readout use detached:
`F.linear(predicted_h, embedding.weight.detach())`. Input/conditioning embedding
paths and source hidden states stay attached. Ordinary CE and those paths still
train E. Globally freezing E would implement a different objective.

The author's newer Nanochat fork also already implements its tied mode,
including optimizer ownership and checkpoint aliases. The public older d20/d34
checkpoints are untied; that tied mode must not be imposed on their pretrained
matrices. Their viable route preserves both matrices and learns the feedback
projection, with the loss of native tying reported explicitly. For a genuinely untied model, a
gradual readout interpolation toward E with continued training/distillation is
possible, but endpoint equivalence is not guaranteed. No such conversion is
needed for the OpenELM route or for an explicitly untied Nanochat adaptation.
Input/readout tying is separate from sharing
the same stack weights across FBT passes.

**4. Independent modes and generalized attention contract**

| FBT | RT | Backbone execution | NextLat |
| --- | --- | --- | --- |
| Off | Off | Ordinary single pass | Independently off/on |
| Off | On | Single pass with selected recurrent layers | Independently off/on |
| On | Off | Ordinary pass0 followed by FBT passes with ordinary attention | Independently off/on |
| On | On | Ordinary pass0 followed by FBT passes with selected RT layers | Independently off/on |

This supports all eight combinations without requiring eight initial training
runs. Preserve **K1=ordinary in the FBT multipass API**. Standalone RT has a
separate mode and does not pay for an unused ordinary pass. NextLat remains
training-only auxiliary supervision, never autonomous predictor rollout.

Default selected set is {0}; top is **27** in the primary 1.1B model (**19** in
the optional 450M fallback), not the OLMo-specific index15. Derive top-only and bottom+top selections from the loaded
configuration; use full causal history. Each block adapter exposes
its native norms, projections, Q/K-head layout and MLP. Preserve native parameter
ownership/layout rather than forcing every family into OLMo fused projections.

For selected layer input x_t, output z_t, native input norm A, native per-head
key norm N_K and rotary R_t, persistent writes use

\[
m_t=(1-\alpha)x_t+\alpha z_t,\quad
k_t=R_tN_K\!\left(\operatorname{heads}(W_KA(m_t))\right),\quad
v_t=\operatorname{heads}(W_VA(m_t)).
\]

Temporary Q/K/V are input-derived, with the checkpoint's Q/K normalization
order and own position. Complete z_t before publishing its persistent pair.
The group map shares each KV head among its corresponding query heads. A tiny
reference may repeat KV **activations** and sum gradients back through the group
map; never create independently trainable duplicate KV parameters. Production
storage should retain native KV-head counts. Record any expanded temporary
working memory during initial profiling.

Our tiled kernel currently assumes full MHA, attention width equal to residual
width, and OLMo-style full-vector Q/K norm. Those are actual changes to make,
along with RoPE and fractional alpha. This is more work than the first OLMo plan.
Reuse tiling/recomputation methods behind native block adapters, not a claim
that arbitrary HF models already fit the kernel.

**Attention backends and layer-wise scaling**

OpenELM's scaling changes internal attention and FFN dimensions, while every
1.1B residual state remains 2048-dimensional. Layer0 uses Q16/KV4 heads of
dimension64, attention width1024 and FFN width1024; layer27 uses Q32/KV8,
attention width2048 and FFN width8192. This is compatible with per-layer RT
caches, 2048-to-2048 FBT fusion and a shared final-state NextLat predictor.
Preserve native dimensions and weights. Do not widen small layers to simplify
kernel shapes. Bottom-versus-top RT comparisons change both location and
recurrent-layer capacity, so they are placement sensitivities within OpenELM,
not controlled tests of position alone.

Both native CoreNet and HF OpenELM use PyTorch scaled-dot-product attention.
Use native-equivalent causal SDPA for ordinary attention and ordinary layers
inside FBT passes; verify actual Flash dispatch in the bounded GPU profile.
The pinned HF code has a mask hazard if `_attn_implementation=flash_attention_2`
is forced without adapting its SDPA call. Preserve explicit causality and test
future-token invariance, including cached offsets; a backend flag is not proof
of correct acceleration.

Selected RT layers require their own exact tiling and recurrent backward.
Persistent K/V depend on completed same-layer outputs, so stock full-sequence
FlashAttention cannot replace the recurrent schedule. FA4 can potentially
accelerate ordinary attention, partial tile updates or reconstruction, but must
preserve temporary self K/V and persistent historical K/V. Current code declares
an FA4 dependency but implements tiled RT with compiled PyTorch, not CuTE.
The current reconstruction also materializes quadratic attention intermediates;
do not assume FlashAttention's memory behavior for this RT backward.

Finite FBT passes have known previous-pass inputs and can run this layer-wise
RT schedule. Exact online feedback decoding remains sequential across tokens;
do not claim that tiled RT makes unknown future feedback inputs available.

After import parity, keep a tiny FP32 oracle and verify the chosen mixed
precision path on the actual OpenELM geometry. During the existing feasibility
stage, record ordinary/RT kernel use, forward/backward memory and throughput.
Only pursue custom FA4/CuTE integration if those measurements justify it. Public
FA4 training uses FP16/BF16 Q/K/V; it does not replace a full-FP32 reference.
Implementation details and pinned sources are in the linked attention audit.

**5. The first RT research experiment: RT x NextLat, without FBT**

Start four paired arms from the same checkpoint; all have FBT off:

| Backbone computation | NextLat off | NextLat on |
| --- | --- | --- |
| Ordinary | Ordinary control | Ordinary + NextLat |
| Bottom-layer RT conversion | RT | RT + NextLat |

The RT arms share a memory-source schedule. The NextLat arms share predictor
initialization, coefficients, masks and activation time, starting together in
the bridge phase. Initialize new modules with an independent RNG stream so they
do not perturb pretrained weights or data order. NextLat is auxiliary training;
neither arm rolls out the predictor as an additional inference recurrence.

Use alpha0 to establish ordinary output **and gradient** equivalence against
the corresponding NextLat-off/on control. Ramp alpha to 1 by nonpadding input
tokens, then hold it at 1. Tentatively allocate at least 10M tokens and 200 updates
to the ramp, then at least the same exposure at alpha1 before SFT. This replaces
the previous1M-ramp/1M-hold proposal; profile and freeze the exact schedule before
comparison. No FBT ramp is present. Keep NextLat coefficients fixed initially;
if calibration requires a ramp, apply it identically in both NextLat arms.

Compare intermediate-alpha gradients explicitly. Tiled backward must send the
memory-source adjoint to both `(1-alpha)x` and `alpha*z`, including norm
derivatives. Alpha is immutable per invocation. **Fractional tiled alpha is
required scope**, rather than an optional follow-on.

The former10M-token efficacy budget was too optimistic as a general adaptation
test. At the reproduction's 524,288-token batch it is only about 19 optimizer
updates. Its new feedback pass initially performed very poorly; among the
sampled checkpoints in our audit, the first lower BPB than ordinary continuation
appears around 1B additional positions. That does not establish a necessary
budget, a precise crossover, or RT's required budget, but it argues for explicit
recovery intervals and against a strong negative conclusion from a smoke test.
See the [audited curve and accounting](fbt-nanochat-budget-audit.md).

Use the following **provisional per-arm budget ladder**, after throughput and
ordinary-baseline learning checks:

| Exposure | Purpose |
| --- | --- |
| Approximately 1M–10M | Smoke/optimization check; not an efficacy decision |
| 100M, then 250M input tokens | Initial recovery/learning screens and review points |
| Approximately 1B, optionally2B | Conditional extension if recovery is credible and measured cost is acceptable |

For FBT specifically, add a 50M–100M recovery review: flat/worsening new-pass loss
deserves investigation even if quality gains are not yet expected. A recovering
branch still slightly behind its ordinary control is different from a branch
that never learns. Review at 500M–1B before a larger extension; do not infer a
universal threshold from the author curve or transfer these timings mechanically
to RT/NextLat. The [budget review](fbt-architecture-and-auxiliary-loss-review.md)
distinguishes failure modes, token exposure and unique data.

Use both tokens and optimizer updates to define the schedule. A 32k–64k-token
effective batch is a starting profiling range, giving roughly 1,500–3,000 updates
per 100M, not a claim of optimal batch size. Preserve a useful physical batch for
RT throughput. At a 100M endpoint, the provisional ramp/hold occupies at least 20M
tokens, with the balance available for focused SFT; all four arms receive the
same phase boundaries. If adaptation remains unsettled at the proposed SFT
boundary, review before changing distributions rather than hiding the issue in
SFT. The FBT-only control has its own feedback calibration and may use continued
pretraining throughout its initial screen.

These are proposals, not automatic authorizations for four billion-token runs.
Compare equal input exposure first, record actual GPU cost, and make each
extension a concrete review point. A substantially shorter SFT-only experiment
can still be useful, but answers a narrower question about that recipe.

Save/evaluate all arms at the ramp endpoint and bridge-to-SFT boundary. Report
general-text retention in each arm's intended inference mode, plus domain CE.
Ordinary-mode evaluation of RT-adapted weights is an additional compatibility
metric, not its main retention score. Material unresolved degradation should
trigger review before SFT.

Prefer programming first, with executable grading. Select a modest public
code-SFT subset after a tokenizer/length/quality audit, then freeze it; a small
fresh program-execution probe provides additional state-tracking diagnostics.
Math is a separate next domain rather than part of the initial mixed objective.
If the ordinary model is at floor on intended tasks, choose an informative
difficulty or reconsider the base before spending on an architectural comparison.

Keep starting parameters, optimizer/schedule policy, training IDs/order, context,
physical batch, loss masks, token budget and generation protocol matched. Use
fresh optimizer state and warmup in all arms by default. Backbone LR around
1e-5 is only a starting proposal; calibrate briefly and freeze before comparisons.
Do not reuse synthetic LR3e-4. Train the backbone; selected-only or LoRA training
would introduce an additional restriction and is not the default.

For SFT, use the same prompt/answer formatting and response-token CE mask in all
arms. Use NextLat regression on valid same-document transitions, including
prompts, and auxiliary KL only where its associated next-next target is a
supervised response token. Run recurrent prefill as well as decode in RT arms.
Ordinary-prefill
followed by RT decode is a separate boundary condition, not an accidental shortcut.
Track valid input tokens, response tokens and optimizer steps independently.

Measure executable pass rate/answer accuracy, answer-token CE, length-conditioned
performance, general-text NLL retention, wall time and GPU-seconds. Use independent
train/development/test task IDs and avoid near-duplicate programs across splits.
Freeze evaluation decoding and max generation tokens. Compare paired task outcomes;
repeat promising findings with a second training seed before claiming robustness.

Retain checkpoints and fixed evaluation prompts for a later frozen-feature
rollout diagnostic, toward the end of the main comparisons. Its provisional
design uses one fixed original OpenELM encoder for every arm, 256 prompts, four
sampled continuations and horizons8/32/128. The unbiased cross-sample CFM
estimator and prompt-level uncertainty are described in the
[evaluation protocol](fbt-architecture-and-auxiliary-loss-review.md). It can be
computed retrospectively and does not delay the early pilots. It supplements
task accuracy and CE, uses no new training loss, and remains separate from
same-token finite-pass/sequential FBT discrepancy measurements.

Review criteria:

- At alpha0, failure of equivalence is implementation failure.
- At alpha>0, transient quality loss is possible; observe recovery and behavior
  after a real alpha1 training interval, not only during the ramp.
- If every arm fails, the test is inconclusive about recurrence.
- If RT remains badly degraded, try at most one documented transition remedy
  (e.g. longer ramp/bridge), keeping a matched ordinary budget, before reviewing.
- If RT is competitive or better at useful cost, extend and replicate.
- Evaluate RT+NextLat even if RT alone disappoints. Compare it with both RT and
  ordinary+NextLat. A separate FBT+RT comparison tests feedback-conditioned
  inputs and is not logically ruled out by the no-FBT comparison.

For a higher-is-better score, record the interaction estimate
`(RT+NextLat - RT) - (ordinary+NextLat - ordinary)`, with paired evaluation
uncertainty where feasible. This asks whether NextLat helps RT more than it
helps the ordinary backbone; it is additional to the practical question of
which arm performs best. A single-seed difference is preliminary evidence.

This first comparison concerns one selected layer and one adaptation recipe.
It does not establish the merit of all-layer RT or every layer placement. Top-only is
the next placement sensitivity if the question warrants it.

**6. FBT and NextLat follow-on comparisons**

Implement FBT-only after ordinary loading, independently of RT backend work.
Keep shifted feedback, cross-gate endpoint, branch scale control, attached
cross-pass gradients and explicit CE reductions from the original brief. Use
post-finalnorm feedback/NextLat states and pre-finalnorm top-block RT writes.
Validate deterministic short-prefix convergence and cached online semantics.

Once the no-FBT four-arm comparison and FBT-only control are interpretable,
extend to FBT, FBT+NextLat, FBT+RT and FBT+RT+NextLat as useful, starting from the
**same original checkpoint** and matched data exposure. Reuse a compatible
FBT-only control rather than retraining it solely because it ran earlier. The
code supports all combinations from the outset. A run beginning from an already
RT-adapted checkpoint is an additional sequential-adaptation experiment and must
include its earlier token/compute budget. Do not silently compare it with a fresh
FBT arm and credit all differences to RT.

NextLat follows the pinned source's one-step hidden regression plus separately
weighted detached-head KL. Source latent/conditioning embedding remain attached;
the target role is stopped, and auxiliary readout detachment does not globally
freeze a tied embedding. SmoothL1 masks valid same-document pairs; KL masks valid
triples. Prompt-only versus response-only auxiliary supervision must be specified
for SFT rather than inherited from synthetic CE masks.

Track ordinary quality and exact sequential hybrid generation as well as finite
K2/K3 prefill; test K4/K8 for extrapolation. Use extra-pass loss averaging, some
K3 training and consistent ordinary/hybrid prefix cache semantics. Compare equal
data exposure first, then actual time/compute; add a compute-matched control
before claiming efficiency. No long full matrix is implied by this plan.

**7. Multi-GPU: correctness first, then useful memory savings**

Two GPUs do not become one GPU with twice the local activation memory. Replicated
data parallelism increases total batch/throughput; it does not enlarge the
physical microbatch seen by one recurrent MLP. Gradient accumulation also does
not improve that per-call matrix size. Sharding model/optimizer storage can free
local VRAM, whereas activation-heavy cases may need checkpointing and head chunks.

The current tiled custom Function hides parameters in its block argument and
internally accumulates `.grad` through repeated nested backward calls. Existing
DDP/FSDP launch code does not prove reducer/sharding compatibility. The old launcher
also resets model parameters after wrapping; a pretrained driver must verify
post-wrap/broadcast weight hashes and must not reinitialize loaded weights.
See [local tiled implementation](../recurrent-transformer/olmo/model.py) and
[historical launcher](../recurrent-transformer/scripts/train.py).

Recommended progression:

1. **Two-rank correctness baseline:** replicate the model, finish complete local
   backward/accumulation, then explicitly all-reduce FP32 gradients once per
   update in bounded buckets. Clip after global reduction, then step. This avoids
   reducer callbacks inside the token loop, at the cost of communication overlap.
2. **Optimizer-state sharding:** test a ZeRO1-style optimizer with weights and
   local gradients replicated. This can free VRAM without changing recurrent
   parameter lifetimes during the scan. PyTorch ZeroRedundancyOptimizer without
   overlap is a candidate, not a pre-validated integration.
3. **Further sharding if needed:** refactor tiled autograd to take parameters as
   tensor inputs and return their accumulated gradients, then validate DDP/FSDP.
   Wrap complete recurrent blocks, retaining their weights through a whole scan
   and its backward. Another option is replicated selected RT blocks with a
   sharded ordinary stack. Do not wrap per-token projections independently.

DDP static-graph support can handle some reentrant cases, but changing K or
optional parameter usage is not automatically compatible with its assumptions.
FSDP changes parameter views/lifetimes and may introduce all-gathers. Defer tensor
or pipeline parallelism across the recurrent sweep; tokenwise communication and
load imbalance may dominate. Pin APIs to the actual container runtime.
[DDP documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html),
[distributed optimizers](https://docs.pytorch.org/docs/2.14/distributed.optim.html),
[FSDP](https://docs.pytorch.org/docs/2.14/fsdp.html).

Idealized FP32 parameter+gradient+two-Adam-moment storage is 16 bytes/parameter.
For the optional 450M fallback this is approximately 6.8GiB before activations; optimizer-only
sharding reduces it to about 5.1GiB/rank on 2 GPUs or4.3GiB on 4.
For OpenELM1.08B this is about 16.1GiB; optimizer-only sharding reduces it to about
12.1GiB/rank on 2 GPUs or10.1GiB on 4. For Qwen1.72B the corresponding estimates
are25.6/19.2/16.0GiB. These exclude activations, new modules, communication buckets,
temporary casts and readout work. Measure, rather than assume, useful batch gains.

The first two-GPU check compares the same global examples with a one-GPU reference:
loss, every gradient, global clipping, actual Adam update and replica equality.
Exercise alpha0/intermediate/1, variable valid-token counts, K1→K3→K2, NextLat
off/on, and fresh-process resume. Globally normalize each loss by its own valid
token/transition count; do not average unequal local means. Synchronize K and
other mode schedules across ranks initially. Globally unused new parameters
must retain skip-update semantics; replacing every missing gradient with zero
can still cause weight-decay/momentum updates.

Then profile actual T128/T512 and physical microbatch candidates, recording
fixed-global-batch and fixed-per-GPU-batch results separately. Report memory per
GPU, tokens/s, communication time, and GPU-seconds/token. Schedule this early
once two GPUs are available, before committing to expensive multi-pass runs.
Single-GPU import and semantic work need not wait for that hardware.

**8. Revised milestone order and review points**

| Stage | Concrete outcome |
| --- | --- |
| A. Checkpoint/import | OpenELM-1.1B native/HF reference pinned, individual late checkpoint mapped, tokenizer resolved, ordinary outputs/gradients/cache and causal attention verified. |
| B. Native-block RT reference | RoPE, per-head QK norm, GQA, variable dimensions and alpha0/intermediate/1, initially bottom layer; same owned weights. |
| C. Tiled/backend platform | Matching backward and cached decoding, shared K3 gradient test, explicit immutable modes; LM NextLat losses independently checked for the early comparison. |
| D. Single-/two-GPU feasibility | Actual attention dispatch and forward/backward memory/throughput; bounded optimizer/resume checks, explicit synchronization first and optimizer sharding if useful. Optional FA4 evaluation only where justified. |
| E. Early learning comparisons | Ordinary/ordinary+NextLat/RT/RT+NextLat, one focused domain and retention evaluation; matched FBT-only control can run earlier while RT backend work proceeds. |
| F. Feedback interactions | FBT±RT±NextLat as motivated; retain compatible earlier controls, then consider placement, checkpoint age and longer budgets. |
| G. Late feature diagnostic | Retrospective frozen-feature rollout evaluation of retained baseline and adapted checkpoints; no feature-matching training objective. |

FBT-only reference development can proceed after A without waiting for C.
The scratch Nanochat miniature in the [audit](fbt-nanochat-budget-audit.md) is now
deprioritized in favor of OpenELM-1.1B, with 450M and public base-d20 retained only
as alternatives. A Nanochat experiment must use its checkpoint's native version:
the public older backbone has no value-embedding bypasses, while the author's
newer backbone does. Keep native block adapters and the same RT/NextLat switches;
avoid building a second independent training platform. Full author d20 pretraining
is not included in the default plan.
Do not run long combinations before a faithful import, validated new gradients
and measured cost. No broad numerical rerun is proposed: use tiny FP32 references,
focused BF16 comparisons at the actual new geometry, and escalate only when a
material concern appears. Eager first; compile/graphs later if worthwhile.

New graphable runs use W&B entity `taylorbollman`, suggested project
`pretrained-fbt-rt-nextlat`. Retain weights, data manifests, checkpoints and reports
under `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/` and persistent project paths.
Use project Docker for all GPU work. Pin source/config/tokenizer/checkpoint hashes,
data IDs, schedule and loss masks, per-rank RNG/cursors and distributed optimizer
state. The decisions still requiring a final run protocol are exact SFT subset,
physical/effective batch, adaptation token budget, and available multi-GPU topology.

**9. Prioritized follow-ups from the two suggested papers**

Feature matching remains a late evaluation connection to the suggested paper,
after the main architecture comparisons. Full EBFT entails sampled-token
policy-gradient training and a frozen feature encoder; that training objective
remains deferred. Preserve checkpoints so the diagnostic can compare initial
and adapted models without rerunning training.

Semantic Tube Prediction remains future work. The reviewed ordinary/FBT x STP
off/on design is a possible later experiment, not an approved or queued stage.
If revisited, choose explicitly between the paper's three-position formula and
the released span-complement implementation, and distinguish within-pass
temporal regularization from a new cross-pass objective.

Neither temporal straightness nor lower feature loss guarantees useful task
behavior or feedback contraction. The [review](fbt-architecture-and-auxiliary-loss-review.md)
records equations, pinned source, evidence limits, masks and minimal monitoring.
Cross-pass distillation/consistency is a separate later method, motivated only
if observed finite-pass versus exact-feedback mismatch warrants it.
