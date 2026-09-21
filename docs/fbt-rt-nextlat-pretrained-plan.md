# Pretrained OLMo with FBT feedback, selected RT layers, and optional NextLat

**Historical initial proposal.** The authoritative
[v3 plan](fbt-rt-nextlat-research-plan-v3.md) now selects original OLMo-1B at
step 60,000 (approximately 252B tokens), after the intervening OpenELM work.
Returning to the OLMo family does **not** reactivate this document's late
checkpoint choice, old milestone order or original conditional approach to
fractional-alpha support. Read the [current handoff](fbt-rt-nextlat-handoff.md).
The equations and design history below remain reference material, subject to
the current plan's explicit decisions.

Prepared 2026-09-21. **Proposal only:** this task investigated sources and code;
it did not download model weights, modify model/training code, or launch training.
The prior three-layer synthetic continuation independently completed at 7,500
updates. This is a new research lineage, not a change to that experiment.

The input is the user's [implementation brief](../../fbt_rt_nextlat_implementation_brief.md).
The recommendation is to preserve a substantially pretrained ordinary OLMo
backbone, add shared-weight execution modes, and establish a useful research
platform in reviewable stages. Late adaptation improving quality is a hypothesis;
it must not be conflated with implementation correctness.

**1. Starting checkpoint and architectural contract**

Start with **`allenai/OLMo-1B-hf`**, pinned to
`aee7752d9c08ee4775e9b0091426d8410e8f6a89`, rather than mutable `main`.
The original family was pretrained on approximately 3T tokens. The public Hub
metadata and immutable config were checked during planning, but actual weight
loading/equivalence remains implementation work. The final-model revision is
distinct from the also-available named late branch `step738020-tokens3094B`
(`c81ab7671941a71f0df5cdb961a7eb3ad0418734`); do not claim the two contain
identical tensors without checking. [Official model card](https://huggingface.co/allenai/OLMo-1B),
[pinned HF config](https://huggingface.co/allenai/OLMo-1B-hf/blob/aee7752d9c08ee4775e9b0091426d8410e8f6a89/config.json).

| Property | Proposed pretrained backbone |
| --- | --- |
| Parameters | 1,176,764,416, before new feedback/predictor parameters |
| Layers / width | 16 / 2,048 |
| Attention | 16 query and 16 KV heads; head width 128; full causal history |
| Position | RoPE, theta 10,000; original 2,048-token context limit |
| MLP | SwiGLU, intermediate width 8,192 |
| Normalization | Non-affine pre-LayerNorm; original final LayerNorm; no Q/K norm |
| Projections | No biases; no QKV clipping |
| Embedding / readout | Tied; 50,304 matrix rows |
| Tokenizer | Checkpoint's OLMo/GPT-NeoX-derived tokenizer, not our T5 or synthetic tokenizers |
| Vocabulary handling | Preserve all checkpoint output rows and reference normalization; distinguish tokenizer IDs from padded matrix rows |

This is a different size and architecture from our 150M-backbone RT experiments
and current small GELU/ALiBi models. We reuse the implementation foundation, not
their dimensions, learned Q/K normalization, task head, or tokenizer.

Two verified alternatives are useful to record, not to implement concurrently:

| Alternative | Why it is not the first target |
| --- | --- |
| `allenai/OLMo-1B-0724-hf`, revision `d7cbab742d80589e714b1a2d7f838dcd21cbe143` | Credible stronger base-model alternative, approximately 1.280B parameters and 3.05T training tokens. Same broad block family, but untied head, QKV clipping at 8, and context 4,096. Existing code supports those features; it is not intrinsically blocked. Original OLMo is chosen to minimize initial variables and retain existing weight tying. |
| `allenai/OLMo-2-0425-1B` | Approximately 1.485B, different normalization placement, learned RMS/QK norms, theta 500,000, and 100,352 vocabulary rows. Requires a separate compatibility contract. |

The July model's published config differs from our local
`recurrent-transformer/configs/official-0724/OLMo-1B.yaml`: that local YAML is
not checkpoint authority. [July model/config](https://huggingface.co/allenai/OLMo-1B-0724-hf),
[OLMo2 config](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json).
No mature public checkpoint for the RT paper's exact 150M/300M experimental
configuration was verified. The repository being an OLMo fork does not establish
that such a checkpoint exists. [RT repository](https://github.com/geniucos/recurrent-transformer).

**2. The computation to implement**

Use zero-based positions/layers/passes. Let K count *all* full-stack evaluations:
K=3 is ordinary pass 0 plus two feedback passes. Set selected recurrent layers
S={0} initially; support S={15} and S={0,15}, and S={} for FBT alone.
Selected layers use full history, with no inherited window-2 restriction.

Let e_t=E[token_t], z_t^{p,L-1} be the final block's residual output, and

\[
h_t^p=N_{\mathrm{final}}(z_t^{p,L-1}),\qquad
\mathrm{logits}_t^p=W_{\mathrm{out}}h_t^p.
\]

The feedback state and NextLat state are **post-final-normalization** h.
The top recurrent block, when selected, writes from its own **pre-final-
normalization** z. These are separate named tensors.

Pass 0 executes every layer as an ordinary transformer, using the original
embeddings and input-derived historical KVs. This holds regardless of configured
K, selected layers, or NextLat. There is one parameter set and one optimizer;
ordinary and recurrent execution are modes of that same set.

For p>0, the brief's cross-gated feedback is

\[
G(e,h)=(W_Uh)\odot\sigma(W_Ge).
\]

The inspected Nanochat-based FBT **reproduction** additionally normalizes the
gate's token input and fused output and returns final-normalized top states.
This does not establish the original paper's exact implementation. The following
is an explicit proposed adaptation branch:

\[
\widehat G(e,h)=s_f\,\operatorname{RMSNorm}
\left((W_Uh)\odot\sigma(W_G\operatorname{RMSNorm}(e))\right).
\]

This is the pinned reproduction's `gate_product` path with a proposed OLMo-specific
output scale s_f. Initialize that scale from the checkpoint's ordinary embedding
RMS so the new branch does not silently replace small pretrained inputs with
unit-scale inputs. Record the measured scale, normalization epsilon, and any
learnable scale parameter. Initialize only new fusion matrices using a recorded
independent RNG stream; do not zero the entire state pathway. Preserve ordinary
embedding scale and all pretrained residual/normalization behavior.
[FBT source, lines 102–107 and 727–775](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py).

The full-feedback input is u_t^p=Ghat(e_t,h_{t-1}^{p-1}); independent document
starts use e_t. Tokens, positions and next-token labels stay fixed across passes.
Rebuild current-pass caches. Previous-pass top states provide feedback; previous-
pass KVs never stand in for freshly computed recurrent memory.

For a selected pre-norm block, with input x_t and completed output z_t, define
its own attention-input normalization A and headwise rotary map R_t:

\[
q_t=R_tW_QA(x_t),\quad
\widetilde k_t=R_tW_KA(x_t),\quad
\widetilde v_t=W_VA(x_t),
\]
\[
z_t=\operatorname{BlockFinish}\left(x_t,
\operatorname{Attn}(q_t,[k_{j<t};\widetilde k_t],
                                  [v_{j<t};\widetilde v_t])\right),
\]
\[
k_t=R_tW_KA(z_t),\qquad v_t=W_VA(z_t).
\]

`BlockFinish` retains the checkpoint's attention output projection, residuals,
FFN normalization and SwiGLU. Write one persistent KV entry only after completing
z_t. The temporary self pair is never added as a second historical entry.
RoPE acts on Q/K at their actual positions, never on values or feedback states.
Both bottom and top recurrence means bottom sweep, ordinary middle layers, top
sweep on every feedback pass.

Optional transition controls remain separate from the endpoint:

\[
u_t=(1-\beta)e_t+\beta\widehat G(e_t,h_{t-1}),\qquad
m_t=(1-\alpha)x_t+\alpha z_t.
\]

Use A(m_t) for persistent writes; full hybrid means alpha=beta=1.
Beta=0 must bypass fusion normalization exactly. Beta=0 alone does **not** disable
RT writes. Pass 0 always takes the actual ordinary branch. Do not jointly ramp
alpha, beta and loss weights in the first diagnostic: it obscures causes.
Implement cheap beta control early; add general tiled alpha only if adaptation
requires it. The reference can exercise alpha0/intermediate/1 independently.
Intermediate alpha needs a direct input-gradient path as well as the output path.

**3. Code organization and the actual missing features**

Use new modules such as `cdrm/fbt_rt_checkpoint.py`, `fbt_rt_model.py`,
`fbt_rt_recurrent.py`, `fbt_rt_losses.py`, `fbt_rt_cache.py`, new configs and
new training/evaluation drivers. Preserve historical synthetic driver/config
contracts and strict resume manifests. Any required shared-kernel changes should
be narrow and tested against existing behavior.

| Existing foundation | New work |
| --- | --- |
| Sequential OLMo, SwiGLU, RoPE, `input_embeddings`, `pre_logits` | Audited HF weight importer and explicit per-call ordinary/hybrid modes |
| Reference and tiled RT forward/backward | RoPE in temporary/persistent projections and both backward reconstruction paths |
| Conversion ownership checks | One canonical parameter set, without constructing a second recurrent copy |
| Ordinary cached attention | Exact recurrent step, hybrid prefill caches, mixed ordinary/recurrent prefix boundaries |
| Synthetic NextLat predictor/regression | LM token alignment, detached-head KL, document/padding masks, per-pass reductions |
| Existing numerical/retention/W&B infrastructure | Focused validation and new run/checkpoint schemas |

Prefer retaining selected blocks' canonical fused QKV weight for the unchanged
ordinary path, with non-owning slices used by recurrent projections. Register and
optimize each parameter once. Import HF Q/K/V in the correct concatenation order;
local SwiGLU combines **up then gate**, not the reverse. Strictly account for
every pretrained tensor and tied aliases; missing keys must not trigger random
replacement of backbone weights.

Current local recurrent paths explicitly reject RoPE, cached decoding and packed
documents. Existing `recurrent_layers` chooses classes at construction time;
it is not a per-pass switch. See [model implementation](../recurrent-transformer/olmo/model.py)
and [configuration](../recurrent-transformer/olmo/config.py).

Two details deserve specific tests rather than a broad numerical campaign:

- The tiled autograd function stores the block and rereads configuration during
  backward. Save mode, positions, masks and alpha in immutable invocation state;
  never toggle a shared global block config while earlier pass graphs remain live.
- Tiled backward internally accumulates parameter `.grad`. Check one combined
  K=3 loss against an ordinary-autograd reference, including direct pass-0 and
  indirect feedback contributions. Do not use parameter-only `autograd.grad`
  as the sole test of this implementation, clear gradients between passes,
  or update weights before all pass losses backpropagate.

**4. NextLat and the objective**

Keep NextLat an optional auxiliary predictor, never a replacement forward step or
an autonomous MLP recurrence. Use the existing residual dynamics architecture
where it matches the pinned source, but implement new LM losses. Our historical
synthetic wrapper has SmoothL1 only and does not supply the requested KL/masks.

For each supervised pass:

\[
\widehat h_{t+1}^{p}=F_\psi(h_t^p,e_{t+1}),\qquad
L_h^p=\operatorname{mean}_{\mathrm{valid},D}
\operatorname{SmoothL1}(\widehat h_{t+1}^p,\operatorname{sg}(h_{t+1}^p)),
\]
\[
L_{\mathrm{KL}}^p=\operatorname{mean}_{\mathrm{valid}}
D_{\mathrm{KL}}\left(\operatorname{sg}(p_{t+1}^p)
\;\Vert\;\operatorname{softmax}(\operatorname{sg}(W_{\mathrm{out}})
\widehat h_{t+1}^p)\right).
\]

Both distributions concern the token after state t+1, i.e. token t+2.
Match the released one-step LM layout: T-1 latent transitions, T-2 KL positions
before boundary masks. Do not apply final normalization again to predicted h.
Keep source h and next-token embedding attached, target h/distribution stopped,
and detach only the auxiliary readout *use*. With tied weights, embedding routes
still train that shared matrix. Mask cross-document and padding transitions
separately from main CE: SmoothL1 requires a valid same-document pair (t,t+1),
while KL requires a valid same-document triple (t,t+1,t+2). Freeze the EOS and
document-ID convention; do not reuse the two-position mask for KL.
[Pinned NextLat implementation](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_nextlat.py).

Set A_p=c_h L_h^p+c_KL L_KL^p and use explicit pass reduction:

\[
L=L_{\mathrm{CE}}^0+\eta_0 A_0+
\mathbf1_{K>1}\frac{\gamma}{K-1}\sum_{p=1}^{K-1}
\left(L_{\mathrm{CE}}^p+\eta_{\mathrm{fb}} A_p\right).
\]

Ordinary CE predicts token t+1 and is averaged over valid target tokens. Latent
and KL terms are averaged over their respective valid transitions (latent loss
also averages coordinates), not equally over variable-length documents.
Start with gamma=1 as the FBT-style choice. This fixes aggregate extra-pass weight
between K2 and K3, but a K1 batch has no extra CE term; log this difference.
NextLat coefficients remain independent and are zero for initial mechanical
feedback tests, then enabled after source-equivalence checks. Choose pilot values
from pinned LM configuration and observed loss/gradient scales rather than
automatically copying synthetic weight1. [FBT Eq.12](https://arxiv.org/html/2608.08888v1#S3.SS3).

**5. Prefill, decoding, and a useful mathematical invariant**

There are three distinct evaluations: ordinary transformer; finite-pass hybrid
prefill using previous-pass states; exact online hybrid using the freshly
completed preceding token's state. A finite-pass prefill generally approximates
the exact online computation; do not call their discrepancy a cache bug.

First support at least one hybrid prompt pass, retain caches from that final
pass, and then decode with one stack traversal per new token. Store previous
top state, absolute/document-relative positions, and cache provenance. A selected
layer's prompt cache contains output-derived writes only when that final prompt
evaluation actually used them.

Next support ordinary-prefill followed by hybrid continuation. The ordinary
prefix keeps input-derived entries; new suffix entries become output-derived.
Prefix-mixin training must switch both input fusion and write semantics at the
boundary. Initial data should use one independent document segment per row;
packing is added only after reset/masking checks, or safely processed as separate
segments. Every layer's attention, feedback carry and auxiliary losses respect
independent-document boundaries.

Our existing ordinary cache stores unrotated keys. Specify a new cache's rotation
convention explicitly, and convert through tested adapters where needed; do not
silently pass rotated keys into the old API. Cached/uncached comparisons must
use identical prefix provenance and the same exact sequential semantics.

**Derived correctness invariant:** with fixed tokens, fixed boundaries,
deterministic operations, no jitter, and exact same-pass RT sweeps, pass p matches
the exact online hybrid through at least position p (zero-based). Pass0 is already
correct at the first token, which has no history or feedback. Inductively,
position p's previous-token feedback is correct on pass p, and its current-pass
causal predecessors are correct too. Thus K=T total passes suffice on T tokens
in exact arithmetic. Test T=4–8 with numerical tolerances.

This finite causal propagation property is our derivation, not a claimed result
from either paper. It does not imply monotonic norm contraction, good K2/K3
approximations, low loss, or stable arbitrarily long generation. Those require
measurement. Freeze prefix masks across passes for this diagnostic.

**6. Milestones and completion criteria**

| Stage | Deliverable and bounded acceptance check |
| --- | --- |
| A — Pretrained ordinary model | Pin weights/config/tokenizer and reference implementation; exhaustive import report; compare logits, NLL, post-finalnorm states and cached next-token outputs on fixed text. Match FP32 math first, then intended runtime. Check tied ownership and save/reload. No feedback yet. |
| B — Exact shared-weight hybrid | Small OLMo-shaped FP32 reference with bottom/top/both selection, RoPE, cross-gated feedback, immutable pass context, K1/K2/K3 and exact sequential reference. Check pass0 equality, causality, write order, parameter reuse, document resets and finite causal convergence. |
| C — Tiled RoPE and caches | Match reference outputs, input/parameter gradients and one optimizer step; include K3 shared-weight accumulation, nonzero position offsets and irregular short lengths. Validate cached decode and both prompt boundary types. Retain internal RT recomputation. |
| D — Optional LM NextLat | Source-level predictor/loss comparison, latent/KL indexing and detached-path tests, tied-head behavior, document masks and a combined feedback+NextLat backward. NL-off must recover the same forward computation and gradients as the preceding milestone. |
| E — Real 1B feasibility | On H10080GB, run a bounded actual-checkpoint forward/backward/update/resume check and profile K1/K2/K3 at T128 then T512, with bottom-only and bottom+top. Choose physical microbatch from measurements. Publish memory/time and remaining scope. |
| F — Short adaptation pilot | Freeze data and training protocol; compare ordinary continued training, FBT alone and bottom-RT+FBT at the same token exposure, then add NextLat. Review ordinary quality retention, hybrid quality and actual sequential generation before extending budgets or top-layer experiments. |

Stages A–B are the first useful review point: verified pretrained loading and
the intended computation, before investing in optimized training. C–E form the
first practical platform milestone. Failure at these stages is localized rather
than interpreted from a long training curve.

Keep numerical work proportionate. Use a tiny FP32 reference, one same-state
BF16/protected-attention check at the real shape, and a trained-state recheck only
if the initial pilot suggests a concern. Follow the [numerical handoff](rt-numerical-handoff.md)
for methods, not its entire historical test matrix or an assumption that previous
ALiBi/single-pass results validate new RoPE/multi-pass behavior.

**7. H100 execution and initial research protocol**

FP32 eager is the semantic reference. For practical 1B training, provisionally
use BF16 matrix operations with FP32 parameters/Adam and the existing protected
recurrent-attention policy, after the bounded comparison. Keep full FP32 feasible
as a low-microbatch fallback if measured overhead is acceptable. Do not assume
the small-synthetic FP32/BF16 speed comparison transfers to this model.

Approximately 1.177B FP32 parameters, gradients and two Adam moments require
about 17.5GiB before activations/workspaces/new modules. This is accounting, not
a measured fit claim. A single H10080GB is a reasonable pilot target. Memory and
throughput must include all retained pass graphs, vocabulary logits and NextLat.
Start physical B1,2,4,8 at T512 as feasible; choose accumulation separately.

Checkpoint ordinary layers as needed; reuse tiled RT's internal recomputation.
Do not wrap the whole multi-pass graph in checkpointing until shared-gradient
recomputation is verified. Chunk CE/KL head work if needed, preserving gradients
to *all* attached pass states; the old head-only detached-leaf trainer cannot be
copied unchanged. Start eager, then consider compile/graphs only if profiling
shows material benefit. Dynamic K requires separate capture contracts if added.

For the first directional pilot, use a fixed, versioned general-text sample
(Dolma-family data is the default recommendation), its official tokenizer, and
held-out documents. T512 is an initial resource choice, not a new model context
limit. A 5/Fuzzy can later be added as separately specified probes; their old IDs
and heads cannot be transplanted into the pretrained LM unchanged.

Use a bounded mechanics run of roughly 32–100 updates, explicitly exercising
K2/K3, before a token-budgeted learning pilot. A provisional next budget is
10M input tokens per arm, reviewed for useful movement before extension; it is
not enough to rule out the research hypothesis. Do not inherit synthetic
LR3e-4. Start a conservative adaptation proposal around backbone LR1e-5 and
new-module LR1e-4 with short warmup, AdamW and norm1 clipping; finalize after
measuring initial loss/state/gradient scales and expected update count. These
are proposed adaptation settings, not paper reproduction claims.

Keep K1 batches present. A provisional mature mixture is K1/K2/K3=75/22/3,
with explicit K3 exercise in the mechanics run so rare sampling cannot hide a
bug. New fusion may need a short beta transition or a fusion-only warm start;
treat either as a recorded adaptation choice. Do not require a long new ordinary
pretraining phase: the selected checkpoint already supplies substantial training.

Use this comparison order, rather than launching every combination immediately:

1. Ordinary continued training and FBT alone establish adaptation and data controls.
2. FBT+bottom RT tests the main addition at identical starting backbone/data.
3. Add NextLat to the same hybrid; include FBT+NextLat and ordinary+NextLat controls
   before attributing a benefit specifically to the interaction.
4. Evaluate top-only/bottom+top placement and RT-only controls as subsequent arms.
   With feedback disabled, repeated deterministic RT-only passes are redundant;
   one extra recurrent evaluation suffices for that control.

Evaluate CE/NLL separately for ordinary mode and K2/K3, diagnostic K4/K8, fixed-
token exact sequential hybrid, and free generation. Record held-out retrieval
and state-tracking probes when the task interface is frozen. Compare equal data
exposure first, and plot quality against actual wall time as well; a later
compute-matched control is needed before claiming efficiency gains.

Log per-pass loss, raw/weighted NextLat terms, gradient/clipping statistics,
gate saturation, latent/input RMS, pass-to-pass changes, predictive-distribution
differences and prefix-conditioned retention. A converged latent is not itself
evidence that feedback or RT is useful. Bottom recurrence can propagate a
feedback-informed state within the same pass, but can also make optimization
harder or harm retrieval; neither outcome is assumed.

Count unique/input-token exposure separately from backbone pass evaluations,
selected-layer recurrent token updates, train seconds, prefill/decode latency,
peak memory and physical microbatch. At the proposed mixture the average is
1.28 stack evaluations per batch, but RT sweeps can cost more than ordinary
layers, so 1.28 is not a wall-time multiplier. FBT's low serving overhead does
not establish the hybrid's prefill/training cost. Measure persistent-write
projection overhead in decoding too.

Use online W&B under `taylorbollman` (suggested project `olmo-fbt-rt-nextlat`),
new local runtime/report paths, and verified artifacts under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/`. Preserve checkpoint/config and
source hashes, tokenizer, data/packing metadata, pass mixture, coefficients and
ramps, optimizer/RNG/data cursor, and a resumable checkpoint at each review point.
All GPU work must use the project container and verify GPU access inside it.

**8. Decisions to retain for the next session**

Recommended first target is original OLMo-1B-hf; July2024 is an available alternate,
not an unsupported feature request. Preserve pretrained RoPE and block semantics.
Do not reinitialize selected blocks or manufacture a pretrained RT checkpoint.
Bottom-only is the first hybrid; top/both are required selectable paths. Use
post-finalnorm feedback/NextLat, pre-finalnorm top-block writes, full history,
attached cross-pass gradients, and exact same-pass recurrent caches.

The unresolved experimental decisions are the final adaptation corpus/sample,
total compute budget, and whether first quality pilots prioritize general LM,
state tracking, retrieval, or reasoning. They do not block stages A–E; freeze
them before long training. A mature checkpoint may require considerably more
than a tiny fine-tuning budget to learn useful new recurrence. A short negative
pilot would diagnose immediate adaptation behavior, not settle that question.
