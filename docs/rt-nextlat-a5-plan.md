# Plan: two-layer RT with NextLat, starting on A5

Prepared 2026-09-11; approved with a scope correction from the user.
Implementation is now authorized. Evaluate transformer backbones only.
The auxiliary MLP serves one-step training; do not implement or run free
latent-recurrence evaluation. The comparison is RT+NextLat against RT alone
and an ordinary Transformer trained with NextLat. The pure-RT run remains
stopped at 100,000 updates.

**Recommendation.** Keep our existing two-layer RT backbone and attach one
NextLat dynamics predictor to its final normalized hidden states. Jointly
train the backbone and predictor from the paired original initialization.
Follow the released A5 NextLat recipe: one-step latent supervision, weight
one, ordinary state-classification CE, and no auxiliary KL or predicted-state
CE in the initial run. Evaluate the trained backbone only.

The question is whether combining RT and NextLat improves backbone state
tracking beyond either component alone. Use saved pure RT and ordinary
Transformer baselines, plus new RT+NextLat and ordinary Transformer+NextLat
runs. The user explicitly excludes adjunct MLP recurrence experiments.

**Source authority and a material discrepancy.** Use the supplied
[NextLat v4 paper](../key_research_docs/2511.05963v4.pdf), particularly §3.2,
§5.2 and Appendices C, E and F.5, together with the released A5 branch pinned
at `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`. The relevant implementation is
[`models/model_nextlat.py`](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_nextlat.py),
[`config/a5/nextlat_a5.yaml`](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/config/a5/nextlat_a5.yaml),
and [`data/a5_data.py`](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5_data.py).
Preserve source attribution and license notices for any ported code.

The paper's Table 5 lists predictor hidden width 1,024. The released A5 YAML
sets `proj_factor: 0.5`, and the implementation multiplies that factor by
the concatenated input width, `2D`. At D512 this gives hidden width **512**.
I recommend using this released configuration first, explicitly recording
the discrepancy. Make hidden width configurable so the Table 5 version is
an inexpensive later sensitivity check, not a silent substitution.

There is another naming trap: the upstream predictor's `LayerNorm` class
uses **RMSNorm** when `bias=false`, as in A5. Match that operation inside the
predictor. Retain our existing LayerNorm in the RT backbone. The source is
[`models/model_base.py:805–831`](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_base.py#L805-L831).

**1. Exact model and latent definition.**

| Component | Proposed initial setting |
| --- | --- |
| Backbone | Existing two tiled RT blocks, both recurrent, rho=1 |
| Backbone width / heads / FFN | 512 / 8 / 2,048, GELU |
| Backbone normalization / position | Existing LayerNorm and learned full-width QK normalization; ALiBi |
| Backbone vocabulary | 60 input operations and 60 state classes; untied embedding/output matrices |
| Latent | Final RT output **after** final LayerNorm, D512 |
| Dynamics input | Concatenate next-operation embedding first, current latent second |
| Dynamics network | RMSNorm(1,024, epsilon=1e-5), Linear 1,024→512, GELU, Linear 512→512, GELU, Linear 512→512 |
| Dynamics output | Add predicted delta to unmodified current latent; no output normalization |
| Dynamics initialization | Linear weights Normal(0, .02), no biases; RMSNorm scale one |
| Dropout | Zero throughout |
| Runtime | Full FP32, autocast/TF32 off, eager execution, no compile or CUDA graphs, one GPU |

The unchanged backbone has **6,357,504 parameters**. The released-width
predictor adds **1,049,600**, giving **7,407,104** total. These counts include
the existing input embedding and output head once each. The Table 5 width
alternative would add 2,622,464, giving 8,979,968 total. Report the backbone,
predictor, total and actually used inference components separately; the
augmented training model is not parameter-matched to pure RT.

Let x_t be the observed permutation operation and y_t the cumulative A5
state after applying operations x_1 through x_t. Define

\[
h_t = F_\theta(x_{1:t})_t,\qquad
p_\theta(y_t\mid x_{1:t}) = \operatorname{softmax}(W h_t),
\]

where F includes both RT blocks and final LayerNorm. Let e_t=E[x_t] use
the RT's existing token embedding. The added dynamics model is

\[
\widehat h_{t+1}=g_\psi(h_t,e_{t+1})
=h_t+W_3\,\operatorname{GELU}
\left(W_2\,\operatorname{GELU}
\left(W_1\,\operatorname{RMSNorm}([e_{t+1};h_t])\right)\right).
\]

There is one predictor after the complete backbone, not one predictor per
RT layer. Its predictions are not inserted back into the RT blocks during
training. The two types of recurrence therefore have distinct roles: RT
recurrence produces training latents; NextLat learns a transition between
those latents.

The embedding E is shared across backbone and predictor conditioning. W is
shared across ordinary RT decoding and predicted-latent decoding. E and W
remain untied to each other. Predicted latents go directly to W: applying
final LayerNorm again would change the released dynamics semantics.

**2. Objective, alignment and gradients.**

For B words of T operations, the initial A5 objective is

\[
\mathcal L =
\underbrace{\frac{1}{BT}\sum_{b,t}
\mathrm{CE}(Wh_{b,t},y_{b,t})}_{\mathcal L_{\mathrm{state}}}
+\underbrace{\frac{1}{B(T-1)D}\sum_{b,t<T,j}
\mathrm{SmoothL1}_{\beta=1}
\left(\widehat h_{b,t+1,j},\operatorname{sg}(h_{b,t+1,j})\right)}
_{\mathcal L_{\mathrm{latent}}}.
\]

This is **d=1, lambda_latent=1, lambda_KL=0, auxiliary lambda_CE=0**.
The source calls its regression term `mse_loss`, but actually computes
Smooth L1. Match its averaging across batch, valid transitions and latent
coordinates; summing coordinates would inadvertently multiply its weight
by 512. Base CE remains active despite F.5's phrase about regression-only
latent supervision. We predict the cumulative state at the same position,
not the next independently sampled input operation.

For a length-12 word there are exactly 11 supervised transitions:
`h[:, :-1]` and `embedding(x[:, 1:])` predict `h[:, 1:]`.
No BOS, language-model shift, transitions between words, or A5 state-label
inputs are introduced. All IDs 0–59 are valid, including identity ID 0.
Do not import LM EOS/padding masks that would discard identity transitions.

| Gradient route | Required behavior |
| --- | --- |
| Main CE to backbone, embedding, output head | Enabled |
| Latent loss to predictor parameters | Enabled |
| Latent loss through source h_t into RT | Enabled |
| Latent loss to next-operation embedding E[x_(t+1)] | Enabled |
| Latent loss through target h_(t+1) in its target role | Stopped |
| Latent loss directly to output head | Absent |

A latent may be a stopped target for one transition and an attached input
for another. There is no frozen teacher backbone, second target encoder or
EMA. Upstream detaches temporary gradient-collecting leaves but explicitly
reinjects their gradients into both the original hidden states and token
embeddings. Copying those initial detaches without the reinjection would
quietly defeat joint training.

Use ordinary attached autograd and form the complete loss before **one
`loss.backward()`**, preserving these routes. Our tiled RT handles its
recurrent backward internally; retain its established `.backward()` route
rather than introducing parameter-only `autograd.grad` as the trainer.

Record `lambda_KL=0` explicitly and require that value in the first A5
implementation. Defer a trainable KL option until an experiment needs it.
For that later extension, its direction is teacher-to-prediction KL, with
detached teacher logits and detached student output-head weights, summed
over classes and averaged over valid transitions. Gradients still reach
the predictor, its source latent and conditioning embedding. The ordinary
CE head stays trainable. No auxiliary hard-label CE is needed initially;
predicted-state CE can be logged as a diagnostic without optimizing it.

**3. Minimal implementation surface.**

Use `backbone(inputs, return_pre_logits=True)`. The existing
[`OLMo.forward`](../recurrent-transformer/olmo/model.py) returns both ordinary
task logits and graph-connected post-final-normalization `.pre_logits`.
No hooks, new hidden-state API, attention rewrite, or RT kernel modification
is needed.

Add a wrapper/predictor and small task drivers in new files such as
`scripts/rt_a5_nextlat.py`, `rt_a5_nextlat_train.py`,
`rt_a5_nextlat_eval.py`, `rt_a5_nextlat_report.py`, with configurations in
`configs/rt_a5_nextlat/`. Reuse the established data loader, A5 metrics,
initialization factory, FP32 context, word order and optimizer grouping.
Keep the architecture selector in the wrapper compatible with ordinary
SEQ as well as RT for the matched SEQ+NextLat control in this milestone.

Leave the historical A5 execution files, OLMo Python tree and
`configs/rt_a5/` untouched. The old resume guard hashes those files, including
every JSON in the old config directory. Even adding an otherwise unrelated
configuration there would invalidate the strict pure-RT continuation. New
hybrid source/config manifests and a new checkpoint schema must include all
new executed dependencies and objective settings.

Construct the backbone with the original canonical seed-1234 SEQ→RT mapping.
Initialize the predictor in a separate recorded RNG scope. Check the
backbone hash against the original initialization, and preserve training
data order independently of module construction. Start joint training from
scratch; adding the predictor to trained RT-100k would be a separate
finetuning experiment with different initial conditions.

Checkpoints must contain the full hybrid model, optimizer, RNG, data order,
objective and predictor configuration, source hashes and the explicit
backbone-only evaluation scope. Each shared parameter should occur only once in the optimizer.
This keeps any later longer run resumable without weakening the old guards.

**4. Evaluate transformer backbones only.**

| Arm | NextLat auxiliary training | Backbone evaluation |
| --- | --- | --- |
| Saved ordinary Transformer (SEQ) | No | Existing 10k development results |
| Saved RT | No | Existing 10k development results |
| New ordinary Transformer + NextLat | Yes | New 10k development results |
| New RT + NextLat | Yes | New 10k development results |

Run RT+NextLat first, then the matched ordinary Transformer+NextLat control.
For both new arms, model evaluation calls the backbone directly and never
invokes the auxiliary MLP. Compute one-step loss/scale diagnostics separately
on bounded development subsets if useful for monitoring auxiliary training;
these are not a recurrent inference route or an additional performance arm.
The MLP is not unrolled or evaluated as an autonomous recurrent model.

Report the same E(t), A(t), M(t), CE and sample identities as the saved
baselines. Compare RT+NextLat against both RT and SEQ+NextLat; the saved SEQ
arm provides the fourth cell for descriptive interaction analysis. A single
seed at a short budget does not establish a reproducible interaction.

**5. Treat training quirks as observations to measure.**

Appendix E reports FineWeb pretraining behavior under learning-rate cooldown:
Smooth L1 can increase while downstream transition usefulness improves.
The authors did not establish a general fix. Their A5 experiment instead
uses constant-LR AdamW. Keep that A5 recipe and do not import Muon, WSD,
loss replacements or target-normalization changes in advance.

Log base CE, raw and weighted latent loss, diagnostic predicted-state CE/KL,
latent/predicted-latent RMS, latent variation, relative prediction error and
the ordinary gradient norm. Use simple scalar summaries rather than an
expensive representation-analysis suite. Inspect actual state predictions
and backbone length curves alongside these values. A change in raw latent loss
alone should not trigger a repair or be labeled a precision failure.

Keep the existing FP32/eager simplification. The released A5 launcher uses
BF16 mixed precision and its YAML enables compilation; our runtime differs
deliberately to match the current baselines and avoid a new numerical
qualification project. Record this difference explicitly.

**6. Bounded correctness checks before training.**

1. Compare the small predictor/loss calculation with the pinned reference
   under copied weights and fixed tensors: normalization, residual, Smooth
   L1 reduction and stop-gradient destinations.
   Include a test that distinguishes a stopped target leaf from an attached
   source leaf and conditioning embedding. Check the equivalence of direct
   autograd to upstream gradient reinjection without importing its trainer.
2. With the auxiliary objective bypassed, recover original RT logits, CE,
   shared gradients and one optimizer update at identical weights/data.
   This is an integration check, not an expectation that the jointly trained
   model will retain the old trajectory.
3. Check all 11 transitions per length-12 word, identity operations,
   same-position labels, no target leakage, no cross-word transitions,
   shared parameter identity and absence of duplicate optimizer entries.
4. Run one tiny FP32 combined-loss check against the existing naive RT
   reference, plus a finite update at the intended B1024/T12. Use existing
   bounded checks rather than reopen BF16, compiler or large-batch tolerance
   studies. Investigate only an actual new failure.
5. Verify backbone evaluation never calls the auxiliary MLP. Run a brief
   small-set overfit check and a tiny fresh-process resume check.

All future GPU model work must run through the project Docker launcher,
after verifying container location and `nvidia-smi`. No GPU work was needed
for this plan. No silent CPU fallback is allowed.

**7. First A5 pilot and review point.**

Each new arm will run **10,000 joint-training updates**, from the paired
initialization, on the existing frozen length-12 corpus. Use physical B1024,
seed and data-order seed 1234, AdamW at constant LR1e-4, betas(.9,.95),
epsilon1e-8, matrix weight decay .01, zero norm decay and global gradient
clipping at one. Apply clipping once across the full trainable hybrid.
Profile a short warmed run first to report memory, throughput and the
actual budget estimate; use that measurement instead of assuming the
predictor adds no cost.

Retain checkpoints at 0, 1k, 5k and 10k. Run small fixed development checks
every 500 updates; perform full 102,400-word short/OOD development evaluation
at the retained trained checkpoints using backbone inference only. Training uses
800,000 words from the frozen million-word corpus. The remaining short and
long development splits and independent confirmation roles stay unchanged.

Report E(t), A(t) and M(t) over positions 1–36: all-prefix exactness,
isolated-state accuracy and mean token accuracy. Emphasize the 12/13/14
boundary and length 36, with integer counts and the existing pointwise
interval convention. Plot both new backbone arms against existing **10k** RT and
SEQ results at matched training exposure. Pure RT-100k is useful longer-run
context, but is not the equal-budget baseline for a new 10k hybrid.

The fixed 10k endpoint is primary; intermediate checkpoints expose learning
and instability rather than provide an opportunity to choose an attractive
endpoint afterward. A weak 10k result does not refute the method: the
paper's reference training lasts 400k updates. Review the joint losses and
backbone comparison curves before authorizing a longer run. Do not automatically
extend pure RT, the hybrid, or any ordinary-Transformer run.

Log under the authorized W&B entity `taylorbollman`, using a distinct
RT+NextLat group in `rt-a5-state-tracking` and separate metric namespaces for
backbone metrics and one-step auxiliary-training diagnostics.
Store new checkpoints, source snapshots, configurations, data references,
metrics and plots under a fresh `gs://fast-chunks/cdrm-w-latent/rt-a5/`
lineage, with verified receipts. Keep the independent confirmation set
unevaluated while making development choices.

The ordinary Transformer+NextLat control is included in this milestone to
answer the user's comparison with either component alone. Both new arms use
identical backbone and predictor initialization values, training words and
optimizer settings. Report both update exposure and wall time: NextLat adds
training compute and parameters even though inference uses the unchanged
backbone. Preserve the three-way primary comparison and the fourth baseline
without claiming broad significance from one seed.

**Reviewable milestones.** First deliver the wrapper/loss/backbone evaluator
and bounded correctness evidence. Then train RT+NextLat followed by
SEQ+NextLat, each to 10k, and deliver backbone length curves, loss/scale
diagnostics, preserved checkpoints and a short interpretation. No free
latent-recurrence implementation, evaluation or experiment is part of this
milestone. The model-specific parts should remain small; no recurrent
attention redesign is expected.
