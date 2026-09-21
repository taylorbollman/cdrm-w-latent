# FBT architecture provenance, pilot budgets, and auxiliary objectives

Prepared 2026-09-21 after the user's final checkpoint/architecture questions and
two suggested papers. Planning only. This supplements the
[current plan](fbt-rt-nextlat-research-plan-v2.md),
[Nanochat budget audit](fbt-nanochat-budget-audit.md), and
[public-checkpoint comparison](fbt-pretrained-nanochat-options.md).

**Current decisions after the user's final scope selection**

- Use OpenELM-1.1B as the primary model. OpenELM-450M remains an optional
  smaller fallback, not a prerequisite pilot; Nanochat is deprioritized.
  A task on which the ordinary model is at floor is not an informative comparison.
- Preserve native OpenELM. Do not transplant every newer Nanochat feature.
- Retain bounded frozen-feature rollout evaluation as a late or post-main
  diagnostic; it is not part of the initial implementation or run protocol.
- Keep Semantic Tube Prediction (STP) as future work, with no ablation queued.
- Defer full Energy-Based Fine-Tuning (EBFT) and any new loss across FBT passes.
  Keep the already-planned early RT x NextLat comparison.

**1. Which Nanochat changes were actually made for FBT?**

The FBT fork descends from upstream Nanochat
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. Comparing its model with the author's
`7037c60924870aca6e30fac95212b0c7caee052d` shows that the attention/MLP backbone
and the listed newer residual/input features were inherited. Their presence
does not show that the FBT author introduced or ablated them for feedback.
[Upstream model](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py),
[FBT model](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py).

| Feature | Provenance | Native OpenELM |
| --- | --- | --- |
| Token-derived value tables on alternating layers | Upstream Nanochat | Absent |
| Learned residual multiplier and layer-0-input skip | Upstream Nanochat | Absent |
| Previous-token embedding mixing and midpoint subtraction | Upstream Nanochat | Absent |
| Additional 1.2 scaling of both Q and K | Upstream Nanochat | Absent |
| SSSL attention-window pattern | Upstream Nanochat | Absent; full causal attention |
| ReLU-squared MLP | Upstream Nanochat | SwiGLU instead |
| RoPE and Q/K normalization | Upstream Nanochat | Present with different native norm/order/geometry |
| Optional tied embedding/readout | FBT fork addition | Already native |
| Shifted cross-gated feedback, pass losses, jitter, prefix handling and feedback decoding | FBT work | New functionality to implement |

Specific upstream commits document
[residual/input mixing](https://github.com/karpathy/nanochat/commit/aa530cdad58123ebfb79ab85d996c4641cfc6c90),
[window attention](https://github.com/karpathy/nanochat/commit/fbc1484e8c2582325e8daa1c1a5000f17aed69e7),
[alternating value tables](https://github.com/karpathy/nanochat/commit/e85db6b4a4351eb562bec220b3bbcaad28be6722), and
[smear/backout/QK scaling](https://github.com/karpathy/nanochat/commit/a825e63f81e62e2e9fd38655e9b2e39417620545).
The author's [initial feedback commit](https://github.com/xidulu/Full-bandwidth-transformer/commit/ca2fe9fe4c417f49ac224668ccec2d2229f0b66f)
is separate. These commits establish provenance, not necessity for FBT.

Two interactions matter for interpretation. The token-value tables look up raw
token IDs even on feedback passes, creating an embedding bypass. The layer-0
skip instead carries the prepared input of the current pass, including fused
state where applicable. They should not both be called raw-embedding skips.

OpenELM is aligned on native tying and Q/K norm, and its SwiGLU resembles the
original paper's MLP more than Nanochat's ReLU-squared does. It does not match
the whole reproduction. Preserve its native norm ordering and residual paths;
stabilize the new feedback branch explicitly instead of altering all pretrained
layers. Retain shifted causality, fusion, bounded input/state scale, attached
cross-pass gradients, pass scheduling and appropriate prompt boundaries.

**2. What model size does the positive Nanochat result establish?**

Do not conflate the public older561M base-d20 with the author's newer model:

| Component in the author's logged d20 | Parameters |
| --- | ---: |
| Transformer matrices |393,217,200|
| Token-value lookup tables |419,430,400|
| Tied embedding/readout |41,943,040|
| Registered feedback alternatives |8,192,000|
| Scalars |66|
| Total |862,782,706|

Only 3,276,800 feedback parameters are active in the cross-gate path; the log
includes dormant alternatives. [Logged counts](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/d20-standard-60k-238647.log).

OpenELM-450M's HF artifact has 457,179,136 parameters, of which49,152,000 are the
tied vocabulary matrix: approximately 408M remain outside that matrix. This
calculation makes the dense-block comparison reasonably relevant, while the
author's extra419M lookup parameters remain real additional capacity.
[OpenELM metadata](https://huggingface.co/api/models/apple/OpenELM-450M),
[pinned config](https://huggingface.co/apple/OpenELM-450M/blob/b53a9c5a731d154b71f8d311ef702f327e0cfa3a/config.json).

The evidence makes a sub-billion fallback plausible, although the current
primary model is OpenELM-1.1B. It does not directly show
FBT gains in a 450M-total-parameter model or establish parameter-matched scaling.
The reported downstream gains remain author-reported; the exact newer math
evaluation artifacts were not found in the inspected repository snapshot.

**3. When should slow adaptation concern us?**

The source curve is summarized in the existing budget audit:20.97B ordinary
input positions before the fork,10.49B additional positions with K2, then math
SFT. Among audited checkpoints, feedback is still behind at 524M and slightly
ahead around 1.05B. The full math result includes roughly 5M SFT examples; exact
SFT token exposure for that lineage was not verified. Do not estimate it by
multiplying examples by maximum context length.

Use this review ladder for our initial FBT adaptation, not as a guarantee:

| Additional exposure | What to assess |
| --- | --- |
|1M–10M|Finite updates, active new-path gradients, intended modes; not efficacy|
|50M–100M|Clear recovery trend in new-pass loss; investigate flat/worsening loss|
|100M–250M|Gap to matched continuation, retained ordinary quality, generation health|
|500M–1B|First serious review of useful quality/efficiency gains if recovery continues|
|1B–2B|Conditional extension for credible but unresolved trajectories|

If the new branch does not learn at all, investigate implementation, scale,
optimizer and curriculum early. If it recovers but remains slightly behind,
that is a different situation. Do not wait for 1B to address NaNs, broken
gradients or severe runaway degradation; do not reject FBT simply for lacking
an advantage at 100M. At 1B without a meaningful trend, review the recipe before
automatically funding10B. These thresholds are engineering decisions informed
by one other model, not universal convergence bounds or RT/NextLat requirements.

Record valid input tokens, unique examples, response tokens, updates, pass counts
and GPU time. At 32k–64k effective tokens/update,100M means roughly 1,500–3,000
updates. OpenELM300k starts after approximately 1.26T nominal positions, a much
more mature checkpoint than the Nanochat fork; this is an additional uncertainty.

**4. Feature matching: retained late diagnostic; EBFT training deferred**

Jelassi et al., *Matching Features, Not Tokens: Energy-Based Fine-Tuning of
Language Models*, targets rollout feature statistics. It uses a frozen feature
encoder and a conditional discrepancy of the form

\[
L_{\rm CFM}=\mathbb E_{c,y}\left\|
 \mathbb E_{\hat y\sim p_\theta(\cdot\mid c)}\phi(c:\hat y)-\phi(c:y)
\right\|^2.
\]

The relevant mismatch is reference prefixes versus model-generated prefixes.
It persists even when parallel and sequential evaluation of the same fixed
tokens are mathematically equivalent. Actual EBFT training uses sampled-token
policy gradients and a frozen encoder, not ordinary latent regression. The
paper reports substantial rollout-training time, so it is a separate project.
[Paper v2](https://arxiv.org/html/2603.12248v2),
[official configuration](https://github.com/sjelassi/ebft_openrlhf/blob/4064b49b1b6e81c614b80b22e2d3905ea96b77c0/configs/qa_code.yaml).

The following bounded evaluation is retained for late or post-main analysis,
adapted to our platform. It is not required before the main learning comparison:

1. Fix 256 held-out prompt/reference pairs and generate 4 continuations per prompt
   up to 128 tokens. Reuse their prefixes at 8,32,128. Maximum131,072 generated
   tokens per arm, excluding prefill. Record length/EOS handling explicitly.
2. Freeze one copy of the original OpenELM checkpoint as a common encoder for
   every arm. Use normalized final-position features at approximately 25%,50%,75%
   depth, with exact layer indices recorded. Cache reference features.
3. Initially use temperature 1 without truncation so the statistic concerns the
   model distribution. Ordinary task scoring may separately use greedy decoding.
   Fix prompt formatting and generation settings across arms. Frozen feature
   extraction can run after generation; it need not reside beside training weights.
4. Use a fixed normalized feature metric as primary. Do not let each arm define
   its own moving encoder or model-dependent whitening geometry. The source's
   whitened training objective is not the same as this evaluation adaptation.
5. Report task success, CE and feature discrepancy together, with uncertainty
   resampled by prompt. Evaluate retained baseline and selected milestone checkpoints, not every
   training update. If feature variance or cost is excessive, adjust the common
   diagnostic protocol before architectural comparison, not separately per arm.

For sampled feature vectors f_i and reference g, use the cross-sample estimator

\[
\widehat L_{\rm CFM}
=\frac{\sum_{i\ne j}f_i^\top f_j}{n(n-1)}
-\frac{2}{n}\sum_i f_i^\top g+\|g\|^2.
\]

The squared sample-mean distance instead includes a model-dependent variance/n
term. Individual unbiased estimates can be negative. A growing discrepancy with
length is not by itself a defect: reference-feature variability contributes a
length-dependent floor. Interpret matched differences at each horizon.
[Source evaluation module](https://github.com/sjelassi/ebft_openrlhf/blob/4064b49b1b6e81c614b80b22e2d3905ea96b77c0/inference_loss/README.md).

Keep three measurements separate:

- Same-token ordinary/RT full-sequence versus cached execution: implementation
  equivalence, with numerical tolerances.
- Same-token finite-K FBT versus exact sequential feedback: feedback training/
  inference approximation; compare states/logits as K increases without changing
  token identities or prompt boundaries.
- Sampled-continuation versus reference features: rollout distribution quality.

RT, FBT and reference-trajectory NextLat do not automatically solve the third
problem. Improvement there would be an interesting experimental finding.

**5. Semantic Tube Prediction: future work, not a queued ablation**

Huang, LeCun and Balestriero propose temporal hidden-displacement alignment:

\[
L=L_{\rm CE}+\lambda\mathbb E_{s<r<t}
 [1-\cos(h_t-h_r,h_r-h_s)].
\]

Its positions are token indices within one pass. It needs no predictor or
additional transformer pass. Evidence includes fine-tuning OpenELM-1.1B-Instruct,
but not our planned base-checkpoint adaptation or RT/FBT. The headline16-fold data reduction uses repeated
epochs to roughly preserve example exposure: it is not16-fold less compute.
[Paper](https://arxiv.org/html/2602.22617v1).

The released `random_span` code aligns a sampled patch displacement with the
sum of before/after displacements, rather than literally sampling the printed
three-point expression. It has no detached teacher, learned predictor or hard
tube-radius constraint. If revisited, pin the variant; released span semantics
with explicit same-document masks would be a reasonable starting choice.
[Official loss implementation](https://github.com/galilai-group/llm-jepa/blob/ea0017c654ad917066ff32afc88276bea8ca5f7e/stp.py#L1172-L1228),
[driver](https://github.com/galilai-group/llm-jepa/blob/ea0017c654ad917066ff32afc88276bea8ca5f7e/run_stp.sh).

No STP experiment is queued. If the user returns to it after the main work,
one possible controlled comparison is ordinary/FBT x STP off/on, holding RT and
NextLat off for that comparison and reusing compatible controls. A later test
could hold NextLat fixed on a promising recurrent combination. Any coefficient,
such as the previously discussed 0.01 starting point, would need calibration
against gradient contribution and development data; it is not validated for
our planned adaptation. One valid sampled span/example is a possible initial
implementation choice.

If STP is revisited for FBT, apply it within each supervised pass and use the same ordinary/extra-
pass averaging convention as CE. Avoid multiplying effective strength by K.
Do not apply this formula across FBT pass indices and call it the published
method. Tokenwise straightness does not imply contraction of the pass update.

Monitor task quality and CE, along with sampled displacement norms, degenerate
near-zero spans, representation rank and auxiliary/main gradient contribution.
Useful state transitions may turn sharply; over-regularization could hurt them,
particularly in state-tracking tasks. Lower geometry loss alone is not success.

A loss specifically coupling FBT passes is a later new-method experiment. If
finite-K/sequential disagreement remains problematic, a direct diagnostic of
that disagreement is more informative first. Adjacent-pass equality can also
encourage the model to ignore feedback or suppress useful refinement, so a
new consistency objective needs controls and should not be bundled into the
initial FBT/RT/NextLat result.
