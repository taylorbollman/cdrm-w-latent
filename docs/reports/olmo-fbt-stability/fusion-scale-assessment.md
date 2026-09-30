# FBT fusion scale and initialization assessment

2026-09-30. Read-only source and retained-report review; no model, training,
probe, or pinned execution source changed.

The raw embedding RMS near 0.04 versus final hidden RMS near 1 does **not**
create a direct 25-fold imbalance in our fusion. The embedding is normalized
before it controls a multiplicative gate. The product is then normalized and
scaled back to the pretrained OLMo input scale. This check supplies no reason
to change the running F-only experiment.

For an eligible position at beta1, the implementation is

\[
R_\epsilon(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+10^{-5}}},\qquad
f_t=s\,R_\epsilon\!\left[(W_Uh_{t-1})\odot
\sigma(W_GR_\epsilon(e_t))\right].
\]

Here `s` is the RMS of the complete pretrained embedding matrix, saved as a
fixed buffer when fusion is constructed. It is neither learned nor recomputed
from each minibatch. RMS reductions use FP32. At beta1 there is no additive
embedding residual on eligible positions; the first position retains its
ordinary embedding. Fractional beta would introduce a blend, but this run
uses beta1. See [native fusion](../../../cdrm/pretrained/olmo_fbt.py#L113)
and [blend](../../../cdrm/pretrained/olmo_fbt.py#L266).

The fixed eight-row F0 panel reports:

| Quantity | RMS |
| --- | ---: |
| Ordinary input embedding | 0.03983261 |
| Final hidden state | approximately 1.000000 |
| Fused stack input, pass4 | 0.03707686 |
| Fused stack input, pass32 | 0.03707686 |

These are recorded observations, not estimates of gate activation. A token
whose embedding RMS is 0.0398 would have normalized RMS about 0.9969 under our
epsilon; normalization therefore largely removes that token's raw small
scale. Per-token values can differ. The approximately 7% panel-versus-global
input RMS difference is far smaller than the raw hidden/embedding ratio.

The paper deliberately places the previous hidden state on the value path
and the embedding on the gate path. It rejects an additive identity shortcut
because a pretrained model could ignore feedback. Appendix C explicitly
normalizes the embedding before fusion and the fused input before the model;
it does not specify our OLMo scale calibration or exact epsilon. Thus the lack
of an identity path is intentional, not an omitted paper component.
See [FBT §3.1 and Appendix C](https://arxiv.org/html/2608.08888).

The retained [author Nanochat code](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py)
also normalizes the gate input and fused product. Its two active fusion
matrices use uniform initialization with bounds `sqrt(3/D)`, as ours do;
neither implementation initializes this path as identity. Nanochat normalizes
ordinary token inputs too, whereas pretrained OLMo consumes its original
unscaled embeddings. Our fixed output calibration preserves that OLMo input
scale. Explicit FP32 reductions and epsilon are additional OLMo adaptations,
so this is not an exact numerical reproduction.

The author's tied Nanochat lookup additionally multiplies by 800 before
normalization to keep epsilon insignificant for its small tied-weight
initialization. Copying that multiplier into OLMo's ordinary input would
change the pretrained function. Nanochat also retains token-value embeddings
inside some attention layers; those provide a separate token route. Its
`x0` residual uses the current pass input, which is fused on feedback passes.
These architecture differences must not be conflated with an identity path
inside the paper's gate-product fusion.

Finally, F0 already contains the earlier
[128-update fusion-only warmup](../olmo-fusion-startup/warmup-results.md).
Initialization arguments do not establish the trained gate's saturation or
sensitivity. We have not measured those here. Matching RMS does not match
directions, token identity preservation, or alignment with the pretrained
backbone/readout. A stable but poor predictor can therefore remain despite
well-controlled scales. F8 settling faster with poor CE is consistent with
that concern, but does not establish its cause. Continue the declared run;
judge improvement in prediction separately from faster hidden-state settling.

Evidence: `.runtime/olmo-fbt-stability/native-f12-to128-01/stability-update-000000.json`;
author snapshot revision `7037c60924870aca6e30fac95212b0c7caee052d`, retained
`gpt.py` SHA256 `bee2292e7f56fe683dbb0e44413c44ef97b47779ee3dd8130bbae6b59673f1a2`.
