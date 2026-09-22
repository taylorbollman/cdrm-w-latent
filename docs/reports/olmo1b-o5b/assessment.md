# O5b assessment: feedback adapts to code but loses language retention

Reviewed 2026-09-22. The two matched arms completed **2,634 updates and
20,855,799 valid input tokens each**, with no restart or health-gate stop.
All full optimizer checkpoints and final comparison evidence have verified GCS
receipts. The subsequent fixed-weight diagnostic also completed and preserved
all recorded model/buffer hashes. No training or evaluation job remains queued.

See [paired results](results.md), [learning curves](learning-curves.pdf),
[post-hoc diagnostic](diagnostic/results.md), [optimization audit](optimization-summary.json),
and [protocol](protocol.md). RT and NextLat were off throughout.

## Main result

The original ordinary model improves with code continuation. Introducing FBT
recovers much of that code performance, but its feedback pass still costs code
quality and has a large general-language retention deficit.

| Final inference path, 512 windows | Code NLL | Retention NLL |
| --- | ---: | ---: |
| Original checkpoint | 1.787902 | 3.040993 |
| Matched ordinary control | 1.699385 | 3.177141 |
| FBT-trained model, ordinary first pass | 1.698216 | 3.183361 |
| FBT-trained model, feedback pass | 1.737004 | 4.969719 |

Relative to the matched control, feedback costs **+0.037619 code NLL**
(95% paired document interval +0.033360 to +0.041884) and **+1.792577 retention
NLL** (+1.631192 to +1.932436). These intervals quantify evaluation-document
sampling, not training-seed variability. The first-pass differences are small:
−0.001168 code and +0.006220 retention. The code difference alone is insufficient
to claim a useful architectural advantage from feedback training.

The learning curves show real recovery. On the separate 128-window subset,
feedback code NLL falls from 1.788459 at ramp end to 1.634389 at the endpoint;
retention falls from 7.200396 to 5.014280. The untrained full-strength feedback
preflight was much worse still. Recovery is slowing but ongoing; this does not
establish its asymptote or prove that longer training cannot help. Compare curve
values only against the matching 128-window control, not the table above.

## What the fixed-weight diagnostic adds

The [diagnostic protocol](diagnostic-protocol.md) was specified after observing
preliminary learning curves, before evaluating the finished checkpoint. It is
post-hoc development exploration, not another training run or held-out selection.

On the same 128-window code/retention selection:

| Inference feedback strength beta | Code NLL | Retention NLL |
| --- | ---: | ---: |
| 0 | 1.599637 | 3.249787 |
| 0.25 | 1.609754 | 3.284040 |
| 0.5 | 1.622601 | 3.367670 |
| 0.75 | 1.620166 | 3.608247 |
| 1 | 1.634389 | 5.014280 |

Partial feedback substantially reduces the retention penalty. **Every positive
beta tested remains worse than beta zero on both measures.** Beta zero is the
ordinary path of these same FBT-trained weights, not the separately trained
control. This supports a problem associated with the new feedback input; it
neither proves its cause nor establishes a beneficial intermediate setting.

On the separate 32-window, maximum-64-token prefixes, additional passes do not
rescue performance. At beta1, code NLL is 2.200133 / 2.200488 / 2.201095 /
2.200238 for K2 / K3 / K4 / exact online. Retention is 5.083958 / 5.013408 /
5.033363 / 5.032388. Exact online is only 0.051569 nats better than K2 on
retention, while remaining about 0.800 nats worse than ordinary online
(4.232553). At beta0.5, K2-to-online differences are smaller still: +0.000303
code and +0.001842 retention. Thus finite-pass approximation alone is not a
convincing explanation for the observed deficit on these short prefixes.
Long-context online behavior and free-running generation remain untested.

## Mechanistic interpretation and next experiment

The selected author's gate-product fusion **replaces** the ordinary embedding
at beta1. Its operation is:

    f(h,e) = s RMSNorm((W_U h) * sigmoid(W_G RMSNorm(e)))
    input(beta) = (1-beta)e + beta f(h,e)

The previous top-layer state supplies the value; the current embedding enters
through the gate. This matches the pinned fusion source. It is not an accidental
omission of an embedding residual. The interpolation and calibration to native
OLMo embedding RMS are explicit port choices.

A newly learned input representation trained only on Python may adapt unevenly
across domains even while the ordinary path preserves native capabilities.
That hypothesis fits these observations, but optimization, initialization,
native OLMo geometry, training budget, and omitted prefix/noise choices remain
possible contributors. The diagnostic does not separate them causally.

I recommend a **small fusion-only domain-adaptation experiment** before a broad
FBT/RT/NextLat sweep or a blind long extension. Branch twice from the retained
FBT endpoint, freeze all native parameters including tied embeddings, and train
only the two fusion matrices. Compare fresh code-only data with a prespecified
code/general-text training mixture, matched for total input-token exposure,
optimizer setup, beta1 and K2. General-text training must be disjoint from all
retention development/test documents. A roughly 5M-token-per-arm diagnostic is
a candidate budget, to freeze after selecting data and profiling this mode.

If mixed-domain fusion training improves retention substantially without moving
the backbone or greatly harming code, that would support insufficient transfer
of the new input path as an important cause. Compare actual code/general exposure
separately because equal total tokens do not imply equal code tokens. No recovery
would weaken this particular explanation at that budget, not conclusively rule
it out. Prefix mixing and state jitter should remain separate, controlled follow-ups.
This proposal is **not launched**. FBT-only results do not veto a later interaction
hypothesis, but resolving this large confound should make such tests easier to
interpret.

## Numerical health, recovery and limits

All recorded objectives/gradient norms are finite; model and AdamW state passed
finite checks at checkpoint boundaries. **All 2,634 updates in each arm were
clipped at norm1.** Preclip norm medians/maxima are 4.871/10.402 for ordinary and
5.043/21.144 for FBT. Both maxima occur on update1491. Median step times are
0.744/0.751 seconds, excluding evaluations/checkpoint work; peaks are
49.305/49.602GiB. The final100 FBT updates have a median fusion squared-gradient
norm share of about0.256%; this is not a measure of parameter-update size or proof
of insufficient learning rate. The frozen catastrophic gate required deterioration
in both domains, so completion does not imply satisfactory retention.

The first100 beta-zero updates match data/counts/LRs and start with identical
losses, but they are **not bitwise-identical training trajectories**. Mean/max
absolute aggregate-CE differences are0.000559/0.004478; update100 code/retention
NLL differences are +0.000254/−0.000095. Some gradient-norm differences are larger.
CUDA backward nondeterminism and subsequent BF16 drift are plausible but unverified
explanations. These records neither identify a new precision bug nor explain away
the much larger feedback retention deficit.

The actual endpoint was loaded safely for the diagnostic after checking its SHA.
Its runtime metadata contains PyTorch's `TorchVersion` subclass, requiring a
scoped allowance with `weights_only=True`. The new diagnostic loader handles it;
a future optimizer resume must use the same narrow context around the frozen
loader. The legacy queue's bare resume invocation does not install it. No frozen
training source or checkpoint bytes were altered; see [usage](../../olmo1b-o5b-usage.md).
Do not conflate the separate tiny/previous-platform exact-resume tests with an
actual O5b full-optimizer replay, which was not performed.

Other limits: one seed; development subsets; no programming execution score;
unknown original pretraining overlap; independent document windows; no packed
attention, no multi-GPU execution, and no reserved-test access. Short retention
prefixes cover only six original documents, versus59 in the full512-window
selection. Native OLMo normalization/embedding geometry differ from Nanochat;
prefix mixing and hidden jitter were intentionally omitted. This is a bounded
adaptation result, not a general verdict on FBT.
