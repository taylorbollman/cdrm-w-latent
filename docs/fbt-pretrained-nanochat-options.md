# Public pretrained Nanochat and OpenELM-450M options

Checked 2026-09-21 after the user supplied public Nanochat checkpoint links.
This updates the availability conclusion in the
[research plan](fbt-rt-nextlat-research-plan-v2.md) and
[author-reproduction audit](fbt-nanochat-budget-audit.md).
Metadata, published code and object availability were inspected; full weights
were not downloaded or loaded, and no GPU work was run.

The user's final choice is OpenELM-1.1B as the primary model, with OpenELM-450M
an optional smaller fallback and Nanochat deprioritized. The
[subsequent review](fbt-architecture-and-auxiliary-loss-review.md) traces Nanochat
changes to upstream and retains feature matching as a late diagnostic and STP
as future work; neither is an initial experiment requirement.

**Correction and decision consequence**

There are usable public pretrained Nanochat models. The previous search only
established that the exact FBT author's newer, tied d20 checkpoint was not
publicly located. That does **not** imply that pursuing Nanochat requires
pretraining from scratch. Native-backbone adaptation is a real alternative.

| Candidate | Parameters | Native tying | Useful role |
| --- | ---: | --- | --- |
| `nanochat-students/base-d20` | 560,988,160 | No | Smaller public base; relatively simple full-MHA RT port |
| `karpathy/nanochat-d34` | 2,217,082,880 | No | Larger pretrained fallback; not a small first pilot |
| `apple/OpenELM-450M` | 457,179,136 in HF artifact | Yes | Optional smaller same-family fallback with native Q/K norm and checkpoint ages |
| `apple/OpenELM-1_1B` | 1,079,891,456 in HF artifact | Yes | Selected primary model for the approximately 1B experiment |

Counts exclude our new feedback and NextLat parameters. Native OpenELM
intermediate checkpoints retain 128 additional vocabulary rows compared with
the HF artifacts, so their exact counts differ slightly.

**The public base d20**

Pin `nanochat-students/base-d20@4c6e4e8d8468a6b66302a472cb950bb73a1e2a61`.
It has 20 layers, width 1280, ten query and ten KV heads, head 128, ReLU-squared
FFN width 5120, vocabulary 65536 and context 2048. The checkpoint is from base
pretraining at update 21400. Its detailed report and effective batch imply
11,219,763,200 processed token positions; the README's opening 2B claim conflicts
with those records. Do not use that opening claim as training exposure.
[Training report](https://huggingface.co/nanochat-students/base-d20/blob/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/README.md),
[metadata](https://huggingface.co/nanochat-students/base-d20/blob/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/meta_021400.json).

The repository contains `model_021400.pt`, `optim_021400.pt`, metadata and its
tokenizer. Unauthenticated model/tokenizer HEAD requests succeeded. The native
model is about 2.08GB; `pytorch_model.bin` is a duplicate object, not a distinct
training stage. Prefer the original metadata/tokenizer contract over assuming
that a community Transformers wrapper preserves every behavior. Optimizer
availability does not yet establish a complete exact distributed resume.
[Artifacts](https://huggingface.co/nanochat-students/base-d20/tree/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61).

Do not substitute `nanochat-students/nanochat-d20`: that is an SFT release.
`mid-d20` and `rl-d20` also represent later training stages. They may be useful
for different questions, but are not the clean base control requested here.
[SFT model](https://huggingface.co/nanochat-students/nanochat-d20),
[mid-training model](https://huggingface.co/nanochat-students/mid-d20),
[RL artifacts](https://huggingface.co/nanochat-students/rl-d20).

**The public d34**

Pin `karpathy/nanochat-d34@c48357d43863a3a6cdc5f5db5b4ec5964e4192d6`.
It has 34 layers, width 2176, seventeen query/KV heads, head 128, context 2048 and
vocabulary 65536. The reported pretraining exposure is88,683,315,200 positions.
Weights, metadata and tokenizer are public; no optimizer checkpoint is listed.
The card describes a base model even though its installation instructions use
a `chatsft_checkpoints` directory. Record its base-training metadata instead of
inferring a training stage from that directory name.
[Model card](https://huggingface.co/karpathy/nanochat-d34),
[metadata and files](https://huggingface.co/karpathy/nanochat-d34/tree/c48357d43863a3a6cdc5f5db5b4ec5964e4192d6).

The metadata lacks a training-code SHA. Historical Nanochat
`5c93a56be5ea24e87c5756126afe8f4b5fb1458b` supplies a candidate compatible model
implementation, not a proven exact training pin. Require state-dict and forward
validation before using it as an oracle.
[Historical source](https://github.com/karpathy/nanochat/blob/5c93a56be5ea24e87c5756126afe8f4b5fb1458b/nanochat/gpt.py).

**Why these weights are usable but not interchangeable with the FBT author's d20**

The public older models have separate embedding/readout matrices, parameter-free
RMSNorm, Q/K normalization after RoPE, full causal attention, ordinary residual
paths and logit softcap 15. Their parameter counts match two vocabulary matrices.
The author's newer d20 uses tied 32768-row weights plus token-value embeddings,
residual/input coefficients, smear/backout, additional Q/K scaling and window
patterns. Identical depth/width in d20 does not imply identical computation.
[Public d20 model code](https://huggingface.co/nanochat-students/base-d20/blob/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/modeling_nanogpt.py),
[author's FBT fork](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py).

If the Nanochat alternative is revisited, preserve the old native trunk and port
the FBT wrapper onto it. Do not load with
missing keys into the new backbone, replace its tokenizer, or force pretrained
untied matrices to share storage. The existing FBT gate contains a learned
D-by-D state projection; it can learn a mapping between hidden and input bases.
The paper motivates tying as a stability/representation aid, not a dimensional
requirement. Retaining untied weights is mathematically viable, but omits a
documented stabilization choice and may make adaptation harder. That last
possibility is a research uncertainty, not a demonstrated failure here.
[FBT stabilization discussion](https://arxiv.org/html/2608.08888v1#S3.SS3).

Older full MHA and uniform dimensions make a native RT adapter structurally
simpler than OpenELM's GQA/variable attention geometry. RoPE, Q/K norm order,
recurrent writes, backward, cached decoding and shared-pass gradient checks
still require implementation. Fewer engineering complications do not guarantee
faster measured training, particularly with the larger vocabulary.

A bounded search of other public d20 releases did not identify a better-verified
match for the author's tied backbone. Several have explicit untied configs;
others change MLP/rotary choices or lack clear provenance. This is not a claim
that no such checkpoint exists anywhere.

**OpenELM-450M remains an optional smaller fallback**

Pin `apple/OpenELM-450M@b53a9c5a731d154b71f8d311ef702f327e0cfa3a`.
It retains tied weights, learned per-head Q/K RMSNorm before RoPE, SwiGLU,
theta 10000 and context 2048. It has 20 layers, residual width 1536, head 64,
12–24 query heads and 3–6 KV heads with 4:1 GQA. Its parameter/state storage is
about 42.3% of the 1.1B HF model's; throughput must be measured separately.
[Pinned config](https://huggingface.co/apple/OpenELM-450M/blob/b53a9c5a731d154b71f8d311ef702f327e0cfa3a/config.json),
[model metadata](https://huggingface.co/api/models/apple/OpenELM-450M).

Apple provides individual50k through 350k checkpoints and full training states.
The 300k model and optimizer-inclusive artifacts returned HTTP200; neither was
downloaded. Nominal exposure is1.258T positions at 300k and 1.468T at 350k.
Its native 32128-row vocabulary requires the same import care as 1.1B.
[Checkpoint table](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/README-pretraining.md),
[training config](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/pretraining_configs/openelm_450M.yaml).

The smaller model does not remove the generalized RT backend work; it can make
same-family experiments cheaper if a fallback becomes useful. It is not a
prerequisite for the selected 1.1B experiment. A negative smaller-model result
may reflect a capability floor or scale dependence and does not reject the
1B hypothesis.

**Current model decision**

- Proceed with OpenELM-1.1B for the intended approximately 1B research comparison.
- Retain OpenELM-450M as an optional fallback; its tying/QK choices and shared
  adapter make it available without requiring an earlier smaller-model run.
- Deprioritize Nanochat, including scratch pretraining. Public base-d20 remains
  an audited alternative if model selection is revisited; preserve and label
  its native untied feedback rather than requiring a tying-conversion experiment.
  d34 is a larger alternative, not an economical first candidate.

Before a learning run, the chosen route needs a bounded native import, ordinary
logit/cache check, targeted gradient check for the actual modification, a small
capability baseline and measured memory/throughput. Fresh matched optimizer
warmup remains appropriate; source optimizer artifacts are not a requirement.
The early ordinary/ordinary+NextLat/RT/RT+NextLat comparison and independent FBT
switches remain unchanged. Public weights remove pretraining expense, not the
need to train and evaluate the new feedback/recurrence paths.
