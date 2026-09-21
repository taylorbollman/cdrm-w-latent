# Author-maintained Nanochat FBT reproduction: evidence and budget audit

Checked 2026-09-21. Planning evidence only: no weights downloaded, model code
changed, or GPU jobs launched. The current decisions are in the
[research plan](fbt-rt-nextlat-research-plan-v2.md).

**Availability update:** the [public-checkpoint follow-up](fbt-pretrained-nanochat-options.md)
verified usable older Nanochat base-d20 and d34 weights, plus OpenELM-450M.
The unavailable artifact discussed below is the exact author's newer tied
checkpoint. Scratch pretraining is not required for Nanochat in general, and
the miniature proposal below is now a lower-priority option.
The [subsequent architecture review](fbt-architecture-and-auxiliary-loss-review.md)
traces the newer backbone additions to upstream Nanochat and clarifies the
863M-total versus 393M-transformer parameter distinction. The public561M d20
is not the model used for the reported FBT gains.

**Provenance and limits**

`xidulu` is Xi Wang, first author of FBT. This is an **author-maintained
Nanochat reproduction**, not an unaffiliated implementation. It remains distinct
from the original paper's Microsoft backbone and experiments. Pin:
`7037c60924870aca6e30fac95212b0c7caee052d`.
[Author identity and paper discussion](https://huggingface.co/papers/2608.08888),
[repository](https://github.com/xidulu/Full-bandwidth-transformer/tree/7037c60924870aca6e30fac95212b0c7caee052d).

The author explicitly treats optimal allocation of training compute as unresolved
in the paper discussion. Do not turn one successful recipe into a scaling law.

**Reported downstream result**

| Evaluation | Ordinary model, standard decoding | Cross-gate FBT, standard decoding | Cross-gate FBT, fused decoding |
| --- | ---: | ---: | ---: |
| GSM8K | 642/1,319 = 48.67% | 632/1,319 = 47.92% | 692/1,319 = 52.46% |
| MATH-500 | 173/500 = 34.6% | 171/500 = 34.2% | 198/500 = 39.6% |

These are **README-reported**, zero-shot, greedy results following pretraining
and one OpenMathInstruct-2 `train_5M` epoch. FBT SFT uses K3. The corresponding
newer SFT scripts/result paths named in the README were absent from the pinned
tree; older checked-in 5-shot artifacts have different lineages. These numbers
are not our reproduction or an isolated adaptation result. The benefit depends
on feedback decoding, and training compute is not matched.
[README](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/README.md).

**Actual pretraining recipe**

The baseline script requests **8 A 100s**, despite `4xh100` in its filename.
It runs 60k ordinary updates, saves every 10k, and uses 524,288 input positions
per update. The configured K2 flag does not activate feedback: its start fraction
is 1.0. The fork resumes the 40k checkpoint and optimization state, switches to
K2 immediately, and runs another 20k updates. It does not use the original
paper's pass-frequency mixture. K2 means two total stack evaluations.
[Baseline script](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/train_d20_standard_60k_4xh100.slurm),
[fork script](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/train_d20_lf_k2_modes_from40k_a100.slurm).

| Phase | Updates | Input positions | Stack-pass position equivalents |
| --- | ---: | ---: | ---: |
| Shared ordinary trunk | 40,000 | 20.97152B | 20.97152B |
| Ordinary continuation | 20,000 | 10.48576B | 10.48576B |
| FBT continuation | 20,000 | 10.48576B | 20.97152B |
| Complete ordinary lineage | 60,000 | 31.45728B | 31.45728B |
| Complete FBT lineage | 60,000 | 31.45728B | 41.94304B |

These are processed positions, not unique tokens or exact FLOPs. They exclude
evaluation and SFT. The resumed logger's 62.91456B pass-equivalent summary applies
K2 retrospectively to all 60k steps and is not the executed lineage.

**What the learning curve adds**

The following values come from explicit L2 validation in the gate-product log,
and validation at the same global update in the ordinary log. Lower bits/byte
(BPB) is better. Generic `Validation bpb` in the FBT log denotes L1, not L2.

| Additional input positions after fork | FBT second-pass BPB | Ordinary BPB |
| ---: | ---: | ---: |
| 0 | 3.138689 | 0.747880 |
| 52.4M | 0.798891 | 0.747347 |
| 104.9M | 0.760616 | 0.747217 |
| 262.1M | 0.749723 | 0.746279 |
| 524.3M | 0.745482 | 0.744604 |
| 1.049B | 0.741251 | 0.741933 |
| 2.621B | 0.731156 | 0.732382 |
| 10.486B | 0.689524 | 0.694023 |

This is recovery evidence from one recipe, not a statistically established
crossover or a guarantee about RT/OpenELM. A new-path improvement test stopped
at 10M would miss almost the entire recovery here.
[FBT log](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/d20-lf-k2-modes-248693_0.log),
[ordinary log](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/runs/d20-standard-60k-238647.log).

**Observed training-time scale**

| Work on 8 A 100-SXM4-80GB GPUs | Timed training hours | Approximate aggregate GPU-hours |
| --- | ---: | ---: |
| Ordinary trunk through 40k | 11.24 | 90 |
| Ordinary baseline through 60k | 16.87 | 135 |
| Additional K2 branch, 40k to 60k | 11.01 | 88 |
| Both endpoints, sharing the trunk | 27.88 | 223 |

Derived from the two logs above. These timers omit evaluation/checkpoint overhead
and most startup work; they are not full job elapsed times. Baseline timing
excludes its first 11 steps. The resumed K2 step sum includes its compilation
step, and its footer already includes inherited pretraining time. SFT is excluded.
Do not translate these directly into H100 hours or RT throughput: those execution
paths and hardware utilization differ. Profile each intended mode locally first.

**Backbone and tying details relevant to a miniature**

The d20 model uses D1280, 20 blocks, ten query/KV heads, head dimension128,
T2048, tied 32,768-row readout, per-head Q/K normalization, ReLU-squared FFNs,
and `SSSL` windows. Native token-value tables, input/residual mixing, smear and
backout make it materially different from OpenELM. Its optimizer combines Muon
and AdamW.

Tying is already implemented: one FP32 parameter, one optimizer group, alias
restoration after materialization/load, and reference-specific input scaling.
Do not substitute upstream Nanochat defaults, which differ.

| Source-derived shape | Without registered feedback | With registered feedback | Value tables alone |
| --- | ---: | ---: | ---: |
| d8 / D512 | 109,052,138 | 110,362,858 | 67,108,864 |
| d12 / D768 | 261,095,906 | 264,045,026 | 150,994,944 |
| d20 / D1280 | 854,590,706 | 862,782,706 | 419,430,400 |

Counts for MHA/head 128 and retained native features, calculated from dimensions;
d20 also agrees with its log. Smaller shapes have not been instantiated/profiled.
Registered feedback includes inactive fusion alternatives.
[Model and optimizer source](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py).

**Availability and practical recommendation**

A bounded check found no downloadable reproduction checkpoint: the repository's
[releases](https://github.com/xidulu/Full-bandwidth-transformer/releases) and
[author's public HF model listing](https://huggingface.co/xidulu/models) were
empty, and scripts reference local checkpoint paths. This is an availability
finding, not proof that no compatible weights exist anywhere. No author was
contacted. Obtaining a suitable 40k checkpoint would change the cost calculation.

Do not make full d20 pretraining a prerequisite for our checkpoint-adaptation
research. A d8/D512 experiment is a reasonable optional learning sandbox:

1. Preserve the tied author-fork model/optimizer features in both arms; use only
   cross-gate FBT. An initial T512 run is an explicit short-context variant.
2. Pretrain one ordinary trunk, provisionally 0.5B input tokens with an optional
   extension to 1B if the baseline is still acquiring basic capability. Save
   intermediate snapshots. This is a proposed budget, not a source result or
   guarantee of useful math performance.
3. Fork ordinary versus FBT-only for 100M, then 250M tokens/arm; measure validation
   loss and exact sequential generation. Extend credible recovery curves toward
   1B/arm only after reviewing measured cost. A short negative result is weak
   evidence about ultimate benefit.
4. If this establishes a useful feedback comparison, add RT and NextLat through
   native block adapters, using matched original snapshots. Do not build a
   second independent optimized trainer/backend.
5. Test two starting ages only if checkpoint maturity itself is the question:
   earlier/later snapshots, each with its own ordinary and FBT continuation.
   A single-age mini-test measures recovery and cost; answering the starting-age
   question requires that second matched age and still does not determine the
   optimal OpenELM checkpoint by proportional scaling.

Native token-value tables already provide embedding bypasses. Selecting bottom
layer0 in an even-depth model avoids a value table in that selected layer, but
does not remove bypasses above it. Results would concern RT added to this
Nanochat backbone, not our earlier plain synthetic RT.

**What can transfer to the OpenELM plan**

OpenELM's configured 4,194,304 input positions/update imply about 209.7B positions
at 50k,419.4B at 100k and 1.258T at 300k. These are nominal positions, not verified
nonpadding exposure. Even its earliest published 50k checkpoint is about ten
times the Nanochat fork's20.97B exposure. Checkpoint fractions such as 40/60 and
300/350 cannot establish comparable maturity or required adaptation duration.
[OpenELM native config](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/pretraining_configs/openelm_1_1B.yaml),
[checkpoint table](https://github.com/apple/corenet/blob/f9f83e616a34d02c422733a06a3fe5bde63ae575/projects/openelm/README-pretraining.md).

Keep 300k for the mature-model question. Use50k as an explicit age sensitivity
with its own control if adaptation age becomes the suspected obstacle. Compare
improvement over each checkpoint's matched continuation, not raw cross-age scores.

Replace the former10M efficacy proposal with a budget ladder: small smoke check,
100M/250M recovery screens, conditional extension toward1B (and possibly2B).
Token budgets must be accompanied by optimizer-update counts. At 524,288 tokens
per update,10M permits only about 19 updates. A practical32k–64k-token effective
batch gives roughly 1,500–3,000 updates per 100M, but is a deliberate optimization
change from the reproduction. Choose the actual batch after profiling; report
tokens, updates, GPU-hours and task-supervised tokens independently.
