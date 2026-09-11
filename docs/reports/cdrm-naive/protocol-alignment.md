# CDRM: current six-block protocol and paper alignment

The current primary experiment is **SEQ-6 versus CDRM-6 on official MAD selective
copying, vocabulary 16, T256, 96 copied tokens**, with zero-based early/late sites
**1/3**. This follows the user's six-block screening update and the selection frozen
at **2026-09-07 13:33:39 UTC** in
[six-comparison-selection.json](../../../.runtime/cdrm-naive/20260907T123830Z/six-comparison-selection.json).
Both SEQ/CDRM seed pairs completed common epoch 45 and passed the full
[artifact/pairing audit](e45-cohort-audit.json). The six unique final evaluations
cover all eight frozen endpoint/best-dev roles and passed the
[independent final audit](final-independent-audit.json). Results show mixed seed
outcomes and no consistent CDRM advantage; complete final values are in
[results.md](results.md).

The earlier 12-block D128 profile, sites 3/8, and original baseline-task corpus are
historical only. The old SEQ-12 run completed 50 epochs; no paired CDRM-12 run was
launched. D256 numerical records are also separate. Neither previous profile is
the current experiment or a substitute for six-block validation.

| Item | Paper/MAD reference and current behavior | Classification |
|---|---|---|
| Depth and topology | Paper compares one Transformer layer with one recurrent layer. Current models have six ordinary backbone blocks. CDRM previews through block 3, scans using post-block-1/post-block-3 states, bridges after block 3, then runs suffix blocks 4/5. | **Deliberately adapted**; CDRM adds parameters and computation. Local SEQ-6 is primary. |
| Per-block dimensions | D128, MLP512, 16 heads, full MHA/head dimension 8. | **Matched paper dimensions**, not paper depth or inherited numerical clearance. |
| Position/normalization | ALiBi maximum bias 8. Local pre-norm, learned normalization/QK-normalization, GELU, bias-free projections/norms and an untied head retain the saved project profile. | ALiBi **matched**; other details are **local choices/unavailable from D.2**. |
| Task implementation | Unmodified MAD generators at revision `0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`, with native targets/masks. | **Matched to pinned MAD**; exact authors' code revision is unavailable. |
| Selected difficulty | Copy count 96 is an official one-field change from baseline count 16; T256/vocabulary 16 remain fixed. | **Established MAD difficulty setting**, not a claim to match the paper's plotted setting. |
| Optimizer | AdamW LR 5e-4, betas (0.9, 0.98), epsilon 1e-8, weight decay 0. | **One matched paper candidate**, not the tuned sweep. D.2 repeats 5e-4; only three LR values are distinct. |
| Precision/clipping | FP32 parameters/computation; TF32/autocast/dropout off; no accumulation; clip norm 1. MAD defaults to BF16. | **Deliberate local profile**; clipping is unspecified in D.2. |
| Data and batch | 12,800 fixed train and 1,280 dev examples. Selected B128 gives 100 updates/epoch; fresh 1,280-example final was generated after role selection and evaluated. | Counts/B128 are **MAD defaults**, not verified authors' plotted-run settings. |
| Duration/schedule | All four SEQ/CDRM seed trajectories completed common epoch 45, chosen by the frozen runtime policy. Cosine horizon 200 epochs, minimum LR 1e-6, no warmup. | Horizon/minimum are **MAD defaults**; epoch stepping is an **adaptation**. Partial runs are not full tuned endpoints. |
| Metrics | Scored-token CE/accuracy and direct all-answer sequence exact match; native objective named separately when different. | Matches **metric types** in Figures 5/6; exact published harness details remain unavailable. |

Depth, dimensions, optimizer, ALiBi and figure statements follow
[paper §7.1, D.2 and E.1](https://arxiv.org/html/2604.21215#A4.SS2). No published values
are inferred from plots. Local settings resolve through
[base_d128_6.json](../../../configs/cdrm/base_d128_6.json) and the
[runner](../../../scripts/cdrm_common.py); batch/data/schedule defaults come from
[pinned MADConfig](../../../vendors/mad-lab/mad/configs.py), not the D.2 paragraph.
Its benchmark construction varies one YAML field at a time. The harder copy-count
and recall-vocabulary settings are established grid entries, not verified settings
underlying the published figures.

**Native supervision is preserved.** Selected copying uses source IDs 0–13, blank 14,
marker 15 at position 159, and 96 aligned targets at positions 160–255 whose inputs
are blanks. Train/dev/final use the same mask: **1,228,800 train targets**
and **122,880 targets each for dev and final**. No copied answer is fed back. Copy 96 also reduces
inserted blanks to 63 at fixed T, so this is not a pure memory-length intervention.

Deferred recall uses vocabulary 128 and configured T128: actual input length
**127**, keys 0–63, values 64–127. Native training has 127 dense next-token targets per
example (**1,625,600** total); separate retrieval labels total **303,639**. Dev has
**30,435** retrieval answers, 15–32/example (mean 23.777). Prior ground-truth values
remain visible to subsequent queries: this is teacher-forced recall, not free
generation. Dense CE includes unpredictable new values/key choices and differs
from answer-only CE. The unused optional copy V128/T256/K16 corpus has 16 answers,
source IDs 0–125, blank 126 and marker 127. See
[screening data/provenance](../../cdrm-mad-screening.md) and the
[pinned generator](../../../vendors/mad-lab/mad/data/instances.py).

The runner consumes labels at the same logit position: **no second LM shift**.
It excludes `-100`, rejects empty-answer examples and directly counts exact match
over each complete answer mask. Exact match is never estimated by exponentiating
empirical token accuracy. The unpadded forward does not exclude legitimate IDs 0/1
merely because model configuration calls them padding/EOS IDs.

**The scheduler implements the intended horizon explicitly.** MAD constructs
`CosineAnnealingLR(T_max=epochs, eta_min=min_lr)` but returns top-level `scheduler`;
Lightning documents `lr_scheduler`, with epoch as its default automatic interval.
Thus source inspection does not prove registration/stepping in an unspecified
historical MAD environment. The local loop explicitly steps after completed epochs
and saves/restores scheduler state, without Stage B warmup or a compressed pilot
horizon. Sources: [MAD wrapper](../../../vendors/mad-lab/mad/model/pl_model_wrapper.py),
[Lightning contract](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.core.LightningModule.html),
[training loop](../../../scripts/cdrm_train.py).

**Development screening selected the task, not the final test.** At epoch 25, SEQ-6
copying seeds 0/1 reached **48.289%/66.184%** dev token accuracy, both **0%** sequence
exact, versus the **12.196%** modal baseline. This establishes local learning with
remaining errors, not a CDRM advantage. Recall reached **4.915%/5.809%**, below its
**9.663%** modal baseline, and was deferred in favor of copying. Near-shortcut recall
does not rule out recurrence; its optimization/task regime remains unresolved.
The third corpus has no model screening run. All four screening-report hashes in
the selection record were independently verified for this audit.

**The completed architecture pairs are intermediate development evidence.** At epoch
25, CDRM seed 0 has CE 1.144615 / token accuracy 40.645%, versus SEQ 0.883124 /
48.289%. Seed 1 has CDRM 0.725343 / 63.584%, versus SEQ 0.746556 / 66.184%.
All four have zero direct exact matches. Each pair has 51 bitwise-equal initial
backbone tensors, two nonzero distinct adapters, matching native sources/data and
2,500 matching ordered update records. Endpoint/best-dev checkpoint identities
and counters are verified in the [seed-0](seed0-e25-audit.json) and
[seed-1](seed1-e25-audit.json) audits. These observations do not substitute for the
selected common epoch 45 or the independent final split.

**Continuation is allocated only by recorded cost.** The 50/45/25 policy was
recorded before either CDRM epoch-25 completion, with progress already visible.
After all four completed, epoch 50 required 5,561.22 seconds including 15% margin
and did not fit the 5,085.70-second remaining window. Epoch 45 required 4,462.77
seconds and was selected. The original 16:18:30 UTC cutoff and 20-minute finalization
reserve remain fixed. The serial SEQ0/SEQ1/CDRM0/CDRM1 queue uses exact epoch-25
checkpoints and the original 200-epoch schedule; all four continuations completed.
The full cohort audit verifies exact inherited histories and all 4,500 ordered
input/target/LR records, checkpoint payloads and initial backbone equality.
No outcome-dependent subset receives extra training.

Uniform answer-vocabulary chance is approximately 7.143% for selected copying and
1.5625% for harder recall; full-vocabulary chance is 6.25%/0.78125% respectively.
Copy's mode baseline ignores output order. Recall's mode counts values over distinct
earlier keys, ignores the current query, and uses native conditional-uniform query
sampling for expected accuracy. Both modal baselines score 0% dev exact match.

**Pairing and holdout boundaries are explicit.** New train/dev seeds 112345/123456
replace native MAD's shared advancing RNG with independent splits while retaining
every draw. All examples passed the oracle; no within/cross-split exact input
duplicates were found. Final seed 134567 was used only after endpoint/checkpoint selection was frozen. Model seeds **0 and 1** share
the fixed corpus and shuffle seed **45678**. CDRM starts from each corresponding
fresh SEQ backbone initialization, not a trained SEQ checkpoint; compatible completed
SEQ screens supply the baseline at common epochs. Adapters are nonzero with a
separate deterministic initialization. Gates remain epsilon 0.1, rho 1 and bridge
lambda 0.01, distinct from weight decay. Sites 1/3 are a topology choice, not tuned
against task outcomes.

The selected copying final split was generated by non-overwriting append **after**
all four epoch-45 completions and the immutable final role-selection plan. The
[append verification](final-data-append-verification.json) and
[final audit](final-independent-audit.json) verify unchanged pre-existing inputs,
all 104 frozen file identities, four appended final files and six actual completed
evaluations. The other screening settings were not promoted to final evaluation.
Endpoint roles use epoch 45; best-dev roles use epochs 45/44 for SEQ/CDRM seed 0
and 41/45 for seed 1, selected by minimum dev answer CE with earliest ties.
Coincident roles share an evaluation; no checkpoint is selected using final scores.
Final reports retain aggregate native/answer counts and metrics, not per-example
predictions. Two paired seeds and development-based allocation support preliminary
evidence, not a complete sweep, published-endpoint reproduction or broad
training-seed uncertainty estimate. Later result-driven changes require fresh
confirmatory data for new claims.
