# Larger-batch mixed baseline and embedding-propagation comparison

**Current execution (2026-09-18):** the batch-2,560 baseline completed and
retained [15,000 total updates](rt-nextlat-mixed-b2560-15000-run.md): Fuzzy
answer accuracy 99.9112%, A5 L36 whole-word accuracy 92.8535%. This supersedes
the earlier 5k cap and proposed 10k budget below. The first fresh embedding
variant has started under the following authorization, keeping batch 2,560
per task and all other learning settings unchanged.

**Subsequent authorization:** after the baseline completes and is summarized,
run all three embedding variants fresh for 15k each, at the same batch and
unchanged task difficulty. See the
[embedding comparison protocol](rt-nextlat-mixed-embedding-comparison.md).
This user instruction supersedes the proposed requirement below to calibrate
a harder Fuzzy condition before comparing variants. Retain the ceiling caveat
and compare learning speed as well as endpoint performance.

**Earlier authorization (2026-09-17):** the user selected **2,560 examples per
task and 2,500 optimizer updates**, then stop for review. That fresh baseline
has been launched; see [the active run handoff](rt-nextlat-mixed-b2560-2500-run.md).
This supersedes the proposed batch and 10k budget below. The rest of this
document is the historical planning record; harder tasks and embedding
variants remain future work, with no additional experiments queued.

Proposed 2026-09-17. This is a plan, not a newly launched experiment. The
user specified an initial **10,000-update larger-batch baseline**, followed
by choosing comparison budgets from its learning curves. Further baseline
training is a later decision. Once the direction is agreed, save and fully
evaluate the current mixed run's stopping checkpoint and switch; completing
its original 20k target is not a prerequisite for the new baseline.

Execution update: the original mixed run was subsequently saved and stopped
at **19,810 updates** for the user's brief throughput recheck. Full endpoint
A5 L36 whole-word accuracy was 57.6582%, and Fuzzy answer accuracy 98.8509%.
The larger-batch learning baseline has not started. The 20k exposure reference
below is the original planned round-number budget; the actual stopped run
consumed 2,535,680 examples per task, about 0.95% fewer than that reference.

A subsequent user-requested throughput trial also completed at **2,048 per
task**: 454,152 tokens/second, 1.8579 seconds/update, 46.58 GiB allocated /
64.94 GiB reserved. This is 1.676x the immediately preceding 1,024-per-task
trial's throughput (271,043 tokens/second). See
[the 2,048 comparison](reports/rt-nextlat-fuzzy-a5/d128-token-throughput-2048-20260917/report.md).
The original 1,024-per-task proposal below remains a proposal; no larger-batch
learning run has been launched. If 2,048 is selected, its 10k budget would
present 20.48 million examples per task and take about 5.16 training hours.

The next requested probe at **3,072 per task** also completed without OOM:
593,574 tokens/second, 2.1323 seconds/update, 69.77 GiB allocated / 71.98 GiB
reserved. This is 30.7% more throughput than 2,048/task and the fastest tested
current-model configuration. See
[the 3,072 comparison](reports/rt-nextlat-fuzzy-a5/d128-token-throughput-3072-20260917/report.md).
Its 10k budget would present 30.72 million examples per task and take about
5.92 training hours. The 2,048 batch retains more memory headroom for variants.
These are throughput-only results; mixed-task learning at these batches is
still untested, and the learning baseline has not been launched.

This supersedes the tentative shorter learning budgets in
[the earlier larger-batch plan](rt-nextlat-larger-batch-directional-plan.md).
The completed throughput measurements in that document remain valid.

## Capacity is already measured with both tasks

The existing H100 80GB profile used the actual D128 mixed model, real A5
length-12 data and real Fuzzy length-400 data, NextLat, full forward/backward,
gradient clipping and Adam. Each physical task batch equaled its logical
batch. It used ten discarded warmup and thirty timed updates per candidate.

| Examples per task per update | Total examples | Mean seconds/update | Peak allocated GiB | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 1.4153 | 3.11 | 4.39 |
| 512 | 1,024 | 1.5262 | 11.80 | 16.68 |
| 1,024 | 2,048 | 1.6311 | 23.39 | 33.00 |

Thus **1,024 A5 plus 1,024 Fuzzy examples per update fits comfortably**.
This is stronger evidence than the old A5-only batch-1,024 runs. Measured
example throughput increased about 6.94-fold relative to 128 per task;
optimizer updates themselves became about 15% slower. This is not a
measurement of convergence speed. No capacity search beyond 1,024 is needed
before this baseline; retain headroom for variants and evaluation.

[Local profile](reports/rt-nextlat-fuzzy-a5/d128-batch-throughput/report.md)
and [W&B measurements](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/bkg4gs26).
The profile configuration hash is
`e77eb47d03d27d6cb7b0a3b9654f708332c30e0ecb48eb2251b38cf55d64d8ea`,
matching `configs/rt_nextlat_tasks/fuzzy_d128.json`.

## Proposed baseline

Start from the same original random initialization and deterministic task
streams as the fresh mixed run, with fresh optimizer state. Do not start
this architectural comparison from either trained mixed or A5-only weights.
Increasing batch size is a new experiment contract, not a strict resume.

Keep two tiled RT blocks, D128/H16/FFN512, Mitchell initialization, ALiBi,
window two in block 0 and full RT in block 1, no embedding bypass, and
NextLat weight one without autonomous latent rollout. The baseline has
479,616 trainable parameters. Preserve FP32 eager execution, LR 1e-4,
AdamW settings, clipping, and all objective and data semantics. Do not
automatically scale learning rate with batch size.

Use `batch_per_task=1024` and `microbatch=1024`. Each update processes
`[1024,12]` A5 and `[1024,400]` Fuzzy batches separately, accumulates
`0.5*(CE_A5+NL_A5) + 0.5*(CE_Fuzzy+NL_Fuzzy)`, then clips and updates Adam
once. Counts and task weights stay equal. There is no padding A5 to 400,
no cross-task sequence concatenation and no state carried between examples.
The current separate symbol ranges and task-local output distributions stay
unchanged. Reuse native dense Fuzzy training labels and masked answer evaluation.

Preserve the finite 800,000-word A5 training corpus and 12,800-example Fuzzy
training corpus. Larger batches regroup the same ordered streams. These
counts measure example presentations, including repeated corpus visits.
Any cross-batch order audit must compare concatenated row-ID prefixes, not
the old per-update chain hashes whose batch boundaries differ.

## Budget and monitoring

Train the baseline to **10,000 optimizer updates**, then review. Estimated
training-loop time is about **4.5 hours**, plus evaluations and retention.
This estimate uses the short measured profile and is not a convergence claim.

| Larger-batch update | Presentations per task | Meaning relative to batch 128/task | Estimated training time |
| ---: | ---: | --- | ---: |
| 1,250 | 1,280,000 | Same exposure as old 10k | 34 minutes |
| 1,875 | 1,920,000 | Same exposure as old 15k | 51 minutes |
| 2,500 | 2,560,000 | Same exposure as old 20k | 68 minutes |
| 5,000 | 5,120,000 | Twice old 20k exposure | 2.3 hours |
| 10,000 | 10,240,000 | Four times old 20k exposure | 4.5 hours |

At 10k, the new run has eight times the presentations of the old 10k run,
but only half the optimizer updates of the old 20k run. Plot updates,
presentations per task and wall-clock time separately. The architecture
comparisons will share the new batch regime, avoiding this confound within
the new experiment series.

Save checkpoints and joint evaluations at 0, 500, 1,000, 1,250, 1,875,
2,500, 5,000, 7,500 and 10,000. Use inexpensive fixed-subset A5 monitoring
every 250 updates and Fuzzy monitoring on the existing 1,280 examples.
Full 102,400-word A5 checks at the principal 1,250/1,875/2,500/5,000/10,000
milestones confirm onset and endpoint behavior. Use representative startup
evaluation/checkpoint checks and existing finite-state checks; do not reopen
precision, compiler or CUDA-graph qualification.

If A5 is still absent at the exposure-matched 2,500 checkpoint, continue
the agreed baseline budget. Eight times the batch does not imply onset in
one-eighth the updates. If the complete 10k baseline has not established
useful joint behavior, first inspect optimization and, if useful, run the
cheap batch-1,024 A5-only control before attributing failure to mixing.
Change learning rate or extend beyond 10k only as an explicitly recorded
follow-up decision. Larger-batch scaling varies substantially by workload
and optimization settings ([Shallue et al.](https://arxiv.org/abs/1811.03600)).

## Is the current benchmark sufficiently demanding?

It is suitable for calibrating larger-batch joint learning, A5 generalization,
interference and learning speed. As the user emphasized after the initial
discussion, Fuzzy performance is too close to ceiling for a useful primary
comparison of retrieval improvements. Establish retrieval headroom before
spending the main comparison budget on the embedding-propagation variants.

The inspected current checkpoint at 19,000 updates had:

| Metric | Accuracy | Scope |
| --- | ---: | --- |
| A5 L36 whole-word | 48.9014% | Fixed 4,096-word development subset |
| A5 L36 mean token | 88.5763% | Same subset |
| Fuzzy answer tokens | 99.0601% | Full 1,280 development examples |
| Fuzzy first-value tokens | 99.9771% | 17,464 / 17,468 first tokens |
| Fuzzy all-answers sequence exact | 78.2031% | Same full examples |
| Fuzzy terminal probe | 97.8329% | Same full examples |
| Fuzzy distance-bin 257–512 | 96.7432% | Actual distances present within T400 |

These are interim same-checkpoint observations, not the final 20k result.
At 15k, full A5 L36 exactness was only 19/102,400 (0.01855%). The observed
emergence makes very early zero exactness an inadequate failure criterion.

For the larger-batch calibration baseline, retain A5 training length 12/evaluation length 36
and Fuzzy length 400, native vocabulary 16, multi-query, no noise, and key/value
motifs up to three tokens. The native held-out keys have length three, while
training keys have lengths one through three. This is not the hardest native
MAD configuration. First-value retrieval is already almost perfect; lower
whole-sequence accuracy alone does not solve the retrieval-comparison problem,
because remaining multi-token answer errors can reflect continuation mistakes.

Assess each checkpoint jointly: A5 L36 exactness and prefix curve; Fuzzy
answer, whole-sequence, first-value, terminal and distance metrics. Do not
average the task accuracies or take each task's best checkpoint separately.
The initial question is whether embedding access improves or accelerates
retrieval while retaining state tracking. Avoid turning tiny near-ceiling
Fuzzy token differences into an architectural claim.

Before launching the variants, add evaluation-only development
stress tests at Fuzzy lengths 512 and 1,024 using baseline checkpoints and
appropriate evaluation microbatches. Label this length generalization, not
training at those lengths. Longer contexts need not be strictly harder:
more repeated keys can help in the small-vocabulary native task. Longer A5
development words are optional if L36 becomes saturated.

If first-value and terminal retrieval still saturate, calibrate a separate
harder training condition: native Fuzzy vocabulary 32 at fixed length 400,
keeping A5 unchanged. Greater symbol/key diversity is a more direct memory
challenge than relying solely on additional context length. Try vocabulary
64 only if 32 also quickly saturates. Use one bounded baseline calibration
at a time; select a condition that learns substantially but retains visible
retrieval errors at the comparison budget, not one stuck near chance.

This requires a new dataset and shared vocabulary/output interface for all
arms. A V16-trained checkpoint cannot directly evaluate new V32 symbols with
untrained embedding/output rows and call that a clean difficulty test. Start
the harder matched baseline and variants with the same expanded interface
and paired initialization. Preserve the V16 baseline as the batch calibration
result. Once a challenge is selected, freeze its generator, splits and
training budget before the mechanism comparison.
Record scored-query counts and terminal first-value accuracy: increasing
key diversity can reduce accidental repeated keys, changing the number of
scored answers per sequence. Use identical held-out examples across all
arms within the chosen condition rather than treating different native
conditions as having interchangeable denominators.

If the longer-context evaluation alone gives useful headroom, it can instead
serve as a shared length-generalization comparison for models trained on the
unchanged T400/V16 mixed task. State this scope explicitly. Increasing native
vocabulary is the preferred next training change if there is no such headroom.

This benchmark mixes separate task examples and separate symbol ranges in
one model. It tests coexistence in shared parameters; it does not require
retrieval and A5 state updates within the same sequence. Confirmation stays
reserved; the current and new screening results use development data.

## Subsequent embedding-propagation screen

Use the baseline curve and retrieval-difficulty calibration to choose common
fixed comparison milestones after its 10k review. Prefer a shorter budget when the baseline has already shown
stable joint learning; retain checkpoints for extension. If substantial
movement continues through 10k, consider extending the baseline before
settling the shorter comparison budget. Do not stop variants solely because
their L36 exactness is zero before the baseline's typical onset window.

Compare one mechanism at a time, applied only in upper block index 1:

1. No embedding propagation: the new larger-batch baseline.
2. Input addition: learned projection of the raw token embedding added to
   the upper block input, initially fixed lambda 0.01 with no ramp.
3. Permanent-value addition: learned embedding projection added only to
   stored upper-layer values, initially fixed lambda 0.01; keys and temporary
   self values keep the original semantics.
4. One existing head reassigned to permanent embedding values, using its
   existing value-projection rows, contextual keys and ordinary query route.
   Other heads and temporary self KV stay unchanged; no added scalar gate.

Use the existing injection normalization conventions, paired common initial
weights and matching data order/batch/optimizer settings. Input/value
projections add 16,384 parameters each; the head reassignment adds none.
Record this small parameter difference rather than changing model width.
The new model's single reassigned head is 1/16 of its value width (8 dimensions),
whereas the old D512/H8 head experiment reassigned 1/8. Document that adaptation.

Reuse existing causal/gradient checks where applicable and add only bounded
checks needed for adapting these paths to the mixed model. No broad numerical
analysis or new precision mode is part of this experiment. Confirm promising
differences or close rankings with another paired seed before stronger claims.

All graphable runs use `taylorbollman/rt-nextlat-fuzzy-a5`; save checkpoints,
resolved configurations and reports in the persistent project tree and verify
retention under `gs://fast-chunks`. Keep the existing current-run and reporting
supervisors unchanged until the switch is requested or their endpoint is reached.
