# A5 follow-up: locate the length-generalization boundary

Prepared 2026-09-11 as a plan, then approved and completed. See the
[implementation and results](reports/rt-a5/length-followup/report.md).
The original approved copy is retained with the follow-up evidence. Numbers
below were read from the completed [10,000-update pilot](reports/rt-a5/pilot/report.json).

The next milestone should be evaluation only. Most requested measurements
already exist in the saved length-36 per-position results. Present those
clearly, verify selected explicit truncations on the trained checkpoints,
then review whether to extend training. Keep the two-layer D512/H8 models,
both RT blocks recurrent, full FP32, and compilation/CUDA graphs disabled.

Revised after the user's protocol clarification: **finish and report RT
first, then perform the limited ordinary-Transformer confirmation.** Model
runs must be sequential. Further training is a separate decision after this
evaluation milestone; do not turn a zero cumulative-accuracy measurement
into a training stopping rule.

## Reconcile Appendix F.5 with Table 5

The number of unique words describes the dataset; the update count describes
how often training batches are drawn from it. The same words can be reused
over many epochs. Thus a million-word corpus and 400,000 updates are
compatible, rather than competing descriptions of the training budget.

There is a small reporting distinction between F.5's prose and the released
implementation. The [generation instructions](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/README.md#state-tracking)
create one million length-12 words. The [data module](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5_data.py)
then uses `test_size=0.2`, leaving 800,000 training and 200,000 validation
words. The [GPT config](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/config/a5/gpt_a5.yaml)
explicitly comments that there are 800,000 training examples, specifies
400,000 training batches and an effective batch of 1,024. Its evaluation
budget is 100 batches, explaining approximately 100k evaluated words:
100 * 1,024 = 102,400.

| Quantity | Completed pilot | Reference recipe |
| --- | ---: | ---: |
| Unique length-12 corpus before split | 1,000,000 | 1,000,000 |
| Unique words assigned to training | 800,000 | 800,000 |
| Words per update | 1,024 | 1,024 |
| Updates | 10,000 | 400,000 |
| Word presentations, including repeats | 10,240,000 | 409,600,000 |
| Nominal passes over training words | 12.8 | 512 |

The reference's 512 nominal epochs in Table 5 agree with this calculation.
Its dataloader drops incomplete epoch batches, whereas our preserved order
carries the remainder forward; nominal passes measure total word exposures,
not identical epoch-boundary behavior. F.5 appears to summarize the generated
corpus without spelling out the split. This is a source-backed reconciliation,
not verification of every historical paper run.

**Recommendation:** keep the frozen corpus, split and existing checkpoints.
Do not add held-out words to training or generate a fresh million words per
update. Report unique training words and total presentations separately.
The 400k budget remains the reference; the current 10k pilot is deliberately
shorter and need not be extended merely to answer the length-sweep questions.

## Resolve the meaning of accuracy

The pinned [NextLat A5 evaluator](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5_data.py#L104-L165)
computes cumulative correctness with a product over positions. This is the
best-supported interpretation of the length curve in Figure 10. The figure's
generic axis label alone does not distinguish metrics, and its original
plot-building script/data have not been independently recovered.

For the correct/incorrect indicator c[b,t], keep three quantities separate:

- **Cumulative prefix exactness E(t)** = mean over words of the product of
  c[b,j] for j=1..t. Every state through t must be correct. At length t this
  is whole-word exact match, and it is the primary paper-style curve.
- **State accuracy A(t)** = mean over words of c[b,t]. Only the state at
  position t must be correct. At length t this is final-state accuracy.
- **Mean token accuracy M(t)** = sum of A(j), j=1..t, divided by t. This is
  the average over all positions that the previous concise summary emphasized.

Uniform state guessing is 1/60, about 1.67%, for A(t). E(t) has no flat 1/60
chance line. Successful early predictions keep M(t) elevated even when later
states are near chance. Do not expect M(13) or M(14) to drop to zero.

## What the saved results already show

These are percentages from the same 102,400 OOD development words at update
10,000. Each row uses their first t positions; no additional model inference
was performed to obtain this table.

| t | RT E(t) | RT A(t) | RT M(t) | SEQ E(t) | SEQ A(t) | SEQ M(t) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 99.1709 | 99.3545 | 99.8717 | 0 | 1.6943 | 39.7259 |
| 13 | 79.8730 | 80.1240 | 98.3526 | 0 | 1.6758 | 36.7990 |
| 14 | 21.5156 | 23.5010 | 93.0061 | 0 | 1.6279 | 34.2868 |
| 15 | 2.8105 | 6.4180 | 87.2335 | 0 | 1.6553 | 32.1113 |
| 16 | 0.2441 | 2.7568 | 81.9537 | 0 | 1.7773 | 30.2155 |
| 18 | 0 | 1.6143 | 73.0442 | 0 | 1.6494 | 27.0411 |
| 36 | 0 | 1.6963 | 37.3588 | 0 | 1.6660 | 14.3542 |

RT therefore has substantial, imperfect success at 13 and a sharp decline
at 14–16. It does not retain near-perfect performance at 13. Its 37.36%
mean token accuracy at 36 mostly reflects strong early prefixes, while
final-state accuracy is near chance and no complete 36-state sequence was
correct in this sample.

SEQ's decline occurs earlier: state accuracy at positions 5/6/7/8 is about
65.46%/10.56%/3.13%/1.94%. By positions 12–14 it is already near chance,
despite mean token accuracy remaining 39.73%/36.80%/34.29%. Its cumulative
exactness is already zero at position 10 in this sample.

The 99.1709% RT prefix-12 exactness above and the previously reported
99.1846% short-development exactness use different development examples.
Their small difference is not evidence that total input length changes the
predictions; that must be checked using identical words.

## Proposed execution order

1. **RT first: export its complete existing length curves.** Read the saved endpoint
   metrics and their counts, source/data identities and checkpoint hashes.
   Produce an RT table for every length 1–36, with separate E/A/M
   columns and denominators. Make E(t) the primary comparison. Report RT's
   last lengths above 95%, 50%, 10% and 1% exactness to avoid an arbitrary
   binary claim that it either succeeds or fails at 13. Use existing records
   at updates 5,000 and 10,000 to show whether this boundary has moved;
   these both evaluated the same full 102,400-row development sets. Preserve
   the fixed 10,000-update endpoint as the primary result.

2. **RT first: check literal shorter inputs once.** Load the trained RT D512
   endpoint without changing weights. For the same first 1,024 frozen OOD
   development words, compare T36 outputs with explicitly truncated inputs
   of lengths 12, 13, 14 and 16. Compare prefix predictions and E/A/M, using
   the established FP32 tolerance for logits rather than requiring bitwise
   equality across tensor shapes. This is five RT forward passes,
   with no backward pass or optimizer updates. Existing
   initialization checks already establish causality; this bounded check
   covers the trained models at the lengths now of interest. If a material
   discrepancy appears, localize it before treating long-input prefixes as
   standalone shorter evaluations. If the check passes, use the existing
   102,400-row curves; no full independent run at every length is necessary.

3. **Publish the RT result before starting new SEQ model evaluations.**
   Plot E(t), A(t), and M(t) in
   separately labeled panels, with the training boundary at 12. Include the
   full 1–36 range and a 10–18 zoom around RT's decline. Put the
   1/60 reference on A(t) only. Show count-derived Wilson intervals for
   E/A when helpful; they describe sampled words at one checkpoint, not
   variation across training seeds. Do not construct mean-token intervals
   by treating positions within a word as independent observations. Label
   observed zero counts as such, rather than population impossibility.

   Deliver the RT interpretation promptly: reliability at 13/14, accumulated
   misses versus loss of individual-state accuracy, and movement between 5k
   and 10k. Reporting RT does not depend on finishing a new paired report.

4. **SEQ second: bounded confirmation using the existing ALiBi model.**
   Export the saved full E/A/M curves and compare 5k/10k, with a zoom over
   positions 4–14. For the same 1,024-word subset, check actual inputs of
   lengths 12, 13 and 14 once, comparing the shared prefixes across those
   shapes. This is three SEQ forward passes after RT is complete. Historical
   full T36 metrics remain explicitly identified as historical measurements;
   no new T36 SEQ forward is required by default. Do not run a separate
   longer-length sweep solely to reconfirm cumulative zeros. Investigate a
   material prefix discrepancy if one appears. This confirms the current
   ALiBi architecture without changing or attributing effects to ALiBi.

5. **Close with the comparison and a training decision.** Add the limited
   SEQ results to the report after the RT report is available. Preserve the
   fixed endpoints, input IDs and independent unevaluated confirmation set.
   There is no automatic SEQ training extension or positional-encoding sweep
   in this milestone.

Both models are causal and use same-position supervision. In exact arithmetic,
their output at t depends only on positions 1..t, so one length-36 evaluation
already provides all shorter-length predictions. Uniformly sampled long-word
prefixes have the appropriate shorter-word task distribution, with the frozen
corpus's existing training-prefix exclusion. New independently generated
datasets at every length would add sampling variation without resolving the
main questions better.

## Early stopping applies to the cumulative length sweep

For fixed checkpoint weights and the same fixed evaluation words, E(t+1)
cannot exceed E(t). Once the **integer number of completely correct prefixes
is zero**, every longer prefix of those same predictions also has zero
exactness: extra tokens cannot undo an earlier error. Use counts, not a
percentage rounded to 0.00%. This permits stopping redundant longer-length
work for SEQ's cumulative-exactness curve. Existing curves can still be
exported in full at essentially no model-compute cost.

The rule has three limits:

- It is a statement about the sampled words and unchanged predictions, not
  a proof of zero population success. Independent datasets at other lengths
  do not inherit that observed zero. For separately truncated forwards, use
  the causal-prefix check before transferring the claim across shapes.
- Later isolated-state accuracy A(t) can recover even when E(t) remains zero;
  mean token accuracy M(t) is also a different metric. Retain their existing
  full curves rather than filling them with zeros or claiming they were
  newly evaluated.
- It is **not a training stopping rule**. Whole-word exactness is commonly
  zero early in training while token loss and state predictions improve.
  For future training, use an explicit update budget and review development
  learning curves; do not terminate SEQ merely because E(12) is currently zero.

The limited T12/T13/T14 SEQ checks above are retained because they answer the
user's direct confirmation request; they are not a new exhaustive sweep past
the first observed zero. RT retains its full saved 1–36 E/A/M curves, including
the requested length-36 token accuracy, regardless of where its E first vanishes.

## Small implementation scope

Add a standalone length-report utility and, if needed, a standalone trained
prefix-check utility. Reuse the existing metrics and model-loading functions;
do not alter the model or trainer. The historical source manifest includes
`rt_a5_eval.py`, `rt_a5_train.py`, and shared helpers. Editing those files
would invalidate strict checkpoint source matching, so preserve them and
record the new utility's source separately alongside the checked original
contract. Do not weaken source validation to make a checkpoint load.

Add only focused tests for any new aggregation, threshold or input-slicing
logic. Keep prior numerical validation closed; no BF16/FP64, gradients,
attention attribution, compiler or architecture ablation is needed here.

Use new output directories and online W&B under `taylorbollman`, project
`rt-a5-state-tracking`, linking back to the pilot. Retain tables, plots,
reports and checks in `gs://fast-chunks/cdrm-w-latent/rt-a5/<new-lineage>/`;
reference existing checkpoint/data hashes without duplicating all artifacts.
Run model forwards only inside the verified GPU container. This round's
GPU work should be measured in minutes including startup; the recorded full
102,400-row RT T36 evaluation itself took about 4.59 seconds. Reporting and
small harness changes will take longer than the forwards.

## Subsequent training decision, outside this evaluation milestone

Our SEQ checkpoint has not learned accurate late states even within the
training horizon. Its present curve cannot establish a clean learned-at-12,
failed-beyond-12 contrast. The reference uses **400,000 updates** (see
[NextLat Table 5](https://arxiv.org/pdf/2511.05963v4#page=29)); our pilot used
10,000. This is a large confound before attributing the difference to ALiBi.
The released GPT also differs in its norm and MLP recipe. Retain the current
architecture for the immediate round.

If we next study whether RT's horizon improves with training, a sensible
bounded follow-up is **RT first to 50,000 total updates**, with a checkpoint
and evaluation at 25,000 and the primary endpoint at 50,000. Reuse
initialization lineage, data order, optimizer state and FP32 execution.
The additional 40,000 RT updates would take about 37 minutes of training-loop
time, plus evaluation/checkpoint overhead, based on the completed pilot.

Extending SEQ to the same budget would then be a separate comparison choice,
run after RT, rather than an automatic requirement for investigating RT's
own learning curve. It would add about 11 minutes of training-loop time;
both extensions together would take roughly 50 minutes including similar
overhead. These are estimates, not new runs. If budgets differ, label them
explicitly and do not describe an RT-50k/SEQ-10k contrast as a matched-budget
architecture comparison. A matched 10k comparison remains available.

Assemble continuation histories without duplicate updates and preserve
explicit checkpoint steps. If a later SEQ extension learns length 12,
reassess the 13–36 drop; if it does not, reassess budget/optimization before
interpreting that drop as pure length extrapolation or starting
positional-encoding ablations. Zero whole-word accuracy alone never triggers
training termination.

Extending toward the 400k reference, adding paired seeds, and changing ALiBi
to RoPE remain later decisions. A single seed at 50k would still not be an
exact paper reproduction or a convergence claim.
