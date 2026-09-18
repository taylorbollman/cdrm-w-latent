# Larger-batch directional A5/Fuzzy pilot

Updated 2026-09-17: the bounded throughput check completed after the mixed
pilot, including W&B reporting and verified GCS retention. The measured
128/256/512/1024-per-task mean update times were 1.4153/1.4549/1.5262/1.6311
seconds, with peak allocated memory 3.11/6.00/11.80/23.39 GiB. At 1024/task,
example throughput was approximately 6.94 times the 128/task result. These
short execution measurements do not establish convergence speed.
See [throughput report](reports/rt-nextlat-fuzzy-a5/d128-batch-throughput/report.md).
The larger-batch A5 and mixed learning pilots below remain proposals and
are not queued. The batch-128-per-task mixed run completed its authorized
10,000 updates, reporting and retention before profiling. The execution plan
below is retained as the prospective record.

The aim of a future pilot is to find out more quickly whether this model can
learn both tasks together, accepting weaker conclusions about comparative
learning efficiency. A larger batch does not make the experiment invalid;
it changes the optimization regime and the comparisons it can support.

## Current evidence and scope

The model remains two tiled RT layers, width128, 16 heads, FFN512, first
layer window2 and second full, ALiBi, Mitchell, NextLat weight1, FP32 eager.
A5 examples have length12; Fuzzy examples have native length400. Measure
A5 length36 by direct backbone evaluation, without latent rollout.

Current logical and physical batch sizes are both128 **per task**. Mixed
updates therefore contain256 examples. Raising only `--microbatch` cannot
improve this configuration: no task batch contains more than128 examples.
Increasing the logical batch while keeping microbatch128 accumulates more
gradients but does not obtain the same hardware parallelism as increasing
the physical batch too.

The actual B128 mixed run uses about1.40seconds/update. Discarded preflight
measured3.12GiB peak allocated and4.38GiB reserved on the H10080GB. These
measurements suggest room to test larger physical batches, but do not
establish their throughput or convergence behavior.

The paired A5-only B128 pilot first recorded positive full length36 whole-word
accuracy at9k and reached23.6865% at10k. Thus a short run with zero whole-word
accuracy can precede a sharp improvement. Fuzzy-only stopped at6,521updates
with99.8854% answer-token accuracy; there is no10k Fuzzy-only endpoint.

## Proposed sequence

1. **Profile after the current run.** Test128,256,512 and1024 examples per
   task, with physical microbatch equal to logical batch where feasible.
   Use discarded fresh-state trials, roughly10warm-up plus30timed full
   optimizer updates, extending warm-up only if timing has not stabilized.
   Record task examples/second, seconds/update, and peak memory. Choose the
   smallest batch near the measured throughput plateau.512/task is the
   provisional candidate; prefer256 if it performs similarly, and consider
   1024 only if it gives a material further speed gain. If gains are small,
   retain128. No new precision or compiler study is needed.

2. **Keep one optimization change initially.** Fresh original paired model
   initialization, same corpora and deterministic example streams, unchanged
   FP32 execution, AdamW settings, constant learning rate1e-4, clip1 and
   NextLat weight1. Preserve equal task counts, separate task means and
   objective `0.5*(CE_A5 + NL_A5) + 0.5*(CE_Fuzzy + NL_Fuzzy)`, with one
   optimizer update. Regroup the same ordered examples into larger batches.
   Do not automatically multiply the learning rate by the batch ratio.

3. **Use an inexpensive A5-only control.** At the selected batch, run the
   same fresh A5-only model with retained checkpoints through up to10k
   updates. This provides a control for whether the larger-batch regime
   still learns state tracking, and when. A5-only updates are much cheaper
   than length400 mixed updates. If it fails, diagnose the batch/optimizer
   regime before interpreting a joint failure; any learning-rate adjustment
   gets its own explicit configuration, not a hidden change to the pilot.

4. **Run one mixed screen.** For the provisional512/task choice, evaluate
   at1,250updates and2,500updates; the latter processes the same number of
   examples per task as the current128/task10k pilot. Keep an optional
   extension to5k as a directional budget, guided by both tasks' curves and
   the A5 control. If A5-only itself needs more than5k updates to emerge,
   a5k mixed failure cannot answer whether joint training prevents learning:
   either grant more updates or label that result inconclusive. Do not stop
   solely because length36 whole-word accuracy is zero at an early check.

5. **Evaluate economically.** Use fixed4,096-word A5 monitoring subsets and
   the existing1,280-example Fuzzy development pool. Use full102,400-word
   A5 checks at the declared endpoint and to confirm an apparent positive
   result. Track whole-word and prefix exactness as well as token accuracy;
   Fuzzy answer-token and answer-sequence exactness remain separate metrics.
   Evaluate both tasks at the same checkpoint. Reuse the existing bounded
   correctness checks; do not reopen broad numerical validation.

6. **Spend on stronger controls only after a useful signal.** The existing
   B128 runs provide context; the new same-batch A5 control helps distinguish
   batch effects from joint-training effects on A5. Defer a new Fuzzy-only
   control and additional seeds until a promising or surprising joint result
   merits a firmer claim. Log to W&B and retain checkpoints/evidence in the
   existing GCS project prefix with a new lineage.

## Comparison budget

All batch counts below are **per task**; mixed total batch is twice that.
“Examples” counts presentations from the same finite corpora, not newly
generated unique examples. With unchanged lengths, matched presentations
also match token exposure separately for each task.

| Batch per task | Updates | Examples presented per task | Use |
| ---: | ---: | ---: | --- |
|128|5,000|640,000|Existing A5, Fuzzy and mixed comparison checkpoint |
|512|1,250|640,000|Match that exposure, with fewer optimizer updates |
|128|10,000|1,280,000|Current A5 and mixed endpoint |
|512|2,500|1,280,000|Primary future exposure comparison |
|512|5,000|2,560,000|Directional extension; twice the current exposure |
|512|10,000|5,120,000|Four times the current exposure; not the initial joint budget |

The first pair supplies a saved Fuzzy-only comparison without extrapolating
its6,521-step endpoint. Data-order auditing across different batches should
compare the concatenated sample stream/prefix, not existing per-update chain
hashes: batch boundaries change those hashes even for identical streams.
The current report's strict same-batch control matcher must not be bypassed;
add a clearly labeled cross-batch comparison when implementing this plan.

Plot outcomes against **examples per task**, **optimizer updates**, and
**wall-clock time**. None alone answers every question. Equal examples mean
fewer optimizer/Adam/weight-decay applications at larger batches; equal steps
mean more examples; equal time is the operational speed comparison. Keep
runtime details and evaluation time visible. A larger batch does not
guarantee earlier convergence or faster time to a useful result.

A strong positive joint result establishes capability under this training
regime. It does not, by itself, establish an architectural improvement or
better sample efficiency than the smaller-batch run. A negative result is
weaker evidence, especially at fewer updates or without successful matched
single-task learning. Pilot checkpoint selection is development screening;
final confirmation remains reserved.

General support for these tradeoffs: [McCandlish et al., An Empirical Model
of Large-Batch Training](https://arxiv.org/abs/1812.06162) describes the
compute/time efficiency tradeoff; [Shallue et al., Measuring the Effects of
Data Parallelism on Neural Network Training](https://arxiv.org/abs/1811.03600)
finds workload-dependent scaling and emphasizes hyperparameters and budget.
Neither study predicts the optimal batch for this particular RT pilot.

## Bounded check execution after interruption recovery

Standalone profiler: `scripts/rt_nextlat_a5_fuzzy_batch_profile.py`; the
existing model and training implementation are unchanged. Thirteen focused
CPU tests passed. Physical and logical batches are 128/256/512/1024 per task,
with 10 warmup plus30 timed updates each. W&B receives timing and memory
charts; resulting weights are discarded and no model checkpoints are made.

Execution lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260916T222000Z-d128-batch-profile/`.
Supervisor PID at launch: 11040. It waits for the mixed recovery supervisor
to finish training, reporting and verified GCS retention before using the
GPU. Its `status.json` records the current phase. `STOP` in this lineage or
`STOP_QUEUE` in the mixed lineage cancels the waiting benchmark. The output
report is `docs/reports/rt-nextlat-fuzzy-a5/d128-batch-throughput/report.md`.
The benchmark has no following learning experiments.
