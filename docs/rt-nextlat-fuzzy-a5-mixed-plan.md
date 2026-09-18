# Two-layer restricted-first RT + NextLat: Fuzzy Recall and mixed A5

Date: 2026-09-16. Status: **proposed implementation and experiment plan**.
The user confirmed that NextLat training should be retained. This document
does not launch training or authorize a numerical/compiler research program.

## 1. Questions and scope

First establish whether our two-layer, restricted-first RT + NextLat learns
native MAD Fuzzy In-Context Recall at long sequence lengths. Then test whether
the same shared model can learn retrieval and A5 state tracking together, with
equal numbers of examples from each task in every optimizer update.

Use **400 → 512 → 1024 tokens**, adding 768 only if helpful to locate a change.
The paper's Fuzzy Recall plot contains lengths 64/128/256/512. The pinned MAD
configuration includes 1024 as its longest length preset; 400 is our added
starting point. The paper's synthetic model is one layer, width 128, FFN 512,
and 16 heads. Our two-layer width-512 experiment uses the paper's task but has
its own architecture and training protocol. The exact Figure 6 training
duration and checkpoint-selection procedure are not established by D.2.
[Paper D.2](https://arxiv.org/html/2604.21215v1#A4.SS2),
[original Figure 6](https://arxiv.org/html/2604.21215v1/all_tasks_TokenAccuracy.svg),
[pinned MAD task configuration](../vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml).

“Longest” is not necessarily “hardest.” Longer words increase retrieval
distances but also repeated evidence. Native held-out three-token keys have
only 7 × 6 × 5 = 210 possible motifs. Keep vocabulary, motif sizes, noise and
training-set size fixed while changing length. Vocabulary expansion or less
training data would be separately named difficulty experiments.

## 2. Model contract

Reuse the architecture of `rt_window2_first` in
[rt_a5_depth_order.py](../scripts/rt_a5_depth_order.py), with the already
checked window implementation in [rt_a5_window.py](../scripts/rt_a5_window.py).

| Element | Proposed setting |
| --- | --- |
| Depth | Two RT blocks |
| Block index 0 | Direct attention to current temporary K/V and immediately previous position's permanent output K/V |
| Block index 1 | Full causal recurrent attention |
| Width / heads / FFN | 512 / 8 heads of dimension 64 / 2048, GELU |
| Normalization / positions | Existing pre-LayerNorm and learned full-width Q/K normalization; ALiBi |
| Initialization / recurrence | Original Mitchell initialization; tiled RT; rho = 1 |
| Embedding additions | None: no input bypass, value bypass, or reassigned embedding head |
| Runtime | Full FP32, eager, TF32/autocast/compile/CUDA graphs off initially |
| Evaluation | RT backbone and task readout; no predictor rollout |

The second block consumes the first block's output normally. Permanent keys
and values still come from the corresponding RT block's processed outputs;
temporary self K/V and queries come from its current input, using our existing
normalizations. The first block's window limits direct reads, **not** the
history carried recursively through previous outputs or gradient propagation.

Retain the original NextLat predictor and objective. With post-final-norm
latent h_t and input-token embedding E(x_t):

\[
\widehat h_{t+1}=h_t+f_\theta([E(x_{t+1}),h_t]),\qquad
L_{\mathrm{NL}}=\operatorname{mean}_{b,t,d}
\operatorname{SmoothL1}_{\beta=1}
(\widehat h_{t+1},\operatorname{stopgrad}(h_{t+1})).
\]

The predictor retains its RMSNorm, three bias-free linear maps/GELU structure,
hidden width 512, independent initialization, and weight-one auxiliary loss.
Only the target role is detached; current latents and conditioning embeddings
remain differentiable. No EMA teacher, extra latent CE, or autonomous MLP
recurrence is introduced.

NextLat receives the **actual next input token**, not a next-token prediction
or a masked supervision label. Thus unpredictable new symbols in Fuzzy Recall
do not make its auxiliary target an impossible unconditioned forecasting task.

## 3. Common task interface, established before standalone training

Use disjoint input symbol ranges in a shared embedding table:

- A5 operations: global IDs 0–59; identity 0 remains an ordinary valid token.
- Fuzzy symbols: native IDs 0–15 mapped to global IDs 60–75.

Use task-local output distributions, implemented as slices of a 76-row output
table: 60 classes for A5 and 16 for Fuzzy. Targets retain task-local IDs. The
task selects the output slice before softmax and CE. No extra task-prefix token
is needed, so original lengths and supervision alignment stay intact.

Both RT blocks, final normalization, and the NextLat predictor are shared.
This tests shared computation with task-specific symbol readouts; it is an
explicit modeling choice rather than a 76-class unconstrained output task.

All new single-task and mixed arms instantiate this same interface. Start from
the original canonical two-layer initialization, retain its core, predictor
and 60 A5 rows, and append Fuzzy rows using a separate recorded RNG scope and
the corresponding initialization rule. Copy the resulting tensors to paired
arms. Verify actual parameter counts; the expected total is **7,423,488**,
including the training-only predictor. Preserve historical factories and runs.

With table parameters shared across tasks, AdamW can decay inactive rows in
single-task arms despite zero gradients there. That does not affect their
active task's computation. Pairing means identical starting tensors, not
permanently frozen unused rows; do not treat a single-task checkpoint as
untouched initialization for a later cross-task warm start.

## 4. Dataset and supervision contracts

Reuse [cdrm/mad_data.py](../cdrm/mad_data.py), its parser/oracle/baseline audit,
and MAD revision `0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`.

For each Fuzzy length, initially freeze 12,800 training examples, 1,280
development examples and 1,280 final-test examples with independent seeds and
manifests. Audit exact input overlap explicitly. Fix vocabulary 16,
multi-query enabled, maximum key/value lengths 3/3, and noise 0. Training key
lengths are 1–3; evaluation keys have length 3. Value lengths are 1–3 in both.

Important adapter details:

- The native generator already aligns next-token labels. Its actual input
  length equals configured T. Do not apply a second shift.
- Native training CE is dense, including left-padding symbol 15; held-out
  accuracy uses the native repeated-key and terminal-query answer mask.
  Preserve this difference. Dense training accuracy is not the paper metric.
- Native padding remains symbolic input. Do not add a padding-attention mask
  that changes the reference task. Retain native within-example transitions
  in NextLat, including these padding transitions. Log the padding fraction;
  auxiliary loss is not restricted to retrieval-answer positions.
- Some native terminal queries have no previously presented mapping. Retain
  their labels in the primary result, and report known-history accuracy and
  oracle availability separately. Availability is not a universal statistical
  ceiling. Reuse annotation of the actual examples rather than attempting to
  recover masks by regenerating with a different training/evaluation flag.

For A5, reuse the frozen corpus from
`.runtime/rt-a5/20260911T154748Z/data`: 800,000 training words of length 12,
the existing short development pool, and the existing length-36 OOD
development pool. Its manifest SHA is
`944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.
A5 targets remain the cumulative group state after the operation at the
**same position**, with no LM shift, BOS/EOS, or target feedback. Keep its
reserved confirmation data separate from screening.

## 5. Implementation and bounded validation

Add a new task-aware factory/objective/trainer/config family, for example
`cdrm/rt_nextlat_tasks.py`, `configs/rt_nextlat_tasks/`, and
`scripts/rt_nextlat_tasks_{prepare,validate,train,report}.py`.
Reuse the existing RT/window/predictor, data and metric primitives.

The current A5 factory/loss has V60/T36 and parameter-count assumptions. The
old CDRM Fuzzy runner deliberately enforces its historical 12-block calibration.
Neither should be repurposed through unnoticed changes to those contracts.
The new interface needs runtime vocabulary, maximum sequence length of at
least 1024, task-specific logits/CE, independent latent-transition masks,
microbatch accumulation, and task-specific evaluation.

Keep correctness work bounded:

1. Check native labels, padding, answer masks and one known causal lookup
   example against the existing adapter. Verify both task ID mappings.
2. Verify the new two-layer factory preserves the original core and window
   semantics, with no embedding modifications. Check causal behavior and
   isolation across examples/forward calls.
3. Check NextLat alignment, target-role detachment and active source/embedding
   gradients. Weight-zero auxiliary loss must recover task CE.
4. Compare gradients from a tiny direct sum of the two task losses with their
   correctly weighted microbatch accumulation. Test short save/resume across
   both data streams, including optimizer/RNG/order state.
5. Perform one small FP32 tiled/reference integration check at a manageable
   length, then bounded forward/backward/update checks at actual T400 and
   T1024 with small physical batches. Escalate only for a concrete failure.

All tensor/GPU work uses the project container with location/GPU checks.
Profile warmed-up update time, evaluation cost and memory for physical
microbatches (e.g. 16/32/64/128 as feasible), leaving comfortable headroom.
Smoke-test the chosen physical microbatch shape and a complete accumulated
logical mixed update before its training run.
The existing window applies a dense mask to the tiled scan; it is not a new
linear-time window kernel. Small parameter count does not establish cheap
T1024 training. Publish measured budget estimates before committing long runs.
If full FP32 is impractical, propose a scoped use of the already validated
precision path; do not silently switch runtime or revive broad numeric studies.

## 6. Standalone Fuzzy experiment

Train fresh, identically initialized models independently at T400, T512 and
T1024, using the common interface. Keep data order seeds and optimizer policy
recorded. A separate train-length/test-length matrix may evaluate their saved
checkpoints across lengths. Label that as length generalization; carrying
weights along 400 → 512 → 1024 would instead be a curriculum experiment.

Provisional initial budget, to be checked against the profile:

- Logical batch: **128 Fuzzy examples/update**, accumulated if needed.
- Optimizer: retain our A5/NextLat AdamW baseline initially: constant LR 1e-4,
  betas (0.9, 0.95), epsilon 1e-8, matrix decay .01, norm decay 0, clip 1.
- Initial endpoint: **10,000 updates = 100 passes over 12,800 examples**.
  Checkpoints at initialization, 1k, 5k and 10k; development monitoring
  initially every 500 updates, adjusted only for measured evaluation cost.
- Start at T400. If learning stalls, first verify training/subset fit and
  native answer metrics. A bounded LR comparison at T400 (1e-4 and 5e-4,
  adding 1e-3 only if useful) may precede the frozen sweep. Give candidates
  equal exposure and select on development only; no broad optimizer search.
- Freeze the selected protocol across lengths. If an early near-ceiling run
  warrants stopping, retain its checkpoint and clearly mark the smaller
  budget. Use fixed common checkpoints for any budget-matched curve.

Primary metric: **native masked answer-token accuracy**. Also report answer
CE, whole answered-sequence exact match, first-token-of-value accuracy,
terminal-probe accuracy, and accuracy versus retrieval distance. These are
teacher-forced measurements, including exact match. Compute the
first-token and distance breakdowns on native scored answer positions only;
do not include ordinary first presentations of a new key/value pair.
Recompute the query-ignoring answer-prefix shortcut on every new corpus; 12.5% uniform
value-token chance alone is insufficient evidence of retrieval. Log distinct
keys, repetition density and number of scored answers per example.

Do **not** reuse A5's “0% length-36 exactness at 10k” stopping rule. A long
Fuzzy sequence can have useful token-level retrieval while no example has
every answer correct. If the restricted model fails, a bounded same-size
two-layer full-window RT + NextLat control is the first architectural
diagnostic, alongside checking optimization/supervision.

Use development results to select an informative Fuzzy setting for joint
training: preferably T1024 if it learns enough to measure interference;
otherwise retain the strongest informative setting and state the unmastered
scope. If all settings saturate, retain the longest and distinguish saturated
task retention from a sensitivity test. Before strong claims, repeat the
selected comparison with a second initialization; do not triple the full
screening sweep automatically.

## 7. Mixed A5 + Fuzzy experiment

Start mixed training from the paired **fresh initialization**. The chronological
order “Fuzzy, then mixed” is experimental scheduling, not a Fuzzy-checkpoint
warm start. Sequential transfer could be a later explicitly separate question.

Each logical batch contains exactly n A5 examples and n Fuzzy examples,
initially **n = 128**. Use homogeneous physical subbatches at each task's own
length. Accumulate them into one optimizer update, rather than padding A5 to
1024, concatenating independent examples into a stream, or alternating
single-task optimizer steps.

Define task-normalized losses:

\[
L=\tfrac12\left(L_{\mathrm{CE,A5}}+L_{\mathrm{NL,A5}}\right)
 +\tfrac12\left(L_{\mathrm{CE,Fuzzy}}+L_{\mathrm{NL,Fuzzy}}\right).
\]

Within each task, normalize CE by its valid training targets and NextLat by
its within-example transitions times latent width. With the fixed lengths,
this also gives equal example weighting within each task. Equal example
counts with a pooled token mean would instead give T400 Fuzzy about 33 times
as many CE terms as T12 A5. The proposed half-weighted task means deliberately
avoid that accidental weighting; equal scalar weights need not produce
equal gradient magnitudes.

Zero gradients once, backpropagate properly scaled microbatch sums, clip the
combined gradient once and take one AdamW step. Retain separate task data
orders/cursors and counts in every checkpoint. Log CE and NextLat separately
per task; sampled per-task shared-gradient norms can diagnose domination if
the curves suggest it, without adding a full gradient study by default.

Matched comparison arms at the selected Fuzzy length:

| Arm | Examples per update | Purpose |
| --- | --- | --- |
| A5-only | n A5 | New batch/interface control for state tracking |
| Fuzzy-only | n Fuzzy | Retrieval control; reuse compatible standalone run |
| Joint | n A5 + n Fuzzy | Shared model learns both simultaneously |

Use identical common initialization, per-task data order, task exposure at
each compared endpoint, optimizer settings and task-local readouts. The
single-task losses retain their normal full weight; the joint objective is
their mean. Equal task exposure, equal optimizer updates and equal compute
are different comparisons. Report all three and make no equal-compute claim.

Initial joint and A5-only endpoint: 10k updates, with 1k/5k/10k checkpoints.
At n128 this is **1.28M A5 training examples**, versus **10.24M** in the old
B1024/10k pilot. Changing batch size also changes optimization; historical
A5 curves are context, not the matched control. If the new A5-only control
is undertrained but improving, propose a bounded 25k extension of it and
joint training together; extend Fuzzy-only if needed for that Fuzzy endpoint.
If A5-only is flat, investigate its optimization before claiming joint-task
interference. Do not diagnose failure solely from old update-count expectations.

Evaluate A5 training-length token/whole-word accuracy and length-36 token,
final-state and whole-word accuracy, plus cumulative prefix exactness
(especially positions 13–16). Evaluate Fuzzy with its native answer mask.
Use fixed development subsets for frequent curves and full frozen development
sets at principal endpoints. Never aggregate both tasks into one “accuracy.”
Assess simultaneous capability using **the same joint checkpoint** for both
tasks at the declared endpoint. Separate per-task best checkpoints may be
diagnostic, but cannot establish that both capabilities coexist in one model.

The first result will answer whether both capabilities coexist at the chosen
budget. It will not establish that NextLat is necessary on Fuzzy or that the
architecture beats parameter-matched alternatives; those require later
ablations if the result warrants them.

## 8. Artifacts, review points and completion

Log graphable runs online under `taylorbollman`, proposed W&B project
`rt-nextlat-fuzzy-a5`, tagged by task, length, architecture, seed and budget.
Report update count, examples and tokens per task, wall time, memory, separate
CE/NextLat losses and separate evaluation metrics.

Save resolved configurations, source/data/initialization hashes, checkpoints
with optimizer and both order/RNG states, metrics and plots under a new
project-local runtime lineage. Retain useful datasets/checkpoints and final
evidence at `gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/<run-id>/` with
verified hashes; SSD-only artifacts are not sufficient. Checkpoint stops and
extensions must be configurable without changing the model or restarting
data orders. Record exact checkpoint IDs in every plotted series.

Suggested milestones:

1. Reviewable task/model integration, bounded correctness results, real-shape
   profile and the resolved training budget.
2. Standalone Fuzzy length results; freeze the joint setting and comparison
   protocol using development data.
3. Matched A5-only/Fuzzy-only/joint results, followed by a selected second seed
   if informative. Final test/confirmation is evaluated only after choices
   are frozen; early diagnostic reports are labeled development results.

This planning change adds documentation only. No new model, dataset, training
job, external project, or numerical experiment was created for it.
