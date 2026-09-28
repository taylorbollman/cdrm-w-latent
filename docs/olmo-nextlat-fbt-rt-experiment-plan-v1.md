# OLMo / NextLat / FBT / RT: proposed experiment campaign

Date: 2026-09-28. **Draft for user review; no training launch is authorized by this document.**

This reconciles the two user-supplied planning briefs with the current repository:
`/home/taylorbollman/olmo_nextlat_fbt_rt_experiment_handoff.md` and
`/home/taylorbollman/olmo_nextlat_fbt_rt_hyperparameter_assessment.md`.
It proposes the next research campaign after the completed functionality and
throughput work. Existing numerical qualifications and historical configurations
remain recorded. This is not a claim that the new campaign recipe is implemented.

## 1. Recommended scope and questions

Use training context **1024**, all eight B/N/F/R/NF/NR/FR/NFR configurations,
and a staged commitment of resources. Primary question: does RT improve the
FBT + NextLat combination enough to justify its cost? Preserve the other three
conditional RT comparisons because RT could help in a different combination.
Also measure the usefulness of N and F relative to ordinary continuation.

Start with a common-recipe, token-matched cohort. Retain compute accounting from
the first pilot; run actual compute-matched extensions for selected comparisons
after costs and early results are known. Equal-token results plus a timing column
are not themselves compute-matched results.

The early endpoint is provisionally **500M additional valid input tokens, then
50M cooldown tokens, then 20M supervised SFT response tokens** per arm. Do not
commit all arms to 1B before the RT review. A one-seed screen guides this project;
it does not establish a general negative claim about recurrence.

## 2. Frozen starting point and treatment definitions

All arms load original `allenai/OLMo-1B`, branch `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`, using the already verified native
weights/tokenizer. Historical token count is approximately 252B; all new budgets
start at zero. The published-batch calculation gives 251,658,240,000, not an
independently recovered trainer counter.

Architecture: 16 layers, width 2048, 16 heads, head width 128, SwiGLU hidden width
8192 per branch, tied 50304-row embedding/readout, native RoPE and non-affine
LayerNorm. Native context capacity is 2048; this campaign trains at 1024. Retain
no added Q/K normalization. All backbone parameters train in the main cohort;
earlier fusion-only experiments are historical references.

| Arm | NextLat | FBT | RT |
| --- | --- | --- | --- |
| B | off | off | off |
| N | on | off | off |
| F | off | on | off |
| R | off | off | on |
| NF | on | on | off |
| NR | on | off | on |
| FR | off | on | on |
| NFR | on | on | on |

- **R:** optimized native RT in layers **0 and 15**, full historical attention,
  recurrence strength alpha=1, original RoPE/normalization, no embedding bypass,
  no restricted window, no learned NextLat rollout. This tests sparse RT in a
  pretrained OLMo, not an all-layer reproduction of the RT paper.
- **F:** four total training passes, shared weights: one without feedback then
  three with causally shifted previous-pass feedback. Full differentiation
  through all passes; one update per effective batch. **RT executes in all four
  passes when R is present.** This is a proposed change from the old bootstrap.
- **N:** existing shared horizon-one predictor, trained within each pass to
  predict the next-position hidden state given the current state and next-token
  embedding. SmoothL1 plus teacher-to-predictor KL, both coefficients 1. No
  separate auxiliary token CE. The predictor is discarded for deployment.
  Preserve detached hidden targets, KL teacher and auxiliary readout, while
  retaining the predictor-input/embedding and inter-pass backbone gradients.

Keep the existing gate-product fusion rather than introducing another fusion
architecture. For previous-position feedback h and current token embedding e:

\[
 f(h,e)=s\,\mathrm{RMS}\left(W_hh\odot\sigma(W_e\mathrm{RMS}(e))\right),
\]

where s is the frozen RMS of the original embedding matrix. At beta=1 this
replaces the eligible input; it is not residual addition to e. The first token
uses its ordinary embedding. Keep current separately seeded fan-in-uniform
matrix initialization and paired fusion/predictor seeds across applicable arms.

Adopt the brief's proposed uniform feedback jitter of +/-0.02 for F training
as an explicit new recipe option, conditional on a bounded implementation check:
add it to the shifted feedback state before fusion, only at eligible positions.
Disable it for evaluation. Record its RMS relative to the actual feedback.
Graph replay must draw fresh noise, recomputation must reuse the appropriate
noise, and restart must restore its RNG. If jitter creates a concrete problem,
resolve it in calibration across the F family rather than changing one arm.
Prefer noise keyed by logical update, pass and document/token coordinates so
changing physical microbatch partitioning does not change assigned noise. A
shared seed alone is insufficient. If that is impractical, disclose the extra
partition-dependent stochastic variation rather than claiming identical draws.

For K=4 with F and K=1 otherwise:

\[
L=\sum_{k=1}^K w_k L_{CE}^{(k)}+
\mathbf 1_N\frac1K\sum_{k=1}^K\left(L_h^{(k)}+L_{KL}^{(k)}\right),
\quad w=(1/2,1/6,1/6,1/6)\text{ for F},\quad w=(1)\text{ otherwise}.
\]

Each component is its own valid-target mean. SmoothL1 also averages hidden
dimensions; KL sums over vocabulary. Full pretraining KL applies on valid
same-document triples, not just the response-half positions used in profiling.
The CE weighting preserves equal total weight for bootstrap and feedback passes;
it is a chosen retrofit recipe, not a literal replication claim.

| Arms | Training parameters | Deployment parameters |
| --- | ---: | ---: |
| B, R | 1,176,764,416 | 1,176,764,416 |
| N, NR | 1,259,491,328 | 1,176,764,416 |
| F, FR | 1,185,153,024 | 1,185,153,024 |
| NF, NFR | 1,267,879,936 | 1,185,153,024 |

RT adds no weights; F adds 8,388,608; N adds 82,726,912 training-only parameters.
Pass count changes compute and activations, not parameter count.

## 3. What must change before this recipe can run

These are scoped extensions, not a new backend project:

| Current implementation/evidence | Required campaign behavior |
| --- | --- |
| FBT always uses ordinary bootstrap | Versioned all-pass RT policy for FR/NFR in eager and static paths |
| Same pass weights for CE, latent and KL; total weight 2 with gamma=1 | Separate normalized CE and auxiliary weights, with matching counters/FLOP estimates |
| No production FBT jitter | Explicit RNG-safe training option and deterministic evaluation |
| CUDA-graph trainer uses one physical batch/update and fixed masks | Correct accumulated updates and realistic variable validity/target masks |
| Real-data adapter is CodeSearchNet/WikiText, limited to 512 | Pinned Dolma and SFT manifests at 1024 |
| Packed independent documents rejected | Preserve independent rows; handle short documents/right padding deliberately |
| AdamW defaults epsilon 1e-8 and decays every matrix | Explicit epsilon 1e-5, tied-embedding exclusion and semantic parameter groups |
| Warmup starts plateau/N | Proposed 10%-to-100% restart warmup |
| FBT likelihood evaluator rejects RT; generation adapter missing | Common evaluation/generation contract for all eight arms |
| Scoped same-process graph reconstruction tested | Fresh-process interruption/restart rehearsal of the actual campaign trainer |

Preserve old defaults/checkpoint meaning. Give the campaign an explicit recipe
version. Do not use a silent edit to reinterpret historical K2 results.

## 4. Data and batch policy

Use a pinned bounded **Dolma v1_5** sample with an explicit source mixture; that
is the release associated with original OLMo-1B, while the current dataset default
is v1_7. Pin source revisions/URLs/hashes, tokenizer, sampling, EOS handling,
document split, windowing and the ordered training manifest. The source mixture
must be resolved for v1_5, not copied from the current v1_7 statistics.
[Official Dolma versions](https://huggingface.co/datasets/allenai/dolma).

Split by document (and duplicate family where available) before windowing. Keep
training, tuning-dev and final confirmation sets disjoint for this continuation.
Exposure during the original 252B pretraining remains unknown; do not claim these
are unseen by the original checkpoint. Use a predetermined extension stream for
later compute-matched training rather than reshuffling the common prefix.
Count any overlapping context tokens as repeated valid input presentations;
record unique source tokens and supervised targets separately.

Start without cross-document packing. Include shorter documents and final
windows using right padding; preserve source proportions by valid tokens. Full
1024-token single-document windows are fine for initial plumbing tests, but are
a biased substitute for the main corpus if used exclusively. Loss boundary
masks alone do not isolate attention or recurrent memory.

Qualify a strict-right-padding fast path: later padding cannot influence earlier
valid outputs under causal attention. Preserve all loss/feedback validity rules
and prohibit carrying padded caches into another document. This can retain
ordinary Flash dispatch without a new attention kernel. CUDA graph mask/index
handling still needs its own implementation; changing a mask tensor is not
sufficient if target-selection indices were baked into the graph. Use bounded
length/layout buckets or graph-safe masked reductions, selected after observing
the data. Report padding utilization and useful valid tokens/s.

**Recommend a common target of 524,288 valid input tokens/update**, about 512
full 1024-token rows. Deterministic logical update membership is shared across
arms; physical batches and accumulation may differ. With variable-length rows,
512 padded rows are not necessarily 524,288 real tokens. Define boundaries by
the shared manifest with small recorded whole-row overshoot; no arm independently
chooses its update's examples. Normalize CE, latent and KL over their respective
whole-update valid counts, not an unweighted average of microbatch means.

For full rows on one GPU, illustrative equivalent partitions are B128 x 4,
B64 x 8, B32 x 16, B16 x 32. These are not measured K4 capacities. Prefer a
physical batch with useful RT utilization and memory headroom. Accumulation
does not replace physical batching for kernel utilization. Clip and update
Adam only after the effective batch; do not divide losses twice. Different
physical partitions need a bounded gradient/update comparison, not assumed
bitwise identity in BF16.

Use the same effective batch for the main comparison. If a different effective
batch materially improves an arm, treat that as a separate tuned follow-up;
it changes updates, noise, moment history and weight decay per token.

## 5. Optimizer and bounded calibration

Use fresh optimizer state in every arm. The pinned source artifact is weights
and tokenizer provenance, not a recovered original optimizer state. Do not
assign Adam step 60000 to zero moments. Separate historical and continuation
counters. Recovering original moments is optional and does not block this plan.

| Setting | Pilot recommendation |
| --- | --- |
| Optimizer | fused AdamW, explicit uniquely owned parameter groups |
| Plateau LR | 2e-4 initially; candidate set 1e-4, 2e-4, 4e-4 |
| Betas / epsilon | (0.9, 0.95) / 1e-5 |
| Decay | 0.1 on eligible matrices; zero on tied embedding/readout, norms, biases/scalars |
| Global gradient clipping | norm 1.0 after accumulation/reduction |
| Warmup | 10% to 100% LR over 52,428,800 valid input tokens, about 100 proposed updates |
| Post-warmup | constant LR on the resumable warm branch |
| New-module LR | 1x, with separate backbone/fusion/predictor telemetry |
| Precision | current BF16-mixed policy with FP32 master parameters/gradients/Adam and native sensitive reductions |

The historical released OLMo YAML specifies peak 4e-4 and a long cosine schedule;
the companion's near-3.95e-4 value at step60000 is a schedule-derived estimate,
not a measured checkpoint optimizer LR. The proposed 2e-4 is a restart/adaptation
choice at a smaller batch. Do not infer LR by dividing by four FBT passes.
[Historical released configuration](https://github.com/allenai/OLMo/blob/v0.4.0/configs/official/OLMo-1B.yaml).

At exactly the target batch, 100/250/500 updates expose 52.4M/131.1M/262.1M
tokens; 500M is about 954 updates. The warmup is about 10.5% of the early warm
budget. Use token-based schedules and record actual update counts when padding
or row boundaries alter these approximations.

Calibrate B, F and NFR at 2e-4 first. Inspect early health around 10/25/50 updates,
warmup-end at about 100, and sustained plateau behavior through about 250 if
needed. Open a neighboring LR only for a concrete learning/stability concern,
not an automatic nine-run grid. Provisional tuning cap: the three anchor pilots
plus at most two focused matched-pair recipe trials, each up to about 131M tokens;
review before exceeding it. If a family needs a change, check affected controls.

If baseline learning is healthy but a combination is unstable, do not force all
arms onto a nonlearning LR. Correct implementation/scaling first; then bounded
LR/warmup sensitivity, then justified auxiliary/module settings. Preserve the
common-recipe cohort and label tuned rescues separately. Consider beta2=.99 or
other RT-paper settings only if observed behavior warrants them.

Every arm then passes its own short pilot. Reuse pilot training when the final
recipe is unchanged; otherwise restart from the original checkpoint. All
discarded pilots count as research expenditure, not primary checkpoint ancestry.

## 6. Diagnostics and inference policy

Reuse prior numerical evidence. Concentrate new checks on all-pass RT, weighted
objectives, jitter, padding, accumulation, and restart. Test small-reference
forward/gradient/causality cases and one representative actual-checkpoint update.
Retain historical BF16 differences; own eager/graph agreement does not erase
them. Escalate precision testing only for concrete new failures. Keep Q/K math
unchanged, monitoring state/logit norms and occasional per-loss gradient scales.

Log each update: valid/allocated/target/pass tokens, per-pass CE/latent/KL,
LRs, unclipped norms, clipping factors, failed/skipped steps, actual updates,
memory, useful throughput and compute. Periodically measure module update/weight
ratios and CE/auxiliary gradient size/alignment on a small fixed probe. Probes
must not alter optimizer state or subsequent data/RNG sequence.

Use small fixed development loss evaluations every 10 updates initially, then
25-50, with a target of roughly <=5-10% evaluation overhead. Tune their size
after timing, not their content after seeing results. Include source/domain
slices and paired per-document loss differences. Broader capabilities belong
at initialization and material token milestones, not every 100 updates.

For F: bounded noiseless 1/2/4/8-pass checks at initialization and pilot reviews;
occasional 16/32-pass and exact-online checks on small selected sequences.
Record CE, normalized state changes, tails and activation percentiles. Extra
untrained passes need not monotonically improve; they are diagnostics, not a
universal-convergence acceptance gate. Assess the trained K4 route and intended
deployment route separately. For R, reuse exact sequential-vs-tiled and cache
checks; verify the newly used combination and context.

**Resolve generation before substantial SFT spending.** Four parallel training
passes and sequential autoregressive feedback are different execution procedures.
Finite-K4 cached prefixes cannot be silently treated as exact-online caches.
Recommended contract:

1. Frequent training-aligned NLL: single pass for non-F, finite K4 for F; report
   all trained passes and predeclare pass4 as the principal finite-F score.
2. Small exact-online teacher-forced NLL for F/FR/NF/NFR, using the same RT
   selection and causal initialization, to quantify the training/inference gap.
3. Prefer exact sequential feedback for primary generated F-model evaluations,
   including exact sequential prefill, if its measured evaluation cost is
   practical. Validate combined RT cache behavior and matching likelihood scores.
4. If exact prefill is prohibitively slow, choose and freeze a clearly labeled
   finite-K4 generation protocol before the cohort, compare it against the exact
   reference on a bounded subset, and budget its cost. Do not silently deploy a
   hybrid cache or change policy per arm/result.

Generated evaluation needs a separate adapter; it is not covered by the current
FBT-only teacher-forced evaluator. Keep common prompts, answer extraction,
generation limits and decoding settings. Primary tasks must fit prompt plus
generation within 1024; a 2048 evaluation is a separate common context extension,
not an unreported advantage or a way to disguise truncation failures.
Determine evaluation eligibility from prompt length plus a fixed generation
allowance, without using reference-answer length or observed success to filter
examples. Report generation-limit failures separately.

Track separately: initial retrofit damage; sustained optimizer health; execution
checks; original held-out recovery; contemporaneous B recovery; and useful
functional benefit. Recovery clocks never reset training-token accounting.
Provisional 0.02 nats recovery or 0.01 nats gain are starting margins to calibrate
on development data, not already validated universal tolerances.

## 7. Training, cooldown and SFT sequence

After calibration, keep B as the moving continuation control. Prioritize NF/NFR
for the user's main RT question, then N/NR and F/FR and B/R so all four RT
contrasts reach the early endpoint. Completed calibration runs can be reused.
Compare at matched saved token milestones even if wall-clock order differs.

Warm checkpoints: approximately 100M, 250M and 500M; later 1B for survivors.
At 500M preserve the full warm state, fork a **50M-token linear cooldown to
zero**, and evaluate warm and cooled states separately. Keep methods/pass count,
data mixture and auxiliary coefficients active. 500M + 50M is a 550M pretraining
endpoint. The later 1B + 100M branch is 1.1B; keep warm ancestors resumable.
Check cooldown adequacy on early B/F development branches before freezing it.

SFT: fresh optimizer and a separately calibrated, common lower-LR recipe.
Propose **2e-5** initially, with one 1e-5/5e-5 neighbor if warranted; about 5%
warmup then a shared decay over response-token progress. These are engineering
starting values, not measured optima. Use roughly **32K supervised response
tokens/update** (16K fallback), giving about 610 (1220) updates over 20M response
tokens. Do not reuse a 524K-response-token batch, which yields only ~38 updates.

Retain the proposed 8M math / 6M code / 4M deduction-state-tracking / 2M general
instruction response-token mix, conditional on actual 1024-token retention:

- MetaMath: group `original_question`/augmentation families before splitting.
- BigCode execution-filtered instructions: use **instruction + response**, not
  its long synthesis `prompt` field. [Dataset schema](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k).
- Fixed deduction/state-tracking development distributions; hold out families
  and complexity/length variants for testing.
- Select genuine general-purpose SmolTalk subsets. Its `all` mixture includes
  math/code sources already budgeted separately; deduplicate across sources.
  [SmolTalk composition](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).

Filter complete templated prompt/response/EOS examples to 1024; record exclusions
by domain and length. Do not truncate away answers. If this leaves too little
useful code/reasoning data, revise the mix or create a separately labeled longer
context branch before drawing conclusions. Report unique examples/repetitions.
No packing initially. Choose explicit loss masks: response-target CE and KL,
and response-position latent targets; all remain within the same document and
may condition on prompt states. Do not exclude the first response merely because
its preceding input is a prompt token. Preserve each objective's pair/triple
causality requirements. Share these policies across all N-containing SFT arms.

Save at 2M/5M/10M/20M response tokens. Include direct-from-original B SFT as a
separate reference, while continued B remains the main control.

Evaluation ladder: frequent held-out Dolma/domain NLL; modest general-capability
development slices; endpoint general suite (PIQA, HellaSwag, WinoGrande, ARC,
OpenBookQA, SciQ, LAMBADA) subject to context feasibility; post-SFT GSM8K,
HumanEval/MBPP with EvalPlus, deduction/state tracking; MATH500 and a fixed BBH
subset secondary if scores are informative. Isolate generated code execution.
Do not interpret a benchmark floor as evidence of equality. Freeze prompts,
datasets/evaluator versions, extraction and the development/final split.

## 8. RT review and compute comparison

Evaluate R-B, NR-N, FR-F and NFR-NF at matched tokens, the 250M-to-500M trend,
the common cooled endpoint and the SFT curve. Compare absolute scores, not only
each arm's improvement from its own starting point. Pick practical gain margins
before main outcomes (e.g. 0.01 nats or 1 point on a chosen domain aggregate as
provisional examples), balancing gains against training/inference cost.

- Clear useful conditional gain: retain that RT arm and its non-RT control.
- Improving or uncertain near the margin: bounded matched rescue or second
  paired seed, possibly carry the pair to 1B.
- All four unpromising after healthy learning and bounded checks: deprioritize
  this RT treatment. Preserve checkpoints and state the tested scope.
- Failure to adapt or execute: record adaptation/implementation failure; do not
  equate it to evidence that a correctly functioning RT has no capacity benefit.

Use paired document/problem uncertainty and training-seed confirmation for close
decisions. A wide interval is inconclusive. Selecting among four conditional
contrasts also needs confirmation; one noisy positive is not decisive. A full
factorial comparison beyond the screen requires keeping all eight arms.

Count input tokens once irrespective of passes. Record pass-tokens separately.
Primary compute matching uses cumulative estimated executed training FLOPs
(including checkpoint recomputation and auxiliary heads), with the estimator
version and limitations. Also report measured accelerator-hours and elapsed
time inclusive of data/evaluation/checkpoint overhead. Hardware throughput can
make FLOP matching and GPU-hour matching disagree; label the chosen question.

For cooled compute-matched arms choose a warm point T_i satisfying
`C_i(T_i) + D_i(cooldown_tokens) = C_target`; use the same cooldown-token budget
within that cohort. State the compared set and anchor, usually the most costly
selected equal-token endpoint after measurement. Include SFT cost too only when
claiming an end-to-end compute match. Equal SFT response tokens alone are not
equal SFT compute. Keep checkpoint ancestry and total tuning/research expenditure
separate. Start with selected paired extensions before an expensive eight-arm
compute-matched campaign.

## 9. Capacity, hardware and retention

Existing T1024 K2 NFR measurement: 11,092 input tokens/s at physical B64,
68.45 GiB reserved; about 12.52 compute-only hours/500M. **Do not reuse that as
the new estimate.** K4, RT on bootstrap, full KL, jitter, padding and accumulation
change the workload. Actual K4 B/F/NFR timing and memory is a readiness deliverable;
profile each remaining arm briefly before allocating its long run.

For orientation only, 500M tokens take 27.8h at 5k valid tokens/s, 13.9h at 10k,
or 6.9h at 20k, before overhead. These are arithmetic examples, not forecasts.
The complete eight-arm 550M endpoint is 4.4B input tokens, plus 160M SFT response
tokens and prompt processing, diagnostics and tuning. This is a multi-day
campaign, not one 12-hour experiment.

**Hardware clarification, 2026-09-28:** the user confirms quality training will
use multiple GPUs; multiple H200 141GB devices are a possibility, with an eight-
GPU H200 allocation currently the available H200 option. Final hardware is not
selected. Do reusable CPU/one-GPU data, objective, masking, optimizer and small
reference checks first; do not build a separate single-GPU performance campaign
or run the full LR calibration there. Move to two GPUs as soon as the distributed
trainer integration is ready, before graph-accumulation/recovery qualification
and sustained calibration. One-GPU reference checks can use one device on that
same machine. Existing two-H100 evidence is reused, with new checks focused on
the changed K4/RT-bootstrap/full-KL/jitter/data/accumulation recipe.

Measure actual K4 per-rank physical batch capacity starting conservatively
(e.g. B8/16, then 32 if feasible); do not assume old B64 fits. Keep Flash SDPA for ordinary
attention and native tiled RT, existing recomputation/checkpointing and accepted
fusions. FA4 remains deferred. Use CUDA graphs after the new objective/layout/
accumulation path passes a bounded check; no broad backend search is a prerequisite.

Two H100s are a useful development platform even if final training uses H200s.
H100 and H200 both have CUDA compute capability 9.0; this supports expected
kernel portability, not guaranteed identical numerics/performance.
[NVIDIA GPU capabilities](https://developer.nvidia.com/cuda/gpus).
On the final hardware, rebuild graphs/JIT caches as needed and run a bounded
kernel/loss-gradient/update check, actual-rank NCCL/accumulation check, fresh-
process checkpoint restart and realistic peak-memory/throughput check. Validate
the actual interconnect and software stack. Identical tensor shapes/dtypes give
useful memory estimates, but kernel workspaces, graph pools, collectives and
checkpoint staging still need measurement. Do not scale batch capacity simply
by 141/80 or assume DDP pools device memory.

GPU-type migration at the same rank count is distinct from 2-to-8-rank resume.
Current distributed checkpoints reject changed world size; ZeRO1 also records
the parameter-to-rank optimizer partition. Several historical test launchers
require exactly two ranks. If mid-run rank-count changes are needed, implement
explicit model/optimizer migration, global data-cursor redistribution and an
RNG policy at a completed update boundary, then test it. Do not promise bitwise
trajectory identity across different reductions/partitions. Starting all pilot
arms on the final rank count avoids that checkpoint conversion requirement.

Keep global valid tokens/update fixed while changing ranks, physical batch and
accumulation. On an eight-GPU node, two-rank diagnostics can expose only a pair,
but do not certify eight-rank execution; qualify the actual training group size.
Before reserving eight GPUs only for development, finish reusable work on a
cheaper machine when available. If the eight-GPU node is already allocated,
other disjoint pairs may run independent readiness jobs. Later compare one
eight-GPU training job against multiple multi-GPU arms (e.g. four two-GPU jobs)
using aggregate campaign throughput and the same effective batch. Use DDP first;
ZeRO1 if state memory limits useful physical batches, ZeRO2 only for a measured
bottleneck. No hardware launch or allocation is authorized by this clarification.

Host disk currently has only about **12 GiB free** (2026-09-28 observation).
Stage bulk data/checkpoints on ephemeral SSD and verify durable GCS copies;
do not fill the persistent filesystem with full optimizer checkpoints. Recheck
capacity during setup. Keep code/manifests/reports in the repo and retained data,
source snapshots and checkpoints under `gs://fast-chunks/cdrm-w-latent/`.

Checkpoint at safe optimizer boundaries approximately every 20-30 minutes,
plus milestones and branch points. Save model, optimizer, schedule, counters,
RNG including jitter, data cursor/order, config and source identity. Restore
from a cloud-verified checkpoint in a fresh process before long runs. Keep a
small rolling local set only after verifying durable copies. Graph warmup or
capture must not silently consume uncounted training updates or data. Log all
graphable work to the authorized `taylorbollman` W&B account with campaign/arm/
seed/recipe/branch identifiers. Restore RNG around graph warmup/reconstruction
as well, so it cannot consume uncounted jitter draws or perturb resumed data.

## 10. Proposed review milestones

1. **Campaign contract and trainer readiness:** implement explicit model/loss/
   optimizer policies, graph-safe accumulation/masks, bounded new-path tests,
   Dolma manifest and restart. Small full-window plumbing fixtures can precede
   representative data. Deliver source-pinned recipes and no hidden default drift.
2. **Inference and capacity readiness:** all-eight small functional checks;
   generation contract and baseline evaluation; realistic B/F/NFR T1024 capacity,
   useful throughput and end-to-end budget. Confirm practical batches and SFT
   source retention before making the large commitment.
3. **Calibration and recipe freeze:** B/F/NFR bounded pilots and necessary matched
   sensitivity; all remaining arms checked independently. Deliver selected common
   recipe, initial retrofit/recovery diagnostics and an actual campaign schedule.
4. **Early eight-arm campaign:** reuse compatible pilots, continue to 500M,
   50M cooldown, common SFT and four conditional RT comparisons. Milestones may
   expose issues that warrant a targeted pause; do not silently retune an arm.
5. **Selection:** retain useful/ambiguous pairs, extend to 1B and chosen
   compute-matched endpoints. Add seeds for close decisions. Feature matching
   can be an endpoint diagnostic after core inference is validated; Semantic
   Tube and other architectural additions remain deferred.

The next implementation milestone is (1), after review of this draft. The main
decision points are measurable readiness/cost, the common LR/effective batch,
generation policy, SFT retention at 1024, and which RT comparisons survive the
early screen. No long-run promise relies on the historical K2 benchmark.

## Local implementation evidence

- `cdrm/pretrained/olmo_fbt.py`, `olmo_static.py`: bootstrap, fusion, online cache.
- `cdrm/pretrained/fbt_training.py`, `nextlat.py`: pass aggregation, loss masks,
  reductions and stop-gradients.
- `cdrm/pretrained/lm_training.py`, `static_training.py`: optimizer grouping,
  eager accumulation, static training and checkpoint contracts.
- `cdrm/pretrained/lm_data.py`, `fbt_evaluation.py`: current data/evaluator scope.
- `docs/reports/olmo-combined-t1024/results.md`: latest context measurements.
- `docs/reports/olmo-two-gpu/results.md`: distributed evidence and qualifications.
- `docs/fbt-rt-nextlat-handoff.md`: historical numerical and execution record.
