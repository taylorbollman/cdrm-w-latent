# Paired NF continuation: KL 1 versus KL 0.1

The user approved this milestone after the
[saved-state feedback diagnosis](../olmo-feedback-diagnostic/results.md).
Run two NF branches from the **same complete update-32 checkpoint**, through
update 64. The only mathematical change between branches is the NextLat KL
coefficient: control 1.0 versus intervention 0.1. This is a bounded directional
test of the diagnosed loss interaction, not a search for a winning model.

## Starting authority and exposure

Parent publication:
`.runtime/olmo-adaptation-pilot/native-nf12-first32-01/checkpoint-publications/update-000032.json`
has SHA256
`ff986113eb32709b904e4cd6b6afe09b3ccda4023948c3637784cfe75f300754`.
Its distributed checkpoint manifest SHA256 is
`3dec8c32d183b8f371a533d0b5176e50cd02867daa67157eeecdf10c174ceb2a`.
Use this authority for both branches, including model, Adam, scheduler and
rank-local state; a weights-only import is not sufficient.

The parent has completed 32 updates, 16,777,216 input tokens, 16,384 packed rows,
1,408 rank microbatches, 16,760,832 CE targets, 16,730,817 latent pairs and
16,684,487 KL triples. Its next ordered chunk is 16,384; update 33 is the first
new update. Preserve these counters, rather than resetting a branch to zero.
Each branch receives 32 additional matched updates and 16,777,216 new input
tokens. At update 64, each has seen 33,554,432 inputs since the paired pilot
origin, plus the separately recorded earlier fusion adaptation exposure.

Retain the original 128-update finite data plan and scheduler prefix. Its
schedule hash is
`993d922340322268d7d0092b3303e8650c9e96a6b33e3c14055353e91041ef1c`.
The stop at 64 is a segment bound, not a new scheduler horizon. Do not shorten
the schedule to 64 or restart the 100-update warmup. Update 33 uses LR 7.76e-5;
the boundary after update 64 sets the next LR to 1.352e-4. Both endpoints remain
inside warmup.

## Settings held fixed

| Setting | Both branches |
| --- | --- |
| Architecture | NF: ordinary OLMo backbone, K4 FBT and NextLat; no RT |
| Feedback | Beta 1; configured first pass; training jitter 0.02 |
| Context and data | T1024; unchanged ordered corpus, chunks, document IDs and masks |
| Parallel execution | Two H100s; physical batch 12 per rank; effective 524,288 input tokens per update |
| Precision and graphs | Original BF16 mixed production path, FP32 masters/Adam, prepared CUDA graphs and accepted checkpointing settings |
| Optimizer | Same complete saved fused AdamW state, parameter ownership, decay groups, clocks and scheduler |
| CE and latent | CE weight 1, latent weight 1; unchanged pass coefficients |
| KL | Control 1.0; intervention 0.1 |

With 512 real T1024 rows per update, each rank executes 22 physical microbatches
of capacity 12; final padding/dummy rows retain the accepted masks. Normalize
using actual global CE/latent/KL counts, not physical capacity. Jitter remains
keyed by the same logical update and row identity. Branch names and checkpoint
paths must not enter the data/noise seed calculation.

The objective is

\[
L_q=\tfrac12 C_1+\tfrac16(C_2+C_3+C_4)
     +\tfrac14\sum_{p=1}^4 A_p
     +q\,\tfrac14\sum_{p=1}^4 Q_p,\qquad q\in\{1,0.1\}.
\]

Each term uses its own eligible-target denominator. SmoothL1 targets remain
detached, and KL remains teacher-to-student with detached teacher and readout
weights. The predictor's embedding input remains differentiable. No kernel,
forward recurrence, predictor architecture or loss mask is changed. Both KL
weights are positive, so enabled terms, target counters, optimizer ownership
and graph participation remain the same.

## Explicit branch transition

The new branch uses the existing execution-identity envelope for compatibility
with verified storage, with an explicit versioned `objective_branch` payload.
It records the parent publication/manifest/state authority, original and
effective objectives, inherited-state policy, branch identity, new helper pins
and unchanged training settings. The control also has its own clearly labeled
branch identity and output/checkpoint namespace.

At first entry, load the parent under **its exact original configuration and
source fingerprint**, using the accepted strict distributed loader. Restore
both ranks' cursors and RNG plus model, buffers, module modes, Adam moments,
Adam steps, scheduler and counters. Validate clocks and the complete boundary.
Only then replace the immutable `lambda_kl` config field before constructing
any objective adapter or graph. Preserve all other wrapper/predictor config
fields individually: historical predictor metadata can differ from wrapper
document-policy metadata. Do not reconstruct either module.

Record exact before/after equality of parameter and buffer contents/ownership,
Adam and scheduler state, counters, cursor and RNG across the transition.
Only objective metadata may change. The actual first adapted update is 33.
Subsequent child checkpoints contain the effective child configuration and
fingerprint; their ordinary resumes must match those exactly. A child resume
does not repeat the parent transition or silently apply another KL override.
The bounded launcher still authenticates the original parent files on a child
resume, so retain the parent checkpoint and pinned report as recovery inputs.

Neither the parent receipt nor saved payload is rewritten. The frozen 200-file
training implementation remains unchanged; new continuation helpers live under
`scripts/` and carry their own explicit source pins. An effective recipe must
describe inherited Adam and the actual KL weight, rather than exposing the
historical recipe's hardcoded `fresh`/KL-1 metadata as current behavior.

## Evaluation and reporting

Use the same fixed `dev-main` prefix of 65,536 input tokens, full-vocabulary CE,
FP32 math and no jitter, with physical evaluation batch 1 per rank. Retain the
accepted restored-boundary evaluation at 32, then evaluate at 48 and 64.
The two update-32 evaluations should have identical raw per-pass results;
their weighted total objectives will differ mechanically. Repeated evaluation
of a restored scheduled boundary is labeled as a repeat, not extra exposure.

`scripts/olmo_kl_evaluation.py` retains the accepted per-pass evaluator and
ordered panel materialization. Its narrowly versioned reducer accepts the
actual KL 1 or 0.1 coefficient while preserving independent global counts and
unscaled sums. Log raw CE, latent and KL separately; compare raw per-pass CE
and later-minus-first-pass gaps across branches. The weighted total objective
is branch-specific and is not a cross-branch quality score.

Track training loss means, pre-clipping gradient norm, clipping frequency and
coefficient, LR, exposure, memory and the existing timing scopes in W&B under
`taylorbollman/pretrained-fbt-rt-nextlat`. Do not interpret the immediate 90%
reduction in KL's weighted contribution as learned improvement. Adam history
and global clipping mean a 10-fold coefficient reduction does not imply a
10-fold reduction in parameter updates.

## Bounded acceptance before native continuation

1. CPU checks cover the declared objective override, preserved parameter/config
   ownership, rejected undeclared changes, evaluator counts/weights, and strict
   distinction between parent import and child resume. Keep old declaration
   source discovery unchanged.
2. Use the existing tiny NFR three-update plan and a saved update-1 checkpoint
   with populated Adam. Verify the KL-1 child through updates 2–3 matches the
   accepted continuation, including input/noise identities and complete state.
3. Run the KL-0.1 tiny child, save at update 2, retain and restore from the pinned
   cloud publication in a fresh process, then run update 3. Compare against its
   uninterrupted KL-0.1 reference. Test wrong-parent/branch/weight rejection
   before state mutation. This exercises the new branch transition and exact
   child recovery without a long numerical campaign.
4. Native entry checks confirm the exact common update-32 boundary, actual
   effective coefficient and unchanged clocks before graph preparation. The
   accepted preparation checks must preserve this boundary. Any native restart
   is reported separately; tiny restart success is not labeled native exact
   restart evidence.

The tiny NFR case checks compatibility with RT present; the actual research
comparison remains NF. No new native precision sweep is required.

## Execution, storage and stopping

Run the two branches sequentially, using both GPUs for each. Parent and child
checkpoint namespaces are distinct. Reuse synchronous completed-boundary SSD
saves followed by the accepted asynchronous CPU upload/verification worker;
never give a worker mutable model/optimizer state. Preserve cadence and
maximum-one-pending behavior, publish before pruning, and drain at termination.
The latest verified cloud publication is the recovery authority. An uploaded
checkpoint still being verified is not yet that authority; unexpected VM loss
can roll back work completed after the most recent verified boundary.

Keep useful reports, immutable sources and checkpoint authorities in
`gs://fast-chunks`, with local progress every 20–30 minutes or sooner. The
earlier 32-update NF stage took about 52 minutes, including preparation,
evaluation and retention; roughly 36 minutes lay in update callbacks. This is
an initial planning reference, not a throughput guarantee. Two native branches
therefore likely take around two hours including checkpoint overhead, in
addition to bounded implementation and acceptance work.

Stop at update 64 and summarize both branches. Stop earlier for nonfinite
values, state/configuration mismatch, graph or evaluation preservation failure,
failed checkpoint publication, or explicit user instruction. Do not continue
past 64 or add a second intervention automatically. Heavy clipping alone is
an observation to compare, not a newly invented stop threshold.

Interpret the outcome as a short, single-seed continuation inside warmup on the
same limited development prefix. Comparable control improvement supports
ordinary adaptation; selective raw-CE improvement under reduced KL motivates
a matched NFR follow-up. Persistently poor later passes motivates feedback
startup investigation. Neither outcome settles RT's value or establishes
long-run model quality.
