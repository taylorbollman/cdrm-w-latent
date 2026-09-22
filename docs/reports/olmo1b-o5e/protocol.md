# O5e: ordinary continuation on the fusion-repair data

Authorized 2026-09-22 after O5d. One bounded ordinary-model continuation is
compared with the completed O5c mixed fusion adaptation and O5d sequential
evaluation. This adds no RT, NextLat, feedback learning or hyperparameter sweep.

## Starting weights, trainable parameters and objective

Start from the **same O5b FBT update2634 native backbone** used to start O5c,
checkpoint SHA256
`99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`.
Do not substitute the separately trained O5b ordinary arm or start from an O5c
endpoint. Its ordinary scores are code1.698216 / WikiText3.183361 on the final
512-window selection. Verify complete model bytes, inherited sources and native
state identity before training; load model weights only and discard parent
optimizer, counters and RNG.

Train all65 native parameters, 1,176,764,416 elements, including the tied
embedding/readout. Keep the two inherited fusion matrices and fixed output-scale
buffer unchanged and unused. Their presence in the checkpoint wrapper does not
make this a feedback model. Explicit mode: FBT disabled, K1, beta0, empty RT
selection, NextLat disabled. Preserve the native OLMo architecture and BF16
mixed/FP32 parameter settings; SDPA, TF32 off, no compile/graphs/distributed.

The objective is **one ordinary CE loss**, coefficient1, normalized by each
update's valid supervised target count. O5c used constant ordinary-pass CE plus
one trainable feedback-pass CE; doubling ordinary CE here would introduce an
extra gradient-scaling difference. Report ordinary CE separately from O5c's
summed objective and compare evaluation NLLs of individual inference paths.

Reset AdamW: native LR1e-5 (the previous native-backbone rate), linear50-update
warmup, betas(0.9,0.95), epsilon1e-8, weight decay0.1, clip norm1, seed20260922.
O5c fusion LR was1e-4. Preserve this difference rather than copying a rate chosen
for two new matrices into the pretrained backbone.

## Exactly shared data and exposure

Reuse authoritative `.runtime/olmo1b-step60000/o5c-data-02/prepared/` unchanged,
manifest SHA256
`f837f7f412dee17a304df5f78f4b65571f15163af440adecfc77b88cca8b1490`.
Use its **mixed** plan, including every segment, original document, target,
context boundary, row order and update boundary. No new data or cycling.

512 updates ×8192 supervised CE targets =4,194,304 targets; each update has
4096 code and4096 general-text targets. Total2,097,152 targets per domain,
4,208,250 valid input tokens and13,946 segments. Maximum length512; physical
chunks at most16 with a variable final chunk. The existing all-target adjacent
same-document mask, genuine-end EOS and independent document contexts remain
unchanged. Full counts and plan identities are checked on startup and resume.

The control matches data, context, supervised exposure and update count, not
trainable capacity or compute. Native training has about140 times as many
trainable parameters as fusion adaptation; it trains different computational
paths at a different LR. This is a practical ordinary-continuation reference,
not a pure architecture or compute-matched test of feedback.

## Preflight, evaluation and checkpoints

Before freezing the resolved configuration, run actual H100 zero-LR steps
(two warmups, three timed) on the first mixed update and the largest-row update
in the fixed plan. Include real AdamW state, backward and clipping. Bound peak
allocated memory to60GiB. One disposable nonzero step must update native weights
while preserving the unused fusion state; restore native tensors exactly, discard
optimizer state and reproduce initial512-window ordinary evaluation. Reuse the
existing CE/evaluator contracts and focused CPU ownership/save-resume tests;
no broad new precision campaign or redundant full-GPU optimizer replay.

Evaluate the same first128 code/WikiText development windows at0,50 and every64
updates, with the final512-window evaluation using original-document records.
Evaluation batch8. Reserved tests remain unscored. Compare final ordinary NLL,
accuracy and perplexity against the shared starting ordinary pass, O5c mixed K2
and O5d mixed exact online. Compare learning curves only on the128-window subset;
do not invent an online learning curve from an endpoint evaluation. Native cached
ordinary fidelity is already established; this run does not repeat the expensive
FBT sequential evaluation or claim a new generation benchmark.

Save complete model/optimizer/scheduler/RNG/counters/data-cursor checkpoints
at128/256/384/512, after600seconds if sooner, and at requested stops. Upload and
verify GCS generation, size, MD5 and SHA before deleting an older local checkpoint.
Keep enough disk space for current and successor full checkpoints; expect about
14GB each. No unrelated files are deleted. W&B logging uses the user's existing
project. Tell the user if the measured runtime is likely to extend to hours.

Stop on nonfinite losses/gradients/state or unused fusion-state mutation. Pause
for diagnosis if either domain's NLL exceeds its initial ordinary value by more
than1.5nats on two consecutive scheduled evaluations. A modest tradeoff is a
result, not a trigger to retune during this fixed run. Resume only the latest
recorded boundary with identical sources/configuration/runtime/data and retry
an interrupted upload before further training. Save current state at safe stops.

## Interpretation and review boundary

Use paired original-document bootstrap differences on the shared final targets,
1000 resamples, seed20260922. Report single-seed/development-only/narrow-WikiText
qualifications and adaptation-capacity/LR/compute differences. If ordinary
continuation obtains comparable or better general-text results, the frozen
ordinary reference from O5c/O5d overstated what could be attributed to feedback.
If FBT retains an advantage, it motivates a more tightly matched follow-up rather
than establishing general efficacy by itself. No automatic budget extension or
RT/NextLat experiment follows this control; assess it at the completed boundary.
