# Mixed A5/Fuzzy learning-rate acceleration pilot

**Completed at 5,000 updates; checkpoint and comparison retained on GCS.**
The higher-rate run reached **83.4307% A5 length-36 whole-word accuracy** and
**99.5960% Fuzzy answer accuracy**. The completed
[results and stability assessment](rt-nextlat-mixed-lr3e4-assessment.md)
recommends keeping 3e-4 for the proposed three-layer baseline. That experiment
has not been started.

Started **2026-09-20 at 18:55:37 UTC**, supervised process **12329**.
Live run: [l1r-rt2-nextlat-d128-b2560-mixed-lr3e-4-warmup100-5000](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/y0mjyvyb).
This document records the prospective protocol and startup checks; terminal
results belong in the report destination below.

Authorized 2026-09-20: repeat the original uninjected D128 mixed model from
fresh, matched initialization for **5,000 optimizer updates**, in FP32, with a
higher learning rate. The purpose is to learn sooner. The original constant
1e-4 run **did succeed**: A5 length-36 whole-word accuracy first became positive
at the full 10,400-update observation and reached 92.8535% at 15k, alongside
99.9112% Fuzzy answer accuracy. Its lack of length-36 whole words at 5k was an
early observation, not an architecture failure.

## Fixed recipe

- Two tiled RT blocks, D128, 16 heads, FFN512, ALiBi and Mitchell initialization;
  block zero has window two and block one full recurrent attention.
- 479,616 parameters, NextLat weight one, no embedding bypass or latent rollout.
- Physical batch **2,560 A5 plus 2,560 Fuzzy examples per optimizer update**.
  Separate task means receive equal weights; A5 is not padded to Fuzzy length.
- Same A5 T12/Fuzzy T400 training data, task vocabularies, initialization and
  independently shuffled task streams as the original large-batch baseline.
- FP32 eager execution; TF32, autocast, compilation and CUDA graphs off.
- AdamW betas (0.9, 0.95), epsilon 1e-8, matrix decay 0.01, vector decay zero,
  global gradient clipping at one. Those settings and the objective are unchanged.

The only optimization change is learning rate. Update **1 uses 1e-4**;
updates 1–100 interpolate linearly to **3e-4 at update 100**; subsequent
updates remain at 3e-4. Step zero stores the initial 1e-4 rate. This schedule
does not depend on the stopping budget, so a later authorized continuation
can preserve it. The present authorization stops at 5k.

Routine evaluation remains every 100 updates on 4,096 A5 examples per length
and all 1,280 Fuzzy development examples. Full 102,400-word A5 evaluations
are scheduled at 1k, 2.5k and 5k, with the existing full recheck on first
positive length-36 monitoring accuracy. Checkpoints include zero, 100, 500,
1k, 2.5k, 5k and the original evaluator's checkpoints at routine evaluations.
An explicit user stop saves and fully evaluates the stopping state.

W&B logs actual learning rate, clipping fraction, task-specific CE/NextLat
losses and the existing A5 prefix/whole-word and Fuzzy answer/first-value
metrics. Compare equal-update and training-time curves against the original
run. At its 5k checkpoint the original had 0% A5 whole-word accuracy at both
lengths, 90.5379% Fuzzy answer accuracy, and 82.5510% Fuzzy first-value accuracy.
Preserve the original 15k trajectory as separately labeled longer-budget
context. A larger learning rate does not imply a proportional speedup.

## Execution and retention

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260920T184640Z-d128-b2560-lr3e4-5000/`.
The prospective `execution-config.json` specifies every argument; `ready.json`
freezes executed sources after bounded schedule/checkpoint checks.
`status.json` is the supervisor state; `train/report.json` and
`train/history.jsonl` are the authoritative training progress.

New trainer: `scripts/rt_nextlat_a5_fuzzy_lr_train.py`; the historical trainer
and model files remain unchanged. The new source/checkpoint identity explicitly
records the warmup, peak rate and current schedule position. Resume validation
must preserve both Adam moments and the learning rate appropriate to the
completed global update, including checkpoints inside warmup.

Supervisor: `scripts/rt_nextlat_a5_fuzzy_lr_queue.py`. After the run stops at 5k
or an explicit user stop, it generates a CPU comparison report and verifies
checkpoint/report retention under
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260920T184640Z-d128-b2560-lr3e4-5000/`.
The runtime `STOP` requests a clean stop. No later experiment or extension is
queued. Final confirmation remains unevaluated.

Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-lr3e4-5000/`.
W&B project: [rt-nextlat-fuzzy-a5](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5).

## Startup verification

Twenty-two focused CPU tests passed for schedule endpoints, exact checkpoint
continuation within/across warmup, and invalid checkpoint rejection. Ten
reporter tests and a rendering smoke check passed. The reporter successfully
read the original three-stage 15k lineage. Independent read-only review found
no blocking issues; four supervisor cases covered completion, a clean stop,
prelaunch cancellation, and rejection of an incomplete endpoint.

The actual initial model's 25 saved tensors and initial Adam state match the
original baseline exactly. Through update 32, both task data orders and example
counts match the baseline, all logged losses/gradient norms are finite, and
the applied learning rate follows the approved schedule. The first update
uses exactly 1e-4. Receipts are in the runtime, including `startup-audit.json`.
Early steady training time is approximately 1.88 seconds/update; 5k updates
should take roughly three hours including evaluation. This is an estimate,
not a completed runtime measurement.
