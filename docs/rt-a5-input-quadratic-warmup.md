# Four-layer input bypass with quadratic lambda warmup

**Terminal update:** training stopped at 13,181 updates, lambda=0.0034747752. Saved-state validation, checkpoint retention and the terminal report passed. [Terminal report](reports/rt-a5/input-quadratic50k/README.md). Terminal plots await visual review; no subsequent GPU run is queued. Archive/readback status is in the lineage's finalize-status.json.

## Current run

User authorized a fresh diagnostic up to50,000 updates, with the option to stop
early, before returning to the value-bypass approach. No subsequent GPU run is
queued. The previous fixed-input run finished20k; its saved-state checks and
reports are complete. The briefly started value run was stopped by user request:
1593 updates were observed in history, but only0/1000 weights were saved.
See [the fixed-coefficient handoff](rt-a5-embedding-injection-diagnostics.md).

New lineage: `.runtime/rt-a5/20260915T160000Z-input-quadratic50k/`.
Training directory: `train-quadratic/`; coordinator PID79637, Docker launcher79638.
Started16:08:15UTC. W&B [db40gut3](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/db40gut3). Read `launch-status.json`,
`training.log`, `train-quadratic/report.json` and the last complete history row
before acting after an interruption. A report marked running may only contain
the latest saved checkpoint counter; history records every completed update.

The full5k checkpoint evaluation is complete: length12 whole-word exactness
97.7607%; length36 E36=0/102400, M36=36.3281%. Lambda=.0005.
[Partial5k figures](reports/rt-a5/input-quadratic50k-through5k/README.md) are
synced to W&B`tly43tdj`; root viewed all four, with readable axes/legends and
consistent full/boundary prefix rows. This is an early snapshot, not the final
50k result. Original state/latent losses are finite.

Checkpoint retention watcher PID89622 uploads newly saved states. The CPU-only
endpoint checker/reporter and terminal evidence archiver are running under
`finalize.py` PID95395, currently waiting for training. Independent full-chain
review passed. They execute automatically after a normal50k finish or graceful
user stop, then verify archived evidence; no further GPU experiment is launched.
`finalize-status.json` records progress/failures. The terminal plots still need
visual review after generation; the earlier partial5k review does not claim that.
A terminal status note will be added to this handoff before archiving.

At10k, full102400-row evaluation: length12 whole-word exactness99.6367%,
length36 E36=0/102400, M36=37.0038%; E13/E14/E16=75.4287%/15.0850%/0.0889%.
Lambda=.002. Training continues under the50k cap; this is not an endpoint stop.
The first waiting finalizer94041 was replaced before it executed any endpoint
work to include an extensionless stop-request file in a future evidence archive.
The trainer, source65 and checkpoint-retention watcher were unaffected.

## Model and schedule

Four total RT layers: window2 at index0, followed by three full RT layers.
Only index1 receives the input injection:

`x_t^(2) = u_t + lambda(s) P_e e_t`

`lambda(s) = 0.05 * min(max(s,0) / 50000, 1)^2`

Here `e_t` is the attached original raw token embedding and `P_e` is the learned,
bias-free512×512 projection. Lambda is **scheduled, not learned**. Existing
index1 normalization follows the addition. All other layers are unchanged.
Initial checkpoint0 has lambda0; before update1 it is2e-11. The saved/evaluated
coefficient at completed update s is the one used for that update:

| Update | Lambda |
| --- | --- |
| 0 | 0 |
| 1000 | 0.00002 |
| 10000 | 0.002 |
| 20000 | 0.008 |
| 30000 | 0.018 |
| 40000 | 0.032 |
| 50000 | 0.05 |

D512/H8/GELU FFN2048, ALiBi, Mitchell initialization at actual four-layer depth.
13,964,800 parameters in44 tensors. Seeds1234 backbone/data order,1235 predictor,
1236 projection. The complete initial tensor digest matches the fixed-input run:
`d9c659b2161eb45e81e648666cc42a48fa110b76f836e2a2ecd5008ca7d60f56`.
Fresh weights/optimizer; no continuation from the fixed-coefficient run.

Original NextLat CE plus weight-one SmoothL1 objective; target latent only is
detached, source hidden and conditioning embedding remain attached. No autonomous
latent predictor rollout. FP32, noTF32/autocast/compile/graphs. AdamW1e-4,
betas(.9,.95), eps1e-8, matrix decay.01/vector0, clip1, foreach/fusedfalse.
B1024, training length12, same800k-word train split and order.
Dataset `.runtime/rt-a5/20260911T154748Z/data`, manifest SHA
`944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.
This changes the input coefficient only, not the learning rate or NextLat weight.

## Checks, tracking and retention

Twelve focused CPU tests passed, including exact initialization, active first-update
projection gradients, exact model/Adam/RNG/schedule resume, off-boundary graceful
stop, scheduled-boundary stop and normal completion. Independent review passed.
Five discarded B1024/T12 updates on the H100 passed finite FP32 model/gradient/Adam
checks for44 tensors, active embedding/projection gradients, and T36 causal
forward behavior at lambda.05. Warm updates take about.104s, excluding evaluation.
No new precision campaign; the recurrent kernels are unchanged.

New factory, driver and config add three files to the historical62-source manifest.
Source65 SHA: `aea19cef549b65563fc9816ce68128520ca5f3a5eb3b829c0688753935b5c942`.
Protocol SHA: `0c2d1d6bd51593a94698379f03b6d6b6ce60d1a9932f8449918ab85a575ae2d3`.
Never edit those frozen training sources while running. Future fixes need new
source-bound lineage; postprocessing helpers stay outside that manifest.

W&B project `taylorbollman/rt-a5-state-tracking`; group is the lineage name.
Lambda is logged directly as `train/injection_coefficient` every25 updates and
`schedule/injection_coefficient` at evaluation, alongside the usual task metrics.
Routine dev/ood_dev evaluation uses4096 rows every500 updates. Checkpoints0/1k,
then every5k through50k; trained checkpoints get full102400-row evaluations.
E(t) is cumulative prefix exactness, A(t) exact accuracy at position t, and M(t)
mean token accuracy through t. Boundary/full figures must use the same36-word
rows. Confirmation remains unevaluated; this is one seed on reused development.
An uninjected four-layer control has not been run; two/six-layer comparisons
also change depth and initialization.

Retain checkpoints and final evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T160000Z-input-quadratic50k/`.

## Stopping and continuation

Only when the user requests a stop, create the lineage's `STOP_AFTER_UPDATE`
file with the request text. The driver checks it after each completed update,
forces full dev/ood evaluation, saves the current weights/Adam/RNG/order/schedule,
and closes W&B. It records status `stopped`, actual completed update and nominal
endpoint50k separately. If a request arrives during evaluation, polling occurs
after the next completed update. Do not hard-kill the trainer for a routine stop.

A future resume must use a fresh output directory and the saved checkpoint;
the source/contract must match, and the global lambda schedule must not restart.
Use a new absent stop-file path. No resume or value follow-up is automatic.
At endpoint: check saved state, verify GCS retention, generate plots from existing
evaluations, visually review them, and archive final source/runtime/report evidence.
