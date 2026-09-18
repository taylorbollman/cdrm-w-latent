# First-window RT without NextLat: matched 80k control

**Training and reporting are complete.** The control stopped at exactly80k
on2026-09-15 at01:25:59 UTC and synced to W&B. No GPU training remains running.
Read the [outcome](reports/rt-a5/window-first-nextlat-ablation-80k/outcome.md),
[whole-word learning curves](reports/rt-a5/window-first-nextlat-ablation-80k/whole-word-vs-updates.pdf),
and [matched boundary plot](reports/rt-a5/window-first-nextlat-ablation-80k/length-boundary.pdf).
Pause here for the user's next experiment; do not extend automatically.

| Training at80k | Length12 whole word | Length36 whole word | Length36 token accuracy |
| --- | ---: | ---: | ---: |
| First-window RT, CE only | 99.9570% | 0.0000% | 37.3472% |
| Matched first-window RT + NextLat | 100.0000% | 100.0000% | 100.0000% |

Each full evaluation uses102400 development words per role. CE-only E(36)
was zero at every retained checkpoint; NextLat's first retained checkpoint
with no observed length36 errors was30k. At the control's80k endpoint,
E(13)=77.4326%, E(14)=18.5957%, E(16)=0.1240%, and A(36)=1.6748%.
This supports a substantial NextLat-training benefit in this paired run,
with the one-seed/reused-development qualifications described below.

The final saved-state audit passed41/41 checks; initialization passed32/32.
Every one of80000 minibatch-order hashes matches the reference, all21
backbone tensors began identically, and all final model/Adam tensors are
finite FP32 with Adam counters80000. The final checkpoint is
`train-control/checkpoints/step-080000.pt`, bytes76346961, SHA256
`9b1837648b0225b3a834121ee8b5f9bc38b36da9e62e53528a3ce08b89929cab`.
Training took4321.84 seconds of updates and4444.81 seconds elapsed.

The independent final report audit passed:1056 metric rows,44 checkpoint
evaluations,24 checkpoint files across the two arms,113 source snapshots,
1600 training bins, and43 reporter artifacts. All five PNG figures were
visually inspected with readable axes and consistent full/boundary curves.
`evidence-audit.json` also binds the generated `outcome.md` supplement.
The first auditor's exact file-inventory check encountered that new supplement;
its original source/log are preserved. `audit_report_with_outcome.py` explicitly
allows only `outcome.md`, verifies every displayed metric table and its report
hash, and retains all required reporter artifact checks. No training or frozen
reporter source changed. `root-final-review.json` records final closure checks.

All12 checkpoints are retained under the GCS prefix below. Final report
[W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/8odwv1dz)
is synced; report JSON SHA256 is
`0ab04c76ee597051e7f41dec1237e9cffaa3b3dbde7d6aa9cbe68bdebd418695`.
The final archive/readback receipts are `evidence-storage.json`,
`evidence-readback.json`, and `evidence-readback-upload.json`; inspect them
for retained object hashes/generations before any future recovery.

The user authorized repeating the successful two-layer RT with window 2 in
the first layer and full recurrent attention in the second, **without
NextLat**, for the same 80,000-update budget. This is a fresh run from the
same backbone initialization, not a continuation of trained NextLat weights.
The lineage is `.runtime/rt-a5/20260915T000224Z-window-first-no-nextlat80k/`;
the pointer is `.runtime/rt-a5/current-window-control-lineage.txt`.

Training launched on 2026-09-15 at 00:11:46 UTC, coordinator PID115743,
launcher PID115745. Live graphs:
[W&B CE-only control](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/fs6vqqtd).
The retention watcher is PID116478. Confirm process/container identity from
the lineage records before recovery; do not start a duplicate after compaction.

The saved training initialization passed all32 audit checks, including exact
equality of every backbone tensor and matching optimizer groups/data state.
See `initial-state-validation.json`. The trainer container is recorded in
`audit-container.json`; its ID begins `11efaab4c108`.

The finalizer started at00:23:39 UTC as PID124176. It waits for a successful,
synced80k training exit, runs the final saved-state audit in a CPU container,
and then generates the comparison report. Its17 frozen dependencies are in
`finalizer-source-freeze.json` (SHA256
`2311a0fd6227819599eb3dbf3d1d3dae00b103b6eb6c9af3fc83919c0da4b660`).
Inspect `finalizer-status.json`; do not launch a second copy. The reporter
passed25 CPU tests and independent review with no findings. After it finishes,
run `audit_report.py` against the completed report, inspect the plots, and
finish the outcome/handoff before archival.

`protocol.json` SHA256 is
`eb8c0833d10344c8f0f90cd033942141f0d6e6733bd738d1fc27702d0dc38549`;
the frozen58-source digest is
`2c72764fc37343df73cd2fc1e320729be72adb6bd93881666cc1f3754a2a5e73`.
All58 implementation/configuration files must remain unchanged through this
run and its report. New reporting/validation helpers are outside that set.

Twelve focused CPU tests passed, including exact D512 initialization,
predictor removal, CE gradients, and model/Adam/RNG resume. Bounded GPU
preflight passed at T12/T36, with all actual initial backbone tensors equal
to the saved reference and ten finite discarded B1024 updates. The preflight
is [W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/6ehissba),
recorded in `preflight/report.json`. It took roughly53ms per update in the
short preflight, using under0.8GB allocated memory. These are operating
checks, not performance evidence for this training run.

The only training objective is the existing A5 same-position state
cross-entropy, averaged over batch and all 12 positions. The NextLat
predictor is absent from the registered model and optimizer. The original
pure-CE update, optimizer, data-order and evaluator functions are reused.
There are no latent diagnostics or autonomous latent rollout. All 21
backbone parameter tensors, totaling 6,357,504 parameters, start identically
to the preceding winning NextLat run's backbone. The reference has four
additional predictor tensors and 7,407,104 total parameters. Global clipping
retains the same rule at norm 1, applied to each model's active parameters.

Both RT blocks retain their original Mitchell initialization, D512/H8,
GELU FFN2048, LayerNorm and learned full-width Q/K normalization, ALiBi,
untied V60 embedding/head, and no biases or dropout. The first block reads
temporary self K/V and the immediately preceding output's permanent K/V.
The second has full-prefix RT attention. States and gradients remain attached;
the short direct attention window does not truncate represented history.

Training uses batch 1024, length 12, 800,000 unique words, seed/order 1234,
and constant AdamW1e-4 with betas(.9,.95), eps1e-8, matrix decay.01/vector0.
The frozen corpus is `.runtime/rt-a5/20260911T154748Z/data/`, manifest SHA256
`944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.
All parameters, gradients, attention and Adam state use FP32; TF32,
autocast, compile and CUDA graphs are off. Do not add precision studies.

The endpoint is prospectively fixed at 80k. Checkpoints are 0, 1k, 5k, 10k,
20k, 25k, 30k, 40k, 50k, 60k, 70k and 80k. Development evaluation every500
updates uses4096 words; checkpoint evaluation uses102400 words per role.
E(t) means all states through t correct, A(t) only state t correct, and M(t)
mean token accuracy through t. Full and boundary curves must use the same
length-36 words. The length-12 development role is separate. Final
confirmation remains unevaluated. Do not stop early or extend the budget
without another user instruction.

The matched reference is the completed first-window RT+NextLat run at
`.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first/`,
[W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/m2klbr0d).
It has zero observed development errors at both lengths at every retained
checkpoint from30k through80k. Both reference and control endpoints were
fixed before their training, but this follow-up choice uses development
results. This is one seed on reused development data, not final confirmation.

The new model/config/driver are `scripts/rt_a5_window_control.py`,
`configs/rt_a5_window_control/base.json` and
`scripts/rt_a5_window_control_train.py`. The58-source manifest extends the
unchanged55-source preceding run with those three files. The bounded
validator is outside the training manifest: it compares CE-only logits and
gradients through the old wrapper with the bare control at T12/T36, checks
all actualD512 initial tensors against the saved reference checkpoint, and
runs ten discarded B1024 updates plus a T36 causality check. Use script-form
`python scripts/rt_a5_window_control_validate.py` for its legacy helper imports.

All GPU work uses the project Docker launcher, verifying container cwd and
`nvidia-smi` inside it. CPU tests/audits use `CDRM_DOCKER_GPUS=none`.
The prospective protocol is resolved by `prepare_protocol.py` after preflight.
`launch.py` refuses existing output, binds source58 and protocol, launches
exactly one fresh control, writes `training-exit-code.txt`, and does not retry
or silently resume. Inspect `launch-status.json`, `training.log` and
`train-control/report.json` after an interruption before taking action.
The host-only `status.py` reads progress without loading model state.

The report destination is `docs/reports/rt-a5/window-first-nextlat-ablation-80k/`.
It will compare only the matched CE-only control and first-window NextLat
reference, including E36 versus updates and both token and whole-word accuracy.
Checkpoint, source, initialization and data-order audits must precede closure.
Keep all closed prior lineages unchanged.

Retain checkpoints and final evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T000224Z-window-first-no-nextlat80k/`.
Run final archive/readback only after training, report, audits, outcome and
handoff notes are stable. Logs for final archival belong outside this lineage;
do not mutate archived members afterward.
