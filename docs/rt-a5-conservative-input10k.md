> **Superseded on2026-09-15:** the user stopped this run at9,665 updates and
> authorized the uninjected control unconditionally. The old conditional coordinator
> was terminated. See [the current handoff](rt-a5-control-linear10k.md).

# Conservative input-bypass ramp and conditional four-layer control

The user requested a fresh four-layer input-injection diagnostic with
`lambda(s) = 0.001 * min(s / 50000, 1)^2`, initially training for **10,000 updates**.
The warmup remains50k; lambda is0.00004 at the10k pilot endpoint. Any extension
requires new user instruction. Lambda is scheduled, not learned; P_e remains
learned. This ramp is50 times smaller than the previous ramp at every update.

The user subsequently authorized a second diagnostic **if the full length36
whole-word exactness is exactly zero at10k**: a fresh10k four-layer L1R RT +
NextLat control with no embedding injection or embedding projection. Its first
layer (index0) has window2; the other three layers use full recurrent attention.
Do not require more user approval for that conditional control. Do not run it
if the conservative10k evaluation has positive E36, and do not substitute a
routine4096-row evaluation or a checkpoint before10k for this condition.
No value-bypass diagnostic is queued.

## Active conservative run

Lineage: `.runtime/rt-a5/20260915T163300Z-input-conservative10k/`.
Training directory `train-conservative/`; coordinator PID102248, Docker launcher102252.
Started16:36:50UTC; W&B [h5m6ufcn](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/h5m6ufcn).
Checkpoint retention watcher PID103083 is active. Read
`launch-status.json`, `training.log`, the atomic training report, and the last
complete history row before acting after an interruption.

Same four-layer architecture as the prior input experiments: index0 window2,
indices1/2/3 full RT, input injection only at index1:
`x_t^(2) = u_t + lambda(s) P_e e_t`. The raw token embedding is attached; the
existing index1 normalization follows the addition. Width512,8 heads,GELU FFN2048,
ALiBi, Mitchell initialization at depth4,13,964,800 parameters in44 tensors.
Fresh seeds1234 backbone/order,1235 predictor,1236 projection. The full initial
model tensor digest equals the earlier fixed/ramped input experiments:
`d9c659b2161eb45e81e648666cc42a48fa110b76f836e2a2ecd5008ca7d60f56`.

Original NextLat CE plus weight-one SmoothL1, detached target only; no autonomous
latent rollout. Original AdamW1e-4, betas(.9,.95), eps1e-8, matrix decay.01/vector0,
clip1, foreach/fusedfalse. Full FP32; noTF32/autocast/compile/graphs. B1024,T12,
same800k training words and data order. Dataset:
`.runtime/rt-a5/20260911T154748Z/data`, manifest SHA
`944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.

Routine dev/ood_dev evaluations use4096 rows every500 updates. Saved0/1k/5k/10k;
trained checkpoints use full102400-row evaluations. E(t) means all states through
t correct, A(t) accuracy at state t, M(t) mean token accuracy through t. Boundary
and full curves must use the same length36 words. Final confirmation is untouched.

Source65 SHA `7e9da2bf91ee61991ee3b05d0f81287db13d72199fa74898c8a1dde78413804d`.
Protocol SHA `80e9551936f4fa646e1c3b4c179a81b9b934c97e898bd3a5521908d0a461a426`.
The previous fixed62 and previous quadratic65 source files remain unchanged.
New files: `scripts/rt_a5_conservative_input.py`, its `_train.py` driver and
`configs/rt_a5_conservative_input/base.json`. Do not edit frozen training sources.

Nine focused CPU tests and independent review passed. They check schedule anchors,
exact common initial values, active first-update projection gradients at lambda
4e-13, and exact model/Adam/RNG/schedule resume. Five discarded actualB1024/T12
GPU updates passed finite44-tensor/Adam checks; T36 forward was finite/causal at
maximum lambda.001. No repeated mixed-precision or compiler study.

W&B project `taylorbollman/rt-a5-state-tracking`, group equals the lineage name;
lambda is logged under `train/injection_coefficient`. Checkpoints/evidence go to
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T163300Z-input-conservative10k/`.
A user-requested early stop can use `STOP_AFTER_UPDATE`, but the normal endpoint
is10k. Do not create that stop file without a user stop instruction.

## Conditional control

Prepared lineage: `.runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k/`.
Use the existing `scripts.rt_a5_l1r_depth_train` driver with `--n-layers 4`, same
width/seeds/data/order/NextLat/optimizer and10k budget. It has13,702,656 parameters
in43 tensors: the projection is removed. The remaining initial tensors should
be identical to the conservative model. No new positional, normalization or
attention change is intended. Its GPU work must wait for the conservative10k
trainer to exit and for the exact full-evaluation gate above to pass.

Root sequencing is running in `run_diagnostics.py`, PID112526, in the conservative
lineage. `diagnostics-status.json` records each stage. It waits for conservative
completion, runs its CPU state check/report, records `control-trigger.json`, and
only on the exact full10k E36zero condition runs the control GPU preflight and
launch. It then runs the control CPU state check and paired report. No automatic
extension. The retainer is authorized to bind/start the control watcher once its
actual gated protocol and training launch exist. CPU pairing of all43 shared
tensors and original optimizer groups/options has passed; helper/sequence/report
reviews found no functional blocker. All82 coordinator-bound files are frozen. Check current run statuses before launching:
never overlap the two GPU trainers or automatically extend either beyond10k.
Both are one-seed diagnostics on repeatedly inspected development pools.

## Previous ramp stopped

The previous lambda.05/50k run stopped cleanly at13,181 updates on user steering,
with its exact model/Adam/RNG/order state saved. Its saved-state check and terminal
report passed; the auditor viewed all four terminal figures without findings.
E36 remained0/102400, E13=79.586%, E14=24.101%, M36=37.361%.
Its state/report/retention/archive/readback closure is complete and must not be disturbed:
`.runtime/rt-a5/20260915T160000Z-input-quadratic50k/`.

The previous ramp archive has300 verified members. An external terminal visual-review
companion records the auditor's inspection without changing archived files.
