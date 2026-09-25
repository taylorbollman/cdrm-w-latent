# Two-H100 progress and interruption handoff

2026-09-25. Active branch `feat/olmo-two-gpu`, base `25bc29d` (PR29).
User authorized the two-GPU plan and requests progressive checkpoint/evidence
retention. No quality-training campaign is queued. Main plan:
`docs/native-rt-single-to-two-gpu-plan.md`; frozen first-stage protocol is beside
this file. GPU work runs only in the project container.

Completed:

- Two H100 80GB devices, NV18 link, bidirectional peer access. PyTorch
  2.13.0a0+8145d630e8.nv26.06, CUDA13.3, NCCL2.30.5.
- Genuine NCCL sums correct through256MiB. Median25MiB all-reduce0.133ms;
  256MiB0.918ms. These are isolated payload timings, not training overlap claims.
  `.runtime/olmo-two-gpu/nccl-01/report.json`, W&B `cc0xn9sa`.
- Eight tiny FP32 combinations pass genuine NCCL/DDP, three accumulated complete
  updates each. Raw gradients and complete Adam states match canonical one-GPU
  rank-major accumulation within frozen tolerances; replicas match exactly.
  Predictor active only in an earlier rank-zero no_sync microbatch and globally
  inactive on update3 are included. `tiny-eager-01/report.json`, W&B `ngrrexb6`.
- CPU eager runtime scoped64 tests; distributed-checkpoint scoped32 tests.
  Real Gloo workers cover coordinated rejection and exact same-world-size resume.
  CPU counts overlap existing preparation tests; they are not GPU evidence.

Actual ordinary and RT eager B1perrank/T512, two complete updates each, now
pass. Gradient L2 ordinary:1.00e-8/5.57e-9; RT:3.73e-9/6.48e-9. Exact inter-rank
gradient and complete-state replicas; reference state/metrics pass the frozen
budgets. Each global update includes two accumulation microbatches per rank
(four documents total). Combined update1 passes. Update2 passes raw gradients,
loss/counts and exact replicas but fails stricter complete-update elementwise
tolerances:20 parameter/7 moment tensors. See `fixed-state-followup.md`; no
budgets changed. The failed overall `actual-eager-01` report is retained in GCS.

Tiny combined eager recovery now passes24checks: full state, next RNG draws,
cursor, raw gradients and next update reproduce exactly. Four physical updates
per rank/logical endpoint3. W&B `zlmvso6v`, `tiny-recovery-01/report.json`.
Tiny actual-DDP graph correctness is running as `tiny-graph-01`; inspect live
processes/logs before restarting. Root owns GPU execution.

Next: complete actual eager checks, save and reconstruct genuine two-rank
RT/combined checkpoints, then real DDP CUDA graphs and measured scaling.
Prepared graph and recovery harnesses are being developed in isolated worktrees.
Native ZeRO-1 follows validated DDP; ZeRO-2 remains conditional.

Persistence: source commits through recovery harness `29b0cb5` are pushed.
NCCL and tiny eager evidence is verified in GCS under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/`
with stage suffixes `nccl-01` and `tiny-eager-01`. Local receipts are under
`.runtime/olmo-two-gpu/retention/`. Main disk has only about19GiB free;
actual model+Adam checkpoint is13–14GiB.
Retain one new actual checkpoint at a time in GCS, verify before cleaning local
bytes, and preserve failed attempts. Never treat local SSD as durable.
For GCS container calls use `env -u GOOGLE_APPLICATION_CREDENTIALS` to select
working mounted ADC; inherited custom ADC path is stale. Never print secrets.
