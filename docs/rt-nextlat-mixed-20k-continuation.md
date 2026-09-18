# Original mixed A5/Fuzzy continuation to 20k

Authorized 2026-09-17: save and stop the warm-A5-checkpoint experiment, then
resume the original fresh mixed run to test whether A5 emerges with more
optimizer updates. Preserve batch 128 per task and the original equal-weight
objective, data order, model, optimizer and FP32 runtime.

**Stopped and retained:** after resuming at 18:04 UTC on 2026-09-17,
the run stopped cleanly at **19,810 total updates** before the user's next
throughput comparison. The user had authorized saving/stopping this run when
moving to the larger-batch direction. Supervisor PID **67101** completed
reporting and verified GCS retention at 21:45 UTC. This is 9,810 additional
updates from the restored 10k parent, slightly short of the original 20k target.
[Continuation W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/atuxgi4p).
Full same-checkpoint development results: A5 L12 whole-word **98.6279%**,
A5 L36 whole-word **57.6582%** (102,400 words each), Fuzzy answer **98.8509%**
and all-answers sequence **74.2188%** (1,280 examples). These joint results
demonstrate observed length-36 success after the original 10k gate failure;
they are one-seed development evidence. See the
[authoritative extension report](reports/rt-nextlat-fuzzy-a5/d128-fresh-mixed-20k-extension/report.md).

Checkpoint: `train-mixed-continue20k/checkpoints/step-019810.pt` in the runtime
below, SHA256 `82c29baee78e4562f010750bc87d70b659d4b42f1e49680ea40076e4e001ec3d`.
`retention/final-receipt.json` verifies its GCS archive. The separate corrected
report closeout also completed and retained its evidence.

CPU readiness
validation passed for the entire committed 0→8k→10k parent history, approved
checkpoint hash, original controls, and unchanged source/CLI settings.

The earlier waiting worker, PID 59404, correctly cancelled when the user
stopped the curriculum. Its executable, configuration and readiness receipt
remain unchanged. The versioned `handoff-v2.py` records the new authorization
and verifies the actual stopped curriculum checkpoint, report and GCS receipt
before launching the same original mixed continuation command.

The fresh mixed 10k result does not establish that state tracking cannot
emerge later. The matched D128 A5-only control improved near 8–9k, so the
original mixed endpoint supplied only a short budget beyond that transition.

## Parent and scope

Resume
`.runtime/rt-nextlat-fuzzy-a5/20260916T182200Z-d128-a5-mixed/train-mixed-resume-8000/checkpoints/step-010000.pt`.
SHA256:
`5c448d02903713139330f88c06b9beecb4464e3d44131a5d8835b1d0a589c69e`.

Restore all model, Adam, RNG and both data streams. Execute 10,000 additional
updates to reach **20,000 total**, with 2.56 million example presentations
per task. This is a continuation of the fresh equal-weight mixed model,
not the A5-warm-start model and not an additional curriculum stage.

Use the frozen `scripts/rt_nextlat_a5_fuzzy_train.py` and existing strict
resume/reporting code. A different serialization hash at the restored
boundary requires an exact full-packet equality receipt accepted by the
existing reporter. Preserve and join the original committed 1–8k history,
the recovered 8k–10k suffix and this 10k–20k extension exactly once.

Evaluate every 500 updates. Retain checkpoints at 11k, 12k, 15k and 20k;
full A5 development evaluations at 12k, 15k and 20k use 102,400 words per
role, with the existing 1,280 Fuzzy development examples. No final
confirmation evaluation or autonomous NextLat rollout.

The original A5-only 10k and Fuzzy-only 6,521 controls remain valid at their
shared checkpoints. Do not label them matched 20k controls or extend their
curves beyond observation. The separate warm-start A5-only continuation
can supply additional context, with its own stage contract and report.

## Execution order and retention

Execution lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260917T153000Z-d128-fresh-mixed-20k/`

The preceding warm-start curriculum was stopped and retained at phase 6,288 /
global 16,288 before this run began. `prerequisite-completion-v2.json` binds
that endpoint to the user authorization and verified GCS archive. Read
`status.json` for the active phase and PID. The original cancelled worker's
records are preserved as `v1-cancelled-status.json` and
`v1-cancelled-supervisor.pid`. `STOP` requests a checkpointed training stop;
`STOP_QUEUE` prevents a queued launch. No further experiments are queued.

After training, the supervisor verifies exact restored packet identity at
the 10k boundary, assembles the accepted full history, reports results, and
retains evidence in GCS. The original 10k A5 screening gate remains historical;
the continuation report must separately assess emergence after 10k and at the
actual endpoint, without treating the earlier gate failure as a 20k result.

The already running supervisor has the frozen legacy reporter loaded in its
execution plan. A separate CPU-only reporting closeout was armed as PID
**75648** and has completed,
under
`.runtime/rt-nextlat-fuzzy-a5/20260917T181000Z-d128-fresh-mixed-extension-closeout/`.
After the legacy report and its verified retention finish, this closeout will
write the authoritative interpretation to
`docs/reports/rt-nextlat-fuzzy-a5/d128-fresh-mixed-20k-extension/`, preserving
the historical gate and separately reporting observations after 10k. It leaves
the training process and original report bytes unchanged. Its separate GCS
archive references the training archive's verified receipt and does not embed
that archive recursively. Read its own `status.json` for reporting completion;
the terminal phase is `authoritative_report_complete_and_retained`.
Five focused CPU fixtures and the wrapper CLI check passed. The wrapper is
`scripts/rt_nextlat_a5_fuzzy_extension_report.py`; the closeout's `ready.json`
binds its code, dependencies, fixtures and driver configuration to hashes.

Execute all training in the required Docker container. Log online under
`taylorbollman/rt-nextlat-fuzzy-a5`. Retain checkpoints and reports in the
persistent project tree and under the corresponding new
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/` prefix.

## Historical batch-size comparison

The older successful two-layer first-window RT + NextLat in W&B project
`rt-a5-state-tracking` used D512/H8/FFN2048, 7,407,104 training parameters,
batch 1,024 and length 12. Its first recorded positive E36 was 46.7471% at
1k updates (1.024 million presentations), versus D128/B128's first recorded
3.3350% at 9k (1.152 million). Width, head geometry, vocabulary and data
order also changed, so exposure alignment is not a controlled attribution.

Mean training seconds/update: historical D512 A5 0.0560, current D128 A5
0.0510, current D128 Fuzzy-only 1.3251, current D128 mixed 1.3899. Accepted
mixed updates 1–10k consumed 3.861 training hours, excluding the discarded
interruption suffix, evaluations and downtime. All these runs used full
FP32 eager execution without compilation or CUDA graphs.

Each mixed update separately executes A5 `[128, 12]` and Fuzzy `[128, 400]`,
accumulates half-weighted task-mean gradients, then clips and updates Adam
once. There is no cross-task padding or concatenation. Native MAD left-pad
symbols, aligned dense training labels and masked answer evaluation remain
unchanged. Token IDs occupy disjoint ranges in one shared table, with
task-local output distributions. The longer Fuzzy recurrent computation,
rather than padding A5 to length 400, accounts for most of the extra runtime.
