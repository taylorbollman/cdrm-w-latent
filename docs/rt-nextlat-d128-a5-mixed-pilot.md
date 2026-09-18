# D128 A5 control and conditional A5/Fuzzy pilot

Current status, reviewed 2026-09-17: both A5-only and mixed training completed
10,000 updates. Mixed reporting and GCS retention completed; the subsequent
batch-throughput profile also completed. Subsequent authorized experiments
are linked below and use separate execution lineages.
The recovery/history sections below describe the completed execution.

Mixed endpoint: A5 L12 token / whole-word accuracy 18.2314% / 0%; A5 L36
7.1777% / 0%. Each whole-word result is 0 / 102,400. Final-state accuracy
is 1.6260% / 1.6387%, near 1/60 chance. Fuzzy answer-token accuracy is
99.4383%, with all-answers sequence exactness 90.1562% on 1,280 examples.
The mixed model started from the original initialization, not trained A5
weights; this is failure to acquire A5 in the tested budget, not demonstrated
forgetting after an A5 warm start. Both arms saw the same 1.28 million A5
presentations and data order, but their objectives differ.

The final checkpoint is `train-mixed-resume-8000/checkpoints/step-010000.pt`,
SHA256 `5c448d02903713139330f88c06b9beecb4464e3d44131a5d8835b1d0a589c69e`.
Its hash and the verified `retention/recovered-final-receipt.json` were
checked during closeout; the five mixed-report PNG figures were reviewed.
See [completed mixed report](reports/rt-nextlat-fuzzy-a5/d128-a5-mixed/mixed/report.md).
The user accepted the smaller-batch A5 warm-start curriculum on 2026-09-17.
Its separate paired continuation is now launched; see
[curriculum protocol and execution handoff](rt-nextlat-a5-warmstart-curriculum.md).
The user subsequently authorized the original fresh mixed model's extension
to 20k after the warm-start experiment; see
[continuation handoff](rt-nextlat-mixed-20k-continuation.md). It preserves
batch 128 per task and the original equal-weight objective.

Historical status at recovery on 2026-09-16: A5 completed 10,000 updates; mixed training
was interrupted after 8,366 valid logged updates and is being resumed from
the last saved checkpoint at 8,000. The endpoint remains 10,000 updates.
Read `active-recovery.json` in the lineage for the current supervisor;
the original `execution-status.json` is stale and preserved as evidence.

The user authorized stopping the near-ceiling Fuzzy run, reporting its saved
endpoint, then training the same model on A5 for 10,000 updates. If the model
achieves greater than zero length-36 whole-word accuracy, proceed directly
to the balanced mixed experiment; no additional approval is needed.

## Completed Fuzzy endpoint

The Fuzzy-only run stopped at **6,521 updates**. Checkpoint
`step-006521.pt`, SHA256
`afeb822e2b0fba636b1e5d95b72d1a3a3606cdfb6a9737f2d982a38bc80d9c6a`.
Development answer-token accuracy 99.8854%, answer-motif exactness 99.8111%,
whole-sequence answer exactness 97.5781%, first-value-token accuracy 99.9542%,
terminal-probe-token accuracy 99.5666%. Results, figures and checkpoint
retention were verified; see [Fuzzy handoff](rt-nextlat-fuzzy-d128-pilot.md).
These are one-seed, teacher-forced development results, not final confirmation.

## Frozen experiment choices

Use the exact existing `configs/rt_nextlat_tasks/fuzzy_d128.json` and
`cdrm/rt_nextlat_tasks.py`: D128, 16 heads of dimension8, GELU FFN512, two
tiled RT blocks, first window2 and second full, ALiBi, Mitchell, NextLat
predictor128, auxiliary weight1. No embedding bypass. Training parameters
479,616 (backbone413,824, predictor65,792). Same shared76-row tables and
task-local60/16 output distributions. Full FP32 eager, TF32/autocast/compile/
CUDA graphs off.

Every arm starts fresh from the original paired initialization: seed1234,
predictor1235, independently appended Fuzzy rows1236. “Same model” means
this architecture and starting initialization, not Fuzzy-to-A5 fine-tuning.

A5-only: 128 length-12 training words per update, existing frozen A5 corpus,
data-order seed5432. Direct length-36 development forwards measure exact
whole-word correctness. Identity0 remains scored; no LM target shift.
The frozen A5 manifest SHA256 is
`944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb`.

Mixed: 128 A5 words plus128 native length-400 Fuzzy examples per update,
separate task microbatches and per-task mean CE plus NextLat losses, equal
half weights, one global clip and Adam step. Independent dataset cursors and
order hashes; Fuzzy order seed2026091604. No padding A5 to400 or concatenating
independent words into a recurrent stream.

Both modes retain constant LR1e-4, AdamW betas(.9,.95), epsilon1e-8, matrix
decay.01, normalization decay0, global clip1. Batch128 gives one-eighth the
A5 examples of historical B1024 pilots at equal update counts; the new
single-task and mixed controls are paired with each other.

## Evaluation and automatic continuation

Train A5-only through10k. Evaluate fixed4096-word development subsets every
500 updates. Full development checks use102,400 words at1k,3k,5k,6521,10k;
a positive subset result can trigger an earlier full check at that same saved
checkpoint. Record integer `whole_word_correct` from actualT36 forwards.

After A5 completes10k, launch a fresh mixed10k run if **any** full102,400-word
L36 development evaluation has at least one completely correct word. Record
the first positive checkpoint, whether positive by3k/5k, and the10k endpoint
separately. A late positive result is not described as early emergence.
If no full check is positive, finish the A5 report and stop without launching
mixed or automatically extending A5. Never substitute token/final-state
accuracy or rounded percentages for the criterion.

Mixed evaluation measures both tasks at the same checkpoint. Save checkpoints
at5k and6521 for comparison with the already retained Fuzzy-only control,
along with the declared10k endpoint. There is no10k Fuzzy-only control;
do not present a10k mixed versus6521 standalone comparison as equal exposure.
Final confirmation remains unevaluated for both tasks.

## Execution and artifacts

New lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260916T182200Z-d128-a5-mixed/`.
Its `protocol.json` records the prospective instructions. Production model,
original Fuzzy trainer and data remain unchanged. New training lives in
`scripts/rt_nextlat_a5_fuzzy_train.py`; new gate/report/validation scripts
are scoped to this follow-up. Reuse completed FP32 kernel checks; validate
the changed task objective, actual A5/mixed shapes, and two-stream resume.

Use the project container for all model execution. W&B entity/project:
`taylorbollman/rt-nextlat-fuzzy-a5`. Retain useful new checkpoints and evidence
under `gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260916T182200Z-d128-a5-mixed/`.
GCS commands use the known command-local ADC override documented in the
Fuzzy handoff; no persistent credential change is needed.

### Validation and unattended execution

The 22 trainer/gate tests and 9 reporting tests passed. Discarded H100 checks
passed for actual batch128 per task and direct A5 length36 evaluation at
batch1024. Separate-task gradient accumulation matched the direct mixed
objective within the existing FP32 tolerance. Fresh-process resume at update1
reproduced uninterrupted update3 exactly for model, optimizer, RNG, both data
cursors and order hashes, using actual A5/Fuzzy corpora. See
`preflight-v2/report.json` and `resume-proof.json` in the new lineage.

The first validation attempt exposed only a tuple/list mismatch when comparing
live initialization metadata with JSON. Actual initial weights matched; the
validator now normalizes JSON values before comparing. The failed attempt
and its disposition are retained. Production model and trainer were unchanged.

The lineage's `execute.py` runs A5 through10k, builds its report and uploads
retained artifacts, checks `mixed-eligibility.json`, and conditionally starts
fresh mixed training through10k. Read `execution-launch.json` for its PID and
`execution-status.json` for the current phase; logs are `train-a5.log` and
`train-mixed.log`. Stop files are `STOP_A5`, `STOP_MIXED`, and `STOP_QUEUE`.
No further GPU experiments are queued. Reports live under
`docs/reports/rt-nextlat-fuzzy-a5/d128-a5-mixed/`.

A5 training W&B: https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/4y1z4tyj.
Supervisor PID at launch: 87698. A reporting CLI ordering bug was caught and
fixed while A5 trained; the added subprocess CLI regression passed. Its
original/fixed source and disposition are retained in `evidence-source/`.
This change does not affect model execution or recorded training results.

Preflight timing was about0.052s/A5 update and1.38s/mixed update, excluding
development evaluations: roughly10–15minutes for A5 and4hours for mixed.

## A5 result and current mixed run

A5 was still at zero length-36 whole-word accuracy on the 6,521-step full
development check. Learning then accelerated near 8–9k updates. The first
positive **full** length-36 check was at **9,000 updates**: 3,415 / 102,400
whole words (3.33496%). It passed the continuation condition, but did not
show positive whole-word accuracy within the first few thousand updates.

At the declared **10,000-update endpoint**, the same checkpoint measured:

| A5 development metric | Length 12 | Length 36 |
| --- | ---: | ---: |
| Mean token accuracy | 97.8502% | 77.0332% |
| Final-state accuracy | 91.6709% | 48.2207% |
| Whole-word accuracy | 88.4736% | 23.6865% |
| Correct whole words / 102,400 | 90,597 | 24,255 |

Checkpoint: `train-a5/checkpoints/step-010000.pt`, SHA256
`0f246e2f39f8a26c8a39fcb336ee1bac4da12990f14bdfaf5676b45ce1364701`.
Report with PDF/PNG curves:
[A5 pilot](reports/rt-nextlat-fuzzy-a5/d128-a5-mixed/a5-only/report.md).
The verified A5 archive is at the lineage's GCS prefix under
`a5/a5-evidence.tar.gz`, SHA256
`8c1ab6d73a2fab1d8f3c9a15b0b81f28380c1f1ab2811a8908bbfa9749c88a5a`;
see `train-a5/retention/a5-receipt.json`. Final confirmation remains unused.

Mixed training W&B:
https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/fxzctush.
Its fresh output directory is `train-mixed/`. It did not inherit trained
A5/Fuzzy weights. The supervisor remains PID 87698. No user approval is
pending; continue the already authorized mixed run to 10k, report both tasks
at the same endpoint, compare the common retained control checkpoints, and
retain the final evidence. Estimated completion is roughly 22:45–23:00 UTC
on 2026-09-16, subject to observed throughput and evaluation overhead.

The actual saved initial checkpoints were checked in the CPU container:
all 25 model tensors (479,616 elements) are identical across Fuzzy-only,
A5-only, and mixed runs. Initial mixed training through update 81 had finite
loss/gradient norms, equal task example counts and objective weights, and
exactly matching per-task data order against both controls. See
`paired-start-audit.json` and `mixed-initial-health.json`. This verifies
startup; it does not assert that the remaining training has completed.

After completion, follow `active-recovery.json` to the recovery status, then inspect the mixed report and
actual PNG figures, and the final GCS receipt before reporting joint results.
Do not describe late A5 emergence as early success, or claim a matched 10k
Fuzzy-only endpoint: that control ended at 6,521 updates.

## Interruption recovery

On resuming the session near 22:12 UTC, the original supervisor/training
processes and container were absent. The old report still said running at
8,350 updates, but the history contained complete rows through 8,366 followed
by a NUL tail. Preserve these files; do not truncate or overwrite them.

The retained `train-mixed/checkpoints/step-008000.pt` matches its recorded
5,812,467-byte size and SHA256
`bc573f157110dfa23c20f50bf5c0e94ab09fe15ef40fb243eb89512a9214b72f`.
The existing frozen trainer resumes that checkpoint into fresh directory
`train-mixed-resume-8000`, restoring model, Adam, RNG and both data cursors.
Steps after 8,000 are replayed; the lost suffix is not counted twice.

Recovery supervisor PID at launch: 6707. Its logs and status are in
`recovery-20260916T221400Z/`. The original supervisor cannot finish the queue.
The recovery supervisor will finish the resumed 10k run, produce a report
combining the validated parent prefix through 8k with the resumed suffix,
and retain the evidence as `recovered-final` in the existing GCS lineage.
Training source and configuration remain unchanged. The container sees an
H100 80GB with the same software/capability and a different device UUID.

The resumed W&B run is
https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/afm9xxon.
The first ten replayed updates (8,001–8,010) exactly matched the original
losses, gradient norms, task metrics and both data-order chains. See
`recovery-20260916T221400Z/replay-observation.json`. This is evidence that
the existing full-state resume recovered the trajectory at the checked steps.

The parent and re-saved child 8k checkpoint packets are also exactly equal
in every field, though serialization hashes differ. Their complete proof is
`recovery-boundary-state-audit.json`. Reporting now follows the bound parent
checkpoint, validates committed history through 8k and joins the new suffix.
The old 366 later valid history rows and damaged tail are preserved but
excluded from the resumed result. Fourteen reporting CPU tests and checks
on the actual parent prefix, controls and boundary proof passed; see
`recovery-20260916T221400Z/reporter-ready.json`. Final evaluation/reporting
still occurs only after the resumed model reaches its endpoint.

The subsequent work is a bounded batch-throughput check after this pilot
finishes. It does not authorize larger-batch A5 or mixed learning pilots.
The waiting throughput supervisor is PID 11040 in
`.runtime/rt-nextlat-fuzzy-a5/20260916T222000Z-d128-batch-profile/`.
