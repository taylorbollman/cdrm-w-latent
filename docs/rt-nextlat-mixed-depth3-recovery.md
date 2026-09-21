# Three-layer mixed-task run: shutdown recovery

Training resumed **2026-09-20 at 23:52:04 UTC** from update **1,200**, toward the
original **5,000 total updates**. The model, FP32 execution, physical/effective
batch of 2,560 per source, task objective and LR schedule are unchanged. LR
remains 3e-4; the completed warmup is not repeated.

[Continuation W&B run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/bqexlzv2)
shares the original run's group. Its global update counter begins at 1,201.
[Interrupted W&B run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/k7hqmegl)
remains separate. Neither the source trainer nor the historical artifacts were
modified to resume.

## Recovered state

The final complete checkpoint is
`.runtime/rt-nextlat-fuzzy-a5/20260920T221646Z-d128-l3-b2560-lr3e4-5000/train/checkpoints/step-001200.pt`,
SHA256 `58b248dcbe1bb515c443b6dbf33a8a4fd7cac180bd5b0d80dc3dbd03f12d0606`.
Its recorded hash matches the current bytes. All 34 parameter/Adam states are
finite FP32, all Adam counters equal 1,200, initialization and the exact
model/data/runtime/source contracts match, and both stream digests were
independently reconstructed through all 1,200 committed updates.

Each task has consumed 3,072,000 examples. A5 resumes at epoch 3, row 672,000;
Fuzzy resumes at epoch 240, row 0. Model, optimizer, RNG state, LR and both
data cursors are restored by the unchanged strict loader.

The interrupted history contains valid rows 1–1,207 followed by 2,828 NUL bytes
from the shutdown. All original bytes are preserved. Only rows through the
saved boundary 1,200 are canonical ancestors of the continuation. Rows
1,201–1,207 are replayed; they must not be counted twice in curves or timing.

Actual restart validation passed: the first seven replayed updates have exact
equality for every logged training value, learning rate, example count and
data-order digest. Timing is deliberately excluded. The resumed report's
contract equals the parent's, and new recorded updates are finite.

## Execution and reporting

New runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260920T234740Z-d128-l3-resume-5000/`.
Supervisor PID at launch: **9549**.

- `recovery-audit.json`: saved boundary, unchanged source hashes and damaged tail.
- `checkpoint-cpu-audit.json`: independently checked model/Adam/RNG/data state.
- `resume-config-audit.json`: only output, resume checkpoint, stop marker and
  W&B run name differ from the original invocation.
- `resume-startup-audit.json`: exact seven-update replay and running-state checks.
- `ready.json`: frozen training and recovery sources before restart.
- `analysis-ready.json`: independently frozen continuation reporting code and
  checks, required before the supervisor generates the final comparison.

The new `scripts/rt_nextlat_a5_fuzzy_depth_resume_report.py` joins only verified
checkpoint ancestors. It audits the resaved boundary state, includes the
original initialization checkpoint, preserves valid earlier evaluations, and
uses resumed history after update 1,200. The two-layer comparison, metric
definitions, plots and qualification remain the existing depth report's.
Its 39 focused CPU tests passed, including an immediate stop with no new
updates. The actual resaved step-1,200 checkpoint exactly matched all 13 parent
packet fields, including model, Adam and RNG state. `analysis-ready.json` now
freezes the validated continuation reporting sources.

The supervisor stops at total 5k, creates the stitched comparison and retains
the continuation evidence. Touch the new runtime's `STOP` for a clean saved
and evaluated stop. No automatic extension or subsequent experiment is queued.

Final report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-l3-b2560-lr3e4-5000-resumed/`.

## Retention

The interrupted parent was already uploaded and verified before restarting:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260920T221646Z-d128-l3-b2560-lr3e4-5000/shutdown-recovery/interrupted-1200-evidence.tar.gz`.
Archive SHA256:
`06f55e24c06143dfdc6089dad6eee93377ccea981b77d661818aa4737e126de9`.

The receipt is copied into the new runtime as `parent-retention-receipt.json`,
so the final continuation archive identifies both pieces of the lineage.
Final continuation retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260920T234740Z-d128-l3-resume-5000/`.
