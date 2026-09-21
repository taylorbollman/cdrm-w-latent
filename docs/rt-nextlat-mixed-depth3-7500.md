# Three-layer mixed run: continuation to 7,500 updates

Started **2026-09-21 at 03:59:06 UTC**, supervisor PID **17796**.
[Live continuation run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/w66moalv).

Authorized 2026-09-21: continue the existing three-layer mixed A5/Fuzzy run
from its completed **5,000-update checkpoint to 7,500 total updates**.
This adds 2,500 optimizer updates. It does not restart initialization or warmup.

The architecture remains restricted RT → full RT → full RT with NextLat,
D128/H16/FFN512, ALiBi/Mitchell and no embedding injection. Execution remains
FP32 eager, with physical and effective batch 2,560 per data source, equal
task weighting, the same AdamW settings and clipping at norm 1. LR stays at
**3e-4**. All frozen training source files remain unchanged.

The purpose is to observe the unresolved learning trajectory for longer.
Routine development evaluation remains every 100 updates. Full A5 evaluations
on 102,400 words per length are scheduled at **6,000, 6,500, 7,000 and 7,500**;
Fuzzy continues to use all 1,280 development examples. No confirmation data
or latent rollout is evaluated. Stop at 7,500 and review; no further run,
automatic extension or learning-rate adjustment is scheduled.

## Checkpoint and lineage

Parent training directory:
`.runtime/rt-nextlat-fuzzy-a5/20260920T234740Z-d128-l3-resume-5000/train`.
Checkpoint: `checkpoints/step-005000.pt`.
SHA256: `06e4944877192d424b30a928dc8c143efd0ae527208ce7b312698fc33d10213a`.
The checkpoint bytes match both the completed report and verified GCS receipt.
The bounded CPU checkpoint audit verified all 34 finite FP32 parameter/Adam
states, Adam counters at 5,000, exact contract/source/initialization identity,
and stream cursors at the next epoch boundaries. The GPU resume loader passed;
initial resumed updates are finite, use LR 3e-4 and start at global update 5,001.

This continues the original lineage 0→1,200→5,000→7,500. The original shutdown
tail remains excluded, as documented in [recovery notes](rt-nextlat-mixed-depth3-recovery.md).
Model, optimizer, RNG state, learning-rate position and both task-stream
cursors are restored through the unchanged strict loader.

The two-layer reference ends at 5,000. Reporting must keep matched depth
comparisons at or below that shared budget and label the additional three-layer
training as longer-budget context. Do not extrapolate the baseline or claim
matched training time at 7,500. Preserve the existing complete 5k report.

## Runtime and automation

New runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260921T035613Z-d128-l3-continue-7500/`.

`execution-config.json` contains exact arguments. `ready.json` freezes training
and supervisor sources before launch; `analysis-ready.json` separately freezes
the endpoint-aware reporting adapter before post-run comparison.
`train/report.json`, `train/history.jsonl` and `status.json` record progress.
Touch this runtime's `STOP` to request a saved and fully evaluated clean stop.

The supervisor uses the existing depth trainer and the new
`scripts/rt_nextlat_a5_fuzzy_depth_extend_report.py` for the longer lineage.
It produces results and verifies endpoint checkpoint retention after stopping.
The reporting adapter passed 45 CPU tests. Its real-lineage smoke check verified
the exact model/Adam/RNG/LR/data state at the resaved 5k boundary and retained
the original 0→1,200→5,000 ancestry. `analysis-ready.json` is complete and frozen.
Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-l3-b2560-lr3e4-7500/`.

Retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260921T035613Z-d128-l3-continue-7500/`.
The parent receipt is copied into this runtime; earlier archives remain intact.
