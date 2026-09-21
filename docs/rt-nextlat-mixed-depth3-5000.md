# Three-layer mixed A5/Fuzzy pilot

**Latest:** the 5k run completed and was retained; the user subsequently
authorized [continuing the same run to 7,500 updates](rt-nextlat-mixed-depth3-7500.md).

**VM interruption recovered:** resumed the intact step-1,200 checkpoint on
2026-09-20 at 23:52:04 UTC, with the same recipe and original total 5k endpoint.
Follow the [continuation run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/bqexlzv2).
The seven valid post-checkpoint updates (1,201–1,207) reproduced every logged
training value and data-order digest exactly after restart; elapsed times are
excluded from that equality claim. See [recovery notes](rt-nextlat-mixed-depth3-recovery.md).

Launched **2026-09-20 at 22:24:04 UTC**, supervised process **27028**.
Live run:
[l1r-rt3-nextlat-d128-b2560-mixed-lr3e-4-5000](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/k7hqmegl).

The user authorized a fresh three-layer run following the successful two-layer
3e-4 pilot. The architecture is **window-2 RT → full RT → full RT**, with NextLat.
Run for **5,000 optimizer updates**, save and compare the endpoint, then stop.
No following architecture trial or automatic learning-rate change is queued.

## Fixed experiment

- D128, 16 heads (head width 8), full-width Q/K/V projections, GELU FFN512,
  ALiBi, rho 1, tiled recurrence.
- Only index 0 restricts direct attention to provisional self K/V and the
  preceding permanent K/V. Indices 1 and 2 retain full causal RT attention.
- 676,736 parameters: 610,944 backbone and 65,792 NextLat predictor.
  The two-layer reference has 479,616 total parameters.
- Fresh canonical Mitchell initialization at the actual three-layer depth,
  with the same seed rule. No transplant from a trained model and no claim
  of identical backbone weights across depths. The separately seeded
  predictor is exactly identical at initialization to the two-layer predictor.
- FP32 eager; no autocast, TF32, compilation or CUDA graphs. No embedding
  bypass, embedding-access head or autonomous latent rollout.
- The same frozen A5 T12/Fuzzy T400 corpora, vocabularies, labels and task orders.
  Each update uses **2,560 A5 and 2,560 Fuzzy examples**, each processed at its
  native sequence length. Physical microbatch remains 2,560.
- Equal task means of CE + NextLat, latent weight 1, one global clipping
  operation at norm 1 and one AdamW update. Betas (.9,.95), epsilon 1e-8,
  matrix decay .01 and vector decay 0.
- LR exactly 1e-4 at update 1, linear warmup to 3e-4 at update 100, then
  constant 3e-4. A future LR reduction would be a separate decision.

Development evaluation remains every 100 updates: 4,096 A5 words at lengths
12 and 36 and all 1,280 Fuzzy development examples. Full A5 evaluations use
102,400 words at 1k, 2.5k, 5k and the existing first-positive full recheck.
Checkpoint/evaluation behavior, task losses and stream ordering use the same
training loop as the preceding LR run. Final confirmation stays unused.

Compare with the two-layer
[3e-4 run](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/y0mjyvyb),
which reached 83.4307% A5 L36 whole-word and 99.5960% Fuzzy answer accuracy at 5k.
Use equal-update and training-time curves; the additional block costs compute
and parameters. A5 monitoring fluctuations should be reported along with the
full endpoint, not replaced by a selected best checkpoint.

## Implementation and checks

The new `cdrm/rt_nextlat_task_depth.py` factory and
`configs/rt_nextlat_tasks/fuzzy_d128_l3.json` describe the actual depth.
`scripts/rt_nextlat_a5_fuzzy_depth_train.py` scopes model construction and
source-manifest overrides around the unchanged LR trainer. Historical source
files stay unchanged. Model/config/initialization guards distinguish the new
checkpoint lineage even though the validated training schema is reused.

Forty-two focused CPU tests passed: 20 depth/factory/adapter checks and the 22
existing LR/checkpoint checks. The actual batch GPU check completed four
discarded-weight updates on the real task data. All 34 parameter tensors,
retained gradients and Adam states were finite FP32. Peak memory was
58.686 GiB allocated and 60.145 GiB reserved, so no accumulation fallback was
needed. The last three check updates averaged 3.001 seconds each; this is a
rough startup estimate, suggesting approximately 4–4.5 hours for the pilot.

The [memory check](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/7kn8nmhi)
is separately labeled in W&B; its weights are discarded before the fresh run.
Independent review found no model/training blocker and passed 13 supervisor
cases, including clean stops and rejection of changed depth/batch/endpoints.
The comparison reporter passed 25 focused CPU tests and a real-reference
parsing/rendering check. Its smoke plots are labeled synthetic rendering
fixtures, not three-layer experiment results.

## Runtime and retention

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260920T221646Z-d128-l3-b2560-lr3e4-5000/`.

`execution-config.json` records exact arguments; `baseline-audit.json` confirms
that the only substantive model-config change is depth and that the previous
51 executed source files are unchanged. `ready.json` freezes the source closure
and checked GPU profile before launch. `status.json`, `train/report.json` and
`train/history.jsonl` record progress. Touching the runtime's `STOP` requests
a checkpointed, fully evaluated clean stop.

The depth supervisor automatically generates the CPU comparison after training
and retains checkpoint/report evidence under
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260920T221646Z-d128-l3-b2560-lr3e4-5000/`.
Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-l3-b2560-lr3e4-5000/`.

The depth comparison is one fresh seed on reused development pools. A new
depth changes Mitchell scaling and RNG consumption; matching seeds and rules
does not make all common backbone tensors identical. Changes in performance
cannot be attributed to parameter count, depth, or initialization separately
from this single comparison.
