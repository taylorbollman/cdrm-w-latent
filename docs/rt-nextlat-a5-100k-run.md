# Original RT + NextLat: stopped at the retained 80k checkpoint

The full-attention run is stopped. On 2026-09-14, the user selected the
**80,000-update checkpoint** after reviewing development curves. The original
plan was 100k; that prospective protocol remains unchanged as historical
evidence. No further full-attention training is running or authorized by
this stop. The later window-2 continuation is a separate experiment; see
[its run note](rt-nextlat-window2-100k-run.md).

Lineage: `.runtime/rt-a5/20260914T185736Z-rt-nextlat100k/`. Its directory name
records the original plan, not the accepted endpoint. The accepted checkpoint
is `train-rt-nextlat/checkpoints/step-080000.pt`, SHA256
`48de43b162f1dbfaeeb5a111f5786b0c0589d544ba02167769dccc672fdf075b`.
It is retained and verified in GCS, with the seven earlier continuation
checkpoints at 20k, 25k, 30k, 40k, 50k, 60k and 70k.

The user requested the stop after training had moved past 80k. SIGINT was
sent inside the verified training container. The process last reported
81,607 completed updates; its 1,607 uncheckpointed updates after 80k remain
in the raw logs/history and are excluded from the accepted model results.
There may also have been an interrupted in-flight step; none of its state
is used. Both the trainer process and its container were confirmed gone.
The raw report records `failed / KeyboardInterrupt` and W&B records
`synced_failed_experiment` because of the intentional stop. This does not
indicate an observed numerical failure. Do not rewrite those raw records
as a completed 80k run.

`endpoint-revision.json` binds the user-directed stop to hashes of the original
100k protocol, raw closed report/history, exit marker and accepted 80k
checkpoint. It records 70,000 accepted continuation updates and
3,784.4136585 seconds of training, plus 86.7657065 seconds for the excluded
1,607-update tail. Raw job time includes evaluation, saving, logging and the
tail. The trainer's raw `train_seconds` and `order_chain` still describe its
last saved 80k checkpoint; its `completed_updates` includes the later tail.
A stopped-run report must interpret this explicitly.

The accepted model is the original **RT + NextLat**, with Mitchell backbone
initialization, ALiBi and full-prefix recurrence in both layers: D512/H8,
GELU FFN2048, LayerNorm and learned full-width Q/K normalization, vocabulary60,
untied head, no dropout. Backbone/predictor counts remain6,357,504/1,049,600,
7,407,104 total in25 tensors. There are no window or sinusoidal overrides.
The loss remains same-position state CE plus weight-one SmoothL1(beta1)
next-latent prediction. Only target-role latents are detached; source
latents and next-operation embeddings remain attached. Accuracy evaluation
uses the backbone, without autonomous NextLat MLP recurrence.

The run resumed the original RT+NextLat10k checkpoint
`.runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat/checkpoints/step-010000.pt`,
SHA256 `c5e425eb0e3321ae87d74a83208bcc944827dfba7ae64af2206abf32ee7c7f5b`.
All46 training sources remain frozen at digest
`1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961`.
Model, Adam, RNG and data-order state were restored exactly under the existing
strict contract. The dataset, batch1024, length12, seeds1234/1235,
constant AdamW1e-4 and full FP32 eager runtime were unchanged. Autocast,
TF32, compile and CUDA graphs remain off. At80k the accepted model has seen
81.92 million words, or102.4 nominal passes over800k training words.

At the full80k development evaluation (102,400 words per role), short-word
token accuracy was99.9404% and whole-word accuracy99.5449%. On length36
outputs, E(13)=90.9102%, E(14)=29.6934%, E(16)=0.2236%, M(36)=37.8208%,
and E(36)=0 observed. The original10k E(13)/E(14) were96.8652%/62.2373%.
E(t) requires every state through t correct; A(t) measures only state t;
M(t) averages token accuracy through t. Full/zoom curves must use identical
length36 output rows. The user-selected80k endpoint is retrospective
development selection, not a prospectively fixed80k experiment. Independent
confirmation remains unevaluated.

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/b77bpb54)
was annotated with the accepted80k checkpoint and excluded tail. The raw
launch name/config remain in local records; `wandb-stop-annotation.json`
records the explanatory remote metadata update.

The original100k finalizer and watcher exited after the intentional interrupt;
their statuses are preserved. Do not restart them. New stopped-run helpers
are `validate_stopped_state.py`, `retain_stopped.py`,
`verify_stopped_evidence.py` and `audit_stopped_report.py`. Saved80k model/Adam
inspection passed31/31 checks, including every Adam counter at80k and all
80k accepted minibatch-order hashes matching the historical stream. Eight
checkpoints and original46 source files/snapshots passed retention checks.

The stopped reporter is `scripts/rt_a5_nextlat_stopped_report.py`; its output
is [the completed 80k report](reports/rt-a5/nextlat-budget-80k/report.md).
See also [the outcome note](reports/rt-a5/nextlat-budget-80k/outcome.md) and
[the comparison in W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/r0s16kq4).
All 11 reporter tests passed. Independent artifact review checked 624 metric
rows, 26 checkpoint evaluations, 14 checkpoint files, 800 accepted training
bins and 36 artifact hashes. It verified the shared full/boundary rows and
the exclusion of the unsaved tail. The boundary and update figures were
visually inspected. No further model evaluation was performed.

For archival closure use `retain_stopped.py archive`, followed by
`verify_stopped_evidence.py`, with command logs outside the archived lineage.
`evidence-storage.json` and `stopped-evidence-readback.json` are the final
archive/readback authorities; they are written after these notes. Do not
modify archived members afterward. No further full-attention training is
needed or authorized by this closure.

GCS prefix:
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T185736Z-rt-nextlat100k/`.
Preserve the original10k checkpoint and data in their historical lineages.
