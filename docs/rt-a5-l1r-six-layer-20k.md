# Six-layer L1R RT + NextLat: continuation to20k

## Completed endpoint

Training completed and W&B synced on2026-09-15 at15:13:19UTC. It stopped at
exactly20,000 updates. The six-layer model reached **80.9629% length36
whole-word accuracy** and **96.7088% token accuracy** on102,400 development
words. Its10k whole-word result was82.0771%; the matched two-layer reference
at20k reached96.3369%. The extra training did not improve the six-layer
endpoint. The six-layer model retains substantial state-tracking ability;
this is not a complete loss of that behavior.

All61 saved model/Adam tensors passed finiteFP32 and exact-counter checks.
The stitched20k minibatch order matches the reference. The final report has
384 primary metric rows, with the full and boundary plots using identical
length36 observations. Root visually reviewed all four figures: labels,
legends, axes and curves are readable without the original10k clipping issue.

- [Report and qualifications](reports/rt-a5/l1r-six-layer-nextlat20k/report.md)
- [Learning curve](reports/rt-a5/l1r-six-layer-nextlat20k/whole-word-vs-updates.pdf)
- [Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/vmg2i60x)
- [Comparison W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/pmajaq1r)

Both15k/20k checkpoints are retained separately; the evidence archive and
readback receipts in this lineage record final closure. Preserve this run and
its10k parent. No continuation beyond20k is authorized.

The next two **four-layer** embedding-injection
diagnostics are authorized,10000 updates each, input injection first and
permanent-value bypass second. Read
[the experiment handoff](rt-a5-embedding-injection-diagnostics.md).

## Original continuation protocol

The user explicitly extended the diagnostic to **20000 total updates** while
the first10k job was around8.7k, unless they return and request an earlier stop.
Resume the exact10k checkpoint with the unchanged model, optimizer, RNG and
data order. Retain15k and20k checkpoints. Do not extend beyond20k automatically.

Completed lineage: `.runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k/`.
Pointer: `.runtime/rt-a5/current-l1r-depth-extension-lineage.txt`.
CoordinatorPID20404 was queued2026-09-15 at14:45:59 UTC. `launch.py` waits for
the parent10k trainer to finish successfully, freezes the resolved continuation
protocol and parent hashes, and starts `train-depth` automatically. It never
retries a failed training job. Check `launch-status.json`, `training.log`,
`train-depth/report.json` and `coordinator.json` after an interruption before
launching anything. The GPU work always uses the project Docker launcher.

Parent lineage: `.runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k/`.
Read [its architecture and initial validation](rt-a5-l1r-six-layer-10k.md).
That job's10k report is still generated and retained independently. The
continuation starts at10000 and ends at20000; no fresh20k retraining or reset
of moments/schedule/data is authorized or intended.

Architecture and training are unchanged: **six total RT blocks**, window2
at index0 then five full RT layers; D512/H8/GELU-FFN2048, ALiBi, Mitchell
initialization at actual depth, attached recurrent states/gradients, original
NextLat objective/predictor, total19998208 parameters. B1024/T12, same800000
words and seeds, constant AdamW1e-4(.9,.95), matrix decay.01, clip1. FullFP32,
no autocast/TF32/compile/CUDA graphs. The unchanged58-source training identity
is `04d3854620854130eb9a67093fd0538b5762ac870b9464291fde8a6899fbc78b`.

The final comparison uses six-layer checkpoints1k/5k/10k/20k against the
two-layer L1R reference at the same updates and full102400-word development
evaluations. The child15k checkpoint is retained but is not a primary matched
comparison because the reference's15k evaluation used4096 words. Stitch the
parent1–10000 and child10001–20000 histories; check every minibatch-order hash.

Reference: `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first/`.
Report destination: `docs/reports/rt-a5/l1r-six-layer-nextlat20k/`.
The20k extension was selected after looking at development results; it was
not the original prospective endpoint. One seed/reused development data,
different-depth Mitchell backbone draws, and unequal parameter/compute budgets
remain qualifications. Confirmation and autonomous latent rollout are untouched.

Retain new checkpoints and final evidence at
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k/`.
Keep parent/previous lineages immutable after their respective archive readback.
Complete saved-state checks, reporting, plot/handoff review, then archive and
verify remote checksums with logs outside the lineage. The companion receipts
record the final archive and checkpoint object generations.
