# Adapted-state precision progress

2026-09-29. Diagnostic complete on `feat/olmo-adapted-precision`, from
`eb42ad2`/PR42. Protocol4fe8b0f, runner9380035, runtime freeze9f4693e.
Read results.md, next-steps.md and storage-receipt.md. No production change,
precision promotion, architecture change or new training occurred.

## Result

Two NF precision cases/four physical backwards completed in63.210seconds.
Backbone BF16/FP32 gradient relative L2 is0.9085% (cold60.8698%), cosine0.9999588;
fusion1.4051% (cold65.2122%). Backbone absolute difference332.723→0.46055 and
fusion56.6953→0.018535: not just a changed norm denominator. Both cases have
finite losses/states/gradients, zero predictor gradients and exact fixed-state,
fixture, source, RNG and import checks. Actual checkpoint SHA verified and68
saved backbone/fusion state entries imported exactly;4fresh predictor entries
retained. No optimizer/cursor/scheduler/RNG history restored.

Record0/pass1 hidden-state difference is12.4366%, worse than cold4.0913%; final
hidden differences1.9277%/2.2201% and incoming cotangent differences0.3848–0.5932%.
Do not describe all forward measures as improved or assume the intermediate
spike harmless. Its position-level cause is unmeasured. These CE-only/no-RT/T16
results do not clear RT, actual auxiliary gradients or packed T1024 training.

W&B [8ymbic44](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8ymbic44)
is synced. Report SHA256
`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.
Both H100s are idle with0MiB allocated and0%utilization at postflight. No further
GPU experiment is queued. Recommended next: two crossed-state precision pairs
to guide startup, with position-level observations; not launched.

## Validation and retention

Independent protocol/runner/import/result review passed. Focused CPU suite:
43passed in4.69s (23import,5runner,15prior recurrence checks), with one harmless
test-only scalar-conversion warning. Log `cpu-final-01.log`, SHA256
`3809e7d6fb42b8f5ac127d63f660f5eba99d263f4c1f86affcf441774da24594`.
Importer SHA `ab5895e968dcecbf38a585a99e91828d3f7d8cdbbbdd063a26147dd41fca7ff5`;
runner SHA `fe081faf4fbcb2caa1449209c28557f9314527e70dd83d50f2ff277c73a4e91f`.
Test files retain the reviewed bytes; warning cleanup was deliberately omitted
once source freeze began. No previous runtime source changed.

Independent baseline audit checks37historical sources,31unchanged/6explicit
migrations,89cold sources/snapshots and original fixture/noise/recipe pins.
Final result review verified104current source/snapshot pairs and report controls.
No numerical budget was relaxed. Runtime root `.runtime/olmo-adapted-precision/`.
The live NVIDIA catalog had no strong custom-PyTorch-numerics match; no installs.

GCS namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T064900Z/`.
Completed `adapted-01` report/log/source evidence is retained. Final independent
cloud readback, closeout and PR details are recorded below when complete.
No new checkpoint or gradient-vector dump; the original O5c checkpoint is reused
without duplicating it in this stage. No local artifact was deleted.
