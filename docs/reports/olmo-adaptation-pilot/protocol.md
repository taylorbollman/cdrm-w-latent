# First 32-update adaptation comparison

The user authorized this cohort after PR51. Execute the prepared B-original and
paired NF/NFR-fusion128 declarations byte-for-byte, using the accepted 200-source
asynchronous runtime. Historical pure-planning `review-draft` and
`launch_authorized:false` metadata are preserved; current user authorization is
recorded separately. No model, optimizer, loss, kernel or runtime changes.

Run B, NF, then NFR sequentially on two H100 80GB GPUs in the required container.
Use T1024, 524,288 valid inputs/update and a CLI segment stop at update32, within
the unchanged 128-update finite declaration. Physical B32/GPU for B, B12/GPU for
NF/NFR; respectively8/22 accumulation slots. Each arm sees the same ordered
16,777,216 new inputs. B starts original weights/fresh Adam; NF/NFR share the
accepted fusion128 weight import, fresh paired predictor and fresh all-active
Adam. Their prior fusion exposure is separate. Do not resume the PR51 diagnostic.

Keep BF16 mixed, graph preparation/capture, activation checkpointing, fused Adam,
LR2e-4 peak with100-update token warmup, clipping1.0, NextLat regression/KL,
K4 feedback and native RT0/15 exactly as declared for enabled arms. No Q/K change.
Record physical capacity and accumulated timing on actual updates; NF's cost is
initially unknown. Budget roughly two hours for the NFR segment.

Evaluate the fixed65,536-input dev-main prefix in common FP32 without jitter at
updates16/32, reporting every trained pass, separate losses and CE gaps relative
to pass1. There is no update-zero evaluation. This is a functionality/adaptation
pilot, not a conclusive scientific comparison; stopping32 is still in warmup.

Save immutable checkpoints on local SSD, then upload/verify in the background.
Keep the600-second save trigger, explicit32-update milestone and terminal drain.
One worker, keep-two verified local states per owned segment. Use the preceding
verified cloud checkpoint if VM/SSD are lost during upload. The trigger is not a
maximum rollback interval. Persistent metadata and W&B remain under the project;
all useful large states are retained in gs://fast-chunks.

Stop on nonfinite values, concrete execution/integrity failures or user request.
Poor CE or clipping alone does not trigger an unplanned configuration change.
At32, assess later-pass gaps, first-pass trajectory and gradient/auxiliary scales.
If improvement is absent or auxiliary losses improve while CE worsens, pause
extension and recommend a bounded saved-state per-loss/fusion diagnostic.
Do not automatically continue to128 or reopen a broad BF16 investigation.

Paths: `.runtime/olmo-adaptation-pilot`, SSD
`/mnt/localssd/cdrm-checkpoints/adaptation-pilot`, reports in this directory.
Checkpoint cloud roots remain those in the pinned declarations, with new unique
arm/segment suffixes. Small evidence uses the new20260929T231346Z namespace.
Root owns all GPU launches. Check reports/processes before resuming after an
interruption; never repeat a completed segment blindly. Save/push progress and
retain stage evidence every20–30minutes where practical, alongside live runtime
checkpoint retention.
