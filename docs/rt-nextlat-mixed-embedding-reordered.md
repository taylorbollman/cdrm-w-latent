# Revised mixed embedding run order

**Stopped by user on 2026-09-18 after the value arm completed.** Value reached
15,000 updates and was reported and retained. The automatically started head
arm was stopped at 368 updates, checkpointed, reported and retained; input
will not resume. Training supervisor 382114 is `stopped_and_retained`, and the
four-model length observer 382778 is `cancelled_before_evaluation`. The launch
notes below are historical. The follow-up is a baseline/value comparison and
an A5-only BF16 repeat, documented in `rt-a5-l1r-bf16-repeat.md`.

The user changed the order on 2026-09-18 to **stored-value addition →
embedding-access head → input addition**. All model, data, batch, NextLat,
precision and 15,000-update endpoint settings remain those in the
[comparison protocol](rt-nextlat-mixed-embedding-comparison.md).

**Active:** supervisor PID **382114** launched at 08:10 UTC on 2026-09-18,
after the baseline's shared length probe completed and was GCS-retained.
Stored-value addition passed its small backward checks and two discarded
B2560 mixed updates (59.17 GiB peak allocated), then began fresh training.
[Stored-value training W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/ucahrd6v)
is online, and initial optimizer updates have been verified in local history.
The reordered control/resume path passed 36 focused CPU tests; the original
training and model source hashes remain unchanged. Later arms run automatically
after each preceding arm is summarized and retained.

Input addition had already started. It was cleanly stopped at **452 updates**,
fully evaluated, summarized and retained. Its remaining training will resume
that exact model/Adam/RNG/data-stream state after the other two arms finish;
its final total remains 15,000 updates. Value and head start fresh from the
original paired initialization. The saved 452 state is not used by either.

Saved input checkpoint:
`.runtime/rt-nextlat-fuzzy-a5/20260918T072040Z-d128-b2560-embedding-comparison/input/train/checkpoints/step-000452.pt`.
SHA256: `bfa24f4cbf19fbc74294da00e4942a7b776c79424f721796c79981321acb3343`.
Consumed examples: 1,157,120 per task; all 26 model/Adam parameter states finite
FP32. Verified archive:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T072040Z-d128-b2560-embedding-comparison/input/final-evidence.tar.gz`.
Its earlier comparison report is an explicitly partial 452-update observation.

The old training supervisor stopped and retained this arm; it will not start
value or head. Its original length-evaluation observer was cancelled before
any GPU evaluation. Original scripts and retained evidence stay unchanged.

## New training supervisor

Runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T075200Z-d128-b2560-embedding-reordered/`.
Read `status.json` for actual launch/progress. `execution-config.json` specifies
the order and preserved input parent. New driver:
`scripts/rt_nextlat_a5_fuzzy_embedding_reordered_queue.py`.

Each completed arm is summarized and GCS-retained before the next starts.
Value/head get the same bounded GPU preflight as originally planned. Input
already passed that preflight, so its remaining training uses strict resume
checks and a CPU exact boundary-packet comparison before final reporting.
The generic boundary proof is emitted in the existing reporter's recognized
schema; training history joins original 1–452 and resumed 453–15,000 once.
No initializer, numerical policy or source-contract requirement is relaxed.

New comparative reports:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-embedding-reordered/after-value/`,
`after-head/`, and `after-input/`. They compare completed full-budget arms
against the baseline; the paused partial input is included when its 15k run
finishes. Checkpoints and reports are retained below
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T075200Z-d128-b2560-embedding-reordered/<variant>/`.

The queue-root `STOP` requests a clean active-arm stop and cancels later arms.
`STOP_QUEUE` prevents the next arm while permitting the current one to finish.

## Shared longer-length evaluation

The user requested a baseline-only check **before** starting these variants.
That check is complete at the original 15k checkpoint: Fuzzy answer accuracy
is **99.9112% / 99.4959% / 79.4008%** at lengths **400 / 512 / 1024**;
whole-sequence exactness is **97.8906% / 82.2656% / 0%**. At T1024, first-value
accuracy is 74.4914% and terminal-query accuracy is 48.9054%, providing useful
headroom beyond the longer sequence's extra opportunities for error. Keep
training and the shared evaluation pools unchanged. Report:
[baseline length probe](reports/rt-nextlat-fuzzy-a5/d128-b2560-baseline-length-probe/report.md);
[W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/ug72unjr).
Runtime: `.runtime/rt-nextlat-fuzzy-a5/20260918T080000Z-d128-baseline-length-probe/`.
Its `final-evidence.tar.gz` is verified under the matching GCS runtime prefix
(SHA256 `0a615e67ed0fc0a30ab2ca1cc5d4deb8fe83ae31044f976006b7e42cde4baa8b`).

The [authorized T512/T1024 protocol](rt-nextlat-fuzzy-length-evaluation.md)
and already prepared shared datasets are unchanged. The replacement observer
waits for this revised value/head/input queue to complete all 15k endpoints:
`.runtime/rt-nextlat-fuzzy-a5/20260918T075200Z-d128-embedding-length-eval-reordered/`.
Its new driver is `scripts/rt_nextlat_fuzzy_length_reordered_queue.py`; it reuses
the already validated evaluator and checks retained endpoints from the new
training paths. Baseline still uses its original completed 15k checkpoint.

The original prepared data/source archive remains verified under
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T074000Z-d128-embedding-length-eval/prepared-evidence.tar.gz`.
Final longer-length results will be retained under the new observer's prefix.
No further user review is needed between the authorized arms.

Replacement observer PID **382778** is running and waiting for the three
reordered 15k endpoints. Seven focused observer checks passed. It will use the
same shared T512/T1024 data as the completed baseline probe, with unchanged
model/evaluation code and retained-checkpoint verification.
