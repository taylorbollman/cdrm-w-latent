# Portable campaign readiness — interruption record

2026-09-28. User authorized initial work effective on the current machine;
training will later be multi-GPU. Save incremental work every 20–30 minutes.
No long training, hardware reservation, or later campaign recipe freeze is
authorized by this milestone. The later-stage plan remains open to review.

Branch: `feat/olmo-campaign-readiness`.

Current scope:

- Opt-in configured RT bootstrap and explicit externally supplied feedback jitter.
- Separate normalized CE/auxiliary pass weights, preserving legacy defaults.
- Explicit recipe, optimizer groups, token schedule and deterministic jitter keys.
- Local tokenized-document/window/update/cursor primitives for later Dolma ingestion.
- Bounded CPU checks and a short GPU integration probe if available, with each
  stage independently saved. No single-GPU throughput or LR campaign.

Work ownership while active:

- `readiness_model`: olmo_fbt.py, olmo_static.py, campaign model tests.
- `readiness_losses`: fbt_training.py, static/distributed objective contracts,
  resource accounting and campaign loss tests.
- `readiness_data`: campaign_data.py and its tests/data-contract note.
- Root: campaign recipe/optimizer/schedule/noise, integration, documentation.

Portable milestone complete at runtime `714f31c`. Broad CPU suite: 706 passed
in 38.88s. Final Unicode JSONL fix: 28 ingest tests passed (overlapping scope,
two new cases). Fresh-process CPU mid-warmup restore reproduces the exact next
update, optimizer, schedule, counters, cursor and RNG output.

Actual-checkpoint GPU probe `gpu-smoke-01` completed all four stages in 41.9s:
canonical/literal loss and gradients match exactly; RT0/15 execute four times
each; all 71 active gradients finite; two fused Adam updates finite. W&B
`kb0lbu1k` synced. All 60 runtime source hashes are unchanged. No full disposable
checkpoint saved. Read results.md for the substantial initial clipping and
qualification limits. This is B1/T16 semantics, not production capacity,
quality or FP32 qualification. No long task or quality training is queued.
Do not rerun the completed probe.

New graph accumulation/dynamic layouts and distributed fresh-process recovery
belong to the next multi-GPU integration. Graph trainers explicitly reject
nonzero jitter until their noise-buffer lifecycle is qualified. Eager DDP needs
a separate tensor-input channel for per-rank jitter; backbone_kwargs metadata
cannot carry it. Historical checkpoint exact configurations must be retained:
new mode fields do not imply an automatic old-schema resume migration. Native
pretrained tensor loading is unchanged; campaign Adam starts fresh.

Next: usage.md dependency order. Padded Flash and graph jitter/mask/accumulation
integration need implementation, then actual multi-GPU update/recovery acceptance.
Current same-plan scheduler does not support unaudited horizon extension.
Retained evidence and object hashes: storage-receipt.md. No test required a
20-minute unsaveable span; maintain 20–30 minute recovery boundaries going forward.
