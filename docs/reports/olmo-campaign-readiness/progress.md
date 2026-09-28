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

Current execution: implementation in progress. CPU container validated; no GPU
probe started. No individual operation planned requires 20 minutes of
uninterruptible execution. Completed tests and remaining limitations will be
recorded here before closure. New graph accumulation/dynamic layouts and
distributed fresh-process recovery belong to the next multi-GPU integration.
