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

Implementation checkpoint: portable model/loss/data/recipe and bounded raw-input
ingest APIs are in place. Broad CPU suite:706 passed in38.88s. Final Unicode
JSONL fix:28 ingest tests passed (overlaps the broad suite; two new cases).
Fresh-process CPU mid-warmup restore reproduces the exact next update, optimizer,
schedule, counters, cursor and RNG output. Runtime sources are frozen for the
actual-checkpoint GPU probe; see protocol.md. Launcher limit is15minutes.
No individual operation planned requires20minutes of uninterruptible execution.

New graph accumulation/dynamic layouts and distributed fresh-process recovery
belong to the next multi-GPU integration. Graph trainers explicitly reject
nonzero jitter until their noise-buffer lifecycle is qualified. Eager DDP needs
a separate tensor-input channel for per-rank jitter; backbone_kwargs metadata
cannot carry it. Historical checkpoint exact configurations must be retained:
new mode fields do not imply an automatic old-schema resume migration. Native
pretrained tensor loading is unchanged; campaign Adam starts fresh.
