# Shared saved-context driver and JSON contract correction

2026-09-29. This new driver replaces only the unlaunched packed GPU driver and
adds a corrected endpoint check for the completed long-fixture cold comparison.
All existing model, data, warmup and original long-probe sources stay unchanged.

The first update128 long probe, `long-128-01`, failed before import/backward
because the live contract contains an empty tuple for selected RT layers while
JSON restores that field as an empty list. Both saved JSON contracts are exactly
equal; every other comparison guard passed. Preserve that failure and its source
snapshot. This is a representation check failure, not numerical model evidence.

The new driver canonicalizes complete JSON-compatible contracts before equality,
without omitting any field or changing its value. Its regression distinguishes
the equivalent tuple/list case from changed RT selection or feedback strength.
The unchanged original long cold loader verifies every old source pin against
the current source inventory, the fixture SHA, original model origin, and both
precision endpoints. Added source entries describe this new driver and tests;
no old source mismatch is allowed. Reuse the completed cold pair without another
cold model run. Long mode permits only the saved update128 endpoint, one pair:
two aggregate/four physical backwards, with the original fixture, model,
objective, runtime, first-pass identity and state/RNG integrity controls.

Packed mode follows `packed-probe-protocol.md`: two B1/T1024 rows from two held-out
source streams, cold and update128, four aggregate/eight physical backwards total.
The prepared artifact has 2048 inputs, 2046 CE targets, 2040 latent pairs and
2032 KL triples, including six actual within-row document boundaries. The
strict original isolated checkpoint import precedes the explicit policy-only
transition. Other configuration/state stays unchanged. Long mode makes no policy
transition. Neither mode constructs an optimizer or changes training progress.

Freeze this driver/tests plus the union of existing long/probe/data source pins
before either endpoint. Focused CPU tests cover policy semantics, actual tiny
packed gradients, reference rejection and JSON normalization. Each GPU stage is
bounded to 900 seconds, requires the container, online W&B, incremental reports,
source snapshots and immutable GCS retention. Numerical differences remain
descriptive; operational completion is not BF16 training clearance.
