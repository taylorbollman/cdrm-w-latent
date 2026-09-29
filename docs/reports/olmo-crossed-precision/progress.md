# Crossed-state precision progress

2026-09-29. Active on `feat/olmo-crossed-precision`, from318948c/PR43.
User approved two crossed-state NF precision pairs. Read protocol.md first.
Four aggregate cases/eight physical backwards, no training or diagonal reruns.

Root owns position observations/tests, protocol/handoff, GPU and retention.
Runner agent owns component assembly/main runner/tests; independent precision
reviewer checks semantics. Data reviewer verifies paired report authority and
results. Prior helpers/importer/tests/protocols remain unchanged.

Preflight: correct project container, both H10080GB GPUs idle (0MiB,0%),
about399GiB persistent disk and447GiB host memory available. GPU0 only will run,
under900s external limit. Runtime `.runtime/olmo-crossed-precision/`.
No model execution yet. Save/push every20–30minutes.
Retention namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T072000Z/`.
