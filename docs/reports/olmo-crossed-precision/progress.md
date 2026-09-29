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

## Implementation validated

Protocol/position baseline pushed at6dd5a16. Independent code and test review
passed. Final focused CPU scope:58passed in3.63s (22crossed,8position,28prior
adapted checks); one previously accepted warning in an unchanged old test.
`cpu-final-01.log` is an overlapping58-test preparation run, not an additional
independent scope. Final `cpu-final-02.log` includes graphable position summaries.
No GPU model execution yet. Root is freezing all new runtime files forcrossed-01.
