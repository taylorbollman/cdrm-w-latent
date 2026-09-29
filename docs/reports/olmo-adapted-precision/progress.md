# Adapted-state precision progress

2026-09-29. Active on `feat/olmo-adapted-precision` from `eb42ad2`/PR42.
User approved the saved adapted-checkpoint comparison. Read protocol.md first.
Two aggregate NF CE precision cases/four physical backwards, no new training.

Root owns protocol, runner, GPU, PR and retention. Import agent owns the new
explicit model-only importer and CPU tests; independent reviewers check precision
semantics and source/fixture evidence. All previous completed sources stay frozen.

Preflight: inside correct project container, two H10080GB devices idle with
0MiB allocated, 0% utilization. About399GiB available on persistent disk.
Runtime root `.runtime/olmo-adapted-precision/`. Stage limit900seconds.
No GPU model execution yet. Save work every20–30minutes.
