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

## Implementation underway

Protocol/handoff pushed at `4fe8b0f`. Independent baseline audit verified37
historical source pins (31unchanged,6explicit revisions) and89cold source pairs;
see baseline-and-controls.md. New runner reuses existing backward/capture helpers
and verifies cold construction without an extra backward. New import adapter
validates architecture/tensors/buffers and preserves current trainability.
Importer tests and final review remain pending before GPU execution.

Live NVIDIA catalog checked via the fallback URL because npx is unavailable;
no strong match for this custom PyTorch numerical check, no installs made.
Retention namespace reserved for this milestone:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T064900Z/`.

## Ready for GPU

Independent protocol/runner/import review passed. Focused CPU suite passed43
checks in4.69s; one test-only scalar-conversion warning, no runtime failure.
Log `.runtime/olmo-adapted-precision/cpu-final-01.log`, SHA256
`3809e7d6fb42b8f5ac127d63f660f5eba99d263f4c1f86affcf441774da24594`.
The suite includes23import,5runner and15prior recurrence tests.

Importer SHA `ab5895e968dcecbf38a585a99e91828d3f7d8cdbbbdd063a26147dd41fca7ff5`;
runner SHA `fe081faf4fbcb2caa1449209c28557f9314527e70dd83d50f2ff277c73a4e91f`.
No previous source changed. Root will launch `adapted-01`, two cases only.
