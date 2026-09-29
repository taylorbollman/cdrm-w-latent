# Fixed-boundary precision progress

2026-09-29. Active branch `feat/olmo-boundary-precision`, from `802a186`/PR41.
User approved the shared-input/shared-cotangent diagnostic and wants a path to
usable BF16. Read [protocol](protocol.md) first. Two full NF CE anchor cases
(four physical backwards) and 12 local VJPs are authorized in this bounded
milestone. No production change or new training run is planned here.

Root writes protocol/handoff, launches GPU and handles PR/retention. Runner
agent owns the new helper/tests; precision reviewer checks module boundaries;
data reviewer verifies prior pins and final evidence. Old sources stay frozen.
Record 0, passes 1/3, fusion and following ordinary stack are the four sites.
Both FP32 and BF16 local outputs must reproduce their original captures.

Preflight confirmed both H10080GB devices idle inside the correct container,
with 0 MiB allocated and about 399 GiB free on persistent disk. GPU 0 alone
will execute this probe; GPU 1 remains unused. Runtime root:
`.runtime/olmo-boundary-precision/`. Each GPU stage is limited to 900 seconds.
Save work every 20–30 minutes. No numerical result has been produced yet.

## Implementation reviewed and ready

New helper and tests passed independent boundary/precision review. Final
focused CPU scope: 31 tests passed in 3.85 seconds, no warnings. This includes
10 new boundary checks plus existing fusion/attention-local tests; the earlier
10-test run overlaps. Log `cpu-boundary-final-01.log`, SHA256
`fe2835620d2720f970c59d1b99819fcc05151ca5534f5f34ecd7c0857382d8b7`.
Helper SHA256 `33d8e6dc0a36fe681935bf425c0582616a28f099d1aad795a03d0ae17a75717d`;
test SHA256 `d3471cdc53444abbdbdb3a06a8f918f9f91077bab4de8910efb4db3166719329`.

Prior 161 source/snapshot pairs are unchanged. The live NVIDIA skill catalog
has no strong match for this custom PyTorch numerical probe; no dependency
was installed. The implementation uses existing tensor encoding and validation
helpers, with explicit layout equality, checkpoint replay exclusion, unused
parameter reporting, training-mode and runtime-flag integrity checks.

Root is freezing the runtime source and launching `boundary-01` on GPU 0.
Retention namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T061000Z/`.
All earlier helpers, tests and protocols remain frozen.
