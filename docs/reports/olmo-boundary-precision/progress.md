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
