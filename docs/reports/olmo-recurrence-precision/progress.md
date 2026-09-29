# Recurrence/precision milestone progress

2026-09-29. Active branch `feat/olmo-recurrence-precision`, from PR40/main
`4eff982`. The user authorized the next bounded numerical milestone.

Read [protocol](protocol.md) first. Initial scope is eight aggregate CE cases:
N/NR/NF/NFR × full FP32 versus production BF16, original T16/two-B2 fixture.
No training or core-model change. The protocol is drafted before execution;
there is no new GPU result yet.

Root owns GPU execution and retention. Runner agent implements the new helper
and tests; precision reviewer independently checks the arm/observer contracts;
data reviewer verifies prior references and matching controls. Old PR40 helpers
and protocols stay frozen. A conditional precision-boundary follow-up needs a
separate written scope after the matrix identifies a useful target.

Runtime root: `.runtime/olmo-recurrence-precision/`. GPU phases are limited to
900seconds and will use device0. Save/push and retain every20–30minutes.
Prior numerical qualifications remain unresolved until evidence supports a
specific resolution; operational completion alone is not clearance.
