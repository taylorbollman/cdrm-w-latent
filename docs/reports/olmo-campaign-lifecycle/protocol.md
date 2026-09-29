# Distributed boundary lifecycle check

2026-09-29. Independent CPU readiness work during the numerical startup study.
Add an opt-in helper and bounded two-process Gloo fault harness. Do not modify
the old campaign runner, active training helper, checkpoint implementation or
model. No GPU execution is part of this protocol.

All ranks must enter the helper in the same order at a quiescent boundary outside
graph capture and in-flight collectives. Agree on the phase and callback
ownership before executing any callback, then agree on ordinary Python callback
exceptions. Rank-zero callbacks execute only there; their return values remain
local. Common failures identify phase, rank and exception type without exception
messages. No automatic retry, rollback, RNG restoration or cursor mutation.
Successful peer callbacks may already have side effects when another fails.

Test eight cases with real CPU Adam state and rank-specific RNG: successful
logging/checkpoint publication; rank-zero logging failure after update2;
rank-zero publication failure after update2 but before atomic rename; rank-one
data preparation failure before update2; mismatched phase; mismatched callback
ownership; invalid local descriptor; and simultaneous distinct callback errors.
Every case starts from a completed update1 checkpoint containing both rank states.

Acceptance requires matching errors on both ranks, expected callback ownership,
no execution of callbacks when descriptors disagree, no subsequent update after
an observed failure, readable unchanged last complete checkpoint after failure,
and exact model/Adam/RNG/cursor replay of completed-but-unsaved update2. This is
checkpoint recovery, not rollback: in-memory update2 remains completed after a
logging/publication error, and recovery discards/replays that unsaved work.

Use a20-second Gloo collective timeout and a60-second parent worker deadline.
Dead peers, process signals, hung callbacks, failed collectives, NCCL/CUDA graph
fault recovery, real service outages and remote storage durability are excluded.
The tiny atomic checkpoint only establishes the test oracle and is not a new
production checkpoint format. A later runner must deliberately integrate this
helper at appropriate safe boundaries; existing runners retain their known gap.
Retain the report, four new source files, CPU logs and tiny checkpoint fixtures
under the persistent project runtime directory.
