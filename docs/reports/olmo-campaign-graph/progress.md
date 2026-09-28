# Campaign trainer integration — interruption record

2026-09-28. User authorizes remaining one-GPU implementation/checks and will move
to two GPUs when real distributed update and restart qualification is needed.
Save source/progress every 20–30 minutes. No long training or allocation.

Branch: `feat/olmo-campaign-graph-integration`, based on PR35/main `50aa1fe`.

Implementation completed at `e2fba4c`; bounded GPU acceptance is next:

- Model agent: opt-in valid-prefix padding fast path; native RT keeps true key
  validity, ordinary causal Flash omits redundant padding masks; mutable prepared
  forward buffers preserve graph storage and zero invalid output positions.
- Loss agent: new fixed-shape dense masked NextLat loss, preserving detachments,
  vocabulary chunking and target counts while allowing graph mask refills.
- Data agent: eager DDP per-microbatch jitter tensor channel; coordinated checks,
  existing independent global denominators and no_sync accumulation preserved.
- Root: new campaign objective/graph accumulation runner, integration tests,
  bounded actual-checkpoint GPU probe and documentation/retention.

The new runner will initially require every positively weighted objective to
have a positive **global** logical-update count; zero local counts and empty
local batches remain supported. This prevents accidental optimizer decay on
globally unused auxiliary parameters. It does not generalize arbitrary changing
global parameter participation within one captured graph.

CPU/Gloo tests are reference diagnostics, not actual multi-GPU/NCCL acceptance.
Use CPU Docker explicitly; all GPU work must use the project GPU container.
Final CPU suite: **867 passed in 51.58 seconds**. This includes all-eight-arm
raw-gradient accumulation references, two-rank CPU/Gloo jitter/empty-slot/Adam
checks, and actual tiny checkpoint publication preserving persistent buffers.
Earlier development failures were fixture/setup mistakes, corrected in this
final passing suite; raw logs remain local. H100 verified idle inside container.

Runtime sources are frozen for source-pinned GPU probe launch. The probe compares
one captured graph under changing padding/noise/counts, empty-slot zero behavior,
and two complete eager versus replay Adam updates. See protocol.md for budgets.
Launcher limit15minutes; each stage saves independently.

GPU `gpu-01` at runtime `0a074ea`: all11 stages passed in55.8s. Both layouts'
losses/raw gradients match eager exactly; two complete Adam updates, parameters,
moments, schedule and counters also match exactly. One graph,13replays, zero
dummy contribution,63unchanged source hashes. W&B `g3sqgfc0` synced.
An eager-after-capture stream warning exposed retention of a completed Python
autograd graph in diagnostic objective output. Detach that scalar after backward
(preserving graph output storage); targeted test added. A second bounded probe
will qualify this lifecycle fix and retain the first passing attempt too.
