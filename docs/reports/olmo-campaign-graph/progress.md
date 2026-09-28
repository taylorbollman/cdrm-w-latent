# Campaign trainer integration — interruption record

2026-09-28. User authorizes remaining one-GPU implementation/checks and will move
to two GPUs when real distributed update and restart qualification is needed.
Save source/progress every 20–30 minutes. No long training or allocation.

Branch: `feat/olmo-campaign-graph-integration`, based on PR35/main `50aa1fe`.

Implementation underway:

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
Runtime sources must be frozen before source-pinned GPU probe launch. GPU probe
stages have bounded timeouts and save reports independently. No GPU run started.
