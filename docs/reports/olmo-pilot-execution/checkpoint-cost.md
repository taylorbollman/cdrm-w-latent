# Native checkpoint cost and practical cadence

The completed B32 native capacity run shows that retaining full model/Adam
checkpoints is a substantial part of current elapsed time. This is an I/O and
validation cost to account for before a longer pilot, not evidence of slow model
kernels or broken training. The functionality-first acceptance keeps its
verification intact; no checkpoint implementation changes are made here.

## Observed B32 regions

The independent `capacity-b32-01/report.json` records the following seconds.
All-rank regions use their recorded maximum rank; retention is rank-zero wall
time. Columns cover consecutive selected regions, not every operation involved
in durable publication.

| Completed update | Boundary observation | Save/write | Post-save state check | Local validation plus cloud retention/readback | Sum of selected regions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.90 | 12.11 | 6.84 | 103.31 | 129.17 |
| 4 | 19.93 | 30.71 | 19.88 | 297.03 | 367.55 |
| 8 | 19.52 | 29.84 | 19.96 | 314.39 | 383.71 |

The origin checkpoint has no populated Adam moments and is not representative
of later full-state checkpoint cost. The two populated-state checkpoints take
about 6.1 and 6.4 minutes in these selected regions. The stage's reported elapsed
time is 1,049.27 seconds, or **17.49 minutes**. Its selected checkpoint regions
sum to 880.43 seconds, approximately **84% of that short stage**. This is a
fixture-specific accounting comparison, not an estimate of a production run's
steady-state checkpoint fraction or tokens/s.

The sum excludes later publication, additional local retained-byte checks,
durable journal/latest-receipt writes and any pruning. It also does not isolate
network transfer from local hashing within the retention timer. The stage timer
excludes earlier CPU resolution and external process/container startup. Other future hardware/storage environments require their own observed costs.

The completed NFR12 run subsequently measured **388.60 and 393.07 seconds**
for these same selected regions at updates 4 and 8, before untimed publication.
That is about 6.5 minutes per populated-state checkpoint, consistent with the
same practical cadence estimate. Its complete stage took 1,613.63 seconds;
the initial preparation alone took 442.81 seconds. These costs remain separate
from graph replay and must not be interpreted as model-kernel throughput.

## Why 600 seconds is not publication every ten minutes

In [`run_loop`](../../../scripts/olmo_campaign_loop.py), `checkpoint()` executes
save, verified retention and publication before resetting `last_save_time`.
The next time-based checkpoint becomes due after 600 more wall-clock seconds,
checked at a completed optimizer boundary. This interval also includes any
evaluation or host work that happens between those boundaries.

For orientation, ignoring boundary overshoot and untimed publication work:

\[
T_{\mathrm{between\ publications}}\approx600\;\mathrm{s}+T_{\mathrm{checkpoint}}.
\]

Using the B32 populated-state measurements yields approximately 968–984 seconds,
or 16.1–16.4 minutes, before the excluded costs. Roughly 16–17 minutes or more is
a sensible planning estimate. It is not a hard bound: a long update, evaluation,
storage retry or publication step can lengthen it. The first setup interval and
explicit update/terminal checkpoints can have different timing.

For a short learning pilot, recommend retaining the **600-second time policy**
and saving at terminal/review update boundaries, rather than carrying the
capacity fixture's every-four-update rule into training. No larger time limit
is needed merely to target the user's 20–30 minute interruption window. The
measured selected costs imply about 38–39% checkpoint time in the simple
`checkpoint / (600 + checkpoint)` illustration. Budget approximately **40% or
more** to allow for unmeasured publication and variation in actual storage costs. This
budget is deliberately an operational estimate, not a new measured result.
The completed NFR selected checkpoint costs imply approximately 39–40% in that
same simplified illustration; publication and other excluded work still add cost.

## A focused later efficiency follow-up

The relevant paths are:

- [`olmo_campaign_ssd_engine.run_segment`](../../../scripts/olmo_campaign_ssd_engine.py): boundary hashes, distributed save, state-preservation checks, local validation, retention and publication.
- [`save_distributed_checkpoint`](../../../cdrm/pretrained/distributed_checkpoint.py): committed model/Adam/scheduler/RNG/cursor serialization and state-file digest.
- [`retain_checkpoint`](../../../scripts/olmo_campaign_loop_run.py): local file digests and state-then-manifest retention.
- [`upload_verified`](../../../scripts/olmo_two_gpu_retain.py): create-only upload followed by full exact-generation download and SHA256 verification.
- [`SSDCheckpointStorage`](../../../scripts/olmo_campaign_ssd_storage.py) and its [ordered metadata adapter](../../../scripts/olmo_pilot_execution_storage.py): ownership checks, verified publication, durable receipts/journal and bounded pruning.

Repeated local hashes may offer savings, but they are not the only cause.
Current durability verification uploads the large state and downloads its
complete pinned bytes again before treating the checkpoint as retained. A
later bounded investigation should first time serialization/copies, local
hashing, upload, full readback and publication separately. Then choose a narrow
change that preserves exact byte/generation authority and interruption safety.
Do not remove full readback or weaken the publication rule merely to improve a
throughput number.

The first short pilot can use the accepted implementation with these costs
budgeted. A longer campaign makes focused checkpoint I/O work worthwhile before
committing substantial GPU-hours. This note neither launches that campaign nor
claims total training throughput from the capacity run's selected compute
regions.

## Evidence

Independent summary:
`.runtime/olmo-pilot-execution/capacity-b32-01/report.json`, SHA256
`14e8a1fa9a4c3ea2a8b8d09f8ab118a2f4adbf98ab91df47c2cd6773cf47fb0b`.

Underlying completed B32 report SHA256:
`432bf8a3f6f09c9879523051f05ce08b8298b6cd90acd82a2ebf2c986c15ccfd`.
The summary retains its input report, declaration, resolved plan and independent
analysis source snapshot. Region sums in this note were recomputed directly
from that summary. No GPU work, network transfer or timing rerun was needed.

NFR12 costs use the completed `capacity-b32-nfr12-01/report.json` summary,
SHA256 `41de9b41f88a5307068684a36a464f41e7bc3f5f0ed3df01d4ecdf2f88b8766b`.
The final combined capacity summary supplies the same independently pinned
NFR report alongside the additional B64 measurement.
