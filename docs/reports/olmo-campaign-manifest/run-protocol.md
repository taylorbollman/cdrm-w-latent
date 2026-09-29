# B-only execution bridge from a resolved manifest

2026-09-29. Separately authorized, explicit operator invocation of a new bounded
adapter; the CPU resolver still grants no launch authorization or numerical
clearance. This is ordinary-B functionality and fresh-process restart acceptance,
not the general eight-arm launcher, a quality pilot or a throughput benchmark.
No previous helper, core file, test or protocol is modified or monkeypatched.

New files are `scripts/olmo_campaign_manifest_run.py`,
`tests/test_campaign_manifest_run.py` and this protocol. Freeze their source
inventory before root launches GPU work. Only root schedules GPU processes.
Use two GPU ranks in the required container, both NCCL asynchronous-error flags
set to zero, and an external **1,800-second bound per stage**, including retained
checkpoints. Ordinary failures are coordinated; an unknown rank/CUDA/NCCL failure
requires launcher teardown and recovery from the last complete retained checkpoint.

The operator supplies both manifest and resolved JSON with separate exact SHA256
pins. Before CUDA initialization each process validates the declarations and
repeats the CPU resolver's complete local byte/metadata resolution. Require exact
equality with the pinned resolved JSON, including source inventory, ownership
ledger, model and tokenizer authorities, source order, logical plan and schedule.
Unsupported values fail before model construction or capture.

Accepted fields are deliberately narrow: only B; original pinned OLMo-1B weights
with fresh Adam and every native parameter trainable; T1024, continuous-stream
packing; two ranks, physical B8 each; three updates with 16,384 valid inputs each;
BF16 mixed with FP32 masters/moments, deterministic ordinary Flash SDPA,
checkpointing, native RoPE, eager pointwise, and prepared CUDA graph execution.
The declared RT backend fields remain present but RT is disabled. No FBT,
NextLat, adapted startup, alternate precision/backend, evaluation or extra update
is accepted. Evaluation must be explicitly deferred. Original weights remain
the reconstructible origin; no redundant checkpoint0 is written.

Unlike the earlier hardcoded heterogeneous base-loop probe, this adapter builds
its own recipe and native model from the validated manifest. It uses existing
`CampaignObjective`, `CampaignDDPGraphTraining`, ownership, packed-data, token
schedule, complete-checkpoint and host-loop primitives without rebinding their
module globals. The actual pure packed plan, model/NextLat configurations,
parameter ownership and scheduler plan SHA must match the resolved evidence.
All recipe seeds and optimizer settings are consumed. The process/CPU-generator
seed is `jitter_seed + rank`; the explicit CUDA generator uses
`jitter_seed + world_size + rank`. No feedback noise is permitted in B.

Execution constants not exposed as campaign choices are explicit here and in
checkpoint configuration: `prepare(warmup=11)` performs 11 synchronized plus
nine accumulation warmup backwards; subsequent `capture(warmup=11)` adds two
captured backwards without repeating preparation. Thus setup performs 20 warmup
and two capture backwards per rank. Use static DDP with no buffer broadcast or bucket views
and a 25 MB bucket. These do not advance Adam, scheduler or committed cursor;
the complete boundary must match before and after preparation. The reference
starts with empty Adam state; resume loads populated Adam before capture, as in
the already accepted base/packed restart path. No new optimizer materialization
or allocation assumption is introduced.

The manifest must declare `checkpoint_every_updates=3`,
`keep_local_completed>=3`, and a positive completed-boundary clock cadence of
at most 600 seconds. The adapter additionally saves **checkpoint1** solely for
the restart diagnostic; checkpoint3 is the declared final cadence. A slow update
may trigger an extra time-based checkpoint2. Retention uses the existing verified
upload/readback primitive with RNG isolated, and publication follows verified
remote generations. The storage prefix must be an immutable stage beneath
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/`; append
`/reference` or `/resume` deterministically. The generic CPU example's different
prefix is rejected before CUDA. At most three checkpoints can be written, so
the declared local count accommodates every possible boundary. There is no
local deletion or pruning implementation; normally only checkpoint1 and3 are
written. Unknown failures cannot roll back already completed work.

The reference executes updates1–3 and retains complete checkpoint1 and3.
Fresh-process resume accepts only the exact reference checkpoint1, pinned by
checkpoint-manifest SHA and complete successful reference-report SHA. Checkpoint
configuration is phase independent and binds the manifest/resolved/source
identities, full runtime/backend/DDP contract, model ownership, source data,
schedule and seed derivation. Restore model, Adam, scheduler, counters, cursor
and all RNG streams before capture. Then updates2/3 must match the uninterrupted
reference bitwise in inputs, raw gradients, metrics and complete boundaries on
both ranks. Retain replay's final checkpoint3. Stop requests act only at safe
completed boundaries and report an incomplete segment rather than acceptance.

All actual updates must be finite, retain tied weights and dormant fusion exactly,
advance only the expected CE/token/data clocks, and keep replicas equal. Log
online to the manifest's W&B account/project/group with explicit optimizer-update
axis. These diagnostic hashes, checkpointing and cloud readbacks are intentionally
included in elapsed time; this is not optimized tokens/sec. Snapshot sources,
manifest and resolved plan, and verify unchanged input/source pins before success.

CPU tests cover strict declaration rejection, exact resolved authority, literal
plan/schedule consumption, actual tiny ordinary CE gradients and optimizer clocks,
strict new-lineage reference binding, unchanged frozen module globals and explicit
retention bounds. Independent data and lifecycle reviews precede the GPU stages.
The caller must explicitly invoke this bounded adapter; a resolver success alone
never starts it. No broader campaign or numerical clearance follows from success.
