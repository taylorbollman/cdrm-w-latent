# Held-out evaluation between live captured-DDP updates

2026-09-29. A bounded functionality supplement, independent of the unresolved
BF16 comparisons. Existing model, captured training, lifecycle and checkpoint
sources remain unchanged. Root alone launches GPU stages after source freeze.

Run two fresh three-update tiny two-H100 jobs from exactly the same source
inventory, seed, training index and configuration. Reuse the guarded lifecycle
driver: FP32/math/eager native RT, NFR K4 with active NextLat, native vocabulary,
physical B2/T16 per rank, and global valid-token targets64/128/192. The first
job is the no-evaluation reference. The second inserts one evaluation after
update1 has completed optimizer/scheduler/counter mutation **and committed the
real data cursor**, before update2. All three training observation rows must
match the reference bitwise, including rank-local RNG; subsequent captured
updates2/3 are the decisive interference checks. Retain existing checkpoint,
stop/error and tracking behavior; do not introduce another training algorithm.

Prepare a small JSON fixture on CPU from the already verified complete-document
corpus, choosing two deterministic dev documents of at least16 tokens. Their
source/document identities and actual token prefixes are pinned. Use one row
per rank:16 valid tokens on rank0 and9 on rank1, each right-padded toT16. This
is25 real tokens and23 within-row CE targets, no intra-row document boundary,
no fabricated EOS, and no overlap with the training split. Both jobs load and
verify exactly the same fixture; no evaluation selection follows the results.

At the completed boundary, retain the training CUDA graphs, gradient storage
and owned input/loss buffers. Run eager `no_grad` evaluation through the canonical
unwrapped model on both ranks; temporarily set all modules to evaluation mode,
disable feedback jitter explicitly, supply no noise, and preserve K4 and native
RT selection. Use the existing full-vocabulary CE arithmetic on the **final
pass only**. All-reduce CE sum and target count; report local elapsed times and
their maximum after device synchronization. This is a functionality statistic,
not the multi-pass combined training objective or a selected production metric.
The NextLat predictor is unused during this CE-only evaluation. The installed
W&B client accumulates explicit-step log calls with commit=False by default;
`dev/functionality/*` and the following training metrics share update1. Both
carry the explicit `update` axis; no custom tracking wrapper is required.

Require exact pre/post model/Adam/scheduler/counters/cursor/RNG, all module modes,
parameter trainability and ownership, gradient values and storage pointers,
owned graph input/loss values and pointers, and graph/DDP identities. Restore
modes and rank-local RNG in `finally`, including named generators. Never
enumerate the peer GPU's RNG. Validate the existing runner before and after
evaluation. Do not mutate graph-owned inputs or clear/reassign gradients.

Local forwards contain no collectives; ordinary local errors are coordinated
before metric reduction. Unknown collective/CUDA failure retains the existing
external teardown policy. The adapter only observes the existing constructor
and boundary callbacks and wraps the host update callback; it restores all
module-local bindings and releases observed graph owners before process-group
teardown. Capture no global `torch` monkeypatches. Reports keep the fixture,
source snapshots, exact controls and W&B metrics; root retains them in GCS.

CPU tests exercise actual tiny-model final-pass CE, no-gradient/no-jitter mode,
state and pointer preservation, exceptional cleanup, an exact subsequent real
update, and callback timing/restoration. Existing coordinated-error/Gloo tests
cover the reused coordination primitive. The actual two-GPU jobs each have
an external180-second bound. This does not qualify pretrained evaluation memory,
BF16, H200, cross-document held-out metrics, online inference, or production
training quality; it checks that an ordinary evaluation interruption leaves
the captured training trajectory unchanged.
