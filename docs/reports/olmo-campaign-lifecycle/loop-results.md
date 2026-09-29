# Captured two-GPU lifecycle integration

The corrected opt-in loop passed fresh-process restart after both an ordinary
stop request and an injected rank-zero logging failure. Every executed resumed
update matched the uninterrupted reference **bitwise**, including model, Adam,
scheduler, raw gradients, loss metrics, input/noise fingerprints, counters,
committed data cursor and each rank's RNG state. The intentionally failed stage
kept failed status, published its completed update-1 checkpoint, and exited
nonzero normally; its fresh continuation then reproduced updates 2 and 3.

This adds an actual host-loop integration to the earlier
[CPU boundary-helper tests](results.md). It uses the frozen original loop through
the additive [guarded driver](guarded-loop-protocol.md). No old runner, core model,
optimizer mathematics or previously retained diagnostic was changed.

| Fresh stage | Executed updates | Outcome | Recorded stage time |
| --- | --- | --- | ---: |
| `guarded-reference-01` | 1, 2, 3 | Uninterrupted reference passed | 30.16 s |
| `guarded-stop-01` | 1 | Stop file honored after completed update; checkpoint published | 16.28 s |
| `guarded-resume-01` | 2, 3 | Both updates exactly match reference | 16.82 s |
| `guarded-log-failure-01` | 1 | Expected failure; completed checkpoint saved and retained | 16.30 s |
| `guarded-failure-resume-01` | 2, 3 | Both updates exactly match reference | 16.96 s |

Times include diagnostic hashes, setup and retention; the report timer ends
before final tracker/process-group teardown. They are not throughput numbers.
The separate launcher recorded normal exit statuses 0, 1 and 0 for the final
resume, deliberate failure and failure-resume stages. All returned within the
180-second inner / 240-second outer bounds, with no manual container teardown.

The model is the seeded tiny NFR architecture: two layers, width 32, four heads,
MLP width 64, with the real 50,280-token corpus vocabulary. It uses four feedback
passes, native RT in both layers, active NextLat losses, FP32/math attention,
eager RT tile calculations, captured DDP backward, and fused AdamW. Physical
batch size is two per rank at T16. The three logical updates use one, two and
three microbatches per rank, testing both local accumulation and synchronized
captured replay. Warmup/capture preserve the restored model/Adam/RNG/cursor
boundary on both ranks in every stage.

The reference sees 64, 128 and 192 input tokens in successive logical updates.
Its final counters are three optimizer updates, 384 input tokens, 360 CE targets,
360 latent pairs, 336 KL triples, 24 packed row presentations and 12 physical
microbatches across both ranks. The counter named `documents` is 24 here; it
counts packed rows, not 24 unique source documents. These first 384 tokens are
within a single long Books document. The T16 index audit separately verifies an
actual cross-document boundary later in that stream; the loop run itself does
not cover one. Across all five stages there were nine logical optimizer-update
executions, representing three unique data updates with deliberate replay.

Twelve immutable checkpoint directories were published across the stages, each
with state and manifest objects retained by create-only GCS uploads and verified
generation-specific download hashes. The independent local audit rehashed each
checkpoint's state/manifest and all **440 source/snapshot pairs** (88 per stage),
compared every recorded update against the reference, and checked the reported
retention controls: **534 checks passed**. It did not repeat cloud downloads or
GPU execution. Online W&B synced each stage, including failed status for the
deliberate error.

Two earlier failures remain part of the evidence. The first unguarded resume
failed its strict comparison. A follow-up observation run showed that all model,
Adam, gradient, input and metric values matched, with only rank-zero Python RNG
state differing by two draws. Cloud retention happened after checkpoint RNG
capture and consumed host randomness. The guarded driver preserves local RNG
around retention, including its failure path; fresh guarded trajectories now
match the complete RNG state too.

That unguarded failed stage also required manual container teardown after its
report and W&B failure synchronization. The old saved exception could retain the
completed stage frame, runner and captured NCCL graph while destroying the
process group. The new adapter retains a text traceback, releases those completed
frames and chained exceptions, and collects unreachable objects before teardown.
The corrected ordinary-error stage exited normally. This establishes the tested
failure path; it does not prove the exact internal cause of the earlier hang or
qualify recovery from a failed CUDA/NCCL operation.

Focused guarded-driver plus original loop scopes passed **28 CPU tests in
7.28 seconds**. Those overlap the earlier loop tests and are not an additional
28 independent behaviors. They include RNG preservation on successful/failed
retention, completed-frame lifetime, module-local adapter restoration, actual
Gloo coordination and checkpoint recovery.

The T16 SQLite index is retained separately at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/lifecycle-index-t16-01`.
Its archive preserves the original rich source-metadata report byte-for-byte,
with an additive verified envelope; the original preparation evidence was not
rewritten. Manifest SHA256 is
`dc7dae59199e75b5d801565f948eebc94d97a67a65e2fc1e13d3d1bdafc452f1`.

Remaining scope: this is a tiny FP32 two-H100 acceptance, not pretrained-scale
campaign training, BF16 or H200 qualification. It covers synchronized ordinary
Python logging failure and stop-file handling at a valid completed boundary.
It does not cover process death, hung callbacks, partial optimizer failure,
NCCL/CUDA faults, remote service outage, or performance at long contexts. The
general policy remains recovery from the last fully committed checkpoint.

Evidence is under `.runtime/olmo-campaign-lifecycle/`. Independent audit report:
`guarded-audit-01/report.json`, SHA256
`aee4b83c99e98c9c99d1d53686bd9ef3c66772423a960ba64e300dbc28ae662a`.
The audit records every stage's report pin, checkpoint pins and W&B URL.
`guarded-remaining-launcher.log` and `run_guarded_remaining.py` retain external
exit-status evidence. The reference W&B run is
[hp37jwsr](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/hp37jwsr).
