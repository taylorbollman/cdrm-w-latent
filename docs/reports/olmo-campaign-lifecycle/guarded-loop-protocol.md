# Guarded tiny lifecycle acceptance

This is an additive correction to the frozen [loop protocol](loop-protocol.md).
The model, packed data, objective, capture, Adam, schedule, checkpoint format,
update counts and numerical comparisons are unchanged. Use
`python scripts/olmo_campaign_loop_guarded.py` under the same two-rank torchrun
launcher and CLI options. The new driver, tests and this protocol are added to
the source inventory; all new reference/stop/resume/error stages must use this
same inventory. Prior checkpoints cannot silently qualify the new lineage.

The initial fresh resume failed the strict equality gate. A subsequent frozen
source run without the reference gate retained the comparison rows: model,
Adam, inputs, gradients and metrics matched exactly, while only rank-zero
Python RNG state differed. Cloud retention occurs after the checkpoint captures
RNG; SDK host work consumed random draws that the fresh process did not replay.
The new driver wraps only retention in the existing local RNG preservation
context, including its error path. Publication order remains immutable local
checkpoint, verified cloud retention, then local latest pointer.

The failed initial stage also did not exit promptly after its failed report and
W&B finalization. A concrete lifetime risk was that its saved exception retained
the completed stage's frame, runner and captured NCCL graph while the process
group was being destroyed. The adapter saves the complete ordinary failure
traceback as text, clears completed frames and chained exception references,
collects unreachable closure cycles, and raises an equivalent LifecycleError
outside the original exception context. It also collects before the legacy
process-group destruction call through a module-local proxy. This is a targeted
lifetime guard; the exact NCCL hang cause is not established by CPU tests.
Unknown update/CUDA/NCCL failures keep their original exception and external
launcher teardown policy. No emergency checkpoint is added for uncertain state.

CPU acceptance checks actual Python/NumPy/PyTorch host RNG plus a mocked device
RNG through successful/failed retention; graph-like owner release with a retained
new exception before a fake destroy; textual diagnosis; unknown-error identity;
module binding restoration; and the expanded source inventory. These do not
substitute for fresh-process CUDA/NCCL execution.

GPU acceptance uses fresh guarded stages: uninterrupted updates 1–3; stop-file
after update 1; fresh resume through update 3 with complete bitwise comparison;
coordinated logging failure after a completed update with a saved boundary; and
fresh resume from that failure checkpoint. The latter stage must exit nonzero
promptly after preserving its failed report, and its continuation must reproduce
the uninterrupted reference. Retention is enabled so the RNG regression is
actually exercised. Use a bounded external timeout; no throughput, BF16,
pretrained-size, H200 or recovery-from-invalid-NCCL-update claim is made.

Pending runtime results are reported separately. A successful ordinary stop or
saved failure boundary alone does not qualify fresh restart or failed teardown.
