# Execution and recovery notes

The current native entrypoint is `python -m scripts.olmo_campaign_execute`, under
`torchrun --standalone --nproc_per_node=2` in the project GPU container. Use the
existing Docker launcher, confirm `/.dockerenv`, working directory and successful
`nvidia-smi`, and set both NCCL asynchronous-error flags to0. Bound the external
launcher. Never run CUDA directly on the host or silently substitute CPU.

Inputs are a separately pinned declaration and its separately pinned prior CPU
resolution, selected arm, new persistent output/checkpoint directories and an
explicit observation choice (`lean` by default). The launcher re-resolves the
native authority before CUDA. Unsupported backend/evaluation/topology choices
fail. A tiny acceptance declaration is a separate schema with explicitly random
weights and FP32/eager RT; it is not a native launch shortcut.

`--stop-after` names an absolute completed-update boundary within the existing
finite plan. It does not change the schedule horizon or data identity. A stop
file or signal requests a checkpoint at the next safe boundary. Checkpoint
cadence is a boundary target; a slow update, preparation or transfer may exceed
it. Distinct segment output/checkpoint roots are required. Local pruning and
production retention automation are not implemented by this milestone.

For same-lineage recovery, pass `--resume <checkpoint-directory>` and
`--resume-manifest-sha256 <independent-pin>`. No reference report argument exists.
A run need not have completed successfully to have a valid earlier checkpoint.
Changed topology, sources, mode, data membership, startup, physical partition or
finite schedule is a new lineage, not an automatic resume. Observer choice and
segment stop do not change lineage. Original model and selected fusion-startup
assets are still required to construct the validated model before restoring the
full distributed checkpoint; data/index dependencies must also be available.

For cloud asset recovery, use the CPU-only container and
`python -m scripts.olmo_campaign_execution_restore --publication <pinned-latest-checkpoint.json>
--publication-sha256 <independent-pin> --output-dir <new-persistent-directory>`.
This command reads exactly the recorded object generations and streams the state
through size, SHA256 and MD5 checks to disk. It publishes the local manifest last.
Resume then points to `<output-dir>/checkpoint`. The asset helper does not load
model tensors, validate the target runtime, restore data or start training.
Partial failed downloads remain visibly partial and require a new destination.

`latest-checkpoint.json` is published only after verified cloud retention. Its
local `directory` field records where the producer saved the checkpoint; the
cloud restore helper never treats that old local path as authority. The receipt
binds both manifest and state generations and hashes. Preserve the receipt along
with the closed stage evidence and its independent GCS retention receipt.

During lean execution, metrics and clocks remain visible in W&B and local JSON,
while full model/Adam/gradient hashing is restricted to startup, checkpoint and
final integrity boundaries. Acceptance observation is for bounded verification.
Native gradients still undergo the accepted finite-value/norm checks, and all
checkpoints retain complete integrity checks. Short acceptance timings include
preparation, hashing, serialization and cloud I/O; do not report them as optimized
steady-state throughput.

Keep the numerical qualifications in the preceding fusion-startup report.
Successful BF16 same-precision recovery is not FP32-equivalence clearance. This
milestone deliberately defers per-pass evaluation and a selected production
mixture, actual pilot token budget, capacity choices and H200 performance checks.
