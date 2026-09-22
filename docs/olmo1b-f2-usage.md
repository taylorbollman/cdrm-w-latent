# OLMo F2 health and physical-batch probes

Read the [protocol](reports/olmo1b-f2/protocol.md),
[results](reports/olmo1b-f2/results.md) and
[handoff](fbt-rt-nextlat-handoff.md). These are bounded operational probes on the
original OLMo-1B step60000 checkpoint, not quality runs. They use the existing
canonical objective and optimizer. RT selects layer0; FBT uses K2 unless named K3.

Enter the GPU container first:

```bash
bash scripts/docker_shell.sh
test -f /.dockerenv
pwd
nvidia-smi
```

The expected directory is /workspace/cdrm-w-latent. Never use a host-shell CUDA
or CPU fallback. All commands below must run inside the container. Choose new
output directories; existing ones are deliberately rejected. W&B is required
online under taylorbollman/pretrained-fbt-rt-nextlat and credentials remain in
the launcher environment. Do not copy environment values into reports.

```bash
python scripts/olmo_f2_health_capacity.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --output-dir .runtime/olmo1b-step60000/f2-health-NEW --stage health

python scripts/olmo_f2_health_capacity.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --output-dir .runtime/olmo1b-step60000/f2-checkpoint-NEW --stage checkpoint

python scripts/olmo_f2_health_capacity.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --output-dir .runtime/olmo1b-step60000/f2-capacity-NEW \
  --stage capacity --cases rt,combined --checkpointing both
```

Health records actual detached activations and reconstructed attention statistics
for B2/T32, then attached per-pass CE/latent/KL gradients. It performs no update.
An observer must cover only one forward and close before backward. Statistics
from reconstructed probabilities are not literal fused-attention intermediates.
The BF16 component-sum closure is descriptive; compare its meaning with the tiny
FP32 closure test before interpreting it as a derivative error.

Checkpoint checks compare one complete BF16 update with ordinary-block
checkpointing disabled/enabled, including exact final model, Adam moments,
scheduler and counters. The opt-in constructor/runtime flag is
ordinary_activation_checkpointing. Record it in run/resume configuration.
It is active only during grad-enabled training, rejects cache use there, and
leaves selected RT reconstruction untouched. It adds no parameters.

Capacity uses physical batches (no accumulation), T512, three warmup plus three
timed complete steps per cell. Inputs and weights change. The initial sequence
is B1,8,16,32,64,128,256,512; stop each arm above65GiB allocated or on CUDA OOM.
Peak reserved memory is reported separately. A capacity OOM ends that arm and
is recorded; nonfinite state/loss/gradients fail the diagnostic. These times
include clipping, AdamW and scheduler, and exclude fixture/report/W&B work.

The independent graph feasibility probe is narrower:

```bash
python scripts/olmo_f2_graph_probe.py \
  --output-dir .runtime/olmo1b-step60000/f2-graph-NEW \
  --length 32 --batch-size 1 --rt-layers 0 --warmup 10 --repeats 10
```

It captures only the unpadded native stack forward and a fixed hidden cotangent
backward, zeroing persistent gradients inside the graph. A disposable SGD update
outside capture tests changed weights. It does not capture CE, NextLat, FBT,
clipping or the optimizer. Check the report for the actual validated length,
batch, precision and backend; a passing short case does not clear larger shapes.
The full canonical combined training path remains ungraphed/uncompiled.

For the validated larger graph reference, use the separate explicit-backend
entrypoint. Its checks and budgets are the original graph probe's:

```bash
python scripts/olmo_f2_graph_backend_probe.py \
  --backend flash --deterministic \
  --output-dir .runtime/olmo1b-step60000/f2-graph-flash-det-NEW \
  --length 512 --batch-size 8 --rt-layers 0
```

This configures deterministic algorithms and cuBLAS workspace before CUDA
initialization. The installed cuDNN backend does not support this strict
deterministic mode. Automatic cuDNN and non-deterministic Flash are retained as
distinct qualified executions; do not impose the same near-bitwise criterion
without first measuring repeat variability. For localization, the separate
olmo_f2_graph_localize.py entrypoint records repeated eager/graph and stream
controls, with optional forced SDPA backends. Its status completed means that
controls ran, not that every numerical comparison passed.

Per-case progress reports survive interruption. No long diagnostic training or
large checkpoint is needed; original native weights are already retained in GCS.
Retain small final evidence with the F2 retainer after assessment; preserve failed
graph attempts explicitly alongside successful controls.
