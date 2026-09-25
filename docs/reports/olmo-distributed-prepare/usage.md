# Distributed preparation usage and scope

`ObjectiveForwardAdapter(model)` exposes the canonical FBT/NextLat objective
through `forward`. Supply the independently summed global CE/latent/KL counts:

```python
from cdrm.pretrained.distributed_training import (
    ObjectiveForwardAdapter, sum_objective_counts,
)

adapter = ObjectiveForwardAdapter(model)
counts = sum_objective_counts([model.counts(batch) for batch in microbatches])
model.zero_grad(set_to_none=True)
for batch in microbatches:
    result = adapter(batch, global_counts=counts, world_size=1,
                     backbone_kwargs={"mode": mode})
    result["objective"].backward()
```

Preserve the caller's precision/autocast policy. For a future real DDP runner,
all-reduce counts over ranks before forward, use the process-group world size,
and invoke the **DDP wrapper's forward**, not the underlying loss method. Default
gradient averaging cancels the adapter's world-size factor. Clip after reduction.
The application must establish rank agreement on objective coefficients and
coordinate failure handling; this helper does not perform collectives.

The adapter owns the original model once, without copying or retieing parameters.
Build the canonical optimizer and checkpoints against `adapter.model` to preserve
names and existing format. Runtime modes, backend flags and FBT gamma remain
configuration, not new adapter buffers. Only the scalar objective is attached;
diagnostic losses are detached so they cannot mislead DDP unused-parameter
traversal. Masks, modes and zero coefficients can change parameter participation.
The adapter makes no static-graph claim.

The prepared actual-model one-H100 diagnostic is:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_distributed_prepare.py --case rt --output-dir .runtime/olmo-distributed-prepare/rt-b2-t512-01'
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_distributed_prepare.py --case combined --output-dir .runtime/olmo-distributed-prepare/combined-b2-t512-01'
```

Execute sequentially after any existing frozen-source queue has finished.
Defaults are B2/T512 and two two-microbatch fused-Adam updates. W&B uses
`taylorbollman/pretrained-fbt-rt-nextlat`, group `olmo-distributed-prepare`.
See `protocol.md` for exact gates and fixed accumulation budgets. Runtime and
source/protocol evidence are retained per create-only output directory; this
file is instructions, not a statement that the GPU checks have run.

CPU preparation tests validate objective algebra, parameter ownership, all eight
feature combinations, canonical clipped-Adam updates, FP32 concatenated-batch
references and failure gates. They do **not** validate DDP reducers, NCCL,
collective ordering, captured backward, sharding or two-GPU recovery. Those
remain the next hardware-dependent milestone. In particular, entirely empty
global updates are rejected; a locally empty objective retains the canonical
attached zero and never fabricates gradients for unused parameters.
