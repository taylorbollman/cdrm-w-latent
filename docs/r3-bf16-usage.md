# Running the bounded R3 BF16 profile

This guide reproduces the NUM/OPS protocol in [the milestone plan](r3-bf16-plan.md).
It does not declare numerical clearance or a performance benefit. Consult the
[evidence directory](reports/r3-bf16/) and individual execution reports for outcomes.
The selected configuration will be retained as
[mqar_t128_bf16.json](../configs/r3_mixed/mqar_t128_bf16.json), with its exploratory
record in [candidate-profile.json](../.runtime/r3-bf16/20260906T225438Z/candidate-profile.json).
Use FP32 by default: `bf16_fp32_state` is an opt-in experimental profile while
numerical flags remain unresolved. Its status and the
[frozen confirmatory criteria](reports/r3-bf16/confirmatory-criteria.md) remain part
of the evidence; this guide promises no speedup.

The operational scope is MQAR, R3 at block 3, D256/H4/12 blocks, rho1, four backward
MLP chunks, physical/global B64, T128, and no accumulation. The config's maximum
sequence length of 512 does not establish T512 training support. SEQ appears only
as the matched benchmark baseline. One short pair does not establish equal final
quality, other-task generalization, one-block RT support, or distributed training.

## Precision contract

Construct OLMo from a config containing `recurrent_precision_policy="bf16_fp32_state"`
and `precision=None`. Parameters and Adam moments stay FP32; projections/MLPs use
CUDA BF16 autocast, while residuals, normalization/CE reductions, sensitive
attention state and temporal adjoints remain FP32. BF16 projected records have
FP32 working attention views. Use no GradScaler and keep TF32 disabled and math
SDPA selected. The harness records effective dtypes and compiler activity.

The essential training boundary is:

```python
# cfg comes from the authoritative saved model configuration.
cfg.precision = None
cfg.recurrent_precision_policy = "bf16_fp32_state"
model = OLMo(cfg).to(device="cuda", dtype=torch.float32)
model.load_state_dict(parent_checkpoint["model"], strict=True)

optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(input_ids).logits
    loss_sum, answer_count = aligned_ce_sum(logits, aligned_labels)
    loss = loss_sum / answer_count
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
```

The runnable harness also sets the saved optimizer/schedule and runtime guards.
Configure the policy **before** constructing OLMo: each block owns a config copy,
so changing only `model.config` afterward does not update the executed block.
No call to `model.bfloat16()` is part of this profile. Backward stays outside
the outer autocast block; the custom backward restores its captured dense
recomputation precision internally. FP32 uses the default `legacy` policy with
outer autocast disabled.

## Container and paths

Run from `/home/taylorbollman/cdrm-w-latent` on the host. The helper below passes
arguments directly to the required container shell and checks its location/GPU
before every command. Execute jobs serially.

```bash
cd /home/taylorbollman/cdrm-w-latent
cdrm_gpu() {
  bash scripts/docker_shell.sh bash -lc \
    'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"' \
    bash "$@"
}

CDRM_BF16_BASE=.runtime/r3-bf16/20260906T225438Z
CDRM_BF16_RUN=$CDRM_BF16_BASE
CDRM_STAGE_B=.runtime/stage-b/20260906T190223Z
CDRM_R3_INIT=$CDRM_STAGE_B/runs/SYN-mqar-R3-seed0/SYN-mqar-R3-seed0-init.pt
CDRM_R3_FINAL=$CDRM_STAGE_B/runs/SYN-mqar-R3-seed0/SYN-mqar-R3-seed0-final.pt
CDRM_SEQ_INIT=$CDRM_STAGE_B/runs/SYN-mqar-SEQ-seed0/SYN-mqar-SEQ-seed0-init.pt
```

These commands show the retained output names. For a replay, set
`CDRM_BF16_RUN` to a **new** `.runtime/r3-bf16/<run-id>` directory; keep
`CDRM_BF16_BASE` pointing at the retained source snapshot. Existing outputs are
not overwritten. Parent Stage B checkpoints/fixtures remain immutable inputs.

## Targeted tests and NUM checks

CPU-only checks use a container with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 CDRM_TEST_DEVICE=cpu python -m pytest -q tests/test_r3_validation_metrics.py tests/test_stage_b_training.py'

CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 CDRM_TEST_DEVICE=cpu python -m pytest -q recurrent-transformer/tests/test_recurrent_foundation.py -k "fp32_attention_helpers_preserve_precision or fp32_inactive_mixed_policy"'
```

For GPU regressions, `CDRM_TEST_DEVICE=cuda` is required even when the container
has a GPU. Without it, this test module defaults to CPU and skips GPU cases.
These targeted tests check dtype/finite-update behavior and inactive FP32
regressions; they do not replace the actual-task numerical comparisons.

```bash
cdrm_gpu env CDRM_TEST_DEVICE=cuda python -m pytest -q \
  recurrent-transformer/tests/test_recurrent_foundation.py \
  -k 'fp32_inactive_mixed_policy or bf16_fp32_state_dtype_contract'

cdrm_gpu python scripts/r3_mixed_validate.py \
  --checkpoint "$CDRM_R3_INIT" --policy bf16_fp32_state --batch 64 --data-offset 1 \
  --baseline-source "$CDRM_BF16_BASE/source/pre-bf16-source.json" \
  --output-dir "$CDRM_BF16_RUN/confirm-init-b64"

cdrm_gpu python scripts/r3_mixed_validate.py \
  --checkpoint "$CDRM_R3_FINAL" --policy bf16_fp32_state --batch 64 --data-offset 1 \
  --baseline-source "$CDRM_BF16_BASE/source/pre-bf16-source.json" \
  --output-dir "$CDRM_BF16_RUN/confirm-trained-b64"

cdrm_gpu python scripts/r3_mixed_write_probe.py \
  --output-dir "$CDRM_BF16_RUN/candidate-v2-write-credit"
```

The CE triad compares naive FP32, naive BF16 and compiled tiled BF16 on the same
aligned answer mask and identical optimizer state. The fresh confirmatory
counters are 1 and 2001. The validator also supports `--scale-check` for a separately
named fixed-forward scaling run; retained exploratory B2 commands live in the
execution ledger. It does not reinterpret a failed historical screen as a pass.
Compiled tiled claims require observed graphs, no graph-break/fallback failure,
and strict recompile-limit handling.

The isolated write probe covers only block 3 at B2/T16/D32 with cotangent at the
last output. It records all 9 block parameter gradients and all 15 prior projected
K/V gradients. The earlier `candidate-v1-write-credit` result is invalid for
candidate confirmation because its tiled block retained the legacy policy;
see [the preserved invalidation note](../.runtime/r3-bf16/20260906T225438Z/candidate-v1-write-credit/invalidation.md).
The corrected probe asserts the executed block config and observed FP32 state/
adjoint dtypes. Its terminal unused-intermediate exception never excuses a
missing intended parameter gradient.

## Paired 100 updates and midpoint recovery

Both training commands import the same retained R3 initial weights into separate
OPS lineages with fresh Adam. They consume exactly the first 100 ordered Stage B
training batches, checked against the frozen arrays. The original 2000-update
schedule and 100-update warmup remain intact; `--stop 100` is not a shortened
schedule. Fixed development evaluation uses the first 256 IID examples at 0/50/100.
No test fixtures are evaluated.

```bash
cdrm_gpu python scripts/r3_mixed_operational.py --mode train --profile fp32 \
  --parent-checkpoint "$CDRM_R3_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/train-fp32"

cdrm_gpu python scripts/r3_mixed_operational.py --mode train --profile bf16 \
  --parent-checkpoint "$CDRM_R3_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/train-bf16"

cdrm_gpu python scripts/r3_mixed_operational.py --mode resume --profile bf16 \
  --parent-checkpoint "$CDRM_R3_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --checkpoint "$CDRM_BF16_RUN/train-bf16/u0050.pt" \
  --reference-final "$CDRM_BF16_RUN/train-bf16/u0100.pt" \
  --output-dir "$CDRM_BF16_RUN/resume-bf16"
```

Train writes `init.pt`, `u0050.pt`, `u0100.pt`, `resolved-config.json`,
`learning-curve.jsonl`, and `report.json`. Checkpoints include model/Adam/RNG,
source and parent identities, runtime math settings, precision policy, schedule,
completed-update count, next-data hash, and numerical history. Per-update
monitoring records CE, gradient/update norms, clipping, finite FP32 state and
block 3 projection-gradient signals. The separate write probe supplies isolated
persistent-credit evidence.

Resume restores the saved state exactly, starts at batch 50, and performs updates
51–100 in a fresh process. It repeats neither dev at 50 nor warmup updates. The
`recovery-comparison.json` compares final model/Adam/RNG/data/schedule and numerical
metrics; runtime timings and compiler histories are excluded. Tensor bit patterns
are checked, with numerical differences retained separately. Cold compilation
can differ from continuous execution; a completed run is not automatically an
exact-recovery pass.

R3 **update-zero exact resume remains disabled**. Importing retained initial
weights into a new diagnostic lineage does not claim recovery of the old Stage B
trajectory. New recovery accepts only its midpoint 50 checkpoint with matching
source, profile, runtime and data identities.

## Four matched benchmarks

Each command uses B64/T128, the same first training fixture, five warmup updates,
and 20 measured updates. SEQ requires its corresponding original paired SEQ
initialization. The benchmark includes actual masked CE, backward, clipping and
AdamW; prepared GPU inputs exclude data transfer from update timing.

```bash
cdrm_gpu python scripts/r3_mixed_operational.py --mode benchmark --topology seq --profile fp32 \
  --parent-checkpoint "$CDRM_SEQ_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/bench-seq-fp32"

cdrm_gpu python scripts/r3_mixed_operational.py --mode benchmark --topology seq --profile bf16 \
  --parent-checkpoint "$CDRM_SEQ_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/bench-seq-bf16"

cdrm_gpu python scripts/r3_mixed_operational.py --mode benchmark --topology r3 --profile fp32 \
  --parent-checkpoint "$CDRM_R3_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/bench-r3-fp32"

cdrm_gpu python scripts/r3_mixed_operational.py --mode benchmark --topology r3 --profile bf16 \
  --parent-checkpoint "$CDRM_R3_INIT" --fixtures-dir "$CDRM_STAGE_B/fixtures" \
  --output-dir "$CDRM_BF16_RUN/bench-r3-bf16"
```

Reports separate warmup/compilation from synchronized steady update times and
include tokens/s, startup and steady allocated/reserved peaks, optimizer bytes,
effective dtypes and compiler audits. New graphs during measured updates reject
the steady-state timing claim. Training-monitor copies/reductions are excluded
from this benchmark. Compare FP32/BF16 within each architecture and R3/SEQ at each
precision; FP32 parameters/moments mean total memory need not halve.

## Retention

Project `.runtime` outputs reside on persistent home storage through the project
bind mount. Local SSD compiler/download caches under `/mnt/localssd/cdrm_runtime`
are disposable and are not checkpoint storage. The archival prefix for this run
is `gs://fast-chunks/cdrm-w-latent/r3-bf16/20260906T225438Z/`. Original Stage B and
FP32-validation objects retain their separate lineages. The scripts do not upload
automatically; archive new source snapshots, fixtures, reports and checkpoints
with checksum/readback verification, and use the resulting storage manifest as
the retention evidence.
