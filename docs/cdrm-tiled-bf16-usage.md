# Five-block tiled CDRM

This profile runs ordinary blocks 0–3 once, retains early/late outputs after
blocks 1/3, applies one CDRM side scan sharing block 1's canonical weights, and
feeds the late bridge into ordinary block 4. See the
[equations and implementation plan](cdrm-tiled-bf16-plan.md).

The [five-block base configuration](../configs/cdrm/base_d128_5.json) preserves
D128/H16, MLP512/GELU, learned pre-normalization/QK normalization and ALiBi.
The prepared task is MAD selective copying V16/T256/K96 with physical B64.
The side gates remain epsilon 0.1, rho 1 and lambda 0.01.

| Backend and policy | Execution |
|---|---|
| `cdrm_backend=naive`, `cdrm_precision_policy=fp32` | Strict FP32 ordinary-autograd reference, including CPU oracle tests |
| `cdrm_backend=tiled`, `cdrm_precision_policy=fp32` | CUDA tiled scan, outer autocast disabled |
| `cdrm_backend=tiled`, `cdrm_precision_policy=bf16_fp32_state` | CUDA tiled scan inside explicit BF16 autocast, FP32 master parameters and residual state |

Keep `precision=None`, `recurrent_layers=[]` and the ordinary sequential block
type. Select mixed execution explicitly with the CDRM policy and outer autocast;
casting the model itself to BF16 is unsupported. The operational harness supplies
the intended context and computes native masked CE in FP32.

Tiled CDRM initially supports rho one, deep source, history reads, unpadded
independent examples, zero dropout and first derivatives. General rho, cached
decoding, packing, activation checkpointing and higher derivatives are outside
this implementation. Autograd's parameter and input gradient interfaces are
supported without a nested parameter `.backward()` side effect. The two adapters
remain the only new checkpoint parameter owners.

`output_cdrm_states=True` exposes true outer-graph candidate, proposed memory,
bridge, preview and temporary-QKV nodes. It does not manufacture graph-connected
views of the custom function's internal permanent records. Use independent
preview leaves and a loss on `hat_m` to diagnose temporal credit before bridge
scaling. Optional internal observers are for bounded diagnosis; remove them for
compiled confirmation and benchmarking.

## Container and retention

All GPU commands run through the project launcher. Use a dedicated persistent
compiler cache for a lineage and reuse it across training and resume processes:

```bash
cd /home/taylorbollman/cdrm-w-latent
run_cdrm_gpu() {
  CDRM_TORCHINDUCTOR_CACHE_DIR=/workspace/cdrm-w-latent/.runtime/cdrm-tiled-bf16/YOUR_LINEAGE/inductor-cache \
    bash scripts/docker_shell.sh bash -lc \
    'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"' \
    cdrm-command "$@"
}
```

Replace `YOUR_LINEAGE` consistently. Compiler-cache persistence supports exact
same-runtime recovery; it is not a guarantee across hardware/compiler changes.
Retain cache contents with the experiment if future exact continuation matters.
The launcher injects `.env`; never print credentials. Graphable runs use online
W&B under `taylorbollman`, with local reports and GCS retention alongside it.

## Prepare and execute a bounded operational pair

Preparation requires the explicitly selected CPU container and new destinations:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python scripts/cdrm_tiled_operational.py --mode prepare --data-root .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/data --output-dir .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/prepare'

run_cdrm_gpu python scripts/cdrm_tiled_operational.py \
  --mode train --precision fp32 \
  --data-root .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/data \
  --initial-checkpoint .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/prepare/init.pt \
  --output-dir .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/train-fp32 \
  --wandb-project cdrm-tiled-bf16-validation \
  --wandb-group YOUR_LINEAGE --wandb-run-name train-tiled-fp32
```

Run the BF16 arm with `--precision bf16` and a new output/run name. Complete the
prospective local and initialization numerical checks before training. Each arm
uses identical saved weights and data order, trains 100 updates and saves update
50/100 checkpoints. The original 200-epoch cosine schedule steps only on complete
epochs; this B64 half-epoch smoke run stays at its initial learning rate.

For recovery, use `--mode resume --precision bf16`, the same data and initial
checkpoint, `--checkpoint .../train-bf16/u0050.pt`, and
`--reference-final .../train-bf16/u0100.pt`, with a new output/run directory.
The harness verifies model, optimizer, RNG, data position and non-timing metrics.
`--mode repeat` provides a bounded repeated-batch learning check.
`--mode benchmark --warmup 5 --steps 20` measures complete warmed updates without
diagnostic monitors and rejects new compilation during measurement.

Prepared checkpoints include config/source/data identities. The three numerical
arms may change only the declared backend, CDRM precision and eager-helper
settings. Source drift fails explicitly; retain the exact source snapshot for
reproduction rather than loading a checkpoint into silently changed code.

## Numerical comparison

Freeze the [numerical contract](reports/cdrm-tiled-bf16/validation-contract.md),
source, initialization and data identities before fresh confirmation. First run
a B2 development diagnostic with `--scale-check`. The validator runs naive FP32,
tiled FP32 and tiled BF16 on the same weights, inputs and optimizer state. It
also probes the side scan on independent saved preview leaves using a fixed,
unnormalized incoming proposed-memory gradient.

```bash
run_cdrm_gpu python scripts/cdrm_tiled_validate.py \
  --checkpoint .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/prepare/init.pt \
  --data-root .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/data \
  --split dev --batch 2 --scale-check \
  --criteria docs/reports/cdrm-tiled-bf16/validation-contract.md \
  --output-dir .runtime/cdrm-tiled-bf16/YOUR_LINEAGE/diagnose-init-b2 \
  --wandb-project cdrm-tiled-bf16-validation \
  --wandb-group YOUR_LINEAGE --wandb-run-name NUM-init-B2
```

For fresh initialization confirmation, select `data/confirmation`, `--batch 64`
and `--example-offset 0`, and use a new output/run name. For the trained check,
load `train-fp32/u0050.pt` and use offset 64 on that same confirmation dataset.
The retained initial policy and criteria apply to both checks. The B2 development
diagnostic supplies the fixed-forward scaling check; the B64 confirmations use
the compiled production helpers with observers removed.

Execution completion does not mean every numerical screen passed: inspect
`machine_screens_pass`, the per-comparison results and the reviewed disposition.
Keep a failed confirmation and any follow-up controls together. The milestone
[results](reports/cdrm-tiled-bf16/results.md) explain the scope of the present
opt-in implementation and its remaining numerical qualifications.
