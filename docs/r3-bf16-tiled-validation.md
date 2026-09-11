# Tiled R3 BF16 validation and recovery

See the [results](reports/r3-bf16-tiled-resolution/results.md) and
[frozen validation contract](reports/r3-bf16-tiled-resolution/validation-contract.md)
before interpreting a completed script as numerical clearance. The numerical
implementation remains `bf16_fp32_state`; BF16 remains opt-in. The measured
scope is MQAR R3, twelve blocks with recurrence at block 3, D256/H4, B64/T128,
rho 1, and the original normalization, ALiBi, loss and optimizer.

The project launcher now sets `TORCHINDUCTOR_CACHE_DIR` under the mounted XDG
cache. `CDRM_TORCHINDUCTOR_CACHE_DIR` overrides that container path. Keep one
compiler cache for an experiment's training and resume commands. The cache
contains compiled code/tuning decisions, separate from autocast's BF16 weight
cache. The recorded runtime, source and cache policy must match for recovery;
moving machines or deleting cache contents does not promise bitwise continuation.
Retain the compiler cache alongside checkpoints when exact future recovery is
important, since local SSDs are disposable.

All model commands must run inside the GPU container. From the project root:

```bash
cdrm_gpu() {
  bash scripts/docker_shell.sh bash -lc \
    'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"' \
    bash "$@"
}
```

Choose a **new output directory** for each invocation; retained runs are inputs.
For example, a four-arm actual-loss check with minimal retained-input hooks:

```bash
cdrm_gpu python scripts/r3_mixed_validate.py \
  --checkpoint .runtime/stage-b/20260906T190223Z/runs/SYN-mqar-R3-seed0/SYN-mqar-R3-seed0-init.pt \
  --policy bf16_fp32_state --batch 64 --data-offset 4096 \
  --include-tiled-fp32 --no-observers \
  --wandb-project r3-bf16-tiled-validation --wandb-group YOUR_NEW_LINEAGE \
  --wandb-run-name confirm-init-b64 --output-dir YOUR_NEW_OUTPUT
```

Counter 4096 was used for this milestone's confirmation and is a replay for a
future investigation. Select fresh confirmation fixtures prospectively.
The trained milestone fixture uses the final checkpoint and `--data-offset 2097`,
yielding counter 4097. `--capture-block-boundaries` supports localization;
`--scale-check` checks fixed-forward homogeneity. `--autocast-cache off` applies
globally across forward and replay for a diagnostic intervention. The deployed
tiled profile keeps it on. `--bf16-reduced-reduction off` is an available
diagnostic, not part of this milestone's cleared scope.

Evaluate retained tensors on CPU, with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'python scripts/r3_tiled_reference_report.py --case-dir YOUR_NUM_OUTPUT --output YOUR_NEW_ANALYSIS.json --criteria docs/reports/r3-bf16-tiled-resolution/validation-contract.md'
```

The evaluator retains old one-sided budgets, reference-centered screens, tails,
and optimizer metrics. Its `machine_screens_pass` is separate from the reviewed
engineering recommendation. `numerical_clearance` remains false in the automatic
analyzer because local correctness and operational interpretation are external
evidence. The trained milestone case still fails the frozen FP32 max/RMS screen.

The local probe accepts `--fixture-seed 9103` for a tiny independent fixture, or
`--source-case <captured tensors.pt> --checkpoint <matching checkpoint>` for
captured full-width inputs. Use `--compiled-check` to compare with execution
without diagnostic hooks. `scripts/r3_resolution_track.py` publishes completed
dense reports as W&B tables and graphs without rerunning a model.

For paired training, benchmarks and midpoint recovery, use the existing
[bounded operational commands](r3-bf16-usage.md), with fresh output directories
and these logging flags on every command:

```bash
--wandb-project r3-bf16-tiled-validation \
--wandb-group YOUR_NEW_LINEAGE --wandb-run-name DESCRIPTIVE_NAME
```

Train and resume with the same explicit compiler-cache directory when retaining
a dedicated experiment cache. For example, insert
`env TORCHINDUCTOR_CACHE_DIR=/workspace/cdrm-w-latent/.runtime/YOUR_NEW_LINEAGE/inductor-cache`
between `cdrm_gpu` and `python`. The controlled successful pair uses exactly
that pattern. W&B failures remain errors, and local records/checkpoints are
retained. Source identities are strict: historical checkpoints require their
recorded source snapshot rather than an arbitrary later working tree.
