# Bounded five-block tiled CDRM learning pilot

This pilot extends the preserved tiled CDRM validation into a matched
SEQ-5 FP32 / CDRM-5 tiled FP32 / CDRM-5 tiled BF16 comparison. The
[frozen protocol](reports/cdrm-tiled-pilot/protocol.md) defines the allocation,
fresh numerical-check roles and retained qualification. Model arithmetic and
the prior numerical contract remain unchanged.

The completed first cohort stops at1000 updates. New whole-model BF16 numerical
failures and observed learning divergence prompted review before the conditional
2500-update or second-seed extensions. BF16 remains experimental; see the
[results and disposition](reports/cdrm-tiled-pilot/results.md). The longer targets
described below are runner capabilities, not an active execution decision.

The first seed's two CDRM trajectories continue the original update-100
checkpoints. SEQ uses the corresponding original backbone with only the two
CDRM adapters removed. The optional second seed has a separately prepared
initialization and shares the same retained data order. The new runner records
parent and child identities; it does not relax the old runner's source guards.

All CUDA commands use the project container. For this lineage, use the retained
parent compiler-cache directory so the resumed runtime contract remains the
same. The parent cache archive is immutable; the raw mutable cache and the
resulting child archive are recorded separately.

```bash
cd /home/taylorbollman/cdrm-w-latent
run_cdrm_pilot_gpu() {
  CDRM_TORCHINDUCTOR_CACHE_DIR=/workspace/cdrm-w-latent/.runtime/cdrm-tiled-bf16/20260907T191403Z/inductor-cache \
    bash scripts/docker_shell.sh bash -lc \
    'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"' \
    cdrm-command "$@"
}
```

The runner selects `--arm seq-fp32`, `cdrm-fp32` or `cdrm-bf16`. Every invocation
requires the shared `--initial-checkpoint`, retained `--data-root`, a new
`--output-dir`, and the frozen `--protocol`. CDRM continuation also specifies
`--checkpoint` with its corresponding trained parent; a new SEQ or second-seed
trajectory starts without that option. Training targets are 1000 and 2500 total
updates; the inherited first100 updates are not repeated.

Keep evaluation every200 updates and retain checkpoints190,210,1000 and2500.
The epoch-boundary recovery invocation loads190, stops at210 and supplies
`--reference-final` pointing to the uninterrupted210 checkpoint. Development
ablation at1000/2500 sets the bridge gain to zero only for evaluation, then
restores it; this measures branch dependence, not benefit versus a separately
trained SEQ model.

Numerical confirmation uses `scripts/cdrm_tiled_pilot_numerical.py`, which
verifies the pilot checkpoint identity and assigned confirmation role before
invoking the unchanged `scripts/cdrm_tiled_validate.py`. It uses the selected
pilot CDRM checkpoint, the new confirmation dataset under
`.runtime/cdrm-tiled-pilot/20260907T212606Z/data/confirmation`, physical batch64
and the offset declared in `prepare/fixture-freeze.json`. The first seed's
FP32-/BF16-trained checkpoints use offsets0/64 at1000 and128/192 at2500; the
second seed uses256/320 and384/448. The final256 examples remain reserved.
Use the exact frozen contract and a new output/run name for every comparison.

For example, the BF16-trained first-seed checkpoint at1000 is checked with:

```bash
run_cdrm_pilot_gpu python scripts/cdrm_tiled_pilot_numerical.py \
  --preparation-report .runtime/cdrm-tiled-pilot/20260907T212606Z/prepare/report.json \
  --model-seed 7500 --updates 1000 --trajectory bf16 \
  --checkpoint .runtime/cdrm-tiled-pilot/20260907T212606Z/seed7500/cdrm-bf16-u1000/u1000.pt \
  --output-dir .runtime/cdrm-tiled-pilot/20260907T212606Z/num/seed7500-u1000-bf16 \
  --wandb-run-name NUM-seed7500-u1000-BF16-trajectory
```

Output directories must be new; the example records an actual retained role and
should not be rerun over its existing evidence. The wrapper checks the fixed
offset and data hash automatically, and preserves failed numerical screens.

The CPU-only `scripts/cdrm_tiled_pilot_report.py` audits explicit run, numerical,
and recovery reports, verifies the matched data order and source identities,
and produces combined CSV, PNG/SVG and W&B curves. Run it in the container with
`CDRM_DOCKER_GPUS=none`. Its `--require-complete-evidence` option checks that all
three arms, both assigned numerical roles and recovery are supplied at the
reported milestone; numerical clearance remains a separate reviewed judgment.

W&B entity is `taylorbollman`, project `cdrm-tiled-learning-pilot`, group
`20260907T212606Z`. Each source, checkpoint, data and configuration identity is
retained locally alongside curves and decisions; durable storage is
`gs://fast-chunks/cdrm-w-latent/cdrm-tiled-pilot/20260907T212606Z/`.
Completed execution and reviewed bounded suitability remain distinct from
passing every frozen numerical screen.
