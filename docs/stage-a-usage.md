# Stage A: R3 correctness and GPU operations

This implementation covers the bounded first deliverable. It does not implement
CDRM, latent objectives, a synthetic research suite, or the proposed large LM
training runs. The supplied research documents are preserved in
[inputs](inputs/README.md), with local decisions in
[semantic-decisions.md](semantic-decisions.md).
The completed checks and measured H100 performance are in the
[Stage A results](reports/stage-a/results.md).

## Open the environment

On this bootstrapped H100 machine:

```bash
bash /home/taylorbollman/cdrm-w-latent/scripts/docker_shell.sh
```

Inside the container, `pwd` must report `/workspace/cdrm-w-latent`; run
`nvidia-smi` there. The home `bash /home/taylorbollman/start.sh` bootstrap also
works and now reuses the existing image. Rebuild explicitly after dependency or
editable-path changes. Project Python ignores persisted user-site packages so
they cannot silently shadow image pins; those files are preserved.

## Configure and convert

```python
from olmo.config import ModelConfig
from olmo.model import OLMo
from olmo.checkpoint_conversion import convert_model

# Start with a verified compatible SEQ ModelConfig. Full fixture settings live
# in configs/stage_a/full.json; tiny settings require no tokenizer or data.
seq = OLMo(config)
r3 = OLMo(config.update_with(
    recurrent_layers=[3], recurrent_backend="naive",
    recurrent_write_rho=0.0, reference_eager=True,
))
report = convert_model(seq, r3)
assert not report.missing and not report.unexpected and not report.new
r3.set_recurrent_write_rho(0.25)
```

The source model's unused rho must remain 1. `convert_model` checks model/block
semantics, all state keys, dtypes, and independent parameter storage before
copying. The report maps fused/split gradients. Reverse projection conversion is
a weights-only warm start and changes computation when recurrence is removed.
It does not convert optimizer moments.

`reference_eager=True` is the small FP32 oracle. GPU BF16 uses FP32 parameters
inside an explicit `torch.autocast("cuda", dtype=torch.bfloat16)` context.
Tiled recurrence supports rho=1 only and requires CUDA. Its training inputs must
carry gradients; frozen-prefix training, distributed execution, and combining
tiled recurrence with activation checkpointing are not validated.

## Numerical validation

Explicit CPU reference tests, launched from the host project directory:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'cd /workspace/cdrm-w-latent/recurrent-transformer && OMP_NUM_THREADS=1 CDRM_TEST_DEVICE=cpu python -m pytest -q tests/test_recurrent_foundation.py tests/test_checkpoint_conversion.py tests/config_test.py --disable-warnings'
```

GPU tests use the same launcher with GPU passthrough:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && cd recurrent-transformer && OMP_NUM_THREADS=1 CDRM_TEST_DEVICE=cuda python -m pytest -q tests/test_recurrent_foundation.py --disable-warnings'
```

Tests enforce structure, learned-norm conversion, mapped gradients, independent
functional recurrence, temporal credit, current-versus-persistent write order,
causality, ownership, strict legacy aliases, and supported recomputation. Tiled
tests include T=1/7/16, eager/compiled helpers, and 1/4 backward chunks. At T=1,
outputs and gradients are tested; shifted CE has no supervised pair.
BF16 gradient checks cover both helper modes and 1/4 backward chunks on the
small bias-free B2/T9/D32 fixture. Full-size runs below establish execution and
performance; they do not extend those numerical bounds to every model size.

Pristine upstream reproduction, failures, and environment details are retained
in [reports/stage-a/environment.md](reports/stage-a/environment.md). Current
fork tests do not retroactively change those results.

## OPS and whole-model benchmarking

All following Python commands run **inside the project container**:

```bash
python scripts/stage_a_ops.py --output-dir .runtime/stage-a/ops-new
python scripts/benchmark_model.py --preset tiny --topology r3 \
  --output .runtime/stage-a/tiny-r3-new.json
python scripts/benchmark_model.py --preset full --precision bf16 --topology r3 \
  --backend tiled --compiled-helpers --microbatch 1 --warmup 2 --steps 3 \
  --output .runtime/stage-a/full-r3-tiled-new.json
```

Use a new output directory/name when retaining a previous run. Start full-size
benchmarks at microbatch 1 and increase only according to measured headroom.
These commands require CUDA and never fall back to CPU. They do not download
data, contact experiment trackers, or launch the provisional research budget.

The benchmark includes shifted CE, backward, actual AdamW state/update, and
optional accumulation. It uses pre-generated synthetic inputs; reported timing
excludes data transfer/tokenization. Compilation warmup is separate, and reserved
memory includes the warmup allocator pool. It checks finite final parameters
outside timed intervals and writes failed/OOM outcomes as JSON.
Compiled entrypoints set a main-thread recompile limit of 64 and reject fallback
via compiler counters; the default limit of 8 caused partial eager fallback at
length 512. In this PyTorch build, configuration overrides are thread-local, so
autograd workers can still see defaults. OPS clears code caches between unrelated
fixture groups, preserves cumulative counters, and keeps each accumulation/resume
comparison intact. Reports record this policy and actual compiler counters.

OPS checks learnability on reused tiny batches, target-count-weighted
accumulation, exact interrupted/resumed trajectories, and fresh-optimizer stream
preservation. Bias-enabled FP32 gradients are compared separately from strict
bias-free optimizer equality because redundant key-normalization bias gradients
can amplify roundoff through Adam's epsilon. These limitations are in the OPS
reports. All smoke checkpoints are labeled **not research pretrained weights**.
The exact-resume and fresh-optimizer contracts here belong to the dedicated
smoke runner. The inherited corpus trainer has early safety/configuration
guards but is not yet an end-to-end validated LM continuation path.

## Next milestone

Use the measured timings/memory to bound a small SYN pilot, pin and test task
generators, and establish a real data/tokenizer/checkpoint plan. Preserve ordinary
SEQ as the parent for future CDRM comparisons. Do not run the entire proposed
LM/topology program automatically from these smoke-test scripts.
