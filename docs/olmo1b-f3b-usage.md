# Native OLMo RT execution options (F3b)

Read the [protocol](reports/olmo1b-f3b/protocol.md),
[assessment](reports/olmo1b-f3b/assessment.md) and
[F3 graph usage](olmo1b-f3-usage.md). These are execution options on the original
OLMo checkpoint, not new model parameters or loss terms.

## Options and mathematical boundary

`OLMoTiledRTForCausalLM` accepts two independent opt-in flags:

```python
base = OLMoTiledRTForCausalLM(
    config,
    attention_backend="sdpa",
    ordinary_activation_checkpointing=True,
    cast_weights_once=True,
    tile_backend="triton",
)
```

Defaults remain `cast_weights_once=False, tile_backend="eager"`. The native
checkpoint/state-dict layout is unchanged. Set flags before preparing a static
training layout or creating a cache. Changing either invalidates an existing
prepared/captured plan or cache; recreate it at a clean boundary.

`cast_weights_once=True` reuses BF16 copies of the large dense weights within
each RT custom forward, avoiding a new conversion at every position. Initial
temporary-QKV projection still runs normally. Copies are local to that
invocation: graph replay runs the conversions again against updated FP32
parameters. Original FP32 weights remain the saved inputs to the existing
batched custom backward. This does not enable a global autocast weight cache.

`tile_backend="triton"` fuses the historical QK, stable online-softmax merge and
PV operation into one kernel. Its online state remains the FP32 unnormalized
numerator, maximum and denominator. It explicitly rounds QK output to BF16
before FP32 scaling, and PV weights/output to BF16 before FP32 state updates.
The temporary self contribution, native RoPE, permanent writes, dyadic schedule,
reverse recurrence and batched parameter-gradient ownership keep their existing
semantics. Fusion can change rounding versus cuBLAS; it is not a bitwise claim.

The current Triton helper supports BF16 Q/K/V, FP32 state, head dimensions
16/32/64/128 and historical rectangles with both sides <=256. Arbitrary strided
reads and key-valid masks are supported. No causal mask is applied inside a
rectangle whose historical relationship is already established by scheduling.
The caller uses eager math for FP32 attention, other projection/key/value dtypes,
larger rectangles or unsupported head dimensions. CPU execution with a requested
Triton RT backend is rejected. The standalone helper itself has no fallback.
At the tested T512/native head128, all 511 in-sequence historical rectangles
meet the fused shape contract. Larger contexts/prefixes need their own dispatch
and numerical coverage; do not infer all-fused execution from the option name.

The first call compiles shape specializations. Prepare/warm all used shapes
before CUDA capture. Ordinary blocks continue to use PyTorch Flash SDPA in the
validated runtime. This RT kernel is implemented in Triton; it does not call
the FA4 package. Backward still materializes full attention probability/error
matrices. Forward fusion does not establish linear backward memory.

## Bounded commands

GPU commands must run inside the project container. The following reproduce
bounded operational fixtures; use a fresh output directory each time:

```bash
bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f3b_validate.py \
  --variant triton --case combined --batch-size 8 --length 512 \
  --output-dir .runtime/olmo1b-step60000/f3b-my-correctness-check'

bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f3b_validate.py \
  --stage capacity --variant triton --case combined --batch-size 64 --length 512 \
  --output-dir .runtime/olmo1b-step60000/f3b-my-capacity-check'
```

Variants are `reference` (both old options), `cast_once` (casts only) and `triton`
(cast reuse plus fused tiles). Cases are RT alone or combined RT+FBT K2+NextLat;
both select RT layer0 only. The correctness runner compares same-state original
and candidate losses/gradients, then same-candidate eager/graph and complete
Adam state. Capacity uses three eager preparation updates, ten backward
warmups, capture and three timed graph updates. Full-step timing includes input
validation/staging, clipping, AdamW and scheduler; data generation is outside it.

`olmo_f3b_tile_probe.py` provides separate frozen-operand and raw-cotangent tiny
block/FP32 checks. Its helper timings include uncaptured submission overhead
and are not full-step or isolated-kernel timings. `olmo_f3b_profile.py` runs
an uninstrumented complete-step measurement before separate eager phase and
graph device profiles. CUDA annotation ranges include gaps and overlap child
kernels; do not add them to kernel durations or to CPU-attributed device totals.
The phase wrappers cover helper primals, not all subsequent local VJP work.

See [resource accounting](olmo-resource-accounting.md) for the parameter and
matrix-FLOP ledger. Those ranges explicitly omit pointwise/optimizer/launch
costs and do not predict wall time.

## FA4/CuTE environment

The installed FA4 wheel is compatible with the installed CuTE DSL. The historical
vendored copy shadows it by default. Select the installed wheel explicitly:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_fa4_smoke.py --output-dir .runtime/olmo1b-step60000/f3b-my-fa4-smoke'
```

Version/path/hash details and the unchanged historical default are recorded in
[the environment note](olmo-fa4-environment.md). The standalone GPU smoke covers
noncausal forward, Q/K/V gradients against FP32 and fixed-input forward capture.
It does not clear FA4 LSE-value numerics, captured backward, or an RT-specific
forward/backward integration. No reinstall was needed for this milestone.

Small evidence retention uses `olmo_f3b_retain.py`; `--dry-run` verifies and builds
the local archive without a cloud call. Exact per-run source snapshots preserve
historical implementations. Reuse original retained weights; these disposable
few-update fixtures are not trained checkpoints to resume for quality work.
