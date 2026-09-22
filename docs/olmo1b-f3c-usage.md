# Native OLMo historical backward tiles (F3c)

Read the [protocol](reports/olmo1b-f3c/protocol.md),
[F3b execution options](olmo1b-f3b-usage.md) and
[current handoff](fbt-rt-nextlat-handoff.md). GPU validation is in progress.

## Configuration and exact scope

`OLMoTiledRTForCausalLM` and `tiled_recurrent_layer` accept
`backward_tile_backend="eager" | "triton"`. The default remains eager. This is
independent of forward `tile_backend` and `cast_weights_once`; it changes no
checkpoint parameters or loss. Set the choice before cache/layout/capture
preparation. Changing it invalidates an existing cache or static execution plan.

The first Triton backward helper fuses one dyadic historical dK/dV rectangle:
three BF16 matrix products with explicit BF16 result rounding, FP32 probability/
error arithmetic and FP32 output increments. The caller accumulates those
increments into its existing FP32 recurrent adjoints. Each reduction spans the
full query rectangle before rounding. No atomics or internal parameter .grad
writes are introduced. The public standalone helper is first-order only and
has no autograd implementation; native RT owns the custom VJP.

Supported inputs: BF16 query/stored values, FP32 probability/attention-cotangent/
row-dot arrays, head widths16/32/64/128 and both rectangle dimensions1..256.
Strided and broadcast reads are supported. Native RT uses eager math for other
precision or shapes; asking for a Triton RT backend on CPU is rejected. The
standalone helper rejects unsupported inputs instead of choosing a fallback.
Warm every required shape before graph capture. At the bounded T512 shape,
all511 in-sequence historical reverse rectangles are eligible.

Reconstruction of full probabilities, the final global error/query-gradient
calculation, prefix gradients, temporary-self derivatives, local writer/finish
VJPs and batched parameter VJPs retain the original implementation. This change
does not establish linear backward memory. Ordinary layers keep their separate
Flash SDPA path; the RT helper is Triton, not FA4.

## Bounded commands

Use fresh output directories and the project container:

```bash
bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f3c_tile_probe.py \
  --output-dir .runtime/olmo1b-step60000/f3c-my-tile-check'

bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f3c_validate.py \
  --case combined --variant triton --batch-size 8 --length 512 \
  --output-dir .runtime/olmo1b-step60000/f3c-my-native-check'

bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f3c_validate.py \
  --stage capacity --case combined --variant triton --batch-size 64 --length 512 \
  --output-dir .runtime/olmo1b-step60000/f3c-my-capacity-check'
```

In the F3c drivers, both variants enable F3b forward fusion and weight-cast reuse.
`reference` keeps the old backward; `triton` also fuses historical backward tiles.
Full-model cases use RT layer0; combined means FBT K2+RT+NextLat. Initial
same-state losses should be bitwise equal because forward is unchanged.
Gradient differences are measured against the frozen engineering screens;
same-candidate eager/graph and full Adam updates are checked separately.

`olmo_f3c_profile.py` takes the same case/variant plus physical batch32/64/128
at T512. It measures three complete graph updates after warmup before adding
any observer, then records separate eager annotations and a graph kernel trace.
Labels distinguish local autograd.grad calls from their projection/MLP primals,
historical reverse tiles, reconstruction and the final batched parameter VJP.
Annotations are inclusive, overlap children and include launch gaps; do not add
them to each other or to actual kernel durations. A same-state all-gradient/loss
check requires the observer to be bitwise neutral. Helper CUDA-event timings
include uncaptured host submission gaps and are not isolated kernel latency.

Few-update fixtures log to W&B and retain exact run-local sources/protocols.
They produce no trained checkpoint for quality evaluation. Existing native
weights are referenced during evidence retention rather than duplicated.
