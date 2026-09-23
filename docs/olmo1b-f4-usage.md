# OLMo F4 training resource cards

Read [protocol](reports/olmo1b-f4/protocol.md) before extending this fixture.
This is a bounded functionality/resource benchmark, not quality training.
The eight switches are explicit; `combined` means RT + FBT + NextLat.
RT uses layers0 and15; FBT uses K2. This is an execution reference, not a
recommendation that later learning experiments use this placement.

## Container commands

From `/home/taylorbollman/cdrm-w-latent`, verify the container GPU:

```bash
bash scripts/docker_shell.sh bash -lc 'test -f /.dockerenv && pwd && nvidia-smi'
```

Choose a new output directory, never overwrite old evidence:

```bash
bash scripts/docker_shell.sh python scripts/olmo_f4_resources.py \
  --case rt-nextlat --stage correctness --batch-size 8 --length 512 \
  --output-dir .runtime/olmo1b-step60000/NEW-f4-rt-nextlat-correctness

bash scripts/docker_shell.sh python scripts/olmo_f4_resources.py \
  --case rt-nextlat --stage capacity --batch-size 64 --length 512 \
  --output-dir .runtime/olmo1b-step60000/NEW-f4-rt-nextlat-capacity
```

Cases: `ordinary`, `rt`, `nextlat`, `rt-nextlat`, `fbt`, `rt-fbt`,
`fbt-nextlat`, `combined`. Capacity supports B64 or B96 at T512. The optional
`--operator-trace` is restricted to correctness B1/T32, outside capacity timing.
Prior source/protocol snapshots must remain intact if future runtime code changes.

## Interpreting a card

- CUDA graphs capture forward/loss/backward. Timed complete updates include
  validation/copy, clipping, Adam and scheduler outside the graph.
- Each card uses three real preparation updates, ten backward warmups and
  three timed graph updates. This is a directional median, not a long-run
  sustained-throughput or stability guarantee.
- `setup_memory` includes graph preparation with initialized Adam;
  `steady_memory` resets peak statistics after capture. Whole-run capacity
  peaks are their maxima. Current reserved memory includes reusable allocator
  blocks and differs from current allocated live tensors.
- Input tokens and CE targets are unique data exposure. FBT pass work is
  reported separately; K2 does not double input-token throughput.
- Matrix FLOPs are analytic estimates with explicit components and assumptions.
  Profiler selected-op FLOPs omit fused/custom work and must not substitute
  for that estimate. CUDA-event/wall timings are measurements; FLOP rates
  derived from the estimate are not hardware utilization.
- Registered weights include inactive frozen fusion; active weights and
  optimizer-owned counts exclude it. NextLat predictor weights are
  training-only. RT adds no parameters.
- Training and finite-prefill/exact-online inference are different workloads.
  This table establishes no new prefill/decode timing, padding, graph recovery,
  accumulation or genuine multi-GPU claim. Native Q/K math remains unchanged.

The [execution plan](reports/olmo1b-f4/execution-plan.json) freezes the intended
matrix. F3e's retained long-context/K3/multiple-layer evidence is complementary;
its timings and update counts are not counted as new F4 measurements.
