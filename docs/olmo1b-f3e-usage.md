# Multi-layer native RT validation

Use the [F3e protocol](reports/olmo1b-f3e/protocol.md) for the fixed experiment
matrix and acceptance thresholds. This is an execution check, not a training or
layer-placement recommendation. The native checkpoint is original OLMo-1B
step60000, and indices below are zero-based.

`scripts/olmo_f3e_validate.py` accepts `--case rt|combined|combined-k3`,
`--layout single|adjacent2|spread2|spread4|all16`,
`--variant reference|recompute`, and `--stage correctness|capacity`.
Layouts select `(0)`, `(0,1)`, `(0,15)`, `(0,5,10,15)` or all sixteen layers.
The single layer is a fresh timing anchor; all16 is a separate stress case.
The user's primary research direction uses multiple selected layers and does
not yet prescribe which layout.

RT-only executes the selected blocks once. Combined K2 executes an ordinary
bootstrap, then one shared-weight feedback stack containing the selected RT
blocks, with NextLat training losses. K3 adds a second feedback stack and is
correctness-only here. Thus block calls are selected-layer count times the
number of feedback passes, not times the total pass count.

Run GPU work only through the project Docker launcher, after verifying the
container and GPU. Output directories must be new:

```bash
bash scripts/docker_shell.sh bash -lc 'test -f /.dockerenv && pwd && nvidia-smi'
bash scripts/docker_shell.sh python scripts/olmo_f3e_validate.py \
  --case combined --layout spread2 --variant recompute --stage correctness \
  --batch-size 1 --length 32 \
  --output-dir .runtime/olmo1b-step60000/f3e-local-smoke
```

Both variants use deterministic ordinary Flash attention, ordinary activation
checkpointing, BF16 mixed precision, cast reuse and Triton historical tiles.
`reference` keeps materialized backward attention; `recompute` uses the F3d
bounded workspace path. Native Q/K math, RoPE, weights and loss definitions are
unchanged. RT uses Triton, not the separately installed FA4/CuTE implementation.
Long forward rectangles above256 retain eager fallback.

Correctness compares every initial parameter gradient and loss against the
materialized path, observes actual per-layer tile dispatch outside capture,
then checks changed-input/weight graph replay and three-versus-three complete
AdamW updates. Capacity records three timed changed-batch graph updates after
three eager preparation updates and ten capture warmup backwards. Input copy,
validation, clipping, AdamW and scheduler remain outside the graph but inside
the reported full-step time. Three-step timing is directional.

Resource cards include actual prepared loss selections, architectural and
resident parameters, gradient participation, deployable parameters and
recompute-aware matrix arithmetic. Capacity cards inspect a live optimizer;
correctness cards explicitly leave optimizer ownership unavailable after the
parity helper releases it. Matrix estimates exclude elementwise operations,
optimizer work, padding, launches and communication. They are not measurements
of hardware FLOPs or utilization.

Each run snapshots all inventoried execution sources and the protocol, verifies
them again on completion and logs to `taylorbollman/pretrained-fbt-rt-nextlat`.
Keep failed attempts distinct from final passing coverage. Reports and retention
must preserve the exact source snapshots, checkpoint identity and W&B outcome.
No disposable few-update model replaces the pinned pretrained checkpoint.

CPU checks use `CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh ...`.
Actual multi-GPU execution, padded CUDA graphs, graph save/resume and graph
gradient accumulation remain separate milestones.
