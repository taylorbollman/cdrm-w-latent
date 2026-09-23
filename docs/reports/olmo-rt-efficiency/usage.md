# Native RT efficiency switches

Stage A adds two independent execution options to
`OLMoTiledRTForCausalLM` and `tiled_recurrent_layer`:

```python
model = OLMoTiledRTForCausalLM(
    config,
    reuse_rope=True,
    kv_only_writes=True,
    # Retain the run's recorded attention, tile, precision and checkpoint options.
)
```

Both default to `False`. They do not change the native architecture, parameter
names/shapes, tied embedding/readout ownership, or checkpoint `state_dict`.
The unchanged sequential implementation remains the reference. These flags are
execution metadata, not fields in `OLMoConfig` or persistent model buffers:
callers must record and restore them explicitly for a run or resume.

| Harness arm | `reuse_rope` | `kv_only_writes` | Meaning |
| --- | --- | --- | --- |
| `control` | False | False | Existing on-demand RoPE and full permanent QKV |
| `rope` | True | False | Reuse native FP32 RoPE tables |
| `both` | True | True | Reuse tables and project only permanent K/V |

`reuse_rope` builds cosine/sine tables using the native FP32 operation order and
actual position IDs, including offsets and nonconsecutive coordinates. The
rotation remains split-half FP32 arithmetic with Q/K dtype restored afterward.
Dynamic forwarding builds tables once per stack invocation; prepared forwarding
shares its owned tables across layers, FBT passes and backward recomputation.
No mutable model-global cache, new positional scheme or fused RoPE kernel is
introduced. Ordinary blocks in the same stack can use the same tables.

`kv_only_writes` takes the K/V row view of the existing packed `att_proj.weight`
for permanent writes, their backward reconstruction and local source VJPs.
Temporary input-derived QKV is unchanged. Autograd returns a full packed
parameter gradient and accumulates the memory and temporary branches normally.
This changes matrix shapes and can change BF16 rounding, so equivalence to the
old writer is screened rather than assumed bitwise. Cache keys remain unrotated.
See [resource accounting](../../olmo-resource-accounting.md) for the analytic
`12BTD²` saving per RT call; RoPE pointwise savings are outside that matrix ledger.

## Prepared layouts, graphs and caches

Set flags before constructing a `PreparedFBTLayout` or `StaticFBTTraining` plan.
Each prepared layout owns its FP32 tables and guards their identity, storage,
version, shape, strides, dtype, device and gradient status. It also guards both
execution flags. Changing tables, positions, flags or ownership requires a new
layout and CUDA graph. Distinct outstanding layouts have distinct table storage.
Parameter value updates through the supported optimizer path remain allowed;
captures must reread those updated weights on replay.

Do not modify table contents or use unsafe `.data` edits. PyTorch version guards
cannot detect arbitrary in-place `.data` mutations. Direct table arguments on
low-level block helpers are trusted prepared metadata and must correspond to the
supplied positions/configuration. The normal public model and prepared-layout
paths construct this metadata for the caller.

Attached online caches record both flags along with their existing execution
provenance. Changing either flag while reusing a cache is rejected. Restore the
saved configuration and rebuild caches/graphs when changing execution strategy;
loading the same weight tensors does not alone recreate a training execution.
These additions do not provide graph save/resume, gradient accumulation,
distributed execution or general padded-graph readiness.

## Bounded harness

The frozen [protocol](protocol.md) and
[`olmo_rt_efficiency.py`](../../../scripts/olmo_rt_efficiency.py) define the
actual checkpoint, fixture, precision and timing scope. Commands below run from
the host project directory through the required GPU container. First confirm
the container location and GPU:

```bash
bash scripts/docker_shell.sh bash -lc 'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi'
```

Use new output directories; the harness rejects existing destinations. Run the
correctness comparisons before timing the corresponding candidate:

```bash
bash scripts/docker_shell.sh python scripts/olmo_rt_efficiency.py \
  --stage correctness --case rt --comparison control-rope --batch-size 8 \
  --output-dir .runtime/olmo-rt-efficiency/rt-b8-control-rope-check

bash scripts/docker_shell.sh python scripts/olmo_rt_efficiency.py \
  --stage correctness --case rt --comparison rope-both --batch-size 8 \
  --output-dir .runtime/olmo-rt-efficiency/rt-b8-rope-both-check

bash scripts/docker_shell.sh python scripts/olmo_rt_efficiency.py \
  --stage capacity --case rt --arm both --batch-size 64 --supervision full \
  --output-dir .runtime/olmo-rt-efficiency/rt-b64-full-both
```

- Correctness requires B8/T512 with the existing half-CE fixture. `ordinary`
  accepts only `control-rope`; `rt` and `combined` accept both comparisons.
  RoPE reuse requires exact same-state outputs/losses/gradients. KV-only uses
  the prospective numerical budgets in the protocol. Candidate eager/graph
  checks and three-update AdamW comparisons use exact equality.
- Capacity uses B64/T512 and defaults to full next-token CE. Run matched
  `control`, `rope` and `both` arms; ordinary has no RT writer and accepts only
  `control`/`rope`. `--supervision half` is an explicitly different workload;
  do not compare it as though it were the full-CE result.
- Optional B128 is limited to `--case rt`. This is a bounded operating point,
  not a maximum-batch search. `--profile` is restricted to RT B64 `control` or
  `both` and runs after timing; its trace must not be counted as a timing sample.
- The checkpoint defaults to `.runtime/olmo1b-step60000/artifacts`; `--artifacts`
  can point to another local copy of the required validated artifact bundle.

All cases use the original 16-layer OLMo-1B checkpoint at step60000, T512 and
CE2048/KL128. RT selects layers0/15 with alpha1; combined adds FBT K2 and
NextLat. Runtime settings are BF16 mixed, FP32 parameters/gradients/Adam,
ordinary activation checkpointing, deterministic PyTorch Flash SDPA, TF32 off,
autocast cache off, explicit per-invocation cast reuse, Triton historical RT
tiles and recomputed backward workspace. CUDA graphs capture
forward/loss/backward; clipping, AdamW and scheduler stay outside capture.

The harness logs online to W&B under entity `taylorbollman`, project
`pretrained-fbt-rt-nextlat`, group `olmo-rt-efficiency`. Each directory retains
source/protocol snapshots, runtime/parameter/resource records, actual optimizer
step counts and failure status. Timing reports separate complete-update input
tokens/sec from forward/loss/backward replay throughput and report setup versus
steady memory. The short runs do not produce useful trained checkpoints; retain
the evidence and reference the original pinned model weights.

This stage does not establish performance parity with the RT authors or resolve
the existing RT+FBT BF16 qualifications. The author-derived RoPE backend,
matched block/all-RT-stack comparison, new attention kernels, Q/K normalization
changes and quality training remain separate work after the Stage A review.

## Summarize and retain a completed queue

Use `scripts/olmo_rt_efficiency_report.py --runs <explicit run names>
--runtime-commit 3fd27e0` to verify the frozen sources and produce the summary
and throughput plots. Run reporting in the CPU-only container. The selection
can include failed reports; a completed summary does not turn failures into
passes or imply every planned case was attempted. Then use
`scripts/olmo_rt_efficiency_retain.py --dry-run` to inspect the evidence selection
and omit `--dry-run` to upload the verified small bundle to GCS. The retainer
requires the final GPU log and validates checkpoint reference, source/protocol
bytes and any declared compressed traces. No diagnostic weights are uploaded.
