# Native RT efficiency — Stage A results

Completed 2026-09-23. **Retain both opt-in improvements as the native reference
for the author-derived comparison.** They preserve the tested behavior and give
modest, repeatable throughput gains: **4.36% for two RT layers** and **2.26% for
RT+FBT+NextLat**. Ordinary OLMo is effectively unchanged (+0.31%). This does not
establish performance parity with the authors or resolve the existing broader
BF16/full-FP32 qualification.

Runtime/protocol: `3fd27e0`; report/retention helpers: `072155e`; base PR22 merge:
`70650a8`. See [protocol](protocol.md), [usage](usage.md), [summary](summary.json),
[profile audit](profile-audit.md), and [throughput PDF](throughput.pdf).
The summary explicitly selects every run, with raw report hashes and W&B URLs.
Online tracking: [pretrained-fbt-rt-nextlat](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat),
group `olmo-rt-efficiency`.

## What changed

`reuse_rope=True` prepares native FP32 cosine/sine tables once per dynamic
stack invocation or immutable prepared layout. Prepared execution shares them
across ordinary/RT layers, FBT passes and backward recomputation. Split-half
rotation, FP32 operation order and Q/K dtype restoration are unchanged. No
model-global mutable cache or new checkpoint buffer is introduced.

`kv_only_writes=True` uses the K/V rows of the existing packed QKV weight for
permanent memory writes, their backward reconstruction and local source VJPs.
Temporary input QKV remains full. Parameter identity, names/shapes, optimizer
ownership and unrotated cache semantics stay intact. The memory branch's packed
gradient has zero Q rows and accumulates normally with the temporary branch.
Smaller GEMMs can change BF16 rounding; they are not asserted bitwise equivalent.

The flags are independent and default off. Historical execution remains
available. Record the flags explicitly for any new run or resume and rebuild
prepared layouts/graphs/caches after changing them. No attention kernel, loss
semantics, Q/K normalization, architecture or production default changed.

## Correctness

**406 scoped runtime/accounting tests plus 14 retention tests pass.** They cover
RoPE offsets/irregular positions and FP32/BF16 gradients; cache/prefix cotangents;
packed gradient ownership; frozen/shared calls; dyadic boundaries and fractional
recurrence; both switches together; static guards; and existing checkpoint,
multi-layer and loss integration. Executed `mm`/`bmm` counts independently verify
the revised matrix ledger. Logs: [runtime](test-results.txt),
[retention](retention-test-results.txt).

Five native B8/T512 comparisons pass **25/25 gates**, including three eager
versus three graph AdamW updates in each report. Exact checks include changed
tokens, repeated gradient overwriting and changed weights, with identical model,
optimizer moments, schedule and counters. All candidate outputs/losses are
bitwise equal to their same-state reference in these comparisons.

| Comparison | Case | Global gradient relative L2 | Worst tensor L2 | Worst normalized max error |
| --- | --- | ---: | ---: | ---: |
| Original → RoPE reuse | Ordinary / RT / combined | 0 | 0 | 0 |
| RoPE reuse → both | RT layers0/15 | 0.3026% | 0.5929% | 1.8987% |
| RoPE reuse → both | RT+FBT K2+NextLat | 0.6543% | 1.0279% | 2.1564% |

The unchanged gradient screens are 1.5625% globally, 3.125% per-tensor L2 and
6.25% normalized maximum error. They are engineering screens, not paper-derived
limits. The K/V differences appear in backward in these fixtures; forward
hidden states and each CE/KL/latent loss remain exact. Small FP32 checks use the
independent CPU sequential oracle. Native GPU checks use the intended BF16
mixed runtime rather than substituting a stricter FP32 run.

## Matched throughput

Original OLMo-1B step60000 (~252B tokens), **16 layers, D2048, H16/head128,
SwiGLU8192 per branch**, tied50304 vocabulary, native nonaffine LayerNorm and
RoPE, no Q/K norm. RT selects layers0/15. Combined uses one ordinary bootstrap
and one feedback pass (K2), with RT in that feedback pass and horizon1 NextLat.

One H10080GB, physical B64/T512, full next-token CE2048 with KL128 and unchanged
auxiliary selections. BF16 mixed, FP32 parameters/gradients/Adam, ordinary
checkpointing, deterministic PyTorch Flash SDPA, RT Triton forward/backward,
cast reuse and recompute workspace. CUDA graphs cover forward/loss/backward;
clipping/AdamW/scheduler remain outside. Each fresh process performs three
preparation updates, ten backward warmups, then five timed complete updates.

Each arm was repeated in reverse order. Values below are the median of the two
run medians, in **input tokens/sec**. These are short execution benchmarks,
not sustained training rates or statistical significance claims.

| Case | Original control | RoPE reuse | Both | Both vs control |
| --- | ---: | ---: | ---: | ---: |
| Ordinary OLMo | 36,656 | 36,771 | — | RoPE +0.31% |
| Two RT layers | 21,487 | 22,216 | 22,425 | +4.36% |
| RT+FBT K2+NextLat | 10,948 | 11,144 | 11,195 | +2.26% |

RoPE supplies most of the gain: +3.39% RT and +1.79% combined. K/V-only adds
another +0.94% and +0.46%, respectively. Both repetitions show the same ordering.
The ordinary change is practically negligible. Its fresh full-CE control also
reproduces PR22's 36.63k/s baseline. Do not compare these full-CE rates directly
with older half-CE timings as if the objectives were identical.

Maximum observed **setup allocated / setup reserved GiB** across repetitions:

| Case | Control | Final candidate |
| --- | ---: | ---: |
| Ordinary (RoPE candidate) | 26.736 / 37.166 | 26.706 / 36.977 |
| Two RT layers | 32.249 / 48.045 | 32.076 / 47.361 |
| Combined | 39.095 / 60.932 | 38.923 / 60.314 |

The summary separately records steady allocator peaks/reservations and warmup+
capture time. Graph pool reuse means steady allocator counters are not a
substitute for setup memory requirements. No new B128/full-model or maximum-batch
search was run: prior B128 RT setup reservation was already tight. Larger
physical batches remain part of the isolated block/stack comparison.

## Parameters, work and profile interpretation

Unique active/deployable counts remain **1,176,764,416** for ordinary and RT.
Their experiment wrapper also holds 8,388,608 frozen, inactive fusion parameters;
resident count is therefore1,185,153,024. Combined has **1,267,879,936** trainable
parameters and **1,185,153,024** deployable parameters; NextLat's predictor is
training-only. Both optimizations add **zero parameters**.

K/V-only saves `12BTD²` matrix FLOPs per RT training invocation: **1.649 TFLOPs**
at B64/T512/D2048, or **3.299 TFLOPs** across the two selected RT calls. The
analytic RT training range changes321.63–341.83→318.33–338.53 TFLOPs/update;
combined684.74–728.02→681.44–724.72. These are matrix-work estimates, excluding
pointwise operations, hardware padding and optimizer work. RoPE reuse changes
pointwise work, not this ledger. See [derivation](../../olmo-resource-accounting.md).

The paired RT profiles show GPU events180,052→146,284, including kernel launches
179,615→145,849. All4,228 cosine/sine launches disappear from graph replay.
Historical RT tile and ordinary Flash durations remain essentially unchanged.
Dense projections and generic pointwise/memory work still dominate; shared GEMM
names cannot uniquely identify writer versus MLP work. The modest speedup is
consistent with removing unnecessary work, not a new faster attention kernel.

## Retention, limits and continuation

All **21 GPU reports pass41/41 gates**, with **158 physical optimizer updates**;
882 run/source pairs and all protocol snapshots match the frozen commit. This
includes the reverse-order repetitions. Exact source snapshots, reports, logs,
two compressed traces, plots, test evidence and the original checkpoint reference
are retained through the [storage receipt](storage-receipt.json). Diagnostic
few-update weights are disposable; the native checkpoint is already retained.

The earlier F4 RT+FBT coordinate miss and roughly18% BF16/full-FP32 initialization
gradient difference remain qualified. No broad precision clearance, long-run
quality result, all-RT performance claim, graph recovery/accumulation, padded
graph clearance or genuine multi-GPU result follows from this stage.

The user authorized proceeding directly after Stage A: next implement native
RoPE in a restricted author-derived reference/tiled backend, then compare
matched blocks and all-RT stacks, preserving useful author compilation/caching
and recording backward/precision differences. Use `reuse_rope=True` and
`kv_only_writes=True` as the improved native arm. The primary Stage A queue
ended with an idle GPU; no learning run was started.
