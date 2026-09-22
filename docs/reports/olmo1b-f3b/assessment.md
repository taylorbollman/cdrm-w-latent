# F3b assessment: native RT historical-tile fusion

2026-09-22. Complete: all12 GPU reports and307 scoped CPU tests pass.
[PR16](https://github.com/taylorbollman/cdrm-w-latent/pull/16).
Read the [results and plots](results.md), [protocol](protocol.md),
[usage](../../olmo1b-f3b-usage.md), [resource derivation](../../olmo-resource-accounting.md)
and [FA4 environment record](../../olmo-fa4-environment.md).

## Result and implementation

This milestone establishes a working fused native RT **forward historical tile**
and a separate weight-cast reuse option. The existing eager implementation
remains available and remains the default. Neither option changes the checkpoint
layout, recurrence placement, normalization, loss, or trainable parameter count.

One opt-in change reuses converted projection weights within each RT forward.
The original path repeatedly converts full matrices because the F3 runtime
disables the global autocast weight cache. Local conversion reuse avoids those
copies while rereading updated FP32 weights on every graph replay. The original
FP32 parameters still supply the existing batched custom VJP.

The second change uses a Triton kernel for historical QK, softmax-state merge
and PV. It keeps FP32 maximum/denominator/unnormalized numerator, explicit BF16
QK-result rounding before scaling, and BF16 PV operands/result before state
accumulation. Temporary-self attention, RoPE, permanent writes, dyadic scheduling
and reverse recurrent gradient propagation are retained. Ordinary layers use
deterministic PyTorch Flash SDPA. The RT tile kernel is Triton, not an FA4 call.

FA4 itself is now usable: the installed FA4 4.0.0b20 wheel matches the installed
CuTE 4.6.0.dev0 dependency. The incompatible vendored source was shadowing it.
An explicit launcher selector chooses the installed wheel while preserving the
historical default. No reinstall or image rebuild was needed. Standalone GPU
forward and Q/K/V gradients agree with FP32 within roughly0.2–0.24% relativeL2;
fixed-input forward capture is exact. This smoke does not validate FA4 LSE values
or establish an RT-specific FA4 backward.

## Numerical evidence

- **Cast reuse:** actual OLMo combined K2+NextLat at B1/T32 and B8/T512 is bitwise
  equal to the original path for per-pass losses and all participating gradients.
  Each also passes changed-token/weight replay and exact three-eager versus
  three-graph AdamW/model/moment/scheduler/counter parity.
- **Frozen fused tiles:** all48cases pass across irregular widths1–256,
  head dimensions16/32/64/128, strided reads, masks, empty/nonzero state and
  large score shifts. Maximum normalized-output relativeL2 versus eager is
  2.4273e-5; maxima are exact and denominator relativeL2 stays below6.15e-8.
- **Tiny native blocks:** all12cases pass, covering alpha0/.37/1, lengths9/17,
  masked attached prefixes, nonuniform positions and raw hidden/exported-KV
  cotangents. Candidate versus eager BF16 outputs/cache/gradients are bitwise
  equal. Maximum global gradient relativeL2 versus full FP32 is0.003464;
  maximum individual tensor relativeL2 is0.005596.
- **Actual OLMo B8/T512:** fused RT and combined K2+NextLat pass the predeclared
  numerical screens. RT global gradient relativeL2 versus original BF16 is
  0.003538; combined is0.010067. Their maximum individual tensor relativeL2
  values are0.005708 and0.014248, and maximum coordinate errors normalized by
  each reference tensor's maximum are0.008734 and0.025661. Maximum per-pass loss
  relative errors are0.0006862 and0.0001255. These are numerical differences,
  not bitwise baseline recovery or quality results.
- Both actual fused candidates pass exact same-candidate eager/graph loss and
  gradient replay, including changed tokens/weights and overwritten gradients,
  plus exact complete AdamW update parity. This update equality compares the
  candidate with itself across execution methods, not fused against original.

The mixed gradient screens remain global1/64, per-tensor1/32 and maximum1/16.
Same-candidate graph checks keep F3's stricter budgets. No failing GPU threshold
was widened. Independent final tile audit verified all source/protocol/fixture
hashes and observed every expected fused call (48standalone +150tiny-block calls).

The FP64-boundary oracle retains one maximum and three denominator strict-state
flags in **both** the old and fused implementation. CPU checks identified this
before GPU testing: FP32/FP64 reductions can land on opposite BF16 rounding ties.
The protocol keeps those state flags diagnostic while requiring the unchanged
strict candidate-versus-eager state screen, independent-oracle output/numerator
screens and matching finite patterns. All required gates pass.

## Profile, throughput and memory

The original combined B64/T512 profile reproduced 10,409 input tokens/s before
instrumentation. Its graph has substantial cast/copy/pointwise work; one BF16
copy-kernel category accounts for466.8ms over12,959calls. Those calls span the
whole graph, so they cannot all be attributed to RT weight conversion. Inspecting
the RT loop identified the repeated full-weight casts addressed here.

Profiler CUDA annotation spans contain launch gaps and overlap child kernels.
For example, eager `_add_tile` annotation spans sum to409.9ms, whereas its
CPU-attributed device total is43.4ms. Neither should be added to the actual CUDA
kernel sums. Helper annotations also exclude subsequent local autograd VJPs;
they are not a full backward partition. The separately measured uninstrumented
complete-step times determine throughput claims.

At combined B64/T512, cast reuse reaches10,717 input tokens/s versus10,409 for
the original graph path (about3.0%). Adding fused tiles reaches10,871/s, about
4.4% total and1.4% above cast reuse. Peak allocated memory remains40.84GiB.
These are bounded three-step medians with warmup, not a randomized speed study.
The small-helper uncaptured timings include host submission gaps; their larger
speedups are not end-to-end speedups.

RT B128/T512 reaches25,588 input tokens/s with cast reuse and26,101/s with both
changes, versus the previously retained F3 reference24,684/s: about3.7% and5.7%
total gains. Peak allocated memory remains42.10GiB. Combined fused B64 reserved
setup peak/current reserved are63.82/43.40GiB; RT B128 values are68.30/44.31GiB.
The historical RT reference uses the same fixture/runtime policy, but is not a
fresh randomized repetition. Native all-gradient correctness was checked atB8;
the larger batches establish finite complete updates and capacity.

The final optimized combined profile independently reproduces10,866 input
tokens/s. Actual graph CUDA kernel invocations fall from149,970 to128,637, and
summed kernel self-device duration from3.0628 to2.9350seconds. These measured
kernel sums are distinct from complete-step wall time and annotated spans.
The fused kernel runs511times, matching all historical rectangles atT512. Its
summed device duration is4.522ms, only0.154% of the graph kernel total, so further
forward-helper tuning alone has little full-step upside. The combined BF16-copy
family falls from15,517calls/476.0ms to12,962calls/402.8ms. Both optimizations
contribute, and the category spans the full graph; this is not RT-only attribution.

Both profiles retain the pre-existing AccumulateGrad stream-mismatch warning.
It is not introduced by Triton and did not prevent capture/replay or the separate
exact gradient/update checks. Keep it visible when adding DDP/multi-GPU support;
this milestone does not clear that future integration.

The resource ledger now records all eight RT/FBT/NextLat combinations analytically.
At B64/T512 with the actual half-position CE/KL fixture and ordinary checkpointing,
RT estimates294.9–316.5TFLOPs/update and combined644.5–689.3TFLOPs/update.
These count multiply-add as two operations and explicitly include RT reconstruction,
FBT passes, predictor, vocabulary losses and checkpointing. Pointwise operations,
optimizer work, memory movement and launches are excluded; implementation padding
can add hardware work. The range is an accounting estimate, not a wall-time model.
Twenty scoped accounting tests include actual tiny forward/backward matmul traces.

## Remaining critical RT work

Forward historical fusion is a useful validated boundary. It does **not** finish
the broader RT execution goal. The current backward still reconstructs full
probability/error matrices and processes local recurrent adjoints sequentially.
At B64/H16, one FP32 attention matrix alone is1GiB atT512 and16GiB atT2048.

The next bounded milestone should use the optimized profile to choose between
the following closely related changes, retaining the current VJP as reference:

1. Fuse dyadic historical dV and dK updates while preserving BF16 product
   boundaries and FP32 accumulated adjoints. Ordinary Flash backward cannot
   replace the whole recurrent backward: incoming attention cotangents become
   available progressively along the reverse recurrent calculation.
2. Replace global probability/error storage with row normalizers and recomputed
   attention tiles. This is the more important memory change for longer contexts.
3. Inspect local writer/finish VJP costs and the full QKV projection whose Q is
   discarded for permanent writes. A KV-only projection is a separate candidate.

Only RT layer0 is selected in full-model measurements; the other15layers are
ordinary. Larger RT-layer selections, longer/padded graph layouts, graph resume
and genuine multi-GPU execution still need their declared checks. Native Q/K
math remains unchanged. No learning win or all-layer RT readiness is claimed.

## Reproducibility and completion

All12 F3b reports pass their declared gates. The four native correctness cases,
four capacity cases and two profiles execute60 physical optimizer updates:
30 eager and30 graph. The standalone tile and FA4 probes perform no optimizer
updates. Warmup/capture/profile backwards are not optimizer steps; the historical
F3 RT B128 reference is excluded from F3b update counts. No learning run is queued.

The307 scoped CPU tests comprise259 kernel/execution/integration/accounting/
launcher tests,23 reporter tests and25 retainer tests. Independent review checked
numerical gates, gradients, actual fused dispatch, source lineage and accounting.
Each GPU report retains its own exact source snapshots, including earlier
successful versions; stricter diagnostic false flags are preserved. See
[test log](test-results.txt) and [machine-readable summary](summary.json).
Small evidence and exact sources are retained at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3b-rt-kernel/`;
the [receipt](storage-receipt.json) records object hashes and generations.
The existing native checkpoint is referenced, not duplicated. No trained weights
from these disposable few-update checks are archived.
