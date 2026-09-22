# F3c assessment: historical RT backward fusion

2026-09-22. Complete: seven GPU reports and432 scoped CPU tests pass.
[PR17](https://github.com/taylorbollman/cdrm-w-latent/pull/17).
Read the [protocol](protocol.md), [usage](../../olmo1b-f3c-usage.md) and
[F3b assessment](../olmo1b-f3b/assessment.md) for the preserved forward reference.

## Implementation and numerical result

The independent opt-in `backward_tile_backend="triton"` fuses the three matrix
products and FP32 error arithmetic for each reverse historical attention tile.
The original eager backward remains the default. Both comparison arms enable
F3b forward fusion and weight-cast reuse, isolating the backward change.
Checkpoint layout, native RoPE, normalization, losses, selected layers and
parameter counts are unchanged. The kernel returns fresh FP32 dK/dV increments;
the existing reverse schedule accumulates them and the batched VJP returns all
parameter gradients normally. BF16 operands and full-product result boundaries
remain explicit. No atomics or per-position parameter .grad writes are added.

The 48 frozen rectangles pass against the current BF16 arithmetic bitwise.
They cover irregular lengths 1..256, head widths 16/32/64/128, strided reads,
masked/zero probabilities and varied cotangent scales. All 12 tiny native blocks
have bitwise primary forward/cache and input/parameter/prefix-gradient equality,
with alpha 0/.37/1, lengths 9/17, attached masked prefixes, nonuniform positions
and raw hidden/exported-KV cotangents. Independent audit verifies 48 standalone
and 144 block fused-backward calls, 150 forward calls per primary arm, and no
fused backward calls in the control.

The independent FP64-boundary oracle's worst frozen global gradient relative L2
is 8.9769e-5, worst tensor L2 1.2633e-4 and maximum-error ratio 0.004133. Thirteen
stricter oracle tensor diagnostics flag identically for eager and candidate;
their declared engineering gates pass and all flags are retained. Maximum tiny
block gradient relative L2 versus full FP32 is 0.003464. FP32/original-BF16
comparisons remain explicitly separate diagnostics; the primary comparison
changes only the backward backend.

Actual OLMo RT+FBT K2+NextLat at B8/T512 passes all five checks. Its initial
per-pass losses remain bitwise equal, all 511 historical reverse tiles call the
kernel, and global gradient relative L2 versus old backward is 0.00139246.
Maximum tensor L2 is 0.00272989 and max coordinate error/reference tensor maximum
is 0.00694445. Same-candidate eager/graph losses and gradients are bitwise equal
with changed tokens, overwritten gradients and changed weights; three eager
versus three graph AdamW updates give exact model/moments/scheduler/counter
parity. This update equality is between execution modes of the candidate.
RT-only B8/T512 also passes all five checks: all initial reference/candidate
losses and gradients are bitwise equal, all 511 fused calls occur, and the
same-candidate graph/replay/complete-update checks are exact.

The prospective gradient gates remain global 1/64, per-tensor 1/32 and max 1/16;
no failing GPU threshold was widened. Native comparison additionally requires
bitwise initial losses, exact ownership and actual fused-call counts. Counted
calls exclude gradient-buffer initialization, which itself performs a backward.
Zero-reference gradients require exact zeros. Source and protocol hashes are
rechecked before success. Both tiny and native tests use raw participating
gradients, not just endpoint loss or finiteness.

## Profile and practical benefit

The fresh original-backward profile reproduces 10,858 input tokens/s for combined
B64/T512, close to the prior F3b 10,866 profile/10,871 capacity measurements.
Allocated peak is 40.84GiB. It passes bitwise observer neutrality for all active
gradients and losses, plus expected phase counts and finite complete updates.

Eager observer CPU-attributed device totals are 37.85ms for historical reverse
tiles, 27.87ms for local writer autograd VJPs, 40.91ms for local finish autograd VJPs,
26.85ms for the final batched parameter VJP and 13.70ms for reconstruction.
Projection/finish primals are labeled separately. These annotations describe
eager execution; they overlap child kernels and include different attribution
boundaries, so they are not graph-region wall times or values to add blindly.
Uninstrumented complete updates determine throughput. Helper CUDA-event elapsed
time also includes uncaptured host submission gaps.

Both large-batch capacity checks pass finite complete updates:

| T512 configuration | F3b forward-optimized reference | With fused backward | Change | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| RT layer 0, B128 | 26,101 input tokens/s | 26,480 | +1.45% | 42.10 GiB |
| RT layer 0 + FBT K2 + NextLat, B64 | 10,858 input tokens/s | 10,973 | +1.06% | 40.84 GiB |

The RT reference is the retained F3b capacity report; combined uses this
milestone's fresh reference profile measured before instrumentation. These
are warmed three-update medians, not randomized repeated speed estimates.
Both variants use ordinary checkpoints and CUDA graphs; full-step timing includes
input staging, clipping, AdamW and scheduler. Initial/changed-input/all-gradient
parity was checked at B8; larger batches establish finite updates and capacity.
Allocated peaks are unchanged. Setup peak reserved and current reserved memory
remain separately recorded: combined63.82/43.40GiB, RT68.30/44.31GiB.

The final candidate profile independently reproduces10,977 input tokens/s
(+1.10% versus the fresh reference). Actual CUDA kernel launches decrease from
128,637 to117,205 (8.89%), and summed kernel self-device duration from2.93613
to2.90897seconds. Exactly511 historical backward kernels execute, taking7.448ms
in total; all511 fused forward tiles remain present, taking4.527ms. Complete
wall time decreases from3.01785 to2.98506seconds. These are distinct measurements.

Direct Triton kernels are incompletely attributed under CPU observer ranges:
the fused reverse-tile span reports only1.565ms of attributed device time,
whereas its actual graph kernels total7.448ms. Therefore the37.85ms old eager
span versus1.565ms new span is not a reliable isolated-kernel speedup. CUDA
annotation spans also contain host gaps. Use the actual device kernel trace and
uninstrumented full-update measurements for the performance conclusion.

## Remaining memory work and coverage

The fused helper consumes already reconstructed probabilities. Backward still
materializes full probability/error arrays, computes final query/prefix adjoints
with the original batched math, and propagates local recurrent credit in reverse.
One FP32 B64/H16 matrix is 1 GiB at T512 and 16 GiB at T2048. This milestone therefore
does not establish linear backward memory.

The next bounded change should retain per-row normalizers and reconstruct
probabilities inside attention/gradient tiles. Preserve full-product BF16
rounding and temporary-self separation when replacing reconstruction, not just
the dK/dV kernel. Final dQ and attached-prefix gradients also need explicit
coverage. Keep the present backend as a reference while removing arrays in
reviewable stages. Local writer/finish work and the discarded permanent Q
projection remain separate optimization candidates.

Full-model measurements select RT layer 0 within the 16-layer OLMo stack.
Ordinary layers use deterministic Flash SDPA; RT forward and historical reverse
tiles use Triton. Standalone FA4 availability was established in F3b, but these
RT kernels do not call FA4. Broadcast-read metadata is supported; production
sliced strides are tested, while broadcast-stride GPU coverage remains untested.
More RT layers, longer/padded layouts, graph resume, accumulation and multi-GPU
still need their stated checks. Native Q/K math is unchanged. The pre-existing
AccumulateGrad stream warning remains visible for future distributed work and
does not block the measured exact graph/update checks. No learning result is
claimed and no quality run is queued.

## Durable completion

All seven reports pass78/78 declared gates. Two native correctness cases, two
capacity cases and two profiles execute36 physical optimizer updates:18 eager
and18 graph. The tile/block probe performs no optimizer steps. The historical
F3b RT B128 reference and backward-only initialization/warmup/capture/profile
calls are excluded from that count. No GPU or quality run is queued.

The432 scoped CPU tests comprise315 integration/kernel/dispatch/accounting,
12 observer-neutrality,15 tile-probe,18 native-validator,29 reporter and43
retainer tests. Independent review verified exact source/protocol/fixture hashes,
fused-call counts, preserved numerical gates, profile attribution and retained
scope. The baseline profile's earlier validator version has its exact snapshot;
all later runs use the frozen current runtime. Stricter diagnostics remain
visible rather than being relabeled passing.

See [results and plots](results.md), [machine-readable summary](summary.json)
and [test log](test-results.txt). Small evidence and exact sources are retained
under `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3c-rt-backward/`;
the [receipt](storage-receipt.json) records hashes and generations. Native weights
are referenced at their verified existing object. No disposable model/optimizer
weights are archived. This closes the historical backward-tile prototype and
leaves quadratic-intermediate removal as the next bounded milestone.
