# Native RT large-batch profile interpretation

The profiles are separate untimed executions after the five canonical timed
updates. Each profiled run adds one real optimizer update (nine total, compared
with eight in unprofiled capacity probes), retains full trace bytes and hashes,
and checks finite state plus changed-weight graph/eager parity afterwards.
Profiler elapsed time must not replace synchronized unprofiled throughput.

## Accounting

The report helper verifies compressed traces and separates kernel events from
overlapping GPU user annotations before computing category shares. Raw
`key_averages` CPU and GPU records can share names; actual CPU phase scopes are
recovered from CPU `user_annotation` events in the trace. CPU phase durations
are inclusive and nonadditive, and do not assign device execution to layers.
The inherited `ordinary_step/` label describes the whole canonical update even
when it contains RT and FBT/NextLat.

Kernel categories are name-based descriptions. Generic matrix multiplication,
normalization, pointwise, casts and copies may belong to ordinary layers, RT or
losses. They cannot be labeled RT finish/writer time. The exact
`historical_recomputed_backward_kernel` name identifies only a historical RT
backward subset. The forward historical tile has the generic name `kernel` and
remains unclassified. Summed kernel durations are not full-step wall time or
measured hardware utilization.

## RT B192

Run `rt-compiled-native-b192-separated-02` passes all operational gates with
nine updates. Its unprofiled median complete update is 3.304515s, or 29,748.39
original input tokens/s. The three separate graph forward/loss/backward samples
have median 3.276580s. The first independent B192 run measured 29,743.65 tokens/s,
so these two run medians differ by 0.016%.

The full-step trace contains 145,616 kernel/device events totaling 3,254.629 ms.
Five GPU annotation intervals totaling 3,346.838 ms are excluded from that sum.
The forward/loss/backward-only trace contains 145,392 events totaling 3,245.621 ms
and has no excluded GPU annotations. These are profiled executions, so their
difference is not a controlled optimizer-cost subtraction.

| Full-step name category | Share of summed kernel time |
| --- | ---: |
| Matrix multiplication | 41.94% |
| Copy/cast | 12.06% |
| FP32 addition | 9.04% |
| Fill | 7.43% |
| FP32 multiplication | 5.83% |
| Concatenation | 4.00% |
| Other pointwise/reduction | 3.82% |
| Normalization | 3.30% |
| Compiled pointwise | 2.83% |
| Ordinary Flash attention names | 2.67% |
| Vocabulary loss names | 2.50% |
| Named native historical RT backward subset | 1.45% |
| Fused optimizer | 0.35% |

The historical backward subset contains 1,022 calls and 47.259 ms. This small
fraction must not be read as total RT cost. Matrix and elementwise work execute
throughout both ordinary and recurrent computations. Copies, fills and simple
arithmetic constitute a visible device-time opportunity despite CUDA graphs;
a later bounded compilation of a contiguous RT finish/writer region is sensible
if its scope is measured. These traces do not justify attributing all such work
to that region or promising a specific gain. The observed capture-memory
boundary remains a separate issue involving full-sequence backward workspace.

## Combined B128

Run `combined-compiled-native-b128-separated-02` passes all operational gates
with nine updates. Its unprofiled median complete update is 5.278578s, or
12,415.46 original input tokens/s; the forward/loss/backward-only median is
5.249121s. The first independent B128 run measured 12,410.94 tokens/s, a 0.036%
difference between run medians.

The full-step trace contains 178,792 kernel/device events totaling 5,216.302 ms.
Five overlapping GPU annotation intervals totaling 5,332.508 ms are excluded.
The forward/loss/backward-only trace contains 178,549 events totaling 5,208.274 ms.

| Full-step name category | Share of summed kernel time |
| --- | ---: |
| Matrix multiplication | 38.40% |
| Copy/cast | 17.84% |
| Fill | 9.91% |
| FP32 addition | 9.66% |
| FP32 multiplication | 4.87% |
| Vocabulary loss names | 3.29% |
| Other pointwise/reduction | 3.13% |
| Concatenation | 3.03% |
| Normalization | 2.65% |
| Compiled pointwise | 2.51% |
| Ordinary Flash attention names | 2.33% |
| Named native historical RT backward subset | 0.62% |
| Fused optimizer | 0.23% |

The historical RT subset again has 1,022 calls, totaling 32.375 ms. Combined
executes RT only in the feedback pass, not in its ordinary bootstrap. Additional
ordinary pass, latent predictor, KL/readout and fusion work contribute to the
larger full-step cost; generic kernel names do not partition that cost among
components. In both profiled configurations, named ordinary attention occupies
a small share and observed FA4 full-step gains are correspondingly small. This
supports closing the present FA4 investigation and taking distributed/recovery
readiness next. More precise RT leaf attribution or backward-memory reduction
can remain a separate bounded follow-up rather than blocking two-GPU testing.
