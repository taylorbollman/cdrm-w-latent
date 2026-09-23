# F4 operator and matrix-accounting audit

The ordinary and combined B1/T32 traces agree with the source-derived dense
matrix ledger, ordinary Flash dispatch, and selected RT execution geometry.
They do **not** measure hardware FLOPs or complete-update throughput. Each
trace contains one untimed eager forward/loss/backward after the correctness
checks; optimizer, clipping, input loading and graph replay are outside it.
Use the separate uninstrumented capacity cards for full-update tokens/s.

Both reports passed their own bounded correctness gates. The combined trace
uses RT layers `(0,15)`, FBT K2 and NextLat. Its ordinary bootstrap has 16
ordinary blocks; the feedback stack has 14 ordinary blocks and two RT blocks.
This short combined result does not clear the separate RT+FBT B8/T512
coordinate-screen failure retained in [continuation-decision.md](continuation-decision.md).

## Observed operators and independent counts

Counts below come from the profiler's aggregated operator records and the
compressed trace's device kernel events. Nested CPU annotations are not
additional physical matrix multiplications.

| Observation | Ordinary | Combined | Source-derived expectation |
| --- | ---: | ---: | --- |
| Ordinary block invocations | 16 | 30 | One stack; or 16 bootstrap + 14 feedback |
| Ordinary Flash forward operators/kernels | 32 | 60 | Forward plus full checkpoint replay per ordinary block |
| Ordinary Flash backward operators | 16 | 30 | One backward per ordinary block |
| Each Flash backward device stage | 16 | 30 | `dot_do_o`, `dq_dk_dv_loop_seqk_parallel`, `convert_dq` |
| RT blocks | 0 | 2 | Selected layers occur only in the feedback stack |
| RT fused historical forward tiles | 0 | 62 | Two blocks times `T-1 = 31` dyadic rectangles |
| RT recomputed historical backward kernels | 0 | 62 | The same rectangles in the reverse scan |
| Aggregated `aten::mm` calls | 260 | 1,320 | Dense projections and selected vocabulary losses |
| Aggregated `aten::bmm` calls | 0 | 10 | Five completed-attention/query-reconstruction products per RT block |

The forward RT device kernel is named `Kernel` in this trace; its 62 calls
agree with the explicit per-layer dispatch counters and the forward Triton
source. The backward kernel is named `historical_recomputed_backward_kernel`.
At T32 all historical forward tiles are fused. These are the existing Triton
RT kernels alongside ordinary PyTorch Flash attention, not native FA4 RT.

For dense-call accounting, an ordinary block uses four forward projections,
four checkpoint-replay projections and eight projection VJPs: 16 matrix calls.
Ordinary therefore gives `16 * 16 + 4 CE = 260`.

An RT block gives 399 dense calls at T32: 129 in forward, two batched
projection reconstructions, 64 local writer forward/input-VJP calls, 192
local finish forward/input-VJP calls, three finish reconstructions, four
batched projection VJPs and five batched finish VJPs. Combined therefore gives
`30 * 16 + 2 * 399 + 6 fusion + 8 CE + 18 predictor + 10 KL = 1,320`.
The predictor runs once per union source position per pass, rather than once
separately for the latent and KL objectives.

The raw combined Chrome trace has 1,594 CPU `aten::mm` annotations. Exactly
274 are nested same-name annotations, accounting for the apparent excess.
Collapsing those nested annotations gives the aggregated 1,320 calls and
their matrix arithmetic below. Summing raw Chrome rows would double-count
28,454,158,336 FLOPs. There is no corresponding discrepancy in the ordinary
trace's 260 matrix calls.

## Arithmetic reconciliation

Multiply-add counts as two FLOPs. These are matrix-operation estimates from
tensor shapes, excluding instruction padding and other hardware details.

| Quantity | Ordinary | Combined |
| --- | ---: | ---: |
| Aggregated dense `mm` FLOPs | 288,064,798,720 | 664,515,444,736 |
| Source ledger, dense terms with full ordinary checkpoint replay | 288,064,798,720 | 664,515,444,736 |
| Aggregated `bmm` FLOPs | 0 | 41,943,040 |
| Profiler-recognized pointwise add/multiply FLOPs | 36,739,118 | 111,469,947 |
| Reported selected-operation total | 288,101,537,838 | 664,668,857,723 |
| Analytic matrix ledger range | 271,161,753,600–288,668,778,496 | 632,888,557,568–665,714,229,248 |
| Matrix work absent from profiler counts, using ledger upper attention estimate | 603,979,776 | 1,156,841,472 |

Dense arithmetic matches exactly. In these traces ordinary checkpointing
replays all four projections; the ledger also retains its prospective lower
case where checkpoint early stopping can avoid the last FF-down projection.
The observed full replay selects the upper dense estimate for this audit.

Ordinary's missing matrix work is the analytic Flash attention estimate:
134,217,728 forward + 335,544,320 backward + 134,217,728 checkpoint replay.
The profiler reports zero operation FLOPs for those fused Flash operators.

Combined's ten `bmm` calls account for RT completed-attention reconstruction
(16,777,216), final-query VJP (16,777,216), and query-side QK recomputation
(8,388,608). Its remaining uncounted matrix work consists of ordinary Flash
(1,132,462,080) and fused historical RT work (24,379,392): forward QK/PV,
reverse-history VJPs, and historical QK reconstruction. Their sum is
1,156,841,472, exactly the residual between the ledger's upper matrix estimate
and the profiler's `mm` plus `bmm` arithmetic.

The recognized pointwise counts are separate from the matrix ledger and are
themselves incomplete. Neither adding them to the matrix estimate nor using
the profiler's selected-operation total establishes complete hardware work.
Flash's causal/padded execution and tensor-core tile padding remain outside
this reconciliation. The recorded 3,872 ordinary and 19,223 combined CUDA
device events also include events such as copies; they are not all distinct
compute kernels and should not be treated as a hardware-operation count.

## Retained evidence

The runtime directories are under `.runtime/olmo1b-step60000/`. Their source
snapshots and exact compressed traces are included by the F4 retainer.

| Runtime directory | Trace bytes | Trace SHA256 |
| --- | ---: | --- |
| `f4-ordinary-correctness-b1-t32` | 766,602 | `0818a616252ae3ac255ad119de3a3a46133ee04715ac2723bbd4bc59328bf153` |
| `f4-combined-correctness-b1-t32` | 3,726,890 | `1913eb993376c6f839cba537d42563eb2018119f0a70d4080a7416fdf8b0955b` |

Both files are named `operator-trace.json.gz`; sizes and SHA256 digests were
checked against each completed `report.json`. The independent derivation
uses `resource_estimates.py`, `olmo_tiled.py`, `olmo_rt_memory.py`, the forward
and recomputed-backward RT kernels, and the static FBT/NextLat loss path. It
required no further GPU run and changed no runtime source or acceptance gate.
