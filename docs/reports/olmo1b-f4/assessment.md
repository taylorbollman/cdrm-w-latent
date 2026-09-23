# F4 training resource assessment

The common B64/T512 training resource matrix is complete. **RT+FBT retains
a failed numerical engineering screen**, with a separate bounded diagnosis;
resource coverage is not all-combinations numerical clearance. No model/kernel,
checkpoint, Q/K normalization or loss-math change was made. All sixteen B64/B96 capacity
checks completed; the GPU is idle and no further run is queued.

## Common comparison

One H100 80GB; RT at layers `(0,15)`; FBT K2; BF16 mixed, ordinary Flash,
checkpointing, CUDA graphs and Triton RT/recompute. Each timed complete update
includes copy/validation, clipping, Adam and scheduler. Three-update medians
are directional. These are full-backbone updates from the same native checkpoint.

| Configuration | Input tokens/s | Seconds/update | Peak allocated / reserved GiB | Matrix TFLOPs/update | Compute hours / 100M input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 31,113 | 1.053 | 26.74 / 37.17 | 281.79–304.87 | 0.893 |
| RT | 19,437 | 1.686 | 32.25 / 47.96 | 308.18–328.38 | 1.429 |
| NextLat | 24,026 | 1.364 | 28.22 / 39.23 | 314.90–337.99 | 1.156 |
| RT + NextLat | 16,444 | 1.993 | 33.73 / 50.51 | 341.29–361.49 | 1.689 |
| FBT | 15,798 | 2.074 | 32.16 / 48.54 | 565.23–611.39 | 1.758 |
| RT + FBT * | 12,136 | 2.700 | 37.61 / 58.37 | 591.62–634.90 | 2.289 |
| FBT + NextLat | 12,177 | 2.691 | 36.01 / 55.48 | 631.45–677.62 | 2.281 |
| RT + FBT + NextLat | 9,892 | 3.313 | 39.09 / 60.46 | 657.84–701.12 | 2.808 |

\* The RT+FBT resource rows retain their initial gradient-screen failure.
Read [roundoff assessment](roundoff-assessment.md) and
[continuation decision](continuation-decision.md).

The exposure fixture has 32,768 input tokens and 16,384 CE targets per B64
update. NextLat adds 32,704 latent pairs, 16,384 KL triples and 32,704 predictor
positions per pass. K2 doubles pass work, not unique data exposure. The hours
column excludes data loading, evaluation, graph setup and checkpoint I/O;
it is not a promise for another dataset/mask or a long training job. Matrix
FLOPs exclude elementwise/optimizer/launch/communication and hardware padding.

At fixed B64, adding two RT blocks costs roughly 0.62–0.63 seconds across all
four relevant pairs. NextLat adds 0.31 seconds with one stack and0.61–0.62 with
K2. These approximate additive costs agree with the execution semantics:
the FBT bootstrap is ordinary, so RT occurs only in the feedback stack, while
NextLat runs on both passes. Do not interpret sub-percent differences as
significant from three samples.

## Larger batch and operating points

| Configuration | B96 tokens/s | Change vs. B64 | B96 allocated / peak reserved GiB | Recommendation |
| --- | ---: | ---: | ---: | --- |
| ordinary | 31,259 | +0.5% | 31.02 / 45.41 | B64; no useful throughput gain |
| rt | 21,618 | +11.2% | 39.18 / 62.45 | B96 is a useful option; B64 remains matched reference |
| nextlat | 23,938 | -0.4% | 32.63 / 49.14 | B64; no useful throughput gain |
| rt-nextlat | 17,833 | +8.5% | 40.79 / 64.97 | B96 is a useful option; B64 remains matched reference |
| fbt | 15,764 | -0.2% | 39.07 / 61.91 | B64; no useful throughput gain |
| rt-fbt | 12,893 | +6.2% | 47.16 / 78.39 | B64; tight setup reservation at B96 |
| fbt-nextlat | 12,049 | -1.1% | 44.22 / 73.89 | B64; tight setup reservation at B96 |
| combined | 10,305 | +4.2% | 48.77 / 78.22 | B64; tight setup reservation at B96 |

All B96 attempts succeeded; no out-of-memory run occurred. Success is not a
comfortable operating point: RT+FBT and combined reserve about78.2–78.4GiB
during setup on a79.65GiB-visible device. FBT+NextLat also exceeds the
prospective72GiB headroom line, without a speed gain. B64 remains the common
development default, leaving about19.19GiB against device capacity even for
the largest observed setup reservation. RT-only and RT+NextLat have useful
B96 gains with peak reservations62.45/64.97GiB. These figures do not promise
headroom after adding layers/features, padding, profiling or other processes.

Setup and steady memory are separately recorded. Peak reservations can be
nonmonotonic between related configurations because allocator/pool histories
differ; do not infer isolated module memory from small peak differences.

## Validation and retention scope

- 23 main GPU reports:22 complete successes and one retained numerical-screen
  failure;46/47 declared gates pass. All16 capacity cards have finite updates.
- Six new successful correctness runs, historical RT-only/combined T512
  evidence and the separate RT+FBT diagnosis cover the graph/update routes.
  This does not imply large-batch full-gradient equivalence for every card.
- 132 optimizer updates in successful main reports (66 eager+66 graph), plus
  six separate diagnostic updates (3+3). The initial failed reference screen
  and failed first diagnostic harness each performed zero optimizer updates.
- 235 scoped CPU tests pass:76 estimator/multi-layer/dispatch,31 actual tiny
  all-eight resource updates and128 reporting/retention checks.
- Runtime/protocol397885b; roundoff initialc8f2311, fixed949731b; reporting
  e4dda74 plus final presentation changes. All run-local sources/protocols,
  errors, W&B URLs, timing samples and compressed operator traces are retained.
- Read [results](results.md), [resource ledger](resource-ledger.json),
  [roundoff assessment](roundoff-assessment.md), [operator audit](operator-audit.md)
  and [final inputs](final-inputs.json). The [storage receipt](storage-receipt.json)
  verifies create-only GCS retention under
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f4-features/20260923T160539Z/`:
  1,206 archive members, 9,119,232 compressed bytes, with checked hashes and
  native-checkpoint identity. Disposable few-update weights are omitted;
  the pinned native checkpoint is retained by reference.

## Parameters and evidence

| Component selection | Active training parameters | Deployable parameters |
| --- | ---: | ---: |
| Ordinary / RT | 1,176,764,416 | 1,176,764,416 |
| + FBT | 1,185,153,024 | 1,185,153,024 |
| + NextLat | 1,259,491,328 | 1,176,764,416 |
| + FBT + NextLat | 1,267,879,936 | 1,185,153,024 |

RT adds no weights. FBT adds 8,388,608 shared fusion weights; NextLat adds
82,726,912 training-only predictor weights. Inactive fusion remains resident
in the common harness, so registered totals exceed active totals when FBT
is disabled. Every card separately records observed optimizer/gradient ownership.

The [operator audit](operator-audit.md) reconciles dense matrix arithmetic
exactly for ordinary and combined traces. Both demonstrate ordinary Flash;
combined also demonstrates Triton historical forward/backward. Fused attention
arithmetic is absent from selected-op profiler FLOPs and is accounted for
analytically. This is not native FA4 RT or measured hardware utilization.

## Numerical qualification and readiness

One original RT+FBT B8/T512 tensor has 6.3492% peak-coordinate error versus
materialized BF16, narrowly outside the unchanged 6.25% screen. Global L2
1.0206% and all tensor L2 values pass. The failure is retained, not waived.
Fixed-state repeats, graph replay and three full Adam updates are exact.
FP32 materialized/recompute globalL2 is3.04e-6. Eager historical backward
removes the coordinate crossing, supporting accumulated reduction/rounding
sensitivity rather than an identified semantic defect.

Both BF16 reference and candidate differ from full FP32 by roughly 18%
gradient L2 at this initialization (cosine about .983–.984); feedback CE differs
by about 0.125%. The broader sensitivity remains a qualification, not a passed
full-precision campaign. No evidence here justifies automatically adding Q/K
normalization. Before substantive learning, a bounded transition-state and
clipped-update comparison is a reasonable follow-up; changing only recompute
would not remove the discrepancy already present in the original BF16 control.

Next readiness work: graph save/resume and accumulation boundaries, padded
batches, then finite-prefill/exact-online resource cards for the four deployment
routes. NextLat is omitted at inference. Genuine two-GPU work still requires
a second GPU; one H100 is exposed. No quality or long training run is queued.
