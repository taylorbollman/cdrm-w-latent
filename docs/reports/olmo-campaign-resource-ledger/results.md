# Campaign resource ledger

2026-09-29. Native-size ledger generation completed on CPU in38.5seconds;
33 focused CPU tests pass. No GPU execution, model training, existing runtime
change or numerical-policy change is part of this task. These are analytic
matrix-work estimates, not measured accelerator FLOPs or throughput.

The tool reuses the pinned packed first update at T1024, B12/rank, two ranks and
22 physical slots/rank. It counts actual module/optimizer ownership separately
from active architecture parameters, and useful supervised targets separately
from dense graph loss work on padded/dummy slots. See protocol and usage for
scope and assumptions.

| Arm | Unique resident parameters | Trainable / optimizer-owned | Deployment parameters | Dense matrix PFLOPs/update |
| --- | ---: | ---: | ---: | ---: |
| B | 1,185,153,024 | 1,176,764,416 | 1,176,764,416 | 4.945–5.416 |
| N | 1,267,879,936 | 1,259,491,328 | 1,176,764,416 | 5.769–6.240 |
| F | 1,185,153,024 | 1,185,153,024 | 1,185,153,024 | 19.860–21.746 |
| R | 1,185,153,024 | 1,176,764,416 | 1,176,764,416 | 5.335–5.747 |
| NF | 1,267,879,936 | 1,267,879,936 | 1,185,153,024 | 23.156–25.042 |
| NR | 1,267,879,936 | 1,259,491,328 | 1,176,764,416 | 6.159–6.571 |
| FR | 1,185,153,024 | 1,185,153,024 | 1,185,153,024 | 21.420–23.070 |
| NFR | 1,267,879,936 | 1,267,879,936 | 1,185,153,024 | 24.716–26.366 |

One PFLOP means10^15 estimated matrix operations. R adds no parameters. The
wrapper retains8,388,608 inactive, frozen fusion parameters in non-F arms; they
are resident but absent from the active/deployment architecture. N contributes
82,726,912 training-only parameters. Tied readout/embedding is counted once.
No optimizer moments were allocated or updated during this inventory.

Every row uses the same declared44 physical slots /528 allocated rows. There
are512 real packed rows,16 dummy rows,524,288 useful input tokens and540,672
allocated input positions. F arms execute four passes:2,097,152 valid pass tokens
but2,162,688 allocated pass tokens. This fixed footprint is a comparison of
accounting conventions, not a batch recommendation for all eight arms.

For NFR, target selection and executed loss work differ even before multiplying
by four passes:

| Work per logical update, once per pass | Selected sparse positions | Dynamic dense positions |
| --- | ---: | ---: |
| CE projection | 523,776 | 540,144 |
| Latent regression | 523,768 | 540,144 |
| Predictor-source union | 523,768 | 540,144 |
| KL teacher/student positions | 523,248 | 539,616 |

The dynamic graph computes padded/dummy and boundary-excluded positions before
masking their losses. NFR therefore estimates24.716–26.366PFLOPs versus
24.562–26.212PFLOPs for selected-loss arithmetic at the same padded backbone
footprint. This0.154PFLOP difference is arithmetic accounting, not an asserted
speedup or numerical-equivalence claim. All-pass RT executes8 selected-block
calls plus56 ordinary-block calls per slot (352 RT and2,464 ordinary globally).
NF executes64 ordinary calls per slot. CE/auxiliary weights do not reduce these
call counts or the actual number of projections.

The estimate reuses established checkpoint/recompute/KV-only formulas. Its
range reflects ordinary attention and checkpoint early-stop accounting choices;
it excludes pointwise kernels, hardware padding, compilation/capture warmup,
optimizer and communication. It cannot be used as frozen-backbone warmup cost,
an H100/H200 memory forecast, MFU or a quality comparison.

Evidence: `.runtime/olmo-campaign-resource-ledger/ledger-01/report.json`, SHA256
`f7d111fdbbac0cc9147766a6b74cd2b09d529e9de856b0608fe5c9bca5f5fea9`.
All49 recorded source files and retained source snapshots match. Source/report
integrity, CPU-only execution and all-eight-arm completion checks pass. The
launcher log is retained beside the report. Cloud retention is owned by the
parent milestone; no separate training run or experimental curve was created.
