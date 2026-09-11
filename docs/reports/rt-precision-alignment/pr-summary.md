# Validate released-style RT mixed precision with CUDA graphs

The existing bounded CUDA-graph path accepted only our protected attention
policy, so it could not directly establish whether the released-style
precision mixture was correct, efficient or adequate for training. This
change makes that policy explicit before model construction and capture, and
adds a reproducible numerical and real-text comparison. The protected
`bf16_fp32_state` default remains in place; `legacy` is an explicit experimental
selection. **The completed two-seed development comparison keeps A as the
default and B experimental:** B is faster and uses less memory, but neither
seed clears the unchanged 0.005-nat quality margin with its one-sided 95%
upper bound. The conditional confirmation evaluation was not run.
[Recorded disposition](../../../.runtime/rt-precision-alignment/20260910T191100Z/reference/development-disposition-500.json).

The tested model is the standard RT with all 12 blocks tiled recurrent:
D1024/H16/FFN4096, rho 1, causal ALiBi, learned Q/K normalization, 151,045,120
backbone parameters and 216,843,264 total. Physical B512/T512, head-only
microbatch 2 and four internal backward MLP chunks remain fixed. Parameters,
residuals, parameter gradients and Adam moments remain FP32. The candidate
preserves the released mixture, including its existing FP32 running sums and
buffers; it is not an entirely BF16 model.

The implementation provides:

- Explicit precision selection through [graph capture](../../../scripts/rt_cuda_graph.py),
  [profiling](../../../scripts/rt_batch_profile.py) and
  [graph validation](../../../scripts/rt_cuda_graph_validate.py). Capture verifies
  every block's policy, retains stable FP32 gradient buffers, clears them once
  per replay and rejects changed configuration/storage/hooks. Clipping and Adam
  remain outside the graph. Setup synchronization/cache release controls
  reservation without changing model arithmetic.
- Identical-state A/B/C comparisons of native loss, all parameter gradients,
  clipping and actual Adam deltas, using cloned weights and moments. FP64 metric
  reductions retain all coordinates and review flags without an absolute
  acceptance floor. Dtype observation checks its own neutrality; recurrent
  credit/scaling and frozen-operand FP64 attention probes localize discrepancies.
  The optional partitioned FP32 reference is separately qualified; physical
  B512 FP32 fit in this experiment, so that fallback was unnecessary.
- Pinned C4/T5 data preparation with retained document boundaries and separate
  diagnostic, development and confirmation roles. Captured training preserves
  identical token order and the released 5,000-update warmup, with explicit
  100/500 endpoints and strict checkpoint/source/runtime/data resume identity.
  Fresh-process resume probes compare loss, gradients, parameters, Adam and
  RNG state exactly at a bounded checkpoint transition.
- Checkpoint-authorized common-FP32/native evaluation with per-target CE and
  document aggregates. The paired analysis independently reconciles those
  aggregates, verifies identities and hashes, then computes token-weighted
  paired document intervals. Native training uses CE sum divided by B×512 over
  511 targets per row; held-out results use CE per supervised token.
- Online W&B records, immutable source/input snapshots, retained checkpoint
  and diagnostic packets, and GCS verification receipts. CPU report builders
  merge only compatible completed segments, bind evaluations/numerical anchors
  to their exact training authorities, and display missing seed/endpoint/role
  coverage instead of silently omitting it.

Existing parameter ownership, masking, autocast restoration, compiler fallback
and graph correctness repairs are retained. No core recurrent arithmetic was
changed for this milestone. The local source snapshots identify the executed
code; the vendor commit alone is not a pristine upstream identity. The
[change audit](change-audit.md) explains retained fixes versus the A/B precision
treatment, and the [usage guide](../../rt-precision-alignment-usage.md) describes
the bounded drivers.

Validation completed:

- **152 integrated CPU tests passed**, with zero failures/errors/skips and CUDA
  initialization forbidden. These cover the graph/head-loss contract, data,
  numerical, evaluation, paired-bootstrap, training and resume harnesses,
  including real saved-checkpoint metadata integration.
  [Test receipt](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/final-integrated-cpu-tests.json).
- **16 separate outcome-report contract checks passed**, using actual completed
  reports and explicitly synthetic continuation metadata. These are reported
  separately from the pytest count. Actual first-seed 100→500 report integration
  and the final two-seed online figure build subsequently completed too.
  [Contract checks](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/outcome-report-integration/report.json)
  and [actual endpoint integration](../../../.runtime/rt-precision-alignment/20260910T191100Z/verification/outcome-report-integration/actual-seed0-500.json).
- On H100 80GB inside the required container, candidate tiny/full-B2 captured
  versus uncaptured execution matched exactly through changed-input Adam
  updates and passed causality. Physical B512 initial loss and all 111
  gradients matched exactly, followed by bounded captured updates. Dtype
  observers were neutral, recurrent credit and fixed-forward scaling checks
  passed, and physical-B512 FP32 references completed.
- All four policy/seed runs completed 500 updates with finite FP32 parameters,
  gradients and moments. All trained numerical screens passed at both first-seed
  policies' 100 and 500 anchors; second-seed gradient anchors are outside the
  predeclared representative numerical scope. Fresh-process resume checks passed; recovery reproduced
  all 69 overlapping saved scalar updates exactly. The independent lineage
  audits verified matching initialization within each seed, distinct
  initialization across seeds, all 2,000 update records, checkpoint hashes,
  source/runtime/data identity and fixed compiler counters.
  [Results and audits](final-results.md).

The completed development comparison shows a small learning difference
despite reassuring trained-state numerics:

| Seed / development checkpoint | Common-FP32 B−A CE | Paired one-sided 95% upper | Frozen 0.005 margin |
| --- | ---: | ---: | --- |
| seed0 / 100 | +0.00940324 | 0.01073532 | Not cleared |
| seed0 / 500 | +0.00892543 | 0.01033497 | Not cleared |
| seed1 / 500 | +0.00416939 | 0.00610802 | Not cleared |

Seed0's interval lies entirely above the margin. Seed1's point estimate is
below it, but its upper bound exceeds it: this is inconclusive relative to the
margin, not proof of degradation beyond 0.005. Both gaps are positive; no pooled
seed inference is made. Native evaluation closely agrees, with endpoint gaps
0.00893324 and 0.00411749, respectively.

Over seed0 resumed updates 103–500, B averaged 5.4501 seconds/update versus
6.1305 for A; seed1 updates 3–500 averaged 5.4463 versus 6.1284. Both give about
11.1% less update time for B. Peak reservation is approximately 39.17 versus
49.00 GiB. Timings omit the first two updates following capture, include
verification and use sequential runs; each paired phase used the same
physical device, but the GPU UUID changed across interruption.
[Two-seed W&B figures](https://wandb.ai/taylorbollman/rt-precision-alignment/runs/lbo30ro4).

Initial query/key rounding flags remain documented; passing trained screens
does not erase them or guarantee equivalent learning. The 0.005 margin and
gradient budgets are prospective engineering criteria, not author-published
tolerances. This bounded C4 slice is not the authors' unavailable processed
corpus. At 500 updates LR is only 0.00019 of the eventual 0.001 peak; neither
peak-LR behavior nor final convergence is cleared.

B was not nominated for adoption after development, so no confirmation
evaluation was run; performance on the confirmation set remains unassessed. The
[frozen endpoint scope](../../../.runtime/rt-precision-alignment/20260910T191100Z/reference/endpoint-check-scope.json)
uses first-seed A/B anchors for representative numerical coverage. No new
kernel repair is justified solely by passing those screens, and the observed
learning gap is not attributed to one isolated arithmetic primitive. A
comparison through the full warmup, or explicitly accepting a speed/quality
tradeoff, would require a separate decision. This completes the bounded
development milestone with defaults and core arithmetic unchanged. This is
a local review summary; no commit, push or external PR was created.
