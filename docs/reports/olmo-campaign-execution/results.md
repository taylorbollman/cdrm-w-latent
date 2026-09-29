# Shared component execution and recovery

**In progress,2026-09-29.** Tiny acceptance and native stop/cloud asset restoration
have passed. Native uninterrupted-reference and resumed-update comparisons are
still running; no complete native replay claim is made yet. PR46.

The common launcher now executes B/N/F/R/NF/NR/FR/NFR with explicit startup and
finite packed-data plans. B means the ordinary backbone, N adds NextLat, F adds
K4 feedback passes, and R selects native RT layers0/15. Original-weight/fresh-Adam
startup supports all eight arms. The initial adapted allowlist is fusion128
weights plus fresh all-active Adam for NF/NFR. Same-lineage recovery uses the
committed checkpoint directly, without needing a successful reference report.

Lean observations retain losses, global counts, gradient norm/clipping estimates,
LR and data clocks while avoiding per-update complete gradient/model/Adam hashes.
Acceptance observations use the identical training path with those hashes added.
Complete checkpoint integrity checks remain in both modes. The native model,
losses, kernels, Q/K behavior and historical acceptance implementations are unchanged.

## Completed evidence

All-eight CPU checks cover parameter/optimizer ownership, actual objective updates,
packed document boundaries, masks, accumulation, empty rank slots, observations,
strict startup/recovery metadata and optimizer/scheduler clocks. Separate operational
helpers verify generation-pinned cloud assets and audit recorded execution evidence.
See the [test ledger](test-ledger.md) for exact test scopes and retained attempts.

Tiny two-H100 acceptance passes:

- Three-update reference versus lean full run: identical metrics and final complete
  model/Adam/schedule/RNG/cursor state.
- Lean stop at update1, GCS generation-pinned restoration, fresh-process updates2/3:
  identical restored origin and every resumed input, raw gradient and step boundary.
- Resume of completed update3: validates and exits with no graph preparation or
  optimizer step; state remains exact.

The tiny fixture uses width32/two layers, T16, FP32 math/eager RT, B2 per rank and
five global rows/update. It checks operations and graph ownership; it does not
exercise native Triton head dimensions or establish BF16 accuracy.

## Native configuration and first boundary

Native OLMo-1B step60000 backbone:16layers,width2048,16heads,MLP8192,tied readout.
NFR adds K4 feedback, RT at layers0/15 in every pass, NextLat latent and KL losses,
with 1,267,879,936 active parameters:1,176,764,416 backbone,8,388,608 fusion and
82,726,912 predictor. Import uses the selected complete fusion-only update128
state, while the backbone/predictor remain at their pinned original authorities.
The new optimizer/data/schedule clocks start at zero. Prior adaptation exposure
is recorded separately; this is not exposure matched to untouched original weights.

Execution is two H10080GB GPUs, BF16 mixed with FP32 master weights/Adam,
ordinary Flash SDPA, native Triton RT forward/backward, backward recomputation,
ordinary activation checkpointing, reused RoPE and prepared CUDA graphs.
T1024, B1/rank, three global rows/update give two microbatches per rank; rank1's
last slot is empty. This tiny physical batch is for bounded recovery acceptance,
not a production batch recommendation or steady-state throughput test.

The native stop-at1 run passed all strict import/transition checks and preserved
the full starting boundary through 20 warmup and two capture backwards per rank.
Adam started empty and acquired state for all71 owned parameter tensors. One update
consumed3,072 inputs,3,069 CE/latent targets and3,066 KL triples. It produced finite
CE6.99714,latent0.80674,KL4.86984. Raw global gradient norm697.082 activated the
existing norm1 clipping (coefficient approximately0.00143455). This clipping is
reported, not evidence that longer optimization is stable.

Observed update regions were6.969s backward and0.097s optimizer/cursor. Per GPU,
after-update allocated memory was23.711GiB and reserved36.373GiB. Preparation
was375.05s. Cloud retention took100.11s for the5.071GB origin and294.34s for the
15.215GB update1 payload. Whole segment elapsed974.27s. These are instrumented
single observations, not throughput or capacity recommendations. Streamed recovery
of the exact update1 generation subsequently verified15.215GB in81.72s; no host
peak-memory or comparative I/O speed claim is made.

## Scope and next work

Existing BF16-vs-FP32 discrepancies remain qualified. Exact same-precision replay
establishes recoverable execution; it does not resolve precision equivalence,
longer-run quality or optimization sensitivity. No Q/K normalization was changed.
The short native prefix contains no internal source-document boundary; independent
CPU fixtures cover that policy. It is readiness data, not a selected pilot mixture.

Next implement [declared per-pass held-out evaluation](next-steps.md), including
live training-state preservation. Then review the actual mixture, budget, startup
exposure and target-hardware capacity/throughput before a quality pilot. See
[operator notes](operator-notes.md) for launch/recovery boundaries and limits.
