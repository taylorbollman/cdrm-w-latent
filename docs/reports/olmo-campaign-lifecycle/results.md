# CPU distributed lifecycle results

The new opt-in boundary helper passed all eight two-rank Gloo fault cases.
Both ranks reported identical errors, rank-zero callbacks stayed on rank zero,
and phase/ownership disagreement prevented callbacks. A failed publication
preserved the prior complete checkpoint; its incomplete newer file was never
treated as committed.

| Case | Completed update in memory | Last complete checkpoint | Outcome |
|---|---:|---:|---|
| Successful logging/publication | 2 | 2 | State and RNG unchanged by logging; checkpoint restores exactly |
| Rank-zero logging error | 2 | 1 | Both stop; unsaved update 2 replays exactly |
| Rank-zero error before checkpoint rename | 2 | 1 | Both stop; old checkpoint remains; update 2 replays exactly |
| Rank-one data preparation error | 1 | 1 | Both stop before next update |
| Mismatched phase | 1 | 1 | Neither callback executes |
| Mismatched callback ownership | 1 | 1 | Neither callback executes |
| Invalid rank-one descriptor | 1 | 1 | Neither callback executes |
| Distinct errors on both ranks | 1 | 1 | Both report both exception classes |

This is recovery from the last complete checkpoint, not rollback of completed
updates. Logging/publication failures leave update 2 in memory while update 1
remains the recovery authority. Restoring update 1 and re-executing update 2
reproduced the CPU model, complete Adam state, RNG and cursor exactly. Tests also
independently read the published and unpublished files: their cursors were 1 and 2,
with different model, Adam and RNG states. Injected private exception text never
appeared in the common report; only phase, rank and exception class were exposed.

Focused tests: **3 passed in 4.74s**, including the real two-process eight-case
exercise. The retained standalone run completed in **3.290s** at
2026-09-29 08:24:20 UTC. All 126 per-case checks and all 5 report checks passed.
No CUDA context was initialized. Four source files were independently rehashed
against both the live files and retained snapshots.

Evidence: `.runtime/olmo-campaign-lifecycle/check-01/`, containing report,
per-rank reports, tiny checkpoint fixtures, source snapshots, launcher log and
independent audit. Report SHA256:
`b033e87285777c6f49ca9b718c403dc2ae6bd6de185cc390d22ebf36a78d3a15`.
The initial test result was captured in tool stdout; the separate retained
standalone run has its complete launcher log.

Existing campaign runners and the active numerical startup runner were not
changed. The known uncoordinated logging path therefore remains to be integrated
with this helper in a later runner revision. These results cover ordinary Python
exceptions at synchronized quiescent boundaries; they do not qualify NCCL,
CUDA graph failures, process death, hung callbacks, remote storage durability or
recovery after a partially completed distributed optimizer operation.
