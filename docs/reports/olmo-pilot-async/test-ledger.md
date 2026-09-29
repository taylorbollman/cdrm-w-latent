# Validation ledger

The runtime frozen at `85e5f78` passed **76 distinct CPU tests in 6.00 seconds**
in the GPU-disabled project container. The separate JSON auditor then passed
**60 additional tests in 1.99 seconds**: **136 distinct new tests** in total.
Earlier overlapping subsets are not added again. Two Google SDK future-version
warnings appeared; no dependency was changed during acceptance.

| Scope | Tests | Main checks |
| --- | ---: | --- |
| New executor/engine integration | 12 | Accepted save/preparation/update math, identical construction, transport identity, invalid policy before CUDA, historical source preservation |
| Asynchronous host loop | 35 | Training can advance while retention is pending; drain before next save and terminal; source/stop/error boundaries; no false publication or recovery collectives after an unknown update failure |
| Worker and storage integration | 29 | One outstanding job, immutable metadata, exclusive ownership, exact existing publication/pruning, CPU child isolation, cancellation/timeout cleanup, changed-byte rejection, preserved prior cloud authority |
| Independent JSON auditor | 60 | Actual schema and policy, source snapshots, gradients/state/counters, named evaluation, original worker files, publication integrity, overlap scope and mutation rejection |

All 192 historical runtime pins remain unchanged. The new runtime contains
200 source pins with canonical inventory SHA256
`d84006ede4e5ad488880da2af387355ca82fc76d186cf87ba16cb007d2279fc5`.
The independent auditor is outside that frozen runtime inventory.

Actual two-H100 tiny acceptance:

- Blocking versus asynchronous retention: **2,847 audit checks pass**. The three
  updates have exact input, raw-gradient, optimizer and complete final-state
  evidence on both ranks. Evaluation is also preserved. The async worker's
  wall intervals overlap updates 2 and 3; the blocking reference has no such
  overlap. This is concurrency evidence, not a native-model speed benchmark.
- Cloud checkpoint 2 restored to a fresh SSD directory, then update 3 in a fresh
  process: **2,427 checks pass**, with exact continuation and final state.
  Recovery uses the actual asynchronous producer's manifest and publication
  receipt, not a relabeled historical checkpoint.

Original audit JSON, source snapshots, per-job files and persistent publication
receipts are retained. A daemon interruption occurred after the async reference
finished; its completed report and W&B sync were recovered, and idle GPUs were
verified. The outer launch-result file was absent, so no missing exit code was
invented and no completed experiment was repeated.

The native accumulated diagnostic is recorded separately in results.md when
complete. Native byte retention is tested; exact new-runtime cloud continuation
is scoped to the tiny fixture. Existing older native recovery evidence remains
separately qualified. No new BF16 equivalence or H200/topology claim is made.
