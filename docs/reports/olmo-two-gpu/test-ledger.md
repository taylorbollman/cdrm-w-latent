# Two-H100 test ledger

2026-09-25. This ledger records validation scope, not a final run inventory.
See [results.md](results.md) for measurements and retained qualifications,
[usage.md](usage.md) for execution contracts, and the generated local audit for
final report/receipt counts. No new test or GPU run was performed to write this
document. The scopes below overlap and must not be added into a distinct-test
total.

## Latest focused CPU scopes

CPU checks ran in the project container with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh
```

The following commands reproduce the recorded test-file scopes inside that
container. They are reconstructed from the execution record: the retained logs
contain pytest outcomes but do not echo the literal historical argument vector.
They do not claim that a future checkout will collect an identical test count.

**Capacity integration: 77 passed, 67 warnings, 8.62 seconds.**

```bash
python -m pytest -q \
  tests/test_olmo_two_gpu_graph.py \
  tests/test_olmo_two_gpu_single_reference.py \
  tests/test_two_gpu_zero1_graph.py \
  tests/test_zero1_training.py
```

Log: `.runtime/olmo-two-gpu/logs/final-capacity-integration-cpu-tests.log`.
This covers replicated graph harness checks/accounting, the matched single-GPU
fixture and timing contract, ZeRO-1 graph integration/accounting, local moment
ownership, optimizer loading and recovery. CPU mocks and Gloo collectives do not
establish CUDA/NCCL graph compatibility; the GPU evidence below supplies that.

**ZeRO-1 transport change: 54 passed, 67 warnings, 11.33 seconds.**

```bash
python -m pytest -q \
  tests/test_zero1_consolidation_transport.py \
  tests/test_zero1_training.py \
  tests/test_two_gpu_zero1_graph.py
```

Log: `.runtime/olmo-two-gpu/logs/zero1-buffer-cpu-tests.log`.
This covers native serialized-byte compatibility, buffer-exporter lifetime,
sender/receiver ordering, subgroup rank mapping, empty and initialized states,
real two-rank Gloo consolidation comparisons, and optimizer/recovery/graph
contracts. Tiny and actual-model NCCL recovery separately validate the transport
change on GPUs. The 67 warnings in both latest scopes are installed PyTorch JIT
deprecation warnings. The container's “driver not detected” banner is expected
for these explicitly CPU-only runs.

## Earlier CPU evidence and corrected attempts

| Retained log under `.runtime/olmo-two-gpu/logs/` | Observed result | Scope/interpretation |
| --- | --- | --- |
| `followup-cpu-tests-02.log` | 91 passed, 67 warnings, 10.05 s | DDP graph runtime/harness, anchored validation, graph recovery and ZeRO-1. Overlaps the later focused scopes. |
| `integrated-cpu-tests-01.log` | 129 passed, one warning, 21.85 s | Earlier integrated distributed implementation checks. Retained result; the log does not contain its full argument list. |
| `graph-input-fix-tests-02.log` | 58 passed, 3.73 s | Corrected static-input graph runtime/harness scope. |
| `graph-harness-fast-check-tests.log` | 8 passed, 2.83 s | Earlier graph harness comparison/check helpers. |
| `graph-input-fix-tests.log` | Two failed, 55 passed, 3.83 s | Two test doubles still declared zero-argument `forward` after the runtime began passing the static token tensor. Their signatures were corrected; the 58-test retry above passes. |
| `followup-cpu-tests.log` | No tests ran | Invocation named nonexistent `tests/test_olmo_two_gpu_zero1.py`; corrected invocation is the 91-test retry above. |

The recorded 91-test file scope was:
`test_ddp_graph_training.py`, `test_olmo_two_gpu_graph.py`,
`test_olmo_two_gpu_validate.py`, `test_olmo_two_gpu_graph_recovery.py` and
`test_zero1_training.py`, all under `tests/`. Earlier handoffs also record
44 recovery/checkpoint, 24 retention, 13 single-reference, 64 eager/preparation
and 32 checkpoint checks. Those are historical overlapping scopes, not extra
independent checks to add to the latest results. This ledger does not reconstruct
unknown older command arguments or claim a repository-wide test-suite pass.

## GPU validation categories

All GPU work ran inside the project container on two distinct H100 80GB devices,
one process per device. Reports freeze source hashes and runtime configuration.
“Exact” below means the tested same-candidate comparisons are bitwise equal;
canonical references with different reduction ordering use the frozen budgets.

| Category | Evidence and verified scope |
| --- | --- |
| Hardware/NCCL | `nccl-01`: topology and peer access inventory; genuine sum collectives across five payload sizes, through 256 MiB. Isolated collective timing is not training overlap. |
| Eager feature combinations | `tiny-eager-01`: all eight RT/FBT/NextLat combinations, three accumulated updates each. Global loss denominators, unequal/local-empty objectives, early `no_sync` predictor use and globally unused parameters are exercised. |
| Actual canonical updates | `actual-eager-01`: ordinary and RT each pass two updates; combined update 2 retains the numerical qualification below. `combined-anchored-01` passes both complete updates from canonical starting states without changing budgets. |
| Replicated optimizer graph correctness | `tiny-graph-03`, `rt-graph-02`, `combined-graph-01`: actual captured DDP/NCCL, initial exact raw parity and two changed-input complete Adam comparisons. Four compound checks per rank, seven physical updates per rank, logical endpoint five. |
| Replicated eager recovery | `tiny-recovery-01`, `rt-recovery-01`: 24 coordinated checks each; exact next raw gradients, loss, full optimizer/model state, scheduler/counters, data cursor and local RNG draws. Four physical updates per rank, logical endpoint three. |
| Replicated graph reconstruction | `tiny-graph-recovery-02`, `combined-graph-recovery-01`: 42 coordinated checks each. Both branches rebuild graphs; exact next gradients, losses and full state. Six physical updates per rank, logical endpoint five. |
| ZeRO-1 update/recovery | `tiny-zero1-01`, `combined-zero1-01` and buffer-transport retries: 13 compound checks per rank, including two exact fixed-gradient full-Adam comparisons, disjoint moment ownership and exact consolidated recovery. Four distributed updates plus two reference Adam steps per rank. |
| ZeRO-1 graph integration | `tiny-zero1-graph-01`, `combined-zero1-graph-01`: initial and terminal changed-weight raw eager/graph parity, parameter replicas, local moment partition/health and step counters pass. Local shards are not incorrectly asserted to be replicas. Actual B1 timings are integration diagnostics. |
| Capacity/scaling | Single, DDP and ZeRO-1 capacity harnesses check initial/terminal raw parity and state ownership around five timed changed-input complete updates, following three preparation updates. Final reports determine which larger shapes passed. Do not infer a passing shape from command availability or an unfinished progress file. |

Compound checks contain multiple assertions. Per-rank checks are checks of
replicas, not independent model scenarios. Warmup/capture backwards are not
optimizer updates. Canonical/reference optimizer steps and repeated recovery
branches are physical work even when discarded afterward; the logical endpoint
is smaller. The generated inventory distinguishes these counts.

## Five retained GPU/launcher failures

| Attempt | Preserved failure and disposition |
| --- | --- |
| `actual-eager-01` | Combined independent update 2 passes loss/counts, raw-gradient budgets (global relative L2 1.3509761003e-5) and exact rank replicas, but 20 parameter and seven moment tensors miss strict complete-update tolerances. Maximum parameter absolute difference is 1.383945346e-6; maximum parameter-tensor relative L2 is 9.096305447e-8. Anchored comparisons pass unchanged budgets; the independent-trajectory failure remains. |
| `tiny-graph-01` | Installed DDP CUDA path rejects empty forward arguments. Fixed by passing the existing static token tensor. Zero optimizer updates. |
| `tiny-graph-02` | Restoring unchanged buffers advances versions and correctly trips the prepared-state guard. Parameter-only restoration with fixed-buffer checks passes. Failed attempt executed four optimizer updates per rank. |
| `tiny-graph-recovery-01` | Missing dependency-recorder output-directory argument; corrected setup call. Zero optimizer updates. |
| `rt-graph-01` | Missing required NCCL asynchronous-error settings; rejected before process-group initialization. Corrected launcher passes. Zero optimizer updates. |

Passing retries do not relabel original reports. No numerical tolerance was
relaxed. Exact local/distributed operational checks do not clear the independent
BF16 trajectory qualification or the earlier native-versus-author BF16 findings.
Any additional failed capacity attempts belong in the final inventory separately.

## Limits of the recovery and integration claims

The probes reconstruct model, optimizer and DDP inside the **same process group**.
They do not launch a fresh `torchrun` process group or test recovery following an
actual VM termination. Loading uses a committed manifest and exact matching world,
source/configuration and ownership. Graph handles are released before saving,
and both recovery branches warm up and capture again; live graphs are not saved.

The tested full-model scope uses the 16-layer OLMo backbone with RT at indices
0 and 15, T512, BF16 mixed and FP32 persistent training state. Fixed graph masks,
counts, shapes and participation are part of the contract. These checks do not
establish all-16-layer RT, dynamic padding, world-size changes, bucket-view
adoption, ZeRO-2, offload, other context lengths, long-run precision equivalence
or model-quality gains. They are bounded evidence that the declared elements
work together and can be measured and checkpointed coherently.
