# Campaign two-H100 test ledger

Recorded 2026-09-29 from completed local reports and verified retention receipts.
The T1024 B8 capacity check passes; the B16 candidate remains pending below. This ledger distinguishes
distributed execution, numerical reference agreement and restart acceptance.
See [results](results.md), [protocol](protocol.md),
[qualification plan](qualification-plan.md) and [storage receipt](storage-receipt.md).

## CPU verification

All CPU tests ran inside the project container with GPU passthrough explicitly
disabled. They do not substitute for the actual NCCL/CUDA checks below.

| Scope | Result | Evidence |
| --- | --- | --- |
| Initial campaign runner/probe/restart checks | 60 passed in 38.61 s | `.runtime/olmo-campaign-two-gpu/cpu-tests-01.log` |
| Campaign/checkpoint/DDP regression after scalar metadata fix | 389 passed in 59.90 s | `.runtime/olmo-campaign-two-gpu/cpu-tests-02.log` |
| Final regression, including historical DDP graph scopes | **445 passed in 60.03 s** | `.runtime/olmo-campaign-two-gpu/cpu-tests-final.log` |
| Capacity helpers after resident-Adam setup change | **19 passed in 2.52 s** | `tests/test_campaign_capacity.py`; focused agent-run output |

These scopes overlap; do not add their counts as distinct tests. The focused
capacity scope extends its previous 17 tests with checks of actual initialized
Adam moments and independent optimizer/scheduler/token-clock snapshots.

The probe CPU checks compare an independently assembled canonical objective
against the prepared dense objective for all eight campaign arms. Fixtures
exercise M1/M2/M3, unequal target counts, a wholly empty rank, empty final
synchronization slots, deterministic keyed jitter and changing masks. Runner
tests exercise two real CPU/Gloo ranks; their GPU qualification is separate.

Development-only fixture corrections included replacing an invalid tile backend
name with `eager` and clearing diagnostic gradients before preparing a new local
runner. Neither changed model math or numerical acceptance thresholds.

## Completed hardware checks

Hardware: two H100 80 GB GPUs, NV18 connectivity, NCCL 2.30.5. GPU commands ran
only in the required project container. Torch is
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, driver 580.178.04. Tiny probes use FP32
math attention/eager native RT. Pretrained probes use BF16 mixed with FP32
parameters, gradients and Adam state, TF32 disabled, forced ordinary Flash SDPA,
native Triton RT recomputation and ordinary activation checkpointing.

| Stage / W&B | Result | Scope |
| --- | --- | --- |
| [NCCL sanity](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ci14andb) | All 5 message sizes pass | Exact reductions, 4 bytes through 256 MiB; not model throughput |
| [Tiny eager](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/axo1gal6) | 88/88 gates pass | All eight arms; independent canonical reference; three Adam updates each |
| [Tiny graph](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/km5w3fog) | 88/88 gates pass | Same scope with two captured backwards and actual NCCL |
| [Pretrained canonical eager](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1ilvjheb) | **FAILED for NFR**; ordinary B passes 11/11 | NFR first raw-gradient comparison fails; loss checks pass |
| [Pretrained prepared eager](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/q9xr9ksc) | 11/11 operational gates pass | NFR; separate independent BF16 qualification remains failed |
| [Pretrained prepared graph](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7kkx637i) | 22/22 operational gates pass | Ordinary B and NFR; three Adam updates each; NFR BF16 qualification remains failed |
| [Full FP32 localization 01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/5gv7m5wl) | Both stages pass | Actual pretrained NFR weights, no DDP/graphs/optimizer |
| [Full FP32 localization 02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/627qt1c0) | Both stages pass | Repeat after correcting diagnostic metadata; no arithmetic change |

The tiny maximum raw-gradient relative L2 is `3.69222e-7`; the maximum relative
error measured against the **parameter update**, rather than pretrained weight
norm, is `5.78202e-6`. In the three-update pretrained prepared NFR checks, maximum
raw-gradient relative L2 is `4.95627e-9`, parameter-update relative L2 is
`1.49771e-8`, and Adam-moment relative L2 is `3.49359e-9`. Replicas agree exactly.

The independent pretrained NFR BF16 failure is **0.0340224273 relative gradient
L2**. It reproduces locally between selected-position and prepared dense-mask
losses before DDP. CE sums are exact; auxiliary sums differ slightly and the
normalized objective differs by `4.19435e-6`, within its frozen budget. Prepared
distributed passes therefore qualify execution of that path, not independent
BF16 equivalence or harmlessness for learning.

The separate full-FP32 comparison passes with gradient relative L2
`7.38198e-7` and objective difference `4.83649e-7`. It uses math SDPA and eager
native RT, so it does not erase the BF16 Flash/Triton qualification. No budget
was relaxed. Reports in prepared-reference mode expose `operational_status`,
`independent_reference_status`, `qualification_failures`, and non-gating
qualification rows explicitly.

## Actual cloud-restored fresh-process restart

| Stage / W&B | Result | Scope |
| --- | --- | --- |
| [Tiny write 01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/edayyrx2) | 11 checks pass | Original checkpoint and uninterrupted live-graph continuation saved |
| [Tiny resume 01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/0rd8zshb) | **FAILED before model mutation** | Safe deserialization rejects a `TorchVersion` metadata subclass |
| [Tiny write 02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/o34f5ijq) | 11 checks pass | Corrected metadata; newly saved and retained checkpoint |
| [Tiny resume 02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7du9uw9g) | 9 checks pass | New torchrun processes load the independently downloaded cloud copy; next update bitwise exact |
| [Pretrained write 01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/gpo6pnho) | 11 checks pass, 187.75 s | Full NFR checkpoint after one M2 update, then original live-graph M3 continuation |
| [Pretrained resume 01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/u3pnp0xi) | 9 checks pass, 129.57 s | Fresh processes restore verified GCS download before DDP/graph reconstruction; next update bitwise exact on both ranks |

The metadata fix converts approved scalar subclasses and string keys to builtin
types. It retains `torch.load(weights_only=True)` rather than relaxing the safe
loader. The first checkpoint and failed resume remain retained as evidence;
their successful upload is not restart acceptance.

The corrected tiny and full pretrained checks compare raw gradients, input and
noise fingerprints, complete model/Adam state, metrics, scheduler/counters,
data cursor, RNG state and actual RNG draws. Full pretrained acceptance is
B2/T16, same world size, hardware and runtime. It is not a T1024 cold-start
capacity result, an H200 portability result, or real-data loader recovery.

## Source lineage and retained attempts

Each report pins individual source hashes and includes a source snapshot. These
hashes, rather than the later branch HEAD, identify the code actually tested.

| Milestone source lineage | Change / applicable attempts |
| --- | --- |
| `79fc75b` | Initial opt-in distributed runner and probes; tiny eager/graph and first tiny write/resume |
| `0c77166` | Safe scalar metadata normalization; corrected tiny restart and original pretrained canonical eager attempt |
| `61d6d2b` | Separate prepared operational reference from independent qualification; prepared eager/graph and the shared source contract used by full pretrained write/resume |
| `28d6a93` | Bounded FP32 localization implementation; first full-FP32 diagnostic |
| `f5f3cd3` | FP32 diagnostic report-label correction; repeated FP32 diagnostic 02 |
| `c2df2f6` | Capacity ordering captures with real resident Adam moments; B8 acceptance passes, B16 pending |

The source contract for the full pretrained restart remains unchanged across
its write and resume phases even though documentation/diagnostic-only commits
advanced the branch. All pinned Python files were compared against the listed
source-content milestones. Reports retain exact per-file hashes.

## Completed B8 capacity check; B16 pending

`capacity-b8-m2-01` passes all **12 stages** on both ranks;
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kkmsgtfg).
This is physical B8 per rank, T1024, M2 accumulation, NFR K4 with native RT at
layers 0 and 15 on every pass, and full CE/auxiliary masks. It performs eight
complete Adam updates: three eager warmups and five measured graph updates.

| Quantity | Measured value |
| --- | ---: |
| Global valid input tokens per optimizer update | 32,768 |
| Measured complete updates | 5 |
| Total timed wall time, summing the slower rank for each update | 47.79149 s |
| Global valid input throughput | **3,428.2253 tokens/s** |
| Median complete update time | 9.56365 s |
| Peak allocated memory per rank, including setup | 34.86480 GiB |
| Peak reserved memory per rank, including setup | **50.35352 GiB** |
| Final sampled free memory per rank | **22.97083 GiB** |
| Whole probe duration, excluding retention | 524.64 s |

DDP preparation uses 20 backwards before Adam initialization. Three eager
updates create the real optimizer moments, then both graphs are captured using
the same DDP wrapper with those moments resident. One replay backward is
discarded without advancing optimizer/scheduler/token clocks. The five timed
updates follow. Gradient/input pointers, RNG preservation, actual Adam moment
residency, finite state and token counters pass throughout. Graph accounting
records two capture backwards and 12 replays: six local and six synchronized,
including the discarded priming update.

Timing includes CPU preflight/refill, graph replay, NCCL, clipping, fused Adam
and scheduler work. Fixture/noise generation, evidence/W&B logging and health
scans are outside the timer. Tokens are counted once globally, not multiplied
by K4. The full-valid single-document fixture does not qualify packed
multidocument semantics. This setup also does not qualify cold T1024 DDP
construction with restored Adam already resident. The older K2 measurements
use a different workload and are not a controlled throughput comparison.

B8 evidence retention is verified; see [storage receipt](storage-receipt.md).
At this snapshot, `capacity-b16-m2-01` is running. Its outcome, any additional
candidate attempts and final closeout retention remain pending for root.
