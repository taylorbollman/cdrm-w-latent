# Remaining two-GPU integration and independent-job isolation

2026-09-30. Prospective bounded functionality scope. No new native or isolation
GPU result is established by this document.

## Independent-job fault isolation

`scripts/olmo_topology_isolation.py supervise --scenario controlled|abrupt
--output-dir <new project directory>` launches two independent one-rank
`torchrun` jobs through the project Docker launcher, on GPU0 and GPU1. Each has
its own container ownership label, rendezvous, output directory, online W&B run
and local checkpoint-retention identity. A local PATH-scoped Docker shim adds
the name/label to `docker run`; the shared launcher is unchanged. Both outer
setup and inner worker lifetime are bounded. Cleanup authenticates ownership
before removing containers and confirms no owned container remains.

The tiny FP32 ordinary campaign model reuses the current campaign CUDA graph
and optimizer path. Both jobs perform two real optimizer updates and save a
hashed, CPU-readback-verified model/Adam/counter checkpoint. The supervisor then
requests either cooperative boundary stop or a deliberate worker `os._exit(73)`
without graph/process-group teardown. The abrupt case synchronizes its known
intent to W&B before aborting; it does not test recovery of an unsynchronized
tracking client. Torchrun's outer launcher commonly returns 1 for a worker exit
73; both values and the launcher's explicit worker-exit evidence are recorded.

The peer keeps its graph and optimizer alive at the two-update boundary. Only
after the supervisor observes the victim's expected exit does it release the
peer to run four more changed-input graph optimizer updates. This deliberate
handshake proves progress before and after failure without confusing a peer
that had already finished with a surviving job. It does not measure throughput
or prove that CUDA work overlapped the exact instant of failure. The peer saves
update6, verifies its bytes and CPU readback, then restores the checkpoint into
a fresh tiny CPU model/Adam and compares state exactly before clean exit.

The expected victim exit must not trigger the allocation benchmark queue's
cancel-both behavior. Unexpected failures stop only resources owned by this
new test. This is independent-job isolation, not recovery from a failed rank
inside one distributed job, native-model precision, H200 acceptance or a claim
that eight concurrent jobs have been tested. Parent closeout owns GCS retention;
worker receipts establish local checkpoint content and distinct identities.

## Native component coverage gap

The [two-GPU campaign integration](../olmo-campaign-two-gpu/results.md) covers
all eight arms in tiny FP32 eager/graph fixtures, while its full native checks
cover B and NFR. Subsequent packed T1024 runs add native F and NF, including
their actual Adam updates and checkpoint/evaluation lifecycle. The completed
[allocation milestone](../olmo-gpu-allocation/results.md) adds matched one-rank
and two-rank native B/NFR performance clones. The older
[F4 resource matrix](../olmo1b-f4/results.md) includes all eight native arms but
uses historical T512/K2 execution and cannot substitute for current integration.

| Arm needing a current native smoke | Passes / modules | Main integration question |
| --- | --- | --- |
| N | K1, NextLat predictor and regression/KL, no RT or active fusion | Predictor gradients and optimizer ownership with fusion disabled |
| R | K1, native RT0/15, no NextLat or active fusion | RT-only graph/reducer ownership and CE normalization |
| NR | K1, native RT0/15 plus NextLat | Joint RT/predictor participation without feedback |
| FR | K4, fusion plus RT0/15 on all passes, no NextLat | Fusion/RT shared-weight pass gradients with predictor absent |

Recommended minimal scope is two ranks, T1024 packed data, conservative B12
per rank, two accumulated slots, and two changed-input optimizer updates per
arm. This tests the production physical shape without spending a full 512-row
learning update; the smaller effective batch must be labeled accordingly.
Include one padded slot or document boundary when practical, but do not repeat
the full tiny exhaustive masking matrix already accepted.

Use the original authenticated OLMo weights and explicitly fresh module/Adam
initialization for a clean functionality fixture. If an existing fusion-only
import is used to moderate FR startup, label that ancestry separately; do not
silently splice a trained full NFR checkpoint into a different active optimizer
layout. These are not matched quality pilots or recommendations for scientific
initialization.

Check the recipe actually selects the intended K/RT layers/objective weights;
global eligible counts and loss denominators; the expected active parameter
and Adam ownership sets; finite losses, gradients, parameters and moments;
changed weights after updates; matching distributed replicas; stable graph
storage; and clean graph/reducer/process-group teardown. Keep captured versus
prepared-eager checks bounded if a new participation concern appears. Existing
precision qualifications remain unchanged: broad FP32 comparisons or a new
BF16 trajectory investigation are not prerequisites for this narrow smoke.

A full 15GB checkpoint upload for every arm would add little new information
after the existing generic native lifecycle coverage. Verify ownership and
serialization metadata for these arms; add a full roundtrip only if a new
state-layout issue is uncovered. Topology migration and independent-job failure
isolation remain separately named acceptance scopes. Eight-rank, H200,
heterogeneous full-node concurrency and physical-batch tuning await the target
hardware.
