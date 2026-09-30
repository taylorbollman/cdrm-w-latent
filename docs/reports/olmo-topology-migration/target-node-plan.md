# Work reserved for the destination GPU node

This is the next hardware-dependent scope, not a claim that the current
migration milestone has already passed. See results/progress for acceptance.

1. Restore the committed code, pinned container/runtime and selected checkpoint
   plus ordered-data assets. Authenticate cloud generations and bytes before
   use. Confirm GPU model/count, topology and actual container CUDA/NCCL runtime.
   Cloud credentials on GCP do not establish credentials on the destination.
2. Repeat a short, fresh-process same-topology restart on that runtime. The
   current H100 results do not certify different drivers, kernels or hardware.
   Declare a new topology transition explicitly when changing rank count; retain
   the original logical cursor, token LR plan and Adam, map surviving RNG streams
   and seed added ranks. Do not relabel a finite 128-update schedule as longer.
3. Qualify one eight-rank graph job and simultaneous independent jobs on the
   actual node. Use distinct device sets, rendezvous, output/checkpoint ownership
   and online run IDs. The two-GPU failure test proves a paused peer's live graphs
   can resume after another independent job exits; it does not prove eight-job
   interference behavior or survival of a failed rank within one distributed job.
4. Calibrate physical batch and memory on H200 before selecting eight singles,
   four pairs or one eight-rank layout. Maintain the chosen real global batch and
   count original input tokens once despite FBT passes. Report padding and
   per-experiment latency as well as aggregate tokens/GPU-second. Larger memory
   may allow faster RT physical batches, but the gain is not established here.
   At a 512-row effective batch, include candidates that divide that budget
   cleanly: for eight ranks, physical B16 or B32 avoids padded rows. A different
   candidate can still win on measured throughput; divisibility is not a rule
   that overrides measurement.
5. Only after runtime/restart/allocation acceptance, choose the learning
   continuation or comparison. Heavy clipping at bare RT startup and whether
   later FBT passes improve CE remain distinct scientific/optimization questions.
   The saved NFR128 scientific endpoint stays available; readiness replay through
   127→128 is not an extension or evidence for update129 onward.

These tests benefit from the actual destination node or additional throughput.
Repeating target-specific eight-rank/H200 questions on two H100s would not answer
those questions. No eight-GPU or H200 performance has been inferred from memory
capacity alone.
