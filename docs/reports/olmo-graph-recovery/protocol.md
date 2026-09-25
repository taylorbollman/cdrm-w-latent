# Single-device graph checkpoint recovery protocol

This is a bounded functionality check on the actual OLMo-1B step-60000 checkpoint.
It does not test learning quality, distributed resume, graph serialization or
saving while an uninterrupted live graph remains resident.

## Fixed configuration

Run `rt` and `combined` separately, physical B2/T512 initially (B8 is an optional
bounded follow-up). Full 16-layer model, native RT at layers 0 and 15; combined
uses K2 FBT and NextLat. Use the large-batch milestone's `compiled-native` arm:
native FP32 RoPE, rounded compiled ordinary SwiGLU, deterministic ordinary Flash
SDPA, fused AdamW, all ordinary layers activation-checkpointed; native RT uses
cast/RoPE reuse, KV-only writes and Triton forward/recomputed backward tiles.
BF16 mixed, FP32 weights/gradients/Adam, TF32 off, autocast weight cache off,
CE position chunks 2048, KL chunks 128, seed 20260922. Preserve full valid CE and
the deterministic real-text fixture rotations. One physical batch per update.
Clipping, optimizer and scheduler remain outside graphs. No numerical tolerance
is relaxed: recovery and own eager/graph comparisons require bitwise equality.

## Boundary and branches

1. Build a fresh model and optimizer; warm up ten backwards, capture and check
   every loss and participating raw gradient against CPU eager references.
   Execute two complete graph optimizer updates.
2. Load the next fixture batch; snapshot graph losses/all raw gradients on CPU.
   Synchronize, release graph result and graph, compare eager, discard the plan,
   clear gradients, then save one complete checkpoint on persistent project disk.
   Save strict runtime flags, fused optimizer identity, model/Adam/scheduler,
   counters, fixture cursor, source fingerprint and all RNG states. Check saving
   did not mutate the boundary.
3. Reference branch retains the same model and optimizer objects, builds a new
   plan/graph, verifies boundary gradients and eager/replay parity, then performs
   two continuation updates. Release its graph/plan, record complete state/RNG/
   cursor digests and free the model/optimizer.
4. Reconstruct the model and fused optimizer from scratch, strictly load the
   saved checkpoint, and compare boundary model/Adam/scheduler/counters/RNG/
   cursor. Build/capture checks must preserve RNG. Repeat the identical two
   continuation updates and compare complete final state and update records.

Both reference and restored branches rebuild their graph. There are six actual
optimizer calls: two preparation plus two per continuation branch, while each
branch's logical final counter is four. Successful optimizer calls are counted
and persisted by post-hook even if later scheduling/metrics fail. W&B operations
preserve experiment RNG.

One checkpoint (~14 GiB) is held at a time. Its bytes and source hashes are
recorded. Only after all checks pass and its hash is reverified is this disposable
two-update diagnostic checkpoint deleted. On failure it remains for diagnosis.
The original pretrained checkpoint remains retained independently. Keep local
JSON/logs/source/protocol receipts and upload them with the milestone evidence;
these short diagnostic weights are not useful training-quality artifacts.

## Execution

Run only inside the validated GPU container. Example:

```bash
python scripts/olmo_graph_recovery.py --case rt --batch-size 2 \
  --output-dir .runtime/olmo-graph-recovery/rt-b2-01
```

Use a new output directory and process for each case. The program checks disk
capacity and requires a project-local destination. W&B project is
`pretrained-fbt-rt-nextlat`, group `olmo-graph-recovery`. CPU tests cover orchestration,
checkpoint contract rejection and exact small-model recovery; only actual GPU
results can clear CUDA capture or fused-Adam integration.
