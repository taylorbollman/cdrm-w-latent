# Next readiness milestone: packed data and recoverable campaign execution

2026-09-29. This is an implementation plan, not a training launch or a choice of
production corpus mixture. Continue the user's functionality/readiness focus.
Reuse the qualified campaign DDP runner and checkpoint format; preserve the
[BF16 qualification](qualification-plan.md). The completed resource check
recommends B12/rank as a starting point: 4,311 input tokens/s across two GPUs,
59.06 GiB peak reserved and 14.23 GiB sampled free per GPU. This does not yet
qualify the cold restart or real-data accumulation described below.

The reusable input is the verified seven-source Dolma v1_5 coverage fixture:
28 complete-document shards, 12,512 documents and 7,054,230 tokens, restored from
GCS on this VM. See [data evidence](../olmo-document-shards/results.md). This is
not the production mixture. Current model adapters reject multiple documents
per row; current `campaign_data.py` produces isolated windows with one-token
overlap. Neither behavior implements the stream policy proposed below.

## 1. Freeze a versioned stream policy, then implement it

Review/freeze these semantics before implementation. The recommendation is
OLMo-style continuous-stream chunks at T1024, not document-isolated attention
packing. Concatenate complete documents in a pinned order, preserve the stored
terminal EOS and true boundary metadata, and take consecutive nonoverlapping
chunks. Do not introduce EOS-aware offsets or reuse the isolated-window stride
of 1023. Literal EOS inside document content is not a document boundary.

| Component | Proposed behavior within one chunk |
| --- | --- |
| CE | Predict token `t+1` from logit `t`, for valid adjacent positions. Include internal EOS and the next document's first token after EOS. The final logit has no target in another chunk. |
| NextLat | Retain same-document latent pairs and KL triples, separately from CE selection. Attribute terminal EOS to its actual document. Preserve all stop-gradient and pass-weight rules. |
| Ordinary attention | Ordinary causal attention across document boundaries; no block-diagonal isolation or EOS reset. |
| RT | Historical attention/memory continue across EOS within the chunk. Start fresh at each chunk; do not carry caches between rows. |
| FBT | Proposed shifted feedback and keyed jitter continue across valid adjacent positions, including EOS boundaries. Position zero uses its ordinary embedding on every pass. This differs from the current same-document eligibility rule and requires explicit policy selection. |
| Positions/tails | RoPE positions run from zero within each chunk. Retain a final incomplete fixture chunk with right padding and audited valid counts; never train on padding. |

For example, `[a, EOS_A, b, c, EOS_B]` includes the CE target
`EOS_A -> b`; its latent pair is excluded, as are KL triples spanning that
document boundary. Removing auxiliary targets does not isolate attention,
RT memory or FBT feedback.

Implement this as an opt-in policy propagated through eager FBT, prepared
layouts and canonical/dynamic loss construction. In particular,
`build_nextlat_masks` currently applies same-document adjacency to **CE too**;
it cannot implement stream CE merely by supplying different masks. Preserve
real document IDs rather than pretending a packed row is one document. Keep
legacy isolated-row behavior and its rejection guards unchanged by default.

Use small CPU oracles for exact chunks/counts, EOS at positions zero/last,
long documents crossing chunks/shards, embedded EOS, short tails and empty
slots. Check eager/prepared forward and per-loss gradients on the eight tiny
arms. Then use a bounded two-GPU B/NFR check with changed boundary layouts,
including an empty final synchronization slot. Compare against the intended
continuous-stream reference, not concatenated independent-document outputs.
Do not build a new attention kernel or launch a broad numerical sweep.

## 2. Connect verified shards to the real training cursor

Add a disk-backed stream/chunk index over `document_shards.py`, retaining corpus,
tokenizer, split, order and packing-policy hashes. Keep the fixture's existing
splits; do not choose a production mixture, resample sources or cycle data
silently. Record source/document offsets so chunk provenance survives both
document and shard boundaries. Preserve complete tokenization for future
context lengths; no retokenization or permanent truncation is needed.

Define logical update membership before rank/microbatch partitioning. Preserve
the proposed common valid-token budget, recording any final partial update
explicitly. Key jitter by logical update/pass and stable chunk coordinates,
not rank or worker scheduling. Report valid inputs, CE/latent/KL targets,
omitted cross-chunk targets and padding separately; FBT passes never multiply
the input-token clock.

Replace the restart probe's synthetic cursor with the actual next committed
stream/chunk position and logical-update index. Pin any shuffle/index state;
unconsumed prefetch must not advance the committed cursor. Keep equal numbers
of physical slots across ranks, filling unused slots with supported empty rows.
The shared global per-term normalization already supports those unequal counts.

Rehearse one save/recovery on a bounded actual-data slice: save at a completed
update, retain and verify GCS bytes, restore into an empty SSD directory, start
fresh processes, and compare the next batch/noise hashes, raw gradients, Adam,
counters and cursor with uninterrupted continuation. Reuse existing safe-load
and live-graph checkpoint machinery; do not invent a second recovery format.

## 3. Qualify the selected T1024 setup and practical batch plan

Start at the measured B12/rank, retaining B8 as a fallback. Test
cold reconstruction at T1024 with loaded model **and Adam state resident before
DDP warmup and graph capture**. Current capacity includes resident Adam during
capture but does not itself establish this cold-start peak. Measure both ranks'
setup peaks and sampled free memory; retain a smaller fallback batch if needed.

The actual-data recovery rehearsal can supply this check, avoiding a redundant
large checkpoint exercise. Include one logical update at the intended
accumulation length: M2 capacity timings alone do not qualify the proposed
524,288-input-token update plan. Add only a brief useful-tokens/s measurement of
the resulting loader/refill path. This is readiness and budgeting, not a new
batch-maximization or quality experiment.

## 4. Resolve or explicitly retain the BF16 layout qualification

The observed NFR sparse/dense raw-gradient difference is about 3.4% before DDP;
prepared local/DDP execution agrees, and sparse/dense full-FP32 agreement passes.
None of those results establishes BF16 harmlessness or a training-quality claim.

Reuse the existing fixed fixture to compare both BF16 paths with a common FP32
reference. Separate CE, latent and KL contributions; if needed, hold hidden
inputs and incoming gradients fixed to distinguish loss/projection arithmetic
from its propagation through RT. Start with the existing initial state and at
most one saved updated state. Inspect component scales and one Adam update,
without a long trajectory or new parameter sweep. Freeze tolerances before
checking a candidate change and retain failures unchanged.

If one operation explains the discrepancy, make the smallest justified change
and rerun its affected checks. Otherwise report the remaining scope explicitly
and make a documented precision choice before claiming numerical clearance.
Packing/readiness work must not silently turn this qualification into a pass.

## 5. Bound work, preserve progress and qualify final hardware

Keep code, manifests and small evidence on persistent storage. Stage data and
checkpoints on SSD; promptly upload and verify recovery checkpoints in
`gs://fast-chunks`, keeping the previous verified checkpoint until replacement
is complete. Save resumable progress at least every 20–30 minutes. If a required
operation cannot meet that interruption bound, report it before starting.

Finish this milestone with a functioning packed-data trainer, a reproducible
recovery example, measured batch/headroom and an explicit numerical status.
Generation/evaluation adapters, final data mixture, LR calibration and quality
training remain subsequent decisions.

Two H100s qualify this tested world size and environment. H200 or a final larger
GPU count requires a short NCCL, update/restart and memory/throughput acceptance
at the selected shape; memory capacity alone does not certify compatibility.
Keep logical data/noise and token budgets fixed across hardware. Changed-world-
size checkpoint resume is currently unsupported and needs separate partition,
cursor and normalization checks if it becomes necessary; do not imply it from
successful same-world-size recovery.
