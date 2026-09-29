# Packed campaign readiness protocol

2026-09-29, from main `7b9c614` / PR 38. The user authorizes the next readiness
milestone on two H100s. This adopts the proposed continuous-stream policy in
the previous next-step plan and does not launch a quality campaign, choose a
production mixture or make an H200 throughput claim.

## Versioned policy

`continuous-stream-v1` concatenates complete documents in recorded order within
one existing split into consecutive nonoverlapping chunks. Stored terminal EOS
is preserved, never doubled. Embedded EOS is content, not boundary metadata.
Right-pad the final partial chunk. RoPE starts at zero for every row; no state
or label target crosses from one chunk to another.

CE includes every selected valid adjacent target, including EOS to the next
document's first token. NextLat latent pairs and KL triples require the same
true document. Ordinary causal attention, native RT history and shifted FBT
feedback/jitter continue across document boundaries within a row. Preserve
the isolated-document default and explicit rejection under that old policy.
One immutable policy must agree between recipe, loss configuration, eager and
prepared forward paths, CUDA graph and checkpoint metadata.

The legacy training `documents` counter counts nonempty physical rows; packed
telemetry will label it as row presentations and separately report actual
document segments/completions, valid inputs, loss targets, cross-boundary
targets, omitted cross-chunk targets, padding and dummy slots.

## Implementation and gates

1. CPU boundary oracles and all-eight-arm canonical/prepared gradient checks.
   Do not simply remove the packing rejection. Exercise moving boundaries,
   EOS at edges/content, masks, short tails and empty slots.
2. Immutable disk-backed chunk index over verified Dolma document shards;
   corpus/tokenizer/split/order/policy pins, true provenance, explicit committed
   cursor and pure peek. Form logical valid-token updates before rank/batch
   partitioning; keyed jitter must follow stable chunk coordinates.
3. Actual two-GPU tiny eager/graph checks under the stream policy. Then bounded
   pretrained B/NFR checks at short length, with the prepared BF16 operational
   reference and separate independent numerical qualification retained.
4. Actual-data T1024 checkpoint/continuation on two GPUs, starting physical B12
   per rank (B8 fallback). Use the proposed 524,288 valid input tokens/update,
   allowing padded final slots rather than changing the common token budget.
   Save at one completed update; compare next update on the original live graph
   with fresh processes restored from a generation-pinned, SHA-verified GCS
   copy. Resume loads actual Adam state before DDP preparation/capture, thereby
   testing cold T1024 setup memory. Compare batch/noise fingerprints, raw
   gradients, full model/Adam, RNG, scheduler and real committed data cursor.
5. Only a short complete-update rate from this loader/accumulation path. Record
   fixture generation separately; do not confuse useful-token rate with pass
   tokens, padding or the older K2 benchmark. No new kernel optimization sweep.

Existing numerical budgets remain unchanged: raw gradient elements atol 3e-5,
rtol 3e-4; loss sums atol 1e-5, rtol 3e-6; normalized objective atol 1e-6,
rtol 3e-6; parameter elements atol 3e-6, rtol 3e-5 and update-relative L2 <=1e-3;
Adam elements atol 3e-5, rtol 3e-4 and aggregate L2 <=1e-3. Restart equality is
bitwise and operational counters/storage/RNG are exact. Retain failures.

## Bounded BF16 follow-up

At the prior isolated initial NFR fixture, compare combined and individual
CE/latent/KL raw gradients for FP32 sparse reference, BF16 sparse, and BF16
dense prepared paths. Report norms, relative L2, cosines and parameter-group
errors. BF16-to-FP32 comparisons are descriptive, not a new claim that a strict
FP32 budget is the appropriate BF16 gate. No threshold relaxation, long
trajectory or optimization sweep. Existing 3.40224% failure remains until a
justified fix or explicitly supported precision decision addresses it.

## Interruption and scope

Root alone launches GPUs, inside the project container. Each stage has a fresh
directory, atomic evidence, source snapshots, W&B and external timeout. Short
GPU probes get 15 minutes; the T1024 real-data write/resume phases may each use
up to 20 minutes. Save/push implementation and retain completed evidence at
least every 20–30 minutes. Never overwrite old evidence/checkpoints or rely on
local SSD for durability. Keep each previous verified checkpoint until a new
one is safely retained. Any longer unsaveable span needs advance notice.

The seven-source corpus is a coverage fixture, not the production mixture.
Recovery remains same-world-size/same-runtime. H200 and changed rank count need
their own short acceptance; no performance extrapolation is qualification.
