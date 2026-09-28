# Combined T2048 benchmark interruption handoff

**Complete, 2026-09-28. Read results.md first.** Latest user instruction: test
Flash SDPA only, then review. No FA4, new T512 run, quality training or kernel
optimization is queued. The GPU was verified idle after all work.

## Result and validated scope

K2 FBT, native RT0/15 in 16-layer OLMo, NextLat SmoothL1+KL on both passes,
native ordinary RoPE. Runtime79e850a; report helper5f6c6ac. Production math and
all37pretrained sources match the historical T512 reference.

- Historical B128/T512:12,361.82inputtokens/s;65.113GiBreserved;12.891GiBfree.
- B16/T2048:7,501.52/s;40.986GiBsteadyreserved;35.977GiBsampledfree.
- Repeated B32/T2048:9,184.55/s;69.014GiBreserved;7.914GiBsampledfree.
- B32 equals the historical65,536inputtokens/update but is25.70%slower in
  throughput, or34.59%moretime. Repeat rates differ only0.0134%.

Four stages pass20/20gates and32physicalupdates; B2 excluded from performance.
Initial/terminal own eager/graph checks are exact.137CPUtests pass;652newsource
pairs verify. No new independent full-Adam trajectory or FP32 qualification.
Prior BF16/native-author qualifications remain. Historical T512 is not a new run.

AtT2048, actual two-RT-layer forward dispatch:4,094historytiles,4,088Triton
and6larger eager tiles. All4,094historybackwardcallsTriton. The6eagerrectangles
cover75.04%ofhistoricalpairarea, not75.04%ofmodeltime. RT batch128→32,4xsequence
and largerattention arithmetic are plausible slowdown contributors; unprofiled.

## Evidence and next action

All4stages retained and verified under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T165900Z/`.
See storage-receipt.md for the separate closeout bundle, hashes and references.
Local evidence: `.runtime/olmo-combined-long-context/`. No new full checkpoint
was needed for disposable timing updates; O1 starting weights remain retained.

The conversation interruption occurred while B32-02 was running; it completed
in the background. Do not rerun it. Every stage is complete and retained.

Pause for review. B32 is fastest tested for this fixed shape; B16 leaves room
for changes. If speed atT2048 becomes the priority, profile the long-context
RT path before choosing a fusion/compilation change. Do not infer authorization
for FA4 or a new training campaign from this benchmark.
