# Numerical localization progress

2026-09-29. Active branch: `feat/olmo-precision-localization`.

The user authorized the next numerical milestone after packed-data/recovery
readiness. The [protocol](protocol.md) is drafted before execution; root will
freeze it with the diagnostic source inventory. At this entry, **no GPU result
from this milestone is available**.

## Starting evidence

- Packed T1024/B12, two-H100 deterministic recovery passed exact next-update
  comparisons, including real data/noise, gradients, Adam/model, cursor and RNG.
  Original nondeterministic resume failure and the isolated Flash repeatability
  checks remain retained. This is operational acceptance, not precision
  equivalence.
- Original isolated NFR sparse/prepared BF16 gradient difference: 3.40224%
  combined; CE alone agrees. Packed NFR separately retains a 1.6953% layout
  failure.
- Both BF16 layouts differ about 86% from the common FP32 combined gradient in
  the initial T16 fixture. That comparison also changes math/Flash SDPA and
  eager/Triton native RT. The responsible operation and practical consequence
  remain unresolved.

See the prior [next steps](../olmo-packed-campaign/next-steps.md),
[precision assessment](../olmo-packed-campaign/precision-assessment.md) and
[storage receipt](../olmo-packed-campaign/storage-receipt.md). Do not repeat or
relabel those old runs as new acceptance evidence.

## Work in progress

| Work item | Owner | State |
| --- | --- | --- |
| Six-case CE/combined precision/backend bridge | Root | Implemented; 13 focused CPU checks pass; GPU validation pending |
| Fixed-hidden-state latent/KL sparse/prepared cotangents | Runner agent | Preparing bounded loss-level probe |
| Protocol and cross-compaction record | Data-review agent | Drafted; awaiting root source freeze |
| GPU execution and artifact retention | Root | Not launched at this entry |

The initial model bridge is FP32 math/eager, BF16 math/eager and BF16
Flash/Triton, using the same initial NFR T16/B2/two-virtual-rank fixture. The
BF16 bridge retains mixed RT attention and other runtime flags; only ordinary
SDPA and RT tile forward/backward selection differ from the BF16 endpoint.
Determinism is configured before CUDA, TF32 remains off, and no optimizer,
DDP or CUDA graphs are part of these new probes.

The bridge comprises six aggregate gradient cases, each with two physical
fixture records: 12 model-backward calls in total. Root's focused bridge and
existing-component CPU suite passed 13 tests. These tests support preparation;
no new GPU result or numerical acceptance is available at this entry.

The auxiliary probe is fixed to eight aggregate loss-gradient cases, each with
two physical records: 16 loss-only backward calls. It reuses the
BF16 Flash/Triton forward anchor exported by the bridge, preserving actual
hidden/embedding dtypes and payload/hash provenance. Readout/predictor values
are reconstructed and hash-checked from the base checkpoint and existing seed;
there is no large duplicate weight dump or additional backbone forward.

Keep each phase bounded by 900 seconds. Save atomic case progress and retain
source/evidence in GCS at least every 20–30 minutes. Run IDs, immutable report
paths and verified receipts will be added here as cases complete. A timeout or
failure is retained rather than overwritten.

## Decision record still pending

After the initial evidence, choose at most one crossed-backend condition or a
targeted cotangent/precision-boundary follow-up. Record its hypothesis and
controls before launch. A real packed-data spot check follows a localized
finding when needed; a broad sweep or Q/K-normalization transition is not
authorized by numerical completion alone. Existing BF16 qualifications remain
open until the results justify an explicitly documented resolution.

Implementation/source freeze: bridge and auxiliary helpers independently reviewed,
no blocking findings. Combined diagnostic CPU scope33 passed in5.14s; final
auxiliary scope20 passed in3.07s (overlap, after reporting/source-pin polish).
GPU queue begins with bridge-01, then auxiliary-01 using its pinned JSON anchor.
Root alone launches. Evidence root `.runtime/olmo-precision-localization/`;
cloud namespace
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T043105Z/`.
These are single-process numerical probes; the retention namespace does not
imply actual DDP or two-GPU execution. Both H100s were verified idle before
launch; only CUDA device0 is used. Protocol and helper source bytes now frozen.

Initial bridge completed operationally:6aggregate cases/12physical backwards,
51.88s, W&Bp6oooxmt. All76source snapshots verified; inputs/recipe/source and
prior endpoints reproduce exactly. ReportSHA39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b.
CombinedBF16math/eager vsFP32=81.4996%; production=85.96195%; production vs
bridge=75.9402%. CEcounterparts=100.0289%,95.9337%,79.8459%. No numericalclearance.
Bridge retained with exported3.52MB JSONanchor SHA
`aeab58a88c7eba15448a1b7630c9af747e492b3760b2364da5cd53380e063b27`.
Auxiliary01 completed8cases/16loss backwards in34.72s, W&Bmdo63etu; reportSHA
`c4946da63c6275a4fcd926292b0296337f57d10553233c7184e94e38b617d5fc`.
Fixedhidden BF16prepared/sparse differences:latent0.1416%,KL0.2016%; FP32below
8e-7. Auxiliary retention running. Adaptiveprotocol freezes onecrossedbackend
CEcondition Flash/eager plus2recomputed reference cases. Precisionagent owns
newcrossscript/tests; oldruntime/protocolsources stayfrozen. Root alone launches.
