# Ordinary OLMo T2048: completed work and interruption handoff

2026-09-28. The authorized one-H100 ordinary-model benchmark is complete.
Read results.md, usage.md, protocol.md and storage-receipt.md in this folder.
No GPU or quality-training job is queued; the H100 was verified idle.

The user explicitly requested reuse of the saved T512 reference. No new T512
run, including FA4 at T512, was performed. Historical B128/T512 gives44,041.286
input tokens/s; new B32/T2048 uses the same65,536input tokens/update.
Repeated B32 rates: Flash SDPA41,479.711/s; FA443,134.687/s. FA4 gains3.99%,
with no meaningful memory saving. Compared with historical T512, rates are
5.82%/2.06% lower. Both T2048 backends reserve37.518GiB and leave40.899GiB
sampled free atB32. B16 gives41.374k/43.065k with50.260GiB free; scaling
B16→B32 adds only0.2–0.3%, so optionalB48/B64 were not needed.

Recommend ordinary B32/T2048 as a comfortable operating point; B16 is useful
when another feature needs headroom. This is not a maximum-capacity result or
an optimal learning batch. FA4 remains explicit opt-in; defaults are unchanged.

Runtimeedf3d0b extends only benchmark selectors/reporting and frozen protocol
requirements.111focusedCPUtests pass (3.23s,67installedJITwarnings). The seven
new GPU reports comprise six passing capacity runs plus one retained loss-only
numerical failure,56physical optimizer updates. All35operational checks pass.
The diagnostic's globalgradientL2.010833/outputL2.006477 and allper-tensor
budgets pass; strictrelativeCE.000775228 fails1e-5, absolute7.2306e-5nats/target.
Do not relabel this stage passed or insert its timing into performance figures.
Owninitial/changedweightgraphchecks are exact; no new independent Adam-trajectory
comparison was claimed. Older precision qualifications remain open.

Active model is originalstep60000 OLMo:16layers,D2048,16heads,1,176,764,416active
parameters. Frozen unused32MiB fusion storage is recorded separately. No RT,
FBT or NextLat executes. Both arms use DaoFP32RoPE, compiledroundedSwiGLU,
fusedAdam, all-layercheckpointing, BF16mixed/FP32trainingstate and CUDAgraphs.
One microbatch/update, noaccumulation; full-valid independent document rows.
The saved T512 environment/package versions and37pretrained runtime sources
match, but it is a historical comparison rather than same-day interleaving.

Local evidence: `.runtime/olmo-ordinary-long-context/`.
Verified GCS prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T163200Z/`.
All7stage receipts plus a separate closeout bundle retained. Audit checks
1,148new source pairs and214dependency pairs without mismatch. Original O1
weights remain retained; disposable capacity updates need nofullcheckpoint.
The closeout also copies the explicitly labeled historical reference report,
sources/dependencies and its originalreceipt. Prior evidence was not modified.

Container-only GPU execution remains required. Use installed Flash-Attention
namespace; FA4 4.0.0b20/CuTE DSL4.6.0.dev0 are already installed. CPUretention uses
GPUdisabled container and env-uGOOGLE_APPLICATION_CREDENTIALS for mounted ADC.
Persistent disk remains about19GiBfree. Existing logs/queue scripts preserve
execution commands; all former sessions33439/58032/70777 have finished.

Next: choose the real data/training workload. This benchmark does not validate
packed/padded data, T2048 RT/FBT/NextLat, accumulated graph replay or long-run
training equivalence. No additional experiment is automatically queued.
