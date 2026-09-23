# Native RT efficiency, Stage A

Prospective bounded protocol, 2026-09-23, authorized after PR22. Stop at the
native optimization review point before the author-derived backend comparison.
No quality training, Q/K change, new attention kernel or production default.

## Fixed execution and three arms

Original native OLMo-1B step60000 (~252B tokens): 16 layers, D2048, H16/head128,
SwiGLU8192 per branch, tied50304 embedding/readout, nonaffine LayerNorm and
native FP32 split-half RoPE, no Q/K normalization. All arms use the same weights
and initialization, real-text fixtures and seed20260922. One unpadded document
per row, T512, one physical batch/update, no accumulation, one H10080GB.
BF16 mixed with FP32 parameters/gradients/Adam; TF32 off, deterministic ordinary
PyTorch Flash SDPA, autocast cache off, ordinary checkpointing on. RT uses
layers0/15, alpha1, Triton historical forward/backward, cast reuse, recompute
workspace. Combined adds FBT K2 and horizon1 NextLat. CE2048, KL128 throughout.

- `control`: original on-demand RoPE and full permanent QKV projection.
- `rope`: invocation/prepared-layout FP32 table reuse; permanent QKV unchanged.
- `both`: table reuse plus K/V-only permanent projection, using the existing
  packed weight's K/V row view. Temporary input QKV is unchanged.

No new parameters or persistent buffers. Both switches default off; preserve
the unchanged sequential oracle and checkpoint layout. Static guards own table
identity/storage/version/metadata and the execution flags. Captures must be
rebuilt after a flag/layout change. Dynamic calls build tables once per stack.

## Correctness gates before capacity

CPU checks cover exact RoPE outputs and source/parameter gradients, FP32 and
BF16 dtype restoration, explicit/offset/repeated positions, cached prefixes,
static ownership and no repeated trigonometric work. K/V checks cover independent
scan equivalence, packed Q-row zero contribution, parameter/input freezing,
shared calls, prefix/cache cotangents, alpha0/.37/1 and dyadic boundaries.
Include a combined-switch cached-prefix case. Existing tiny feature integration,
checkpointing, static-training and tiled suites remain the regression controls.
Matrix accounting is checked against executed mm/bmm counts.

Native B8/T512, half CE: ordinary, RT, combined control-versus-rope; RT and
combined rope-versus-both. Compare all-pass final hidden states, individual loss
sums, counts/ownership and every participating raw parameter gradient at the
identical checkpoint. RoPE reuse must be bitwise exact. For K/V projection,
global gradient relative L2<=1/64, each tensor L2<=1/32 and max absolute error
normalized by the reference tensor maximum<=1/16; relative loss L2<=1e-5;
hidden-state L2<=1/64 and normalized max<=1/16. Handle zero norms explicitly.
All losses, outputs and gradients must be finite. These are engineering screens,
not thresholds attributed to the RT authors. Preserve failures; do not widen
budgets after results. The prior F4 RT+FBT coordinate miss and broader roughly
18% BF16/full-FP32 initialization gradient difference remain qualified.

For each candidate: ten warmup backwards, capture, exact same-candidate eager/
graph losses and all gradients, changed tokens, repeated overwrite, three eager
versus three graph AdamW/clipping/scheduler updates with exact full model/moments/
counters, then changed-weights graph replay. Six physical updates per successful
correctness report, five reports expected. No long learning or checkpoint needed.

## Throughput and attribution

Fresh B64/T512 full-CE processes: ordinary control/rope, RT control/rope/both,
combined control/rope/both. Each: three preparation optimizer updates, ten warmup
backwards plus capture, five complete graph-backed update timings, then three
forward/loss/backward replay timings. Report synchronized wall/CUDA-event samples
and medians. Full update includes validation/copies, clipping, AdamW and schedule,
excludes logging/filesystem/profiler. Separate setup and steady allocated/reserved
peaks. Eight physical updates per successful timing report; eight reports expected.
Optional B128 RT control/both only if comfortably feasible; do not maximize VRAM.
Repeat a close comparison in reverse order before claiming a small gain.

Capture separate untimed RT control/both device traces after timing. Count actual
CUDA kernels and their durations; categorize RoPE/pointwise, permanent projection,
MLP/local backward and historical tiles where identifiable. Device-time sums and
nested annotations are not complete-step wall-time decompositions. A trace cannot
reliably separate all shared GEMMs; state that limit. Backward timer includes
forward/loss/backward; separate forward/backward timing belongs to the later block
comparison, not a new lifecycle change here. Profile results are explanatory;
the paired complete-update timings determine measured benefit.

Expected primary queue:13 reports,94 physical optimizer updates. B64 is the
conservative integrated comparison; it is not the later all-RT stack/B512 test.
Report unchanged parameter counts and revised analytic matrix FLOPs. The K/V-only
change saves12*B*T*D^2 operations per RT block call in this training execution
(forward plus backward/reconstruction), not one third of full-model cost.

## Provenance and retained evidence

Freeze sources and this protocol in Git before GPU work. Per-report source and
protocol hashes/snapshots, native checkpoint pin, device/runtime, explicit failures
and actual optimizer-step counters are required. Online W&B entity`taylorbollman`,
project`pretrained-fbt-rt-nextlat`, group`olmo-rt-efficiency`. Retain selected reports,
logs, traces, CPU results, summary, exact source snapshots and native checkpoint
reference in `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-rt-efficiency/`.
No weights or secrets in the small evidence bundle. Graph recovery/accumulation,
padded graphs, genuine multi-GPU and online readiness remain separate V4 work.
