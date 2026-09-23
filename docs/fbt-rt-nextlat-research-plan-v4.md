# OLMo / RT / FBT / NextLat: functionality and execution plan

Updated 2026-09-23. **Current authoritative forward plan.**

This supersedes the next-experiment queue in [v3](fbt-rt-nextlat-research-plan-v3.md).
The user's new priority is confidence in functionality, numerical health,
integration and reasonable execution cost, before asking which model wins.
Completed O1–O5e evidence remains valid within its recorded scope.

**Status: approved; F1/F2 and F3 graph, forward/backward fusion and bounded
backward workspace are complete within their measured scopes.** Read the
[F3d assessment](reports/olmo1b-f3d/assessment.md),
[results](reports/olmo1b-f3d/results.md) and [usage](olmo1b-f3d-usage.md).
Ten final GPU reports pass 90 gates and 428 scoped CPU tests pass . RT and combined
K2+NextLat have exact same-candidate graph/full-Adam parity throughT1024.
Initial gradients versus materialized F3c pass unchanged budgets, with maximum
aggregate relative L2 .00237615 across the three native checks.

Opt-in `backward_memory="recompute"` removes full backward probability/error
arrays. In three fresh paired capacity cases it saves 1.75–3.50 GiB allocated
with 0.39–0.74% lower measured throughput. RT B128/T512 now reaches26.26kinput tokens/s
at 38.60 GiB; combined B64/T512 10.93k/s at 39.09 GiB; combined B16/T1024 8.58k/s
at 31.29 GiB. Isolated reconstruction workspace grows approximately linearly through
T2048. Complete-model memory is not claimed linear. Materialized remains default.

Ordinary layers use deterministic PyTorch Flash, selected RT tiles use Triton.
Forward rectangles larger than 256 retain eager fallback; recompute backward
supports historical rectangles through 2048. Installed FA4/CuTE passed its separate
F3b smoke, not native RT integration. Native checkpoint/RoPE/loss/QK math and
parameter counts stay fixed. All F3d full-model checks select only layer 0 for RT.

**Next:** bounded multi-layer/all-layer RT and longer-context integration/resource
coverage using the new path, then complete F4 runtime cards. The analytic ledger
covers all eight combinations; its current estimator remains materialized and
now documents the recompute correction. Native T2048 complete updates, padded
graphs, graph recovery/accumulation and actual two-GPU execution remain open.
Local writer/finish VJPs and discarded permanent-Q computation are separate
performance opportunities. Preserve native Q/K math. No GPU or quality run is
queued; deferred quality comparisons need a later decision. Read the
[handoff](fbt-rt-nextlat-handoff.md) first after compaction.

## 1. Scope and working principles

Keep original OLMo-1B, step 60,000 / approximately 252B pretraining tokens.
The pinned native revision is 81b71efbce6f4dada57c94860301af4298bcd351;
source, tokenizer, checkpoint hashes and retained artifacts remain in the
[selection audit](olmo-1b-250b-checkpoint-selection.json) and handoff.
This is the 16-layer, width-2048 model with 16 full-MHA heads of width 128,
SwiGLU intermediate 8192, tied 50,304-row embedding/readout, native RoPE,
non-affine LayerNorm and **no Q/K normalization**. Native context is 2048.
Do not silently substitute OLMo 2, OpenELM or the historical A5 architecture.

Build confidence in all eight independent on/off combinations of RT, FBT
and NextLat. This means short operational checks, not eight learning contests.
Keep a precise ledger of what passed at which layer selection, length, batch,
precision and backend. Finite gradients alone do not establish correctness;
agreement with an independent reference alone does not establish efficiency.
Neither establishes useful language modeling without later experiments.

Three qualifications improve the user's proposed sequence:

1. Take brief execution measurements early enough to identify the real
   bottleneck, then make the final resource comparison after relevant changes.
2. Treat Q/K normalization as a model change, separately from arithmetic
   precision and kernel choice. Diagnose before changing the pretrained math.
3. Begin two-GPU correctness on the existing validated backend when hardware
   is available; it need not wait for an optimized RT attention kernel.

Reuse existing checks. Add focused tests for changed behavior and uncovered
integration risks. Escalate numerical investigation only for a material
discrepancy, nonfinite value, missing gradient, or concerning scale behavior.
A short fixed-batch learning check can help establish that optimization works;
outperforming the ordinary model is not an acceptance criterion.

## 2. Terminology and exact current semantics

| Term | Meaning in this project |
| --- | --- |
| Ordinary / base | The native pretrained OLMo stack without RT, FBT or NextLat. |
| RT | A selected attention block stores keys/values derived from its completed output, creating recurrence through time inside that block. |
| FBT | The previous token's top-layer state feeds back into the input of the shared stack through fusion. |
| Fusion | Two learned matrices combine the previous top state with the current token embedding; the result supplies the feedback-conditioned stack input. |
| K1 / K2 / K3 | One / two / three total whole-stack passes in finite-pass FBT. K includes the ordinary bootstrap pass. |
| Exact online | Sequential execution using the freshly completed previous-token feedback state. “Exact” describes the recurrence semantics, not exact arithmetic. |
| NextLat | A training-only next-latent predictor with regression and distribution-matching losses. It does not generate a separate latent rollout at inference. |
| Alpha | RT input-derived versus output-derived memory blend. |
| Beta | Ordinary embedding versus fused-feedback input blend. |
| Gamma | Weight of the aggregate extra-pass training loss. |

**K2 is an ordinary pass followed by one feedback pass. K3 adds a second
feedback pass.** These passes reuse one backbone; K is not the layer count
or the number of independently parameterized models.

### RT

For a selected block, let x_t be its input and z_t its completed output.
Its current query and temporary key/value come from native LN(x_t).
After processing position t, the block publishes permanent memory from

\[
m_t=(1-\alpha)x_t+\alpha z_t,\qquad
k_t=R_t W_K\operatorname{LN}(m_t),\qquad
v_t=W_V\operatorname{LN}(m_t).
\]

Attention at t uses permanent memory from earlier positions plus the current
temporary entry. It must not use the current permanent entry before z_t exists.
R_t denotes native RoPE; stored keys remain unrotated under our cache contract.
Alpha 0 recovers ordinary memory; alpha 1 is full output-derived RT.
The current RT transformation adds no parameters.

The initial pretrained-model selection remains **layer index 0 only**;
the other 15 layers are ordinary. This is distinct from the earlier synthetic
restricted-first-layer architecture. Additional RT layer selections must be
named in each configuration and separately exercised.

### Fusion and FBT

Let h be the previous token's post-final-LayerNorm top state and e the current
token embedding. The implemented fusion is

\[
f(h,e)=s\,\operatorname{RMSNorm}
\left[(W_Uh)\odot\sigma\left(W_G\operatorname{RMSNorm}(e)\right)\right],
\qquad
x^{\mathrm{stack}}=(1-\beta)e+\beta f(h,e).
\]

The new branch uses FP32 RMS reductions, epsilon 1e-5, and fixed scale
s = 0.03707655891776085, measured from the original embedding matrix.
W_U and W_G are learned 2048-by-2048 matrices. At beta 1 the fused input
replaces the ordinary embedding input; this is not simply an additive skip.
Document starts retain ordinary input and reset feedback.

Finite pass p uses h_(t-1) from pass p-1, with cross-pass gradients attached
and fresh layer history per pass. Pass 0 is ordinary. **When FBT is on, RT
selection applies to extra passes, so K1 does not exercise RT.** With FBT off,
one standalone RT pass is supported. Beta 0 bypasses fusion exactly but does
not disable RT in extra passes.

Exact online execution uses the freshly completed h_(t-1), rather than its
previous-pass approximation. Consequently K2 and online outputs need not
match. Compare identical execution semantics for precision checks; use the
existing tiny causal-convergence fixture for finite-versus-online correctness.

### NextLat and objective weights

The predictor maps (h_t, e_(t+1)) to a prediction of stop-gradient h_(t+1).
Its latent regression uses same-document pairs; its teacher-to-student KL
uses valid triples and concerns token t+2. Source states and conditioning
embeddings remain attached. Target states and the auxiliary readout use are
detached; ordinary CE and input lookup still train the tied embedding.
One shared predictor serves all passes and is omitted from ordinary LM inference.

Preserve the current finite-pass objective during functionality work:

\[
L=L_0+\frac{\gamma}{K-1}\sum_{p=1}^{K-1}L_p \quad(K>1),\qquad
L=L_0 \quad(K=1),
\]

where each L_p combines CE, weighted latent regression and weighted KL,
each divided by its own valid-target count. Report final-pass CE separately.
At K2/gamma1 this has two units of CE weight. The earlier suggestion to average
all pass losses concerns later comparison design; it is not a required fix.
Alpha, beta and gamma are configured or scheduled controls, not learned gates.

## 3. What already works, and what remains unestablished

| Area | Existing evidence | Remaining scope for this program |
| --- | --- | --- |
| Native import and RT math | O1/O2 native checkpoint fidelity, independent sequential oracle, tiled forward and first-order backward, masks/positions/caches and fractional alpha | New execution paths, representative multi-layer selections and intended runtime lengths |
| NextLat and optimizer | O3 loss indexing, detachments, gradients, accumulation, checkpoint recovery and bounded single-H100 complete steps | Combined repeated-update health and distributed integration |
| Eight combinations | O5a tiny independent objective/all-active-gradient checks for all eight; short actual-checkpoint combined FP32 and BF16 checks | Repeated actual-checkpoint updates for every combination, then bounded T512 operation |
| FBT finite/online | Independent causal/cache checks; O5d fixed-weight learned K2/K3/K4 and online evaluation through T512 | RT+NextLat online integration at representative lengths and efficient execution |
| Learning stress evidence | O4–O5e finite training and retained checkpoints; adaptation sometimes helps and sometimes hurts | Use these observations to choose health probes, not to require a quality win now |
| Attention performance | Ordinary SDPA and eager native tiled RT; some historical bounded timings | Actual backend dispatch per layout; optimized native RT forward/backward and integrated costs |
| Multi-GPU | Not validated | Genuine two-GPU complete updates, resume, scaling and memory measurements |

Sources: [O2](reports/olmo1b-o2/results.md), [O3](reports/olmo1b-o3/results.md),
[O5a](reports/olmo1b-o5a/results.md), [O5d](reports/olmo1b-o5d/results.md),
[O5e](reports/olmo1b-o5e/results.md). Historical reports retain their dated
“next step” language; the present plan determines what comes next.

The short O5a combined BF16 fixture had approximately 1.51% aggregate gradient
relative-L2 difference from FP32. This is useful prior evidence, not a universal
acceptance threshold. Preserve O2's documented FP64-backed raw-input roundoff
qualification and existing meaningful error budgets.

## 4. F1 — Common integration checks and capability ledger

**First authorized implementation PR.** Add a small, reusable configuration-driven
runner and machine-readable results, extending the current platform rather than
creating eight training scripts.

- Enumerate all eight feature combinations. Default FBT checks use K2; add a
  representative K3 all-three case. Keep native checkpoint/source pins fixed.
- Reuse existing tiny independent objective/gradient checks. At actual model
  size, start with B1–2 / T32–64 and approximately 3–5 complete BF16 mixed AdamW
  updates per combination. Keep FP32 parameters, gradients and optimizer state.
- Verify active-gradient ownership, finite losses and updates, tied parameter
  identity, inactive parameter invariance, and repeated calls to shared weights.
  An intentionally disabled branch need not have a nonzero gradient.
- Exercise zero, fractional and full alpha/beta transitions in selected cases,
  avoiding a full combinations-times-precisions-times-coefficients sweep.
- Include irregular padding/document boundaries, unequal microbatch target
  counts, cache continuation and rejection after incompatible changes, and
  correct CE/pair/triple reductions and detachments.
  Preserve the current one-document-per-training-row contract and rejection of
  packed multi-document rows. Test online document/cache resets separately;
  adding packed training is outside this milestone.
- Check save/reload and the next complete update for representative ordinary
  and all-three cases. Use changing inputs and weights, not only one backward.
- Add a short two-selected-layer RT case, for example indices 0 and 15.
  All-16-layer RT is a separately named stress configuration; it is not cleared
  by a layer-0 result. Test it before advertising it as ready for use.
- Extend representative cases to T128 and T512 if the initial checks pass;
  perform the common T512 resource screen in F4.

Record mode, exact objective, precision, layer selection, batch, sequence length,
parameter ownership, observed backend and explicit limitations for every case.
Brief timings identify likely bottlenecks; they are not optimized benchmarks.
No multi-million-token learning comparison is needed.

**Done when:** every intended feature combination completes the bounded actual
runtime checks, or a specific failing combination is isolated and repaired;
the ledger distinguishes passed, untested and unsupported configurations.

## 5. F2 — Numerical health, Q/K decision and auxiliary-loss integration

Instrument only bounded diagnostics; avoid storing every attention matrix.
Use synthetic inputs for reproducibility plus a few existing real-text batches.
Compare native ordinary, RT, FBT and all-three cases under matching semantics.

Observe Q/K RMS and tails, scaled-logit ranges, attention entropy/concentration,
hidden/fusion-state scale over positions and passes, pre/post-clip gradient norms,
update-to-weight ratios, and CE versus auxiliary gradient contributions.
Sharp attention, clipping or a worse short loss is not by itself a bug.
Look for nonfinite values, extreme new scale growth, reproducible incorrect
derivatives, or an auxiliary objective overwhelming the intended learning signal.
Diagnostic instrumentation must not silently change the measured backend.

Use the retained FP32/sequential references on small fixtures. Compare losses,
aggregate and per-tensor gradient relative L2, direction and scale-aware maximum
error, then a few complete BF16 updates. Do not treat relative error at nearly
zero coordinates as decisive. Only expand precision testing if these checks
leave a concrete concern unresolved.

### Q/K normalization decision

Start with native OLMo's absence of Q/K normalization. That is a valid pretrained
architecture; feedback and RT introduce new input distributions whose scale
still deserves measurement. If health is acceptable, retain native math and
record the measured scope. We cannot prove the absence of future scale problems.

If a meaningful scale/saturation problem remains after checking masks and losses,
prototype Q/K normalization in a **separate opt-in branch**. A possible transition:

\[
\widetilde q=(1-\kappa)q+\kappa s_q N(q),\qquad
\widetilde k=(1-\kappa)k+\kappa s_k N(k),
\]

where N is per-head RMS normalization, kappa starts at zero, and initial gains
s_q/s_k are calibrated on bounded native activations. Kappa 0 must execute the
native branch exactly, including its gradients. Calibration can approximately
preserve scale; it does not make the normalized endpoint functionally identical.

Apply the chosen rule consistently before RoPE to queries, temporary keys,
permanent RT keys, cache writes and backward reconstruction. Do not reuse caches
after a normalization/gain/weight change. Validate a tiny independent reference,
then a short gradual training transition, holding other schedules fixed.
Specify whether gains are fixed or learned and include new parameters in the
ledger. Start ordinary, then RT/FBT integration. This is a conditional adaptation
milestone, not an automatic alteration of every baseline.

### NextLat startup

O4 already showed a substantial deficit while recurrence was still off; a
randomly initialized predictor with unit auxiliary weights can disturb training.
First verify indexing, detachments, normalization and gradient routes.
If implementation is correct but auxiliary gradients dominate, test a bounded
predictor-only warm start or individual auxiliary-weight ramp. Record the changed
trainable set, exposure and schedule. Do not require a language-quality victory
to establish that the combined objective works as intended.

**Done when:** no unresolved material numerical/gradient concern remains in the
tested configurations, and the Q/K/auxiliary startup decisions and limits are
documented. A new normalization branch, if needed, gets its own checked scope.

## 6. F3 — Attention backend and native RT efficiency

**Completed slice (2026-09-22):** fixed-layout CUDA graphs now cover canonical
RT/FBT/NextLat training with ordinary activation checkpointing. The bounded
correctness and B32/64/128 results are in the F3 report. F3b subsequently validates
weight-cast reuse and a fused historical forward tile, with matched capacity
checks and before/after profiles. Ordinary Flash dispatch is verified for these
unpadded graph layouts. F3c now validates historical dK/dV fusion and its full-update integration.
F3d removes quadratic backward probability/error buffers in an opt-in checked path.
Broader layer/context/backend coverage still keeps the full gate open.

### Current distinction

Ordinary OLMo attention calls PyTorch SDPA. A prior bounded fixture observed
cuDNN fused attention, but the configuration label “sdpa” does not guarantee
FlashAttention for every shape and mask. The public eager FBT path supplies an
explicit validity/causal mask; F3's prepared path lowers proven all-valid
single-document rows to an equivalent implicit causal mask. Padded and online
dispatch still need their own evidence, so a mask-free ordinary trace is
insufficient for them.
PyTorch selects implementations subject to input constraints.
[SDPA documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html),
[backend selector](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html).

Selected native OLMo RT blocks retain the PyTorch dyadic schedule and custom VJP.
F3b adds an optional Triton historical tile for BF16 Q/K/V, head dimensions
16/32/64/128 and rectangles up to256 on either side; unsupported shapes/precision
use the recorded eager fallback. At the tested T512 all511 historical rectangles
are fused. This is not an FA4 call. Forward retains inputs and completed outputs
for recomputation. The retained materialized backward reconstructs quadratic probability/error
intermediates; F3d now offers bounded attention workspace via row statistics
and tile recomputation. Retained forward activations alone do not establish
linear complete-model memory.

The relevant vendor RT implementation also uses compiled PyTorch tile helpers;
its ordinary Flash hooks and installed FA4 dependency are not evidence of a
ready-made native RoPE Flash RT path. In particular, do not import its old
internal parameter-gradient writes or unsupported RoPE assumptions.
Our native custom backward returns parameter gradients normally.

### Staged implementation

1. **Trace ordinary attention dispatch** for plain causal, padded FBT, cached
   online and representative training shapes. Add explicit backend selection
   for diagnostics, and make unsupported forced-backend cases clear. Use a
   mask-free causal fast path or supported variable-length layout only where
   mathematically equivalent; preserve padding, document boundaries and RoPE.
2. **Prototype fused historical tile updates for RT.** Preserve a mergeable
   softmax state: maxima/denominators/numerators, or normalized output plus
   log-normalizer. Keep the temporary self contribution distinct from permanent
   historical KV. Ordinary normalized SDPA output alone is insufficient to
   merge disjoint history tiles correctly.
3. **Treat backward as separate work.** Recurrent adjoints arrive in reverse
   dependency order; ordinary Flash backward is not a drop-in replacement.
   Compute correct tile-local historical and temporary-self contributions,
   return all parameter gradients explicitly, and address the quadratic
   probability materialization deliberately.
4. Test tile boundaries, odd lengths, masks/positions, attached cache prefixes,
   frozen-input parameter gradients and shared FBT calls against existing
   independent references. Add bounded mixed-precision complete updates.
5. Measure complete-step benefit before expanding the kernel work. Small tiles
   may favor compiled math. Keep an explicit correct fallback with a recorded
   reason; do not describe that fallback as Flash.

Choose between the available Flash/CuTE interface, a small custom kernel and
compiled PyTorch based on API feasibility and measured bottlenecks. FA4's official
implementation targets Hopper/Blackwell, but its availability does not prove that
its public interface supplies this RT forward/backward contract.
[Official FlashAttention repository](https://github.com/Dao-AILab/flash-attention).
The first fused-tile prototype is a review point before a substantial rewrite.
F3b completed that forward review point, and F3c subsequently validated dyadic
historical dK/dV fusion with FP32 adjoints and BF16 product boundaries. F3d now
replaces full probability/error storage with row normalizers and recomputed tiles
within the tested opt-in scope.
The optimized profile also motivates inspecting local writer/finish VJPs and
the full permanent QKV projection whose Q is discarded. Stage these separately
so numerical ownership and memory improvements remain attributable.

Preserve RT's input/output reconstruction advantage. Consider selective
activation checkpointing for ordinary blocks and pass storage separately;
blindly checkpointing a whole RT block/pass may replay expensive recurrence.
Vocabulary loss projections already have chunked checkpointing.
Consider compile/CUDA graphs only if a profile shows worthwhile launch overhead,
after semantics stabilize; check changed-input and changed-weight replay.

**Done when:** actual fused-backend coverage is explicit, the supported optimized
RT path agrees with its oracle in the bounded cases, and complete-step timing
and memory justify the implementation. If a fused RT path is impractical within
the bounded prototype, report that unresolved goal and fallback costs rather
than implying that existing tiling already meets it. No requirement that RT
match ordinary-transformer throughput despite its sequential dependencies.

## 7. F4 — Single-GPU parameter, throughput, memory and FLOP cards

**F3b accounting slice complete:** [resource derivation](olmo-resource-accounting.md)
and the F3b report provide all eight analytic cards, with20 tests including actual
small matmul traces. The estimated RT/combined B64/T512 update costs for the
recorded loss fixture are294.9–316.5 and644.5–689.3 TFLOPs. These exclude pointwise,
optimizer and launch costs. Do not infer eight measured runtime cards from eight
analytic cards; finish runtime coverage after relevant RT optimizations.

Count each shared/tied parameter once. Current unique parameter totals are:

| Configuration | Unique parameters |
| --- | ---: |
| Ordinary or RT | 1,176,764,416 |
| Ordinary/RT + FBT | 1,185,153,024 |
| Ordinary/RT + NextLat | 1,259,491,328 |
| Ordinary/RT + FBT + NextLat | 1,267,879,936 |

Fusion adds 8,388,608; the configured shared NextLat predictor adds 82,726,912.
RT and additional K passes add zero parameters. Report registered/resident,
executed, trainable, optimizer-owned and deployable-inference totals separately:
an inactive module can remain resident, and NextLat is training-only.
Frozen-backbone fusion adaptation has only 8.39M trainable parameters.

Give all eight configurations a brief complete-update T512 screen, with warmup
and repeated changed-input/weight updates. Compare one common feasible physical
batch and logical target budget first, then try a few larger microbatches per
configuration for a comfortable operating point. Keep explicit memory headroom;
do not maximize the last GiB. Add selected T1024/T2048 stress cases within native
context and representative K3/multi-layer-RT cases; label untested cells.

Report:

- Valid input tokens/s and CE targets/s, global versus per-GPU batch, accumulation
  and update time. Separately count latent pairs, KL triples and pass work.
  Reprocessing a token in K passes does not mean K times the data exposure.
- Peak allocated and reserved memory after optimizer state is initialized,
  with persistent state, temporary activations and cache/recompute behavior.
- Finite-pass training separately from finite prefill and exact online/decode.
  Online inference omits the NextLat predictor and has a different dependency path.
- Steady compute separately from instrumentation, evaluation and checkpoint I/O.
  O5e spent about 191 seconds in optimizer steps but 23.8 minutes end to end,
  largely because of full-checkpoint retention; this matters for planning pilots.

Estimate FLOPs/update with multiply-add counted as two operations. Include
projection/MLP and attention forward/backward, permanent RT KV projections,
recomputation, K passes, fusion, predictor, full-vocabulary CE and KL readouts;
state whether optimizer arithmetic is included. Explain mask/padding and actual
versus ideal useful work. Parameter count alone or a generic 6NT estimate misses
material work here.

Use a short operator trace to audit the analytic estimate, not as a complete
FLOP counter: PyTorch's automatic estimate covers selected operators and can
miss fused/custom work. Report an estimate or range with coverage assumptions.
[Profiler documentation](https://docs.pytorch.org/docs/2.14/profiler.html).
Verify APIs against the installed runtime before using them.

**Done when:** a compact resource table lets us budget meaningful future token
exposures and understand the parameter/compute cost of each mechanism.

## 8. F5 — Genuine two-GPU execution

The read-only inventory on 2026-09-22 exposes **one H100 80GB**. An actual second
GPU is a hardware dependency. CPU collectives or two processes sharing one GPU
do not establish two-GPU readiness.

Start with DDP for correctness and throughput; it replicates the model and
optimizer and does not pool GPU memory. Use the current validated backend first,
then repeat representative checks after relevant kernel changes.

The existing single-process optimizer helper calls a loss method directly.
A DDP forward adapter is needed; simply wrapping the object is insufficient.
All-reduce separate global CE/pair/triple counts. With default gradient averaging,
each rank scales its local loss sum by world_size/global_valid_count. Clip after
gradient reduction. Match collective order and coordinate nonfinite failures
before any rank steps.
[PyTorch DDP documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

Test:

- All eight combinations at small two-GPU sizes, then representative actual-model
  ordinary, RT and all-three complete updates.
- One-GPU accumulation versus two-GPU execution on identical global data, with
  stronger full-update equivalence for ordinary and all-three. Include unequal
  valid counts and a rank with zero targets for one objective, without skipping
  required collectives.
- Shared backbone calls, tied embedding ownership, inactive fusion/predictor
  parameters and coefficient-zero schedules. Do not assume static gradient
  participation where schedules can change it.
- A few optimizer steps and same-world-size save/resume, including per-rank
  RNG/data cursor. Checkpoint/artifact and W&B ownership belongs to rank 0;
  reported metrics must aggregate correctly.

Then measure equal-global-batch speedup and maximum-comfortable aggregate
throughput separately, with GPU model/topology/interconnect and per-rank memory.
If memory limits useful batches, consider optimizer-state sharding first.
FSDP/parameter sharding is a separate compatibility task: custom replay accesses
layer weights, FBT reuses them, and tied readout/cache ownership must survive
materialization. Do not promise that DDP alone reduces per-GPU model-state memory.

**Done when:** real two-GPU updates, recovery and resource measurements pass in
the declared scope. Changed-world-size recovery and broader cluster scaling
remain separate unless needed.

## 9. F6 — Readiness review, then choose learning experiments

Deliver one handoff linking:

1. The eight-mode capability matrix, precise semantics and supported layer/shape
   scope, with failures fixed and open limitations explicit.
2. Bounded numerical/gradient evidence, Q/K decision and any startup schedule.
3. Actual attention backend traces, tiled RT correctness and memory/efficiency
   evidence, with optimized and fallback paths distinguished.
4. Parameter and single/two-GPU resource cards, including full-step FLOP estimates
   and recoverable checkpoint procedures.

This is the review point for deciding whether to continue an ordinary baseline
for substantial tokens, run short mechanism comparisons, or first fix a capacity
or throughput bottleneck. Do not automatically launch the old joint mixed-data
FBT learning comparison or an eight-arm quality sweep.

Later comparison design should track initialization, data/targets, objective
weights, trainable capacity and compute. Multi-seed studies, broader retention,
math/code generation and Feature Matching remain later research. Semantic Tube,
autonomous NextLat predictor recurrence and a full paper-scale reproduction
remain deferred.

## 10. Execution order and durable operation

- Completed: F1 integration, F2 health/checkpointing, F3 canonical CUDA graphs,
  F3b forward historical-tile fusion/cast reuse, F3c historical backward fusion
  and F3d bounded backward attention workspace,
  each with before/after profiles and bounded native checks.
  F4 analytic parameter/FLOP cards are complete; runtime coverage remains partial.
- Next: review F3d, then broaden optimized RT layer/context integration and
  finish feature/resource cards. Keep native Q/K math and materialized reference.
- F5 starts when a second GPU is available, independent of quality results
  or completion of the fused-kernel work. Then perform F6 readiness review.
- Kernel engineering is the largest uncertain effort. Initial checks and profiles
  should be bounded; no automatic long learning run is needed. Give advance notice
  of any multi-hour run and retain occasional recoverable checkpoints.
- GPU execution must use the project container with successful in-container
  nvidia-smi; never silently fall back to CPU. Do not disturb completed run records.
- Log graphable diagnostics to W&B under taylorbollman. Retain useful weights,
  data and evidence at gs://fast-chunks; record hashes/generations. Keep full
  recovery checkpoints at a sensible cadence and report upload cost separately.
  Disposable few-step checks need not create repeated multi-gigabyte archives.

Relevant implementation starting points:
[native stack](../cdrm/pretrained/olmo.py),
[tiled RT](../cdrm/pretrained/olmo_tiled.py),
[sequential oracle](../cdrm/pretrained/olmo_recurrent_oracle.py),
[FBT/fusion](../cdrm/pretrained/olmo_fbt.py),
[training objectives](../cdrm/pretrained/fbt_training.py), and
[historical numerical methods](rt-numerical-handoff.md).
Retain these independent references when optimizing the candidate path.
