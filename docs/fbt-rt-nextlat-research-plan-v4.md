# OLMo / RT / FBT / NextLat: functionality and execution plan

Updated 2026-09-24. **Current authoritative forward plan.**

This supersedes the next-experiment queue in [v3](fbt-rt-nextlat-research-plan-v3.md).
The priority is functionality, numerical health, integration and reasonable
execution cost before quality comparisons. Completed O1–O5e evidence remains
valid within its recorded scope.

**Next milestone authorized,2026-09-24:** prioritize native RT physical-batch
scaling after bounded integration of the accepted ordinary fusions. See the
[large-batch plan](native-rt-large-batch-plan.md) for B64/128 upward toward512 at
T512, RT-only and combinedK2+NextLat, and separate setup/capture/steady memory
accounting. B64 is not an established throughput optimum. The paper's utilization
argument concerns per-device batch, not gradient accumulation. This milestone
precedes RT leaf compilation and the older recovery/online queue. User authorized
execution plus a conditional ordinary-FA4 memory check near capacity. Preserve the existing precision qualifications and native math.

**RT decision,2026-09-24:** user adopts optimized native RT and moves on from
the backend-selection comparison. Keep author-derived as an experimental
reference; preserve its isolated B128 throughput/memory advantages and unresolved
full-model BF16 qualification. Full-model B64/T512 rates were effectively tied,
and the latest ordinary fusions were not part of those historical comparisons.
No new backend-adjudication run is queued. Use the already validated rounded
compiled ordinary SwiGLU; prioritize bounded compilation of the RT finish/writer
region if further RT profiling supports it. Evaluate Dao/Liger/xFormers/custom
activation alternatives only against that measured path when useful.

**User accepted,2026-09-24:** select the PR27 Dao-RoPE/fused-Adam ordinary T512
configuration for future runs, with rounded compiled SwiGLU, Flash SDPA and
all-layer checkpointing. Historical controls/default reproduction remain
available; record execution options in new run/checkpoint configurations.
[Clarification and candidate follow-ups](rt-backend-numerical-clarification.md)
records Dao standalone SwiGLU, measured absence of FA4 memory gains, and why
the31%/16% RT backend gradient mismatch is not an error measured against FP32.
No new GPU run is launched by this explanatory update.

**Latest milestone complete,2026-09-24:** native-FP32 Dao RoPE and fused AdamW
are implemented as independent opt-ins; read
[ordinary fusions results](reports/olmo-ordinary-fusions/results.md) and handoff.
Repeated B64/T512 full-CE39.18k→43.61k (+11.30%); one B16/T2048 pair36.99k→41.03k
(+10.92%). T512 numerical screen passes; T2048 retains a tiny absolute CE-only
relative-loss miss. All own graph/full-Adam checks pass; fixed-gradient optimizer
comparison passes.12reports,96updates,410distinct scoped CPU plus75evidence tests.
Defaults and prior RT qualifications remain. GPU idle, no additional queue.
Section8 now explicitly records DDP→ZeRO1/2 and bounded contiguous RT leaf
compilation as future candidates. Other RT/FBT/NextLat combinations and online
execution need their own checks before these ordinary options are called ready.

**Previous milestone complete,2026-09-24:** ordinary-model efficiency precedes
the older functional queue. Read [results](reports/olmo-ordinary-efficiency/results.md)
and the current handoff. Opt-in rounded SwiGLU gives repeated B64/T512
36.75k→39.16k inputtokens/s (+6.55%) at46.17GiB reserved; compiled+alternating
checkpointing atB32 gives34.43k→40.37k (+17.25%) at54.75GiB. Both pass the
unchanged actual-checkpoint numerical screen and exact own graph/Adam checks.
FA4's small loss-only misses remain qualified, despite healthy outputs/gradients
and exact own operational checks; its directional full-step gain is0.65% atT512
and3.55% atT2048. Two graph-capture OOMs bound reduced-checkpoint capacity.
No defaults, Q/K normalization, RT backend or quality-training state changed.
This is the prior PR26 snapshot. The subsequently authorized ordinary-fusion
follow-up is recorded above; its results supersede the recommendation to wait
before implementing native-FP32 RoPE fusion. The RT-backend decision and older
functional queue remain separate.

**Latest approved ordering (2026-09-23):** the user approved the
[native RT efficiency and author-comparison plan](olmo-rt-efficiency-and-author-comparison-plan.md).
It places RoPE-table reuse and K/V-only permanent writes before a matched
author-derived native-RoPE backend comparison, with an initial review after the
native improvements. Stage A is complete: [results](reports/olmo-rt-efficiency/results.md).
RoPE reuse and K/V-only writes pass420 CPU tests and41 GPU gates, and improve
matched native RT/combined throughput by4.36%/2.26% with unchanged parameters.
The user subsequently lifted the review stop and authorized proceeding directly
to Stages B/C after finishing Stage A. Graph recovery/accumulation and online
readiness remain subsequent work.

**CE integration and original 16-layer baseline complete (2026-09-23).** Read
[results](reports/olmo-ce-integration/results.md) and
[usage](reports/olmo-ce-integration/usage.md). Optional independent CE2048/KL128
preserves historical defaults and loss semantics. Six GPU reports pass 18 gates,
36 physical updates, and 152 scoped CPU tests. Native B64/T512 matched half-CE
throughput improves 31.11k to 39.19k input tokens/s (+26%); full CE reaches 36.63k,
with unchanged 26.74 GiB allocated / 37.17 GiB setup reserved peaks. Native
ordinary/NextLat/combined chunk-comparison gradients pass; same-candidate graph
and three-update Adam comparisons are exact. Dao CE was source/import-audited,
not adopted or GPU-tested. No default, Q/K, RoPE or RT math change.

After the RT backend comparison: resume graph recovery/save-resume, accumulation, padding and online
readiness. Use recorded CE2048/KL128 in new bounded development and matched CE
settings in future throughput comparisons. Existing RT+FBT precision caveats
remain open; require a second GPU for distributed checks. No long quality run
or additional GPU experiment is queued.

**User-requested ordinary throughput investigation complete (2026-09-23).**
Read its [results](reports/olmo-ordinary-throughput/results.md). Seven bounded
six-layer runs identify avoidable CE chunk overhead:128→2048 positions improves
B64 halfCE57.4k→92.8k inputtokens/s and physicalB51243.9k→93.1k. FullCE B512
reaches73.5k; B32 without ordinary checkpointing88.9k. The paper's ordinary
recipe uses micro32/global512, GELU, smaller vocabulary, compilation and no
ordinary checkpointing. The quoted153k is not reproduced, and F4's31k is not
an optimized ordinary baseline. No production defaults or math changed. Its
larger-CE-chunk follow-up is now complete as recorded above. Retain existing
numerical qualifications.

**F4 training resource matrix is complete with a retained numerical qualification.**
Read the [F4 assessment](reports/olmo1b-f4/assessment.md),
[results](reports/olmo1b-f4/results.md) and
[roundoff diagnosis](reports/olmo1b-f4/roundoff-assessment.md).
All eight modes have finite B64/B96 full-update measurements; B64 remains
our common default. RT+FBT narrowly failed the unchanged BF16 coordinate
screen; graph/Adam checks are exact, and full FP32 memory implementations agree
closely. Both BF16 control and candidate show about18% gradientL2 difference
from full FP32 at this initialization. This remains a qualification, not a
blanket precision clearance. No core math or Q/K change was made. Read the
handoff for235 scoped tests, retained evidence and resource operating points.
Next: recovery/accumulation, padding and online resource readiness, plus a
bounded transition/clipped-update precision follow-up before substantial
learning if warranted. Genuine multi-GPU requires a second GPU.

**Status: F1/F2 and F3 through F3e are complete within their measured scopes.**
Read the [F3e assessment](reports/olmo1b-f3e/assessment.md),
[results](reports/olmo1b-f3e/results.md) and [usage](olmo1b-f3e-usage.md).
Sixteen F3e GPU reports pass48 gates;474 scoped CPU tests pass. Actual native
checks now cover multiple selected RT layers, shared K3 feedback and T2048.
The user expects more than one RT layer but has not prescribed all layers or
placement. Primary layouts are `(0,1)`, `(0,15)` and `(0,5,10,15)`; all16 remains
separate stress, with B1/T32 full equivalence and B8/T512 finite capacity updates.

Recompute-versus-materialized BF16 gradients pass the unchanged budgets
(maximum global relative L2 .007477). All initial forward losses, same-candidate
graph checks and complete Adam/state comparisons are exact. This extends
bounded functionality coverage, not sustained learning stability or a new
full-native BF16-versus-FP32 comparison. Native Q/K math remains unchanged.

With FBT K2+NextLat at common B64/T512, single/two/four RT layers reach
10.93k/9.89k/8.31k input tokens/s, all around39.09GiB peak allocated and
60–61GiB peak reserved during setup. RT-only spread2 B64 reaches19.45k/s at
32.25GiB allocated/47.96GiB peak reserved; B128 reaches22.72k/s but reserves
76.80GiB during setup, so B64 is the conservative development reference.
Combined spread2 B8/T2048 reaches4.71k/s at31.29GiB allocated. These are
three-update medians with explicit batch/length scope, not maximum-batch searches.

Ordinary layers use deterministic PyTorch Flash/checkpointing; selected RT tiles
use Triton and F3d bounded-workspace backward. Forward rectangles larger than256
retain eager fallback: atT2048 only six two-layer calls account for75.04% of
historical attention pair area, not full-step work/time. Historical recompute
backward remains fused. Native RT does not use FA4/CuTE. Defaults still retain
materialized reference; no architecture, checkpoint, RoPE or loss change.

**Next after F4 training cards:** graph recovery/accumulation, padding and
online readiness, retaining the F4 numerical qualification. Analytic parameter/FLOP cards now explicitly support recompute and
actual loss-mask counts. Combined training/deployable parameters are
1,267,879,936/1,185,153,024; RT selection and pass reuse add no weights. Actual
multi-GPU work needs a second GPU. Local writer/finish VJPs, discarded permanent-Q
and longer forward kernels remain profiling-led performance opportunities.
No GPU or quality run is queued. Read the [handoff](fbt-rt-nextlat-handoff.md)
first after compaction.

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

**F4 training-resource matrix complete (2026-09-23), with qualification:**
[assessment](reports/olmo1b-f4/assessment.md), [results](reports/olmo1b-f4/results.md),
[frozen protocol](reports/olmo1b-f4/protocol.md). All eight independent toggles
were measured at common B64/T512 and B96/T512, RT `(0,15)`, K2, same checkpoint
and masks. RT+FBT retains a failed coordinate screen and the broader BF16
sensitivity is documented. All capacity checks are finite; that does not clear
numerical qualifications. B96 materially helps RT and RT+NextLat; the larger
FBT combinations have tight setup headroom. Ordinary/all-three B64 reach
31.11k/9.89k input tokens/s, with26.74/39.09GiB allocated and37.17/60.46GiB
reserved peaks. Operator traces reconcile the dense ledger. This completes
training cards only; finite prefill/exact-online cards remain in readiness.
F3e long-context/K3 evidence retains its original scope. No quality sweep.

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
User refinement,2026-09-24: explicitly compare DeepSpeed ZeRO stage1 (optimizer
state sharding) and stage2 (optimizer and gradient sharding) after the DDP
correctness baseline. Keep parameters replicated initially. Measure both
equal-global-batch scaling and the larger physical microbatch made possible by
memory savings; RT throughput can benefit from the latter, but sharding itself
does not remove its sequential dependency. Start without CPU/NVMe offload.
[DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/).

Treat integration as an execution change: preserve global per-objective loss
normalization, tied parameter ownership, FP32 master/moment policy, clipping
after reduction, and same-world-size recovery. Verify the custom RT backward
and shared FBT parameters work with gradient hooks. The current captured plan
owns persistent gradient buffers, so do not assume its storage contract survives
ZeRO gradient partitioning. Establish eager distributed updates first, then
validate capture/replay and collective ordering with the installed versions.
Compare resident and setup-peak memory as well as communication/full-step time.
Do not replace the optimizer, graph boundary and sharding policy simultaneously.

If memory limits useful batches, consider optimizer-state sharding first.
FSDP/parameter sharding is a separate compatibility task: custom replay accesses
layer weights, FBT reuses them, and tied readout/cache ownership must survive
materialization. Do not promise that DDP alone reduces per-GPU model-state memory.

**Done when:** real two-GPU updates, recovery and resource measurements pass in
the declared scope. Changed-world-size recovery and broader cluster scaling
remain separate unless needed.

### Sequential RT fusion follow-up

User refinement,2026-09-24: profile the recurrent leaf path, including writer and
finish projections, normalization, SwiGLU and residual operations, at realistic
large physical batches. Ordinary compiled SwiGLU savings do not establish the
size of an RT improvement. Compare a compiled contiguous leaf/writer/finish
region against activation-only fusion; preserve the tested precision boundaries
and weight-cast reuse. Keep historical attention in its validated tile backend
and retain the dependency between successive recurrent positions. Compilation
can reduce launches/intermediate traffic but cannot parallelize away that
dependency. CUDA graphs already reduce host launch cost, so measure the added
device-time benefit rather than assuming speedup. Bound compile time/code size;
do not unroll an entire long recurrence merely to fuse its pointwise operations.
Require shared-cotangent gradients, own graph/full-update checks and a full-model
throughput measurement before adopting a candidate. This is a future bounded
RT optimization milestone, not a change to the current ordinary-only queue.

Concrete native candidate: `olmo_tiled._finish`, then a pure one-token
finish/interpolation/writer/RoPE helper. Keep cache mutation and `_add_tile`
outside. Later examine the two reverse-loop local input VJPs without regressing
the already batched parameter VJP to per-token weight-gradient computation.
The author port already compiles several batched helpers; its sequential
finish/writer calls remain eager, so this may benefit either backend. Preserve
FP32 norm/residuals, native packed `[value,gate]`, temporary-self behavior,
alpha and K/V-only semantics. A SwiGLU library substitution must respect both
packing and the established BF16 rounding boundaries.

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
  F3d bounded backward attention workspace and F3e multi-layer/context
  integration, each with bounded native checks and resource evidence.
  F4 analytic parameter/FLOP cards are complete; runtime coverage remains partial.
- Next: complete F4 feature/runtime cards with multiple selected RT layers,
  then remaining graph/online readiness. Keep native Q/K math, the materialized
  reference and the separate scope of all16 stress evidence.
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
