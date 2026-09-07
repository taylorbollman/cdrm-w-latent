# Implement the naïve FP32 CDRM reference

Implement Cross-Depth Read-Conditioned Recurrent Memory (CDRM) in the existing OLMo-based project. The immediate deliverable is a trustworthy, runnable ordinary-autograd FP32 implementation, bounded numerical/operational evidence, and an executed preliminary pilot on selected D.2/MAD synthetic tasks, followed by configurations for broader joint SEQ/R3/CDRM experiments.

The user has now authorized the coding agent to include the bounded synthetic pilot below in this PR and execute it after the applicable correctness checks pass, without waiting for another confirmation. This updates the earlier implementation-only stopping point.

This is the current research priority. A standalone R3 learning advantage, BF16 clearance, custom CDRM tiling, and completion of the entire control/topology grid are not prerequisites. This staging supersedes earlier instructions that would hold core CDRM implementation behind those milestones. Preserve existing supported paths and use the latest project instructions and saved configurations for execution.

## Context and scope

Stage B compared SEQ and R3 on small symbolic tasks. Neither learned reliable retrieval/state computation under the initial budget. It did not test CDRM. The subsequent FP32 backward investigation cleared the tested R3/rho1/no-accumulation MQAR regime. Keep those results and artifacts intact.

BF16 mixed precision is implemented experimentally but remains incompletely cleared and slower at the tested B64/T128 shape. Retain it as opt-in; do not repair or optimize it as part of this milestone. Use FP32 parameters/computation, autocast and TF32 off, and the established deterministic attention policy for the new reference. Verify that recent precision-policy changes do not inadvertently activate BF16. Existing compiled ordinary helpers may remain where already validated; the new side scan must use ordinary autograd, without a new custom backward.

The first substantive shape remains 12 blocks, D256, H4/full MHA, MLP1024/GELU, pre-norm ALiBi with the existing learned normalization/QK-normalization settings, and task-native vocabulary. Start with tiny numerical fixtures, then T128 and measured microbatch sizes on one H100 80 GB. The saved configuration is authoritative for detailed normalization, projection, bias, clipping, initialization and loss settings. Do not infer them from the paper's debug script or change them silently.

Use zero-based indices: early site 3, late source/bridge site 8, suffix blocks 9–11. Scope excludes cached decoding, document packing, GQA/RoPE expansion, distributed execution, gradient accumulation, late recurrence, latent objectives, feedback-to-embedding, multi-depth writers, and custom tiled CDRM.

## Exact architecture and equations

Run ordinary causal blocks 0–8 across the full sequence. Save:

- `p3`: output **after ordinary block 3**, shape `[B,T,D]`.
- `p8`: output **after ordinary block 8**, same shape.

Then execute one side memory scan, bridge its result into `p8`, and execute ordinary blocks 9–11. CDRM does not replace preview block 3 with R3, replay blocks 4–8, or retroactively change the preview. Adding both R3 replacement and this side scan would be another architecture.

Define `Pre3` as block-3 Q/temporary-KV preprocessing; `Read3` as scaled attention including head merging and `attn_out`; `Post3` as the existing residual/MLP processing after that projected attention; and `KV3` as the validated persistent-record normalization/projection/key-normalization path. Apply each operation exactly once.

Construct the deep-conditioned candidate:

\[
c_t=p3_t+\epsilon A_d\!\left(N_d(p8_t-p3_t)\right).
\]

Construct queries and temporary records from **p3, not c or p8**:

\[
(q_t,k_t^{\mathrm{tmp}},v_t^{\mathrm{tmp}})=\operatorname{Pre3}(p3_t),
\]
\[
a_t=\operatorname{Read3}\!\left(q_t,
\{M_i:i<t\}\cup\{(k_t^{\mathrm{tmp}},v_t^{\mathrm{tmp}})\}\right).
\]

Combine current deep information and retrieved history, then write:

\[
\hat m_t=\operatorname{Post3}(c_t,a_t),\qquad
m_t=(1-\rho)p3_t+\rho\hat m_t,\qquad
M_t=\operatorname{KV3}(m_t).
\]

Bridge into the late preview:

\[
v8_t=p8_t+\lambda A_b\!\left(N_b(\hat m_t-p3_t)\right).
\]

Feed `v8` to block 9. The bridge uses **proposed state hat_m**, not interpolated memory state m. Its subtraction anchor is p3, not p8 or c. `m` and `hat_m` are residual-space vectors; `M` is a K/V pair. Never add raw K/V tensors to p8.

| Gate | Meaning |
|---|---|
| epsilon | Explicit deep-source contribution to the candidate |
| rho | Proposed-output contribution to persistent storage |
| lambda | Correction strength at the post-block-8 bridge |

Use rho=1 for the initial CDRM training profile. At rho=0, memory contains projected p3 while the reader/writer/bridge still operate: this is **not** a SEQ switch. Lambda=0 gives SEQ logits with an otherwise ordinary backbone and controlled deterministic execution.

The scientifically distinctive dependency is that a memory record depends on both the deep current candidate and a read of earlier records. Addressable deep records alone are a separate control hypothesis.

## Reference execution and causality

Queries, temporary pairs and candidates can be computed for all tokens before the scan. The writer and permanent records remain sequential. The following is dependency pseudocode, not a requirement for new API names:

```python
p3, p8 = ordinary_preview_through_8(tokens)
q, k_tmp, v_tmp = pre3(p3)           # [B,H,T,head_dim]
c = p3 + epsilon * deep_adapter(deep_norm(p8 - p3))
history_k, history_v, proposed, records = [], [], [], []

for t in range(T):
    k = torch.cat(history_k + [k_tmp[:, :, t:t+1]], dim=2)
    v = torch.cat(history_v + [v_tmp[:, :, t:t+1]], dim=2)
    a = read3(q[:, :, t:t+1], k, v, position=t)
    hat_m = post3(c[:, t:t+1], a)
    m = (1 - rho) * p3[:, t:t+1] + rho * hat_m
    kt, vt = persistent_kv3(m)      # compute AFTER current read
    history_k.append(kt)
    history_v.append(vt)
    proposed.append(hat_m)
    records.append(m)

hat_m = torch.cat(proposed, dim=1)
v8 = p8 + lambda_ * bridge_adapter(bridge_norm(hat_m - p3))
logits = ordinary_suffix_from_9(v8)
```

Start with unpadded independent examples. Reset side memory on every independent forward/example; never carry it between minibatches or gradient-accumulation steps. Only earlier permanent records and the current temporary pair enter a read. Future temporary pairs are forbidden. The current permanent write is unavailable until the current output has been computed.

Use absolute sequence positions when slicing ALiBi: row t and columns 0 through t for the scan's packed prefix. Reuse the established slope/bias builder and apply attention scaling once. If using SDPA with a one-row query and a physically restricted prefix, do not blindly set `is_causal=True`: PyTorch specifies upper-left causal alignment for non-square attention, which does not represent this query at absolute position t. Explicit prefix construction with the correct bias and `is_causal=False` is appropriate when no additional padding mask is needed. See the [PyTorch SDPA mask semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). The ordinary preview and suffix still require their usual combined causal/ALiBi masking. Audit direct block calls so they do not bypass that contract.

Keep every training memory record connected to autograd. Do not detach history, write into detached inference buffers, or truncate temporal credit to make the pilot fit. List accumulation and concatenation are acceptable in this oracle. Concatenated prefixes and saved tensors can have quadratic training-memory cost even though final K/V storage is linear; measure it. Reduce microbatch or sequence length for diagnosis rather than silently changing the computation.

Do not force the terminal write to receive a gradient when it has no consumer. An unused intermediate can legitimately have None/zero gradient. Diagnose parameter gradients and active temporal paths separately.

## Parameter ownership, adapters and configuration

Initially share block-3 parameters between its ordinary preview computation and the side reader/writer. Keep one canonical parameter owner and preserve backbone checkpoint names where possible. A functional interface receiving the owning block is a reasonable implementation. Do not create a copied trainable recurrent block for the side scan and call that weight sharing.

The ordinary block may retain fused QKV. Use differentiable slices of its canonical fused weights/biases for the side projections, respecting actual projection ordering and dimensions. Do not wrap slices in new `nn.Parameter` objects or duplicate optimizer entries sharing storage. Gradients from preview, reader, writer and persistent projections must accumulate into the same intended weights. Check registration, optimizer ownership and serialization explicitly; default deduplication by `named_parameters()` alone is insufficient evidence about optimizer groups or checkpoint aliases.

Use small ordinary PyTorch adapters and explicit normalizers. A simple provisional choice is separate bias-free D-to-D linear adapters and stateless FP32 RMS normalization with a recorded epsilon. These are extra adapter normalizers, not a replacement for the backbone's established normalization type. Preserve a prior explicitly frozen adapter specification if one exists in the project and document it.

Initialize both adapters nonzero. For initial smoke testing, rho=1 with small fixed nonzero epsilon and lambda is sufficient; epsilon=0.1 and lambda=0.01 are provisional starting values, not scientific optima. Inspect candidate-correction/p3 and bridge-correction/p8 RMS ratios at initialization. If these are grossly disproportionate, make a documented initialization-scale adjustment before comparative training. Do not tune gates independently against held-out research outcomes. Fixed gates/config buffers simplify the first patch; learned gates are optional later. Save the values in checkpoints.

Lambda=0 is an equivalence-test setting, not the initial writer-learning setting. Do not initialize both a bridge matrix and its gate to zero. With the bridge disabled and no auxiliary objective, the side branch receives no backbone-loss gradient through that bridge.

Add only the configuration needed for topology, early/late indices, reference backend, gates, adapter specification and requested diagnostic states. Validate indices and reject incompatible combinations. CDRM's preview replacements are empty; enabling CDRM must not accidentally retain R3 replacement at index 3. Do not build a general graph/topology framework in this patch.

Preserve existing hidden-state conventions: the state entering block 9 is v8; raw p8 is a separately named preview. Optional outputs may expose p3, p8, hat_m and m for diagnostics and future latent objectives, without inserting extra entries into the ordinary block-indexed hidden-state list. Log detached summaries; retain graph-connected tensors only when required by the computation or an explicit diagnostic.

## Numerical acceptance tests

Reuse existing infrastructure and tolerances where appropriate. These tests establish computation and gradient correctness; they do not establish task learning.

1. **Topology and bypass.** Exactly ordinary blocks 0–8, one side scan, and ordinary blocks 9–11 execute. No replay or hidden replacement. With copied backbone weights and lambda=0, SEQ and CDRM logits, actual masked loss and common-backbone gradients agree. Account for expected zero/unused extra-branch gradients. First compare deterministic paths with matched execution settings.
2. **Independent tiny oracle.** Cross-check the production reference against a small explicit attention/scan formulation or meaningful directional derivative check with independent indexing/projection logic. Sharing the same mistaken helper in both arms is not an oracle. Reuse proven primitives and focus independent checks on the new composition, projection sharing and masks. A rho=0 static-K/V parallel side computation is an additional useful endpoint check, but is not global SEQ equivalence.
3. **Causality and boundaries.** Future-token perturbations cannot affect earlier outputs. No own permanent record is read; one pair is written per position. At T=1, changing rho cannot change the current bridged output, because rho changes only the unused stored record. Evaluate outputs, not an internally shifted loss with no valid target. Use a non-power-of-two short sequence as an indexing check.
4. **Specific deep-memory path.** Supply independent leaf p3 and p8 tensors to the side scan, with nondegenerate gates/weights, and use loss only on a later bridge correction. At rho=1, verify gradients reach an earlier p8 through stored records. At rho=0 that earlier-deep-source route is absent. This avoids confusing the ordinary preview/suffix gradient with the new deep-memory contribution.
5. **Read-conditioned writing.** Hold a current candidate/query/temporary pair fixed and perturb an earlier record. Verify that the current read and current proposed/persistent write respond at rho=1. This distinguishes the intended writer from one whose stored contents ignore retrieved history.
6. **Shared gradients.** Verify that preview and side contributions add correctly into the canonical weights, including fused QKV slices and learned norms. A tiny untied diagnostic clone, initialized identically, can provide a sum-of-gradients reference. Check intended parameters occur exactly once in optimizer groups and survive save/load with their ownership intact.
7. **End-to-end FP32.** On a small microbatch at D256/T128, verify finite logits, real masked CE, gradients, clipping and an optimizer step. Increase microbatch toward B64 only after measuring memory/time. Clearance must state tested shapes and any limits. Recheck SEQ/R3 only where changed helpers or construction could affect their supported paths.

Do not reuse the R3 custom tiled backward for CDRM. Additional deep-source, adapter and shared-preview gradient contributions make that a separate future implementation task.

## Minimal controls and interpretation

Prepare SEQ, existing R3/rho1, and CDRM as the core joint comparison. R3 need not beat SEQ before CDRM can be studied. Support lambda=0 and a current-only side read as inexpensive diagnostics if straightforward; do not make every historical control a prerequisite for the first core deliverable.

Prepare an **active same-depth writer** for the subsequent attribution comparison. Simply substituting p8=p3 in the main candidate formula makes its difference input zero and can disable the adapter. That is a valid deep-source-removal ablation but is not an equally active capacity control. A concrete active variant is:

\[
c_t^{\mathrm{same}}=p3_t+\epsilon A_d(N_d(p3_t)).
\]

Keep the reader, persistent-write rule, bridge and ordinary p8 unchanged. This uses the same adapter inventory with an active shallow input; document the different input statistics and do not claim that every confound is eliminated. Do not replace the bridge's p8 with p3. Label the zero-difference removal ablation separately if retained.

DM (pointwise deep mixing) and DW0 (addressable deep records whose writes do not depend on the read) remain useful later controls. Their implementation can follow the core reference. Gains over R3 alone do not isolate the deep source, because R3 and CDRM also differ in where the computation executes and enters the backbone.

## Operational milestone and joint-run preparation

After numerical checks, run bounded training smoke tests with active gates, real aligned answer CE, and task-native data. Demonstrate fitting on a tiny reused batch separately from a short fresh-batch run. Track loss, active adapter/shared-weight gradients, gradient norms, write norms, and candidate/bridge correction ratios. These are operational checks; do not label memorization or a short loss decline a synthetic research advantage.

Use corresponding standard-backbone initialization for SEQ and CDRM, and the established mapping for R3. A retained trained SEQ checkpoint can support zero-bridge equivalence tests. Do not silently initialize CDRM's ordinary preview from a trained R3 checkpoint: returning block 3 to ordinary behavior changes the function even if weights transfer. Record any deliberate checkpoint adaptation as its own lineage.

Validate checkpoint round trip and bounded midpoint resume for CDRM, including adapters, gates, optimizer state, RNG, configuration and data position. The side memory is rebuilt from each input sequence and is not unrelated minibatch state. An initialization artifact that can be loaded as weights is not automatically a validated resumable training checkpoint; distinguish those roles and respect known runner restrictions.

Benchmark actual forward/backward/optimizer steps for SEQ, R3 and CDRM at matched shapes and batches where they fit. Include preview/scan/bridge/suffix timing where useful, overall tokens/s, peak allocated/reserved memory, dtype/backend, warm-up and source identities. Do not extrapolate CDRM's memory from the roughly 2.1 GiB measured for R3. If different microbatches are necessary, disclose that and keep throughput versus update-count comparisons distinct. Accumulation remains outside the supported training profile until separately cleared.

Execute the bounded D.2-aligned pilot specified below and provide ready-to-run manifests for subsequent joint experiments. A larger training campaign remains outside this milestone. Use common task definitions, examples, evaluation masks and declared budgets. Preserve all old Stage B endpoints. Treat its previously inspected tests as historical exploratory evidence, use development data for new decisions, and reserve fresh confirmatory examples for later claims. If all models remain near shortcuts, improve the task/optimization regime before drawing a strong negative conclusion about CDRM. Do not require the ordinary model to win or solve every condition as a gate on a possible CDRM advantage.

The paper's one-block comparison and the deeper CDRM topology experiment are separate configurations. Optional one-block reference runs may accompany the pilot below; they are not a prerequisite for starting the main CDRM/SEQ pair.

## Authorized extension: bounded D.2-aligned synthetic pilot in this PR

**Implement and run this extension once the necessary CDRM checks pass.** The user will be away for several hours and wants preliminary learning evidence as part of the same work. Do not stop at NUM/OPS plus proposed commands if there is sufficient time to execute the pilot. Conversely, a genuine unresolved computation/gradient problem takes priority over generating research curves.

### Sources and what counts as alignment

D.2 specifies one block, D128, MLP512, 16 heads, ALiBi maximum bias 8, and AdamW with betas (0.9, 0.98), epsilon 1e-8, learning-rate candidates 1e-4/5e-4/1e-3, and weight decay 0 or 0.1. Its printed LR list repeats 5e-4; do not infer a fourth distinct value. Figure 5 reports sequence exact match; Figure 6/E.1 reports token accuracy. [RT paper, D.2 and E.1](https://arxiv.org/html/2604.21215#A4.SS2)

The current MAD configuration supplies defaults of batch 128, 200 epochs, 12,800 training examples, 1,280 evaluation examples, cosine scheduling, minimum LR 1e-6, and LR 5e-4. **These are MAD defaults, not independently verified settings for the authors' exact plotted runs.** D.2 does not explicitly give batch size, training duration, or all task-generator settings. [MAD configuration](https://github.com/athms/mad-lab/blob/main/mad/configs.py)

Use pinned official MAD generators and task configurations when the exact RT synthetic harness is unavailable. Retain their inputs, targets, ignore indices, special-token handling, and stochastic distributions. Reuse our trainer/model interface; a minimal adapter to MAD data is sufficient. Do not port the whole MAD trainer or install unrelated architecture dependencies merely to generate examples. Record the MAD revision and any task-code adaptation. A familiar task name is insufficient: the Stage B custom MQAR/noisy-recall generators are not interchangeable with MAD.

### Task priorities

| Priority | Task and initial setting | Purpose |
|---|---|---|
| 1 | MAD `in-context-recall`: T128, configured vocabulary 16, 12,800 train examples, multi-query enabled | Establish basic associative retrieval in an established task implementation. |
| 2 | MAD `selective-copying`: T256, configured vocabulary 16, 16 tokens to copy, 12,800 train examples | Test ordered retrieval while ignoring intervening irrelevant material. |
| 3, optional | MAD `fuzzy-in-context-recall`: T128, configured vocabulary 16, 12,800 train examples, multi-query, motif settings 3/3 | Test retrieval with multi-token keys and values. |

These settings come from the official [recall](https://github.com/athms/mad-lab/blob/main/configs/tasks/in-context-recall.yml), [selective-copying](https://github.com/athms/mad-lab/blob/main/configs/tasks/selective-copying.yml), and [fuzzy-recall](https://github.com/athms/mad-lab/blob/main/configs/tasks/fuzzy-in-context-recall.yml) baseline configurations. Resolve the generator's actual token-ID range, including reserved symbols, before setting embedding/head vocabulary dimensions; the configured vocabulary number need not be the final model vocabulary size.

Aim for one useful paired task before spreading the budget across several. Add task 2 if its T256 numerical/memory check and timing permit. If its longer scan is too costly, task 3 can be the second task instead; select this substitution from measured cost before comparing learning outcomes. A third task is optional. Do not add compression in this first pass: its official autoencoder/readout contract is another adaptation. Do not call ordinary copying and selective copying the same task.

### Model and optimizer configurations

For this paper-oriented pilot, add a **separate 12-block D128/H16/MLP512 research configuration**, retaining sites 3 and 8 and the established pre-norm, GELU, QK-normalization and FP32 conventions. This matches the paper's reported per-block dimensions while preserving our intended topology. Keep the earlier D256 configuration and its validation as their own records. D128/H16 has head dimension 8, so perform a targeted end-to-end NUM/OPS check at this shape; old H4/D256 clearance alone does not cover it. If the new shape exposes a genuine unresolved issue, retain the working D256/H4 profile as a documented fallback rather than silently claiming the dimensions match.

The required research pair is **SEQ-12 versus CDRM-12**, with corresponding backbone initialization, identical ordered data, and equal completed updates/epochs at comparison endpoints. CDRM uses rho=1 and the active epsilon/lambda initialization already specified above; do not tune those gates for this pilot. Start fresh symbolic training; preserve all Stage B checkpoints and lineages.

Use AdamW LR=5e-4, betas=(0.9,0.98), epsilon=1e-8 and weight_decay=0 for both arms. This is one declared optimization choice, not a reproduction of the authors' tuned sweep. The optimizer's weight decay is separate from our bridge gate lambda. Keep the existing clip-norm policy (initially 1) and record it as a local choice unless an authoritative matching RT synthetic setting is found. Keep dropout off. Use FP32 throughout, with neither accumulation nor BF16.

Target **physical batch 128 on one H100 80 GB**, conditional on measured fit. Validate small batches first, then B128 with real loss/backward/update. If it does not fit, use a common smaller physical batch (64, then 32) for the pair, record the changed updates per epoch, and classify batch alignment accordingly. Do not enable unvalidated accumulation to preserve the label B128. Reduce batch before altering the selected task length or task data distribution.

R3-12/rho1 on the same task/config is a useful next arm if time permits; use its existing FP32 path only after the necessary new-shape check. Optional one-block SEQ/RT runs at D128/H16/MLP512 provide a closer link to the paper. The main CDRM/SEQ pair takes priority, followed by a second task or R3 according to the measured remaining budget. A full 12-block recurrent model and the full optimizer sweep are deferred.

### Data, metrics and contamination checks

1. Generate the fixed training set once and reuse it over epochs, with reproducible epoch shuffles shared by the pair. Do not silently substitute a fresh infinite stream. Make independent development and final-test sets, initially 1,280 examples each, with separate seeds. Record any generator collision/overlap handling; do not rewrite the distribution to manufacture novelty.
2. Inspect a few decoded examples and validate targets with a deterministic task oracle. Verify counterfactual changes to relevant keys/copy tokens change the expected answers. Preserve official supervision alignment: some synthetic targets are already aligned with their prediction position. Do not apply a second automatic LM shift.
3. Report answer-token CE, accuracy on scored positions only, and whole-example exact match over all scored positions. Compute exact match directly, not as token accuracy raised to a token count. Exclude ignored positions and flag any example with no scored target. Also retain upstream metric outputs if definitions differ, with clear names.
4. Verify causality on the task tensors and that the target token is not inadvertently available at its prediction position. If teacher-forced prior answers are legitimately part of a task, record that protocol; do not compare it with free-generation accuracy as though identical.
5. Include oracle and simple task-appropriate chance/query-ignoring baselines. Development curves are for pilot decisions; final-test evaluations occur at the declared endpoint and best-development checkpoint after choices are fixed. One training seed is acceptable here, explicitly exploratory. Do not treat an example bootstrap as training-seed uncertainty.

### Time budget, scheduling and checkpoint milestones

Use a **provisional four-hour wall-time window for this coding/validation/pilot session**, beginning when this updated work starts, on the already supplied single H100. This is a bounded interpretation of the user's request, not a runtime prediction. Honor any concrete session budget already supplied by the user. Reserve roughly 20 minutes at the end for checkpoint completion, evaluation, retention and reporting. If implementation itself consumes the window, deliver the implementation status and exact pending run commands; never bypass the numerical gate to produce a result.

After warm-up, measure 10–20 real CDRM updates and evaluation time. Estimate the cost of a complete pair before allocating further work:

\[
U_{\rm epoch}=\lceil N_{\rm train}/B\rceil,\qquad
T_{\rm pair}(E)\approx E\,U_{\rm epoch}\,(t_{\rm CDRM}+t_{\rm SEQ})+T_{\rm eval/save}.
\]

For 12,800 examples and B128, one epoch is 100 updates; 200 epochs is 20,000 updates per arm. Include compilation, data and checkpoint overhead separately. Do not extrapolate the earlier tiled-R3 timing directly to the new naïve CDRM scan.

Use a **200-epoch cosine schedule horizon**, with the scheduler's verified epoch/update stepping convention, and pause early if necessary. A 25-epoch pilot should be the first 25 epochs of that declared schedule, not a silently compressed 25-epoch cosine run. Preserve scheduler state so continuation to 200 does not restart or reinterpret the schedule. Do not import Stage B's different warm-up schedule automatically.

Before examining development results, choose a feasible first shared endpoint from approximately **10, 25, or 50 epochs**, based on measured cost. Prefer 25–50 when affordable. Retain milestones at epochs 0, 1, 5, 10, 25, 50, 100 and 200 when reached, plus latest and best-development checkpoints. Use existing economical retention policies; identical checkpoint roles may share an artifact with separate ledger entries. Prefer finishing one pair over leaving multiple isolated CDRM arms.

Continuation to a later milestone, including all 200 epochs, is already authorized if it fits the remaining window. Record why it was chosen, preserve earlier endpoints, and compare both arms at common epochs. Additional tasks and longer training compete for the same budget; neither is mandatory. Do not add independent hyperparameter sweeps or silently stop only the weaker arm based on its outcome. If wall time interrupts an arm, save recoverable state and report comparisons at the latest common checkpoint.

Save initial weights, optimizer/scheduler state once created, RNG/sampler/data position, source/config identities, gates/adapters and completed epoch/update counters. Preserve the existing distinction between an initial weight artifact and a verified resumable checkpoint. Save learning curves to local CSV/JSON and the project's existing W&B setup if configured; lack of W&B connectivity must not block training or local logs.

### Required preliminary report

Add a short report to this PR with the actual completed runs, exact configuration/task provenance, common-epoch learning curves, token and sequence metrics, baseline scores, time/memory cost, and checkpoint links/resume commands. Separate NUM, OPS and SYN evidence.

Include a **paper-alignment table** listing task semantics, sequence length, vocabulary, model depth/width/heads, optimizer, physical batch, epochs and metric definition. Mark each as matched, deliberately adapted, MAD default, or unavailable from the paper. The paper's figures are context; improvement over its one-block model is not evidence that CDRM is better than a standard 12-block model. Local SEQ-12 is the primary comparison. Do not present a partial run with one hyperparameter choice as comparable to a fully trained, tuned published endpoint.

If published numerical values are not supplied, reference the figures without inventing exact values; label any visually estimated values as approximate. Report genuine learning progress even when sequence exact match is still low. An advantage over local SEQ is preliminary evidence worth following up; both models remaining near shortcuts leaves the learning regime unresolved. Strong SEQ performance with no CDRM gain identifies a task where the added pathway has not yet helped, rather than settling all CDRM hypotheses.

The successful extended deliverable is the implemented reference **plus at least one bounded paired synthetic learning trajectory when the implementation, correctness and time budget allow**, with subsequent tasks/runs selected as above.

## Deliverables and stopping point

Deliver the core implementation and targeted tests, resolved FP32 CDRM/same-depth configurations, a usage guide with exact commands, a bounded NUM/OPS/performance report, the authorized preliminary synthetic-pilot report and checkpoints, and joint-experiment manifests. The guide should show the user how to set topology, gates, sequence/microbatch size, output location and seed; launch a smoke run; load weights; resume a supported checkpoint; and find logs/checkpoints.

Retain initial, midpoint/resume, final and any best-development operational artifacts with their role, update count, source/config/data identities and verification scope. Use the project's existing persistence/checksum conventions rather than creating new infrastructure. Never overwrite the Stage B or precision-validation lineages.

The milestone ends with a correct, trainable reference and the bounded pilot results where feasible, plus a concrete next comparison, or an explicit bounded implementation issue with its evidence. BF16 optimization, custom tiling, larger LM runs, latent objectives and the rest of the TopologyGrid remain later work; keep minimal state/config hooks for them without implementing speculative machinery now.
