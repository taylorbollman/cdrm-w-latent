# Coding-agent brief: sparse temporal recurrence, cross-depth memory, and the TopologyGrid

Updated 2026-09-04. Audience: a frontier coding/research agent starting without prior conversation context. This document is the implementation handoff; it supersedes the earlier brief. It specifies a research program, not established performance results.

## 1. Mission, scope, and first deliverable

Build a PyTorch research implementation based on [geniucos/recurrent-transformer](https://github.com/geniucos/recurrent-transformer). Begin with a standard causal Transformer containing **one Oncescu-style recurrent block at index 3**. Establish a reliable reference, then investigate a localized cross-depth memory pathway and selected configurations with a second recurrent site near index 8.

The motivating question is whether one or two temporal recurrence sites and one localized cross-depth memory pathway can capture useful serial computation while retaining token-parallel execution in the ordinary backbone. The larger scientific program is the **TopologyGrid**, with a NextLat-style auxiliary objective tested at a few selected stages. Neither a larger model nor the most elaborate topology is presumed to win.

Initial model convention: 12 blocks, indexed **0–11**, early anchor `r=3`, late source/merge site `w=c=8`. The pair 3 and 8 is symmetric in this stack. Parameterize the positions for larger models; do not automatically increase the number of recurrent sites or depth sources with total layer count.

**First coding deliverable:** reproduce upstream, enable per-block selection, implement and test conversion from a sequential model to `R3`, and provide a tiny numerical/causality test suite plus a whole-model benchmark entrypoint. Do not implement the entire research program in the first patch.

The user works in PyTorch and can potentially access an eight-H100-80GB node, preferably using fewer GPUs. Initial sequence length is 512, after tiny tests. These are available directions for experiments, not authorization to rent GPUs or launch a large training run.

## 2. Research context and sources

The central starting point is Oncescu et al., [The Recurrent Transformer: Greater Effective Depth and Efficient Decoding](https://arxiv.org/abs/2604.21215). Its defining change is that later positions attend to K/V derived from a layer's processed output. Current-position attention uses temporary input-derived K/V to avoid a circular definition. Its exact tiling exploits queries being available before persistent writes; it improves memory movement without eliminating token-to-token dependence or making dense attention linear in arithmetic cost.

We narrow that recurrence to selected sites and investigate which representation should query history, which should participate in writing it, and where the result should rejoin the backbone. Cross-depth storage, same-depth temporal recurrence, and an auxiliary latent objective are separate experimental resources.

Relevant context:

- Mozer et al., [The Topological Trouble With Transformers](https://arxiv.org/abs/2604.17121): motivates investigating limits of fixed-depth state tracking and access to deeply processed information. It does not establish that our topology solves those limitations.
- Mozer et al., [Recirculation](https://arxiv.org/abs/2608.17981): a related recurrence/belief-state direction. Our initial implementation is not an implementation of that method.
- Wang et al., [Full-bandwidth transformer](https://arxiv.org/abs/2608.08888): deep hidden-state feedback motivates a later comparison. Top-to-next-input feedback is outside the initial architecture.
- [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/abs/2511.05963), with [official NextLat code](https://github.com/JaydenTeoh/NextLat): motivates token-conditioned prediction of future hidden states and the selected auxiliary studies below. Treat our intermediate-memory variant as an adaptation.
- [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852): important precedent for localizing recurrence within a larger stack, with a different state/update design.
- [WhiteMatter](https://arxiv.org/abs/2608.18486), [FusedKV](https://arxiv.org/abs/2512.03870), and [Depth-Attention](https://arxiv.org/abs/2606.05014): relevant cross-depth mixing comparisons.
- [MAD](https://arxiv.org/abs/2403.17844) and [Zoology](https://arxiv.org/abs/2312.04927): useful starting points for controlled sequence-model diagnostics.

These references motivate controls, not a claim that generic recurrence, latent prediction, or cross-layer mixing is new. The potential contribution is the specific division of addressing, read-conditioned writing, temporal processing, and reinjection, together with evidence locating when each helps.

### Upstream snapshot and code audit

The locally inspected reference is commit [a21b42d2bc292edb86ed1b62cee4bcab809a9d21](https://github.com/geniucos/recurrent-transformer/tree/a21b42d2bc292edb86ed1b62cee4bcab809a9d21), dated 2026-04-22; code rechecked for this brief on 2026-09-04. Pin the actual revision used. If upstream differs, document relevant changes before importing assumptions.

| File / symbol | Why it matters |
| --- | --- |
| `olmo/config.py`: `ModelConfig`, `BlockType`, `Config.update_with` | Block type is currently global. Derive per-block configs without mutating the shared config. |
| `olmo/model.py`: `OLMoBlock.build`, `OLMo.__init__` | Dispatch and construction of the mixed stack. |
| `PreAttentionBlock` | Input normalization, separated Q/K/V projections, head reshaping and optional Q/K normalization. |
| `PostAttentionBlock` | Attention residual plus MLP residual. Its attention argument is already output-projected. |
| `OLMoRecurrentBlockBase.init_from_sequential_block` | Partial weight conversion; learned norm copying is commented out at the inspected revision. Extend it before using trained weights. |
| `OLMoRecurrentAutogradBlock._real_forward` | Readable token loop; primary reference for a single recurrent block. |
| `OLMoRecurrentBlockTiledFunction` | Custom autograd forward/backward; has manual gradient accumulation and recomputation. Modify only after reference semantics are fixed. |
| `OLMo.forward` | Backbone orchestration, masks, hidden-state returns and final norm/head. |
| `debug_utils.py` | Existing conversion helpers assume all destination blocks are recurrent. |
| `debug_test_recurrent.py` | Upstream naïve-versus-tiled numerical/gradient comparison. |
| `benchmark_block_latencies.py` | Block benchmark template; insufficient by itself for total training time/memory. |
| `configs/kempner/base-c4-t5.yaml` and `models/150m.yaml` | Initial paper-family setup: 12 blocks, width 1024, 16 heads, MLP width 4096 in this overlay. Verify the merged effective configuration. |

Do not rely on historical line numbers. Search the symbols and inspect callers, including all global `block_type` conditionals outside block construction. The tiled path is custom PyTorch autograd with compiled helpers; “tiled kernel” does not imply there is a standalone CUDA/Triton kernel to edit.

Initial supported configuration: pre-norm (`norm_after=false`), ALiBi, RoPE off, full MHA (`effective_n_kv_heads == n_heads`), `block_group_size=1`, no cached decoding or packed-document mode. At this snapshot the model rejects ALiBi together with its FlashAttention option; preserve a supported baseline attention path.

The base YAML itself defaults to RoPE and sequence length 1024; set and verify the ALiBi/512 overrides explicitly. The `150m` name refers to approximately non-embedding parameters, not necessarily total training parameters. The inspected debug script currently enables only its ALiBi/post-norm case despite the README describing four combinations. Add an explicit pre-norm comparison for this brief's initial contract; do not assume the default command covers it. If the chosen real checkpoint uses post-norm, preserve that fact and use the supported rho=1 recurrence or design/test a separate homotopy; do not silently change normalization to obtain an equivalence test.

Start with dropout off for numerical comparisons. Disable optional model-level compilation/CUDA graphs for the oracle; note that upstream helper functions already have `torch.compile` decorators. If a fully eager reference is needed, explicitly bypass those wrappers. “compile=null” alone does not remove every decorator.

A repository configuration or training command is **not** a released pretrained checkpoint. Prefer the paper's compatible architecture and an actual verified checkpoint if available; otherwise make tiny conversion fixtures and identify a usable checkpoint separately. Do not promise checkpoint availability or substitute a different tokenizer/positional encoding silently.

## 3. Vocabulary, experiment identities, and migration

Use architecture identity separately from execution backend. `naive` and `tiled` should mean the same mathematical model when numerical equivalence has been demonstrated.

| ID | Meaning |
| --- | --- |
| `SEQ` | Ordinary causal backbone. |
| `R3` | Replace block 3 by an Oncescu recurrent block; all others standard. |
| `R8` | Replace block 8 only; late-site control. |
| `R3R8` | Replace both blocks in the ordinary stack; conventional two-site control. |
| `DM` | Nonrecurrent depth-mixing/extra-compute bridge from preview states. |
| `CDRM` | Cross-Depth Read-Conditioned Recurrent Memory: early query, deep current-state writer input, one shared growing memory, bridge after the late source. |
| `DW0` | Deep-write-without-read-conditioning control defined below; controls direct transport of processed deep information. |
| `CDRM-A` | Deep recurrent refinement first, then CDRM, then the suffix. |
| `CDRM-B` | CDRM first, then deep recurrent refinement, then the suffix. |
| `CDRM-A-replace` | Explicit lower-compute variant replacing ordinary block 8 instead of adding refinement. |
| `NL1 / NL2 / NL3` | Three selected auxiliary-study entry points, not three compulsory new architectures. |
| `MultiDepthWriter` | Deferred original idea: several depth candidates processed in the writer, aggregated before one persistent write. |
| `DQ / DS` | Deferred shared-memory multi-query / independent-memory-stream alternatives. |

Old `R1-N/T` means `R3` with naïve/tiled execution; “1” counted recurrent blocks, not the index. Keep aliases only when importing old records.

**Resolve the old DW ambiguity explicitly.** The old `DW-N-B8` batched depth candidates through `PostAttentionBlock(candidate, a_t)` before aggregation. That already uses retrieved history in its write. It cannot serve as a “no read-conditioned write” baseline against CDRM. Preserve that construction as `MultiDepthWriter`; do not rewrite old experiment metadata as if it meant `DW0`. The latter is a new, precisely specified ablation, not a claim about what the old code did.

Old `PW-CDRM` was a final-only, possibly token-unconditioned future-latent proposal. The current plan allows selected NextLat-style studies earlier. Log token-conditioned and token-unconditioned objectives separately.

## 4. Tensor and recurrence contract

Let `B` be device microbatch size, `T` sequence length, `D` model width, `H` attention heads. Backbone states have shape `[B,T,D]`; initial MHA K/V have shape `[B,H,T,D/H]`.

- `x[j]` is the input to block `j`.
- `p[j]` is an ordinary causal preview's output after block `j`.
- `p3` means post-block-3 preview, not the input to block 3.
- `p8` is the unmodified post-block-8 preview.
- `u8` is an output of a late recurrent transform.
- `v8` is a state after applying the CDRM bridge.
- `m[t]` is a residual-space memory record; it is not a raw K/V tensor.
- `a[t]` is the read output after head merging and `attn_out`, before the writer's residual/MLP.

All preview states at position `t` depend only on tokens `<=t`. Each scan reads persistent entries `<t` and its current temporary entry. It writes entry `t` only after the writer completes. No future temporary entries participate.

### 4.1 Ordinary single-block recurrence: R3

Let `Pre3` perform the existing Q/temp-KV preprocessing, `Post3` the existing post-attention computation, and `KV3` the persistent projection including the selected normalization/clipping/key-normalization policy:

\[
(q_t,k_t^{tmp},v_t^{tmp})=\mathrm{Pre3}(x[3]_t),
\]
\[
a_t=\mathrm{Read3}(q_t,\{(k_i,v_i):i<t\}\cup\{(k_t^{tmp},v_t^{tmp})\}),
\]
\[
z_t=\mathrm{Post3}(x[3]_t,a_t),\qquad
w_t=(1-\rho)x[3]_t+\rho z_t,\qquad
(k_t,v_t)=\mathrm{KV3}(w_t).
\]

Pass `z` to block 4. The interpolation `w` affects what is stored, not what is passed upward. `rho=1` is the intended output-derived recurrent write. `rho=0` is a proposed warm-start homotopy endpoint and must match ordinary attention only under a compatible projection/normalization contract.

At the inspected upstream revision, persistent K/V always apply `attn_norm(out_t)`, while temporary K/V skip that normalization when `norm_after=true`. Therefore **do not assert rho=0 equivalence for arbitrary upstream configurations**. Initially require pre-norm; test all copied norm, clipping, bias and Q/K-normalization behavior. Supporting post-norm later requires an explicit endpoint-consistent design or a separately labeled homotopy.

Only block 3 is recurrent in R3. Run blocks 0–2 for all tokens, scan block 3 across time, then run blocks 4–11 for all tokens. Teacher forcing supplies tokens; it does not remove the within-block sequential dependence.

### 4.2 Primary CDRM: preview, read, write, bridge

CDRM initially uses **ordinary blocks 0–8** for a causal preview. It saves `p3` and `p8`, executes the side memory scan, and passes a corrected `p8` to ordinary blocks 9–11.

It does not replace preview block 3 by R3 and also apply a side scan. That would be an additional topology. It also does not replay blocks 4–8 after the scan.

For the default shared-block writer:

\[
c_t=p3_t+\epsilon A_d\bigl(N_d(p8_t-p3_t)\bigr),
\]
\[
(q_t,k_t^{tmp},v_t^{tmp})=\mathrm{Pre3}(p3_t),
\qquad
a_t=\mathrm{Read3}(q_t,M_{<t}\cup\{(k_t^{tmp},v_t^{tmp})\}),
\]
\[
\hat m_t=\mathrm{Post3}(c_t,a_t),\qquad
m_t=(1-\rho)p3_t+\rho\hat m_t,
\qquad
M_t=\mathrm{KV3}(m_t),
\]
\[
v8_t=p8_t+\lambda A_b\bigl(N_b(\hat m_t-p3_t)\bigr).
\]

Here `A_d` and `A_b` are residual-space adapters; `N_d` and `N_b` are stable normalizers. Use ordinary PyTorch modules. Reuse block-3 parameters initially for `Pre3`, `Read3`, `Post3` and `KV3`; register each parameter once under a canonical owner. A separate writer is a later parameter/compute-controlled variant.

These equations choose an **anchor-relative correction** consistently. Alternatives such as `m-p8` or `m-c` change the bridge and must be separate configurations, not silent substitutions. At `rho=1`, `m=hat_m`. The bridge deliberately uses `hat_m` so that a recurrence-strength ramp does not also switch off same-token writer computation.

The design invariants are more important than this initial adapter parameterization:

1. The query/temp pair is formed from the saved early preview.
2. Retrieved history and the current deep preview can both affect the write.
3. Exactly one persistent pair is produced after the final write representation.
4. The correction is a residual-space tensor; never add raw K/V to `p8`.
5. The memory is a growing collection of records, not a single overwritten state.
6. There is no same-position feedback into the preview that produced the query.

**Location is representational, not execution order.** “Read at index 3” means the reader uses `p3` and block-3 projections. The reader executes after the deep preview is ready. Blocks 4–8 have not consumed that read. This architecture gives later memory queries access to deeply conditioned records and gives the suffix a correction; it does not make the ordinary shallow preview itself see recurrent memory.

With `lambda=0` and no other altered backbone blocks, CDRM logits equal SEQ logits under the deterministic test setting. Extra branches can consume dropout RNG, so an exact training-mode comparison must also control stochastic masks. `rho=0` instead removes output-derived persistent updates while retaining the read/writer/bridge; it is not a global SEQ switch.

### 4.3 Minimal controls before elaborate writers

**DM:** form a pointwise deep candidate `c_t`, compute `f_t=Post3(c_t,0)`, and bridge `f_t-p3_t` using the same adapter and gate. There is no extra memory scan. This retains deep-source and nonlinear writer compute. Ordinary attention in the backbone remains present.

**DW0:** compute `m_t=Post3(c_t,0)` independently of the memory read, project its permanent K/V, and let the early query retrieve those records from positions `<t` plus its current temporary pair. Use a separate, matched bridge path from the read and current candidate, e.g. `Post3(c_t,a_t)-p3_t`. The exact bridge is recorded and reused for comparisons.

DW0's stored record is independent of `a_t`. Its persistent records can be precomputed in parallel. It provides addressable deep memory without a recursive read-to-write chain. An oracle may still loop to simplify comparison, but this is not evidence of a true recurrent writer. Account for any additional writer/bridge computation in the comparison.

Additional surgical CDRM controls:

- **Current-only:** remove historical entries, retain own temporary pair, writer and bridge.
- **Read-to-write removed:** substitute a fixed zero read in the writer; label any simultaneously removed output/read path.
- **Same-depth:** replace `p8` by `p3` only in the writer source, preserving adapter/bridge capacity.
- **Stopped temporal credit:** stop gradient through selected historical writes without changing their forward values. This tests optimization/credit, not removal of recurrence.
- **Linear/normalized depth mixture:** replace the nonlinear consolidation with a simple depth fusion.
- **Memory-size/extra-compute controls:** match extra K/V storage or extra transforms separately where practical.

No one ablation perfectly matches all resources. Record the dependency removed and residual differences rather than presenting a misleading matched comparison.

### 4.4 Reference scan pseudocode

The following specifies dependencies, not a drop-in upstream API:

```python
# All preview sources exist before this scan. Shapes preserve a length-1 axis.
q, k_tmp, v_tmp = pre3(p3)
candidate = make_candidate(p3, deep_source)
records, outputs = [], []
history_k, history_v = [], []

for t in range(T):
    keys = torch.cat(history_k + [k_tmp[:, :, t:t+1]], dim=2)
    vals = torch.cat(history_v + [v_tmp[:, :, t:t+1]], dim=2)
    read = read3(q[:, :, t:t+1], keys, vals, bias_for(t))
    proposed = post3(candidate[:, t:t+1], read)
    record = (1 - rho) * p3[:, t:t+1] + rho * proposed

    # Write AFTER consolidation; this slot is not read by the current token.
    kt, vt = permanent_kv3(record)
    history_k.append(kt)
    history_v.append(vt)
    records.append(record)
    outputs.append(proposed)

memory_states = torch.cat(records, dim=1)
proposed_states = torch.cat(outputs, dim=1)
bridged = deep_source + bridge(proposed_states - p3)
```

Do not detach these lists/tensors in the training oracle. Do not copy into mutable inference buffers to save memory. Such optimizations require their own correct backward design. Keep padding/document masking explicit; the first run can use unpadded single-document examples.

## 5. Two recurrent sites: schedules A and B

A second recurrent site is an optional TopologyGrid point. Core CDRM does not require it. **Schedule C, a joint/interleaved scan, is excluded from the current plan.** A/B each complete one sequence-wide scan before the next begins.

### Resolve replacement versus refinement

“Make index 8 recurrent” has two valid implementations that must not share an experiment ID:

- **Replacement:** `R8(x[8])` takes post-block-7 inputs and replaces ordinary block 8.
- **Refinement:** `R8_refine(p8)` takes a state that has already passed through ordinary block 8. It adds one more attention/MLP transform.

For a clean initial A/B ordering comparison, use a **common ordinary preview through index 8** and the **same additional recurrent refinement module** in both. This follows the preview-then-two-scans construction and gives both schedules the same transform inventory. Initialize the refinement module from a copy of block 8; do not tie it to preview block 8 by accident. Record its logical location and extra parameter/FLOP cost.

Also expose the replacement version as `CDRM-A-replace` if desired. It is a distinct, cheaper topology, not an optimized implementation of identical mathematics.

### Schedule A: deep recurrence, then cross-depth memory

```python
p3, p8 = ordinary_preview_0_through_8(tokens)
u8 = recurrent_refine8(p8)              # complete entire temporal scan
m3, delta3 = cdrm_scan(p3, deep=u8)     # complete entire temporal scan
v8 = u8 + bridge(delta3)
logits = ordinary_suffix_9_through_11(v8)
```

The late recurrence records K/V from `u8_t` before the later bridge. The CDRM correction does not retroactively overwrite its cache. CDRM has a separate memory stream. This order makes the CDRM writer's deep source itself temporally recurrent.

**A-replace variant:** run ordinary blocks 0–7, apply recurrent block 8 to `p7`, run CDRM using `p3,u8`, bridge, then run 9–11. Do not also compute ordinary `p8` unless a separately justified target/control needs it.

### Schedule B: cross-depth memory, then deep recurrence

```python
p3, p8 = ordinary_preview_0_through_8(tokens)
m3, delta3 = cdrm_scan(p3, deep=p8)     # complete entire temporal scan
v8 = p8 + bridge(delta3)
u8 = recurrent_refine8(v8)             # complete entire temporal scan
logits = ordinary_suffix_9_through_11(u8)
```

Here late recurrence consumes CDRM-corrected inputs; its persistent writes reflect that correction. CDRM's own current write uses raw `p8` and cannot read the later `u8`. Replacing its source with `u8` would create another dependency/cycle and is not Schedule B.

Both schedules allow all queries for a particular scan to be formed once that scan's input sequence is ready. Each can reuse a validated standalone late recurrent block, including its unmodified tiled implementation. CDRM itself still needs a separately validated backend.

A and B are different functions; do not write an equality test between them. With two comparable scan modules, adjacency alone does not justify calling one dramatically faster or harder. Benchmark actual execution. The released tiling relies on completed input sequences, not on fusing two recurrent modules into a single loop.

### Count ordinary parallel regions honestly

| Topology | Ordinary token-parallel regions |
| --- | --- |
| R3 | 0–2; 4–11: two |
| R8 | 0–7; 9–11: two |
| Conventional R3R8 | 0–2; 4–7; 9–11: three |
| CDRM | 0–8 preview; 9–11 suffix: two |
| CDRM-A / B | 0–8 preview; 9–11 suffix: two, with two scans between |
| CDRM-A-replace | 0–7 preview; 9–11 suffix: two, with two scans between |

The “two regions” property describes the main execution layout. It is not a claim of constant overhead, full parallelism, or a property of every grid point. Ordinary blocks are parallel across positions, not across layers.

## 6. The TopologyGrid is the experiment, not an escalation ladder

Keep topology, backend and auxiliary loss as distinct configuration dimensions. The grid does not require a full Cartesian product, nor an ordering such as `SEQ < R3 < DM < CDRM`.

Record these independent properties:

| Property | Initial choices / what it tests |
| --- | --- |
| Temporal recurrence sites | None, early only, late only, conventional early+late, CDRM, CDRM+late refinement |
| Query source | Early preview by default; a deeper query is a later ablation |
| Write source | Same-depth versus deep-preview versus temporally processed deep state |
| Read-to-write edge | Present, absent in forward, or stopped only in backward |
| Merge location | Late bridge initially; replay only if evidence calls for it |
| Addressable memory | Number of streams, number of stored slots, K/V width, accessible history |
| Schedule | None, A, B, or explicitly A-replace |
| Auxiliary objective | None or the selected NL1/NL2/NL3 study |
| Backend | Reference, validated upstream tiled, or later custom CDRM tiled |

Use three initial task families before a large language-model study:

1. **State updates:** events modify a latent state and the model must report it after variable delays; vary the number/composition of updates independently of sequence length.
2. **Delayed associative retrieval:** vary the number of distractors and retrieval distance without increasing the required update chain.
3. **Retrieve–update–retrieve:** retrieve a prior entity/event, combine it with a new observation, then query the revised result later; vary whether successful writing needs historical context.

Use held-out lengths, event compositions and entity identities. Include generator tests and simple baselines to expose shortcuts. Do not merely rename a language-model perplexity gain “state tracking.”

The operational hypothesis is that a stored record `m_t=F(deep_t,a_t)` can make a later relevant state easier to recover than a write based on the deep state alone. An information-theoretic intuition is the possible predictive contribution of `a_t` beyond the deep state, or vice versa. Because both are deterministic computations of overlapping prefixes, this is not new input information and does not imply positive conditional mutual information in every distribution. Improved accessibility/learnability under bounded computation is the target.

Use held-out loss differences, interventions and task scaling to test that hypothesis. Attention entropy, probes, gradient sensitivity and rank are diagnostics, not direct mutual-information estimates or proof of belief-state sufficiency. A growing K/V collection should not be described as one compact Markov state without a separate argument.

### Budget and attribution

For each comparison record parameters, persistent state size, training tokens, optimizer/data schedule, measured forward+backward cost, peak memory, and wall time. Use fixed-token comparisons and a selected matched-compute or matched-wall-time comparison; do not imply those budgets can all be equal at once.

Shared weights save parameters, not repeated FLOPs. Extra memory can improve a result independently of recurrence. Reusing a writer MLP K times incurs K evaluations even if flattened batching improves utilization.

Train the most important ablations, rather than relying only on test-time lesions. A trained no-history control asks whether history was necessary to learn the task; a test-time lesion asks whether a trained model currently relies on it. Their conclusions differ.

For shuffled-history checks, preserve validity/causality and specify what is shuffled. Jointly permuting K, V and their position metadata can leave attention unchanged. Shuffling content across fixed positions, corrupting associations, and replacing records with matched unrelated records test different things. Never allow a “shuffle” to introduce future information.

## 7. TopologyGrid plus latent dynamics: three selected entry points

Latent prediction is a related auxiliary-training question that can be tested at selected topologies. It is not a requirement to augment every grid point and not restricted to the end of the project.

| Entry point | When | Initial source and target | Comparison |
| --- | --- | --- | --- |
| **NL1 — basic temporal recurrence** | After SEQ/R3 conversion and recurrence tests are trusted | Final hidden state `h_t` plus next-token embedding predicts stopped `h_(t+1)` | Same head/loss on SEQ and R3 |
| **NL2 — cross-depth memory** | After core CDRM and its controls work | Memory record `m_t` plus next-token embedding predicts stopped raw deep preview `p8_(t+1)` | CDRM with/without objective, ordinary deep-source predictor, and relevant write control |
| **NL3 — selected final topology** | After the planned topology study | Reuse the best justified objective on selected finalist(s), including A/B only if retained | Best architecture with/without objective and matched simpler baseline |

NL1 and NL2 are optional branch points with their own small budgets. NL3 is the final consolidation experiment, not an obligation to repeat every earlier combination. Do not let auxiliary work delay correctness of the core recurrence.

### Initial one-step objective

Let `e_(t+1)` be the embedding of the actual next observed token. For a chosen causal source `s_t` and target representation `y_(t+1)`:

\[
\hat y_{t+1}=g_\phi(s_t,e_{t+1}),\qquad
\mathcal L_{\mathrm{lat}}
=\frac{1}{|\mathcal V|}
\sum_{t\in\mathcal V}
\mathrm{SmoothL1}\left(\hat y_{t+1},
\operatorname{sg}[N(y_{t+1})]\right),
\]
\[
\mathcal L=\mathcal L_{\mathrm{LM}}+\beta\mathcal L_{\mathrm{lat}}.
\]

Here `V` is the set of valid adjacent pairs. Average over latent dimensions explicitly; log both raw and weighted auxiliary loss. Start with horizon one and a small predictor matching the chosen NextLat implementation or a clearly labeled lightweight MLP adaptation. Sweep a small set of `beta` values including zero.

This predicts the latent **after incorporating a known next token**. It is not the same problem as predicting that token or forecasting a future state without observing its intervening input. A token-unconditioned predictor is an informative later ablation, with a distinct name.

Default target conventions:

- NL1: use the model's final normalized hidden state, consistently before its LM head. Avoid a second accidental final norm.
- NL2: use the raw ordinary `p8` before the bridge, optionally with a fixed, nonlearned normalization for scale. The target is not `v8` or `u8` unless specified as a separate experiment.
- NL3 on common-preview A/B: retain raw `p8` when testing the same memory-to-preview question. A's memory is conditioned on `u8`, but its target can remain raw `p8` for comparability. For A-replace there is no ordinary `p8`: choose a new documented target such as prebridge `u8` and treat that change explicitly, rather than computing an otherwise unused preview without accounting for it.

Default source gradients remain enabled so the auxiliary loss can shape the backbone or memory writer. The target branch is stopped. The same target tensor can still receive gradients from ordinary LM computation through its other uses.

Compute all valid one-step predictions as a batch after the causal forward. Do not add a recurrent predictor loop for horizon one. Discard the auxiliary predictor at ordinary inference; no future token/target enters the memory update, attention, bridge or current-token LM logits.

```python
source = states[:, :-1]                  # final h, or CDRM memory m
next_embed = input_embeddings[:, 1:]
target = target_normalize(target_states[:, 1:]).detach()
prediction = latent_predictor(source, next_embed)
per_pair = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
latent_loss = masked_mean(per_pair, valid_adjacent_pairs)
loss = lm_loss + beta * latent_loss
```

### Optional distribution-consistency term and gradient audit

NextLat also motivates a token-distribution consistency loss. Defer it until the one-step latent objective is understood, or implement it separately when reproducing the paper exactly. A full-vocabulary KL can add substantial memory/cost.

For final hidden states, a detached target distribution and frozen LM-head weights can define a KL while preserving gradients through the predicted latent. **Freezing head parameters does not detach the head's input or prevent gradients reaching the backbone through the predictor.** Do not wrap the predicted branch in `no_grad`. Test gradient ownership rather than inferring it from paper prose.

For an intermediate `p8` target, the final LM head is not automatically a calibrated decoder. Do not reuse it silently; begin with latent loss only. A learned intermediate decoder/teacher is another experiment with extra parameters and supervision.

Do not add an EMA teacher, variance regularization or multi-step rollout in the first auxiliary patch unless a measured issue requires it. Stop-gradient plus LM training does not mathematically guarantee noncollapse. Monitor representation variance/rank and downstream validation, not only latent loss.

For multi-step rollout, NL1 has a natural repeated hidden-state transition. NL2 maps `memory -> deep preview`, which is not closed on one state space. Feeding its prediction back as though it were another memory record requires a new transition definition; postpone it.

The paper's belief-state reasoning does not automatically transfer to one intermediate record in our growing-memory model. Label “JEPA-like” as the predictive-latent connection, not a proof of a JEPA architecture or sufficient belief state.

## 8. Configuration and ownership

Favor a small validated configuration over many ambiguous booleans. The following is a semantic schema, not an upstream-ready YAML command:

```yaml
experiment:
  topology: R3              # SEQ, R3, R8, R3R8, DM, DW0, CDRM, CDRM-A, CDRM-B
  backbone_layers: 12
  early_index: 3
  late_index: 8
  recurrent_backend: naive
  write_rho: 1.0

  memory:
    query_source: preview_3
    write_source: preview_8
    mode: shared
    read_conditions_write: true
    projection_owner: block_3
    writer_owner: block_3

  bridge:
    mode: anchor_delta
    gate: 0.001
    deep_source_gate: 0.001

  late_recurrence:
    placement: none         # none, replacement, post_preview_refinement
    schedule: none          # none, A, B
    parameter_source: copy_block_8

  latent:
    enabled: false
    stage: none             # NL1, NL2, NL3
    horizon: 1
    condition_on_next_token: true
    source: final_hidden
    target: final_hidden
    target_stop_gradient: true
    loss: smooth_l1
    weight: 0.0
```

Validate topology-specific fields; unused fields should not silently activate behavior. Derive ordinary-block replacements from the topology. In CDRM, block 3's *preview* remains standard even though the side reader/writer uses its parameters.

Initially reject unsupported RoPE, GQA, grouped blocks, cache arguments and document packing. Validate unique/in-range indices and `early < late < n_layers`. Use separate cache objects for CDRM and the late recurrent module; prohibit aliasing. A/B require explicit placement and parameter ownership.

Ensure shared modules have one canonical registration, optimizer parameters are not duplicated, and state-dict aliases are understood. Inspect upstream aliases in `pre_attention_block` and `post_attention_block`; do not add a second independent owner to make calling convenient. Copying a block is different from tying it.

Suggested interfaces:

- `block_type_for_layer(config, index)`
- `convert_checkpoint(source, target, conversion_spec) -> ConversionReport`
- `ordinary_preview(tokens, capture_indices) -> PreviewStates`
- `pre_attention(x)`, `read_history(...)`, `post_attention(residual, read)`
- `persistent_kv(record)`
- `cdrm_scan(p3, deep_source, masks) -> MemoryResult`
- `apply_bridge(deep_source, proposed_state, anchor_state)`
- `run_late_recurrence(input_states)`
- `latent_loss(states, embeddings, valid_pairs, spec)`

Use typed result containers. Return intermediate states only when needed; returning every state unconditionally prolongs graph lifetimes and changes memory measurements. Do not detach trainable source states to reduce memory; diagnostics may detach their own copies.

Keep upstream hidden-state output semantics. At the inspected revision entries are appended before blocks, with a final state afterward. The state entering block 9 is `v8` for CDRM/A and `u8` for B. Expose raw previews through a separate named result rather than misleading `hidden_states` indices.

## 9. Checkpoint conversion and gradual activation

### Sequential to mixed

Split fused `att_proj` into Q and K/V using the actual fused dimensions, including biases. Copy attention output, MLP, attention/MLP norms and optional Q/K norms. Copy all unchanged blocks, embedding, final norm and head exactly. Test nondefault learned norm parameters; the inspected upstream helper omits their copying.

Return a report of copied/transformed/new/missing/unexpected keys. A successful `strict=False` load is not a conversion test.

### All-recurrent to mixed

Concatenate recurrent Q and K/V projections for blocks converted back to sequential, and copy the other parameters. This is a **weight warm start**, not a function-preserving conversion: removing recurrence changes those blocks' computation. Rho-zero tests only establish equality to a constructed sequential counterpart under the supported contract; they do not preserve the original all-recurrent model.

For CDRM, retain a standard preview and its weight ownership. Starting with a trusted SEQ checkpoint is the cleanest default. Moving from an R3-trained checkpoint back to a standard preview changes behavior even if weights transfer perfectly.

### Optimizer and initialization

Initialize a fresh optimizer and re-warm learning rate for the initial continuation study; give the sequential control the same optimizer reset/re-warm. Exact optimizer continuation requires explicit conversion of fused/split parameter moments and parameter IDs.

For R3, test `rho=0` exactly, then ramp to one as a separate experimental variable. A modest ramp over an initial fraction of the continuation budget is a starting hypothesis, not an upstream requirement.

For CDRM, `lambda=0` is an equivalence-test setting. Use a small nonzero bridge gate for training so the writer gets LM gradients. Avoid zeroing both the bridge matrix and gate: that can block all learning in that path. With a nonzero gate and zero matrix, the matrix can learn first but upstream writer gradients initially vanish; document this staged effect if chosen. A simple initial choice is a nonzero adapter initialization with a small gate.

Use a small deep-source gate with a nonzero adapter if immediate deep-source gradients are desired. Ramp one coupling at a time when diagnosing instability. Record `rho`, deep-source strength and bridge strength separately.

A zero CDRM bridge does not turn off an R8 transform elsewhere. A refinement with `rho8=0` is still an extra ordinary transform; it does not reproduce a 12-block SEQ model. Use explicit module bypass for the relevant equivalence test.

## 10. Development phases and acceptance gates

### Phase 0 — Reproduction and provenance

Inspect `AGENTS.md`, the existing checkout and user changes. If starting empty, clone the official repository and pin a revision. Use an isolated branch/worktree when appropriate; preserve user work.

Read the effective configuration and upstream tests before execution. Install a compatible environment, record Python/PyTorch/CUDA/GPU versions, and run the smallest upstream naïve/tiled comparison. If no CUDA is available, complete code inspection and CPU-capable pure-reference tests; report GPU tests as pending. Do not invent measurements.

Verify actual checkpoint and data availability independently. Do not download a large corpus or claim a paper-scale reproduction from a config file alone.

**Gate:** known upstream behavior and a saved baseline report, including any pre-existing failures. Diagnose failures rather than hiding them by loosening tolerances.

### Phase 1 — R3 foundation

Implement per-layer config selection, complete conversion, the pre-norm homotopy and mixed-model tests. First use tiny `B,T,D`. Then establish R3 at sequence 512 on the actual hardware.

Where feasible, run unmodified upstream tiled R3 at `rho=1` as an efficiency comparator. There is no reason to wait until all CDRM experiments finish before using the released exact tiled block for ordinary R3. Fractional-rho support in the custom tiled backward is separate work.

**Gate:** correct mixed topology, outputs/gradients, causal dependence, conversion and measured memory. NL1 may branch here.

### Phase 2 — Preview, bridge and simple controls

Implement standard preview capture and suffix orchestration. Prove the bypass path matches SEQ. Add DM and the precise DW0/read-disabled controls. Fix and record the bridge convention.

**Gate:** controlled depth/extra-compute baselines, no future leakage, clean zero-gate behavior and measured added cost.

### Phase 3 — Core CDRM

Implement the reference equations using one query, one memory stream, one write per token. Compare to the surgical controls. Run small state-tracking and retrieve–update–retrieve pilots, then a bounded LM continuation if justified.

**Gate:** outputs/gradients are trustworthy and results can distinguish read-conditioned storage from depth mixing or extra transforms. NL2 may branch here. A negative result need not halt a distinct latent hypothesis, but give it a bounded explicit budget.

### Phase 4 — Selected topology comparisons and late recurrence

Add R8 and conventional R3R8 controls. Implement common-preview Schedule A using standalone scan modules first, then B if it answers a live question; its code should mostly change orchestration. Treat A-replace as a separate optional compute-saving architecture.

Compare two-site benefits against extra-transform and extra-memory controls. Measure actual activation lifetime; freeing forward caches does not necessarily free tensors saved for backward.

**Gate:** interpretable topology effects rather than an unbounded sweep. NL3 is available after topology selection.

### Phase 5 — Optimize selected semantics

Profile the actual bottleneck. Reuse tiled R3/R8 unchanged where valid. Port CDRM to a custom tiled/autograd implementation only after freezing its equation and parameter ownership.

CDRM's early queries and deep candidates are known before its scan, which is compatible with the key availability condition behind tiling. It does **not** make the upstream custom backward a drop-in replacement. New gradients reach both early and deep sources, shared preview/writer parameters, adapters and memory writes.

A tiled CDRM must match reference outputs, input gradients and every parameter gradient, including multiple contributions to shared weights. Test gradient accumulation and recomputation; keep the naïve oracle as the authority.

**Gate:** measured whole-model benefit and verified equivalence. Optimization must not silently substitute independent streams, detached history or truncated recurrence.

## 11. Required correctness tests

These tests protect the research question. Keep them small and meaningful rather than duplicating implementation details.

1. **Structure/config:** only specified replacement blocks are recurrent; CDRM preview block 3 is ordinary; unsupported combinations fail clearly.
2. **Conversion:** exhaustive key report, nondefault norms/biases, state-dict round trip and no duplicated optimizer parameters.
3. **R3 endpoints:** pre-norm `rho=0` matches the sequential counterpart in logits, loss and gradients; `rho=1` matches the reference recurrence. At `T=1` changing persistent-write strength cannot change the current output.
4. **Preview/bypass:** zero bridge with otherwise ordinary backbone matches SEQ; output-hidden-state entries reflect actual suffix inputs.
5. **Temporal causality:** perturb tokens after cutoff `c`; outputs up to `c` do not change. All memory indices obey the same document/causal boundary.
6. **Read/write order:** current output uses current temporary K/V, never its own persistent K/V; exactly one permanent slot is appended per token.
7. **Temporal gradients:** a later loss can reach an earlier write through history in a nondegenerate fixture; no accidental detach or in-place version error.
8. **Vectorization:** a small explicit loop matches candidate batching or other vectorized primitives in outputs and gradients.
9. **Schedule semantics:** A's late cache comes from prebridge `u8`; B's late transform consumes `v8`; streams never alias; no expectation A equals B.
10. **Auxiliary gradients:** targets are stopped in the auxiliary path; predictor/source gradients are present as specified; final-head freezing preserves predicted-input gradients if KL is enabled.
11. **No auxiliary leakage:** requesting latent loss does not alter causal LM logits; future tokens/targets never enter the forward memory path; adjacent targets respect padding/document boundaries.
12. **Tiled equivalence:** compare every trainable parameter, inputs and outputs under supported settings. A global aggregate error metric must not hide skipped/zero gradients.

Use deterministic FP32 tiny tests before BF16. Document absolute and relative tolerances and inspect outliers. Different kernels need not be bitwise equal.

For the reference, ordinary PyTorch autograd should own trainable recurrence. The upstream custom tiled function performs inner backward calls and parameter-gradient side effects; do not assume compatibility with arbitrary FSDP, DDP, activation checkpointing or higher-order gradients. Before scaling, verify one-GPU accumulation, then the specific distributed configuration with a small comparison to a known-correct result.

## 12. Memory, latency and practical execution

Benchmark whole training steps, not only forward blocks. Include loss, backward, optimizer step, activation checkpointing policy, accumulation and distributed communication. Separate compile/capture warm-up from steady-state time and synchronize CUDA around timed regions.

Log device microbatch `B`, sequence length `T`, GPU count and accumulation. Effective sequences per update are:

\[
B_{\mathrm{global}}=B_{\mathrm{device}}\times n_{\mathrm{data\ parallel}}\times n_{\mathrm{accumulation}}.
\]

Gradient accumulation does not enlarge the simultaneous MLP batch inside the recurrent forward. A forward token-loop writer sees roughly `B` rows per step; an ordinary token-parallel MLP sees roughly `B*T`.

For a rough replacement-only timing decomposition, if one recurrent layer costs `r` times a typical standard layer and all others are similar, total layer time scales approximately as `(L-1+r)/L`. This is a bookkeeping model, not a benchmark prediction. CDRM and post-preview refinement add work and require their own sum of measured components.

A single persistent MHA memory stream requires approximately

\[
2BTD\cdot\mathrm{bytes\_per\_element}
\]

for K and V alone. Actual training memory includes the entire autograd graph. In the naïve implementation, concatenated growing K/V prefixes can yield an additional `O(BDT^2)` footprint. If both BF16 prefixes are retained at every step, their sum is approximately `2BDT(T+1)` bytes: about 8 GiB at `B=16,T=512,D=1024`, **before** other allocations. This is a conditional accounting estimate, not a guaranteed peak.

Replacing a standard block with recurrence does not automatically add a whole extra inference K/V stream; adding a CDRM side memory does. For equal-width full MHA, one added stream is roughly `1/L` of the ordinary backbone's K/V storage, excluding additional refinements, temporary state and training activations. Width/head choices and actual cache lifetimes matter.

Start tiny, then increase microbatch on one GPU with headroom. Use accumulation for the target effective batch. Add GPUs only after measuring the actual limit; DDP replicates model/optimizer state and does not combine GPU memory into one pool. Sharding adds a separate implementation and communication question.

Required benchmark outputs: topology/backend, effective config, forward and training-step time, tokens/s, allocated/reserved peak memory, optimizer state, dtype, sequence/microbatch, warm-up, hardware and commit. Capture OOMs as measured limits. Do not repeat unsupported claims of a particular slowdown or H100 fit.

## 13. Diagnostics, decisions and interpretable outcomes

Continuously log a modest set of scalar diagnostics:

- Bridge correction RMS relative to its receiving state; deep-source and recurrence gates.
- Permanent K/V RMS versus position, history/current attention mass and attention entropy.
- Memory-record variance/effective rank and source/read contributions.
- Gradient norms for writer, adapters, bridge and selected earlier writes.
- Task accuracy/LM validation, tokens/s and peak memory.
- With latent loss: raw/weighted objective, predictor/source gradient norms, target scale and downstream performance with the predictor unused at inference.

Sample expensive lag-sensitivity or representation diagnostics periodically, not every step.

| Observation | Supported interpretation / next check |
| --- | --- |
| R3 improves on SEQ, with stronger gains as update chains lengthen | Evidence that one temporal recurrence site is useful; check token/compute budgets and trained history controls. |
| DM matches CDRM | Depth mixing or extra transforms may explain the gain; do not credit recurrent consolidation. |
| DW0 matches CDRM | Addressable deep records may suffice without recursive read-to-write updates. |
| CDRM gains over matched controls and loses the gain when read-conditioned writing is removed | Evidence for the specific consolidation hypothesis; test longer chains and held-out composition. |
| A and B differ consistently | Ordering and which state becomes recurrent memory matter; inspect matched operations and target choices. |
| Two sites help only with more parameters/compute | Capacity/compute remains a plausible explanation; include the matched extra-transform baseline. |
| Gains disappear when history is removed | Evidence the trained solution uses history, subject to lesion distribution shift; this is not evidence that history was irrelevant. |
| No gain, gates/gradients near zero | The mechanism may not have been learned or exposed; perform bounded initialization/optimization diagnostics. |
| No gain despite verified use and adequate training | A meaningful negative result for that topology/task/budget; retain the simpler architecture. |
| Latent objective improves only latent prediction | More predictable states have not yet produced useful task computation. |
| NL improves CDRM and the simpler baseline equally | Generic auxiliary training benefit; no evidence of a special interaction with memory topology. |
| NL helps CDRM disproportionately under matched controls | Evidence of an interaction worth investigating, not proof of a sufficient belief state. |

Predefine primary metrics, budgets and a small set of ablations before larger runs. Use multiple seeds for deciding claims, confidence intervals where appropriate, and held-out tests. A short pilot selects experiments; it does not establish a universal mechanism.

Do not require a monotonically improving architecture ladder. A useful outcome may be a cheaper topology, a task-dependent map, or a clean negative result establishing that a proposed read/write edge adds cost without benefit.

## 14. Deferred options and original motivation

Keep these possibilities documented without making them prerequisites:

- **MultiDepthWriter:** take several ordinary preview depths, adapt each into the writer's residual space, process candidates as `[K,B,T,D]` or per-step `[K*B,1,D]`, then normalize/aggregate the candidate outputs **before** producing one shared permanent K/V pair. Use a mean or convex weights initially, not an unscaled sum. Depth sets such as `[3,8]` or `[3,5,6,8]` are options; all intermediate depths are not compulsory.
- This batching idea motivated the project: it may use the writer MLP more effectively at small device microbatches while adding useful depth information. It performs extra FLOPs, so speed or quality benefits must be measured. It remains an open experiment, not a dismissed possibility or a central initial requirement.
- **DQ:** multiple queries read one shared history, with one combined write. Avoid physically duplicating history simply to batch queries.
- **DS:** independent recurrent depth streams flattened into batch, each with its own memory. The upstream tiled block may make this easy to test, but it changes the architecture and cache budget; it does not approximate shared-write recurrence merely by reshaping.
- **Replay:** reexecute part of the middle corridor after the recurrent result. Only test if a late bridge appears insufficient.
- **Full-Bandwidth-style feedback:** top-state-to-next-input or mid-depth-to-next-input feedback as separate later integrations. The current plan contains neither. Such feedback changes prefill dependencies, checkpoint semantics and deployment state; a one-token shift in a training tensor is not by itself proof of matching autoregressive execution.
- RoPE, GQA, incremental decoding, production cache management, long-context kernels and joint/interleaved scans are outside initial scope. Schedule C remains excluded unless the research plan is explicitly revisited.

## 15. First-session instructions to the coding agent

Read this brief and local project instructions; inspect the actual repository before editing. Work autonomously on ordinary engineering choices, preserve user changes, and maintain a short semantic decision log.

Complete **Phase 0 and the R3 correctness portion of Phase 1**:

1. Establish the pinned upstream baseline and environment.
2. Add mixed-block selection with exactly index 3 recurrent.
3. Implement exhaustive weight conversion, including learned norms.
4. Add the supported pre-norm write homotopy to the naïve path.
5. Test structure, conversion, outputs, gradients, current-versus-persistent K/V order and causality.
6. Add a small whole-model benchmark command; execute only within available/authorized hardware and budget.
7. Report changed files, commands, actual results, unsupported features, and a concrete next benchmark/patch.

Do not start by writing a custom kernel or an all-purpose topology framework. Do not begin CDRM, A/B or latent-loss implementation until their preceding contracts are tested. Leave minimal interfaces ready for preview capture and auxiliary states rather than speculative infrastructure.

Keep a machine-readable experiment manifest with: topology ID, index map, input/output state definitions, parameter ownership, cache sources, gradient stops, masks, backend, gates, checkpoint provenance, optimizer policy, data/tokenizer, auxiliary source/target and budget.

Nonblocking decisions to resolve when they become relevant: actual downloadable checkpoint and dataset, continuation budget, selected synthetic task generators, final adapter dimensions, and which late refinement/replacement comparison to run first. The defaults above are sufficient for the first coding session.

The deliverable is a trustworthy experimental platform in which a named topology means one precise computation and each result has an interpretable control.
