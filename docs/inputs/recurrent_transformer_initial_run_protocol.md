# Provisional initial run protocol: SEQ, R3, and reusable research checkpoints

Date: 2026-09-06. Audience: Taylor and the coding AI working in the project repository.

This is an operational supplement to `recurrent_transformer_coding_agent_brief.md`. Preserve that brief's architecture definitions and correctness requirements. The budgets and schedules below are proposed defaults for our experiments, not prescriptions from the Recurrent Transformer paper. Creating this protocol does not mean that any training or evaluation has run, or that any named checkpoint already exists. Record actual status separately as planned, running, completed, or verified.

## 1. Objectives and scope

Establish a trustworthy 12-block standard causal Transformer (SEQ), replace only block index 3 with temporal recurrence (R3), and ask whether continued training improves learning per token and per unit of training time. Preserve a common standard checkpoint and evaluation fixtures so later DM, DW0, CDRM, and latent-objective experiments can make comparable measurements.

R3 executes ordinary blocks 0–2, scans block 3 across positions, and executes ordinary blocks 4–11. The scan reads earlier permanent K/V and the current temporary K/V; only after producing the current output does it append that position's permanent record. The initial supported interpolation is:

\[
w_t=(1-\rho)x_t+\rho z_t.
\]

Project persistent K/V from `w_t` using the validated pre-norm contract; pass `z_t` upward. `rho=0` must reproduce the compatible SEQ block, and `rho=1` is full recurrence. The interpolation is our conversion experiment, not an upstream feature to assume exists.

Upstream foundation: [Oncescu et al., The Recurrent Transformer](https://arxiv.org/abs/2604.21215), [official implementation](https://github.com/geniucos/recurrent-transformer). Begin from pinned revision `a21b42d2bc292edb86ed1b62cee4bcab809a9d21`, recording our fork revision separately.

## 2. Keep four categories of evidence separate

| Label | Question | Examples | What passing establishes |
|---|---|---|---|
| NUM | Does the code compute the intended function and gradients? | Conversion equality, causality, reference/tiled agreement, save/resume | Implementation correctness within tested settings |
| OPS | Can this configuration run reliably at an affordable cost? | Tiny overfit, full training-update benchmark, memory measurement | Operational readiness; not a generalization result |
| SYN | What computations does the architecture learn? | MQAR, noisy recall, state updates on unseen sequences | Mechanistic evidence under a specified synthetic training regime |
| LM / DOWN | Does the trained language model improve? | Held-out C4 CE; PIQA/HellaSwag scoring | Language-model or downstream evidence |

Generator oracle checks are NUM; learned-model synthetic accuracy is SYN. Loss reduction on a repeatedly reused tiny batch is OPS; it is not held-out LM performance. A numerical failure blocks results from the affected path. A scientifically valid negative SYN or LM result does not imply a numerical failure.

## 3. Initial language-model contract

| Item | Initial choice |
|---|---|
| Backbone | 12 blocks, indices 0–11 |
| Dimensions | Width 1024; 16 heads; head width 64; MLP width 4096 |
| Parameter label | Approximately 150M non-embedding parameters; report actual total and trainable counts |
| Architecture | Pre-norm, ALiBi, full MHA; preserve upstream activation and normalization details |
| Initially excluded | RoPE, GQA, document-isolated packed attention, cached decoding, latent losses, cross-depth feedback |
| Sequence length | 512 for substantive LM runs |
| Global batch | 512 sequences per optimizer update |
| Device microbatch | Measured on one H100 first; use accumulation to reach global batch |
| Precision | FP32 numerical fixtures; BF16 training after validation |
| Stochasticity | Dropout off initially |
| Backend | Eager SEQ and naïve R3 first; validated tiled R3 at rho=1 may follow |
| Data/tokenizer | Verified C4 training/validation pipeline with the repository-compatible T5 tokenizer and IDs |

Freeze tokenization, vocabulary, EOS/BOS treatment, chunking, and whether ordinary fixed-length corpus chunks cross document boundaries. Do not accidentally introduce a different masking regime in one architecture. Keep validation separate from training. Record data revisions, file hashes or stable manifests, and sampling order. Do not assume the repository's original cluster paths are available.

The released [150M sweep](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/sweep_olmo_150m_512.yaml) specifies a 12,500-update training horizon. At the batch and length above:

\[
\text{input tokens/update}=512\times512=262,144.
\]

| Updates | Input tokens |
|---:|---:|
| 200 | 52,428,800 |
| 2,500 | 655,360,000 |
| 5,000 | 1,310,720,000 |
| 12,500 | 3,276,800,000 |

Also log actual supervised target-token counts: padding and the next-token shift can make these smaller. All update counts here refer to optimizer updates, not microbatches. With one GPU and microbatch 16, accumulation 32 gives global batch 512. Accumulation does not enlarge the writer's simultaneous per-token batch.

## 4. Stage A — numerical and operational foundation

Do this before substantial pretraining. It corresponds to the correctness part of implementation Phases 0–1.

1. Complete the environment/provenance audit, explicit CPU reference path, mixed-block construction, and exhaustive checkpoint conversion.
2. Repair/audit masking at every ordinary-block entrypoint: normal forward, direct block calls, benchmarks, and any graph wrappers. Raw ALiBi is not a causal mask. Keep CUDA graphs disabled until their actual paths pass causality checks. See the pinned [capture helpers](https://github.com/geniucos/recurrent-transformer/blob/a21b42d2bc292edb86ed1b62cee4bcab809a9d21/olmo/efficient_utils.py).
3. Run tiny deterministic NUM fixtures: exactly block 3 replaced; copied nondefault norms and biases; rho=0 logits/loss/input-gradient/mapped-parameter-gradient equality; rho=1 reference agreement; no current-write self-read; no future leakage; later loss reaches earlier writes; unique optimizer ownership. At sequence length one, compare outputs, since an internally shifted LM loss has no valid target pair.
4. Validate checkpoint round trips and an interrupted/resumed tiny training trajectory. Validate accumulation against an equivalent full-batch update, using correct target-count weighting. A fresh-optimizer experimental fork is a different operation from exact resume.
5. Run a tiny fixed-data overfit test for SEQ and R3. Its purpose is learnability and plumbing, not research evidence.
6. At the substantive model dimensions and length 512, benchmark real next-token CE, backward, and optimizer updates. Warm up sufficiently to allocate optimizer state. Record steady-state update time, forward time, peak allocated/reserved memory, throughput, precision, and backend. Start on one GPU; increase hardware only after measuring fit and throughput.

Retain `NUM-fixtures-v1`, the conversion audit, hardware report, and `OPS-smoke-SEQ` / `OPS-smoke-R3` checkpoints as debugging artifacts. Explicitly label them **not research pretrained checkpoints**. Do not initialize the main LM baseline from weights trained on arbitrary smoke-test batches.

Gate: all relevant NUM tests pass; OPS runs are stable; a bounded projected GPU-hour budget for subsequent stages is recorded. Do not fabricate a speed estimate from the fraction of recurrent layers.

## 5. Stage B — small synthetic mechanism pilot

This can run once Stage A's small-model implementation is trusted, before paying for the 5,000-update LM baseline. Do not wait for an expensive pretrained checkpoint to debug synthetic generators.

### 5.1 Initial tasks, in priority order

| Task | Source / status | Main question | Initial measurements |
|---|---|---|---|
| Multi-query associative recall (MQAR) | Adopt a pinned [Zoology generator](https://github.com/HazyResearch/zoology); [paper](https://arxiv.org/abs/2312.04927) | Does recurrence preserve or improve access to stored associations across delays and competing records? | Accuracy and CE at designated answer positions; breakdown by delay and number of associations |
| Noisy in-context recall | Adopt a pinned [MAD generator](https://github.com/athms/mad-lab); [paper](https://arxiv.org/abs/2403.17844) | Can useful associations survive irrelevant intervening tokens? | Answer accuracy/CE as distraction and retrieval distance vary |
| Ordered state updates | Our custom diagnostic, not an official MAD/Zoology task | Can the model accumulate ordered transformations of an entity's state? | Final-state accuracy/CE by update count, intervening entities, and delay |

MQAR is a retrieval control, not a direct proof of read-conditioned writing. MAD noisy recall adds a distractor axis. The custom task supplies the stronger temporal-computation test that plain recall does not provide. Do not average these into an allegedly standard MAD or Zoology score.

For the custom task, initialize entity states in a small finite set and emit interleaved entity-specific operations drawn from fixed noncommuting permutations, then query an entity's final state. Learn individual operation meanings during training and hold out operation compositions. Start with one terminal query and no intermediate answer tokens: teacher-forced intermediate answers must not reveal a state that shortcuts later computation. Last-assignment retrieval alone should not solve the task. Preserve a simple exact interpreter as the oracle.

After CDRM exists, extend this diagnostic to retrieve–update–retrieve: an operation must retrieve an earlier source entity's state, transform it, store it for a destination entity, and later answer a query about that destination. Make every dependency backward in time and label this an additional custom task.

### 5.2 Model and training defaults

Use a **12-block small diagnostic model**, width 256, four heads, MLP width 1024, ALiBi, and task-native symbolic embeddings/output vocabulary. This retains the recurrence location while reducing cost. It is not the 150M LM checkpoint, and its vocabulary/head cannot be silently substituted into that checkpoint.

For each task, train paired SEQ and R3 from corresponding random weights, with rho=1 for R3. This tests learning an architecture from scratch, separately from later LM conversion/ramping. No latent objective initially.

Proposed pilot: one training seed, 2,000 updates, global batch 64, training length 128. Use the same generated training examples for each pair. Begin with AdamW at 1e-3, betas (0.9, 0.95), epsilon 1e-8, zero weight decay, gradient clipping 1.0, and a 100-update warmup. Specify a common cosine schedule with a final LR multiplier of 0.1. These are pilot defaults, not source-paper hyperparameters. If necessary, allow one equal-budget 3e-4 alternative for both architectures; retain both outcomes.

Use task-native label placement and loss masking. Inspect the imported generator/training interface: some return already-aligned next-token labels. Do not shift them a second time. Train and score answer positions; prediction of random context tokens should not dominate the task objective.

Initial difficulty: approximately 8–16 associations for MQAR where the generator permits; low and moderate distractor settings for noisy recall; 2–4 entities and 4–8 relevant updates for state tracking. Calibrate feasibility on development data and freeze an explicit configuration before the result run. Do not silently overwrite the configuration if both models hit floor or ceiling.

### 5.3 Evaluation and interpretation

- Fixed development set: 1,024 examples per selected condition; evaluate every 200 updates. Separate final test set: 4,096 examples per condition, used after configuration selection.
- First test unseen sequences and mappings at length 128. Then test 256 and 512 after backend/position-length validation. Vary delay at fixed association/update count separately from varying the number of associations/updates. Record changed variables.
- For custom state tracking, test unseen operation compositions and entity-role assignments. Holding out entirely untrained symbol embeddings is a different experiment and is not the initial generalization claim.
- Validate oracle answers, legal masks, train/test separation, and counterfactual sensitivity to relevant earlier operations. Include chance and a restricted-history or last-record heuristic to expose shortcuts.
- Report answer-position CE/accuracy and, when relevant, all-answers-correct sequence accuracy. A low average loss over mostly unscored context is not a success metric.
- A promising or ambiguous pilot earns three paired training seeds total. All-seed endpoints and dispersion are primary; best-development checkpoints are secondary. Pairing applies to initialization and data, not an assumption that the functions stay equal at rho=1.

Retain per task/topology/seed: initialization, final checkpoint, best-development checkpoint, complete configuration, generator revision, generated held-out fixtures, per-example predictions, and learning curves. Naming: `SYN-{task}-{SEQ|R3}-seed{s}-{init|final|bestdev}`. Save the exact update count and metrics in the manifest.

If both architectures fail, check task learnability and bounded optimization alternatives before interpreting a negative result. If both saturate, increase one difficulty axis. A small-model win motivates testing at full width; it does not establish a 150M-model or language-model gain. Run a representative full-width confirmation only after a useful diagnostic condition is identified.

## 6. Stage C — build and preserve the standard LM parent

Use an existing verified compatible intermediate SEQ checkpoint if one becomes available; record its actual provenance and counters rather than relabeling it as our run. Otherwise train a fresh SEQ baseline as follows.

1. Freeze the data stream, evaluation sets, optimizer recipe, and initialization seed. Save `LM-SEQ-s0-init` before the first update.
2. Provisional baseline recipe: AdamW, peak LR 1e-3, betas (0.9, 0.95), epsilon 1e-8, zero weight decay, gradient clipping 1.0. Use a 12,500-update cosine-with-warmup horizon, 5,000 warmup updates, and initial/final LR multipliers of 0.1. Verify how the repository scheduler realizes those values. This chooses one member of the released sweep rather than claiming to reproduce its best run.
3. Run the first 200 updates as an operational check using that same planned schedule. If stable, continue the same run to 5,000 updates. Do not shrink the scheduler horizon to the stop point.
4. Permanently protect `LM-SEQ-s0-u1000`, `LM-SEQ-s0-u2500`, and especially `LM-SEQ-s0-u5000`. The latter becomes the common parent for the first comparison and later topology branches.
5. Evaluate held-out C4 at initialization and each specified interval, and perform the larger endpoint evaluation plus downstream evaluation at u5000. Save evaluation outputs beside the checkpoint identity.

The standard run need not reach u12500 before the first R3 experiment. If later resumed without optimizer reset, preserve u7500, u10000, and u12500. Label it `SEQ-uninterrupted`; it is useful context but does not replace the reset-matched control below.

## 7. Stage D — paired SEQ and R3 continuation

Fork two runs from the exact `LM-SEQ-s0-u5000` weights and next-data position:

```text
LM-SEQ-s0-u5000 [immutable parent]
  ├─ SEQ-cont: ordinary architecture; fresh optimizer + continuation schedule
  └─ R3-cont:  convert block 3; fresh optimizer + same schedule; ramp rho
```

Conversion is not training. First save `LM-R3-s0-parent5000-c0000-rho0` and verify NUM equality to the parent using trained, nondefault weights. Save a conversion coverage report. A rho=1 evaluation without training is an optional labeled intervention; it is not a trained recurrence result and must not mutate the rho=0 starting checkpoint.

### Matched continuation defaults

| Item | Choice |
|---|---|
| Budget | 2,500 additional optimizer updates per branch |
| Operational pause | Check at continuation update 200; resume the same planned run if stable |
| Optimizer | Fresh AdamW in both branches; same betas, epsilon, weight decay, clipping as above |
| LR | Provisional peak 1e-4; 100-update re-warm; 2,500-update cosine horizon; initial/final multipliers 0.1 |
| Recurrence ramp | R3 rho(c) = min(c/250, 1), with c completed continuation updates before the next update |
| Training data | Identical next examples in identical global batches; preserve the parent's data offset |
| Trainable weights | All model weights; no freezing initially |
| Backend | Naïve R3 initially; any validated tiled switch at rho=1 recorded as a backend change |

The ramp is over training updates, not a growing selection of token positions. Each sequence uses one recorded rho setting. Reset temporal K/V between independent training examples; permanent within-sequence memory is not a state to carry between unrelated minibatches.

Keep separate counters for parent updates/tokens, continuation updates/tokens, and lifetime tokens. A weights-only load must not silently restart the corpus, and restoring the corpus position must not inadvertently restore the old LR schedule. Validate this separation in Stage A.

Evaluate/log at c0000 and retain protected checkpoints at c0200, c0250 (ramp completed), c0500, c1000, and c2500 for **both** branches. Store the gate state so c0250 resumes at rho=1. Also retain each branch's best-development checkpoint, separately labeled. Example: `LM-R3-s0-parent5000-c2500`; the endpoint has 7,500 lifetime updates but belongs to a distinct lineage from uninterrupted SEQ-u7500.

Primary research endpoint: equal additional tokens at c2500, not whichever checkpoint happened to have the lowest noisy validation loss. Compute:

\[
\Delta L_{\mathrm{C4}}=L_{\mathrm{R3}}-L_{\mathrm{SEQ-cont}}.
\]

Negative favors R3. Show the full transition curve, including ramp and recovery, along with loss versus training GPU-hours. A secondary equal-time comparison uses a prespecified common training-time budget and saved intermediate evaluations; it need not consume equal tokens. Report whether evaluation/checkpoint I/O is included, and report total job cost separately. Do not extrapolate sparse curves into a claimed measured win.

One parent and one pair is exploratory. If useful, run two more paired continuation seeds from the same parent; these measure variation conditional on that parent. Stronger replication later includes independently pretrained parents. Shared-parent repeats cannot quantify pretraining-seed uncertainty.

## 8. C4 logging and evaluation contract

Training CE is diagnostic; held-out CE is the primary LM metric. Aggregate summed negative log likelihood divided by valid target count, not an unweighted average of unequal batches. Keep the standard causal next-token shift and identical token masking in every topology.

| Record | Default cadence |
|---|---|
| Training CE, LR, gradient norm, rho, update/token counters | Every 20 optimizer updates |
| Throughput and memory | Regular samples plus dedicated clean benchmark |
| C4 development CE | Initially, every 250 updates, and all protected transition checkpoints |
| Larger fixed C4 evaluation | Parent u5000 and both c2500 endpoints |
| Downstream scoring | Parent u5000 and both c2500 endpoints |

Proposed fixed C4 development set: 1,024 sequences of length 512. Proposed disjoint larger evaluation set: 8,192 sequences. Freeze example IDs and report actual valid-token counts. The larger set is a confirmatory evaluation for this pilot, not a tuning set; if repeatedly used to guide later research, acknowledge its development role and reserve a fresh final set.

Use one W&B run per lineage, grouped by protocol and parent identity. Capture ordinary learning curves rather than requiring an expensive model-graph trace. Persist resolved configs and local machine-readable metrics as well as W&B IDs. Match global batches even when device microbatches differ.

## 9. Downstream checks: PIQA first, HellaSwag second

These are frozen-checkpoint evaluations: no downstream fine-tuning in the initial protocol. Start with zero-shot scoring using pinned [PIQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml) and [HellaSwag](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml) task definitions from the EleutherAI evaluation harness, or an exactly documented adapter reproducing their prompt/scoring semantics.

- **PIQA:** primary downstream task; score the full labeled validation split.
- **HellaSwag:** optional second task; initially use a fixed seeded 1,000-example validation subset. Expand to the full labeled validation split for a confirmatory comparison if cost permits. Name the subset explicitly.

For each, report standard accuracy and the task's configured length-normalized accuracy where provided. Also report token-weighted NLL of the gold continuation and per-example scores as smoother diagnostics. Gold-continuation NLL is our additional metric, not automatically the benchmark's standard score. Accuracy may be weak at this model scale; do not substitute a near-chance downstream result for the more sensitive C4 and mechanism measurements.

Implement scoring through full causal forward passes; cached decoding is not required. Score only continuation tokens, initialize independent recurrent memory for every candidate, and test token-boundary alignment against manually computed fixtures. Preserve tokenizer BOS/EOS conventions. Do not leak gold answer identities into prompts.

If prompt plus continuation exceeds 512, use one documented, identical truncation policy across models, preserving the full candidate continuation where possible. If a continuation alone cannot fit, exclude the whole example for every model and report counts. Label truncation/subsetting adaptations; they limit direct comparison with published full-context benchmark numbers.

Preserve per-example likelihoods and choices to support paired comparisons and bootstrap uncertainty. Resample whole examples, and for C4 use document clusters when available. Evaluation-item uncertainty does not replace training-seed replication. Do not tune the architecture or LR on these downstream results in the first pilot.

## 10. Later checkpoint-based synthetic transfer and CDRM readiness

Stage B's symbolic-model training and evaluation of the pretrained LM are different studies. Do not feed arbitrary task-native symbol IDs into the T5 LM and interpret failure as lost memory capacity.

After the first LM pair, optionally select the most informative synthetic task and define a stable text serialization using the existing LM tokenizer. First audit lengths, answer boundaries, and chance/shortcut baselines. Clone the parent and matched SEQ/R3 endpoints into dedicated task-adaptation branches; use identical examples, update budgets, and answer-only loss. A bounded first transfer run is 500 updates at global batch 64, peak LR 1e-4, 50-update warmup, final multiplier 0.1. Validate fit in 512 tokens, and log actual tokens because text lengths vary. Treat this as supervised adaptation, with separate names and checkpoint lineage. Never overwrite the LM checkpoints or use task-adapted weights in the frozen downstream comparison.

When implementation Phases 2–3 are ready, first verify that ordinary preview through block 8 plus zero bridge matches SEQ. Then run small SYN comparisons for DM, DW0, same-depth controls, and CDRM, including the custom retrieve–update–retrieve extension. DM tests extra pointwise/depth computation; DW0 has writes independent of retrieved history; CDRM has a read-to-write dependency. Preserve their distinct definitions from the main brief.

For their first LM comparisons, initialize from the protected **SEQ-u5000** parent with the same 2,500-update continuation budget and a reset-matched SEQ control. Reuse an earlier control only if its complete protocol, data sequence, evaluator, and relevant implementation semantics match; otherwise rerun it. CDRM is not automatically initialized from R3-cont because its preview is ordinary and its information flow differs.

A negative R3 result does not automatically cancel CDRM or the latent hypothesis. It changes the evidence and should motivate a bounded, explicitly justified follow-up. NL1 can branch after trusted R3/SEQ; NL2 after trusted CDRM. Keep latent-loss runs in separate lineages with objective-weight-zero controls. RoPE remains a later compatible-backend extension rather than part of this first comparison.

## 11. Protected checkpoint ledger

The coding AI must maintain this ledger with actual paths, hashes, creation status, verification status, parent identity, and associated results. The entries below are targets, not assertions of existing files.

| Stage | Protected artifacts | Why retain them |
|---|---|---|
| A | NUM fixture bundle; OPS smoke checkpoints | Reproduce bugs, conversions, and backend comparisons |
| B | SYN init/final/bestdev for each task, topology, seed | Repeat mechanism experiments and test new backends |
| C | LM-SEQ-s0-init, u1000, u2500, u5000 | Recreate training and study conversion at different maturities |
| D initialization | SEQ-cont-c0000; R3-c0000-rho0; conversion report | Exact common-parent comparison and conversion audit |
| D transition | Both branches c0200, c0250, c0500, c1000 | Diagnose disruption, ramp completion, recovery, and time tradeoffs |
| D endpoint | Both branches c2500; separate bestdev checkpoints | Primary comparison and downstream/transfer starting points |
| Optional baseline completion | Uninterrupted SEQ-u7500, u10000, u12500 | Standard learning trajectory and later conversion parents |
| Optional task adaptation | Separate SYN-transfer init/final/bestdev | Measure adaptation without altering original LM results |
| Later CDRM/control branches | Initialization, transition, endpoint for every selected topology | Comparable TopologyGrid experiments |

Protect all listed milestones from rolling-checkpoint deletion. Other restart checkpoints may use bounded retention. Each protected training checkpoint should have full resumable state; a separate portable weights export is useful where supported. Preserve model config, canonical parameter ownership, optimizer/scheduler state, gate schedules, RNG state, dataset offset/sampler state, precision/backend, both repository revisions, and evaluated metrics. Save weights-only conversions with an explicit fresh-optimizer policy.

Never describe a checkpoint as verified until loading it and the relevant round-trip/resume checks have succeeded. Do not delete a milestone just because a later checkpoint has lower loss.

## 12. Initial execution order and decisions

1. Finish Stage A NUM/OPS and the checkpoint/metrics infrastructure.
2. Run Stage B's small SYN pilot and resolve invalid generators or uninformative difficulty settings. Scientific improvement is not required to proceed to a distinct LM hypothesis.
3. Record measured cost and the resolved experiment budget, then run Stage C to the protected SEQ-u5000 parent.
4. Run Stage D's paired 200-update operational check; extend the same runs to c2500 if numerically valid and operationally viable.
5. Produce a paired report: C4 curves and endpoints; GPU-hours and memory; synthetic capability curves; PIQA and optional HellaSwag; checkpoint ledger. Keep NUM/OPS outcomes in a separate part of the report.
6. Decide on replication, full-width synthetic confirmation, later conversion points, and the Phase 2–3 controls/CDRM. Do not launch the entire topology grid as an automatic sweep.

Before launching, resolve the actual data/checkpoint availability, GPU allocation, measured cost, and experiment budget with the project's existing authorization. If one is unavailable, complete the useful implementation and report exactly what is pending. An unavailable dataset is not a negative model result.

The useful outcomes include: better C4 loss at equal tokens; better quality at equal training time; improved ordered-state computation even if recall is unchanged; preserved recall with cheaper or more useful later topology options; or a well-controlled null result showing that a specific dependency adds no value in the tested regime. Claims should match the training regime and scale actually tested.
