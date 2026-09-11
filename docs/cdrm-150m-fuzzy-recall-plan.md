# Proposed 150M-family CDRM versus Transformer fuzzy-recall experiment

Status: plan only, 2026-09-08. No implementation, dataset generation or training
is authorized by this document alone. Model counts were checked using formulas
and CPU/meta construction; no GPU experiment was run for this plan.

The primary comparison is our tiled CDRM against the paper's ordinary,
non-recurrent Transformer backbone, trained from scratch on native MAD symbols.
Use a small MLP-width adjustment to match their parameter counts. Do not start
from C4-pretrained checkpoints or carry over the selective-copying pilot weights.

Figure 6 is a sequence-length sweep for a one-layer, D128/H16/MLP512 synthetic
model, not the paper's 150M language model. The proposed experiment combines the
150M architecture family with that diagnostic task. It is a new scale comparison,
not a reproduction of the published curve. The PDF and its SVG indicate fuzzy
points at 64, 128, 256 and 512; use 64/128/256 within the requested range and add
300 as an explicit endpoint. Exact published training seeds, duration and
checkpoint selection are not established by the available synthetic description.
See [paper D.2, E.1 and E.3](https://arxiv.org/html/2604.21215v1#A4.SS2) and
[Figure 6](https://arxiv.org/html/2604.21215v1#A5.F6).

1. **Freeze the architecture and parameter comparison.**

   Use 12 ordinary blocks, width 1024, 16 full-MHA heads of dimension 64, GELU,
   pre-norm learned LayerNorm and learned Q/K normalization, ALiBi maximum 8,
   zero dropout, bias-free projections/norms, and untied 16-symbol input/output
   tables. Retain the released 150M configuration's normalization and
   initialization semantics; record all resolved overrides to its C4 configuration.

   | Arm | Ordinary MLP width | CDRM | Actual unique parameters |
   | --- | ---: | --- | ---: |
   | Primary CDRM | 4096 | One shared side fabric | 153,175,040 |
   | Primary matched Transformer | 4192 | Disabled | 153,437,184 |
   | Optional unchanged Transformer control | 4096 | Disabled | 151,077,888 |

   The matched Transformer is 0.171% larger than CDRM. Its MLP width increases by
   96, keeping depth, attention width, head geometry and vocabulary unchanged.
   This is a simple uniform configuration change. The unchanged control is useful
   at T256 for checking the effect of the small capacity adjustment; it need not
   become a third complete length/seed sweep initially.

   With L blocks, width D, MLP width F and vocabulary V, these configurations have
   `P_SEQ = L*(4*D^2 + 2*D*F + 4*D) + D + 2*V*D`, and
   `P_CDRM = P_SEQ + 2*D^2`. Count the shared fabric weights once. The two
   separately owned D-to-D adapters add 2,097,152 parameters at D1024.
   Report total, backbone-only and adapter parameters explicitly: the original
   T5-vocabulary standard model has 216,843,264 total parameters, despite its
   nominal 150M label. A helper that excludes only the input embedding is not a
   backbone-only count because the head is untied.

   Restore CDRM's original zero-based sites 3 and 8: ordinary blocks 0–8 run once;
   the side fabric shares block 3; its deep candidate uses the block-8 versus
   block-3 preview difference; its read-conditioned write supplies a correction
   after block 8; ordinary blocks 9–11 consume the bridged state. Keep rho=1,
   epsilon=0.1 and lambda=0.01 fixed, with the existing nonzero adapter
   initialization. There is no separate recurrent replacement of an ordinary
   block, no learned gate, and no additional latent objective. The five-block
   numerical pilot was a minimal test profile, not the target size topology.

2. **Implement and independently audit native fuzzy-recall data.**

   Use pinned MAD revision `0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`, native
   V16, multi-query mode, maximum key/value motif lengths 3/3, and separate
   training, development and final seeds. The
   [task YAML](../vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml)
   establishes V16/T128/12,800 training examples; 256 is an established YAML
   length change. Keep every setting other than sequence length fixed.

   Fuzzy recall uses variable-length symbolic keys and values. Training keys
   have lengths 1–3; held-out keys have length 3. Values have lengths 1–3 in
   both. Keys occupy IDs 0–6, values 7–14, and padding is ID 15. Preserve this
   distribution difference rather than accidentally making evaluation easier.
   The native generator returns actual input length T after its own shift.

   Preserve native dense next-token training targets, including native padding
   targets, and held-out masks for repeated-key values. Do not shift labels
   twice, score padding as retrieval success, or silently replace native loss
   with answer-only training. Keep native padding-token treatment explicit;
   adding a new attention padding mask would be another semantic change.

   Extend `cdrm/mad_data.py` with a variable-length key/value parser and causal
   answer oracle. Its current two-task configuration and simple-recall parser
   are insufficient. In particular, replaying fuzzy generation with a different
   `is_training` flag changes the sampled inputs, so it cannot recover masks for
   an identical training example. Verify native arrays and labels, motif
   boundaries, first/repeated keys, final probes, padding, partial final values,
   causal indexing and split overlap. Retain native samples without silently
   rejecting or resampling inconvenient examples.

   Start with 12,800 training, 1,280 development and 1,280 untouched final
   examples per length and replicate. Pair the same datasets and shuffles
   across architectures. Use three paired model/data replicates in the completed
   comparison; keep development-calibration seeds separate from final replicates.
   Final data is generated and frozen after protocol selection. Report the
   number of scored tokens and sequences, not just nominal dataset size.

3. **Integrate the precision placement and validate the larger shape.**

   Introduce an explicit model option for FP32 ordinary attention, covering all
   12 blocks, with BF16 ordinary MLPs/head and tiled fabric. Parameters,
   residual/recurrent state, established normalization arithmetic, loss and Adam
   retain their FP32 policy. Apply the same ordinary-attention policy to the
   standard baseline. The present numerical wrapper explicitly lists blocks
   0–4; copying it unchanged would leave seven blocks outside the intended policy.

   Validate the option against the archived five-block wrapper before testing
   the new shape. Then use the actual D1024/H16 models and fuzzy supervision:
   tiny/short naive-FP32 versus tiled-FP32 checks; tiled mixed versus tiled FP32
   at T128/T256/T300, initially small batches; and bounded full training-batch
   checks at the longest length. Include causality, adapter/shared-weight
   gradients, real-loss and independent unscaled fabric gradients, scalar
   cotangent scaling, and same-state optimizer updates. Cover initialization
   and an early-trained development checkpoint. Test T300 explicitly because
   it exercises a non-power-of-two tiled boundary.

   Carry the existing numerical criteria and qualifications forward. In
   particular, a passing initial Adam cosine does not establish close first-step
   relative L2 agreement. Keep the first-step distance and sign-sensitive
   coordinates visible. A smaller batch alone does not clear the actual training
   batch. If the selected mixed policy fails at this scale, localize that result
   before a broad training sweep; an FP32 run is a separately recorded fallback.

   Current tiled support excludes owner activation checkpointing, dropout and
   FlashAttention. Do not silently enable these to make the experiment fit.

4. **Build a new resumable runner and measure the H100 budget.**

   Reuse validated aligned losses, metrics, optimizer/state accounting, W&B and
   checkpoint machinery, but add a new task/size-aware runner and lineage. The
   old tiled pilot deliberately freezes five blocks, selective copying, B64 and
   a 2,500-update cap; preserve its identities and guards.

   Enter the project's Docker launcher, verify container working directory and
   `nvidia-smi`, then benchmark real forward/backward/Adam updates after compilation
   and warmup. Try physical B32, B64 and B128 at T300. Target B128 to match MAD's
   batch default; if it does not fit both arms, freeze the largest common tested
   batch for every length. No accumulation is required for the first protocol;
   introducing it would require separate loss-normalization and update-parity
   validation. Record peak allocated/reserved memory, steady-state seconds per
   update and evaluation cost. Full-width fit and speed are unmeasured today.

   Verify save/load and a bounded interrupted/resumed run including optimizer,
   scheduler, RNG, data order and cursor. Each independent sequence rebuilds
   recurrent memory. Keep all parameters canonically owned once.

   Report the measured total budget before launching the complete sweep. For
   two architectures, four lengths, three seeds and E epochs with N=12,800,
   training time is approximately
   `3 * sum_over_lengths((N/B)*E*(seconds_SEQ[T] + seconds_CDRM[T]))`, plus
   calibration, compilation, evaluation and checkpoint overhead. At B128 and
   E=25, this is 24 runs and 60,000 updates in total; E=50 doubles it.
   Parameter matching does not match repeated computation or wall time.

5. **Calibrate at T256, then freeze a bounded comparison.**

   Start from fresh task-native initialization. Use AdamW betas (0.9,0.98),
   epsilon 1e-8, weight decay 0 and clipping norm 1. Consider the paper's three
   distinct learning rates 1e-4, 5e-4 and 1e-3 equally for both architectures,
   using one separate calibration replicate and a maximum of 10 epochs each.
   The duplicated 5e-4 in the paper is not a fourth distinct candidate. This
   fixes weight decay rather than reproducing its entire tuning grid; record
   that reduction. Preserve native supervision during this calibration.

   Use one common explicit cosine horizon of 50 epochs, minimum LR 1e-6 and no
   warmup, evaluated after each completed epoch. This is a bounded experimental
   choice, not the undocumented exact schedule behind Figure 6. Retain the same
   schedule during calibration and the main comparison. Select one LR per
   architecture using development token accuracy, with development answer CE as
   a predeclared tie-breaker; freeze those choices across all four lengths.
   Record all calibration outcomes and equal tuning expenditure.

   The proposed first main endpoint is 25 epochs, with a common 50-epoch endpoint
   if calibration indicates that the shorter budget is clearly premature and
   the measured compute budget supports it. Freeze the endpoint for all lengths
   and both architectures before final-replicate training. At B128, these are
   2,500 or 5,000 updates per run. Neither endpoint is a claim of reproducing MAD's
   200-epoch maximum. Do not change the selected endpoint using final-test scores.

   For the resized baseline, identical full-backbone initialization is
   impossible because MLP tensor shapes differ. Use paired seeds, identical
   data/order, copy shape-identical embeddings/attention/norms, and initialize
   resized MLPs with their declared native variance rules. Record precisely what
   is shared. The optional unchanged-MLP baseline can use an exact CDRM backbone
   copy and gives a cleaner capacity-addition control at T256.

   If calibration immediately saturates, retain early learning curves and report
   the task ceiling; do not infer an architecture advantage from a negligible
   endpoint gap. If both fail to learn, check supervision, tiny-batch fitting
   and optimization before the full sweep. Any extension of the tuning or
   training budget is symmetric and uses development data only.

6. **Train separately at each length and report the requested curve.**

   The main matrix is two architectures × T{64,128,256,300} × three paired
   replicates. Each length receives its own fresh model trained at that length.
   Training once at T128 and evaluating at other lengths would be a separate
   length-generalization experiment. Equal examples/epochs at different lengths
   do not mean equal tokens across lengths; within each length, data exposure
   and update budgets are matched between architectures.

   Primary output: final held-out token accuracy versus actual sequence length,
   with individual seeds, mean/spread and paired differences. Use only native
   answer positions. Secondary outputs: exact match over all scored tokens in
   each sequence, loss, time/updates/tokens to fixed development accuracy levels,
   and development learning curves against both tokens and elapsed time.
   Bootstrap whole sequences for within-seed sampling intervals, and distinguish
   those from variation across training seeds. Multi-token answers are teacher
   forced; masked exact match is not free-running answer generation.

   Include a uniform eight-value predictor (12.5% expected token accuracy),
   simple frequency/answer-prefix shortcuts and the exact causal retrieval
   oracle. A learner exceeding uniform chance may still use a shortcut.
   Keep dense training metrics separate from held-out retrieval metrics.
   Do not overlay the paper's one-layer results as a parameter-matched baseline.

   Log online under `taylorbollman/cdrm-150m-fuzzy-recall`, with groups identifying
   the frozen protocol, length and replicate. Retain configurations, dataset
   manifests, initialization, resumable checkpoints, numerical probes and final
   reports under a fresh `gs://fast-chunks/cdrm-w-latent/` prefix with checksum
   verification. Preserve the prior numerical lineage unchanged.

The first reviewable milestone is the new task/runner and 12-block precision
option, validated at full width, plus T256 calibration and measured sweep cost.
The next milestone produces the frozen four-length comparison. Improvements
over the matched Transformer would support this CDRM configuration on this task;
isolating the deep-source contribution from extra computation would still call
for the existing same-depth or another computation control in a later study.
