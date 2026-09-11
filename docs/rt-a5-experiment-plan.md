# Plan: two-layer Transformer versus two-layer RT on A5

Prepared 2026-09-11. This is an implementation and experiment plan; no model,
dataset, training job, or GPU validation was created or launched for it.

Implementation follow-up: the plan below was subsequently approved and its
first paired 10,000-update pilot completed. See
[the results](reports/rt-a5/pilot/report.md) and [handoff](rt-a5-handoff.md).
The remaining longer-run stages are still prospective.

Revised after the user's request to minimize numerical-validation work:
**use FP32 throughout both models, disable compilation, and start without
CUDA graphs**. Keep a few bounded correctness checks for the new task, then
proceed to the learning pilot. Mixed precision and execution optimization are
optional responses to a measured runtime problem, not prerequisites.

The first comparison will use our existing standard Transformer components
and our existing tiled Recurrent Transformer, with **two blocks in each model
and both RT blocks recurrent**. The initial question is whether recurrence
helps learn and generalize permutation-state tracking. NextLat's auxiliary
objective and its separately rolled-out latent dynamics model belong to a
later experiment; neither is part of these two arms. Our RT retains attention
memory and is not the compact-state RNN evaluated in NextLat's Figure 10.

**Reference experiment and source authority.**

The supplied [NextLat v4 PDF](../key_research_docs/2511.05963v4.pdf) is byte
identical to [arXiv v4](https://arxiv.org/pdf/2511.05963v4), SHA256
`7053798fe919915d0be1f03165f69f96b4e4076114077197c34d99efb767c33f`.
Section 5.2, Table 5 and Appendix F.5 provide the experimental reference.
The authors use two-layer, width-512, eight-head Transformers, train on
12-operation words, and evaluate out to length 36. Their positional encoding
is RoPE; ours will remain ALiBi in both arms. Its released GPT also uses
RMSNorm and SwiGLU, whereas our components use LayerNorm and GELU. This is an
architecture comparison inspired by that experiment, rather than an exact
reproduction of its GPT.

Two repositories resolve details that matter for implementation:

- [NextLat's A5 branch](https://github.com/JaydenTeoh/NextLat/tree/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9),
  commit `b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`, is the authority for the
  requested experiment's serialization and evaluation. It uses no BOS token
  and computes cumulative prefix exactness. One million length-12 examples
  are split 80/20, explaining the reported 512 nominal epochs at 400,000
  updates of batch 1,024. The released code repeatedly evaluates its held-out
  splits; our additional final confirmation set will remain separate.
- [word-problem](https://github.com/jopetty/word-problem/tree/8f910f92e1c70455dcd9376f56032dfc55126188),
  commit `8f910f92e1c70455dcd9376f56032dfc55126188`, supplies the mathematical
  generator contract. Its trainer adds BOS and can mix shorter words into
  training; we will not inherit those choices for the NextLat-style experiment.
  The [earlier paper, Section 6](https://arxiv.org/html/2404.08819v3#S6), studies
  depth at increasing training lengths, a different protocol from training
  once at 12 and extrapolating to 36.

**1. Freeze the task, label alignment and metrics first.**

Let \(G=A_5\), the 60 even permutations of five objects. Sample input operations
independently and uniformly from all 60 elements, including identity. For a
word \(g_1,\ldots,g_L\), define

\[
s_0=e,\qquad s_t=s_{t-1}\circ g_t,\qquad
(p\circ q)(i)=p(q(i)).
\]

The rightmost permutation acts first in this convention. IDs 0 through 59
enumerate even permutations of `(1,2,3,4,5)` in lexicographic order; identity
has ID 0. Each position receives the state label **after its input operation**:

```text
input:   g1  g2  g3  ... gL
target:  s1  s2  s3  ... sL
```

There is no BOS, padding, interleaved answer, target feedback or shifted
language-model loss. A length-12 word occupies exactly 12 model positions.
This is 60-class prefix tagging, rather than binary identity membership or
prediction of the next randomly sampled operation.

For logits \(a_{b,t}\in\mathbb R^{60}\), use

\[
\mathcal L=\frac{1}{BL}\sum_{b=1}^{B}\sum_{t=1}^{L}
\operatorname{CE}(a_{b,t},\operatorname{id}(s_{b,t})).
\]

Define \(c_{b,t}=\mathbf1[\arg\max a_{b,t}=\operatorname{id}(s_{b,t})]\).
The principal Figure-10-style curve is

\[
E(t)=\frac1N\sum_{b=1}^{N}\prod_{j=1}^{t}c_{b,j},\qquad t=1,\ldots,36.
\]

This is exact correctness of the entire prefix, not accuracy only at position
\(t\). Also report isolated state accuracy \(A(t)=N^{-1}\sum_b c_{b,t}\),
mean token accuracy, CE, and whole-word exact match. At a word's final
position, \(A(L)\) is final-state accuracy and \(E(L)\) is whole-word exact
match. Reporting both separates accumulated occasional errors from loss of
state-tracking ability. The first state equals the first input operation and
is especially easy; do not let it dominate interpretation. Uniform guessing
has 1/60 isolated-state accuracy, not a flat 1/60 cumulative-exactness curve.

The metric interpretation follows the
[released evaluator](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5_data.py#L104-L165).
No released A5 plot-building script, underlying figure CSV or checkpoints
were found; the exact published figure was not independently regenerated.

Implement a small native generator and multiplication table, rather than
installing the original SSM training stack. Preserve MIT attribution for any
adapted word-problem code. Save the ordered alphabet, table, generator version,
seeds, split indices, data hashes and complete resolved protocol.

A read-only source audit already independently reproduced all 3,600 pairs
and prefix labels in the pinned repository's `A5=2.csv`, with zero mismatches.
Its SHA256 is
`5fa1b247ecf078d027a71fff7c486d6e4f12df4d0f8ff4526bc91cf759bbf307`.
This is a useful external oracle; the new implementation will still need its
own tests. Include identity, inverses, noncommuting pairs, associativity,
prefix composition, deterministic generation and complete-word split
disjointness. Shared short prefixes across splits are expected in this task.

**2. Construct two genuinely matched models.**

The primary size will match NextLat's layer count, width and head count while
preserving our own block recipe. Smaller widths are subsequent capacity
ablations, not substitutes for the primary comparison.

| Configuration | Blocks | Width | Heads | GELU FFN width | Parameters per arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 2 | 512 | 8 | 2,048 | 6,357,504 |
| Smaller | 2 | 256 | 4 | 1,024 | 1,605,888 |
| Smallest | 2 | 128 | 2 | 512 | 409,728 |

These are source-derived counts, to be checked against constructed models.
For width D, FFN 4D and separate 60-row embedding/head, both arms have
\(24D^2+129D\) parameters. Our primary count is approximately 6.36M, slightly
different from the paper's reported 6.43M. Retaining width 1,024 from our old
150M backbone would be a separate, larger two-layer setting; I recommend
starting at 512 for this mini-model experiment.

Use the following settings explicitly; do not rely on `ModelConfig` defaults:

- Exact GELU (`approximate="none"`), two pre-norm blocks and FFN width 4D.
  Our saved core recipe uses GELU, although the config class defaults to SwiGLU.
- Ordinary LayerNorm with epsilon 1e-5, learned gain and no bias. Learned Q/K
  normalization operates across the full model width before head splitting.
- Full multi-head attention with head dimension 64, causal ALiBi and no RoPE.
- Zero dropout, no dense biases, no embedding LayerNorm or learned position
  embeddings, final LayerNorm, and an untied bias-free 60-class output head.
- Mitchell initialization; no CDRM fabrics, gates, bridges or additional loss.

The ordinary arm uses `block_type="sequential"`. RT uses
`block_type="recurrent"`, `recurrent_backend="tiled"`,
`recurrent_write_rho=1`, and `recurrent_layers=None`, making both blocks
recurrent. Preserve the current tiled internal recomputation; no outer
activation-checkpoint wrapper is needed initially.

Ordinary attention reads keys and values derived from block inputs. RT reads
the current provisional self entry plus permanent entries written from earlier
block outputs, and writes the current permanent entry from its completed
output. The same projection parameters are reused. Consequently parameter
counts can match exactly while the computations intentionally differ.

Use the checked weights-only
[conversion utility](../recurrent-transformer/olmo/checkpoint_conversion.py)
to create corresponding initial states, then initialize fresh optimizers.
The same random seed alone is insufficient: fused ordinary QKV and split
recurrent projections consume initialization draws differently. Verify all
mapped weights, norms, embeddings and the task head explicitly. Keep precision
config fields compatible for the common FP32 execution; no new conversion
exception or precision-policy mapping is needed for this experiment.

Set `vocab_size=60`, `embedding_size=None`, `weight_tying=False` and request
all-position logits. The native output head already fits the task; no unused
C4 vocabulary head or replacement classifier is necessary. Allow evaluation
context at least 36, and verify runtime lengths 12 and 36 directly. Static
source review found no tile-divisibility requirement at these lengths.

**3. Add a small task harness without rewriting the historical C4 drivers.**

Proposed files, to be created during implementation:

- `configs/rt_a5/`: explicit shared task/optimization config and paired model
  configs, with width variants represented consistently.
- `scripts/rt_a5_data.py` and `scripts/rt_a5_common.py`: generator, manifests,
  model construction, pairing, loss and metrics.
- `scripts/rt_a5_validate.py`: a small set of task and FP32 backward checks,
  not a new general numerical-analysis framework.
- `scripts/rt_a5_train.py` and `scripts/rt_a5_eval.py`: bounded/resumable
  training, fixed evaluations and result export.
- Relevant focused tests, a usage note and a report under `docs/reports/rt-a5/`.

Reuse existing tracking, source provenance and optimizer practices.
The old `head_chunked_backward` owns a shifted C4 objective, and
`CapturedRTBackward` requires all-tiled RT. They cannot be reused unchanged
for this paired experiment. Implement a task-specific forward/loss/backward
body supporting both arms with an ordinary uncaptured training loop. The
60-class head needs no head microbatching. A task-specific CUDA graph wrapper
is deferred unless measured execution overhead makes it worthwhile.

**4. Use a conservative execution mode and only essential checks.**

Read [the numerical handoff](rt-numerical-handoff.md) before implementing.
The established FP32 tiled implementation is a suitable starting point. Its
previous checks need not be repeated as a research project. We still need
small checks for the new objective, short lengths and training harness.

| Execution choice | First experiment |
| --- | --- |
| Attention, projections, MLP, head and loss | FP32 in both architectures |
| Parameters, gradients and optimizer moments | FP32 |
| Autocast and TF32 | Explicitly disabled |
| Whole-model and RT helper `torch.compile` | Disabled |
| CUDA graph capture | Off initially; optional later if timing justifies it |
| RT implementation | Existing tiled recurrence and internal recomputation |

Construct both models with `reference_eager=True`. This bypasses compiled RT
helpers and internal block autocast without replacing tiled recurrence with
the naive algorithm. Explicitly disable outer autocast as well: the eager
flag does not override an enclosing autocast context. Set FP32 model/runtime
precision and verify the constructed parameters are FP32. Explicitly select
PyTorch's math SDPA backend for ordinary attention, rather than relying on
automatic backend selection; a runtime context can do this without changing
the paired model configs.

Using FP32 just for attention, with BF16 dense layers, is a viable later
performance option. For this first comparison, FP32 throughout is simpler:
it removes the BF16-versus-FP32 comparison entirely and avoids the unequal
precision boundaries of the existing ordinary and RT protection policies.
It does not make the implementations mathematically exact or eliminate the
need to check newly written task code.

Keep these bounded checks:

1. **Task and metric correctness.** Test the group oracle, label alignment,
   loss denominator, cumulative exactness, causal prefix behavior and
   batch-order invariance. Reuse small fixtures; no large diagnostic corpus
   or attention instrumentation is needed.
2. **One small FP32 backward regression.** Compare tiled RT with the existing
   naive-autograd FP32 implementation using the new loss, one seed and a
   small batch at lengths 12 and 36. Check outputs and participating parameter
   gradients. This is a pair of short test cases, not a width/seed/precision
   sweep. SEQ and RT gradients are intentionally different and are not
   compared for equality.
3. **Training and recovery smoke checks.** Overfit a small easy set in each
   arm; check finite loss/gradients and actual parameter updates at the intended
   training batch; exercise length-36 evaluation; verify a short save/resume
   continuation. No trained-state gradient attribution or Adam-delta matrix
   is required unless these checks expose a concrete problem.

After these pass, proceed to the pilot. Do not add BF16 qualification, local
FP64 oracles, layerwise attention attribution, precision ablations or new
kernel repairs to the initial milestone. A failure should trigger a targeted
investigation of that failure, not an automatic return to the historical
numerical-analysis program.

The NextLat [README](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/README.md#reproducibility) reports A5
inconsistencies with `torch.compile`, especially on Hopper, and recommends
disabling it for those experiments. Follow that recommendation. Their released
YAML still enables compile, so we should not assert that every published run
was uncompiled. We do not need to reproduce or diagnose their compiler issue.
Disabling whole-model compile alone is insufficient; disable our internal
helper compilation too.

**Measure cost once, then decide whether optimization is needed.** Profile a
short warmup and timed window for each FP32 arm at B1024/L12, plus a length-36
evaluation sample. Extend warmup only if timings have not stabilized. Record
step time, peak memory and estimated 10,000-/400,000-update runtime, including
an allowance for evaluation/checkpoint overhead. These are short sequences
and small models, but 400,000 updates can still be substantial. Short length
reduces attention cost; it does not remove dense MLP cost or eager kernel
launch overhead. No GPU-time saving from full FP32 is assumed in advance.

If the measured cost is acceptable, keep this execution mode for the pilot
and subsequent comparison without further numerical work. If execution
overhead is a material bottleneck, consider **CUDA graphs while retaining
FP32 and uncompiled helpers** first. Graphs are distinct from `torch.compile`;
adding them would require a small captured-versus-uncaptured check covering
loss, gradients and several changed-input/changed-weight optimizer steps.
If dense arithmetic is instead the bottleneck, reconsider the existing FP32
attention/BF16 dense path with a proportionate same-state check. Neither
optimization is part of the default first milestone, and compiler tuning is
deferred. Any adopted execution change must be documented and applied
consistently to the paired comparison.

**5. Use the paper's learning setup, with staged execution and a fresh test.**

| Setting | Proposed primary value |
| --- | --- |
| Training words | One million unique length-12 words, split 800,000 train / 200,000 development |
| Sampling / split seeds | 444 for length-12 pool, independent 445 for OOD development pool, 446 for final confirmation; split seed 42; save actual corpus/split hashes |
| Effective batch | 1,024 words; prefer physical 1,024 after profiling |
| Precision / execution | Full FP32, autocast/TF32/compile off, initially uncaptured |
| Optimizer | AdamW, LR 1e-4, betas (0.9, 0.95), epsilon 1e-8 |
| Regularization | Weight decay 0.01, gradient clipping at norm 1, dropout 0 |
| Schedule | Constant LR, no optimizer warmup, matching this task's released recipe |
| Full reference budget | 400,000 optimizer updates, or 512 nominal passes through 800,000 training words |
| Primary evaluation | Cumulative prefix exactness and isolated state accuracy through length 36 |

These learning settings come from the released
[A5 GPT config](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/config/a5/gpt_a5.yaml),
[defaults](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/defaults.yaml)
and [generation instructions](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/README.md#L76-L84).
Their released recipe uses BF16 mixed precision. Our full-FP32 execution is
an explicit simplification for this architecture comparison.

The earlier C4 warmup schedule does not transfer to this task. Execution
warmup for stable timing/capture is separate from an optimizer LR schedule.
Define and record weight-decay parameter groups rather than assuming that
two AdamW constructors treat norms and embeddings identically. Preserve
the same initialization mapping, training order, batches, schedule and
evaluation examples across arms within a seed.

Use fixed-length batches without packing or padding. Define a deterministic
epoch/remainder policy and record actual word exposures. If physical batch
1,024 is impractical, use matched effective batches with correctly normalized
gradient accumulation, clipping and updating once per effective batch; that
path needs a small equivalence check. The default uncaptured loop avoids
captured replay's gradient-clearing complication.

Suggested execution order after implementation approval:

1. Complete the bounded checks above and a short timing/memory profile on the H100.
   Finish a reviewable first PR with a runnable paired harness, exact resolved
   counts/configs, focused tests and smoke results. Extensive numerical
   analysis or capture support is not required to finish it.
2. Run one paired width-512 development pilot for **10,000 updates**, keeping
   checkpoints at 0, 1,000, 5,000 and 10,000. Log small fixed development
   evaluations during training and fuller evaluations at 5,000/10,000. This
   is 2.5% of the reference budget and cannot establish final failure or
   convergence. Use its measured throughput to estimate the long-run cost.
3. Review optimization health, curves and runtime, retain FP32 unless measured
   cost warrants a separate execution change, and freeze the long-run budget.
   Run the primary width-512 comparison with **three paired
   initialization/data-order seeds**. Compatible pilot runs may continue;
   changes to optimization or precision start an explicitly new comparison.
4. Add width-128 and width-256 pairs as capacity ablations. Screen them on
   development data first; extend selected settings across seeds. If the
   length-12 results leave uncertainty about extrapolation versus learning
   difficulty, add a paired **train-at-36** control. Do not interpret a short
   training plateau or the complexity-theoretic discussion as proof of
   impossibility at these finite sizes.

For length generalization, evaluate length-36 words and compute all prefix
metrics from the same examples. This yields the curve from 1 through 36 in
one causal forward pass per batch. Verify this agrees with explicitly
truncated inputs at selected lengths. Mark the training boundary at 12.

The released NextLat setup generates a separate million length-36 words,
takes a 20% split, and evaluates about 102,400 rows at batch 1,024. We can
use those sizes for development comparison, with an independent sampling
stream. Its instructions reuse seed 444 for both lengths; source inspection
indicates that the common PRNG stream can couple length-36 prefixes to
length-12 words. We have not generated and measured that published-recipe
overlap, so do not report an observed overlap rate. Our generator will use
different streams and explicitly audit complete 12-prefix intersections.
Also generate a **fresh,
disjoint 102,400-word length-36 confirmation set**, with an independent seed,
kept unused until settings and checkpoint-selection rules are frozen. Its
12-prefixes must be checked against training and development before claiming
fresh length-12 results. Exclude any such intersections, and check complete
36-word disjointness as well. Shorter shared prefixes are expected. If repeated OOD development
curves guide any choices, report that selection explicitly. The final set
must not guide it.

Plot both cumulative and isolated accuracy, learning curves versus updates
and processed words, and efficiency versus wall time. Report every seed and
their variation; word-level uncertainty is not a replacement for seed
variation. Compare endpoints at matched training budgets as well as any
prospectively selected development checkpoint. Do not choose whichever
architecture checkpoint happens to look best on the final test.

**6. Tracking, persistence and the first stopping point.**

Use W&B entity `taylorbollman`, proposed project `rt-a5-state-tracking`, with
architecture/width/seed/precision tags and grouped comparisons. Preserve
resolved config, exact source snapshot, runtime, initial mapped weights,
optimizer/RNG/data-order state, checkpoints, local metrics and evaluation IDs.

Use a new local lineage such as `.runtime/rt-a5/<run-id>/` and retain reusable
data, checkpoints and evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/<run-id>/`. Record verified storage
receipts; SSD-only artifacts are temporary. Keep historical RT precision
directories and caches closed. All future GPU commands must run through this
project's Docker launcher after container/GPU verification, per
[AGENTS.md](../AGENTS.md).

The first implementation milestone ends when the paired models, task,
metrics, restart path and bounded correctness checks are working and the pilot
can run reproducibly. The first learning milestone ends with the 10,000-update
paired report and a measured cost estimate. Those are sensible review points
before committing to the full seed/width matrix. This plan does not assume
that RT will solve the task or that the ordinary Transformer must fail.
