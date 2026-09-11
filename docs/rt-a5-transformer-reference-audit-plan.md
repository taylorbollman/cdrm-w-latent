# A5 Transformer curves: consistency audit and reference-alignment plan

2026-09-11. This is an analysis and proposed next step, not authorization for
new training. No models were evaluated or trained for this audit. Existing
reports, checkpoints and closed evidence archives were not modified.

## The local plots are consistent

`reports/rt-a5/nextlat-pilot/length-full.pdf` and `length-boundary.pdf` use
the same checkpoint, data and curves. The latter changes only the horizontal
view from positions 1–36 to 10–18. Its three panels retain the same meanings:

- E(t): fraction of words with **every state through t correct**.
- A(t): accuracy of the state at t alone.
- M(t): mean token accuracy across positions 1 through t.

The source is the shared loop in `scripts/rt_a5_nextlat_report.py`, lines 324–339.
The old pilot's `length-generalization.pdf` plots E and A for the very same
saved SEQ model. The newer right-hand M panel is an additional statistic.

Audit evidence: the old and new SEQ endpoint metrics and checkpoint identity
are identical. Both newer PDFs and PNGs match their recorded artifact hashes.
An independent read of the blue Transformer vector paths in the PDFs checked
108 full-view points and 27 cropped-view points against the saved E/A/M values;
differences were below 2e-9 in fractional accuracy from PDF coordinate rounding.

| Position | E, every state correct | A, state at position | M, mean token accuracy |
| --- | ---: | ---: | ---: |
| 4 | 91.6328% | 92.0176% | 97.2080% |
| 5 | 61.8926% | 65.4619% | 90.8588% |
| 6 | 8.6055% | 10.5645% | 77.4764% |
| 8 | 0.0732% | 1.9414% | 58.7410% |
| 10 | 0% | 1.7354% | 47.3410% |
| 12 | 0% | 1.6943% | 39.7259% |
| 14 | 0% | 1.6279% | 34.2868% |
| 18 | 0% | 1.6494% | 27.0411% |

The crop hides the SEQ drop around 4–8. Elevated M in that crop reflects
correct early predictions, not accurate late states. Future plots should
explicitly title this view “zoom: positions 10–18” and use an E-only full-range
panel when comparing with Figure 10. There is no numeric plot correction to make.

## Comparison with the NextLat paper

The released [A5 evaluator](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5_data.py)
uses a cumulative product of per-position correctness, matching E(t). This
supports using our left panel for Figure 10 comparison. The original figure's
plotting data/script were not located, so this is an interpretation supported
by the released evaluator rather than an independently reconstructed figure.

Our SEQ result is a short-budget architecture experiment. It has not learned
accurate late states even within its length-12 training range. Its current
curve therefore cannot establish the paper's learned-through-12, failing-beyond-12
behavior, or a converged capacity limit at 5 tokens.

The [released GPT configuration](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/config/a5/gpt_a5.yaml)
specifies 400k updates at batch 1024; our SEQ checkpoint has 10k at that batch.
That is 40 times more training in the reference: 409.6M versus 10.24M word
presentations. The reference corpus is also split into 800k training and 200k
validation words. The paper's one-million-word description concerns the
generated pool; corpus size alone is not a demonstrated mismatch.

Architecture details from the pinned [GPT implementation](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_gpt.py)
and its [normalization implementation](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/models/model_base.py):

| Setting | Our SEQ Transformer | Released A5 GPT |
| --- | --- | --- |
| Layers / width / heads | 2 / 512 / 8 | 2 / 512 / 8 |
| Position encoding | ALiBi | RoPE |
| Block/final norms | LayerNorm | RMSNorm |
| Q/K normalization | Learned full-width normalization | None |
| Feedforward | GELU, hidden 2048 | SwiGLU, hidden 1408 |
| Initialization | Existing Mitchell initialization | Normal(0,.02) weights |
| Embedding / head | Untied, 60 classes | Untied, 60 classes |
| Parameters | 6,357,504 | 6,486,528 |

Thus replacing ALiBi with RoPE isolates one difference; it does not reproduce
the released GPT. Base optimizer settings substantially agree. Our full FP32
runtime differs from the released mixed-precision launcher. Compilation can
remain off: the authors explicitly recommend that for A5 on Hopper in their
[reproducibility notes](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/README.md#reproducibility),
despite the YAML's `compile: true`. No new RT numerical campaign is warranted.

A further data detail needs care in any literal reproduction. The published
[generation commands](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/README.md#state-tracking)
reuse seed 444 for the short and long corpora. The sequential draws in the
[generator](https://github.com/JaydenTeoh/NextLat/blob/b37d3411ab9b17be8638abbddb9529f0f3a0a5f9/data/a5/prepare.py)
can then yield shared 12-token prefixes. A bounded standard-library check
confirmed this property of those instructions. Our long corpus uses an
independent stream and excludes full 12-token prefix overlap. This does not
establish which data the paper actually used or explain its performance.
Do not silently change our primary held-out set to mimic that property.

## Recommended next experiment

1. **Continue existing SEQ from 10k to 100k total updates.** Preserve its
   architecture, data, optimizer and exact resume contract. Save and evaluate
   at 25k/50k/100k, tracking E/A/M over 1–36 and the separate length-12 development
   set. This directly tests the short-training explanation. Estimated additional
   time is 25–30 minutes from the measured original run. Existing pure RT 100k
   provides a matched-budget comparison without another RT run. Keep the 10k
   NextLat experiments separate from that budget comparison.
2. **Establish a released-GPT-architecture control.** Use the pinned authors'
   GPT module, including RoPE, RMSNorm, SwiGLU, absent QK normalization and its
   initialization. Train on our same frozen data for a controlled comparison,
   with 10k and 100k readouts. Keep FP32/eager initially and document that this
   matches architecture rather than every historical execution detail. Check
   simple task/loss alignment and finite updates, then train; no RT backward
   redesign or extensive precision qualification is involved. Profile this
   model before estimating its training cost.
3. **Use RoPE-only as a targeted follow-up if a gap remains.** Change only
   position encoding in our SEQ architecture; compare at matched budgets and
   data. This separates the positional effect from the rest of the released
   recipe. If budget rather than architecture explains the gap, this ablation
   need not precede useful RT+NextLat follow-up work.

100k is an intermediate decision point, not the paper's endpoint. If these
controls still do not reproduce the reference pattern, selected runs should
reach 400k before declaring disagreement with the reported result. A literal
reproduction would additionally audit the source-native dataset, execution
precision and evaluation mapping, and report any departures. Compare that
source-native benchmark separately from our independent-prefix benchmark.

No reusable official A5 GPT checkpoint or Figure 10 CSV was found in the
bounded source/release searches. That does not rule out unpublished or
unlinked artifacts. One seed can localize a large discrepancy; confirming
an architectural conclusion should subsequently use additional seeds.

The user excludes autonomous NextLat MLP recurrence experiments. This plan
does not reintroduce them. Independent confirmation remains unevaluated,
and any future runs continue to require container GPU verification, online
W&B tracking and retained checkpoints under `gs://fast-chunks`.
