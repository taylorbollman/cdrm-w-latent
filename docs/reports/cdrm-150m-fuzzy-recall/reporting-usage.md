# Phase-one reporting

`scripts/cdrm_fuzzy_report.py` reads only the supplied fuzzy-recall lineage. Run it inside the project container after the reports to be summarized are closed, with `--lineage` and a new `--output-dir`. It verifies retained source, checkpoint, numerical-packet, dataset and epoch-order references; preserves failed attempts and numerical flags; and produces JSON, Markdown, CSV, PNG and SVG summaries. Optional online W&B publication uses the authorized `taylorbollman/cdrm-150m-fuzzy-recall` project. It never runs a model, trains, or generates data.

The six learning-rate calibration endpoints must be complete and have consistent paired initialization, data and execution identities before the reporter selects learning rates. Incomplete runs remain visible. Initial Adam update cosine and trained Adam update relative error are separate criteria; initial update relative error is descriptive. Projected 24-run costs use explicitly labeled measured timing proxies and exclude compilation, evaluation, checkpointing and tracking overhead. These phase-one reports are not the final length sweep or a held-out test result.

## Native target and padding semantics

The native vocabulary uses key symbols 0–6, value symbols 7–14 and left-padding symbol 15. Training retains dense, already shifted native targets, including padding. Development uses the native masked value targets, with `-100` as the only ignored target. Previous answer tokens are teacher-forced. Padding is not counted as retrieval success, and no extra padding attention mask is introduced.

The model configs retain upstream `pad_token_id=0` and `eos_token_id=1`. These are inactive metadata in this experiment's direct forward and external loss path:

- `recurrent-transformer/olmo/model.py:1970` creates the embedding without `padding_idx`; direct forward embeds all supplied symbols at line2244.
- Its forward only uses an attention mask when the caller explicitly supplies one (lines2263–2298). The direct fuzzy update/evaluation calls supply none (`scripts/cdrm_fuzzy_common.py:267` and line301).
- `pad_token_id` has no use in this model implementation. `eos_token_id` is read by `generate` when constructing beam search (`olmo/model.py:2571`); this experiment does not call generation.
- The external loss and metrics use explicit labels and `ignore_index=-100` (`scripts/cdrm_tiled_common.py:212–229`), without consulting either token ID.

Consequently key symbols0/1 receive ordinary embeddings and supervision, and native padding15 follows the supplied dense/masked labels. This conclusion is scoped to the audited direct forward/loss path; generation APIs would require their own token metadata setup. Frozen configs and source were not changed for this clarification.

The native generator can also score a terminal query whose mapping never appeared. All such examples and targets remain in the primary metric. In calibration development data, 13 of 16,143 scored tokens lack an earlier exact mapping. The independent oracle reports availability separately, without claiming an accuracy ceiling. The query-ignoring, answer-prefix shortcut reaches 44.3536% token accuracy on that same development corpus; it is more informative than a uniform-token baseline for interpreting learning curves.

## Selected-checkpoint error analysis

After the six-run cohort is audited and the learning-rate decision is frozen, `scripts/cdrm_fuzzy_ceiling_eval.py` can analyze each selected `calibration/*-e10` checkpoint. It first requires exact replay of that report's native BF16 metrics on the same 1,280 development examples and B128 evaluation order. It then reloads the same weights for strict FP32 evaluation. This comparison changes evaluation precision only; it does not measure how an FP32-trained trajectory would have learned.

The available/unavailable lookup masks partition the 16,143 native targets into 16,130 and 13 targets, respectively. Subset exact match includes only examples with at least one subset target; the unavailable subset therefore has nine eligible examples. Empty subsets yield null rates rather than vacuous successes. Token errors partition across subsets, but sequence-error counts need not add because an example can fail both. Per-position FP32 CE is summed in FP64 for the subset analysis, separately from the frozen native batch-reduction arithmetic used for exact replay.

The report retains native metrics, subset metrics, prediction flips/corrections/regressions, logits, predictions, masks and error coordinates for both precisions. Unavailable lookup targets can still contain statistical information through teacher-forced answer prefixes and the value-generation rules. The helper's filename does not imply an established statistical ceiling, and these development diagnostics do not modify the selection rule or constitute final-test evidence.

The completed selected-checkpoint checks replayed native BF16 metrics exactly for both arms, and strict FP32 evaluation changed no scored predictions. CDRM retained 16 native errors: five among the 16,130 available targets and 11 among the 13 unavailable targets. SEQ retained 20: eight available and 12 unavailable. Exact match over available targets was 1,275/1,280 examples for CDRM and 1,272/1,280 for SEQ. These are one calibration replicate's post-selection observations; they do not establish agreement between training trajectories or a robust architecture difference.

The audited [W&B aggregate](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/gmibkb9d) contains all six development curves. The detailed [CDRM](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/2bxainuy) and [SEQ](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall/runs/floch77w) runs retain the post-selection diagnostic metrics. Artifact verification is recorded in the [storage receipt](storage.json); report text does not depend on its eventual manifest digest.
