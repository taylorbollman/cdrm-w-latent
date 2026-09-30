# Saved NF/NFR comparison with F-only pass curves

This is an observational follow-up to the F-only study. It does not change the
frozen F, NF, NFR or KL-continuation training protocols, weights, optimizer
states or source inventories.

## Question and scope

Compare the pass dynamics of saved NF at updates 0, 32 and 64, and saved NFR at
update 32, against F-only curves at available training milestones. NF64 can be
the explicitly identified KL1 control or KL0.1 branch; these are distinct
conditions and must never be combined under an unlabeled NF64 name.

The measurement uses all eight original `dev-main` packed T1024 rows, exactly
the same row membership, order, token IDs, document boundaries and loss masks
as the F-only probe. Each row is evaluated independently at physical B1. All
32 total passes are measured under common FP32 math, with feedback beta1,
zero feedback jitter and the condition's original RT selection. The first pass
is included. In NFR, configured RT remains active in every pass.

NextLat's predictor and its saved tensors remain resident and unchanged, but
the predictor is **never called**. This measures the inference dynamics of the
backbone and fusion after NextLat training, without executing auxiliary losses
or introducing a predicted-latent inference recurrence.

## Execution and preservation

`scripts/olmo_fbt_component_curves.py` authenticates the old saved training
report, declaration/resolution, checkpoint manifest/state hashes, parameter
aliases and exact row cursor. NF64 additionally authenticates its declared KL
branch. Weights are copied into the matching construction with `assign=False`.
No optimizer, scheduler, rank RNG or data cursor is restored or advanced.

The old feedback diagnostic cannot directly serve this purpose: it hardcodes
four passes and evaluates the full NextLat objective, including predictor
readouts. The new bounded adapter calls the unchanged canonical FBT core once
for the requested pass count, retains only its hidden states plus small scale
observations, and reuses the F-only metric definitions, vocabulary chunking,
region masks and raw-statistic reducer. It introduces no new recurrence math.
At native B1/T1024/D2048, 32 retained FP32 hidden states occupy approximately
256 MiB; full-sequence vocabulary logits are not retained.

Runtime preservation uses the existing common-FP32 evaluation scope. Complete
parameter/buffer hashes and RNG state are additionally compared before and
after the diagnostic, and gradient buffers remain absent. Temporary hooks
are removed on success and exceptions. A guard fails if the NextLat predictor
is unexpectedly executed.

## Metrics and interpretation

Report per-pass CE, full-vocabulary predictive entropy, hidden and pre-final-
norm scales, input scales, consecutive hidden changes, normalized hidden
changes and cosine similarity. Regions are all positions, each position
quartile, the final 128 positions, and the still-unsettled causal suffix.
Numerators and counts are pooled before means or ratios are calculated.

These are the same FP32 observations as F-only, not new BF16 compatibility or
quality acceptance criteria. A state curve settling to poor CE is not useful
refinement. Curves at different updates are different learned weights. Curves
across architectures isolate neither optimizer history nor every effect of
auxiliary training; the first 32 updates nevertheless share the declared data
and hyperparameter prefix.

## Resource and publication policy

`--preflight-only` performs CPU authority and fixed-panel checks without model
construction or GPU execution. Model execution requires the project container
and a working CUDA device; there is no CPU fallback. Default `--max-passes 32`
produces the complete curve. Explicit `--max-passes 4` or `8` can estimate NFR
diagnostic cost first and is labeled as a shorter curve, not substituted for a
32-pass result. The FP32 eager RT reference may be substantially slower than
ordinary F/NF evaluation.

Each finished row is written immediately; the complete report is published
only after all eight rows and preservation checks pass. W&B records scalar
curves against total pass count. Source snapshots and input SHA pins are saved
with the report. All new files are separately pinned and do not extend any
previous training runtime inventory.

The separate CPU-only `scripts/olmo_fbt_component_curves_summary.py` accepts
completed F probe JSON and completed component reports with `--inputs`, writing
CSV, PDF, PNG and immutable input snapshots under a fresh `--output-dir`.
It rejects mismatched panel membership, target counts, precision or beta, and
rejects duplicate condition/update/objective inputs. NF64 KL1 and NF64 KL0.1
receive different labels. It performs no model execution or GPU operations.
