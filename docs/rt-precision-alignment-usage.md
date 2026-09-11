# Standard RT mixed-precision experiments

The protected precision policy remains the default. Selecting `legacy` is an
explicit experimental choice made before constructing the model and capturing
its CUDA graph. This milestone concerns the standard 12-layer all-recurrent
D1024/H16/FFN4096 model at T512 and physical B512, with 151,045,120 backbone
parameters and 216,843,264 total parameters.

The completed 500-update, two-seed pilot passed its trained-state numerical
screens but did not clear the frozen development-loss margin for `legacy`.
The protected default is unchanged. See the
[results](reports/rt-precision-alignment/final-results.md) and
[implementation summary](reports/rt-precision-alignment/pr-summary.md).
Confirmation remains unused, and the 500-update prefix does not validate the
full 5,000-update warmup or peak learning rate.

Run GPU commands through `bash scripts/docker_shell.sh bash -lc '…'`, verify
`/.dockerenv`, `/workspace/cdrm-w-latent` and `nvidia-smi` inside the container,
and use a fresh output directory per execution. Keep a fixed, explicitly named
`TORCHINDUCTOR_CACHE_DIR` for each experiment lineage. Every graphable run logs
online to the `taylorbollman/rt-precision-alignment` W&B project.

The authoritative execution lineage for the initial investigation is
`.runtime/rt-precision-alignment/20260910T191100Z`. Its frozen
`reference/protocol.json` defines thresholds, seeds, schedule and stopping
rules before the fresh precision comparisons. The originally approved plan is
preserved as `reference/approved-plan.md`; corrections and results belong in
the current plan/report rather than replacing historical evidence.

## Drivers and scope

| Driver | Purpose |
| --- | --- |
| `scripts/rt_precision_dtype.py --arm A\|B\|C` | Tiny compiled tensor-boundary observation; loss and all gradients must match observer-free execution bitwise. |
| `scripts/rt_cuda_graph_validate.py --tier tiny\|full --precision bf16 --policy legacy` | Same-policy captured/uncaptured gradient, changed-input Adam, state and causality checks; full tier uses B2/T512. |
| `scripts/rt_batch_profile.py --batch 512 --updates 10 --warmup 2 --cuda-graph --policy legacy` | Actual physical-B512 initial gradient equivalence and bounded operational timing. Random-token updates are discarded. |
| `scripts/rt_precision_compare.py --tier full --batch 2 --seed 20260910 --ids-path …/diagnostic.npy --observe-layer-outputs` | Identical-state A/B/C native-loss, gradient, clipping and Adam comparisons on real tokens. Optional checkpoint supplies both weights and Adam state. |
| `scripts/rt_precision_credit.py` | Terminal-output credit and fixed-forward cotangent scaling by 1/32 and 32; the unused vocabulary head is explicitly excluded. |
| `scripts/rt_precision_attention_probe.py --case-dir …/full-b2-seed0 --layer 9` | Conditional FP64/FP32 attention derivatives using the actual legacy operands and incoming gradient, with exact retained-case reproduction. |
| `scripts/rt_precision_partition.py` | Physical versus microbatched FP32 reference check, available if a future reference needs partitioning. The initial physical B512 FP32 case fit without this fallback. |
| `scripts/rt_precision_data.py` | Pinned fresh C4/T5 matrices and document boundaries for disjoint diagnostic/development/confirmation roles. |
| `scripts/rt_precision_train.py` | Paired captured real-C4 training under the released 5,000-update warmup; explicit 100/500-update endpoints and checkpointed continuation. |
| `scripts/rt_precision_eval.py` | Saved-checkpoint held-out CE under common FP32 or native BF16 evaluation, with per-target losses and document aggregates. |
| `scripts/rt_precision_pair_eval.py` | GPU-disabled CPU analysis of explicitly paired evaluations: verifies token/document/checkpoint provenance and applies the frozen document-bootstrap margin. |
| `scripts/rt_precision_resume_probe.py` | Fresh-process capture and checkpoint-resume proof using actual model, gradient, Adam and RNG bytes. |
| `scripts/rt_precision_pilot_report.py` | CPU plots of explicitly closed 100-update training and development reports. |
| `scripts/rt_precision_outcome_report.py` | CPU figures from explicit closed training segments, paired evaluation summaries and trained numerical anchors, with missing coverage stated. |

All drivers require `--output-dir`; use `--help` for the remaining explicit
inputs. Training requires the frozen data manifest and protocol. Evaluation
also requires the completed training report as its checkpoint/source/data
authority; it checks saved tensors before loading to avoid silently promoting
lower-precision saved weights. W&B grouping and names identify the arm, seed,
batch and phase. These
scripts do not modify the project model defaults or enable the older
`make_graphed_callables` wrappers.

Run CPU-only analysis in the container with `CDRM_DOCKER_GPUS=none`. For a
resumed 500-update curve, supply both its original 100-update report and its
completed 101–500 continuation to the outcome driver. Exclude the interrupted
`A-seed0-500` directory; the accepted replacement is
`A-seed0-500-restart1`. The final development disposition is recorded in
`reference/development-disposition-500.json`. Keep the existing lineage and
its frozen compiler cache unchanged; any further experiment needs a new
output directory and an explicit scope.

The numerical driver runs the compiled model without capture so it can retain
raw diagnostic packets. Its results apply to captured execution only alongside
the separately measured same-policy graph checks. `--reference-microbatch N`
partitions only the FP32 reference while preserving the global B×T denominator
and performing clipping/Adam once. That changes the FP32 batch/reduction order;
validate it against physical FP32 where feasible before relying on that fallback.
It does not establish physical-B512 FP32 capacity.

## Interpretation and retention

Review flags use every parameter coordinate and FP64 metric reductions without
a model-size-dependent absolute error floor. They are engineering prompts for
investigation, not tolerances reported by the authors. Initial Adam update
distance and small-gradient sign sensitivity remain visible even when update
cosine similarity is high. Graph equality, arithmetic closeness, optimizer
behavior and short-run learning each answer different questions.

The real-text pilot preserves the released cosine scheduler's `alpha_0=0.1`,
`alpha_f=0.1`, 5,000-update warmup and 12,500-update horizon, with peak LR 0.001.
Update 1 uses 0.00010018; update 100 uses 0.000118; update 500 uses 0.00019.
A 500-update pilot therefore does not test peak learning rate or establish
long-term quality. Report CE per supervised token for held-out comparisons;
the training objective uses CE sum divided by B×512 over 511 targets per row.

Retain reusable data, checkpoints, numerical packets, code snapshots and
compiler caches at
`gs://fast-chunks/cdrm-w-latent/rt-precision-alignment/20260910T191100Z/`, with
hash-verified receipts. Home/project files persist, while SSD-only caches do
not. Data manifests describe this freshly tokenized C4 slice; it is not the
authors' unavailable preprocessed corpus.
