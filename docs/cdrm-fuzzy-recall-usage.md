# Native MAD fuzzy recall at the 150M architecture scale

This milestone implements the approved
[experiment plan](cdrm-150m-fuzzy-recall-plan.md). Its first scope is task support,
the production precision option, full-width numerical validation, operational
measurements and development calibration. The final four-length comparison is
a subsequent frozen experiment. Check the retained reports for completed scope;
the existence of a configuration does not establish numerical clearance.

## Models and precision

Both arms have 12 ordinary blocks, D1024, 16 attention heads, learned LayerNorm
and Q/K normalization, GELU, ALiBi, and untied V16 input/output tables. The CDRM
arm has F4096 and 153,175,040 unique parameters. The matched standard Transformer
uses F4192 and 153,437,184 parameters, 0.171% more. Native MAD symbols replace the
language-model vocabulary; these are fresh synthetic-task models.

CDRM executes ordinary blocks 0–8 once. Its side fabric shares block 3, combines
the early and deep previews from blocks 3 and 8 through the established
read-conditioned write recurrence, and bridges after block 8 before blocks
9–11. Its two D×D adapters are separately owned; shared block weights are counted
once. Rho=1, epsilon=0.1 and lambda=0.01 are fixed. There is no learned gate or
separate recurrent replacement of an ordinary block.

The explicit model option `ordinary_attention_precision_policy="fp32"` puts
ordinary attention in FP32 for every block and both arms. Ordinary MLPs, the
head and tiled fabric use BF16; parameters, residual/recurrent state, established
normalization arithmetic, aligned cross entropy and Adam state retain their
FP32 policy. The option defaults to `legacy` for existing callers. Unsupported
dropout, FlashAttention and owner activation checkpointing are rejected. The
archived five-block wrapper and previous checkpoint identities remain separate.

## Native data semantics

`cdrm/mad_data.py` supports `fuzzy-in-context-recall` from pinned MAD revision
`0f49a452b84ca0d13f8eb9c1ffa649032376fb1b`. Key tokens are 0–6, value tokens 7–14,
and padding is 15. Training keys have length 1–3; held-out keys have length 3.
Values have length 1–3 in both splits. Requested length T is the actual input
length after the generator's own shift.

Training uses the native dense aligned targets, including padding. Development
uses the native repeated-key/terminal-probe answer mask. Labels are not shifted
again. Teacher forcing exposes earlier answer tokens; masked exact match is not
free-running generation. Training answer masks are independently annotated
without regenerating inputs under a different training flag.

Native generation occasionally emits a terminal query whose value was not
inserted into the input. These examples are retained with their original native
targets. The independent `oracle_available_mask` identifies positions where
causal lookup has a known answer. Conditional oracle accuracy and coverage are
reported separately; coverage is not a universal statistical ceiling. In the
calibration development corpus, 13 of 16,143 answer targets lack that lookup
history. The query-ignoring answer-prefix shortcut reaches about 44.35%, so
exceeding uniform eight-value chance alone does not demonstrate key recall.

## Entry points

- `scripts/cdrm_fuzzy_prepare.py` freezes native arrays, masks, source hashes,
  baseline/oracle checks and the 50-epoch shuffle order under an existing protocol.
- `scripts/cdrm_fuzzy_train.py --mode prepare-init` saves paired fresh models.
  The 75 shape-identical embedding, attention and norm tensors are copied
  bitwise; 24 resized MLP tensors retain their native independent initialization.
- `scripts/cdrm_fuzzy_validate.py` compares the same saved state under strict
  FP32 and the mixed policy. CDRM can additionally compare naive FP32, independent
  unnormalized fabric gradients, scaling and causality. Raw packets are retained.
- `scripts/cdrm_fuzzy_profile.py` measures the actual update helper after warmup
  on numerical fixtures. Its discarded updates are operational measurements.
- `scripts/cdrm_fuzzy_train.py --mode train` supports the frozen T256 calibration
  only: 12,800 training and 1,280 development examples, three declared learning
  rates, paired initialization seed 86100 and at most ten epochs. It deliberately
  does not launch the final length/seed sweep.
- `scripts/cdrm_fuzzy_report.py` aggregates retained evidence and plots. Incomplete
  or failed cases remain visible. LR selection requires all six matching
  calibration endpoints.
- `scripts/cdrm_fuzzy_ceiling_eval.py` evaluates an explicitly selected epoch-10
  checkpoint after calibration. It first requires exact replay of the official
  BF16 native development metrics, then reports errors on targets with and
  without an earlier lookup mapping and compares strict FP32 evaluation at the
  same weights. It retains both sets of logits/predictions and their masks.
  These secondary diagnostics do not change native labels or learning-rate
  selection; subset exact match counts only examples containing subset targets.

Use `--help` for each entry point inside the project container. All GPU execution
must use `bash scripts/docker_shell.sh bash -lc '<command>'`, verify
`/.dockerenv`, working directory `/workspace/cdrm-w-latent` and `nvidia-smi` inside
that container. Set `CDRM_DOCKER_GPUS=none` explicitly for CPU-only container work.

## Calibration and exact continuation

The optimizer is AdamW with betas (0.9,0.98), epsilon 1e-8, no weight decay,
clipping norm 1, and explicit non-fused/non-foreach updates. The cosine schedule
has a fixed 50-epoch horizon, minimum LR 1e-6, no warmup and steps after each
completed epoch. The candidates are 1e-4, 5e-4 and 1e-3 for both architectures.
Native development token accuracy at the common ten-epoch endpoint selects the
LR, with answer CE breaking ties. Final data remains untouched during selection.

Every invocation uses a new output directory. A continuation supplies both the
original `--initial-checkpoint` and a trained `--checkpoint`. Preserve the
protocol, source closure, native corpus, physical batch, LR, precision, monitor
and checkpoint calendars, and explicit retained Inductor cache path. The runner
checks optimizer/scheduler/RNG restoration, the complete batch history and next
batch, as well as model/configuration identities. It rejects incompatible
continuations rather than silently changing their meaning.

For a recovery comparison, first save a checkpoint *within* the uninterrupted
reference invocation, then resume that checkpoint to the same endpoint with
`--reference-final` pointing to the reference endpoint. This preserves the
development evaluation calendar. Compare across an epoch boundary to exercise
the learning-rate transition. Continue the uninterrupted endpoint as the
calibration authority once numerical gates are assessed.

## Numerical interpretation and retention

The existing [numerical contract](reports/cdrm-tiled-bf16/validation-contract.md)
is unchanged. Passing the initial Adam cosine criterion does not establish a
small first-step relative distance; report both. Trained same-state Adam updates
use the stricter relative-L2 criterion. Raw FP32 coordinate flags remain visible
alongside their scale-aware assessment. Small-batch results do not clear the
chosen full training batch, and a T256 trained check does not alone clear the
later trained T300 path.

This lineage is `.runtime/cdrm-150m-fuzzy-recall/20260908T031418Z`, with an
immutable initial protocol in `decisions/protocol-v1.json`. Later batch and
calibration decisions must be new linked records. Source files in the paired
initialization's dependency closure are frozen for that lineage.

Graphable runs log online to
[taylorbollman/cdrm-150m-fuzzy-recall](https://wandb.ai/taylorbollman/cdrm-150m-fuzzy-recall).
Artifacts are intended for
`gs://fast-chunks/cdrm-w-latent/cdrm-150m-fuzzy-recall/20260908T031418Z/`.
Only verified upload receipts establish retention; a configured destination does
not establish that an upload completed. Closed scientific objects may be staged
with immutable receipts while work continues, followed by a complete final
manifest. The prior numerical lineage is retained by its existing manifest and
verified GCS reference.
