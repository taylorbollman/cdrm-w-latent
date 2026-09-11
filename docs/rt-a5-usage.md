# Two-layer A5 experiment: usage

This implements the task and paired models in the
[approved experiment plan](rt-a5-experiment-plan.md). The initial execution
uses FP32 throughout, no autocast or TF32, no `torch.compile` (including RT
helpers), and no CUDA graphs. It compares two ordinary Transformer blocks
with two tiled recurrent blocks. Neither arm includes the NextLat auxiliary
objective or a separately rolled-out latent model.

## Container and storage

Run project commands from `/home/taylorbollman/cdrm-w-latent` on the host.
CPU dataset preparation and focused CPU tests use the explicit CPU container:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python -m pytest -q tests/test_rt_a5_data.py tests/test_rt_a5_common.py'
```

GPU commands must enter the project container and verify the GPU first:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && python scripts/rt_a5_validate.py --mode checks --output-dir .runtime/rt-a5/NEW-RUN/checks'
```

Replace `NEW-RUN` with a fresh lineage. Validation refuses to overwrite its
output directory. Graphable runs use online W&B under `taylorbollman`, project
`rt-a5-state-tracking`; the launcher supplies credentials without printing them.
Retain reusable data, checkpoints and evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/<lineage>/`. Retention receipts belong in
the run report; the initial corpus upload is verified in
[storage-data.json](../.runtime/rt-a5/20260911T154748Z/storage-data.json).
Project/home files persist, while SSD-only storage is temporary.

## Preparing and loading data

The complete initial corpus already exists at
`.runtime/rt-a5/20260911T154748Z/data`. Reuse that corpus for paired runs. To
prepare an explicitly new lineage with the same default recipe:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python scripts/rt_a5_data.py prepare --output-dir .runtime/rt-a5/NEW-RUN/data'
```

For a small integration fixture, add `--pool-size 100 --confirmation-size 17
--ood-eval-size 13`. Default operation lengths remain 12 and 36. Preparation
refuses any existing output directory, including a partially prepared one.
The full command generates data and integrity evidence, not model evaluations.

The alphabet is exactly the 60 even permutations of `(1,2,3,4,5)`, sorted
lexicographically. ID 0 is identity. There are no extra tokens: all input and
label values are in `0..59`. No BOS, EOS, padding, interleaved answers or target
feedback is added. The model configuration's unused EOS/pad metadata does not
make identity special; identity positions participate in the loss normally.

For operations `g1, g2, ...`, the label at position `t` is the accumulated state
after `gt`, with `state = state compose gt` and `(p compose q)(i) = p(q(i))`.
Thus the first label equals the first input. The task loss is 60-class CE at
every position, averaged over all words and positions, with no language-model
target shift or ignored identity labels.

| Role | Length | Stored words | Words returned by `load_split` | Intended use |
| --- | ---: | ---: | ---: | --- |
| `train` | 12 | 800,000 | 800,000 | Primary training |
| `dev` | 12 | 200,000 | 200,000 | In-distribution development |
| `ood_dev` | 36 | 200,000 | 102,400 | Development length-generalization curves |
| `train36` | 36 | 800,000 | 800,000 | Reserved later train-at-36 control |
| `confirmation` | 36 | 102,400 | 102,400 | Separate frozen final protocol only |

Inputs and labels are read-only uint8 NumPy memmaps:

```python
from scripts.rt_a5_data import load_split, validate_manifest

root = ".runtime/rt-a5/20260911T154748Z/data"
manifest = validate_manifest(root)
inputs, labels = load_split(root, "train")
```

Convert selected batches to int64 when constructing model input/label tensors;
do not change or train directly into the read-only arrays. `load_split` checks
all file hashes and array schemas, caching verification within the process
until manifest/file metadata changes. Its `ood_dev` result is the first 102,400
rows of the frozen 200,000-row long development split. The remaining stored
rows are not part of that normal evaluation interface.

The native generator deliberately uses NumPy `Generator(PCG64)` uint8 draws
instead of the upstream Python/set generator. It preserves the mathematical
and serialization contract, but does not claim byte-for-byte identity with an
upstream dataset produced with the same integer seed. Unique words retain
first-occurrence draw order. Sampling seeds are 444, 445 and 446 for the short
pool, long pool and confirmation respectively. `RandomState(42)` determines
the retained split indices: shuffled development rows first, then training.

The long pool excludes complete length-12 words from both short-pool roles as
prefixes. Confirmation excludes length-12 prefixes from the entire short and
long pools, including the reserved train-at-36 control. The manifest records
rejections, all pairwise training-length prefix intersections, complete-word
intersections, within-role duplicates, exact source indices, file hashes and
generator provenance. Shorter shared prefixes are expected and unrestricted.
The initial corpus needed zero collision rejections and has zero recorded
12-prefix or complete-word intersections. Integrity checks of confirmation
data are permitted; using its model outcomes for development is not.

## Matched model interface

[rt_a5_common.py](../scripts/rt_a5_common.py) provides `model_config`,
`build_model`, `configure_fp32_runtime`, `fp32_context`, `task_loss`,
`A5Metrics` and `make_optimizer`. Explicit model configuration snapshots live
under [configs/rt_a5](../configs/rt_a5).

| Width | Heads | GELU FFN width | Parameters per architecture |
| ---: | ---: | ---: | ---: |
| 512 | 8 | 2,048 | 6,357,504 |
| 256 | 4 | 1,024 | 1,605,888 |
| 128 | 2 | 512 | 409,728 |

Width 512 is the primary experiment. Both architectures use two pre-norm
blocks, LayerNorm and learned full-width Q/K normalization, exact GELU, causal
ALiBi, zero dropout, no dense biases, final LayerNorm and separate 60-row
embedding/output matrices. Both RT blocks use tiled recurrence with write
rho 1 and the existing internal recomputation. No outer checkpoint wrapper
is introduced. These preserve our model components; NextLat's released GPT
uses a different RoPE/RMSNorm/SwiGLU recipe.

`build_model("seq", width=512, seed=0, device="cuda")` and the corresponding
`build_model("rt", ...)` initialize from the same canonical CPU ordinary
Transformer and use the checked weights-only conversion for RT. Construction
verifies exact parameter counts and canonical parameter hashes. The attached
`a5_initialization` metadata records pairing and conversion coverage. Fresh
optimizers are initialized afterward; ordinary and recurrent gradients are
not expected to equal each other.

Call `configure_fp32_runtime()` before model work and wrap forward/backward
with `fp32_context("cuda")`. The context disables enclosing autocast and
selects ordinary math SDPA after model construction. `reference_eager=True`
bypasses internal RT compilation and autocast. The policy fields labeled
`legacy` do not enable mixed precision in this execution mode.

The optimizer uses AdamW at constant LR 1e-4, betas `(0.9, 0.95)`, epsilon
1e-8, matrix weight decay 0.01 and no decay on one-dimensional norm parameters.
The training loop clips the global gradient norm at 1. Execution timing warmup
is separate from the task's constant learning-rate schedule.

`A5Metrics` reports both isolated state accuracy and cumulative prefix
exactness. At position `t`, cumulative exactness counts a word only if all
predictions through `t` are correct. Its final value is whole-word exact match;
the final isolated accuracy is final-state accuracy. Keep these separate in
plots. Position 12 marks the primary training boundary in length-36 curves.

## Bounded checks and evidence

The initial focused data suite passed all 16 tests in the explicit CPU
container. It covers all 3,600 products against permutation matrices,
associativity, identity/inverses, noncommuting composition, same-position labels,
deterministic draw order, collision exclusion, split ordering, immutable
outputs, memmaps and manifest corruption. The shared model tests cover exact
counts and paired initialization, configuration snapshots, loss denominator,
metrics, FP32 contexts and optimizer parameter groups.

A separate read-only CPU audit of the complete prepared corpus on 2026-09-11
verified all 17 artifact hashes, all labels using an independently reconstructed
permutation-matrix multiplication table, all input/label ranges, expected role
shapes, saved split ordering, first 32 source-pool draws against each sampling
seed, and every recorded overlap count. It found zero label mismatches or
unexpected overlaps. Confirmation was checked only for data integrity; no
model was evaluated on it.

The audit also confirmed that the generator source still matches the manifest:

```text
manifest SHA256:
944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb

generator source SHA256:
dc857d457a70311899db4db7690a0d22e6f8a5c842341693b25fea9e16ffff23
```

The GPU validation entry point has three bounded modes: `checks` for task
causality and a small tiled-versus-naive FP32 backward comparison at lengths
12/36; `overfit` for an easy small set; and `profile` for primary-width
B1024/L12 timing plus a length-36 evaluation sample. Use a separate fresh
output directory for each mode. Numeric CPU tests and corpus-integrity results
do not establish GPU correctness, throughput or task performance. Those results
must be reported from actual validation/training artifacts.

## Training and development evaluation

The default trainer runs the primary width-512 model for 10,000 updates at
physical batch 1,024. Run each arm sequentially on the same H100. For example,
after the bounded checks pass:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && python -m scripts.rt_a5_train --data-dir .runtime/rt-a5/20260911T154748Z/data --output-dir .runtime/rt-a5/NEW-RUN/train-seq --architecture seq --wandb-group NEW-RUN-pilot'
```

Use `--architecture rt` and a separate `train-rt` output for its pair. Match
`--seed`, `--data-order-seed`, width, batch and total updates; defaults are
1234, 1234, 512, 1,024 and 10,000. Output directories must be fresh. The first
pilot uses `20260911T154748Z/train-seq` and `train-rt`; do not overwrite them.
All training/evaluation entry points require the GPU container and online W&B.

`WordOrder` derives each epoch's permutation from `(data_order_seed, epoch)`.
It carries the end-of-epoch remainder into the next full batch, drops no
examples, and resumes from the absolute word offset. Training saves an
incremental hash of the actual row indices, so matching final order hashes
verify the two arms consumed the same ordered batches. Every update uses
same-position CE over `B * 12` targets and clips/updates once. There is no
gradient accumulation or graph replay in this implementation.

Default local checkpoints are at updates 0, 1,000, 5,000 and 10,000. They
contain model/optimizer tensors, RNG state, word offset/order hash, paired
initialization audit and source/data/model/runtime identity. `history.jsonl`
records each update; `report.json` records checkpoints, evaluation curves,
tracking status and the source manifest. Exact executed source files are
copied to each run's `source/` directory. The live W&B training series log
25-update means, while local history retains every update.

Development evaluation runs every 500 updates on fixed 4,096-word subsets
of both `dev` and `ood_dev`. At updates 5,000 and 10,000 it uses 102,400
examples per role. Checkpoint choice for this pilot is the fixed 10,000-update
endpoint. These are development results; confirmation remains unused.

Resume into a fresh continuation directory, using an absolute later endpoint:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && python -m scripts.rt_a5_train --data-dir .runtime/rt-a5/20260911T154748Z/data --output-dir .runtime/rt-a5/NEW-CONTINUATION --architecture rt --resume .runtime/rt-a5/ORIGINAL/train-rt/checkpoints/step-005000.pt --updates 10000 --wandb-group ORIGINAL-pilot'
```

Model dimensions, seeds, data identity and executed source/runtime must match
the checkpoint contract. A changed experiment is not an exact resume.
Continuations have separate W&B runs and local histories, starting after the
saved update. Do not count an interrupted tail and a replayed continuation
twice. The initial bounded RT fresh-process proof is in
[resume-proof.json](../.runtime/rt-a5/20260911T154748Z/resume-proof.json).

For additional development evaluation of a retained checkpoint, use
`python -m scripts.rt_a5_eval --checkpoint PATH --data-dir DATA --output-dir NEW
--role dev` or `--role ood_dev` inside the verified GPU container. It checks
checkpoint source/data identity and raw FP32 tensors before loading. The CLI
deliberately does not expose the final confirmation role in this milestone.

The 400,000-update, three-seed comparison and smaller-width ablations remain
subsequent work. The 10,000-update pilot is a bounded check of learning and
runtime, not an assessment of final convergence or a reproduced paper result.
