# A5 Transformer reference architecture comparison

## Approved scope

The user approved steps 1 and 2 of
[the consistency audit and plan](rt-a5-transformer-reference-audit-plan.md):
continue the original SEQ Transformer from 10,000 to **100,000 total updates**,
then train the released NextLat GPT architecture from scratch to **100,000**
through our existing A5 training and evaluation code. Pause to review both
before the proposed RoPE-only ablation. While the reference run was active,
the user subsequently authorized that ablation to **50,000 updates without
pausing**. It has its own frozen protocol in
`.runtime/rt-a5/20260911T210811Z-rope-only/`; pause after that follow-up instead.
An extension to 400,000 updates remains outside the approved scope. No
autonomous NextLat MLP recurrence is included.

The primary comparison is the fixed 100k endpoint. The 10k/25k/50k checkpoints
show learning progress, not an opportunity to select the best result. Existing
pure RT checkpoints provide matched-budget context without another RT run.
The previous RT+NextLat pilot remains a separate 10k experiment.

## Model and shared harness

Both models have two layers, width 512, eight heads, 60 operation/state IDs,
untied embedding and classifier, no dropout, and same-position supervision.

| Model detail | Original SEQ | Released GPT architecture |
| --- | --- | --- |
| Position encoding | ALiBi | RoPE, base 10000, source positions 1..T |
| Block/final normalization | LayerNorm | RMSNorm, epsilon 1e-5 |
| Q/K normalization | Enabled | Disabled |
| Feedforward | GELU, hidden width 2048 | SwiGLU, hidden width 1408 |
| Weight initialization | Existing Mitchell rule | Source Normal(0, 0.02) rule and construction order |
| Parameters | 6,357,504 | 6,486,528 |

The new model is a compact port of the model operations from NextLat commit
`b37d3411ab9b17be8638abbddb9529f0f3a0a5f9`, with MIT attribution in
`scripts/rt_a5_reference_gpt.py`. It returns `.logits` to our existing code.
It does not use the authors' trainer. Source-pinned model classes are loaded
only in bounded equivalence tests, which check initialization, logits,
gradients and an update using our AdamW path.

`scripts/rt_a5_reference_train.py` directly imports the historical `train_step`,
`evaluate_arrays`, `WordOrder`, `batch_tensors`, `make_optimizer`, `task_loss`,
and `fp32_context`. Tests check function identity. The new driver handles
model construction, tracking, and a separate checkpoint/source contract.
Historical executed source files are preserved to keep old resumes valid.

Both runs use the same frozen 800k training words, deterministic order seed
1234, batch 1024, length 12, AdamW with constant learning rate 1e-4, betas
(0.9, 0.95), epsilon 1e-8, matrix decay 0.01, vector decay zero, and global
gradient clipping at 1. The state loss is unshifted CE averaged over B*T.
Runtime is full FP32, math SDPA, TF32 disabled, eager, without compilation
or CUDA graphs. Initialization seed is 1234; distinct architecture families
are not claimed to have equal initial weights.

## Current lineage and execution

Lineage: `.runtime/rt-a5/20260911T201820Z-transformer-reference/`.
The frozen `protocol.json` and `approved-plan.md` precede training.

- Original data: `.runtime/rt-a5/20260911T154748Z/data`.
- SEQ parent: original pilot `train-seq/checkpoints/step-010000.pt`.
- New SEQ continuation: lineage `train-seq/`.
- New released-GPT control: lineage `train-reference-gpt/`.
- Existing pure RT 100k: `.runtime/rt-a5/20260911T171239Z-rt100k/train-rt/`.
- W&B project: <https://wandb.ai/taylorbollman/rt-a5-state-tracking>.
- W&B group: `20260911T201820Z-transformer-reference`.
- GCS prefix: `gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T201820Z-transformer-reference/`.

Never run GPU code in the host shell. Use `scripts/docker_shell.sh bash -lc`,
verify `/.dockerenv`, cwd `/workspace/cdrm-w-latent`, and `nvidia-smi` before
execution. CPU tests use `CDRM_DOCKER_GPUS=none` explicitly. This lineage was
started on an H100 80GB. No parallel training jobs are intended.

Run scripts, training logs, checkpoint metadata and source snapshots live in
the lineage. `history.jsonl` records every update; the main `report.json` is
refreshed at checkpoints and completion. Check the run exit code and final
status rather than treating a quiet launcher as successful completion.

## Evaluation and interpretation

Small development checks use 4,096 rows every 5k updates. Saved checkpoints
use 102,400 short development words and 102,400 independent length-36 words.
SEQ saves 25k/50k/100k; the reference saves 0/10k/25k/50k/100k. Independent
confirmation remains untouched.

- E(t): every predicted state through position t is correct.
- A(t): predicted state at position t is correct.
- M(t): mean token accuracy through position t.

The released A5 evaluator supports comparing Figure 10 to E(t). Display
full positions 1–36 and label the 10–18 view explicitly as a zoom. Do not
interpret elevated M(t) as accurate late states: correct early tokens can
raise this average when late predictions are near chance.

This control matches the released **architecture on our harness**. It is
not a literal reconstruction of the paper's 400k result: the current budget
is 100k, runtime is FP32, optimizer execution is our nonfused path, and data
ordering/splitting follow our existing code. Our long-word set excludes
training-prefix overlap. Preserve those conditions across both arms. The
authors' README recommends eager A5 execution on Hopper despite the YAML's
compile setting. A single seed can identify a large discrepancy but does
not establish a general architectural conclusion.

## Completion checklist

After both runs complete, generate `docs/reports/rt-a5/transformer-reference-100k/`
with the new standalone reporter. Check that overlapping update-order hashes,
data identity, optimizer settings, full evaluation counts, source snapshots,
finite endpoint model/Adam state and fixed endpoint selection agree. Retain
all new checkpoints and the bounded evidence archive in GCS with verified
checksums. Record final metrics and links here. The later authorized RoPE-only
run follows sequentially and pauses at 50k for user review.

Both SEQ and the reference GPT completed the approved 100k endpoints.
The combined 27 model/trainer CPU tests and
10 reporter tests passed. The disposable GPU check passed 75 updates at
D512/B1024/T12 and a B1024/T36 forward evaluation, with 25 warmup updates
excluded from timing. Reference updates take about 17 ms. Its initialization
checkpoint starts fresh after this disposable check.

Historical 43-file source SHA:
`6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea`.
Expanded 46-file source SHA:
`851d80ef1192a41d1b36ade785a96dcc1974ed554f5bbfc02f2dd9467409703c`.
The complete freeze receipt is `checks/source-freeze.json` in the lineage.

SEQ 100k: short development token accuracy 62.4946%, whole-word exactness
0.1514%; on length-36 words, E(12)=0.1504%, E(13)=0.0215%, E(36)=0%, and
M(36)=21.9672%. Its E≥50% horizon moved from position 5 at 10k to position 7
at 100k. This is improvement without solving the training length.

W&B training runs:
[SEQ continuation](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/wefqnb7i)
and [reference GPT](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/z6maptz5).

The [completed report](reports/rt-a5/transformer-reference-100k/report.md)
and [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/kgqggfl6)
contain the full 1–36 curves, explicit 10–18 zoom, E-only view, checkpoint
tables and training histories. At 100k:

| Backbone | Short dev token accuracy | Short dev whole-word exactness | OOD E(12) | OOD E(13) | Last OOD prefix with E≥50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Our Transformer | 62.4946% | 0.1514% | 0.1504% | 0.0215% | 7 |
| Authors' GPT architecture on our harness | 90.7859% | 5.8662% | 5.7686% | 0.1416% | 11 |
| Our RT, existing 100k checkpoint | 99.9826% | 99.9111% | 99.9199% | 48.7432% | 12 |

All three use the same 102,400-word evaluation subsets and all 100,000
per-update order hashes agree. The extended SEQ shifts its early boundary,
but the released architecture shifts it much further on the same harness.
Its drop is now in the 11–13 region. This is a qualitative movement toward
the figure's pattern, not a verified numerical reproduction of its 400k run.
Several architecture components changed together, so this result alone does
not identify RoPE as the cause. The separately authorized RoPE-only 50k
control is described in [its handoff](rt-a5-rope-only-handoff.md).

Stored-state audit passed for all 19 SEQ and 15 reference GPT parameter/Adam
states, including finite FP32 moments and exactly 100k optimizer counters.
All eight new checkpoints are retained with verified GCS checksums; final
archive receipts are `evidence-storage.json` and `evidence-storage-upload.json`
in the lineage. Preserve the closed training sources and generated reports.

## Shared repeated-prefix limitation

A bounded read-only diagnostic found a shared attention-only limitation.
For a word starting with the same nonidentity group element twice, `(g, g)`,
both positions start with the same embedding. ALiBi and RoPE modify attention
scores, while the values remain identical; their weighted average is the
same. With no BOS token or positional contribution to the values/residual,
both blocks therefore produce the same logits at these positions in exact
arithmetic. The correct states are `g` and `g²`, which differ for nonidentity
`g`. At least one must be wrong, so the prefix cannot be wholly correct.

There are exactly 1,706 such words among the evaluated 102,400 short dev
words and 1,692 in OOD dev. The corresponding E(2) bounds, 98.333984375% and
98.34765625%, exactly match the recorded values in both SEQ and the reference.
Two bounded untrained D512 CPU examples per architecture gave identical
argmax with position-logit differences at most 8.35e-7. This supports a
structural explanation; saved per-word predictions were not inspected, and
the bound describes exact arithmetic rather than exploiting roundoff.

This limitation is shared by the released no-BOS GPT architecture. It does
not explain the large later-position difference between the two Transformers.
It also means an attention-only reference need not reach literally 100% E(12)
to reproduce the intended learned-through-12 pattern. No model or data
change was made. See lineage `checks/repeated-prefix-structural-note.{md,json}`.
