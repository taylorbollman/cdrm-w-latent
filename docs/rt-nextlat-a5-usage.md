# RT + NextLat A5 implementation and continuation notes

This milestone trains a NextLat auxiliary predictor with the existing two-layer
RT, then trains an ordinary Transformer with the same auxiliary objective.
**All task accuracy comes from the backbone.** There is no autonomous MLP
latent recurrence implementation or evaluation. The user explicitly excluded
that direction on 2026-09-11.

The approved scientific specification is [the plan](rt-nextlat-a5-plan.md).
The fixed execution lineage is `.runtime/rt-a5/20260911T191702Z-nextlat/`.
The primary comparison is at **10,000 updates per arm**, against the saved RT
and ordinary Transformer at 10,000 updates. Pure RT at 100,000 is longer-run
context, not the equal-budget control. Stop after this pilot and report;
do not extend any run automatically.

## Model and objective

Both backbones retain width 512, eight heads, FFN 2048 with GELU, ALiBi, the
existing normalization, and separate 60-row input and state-output matrices.
Both RT blocks recur with tiled rho=1. Each backbone has 6,357,504 parameters.

The predictor receives the next observed operation embedding followed by the
current **post-final-LayerNorm** latent. RMSNorm with epsilon 1e-5 and a
three-linear-layer GELU MLP predict a residual added to the current latent.
At D512 its hidden width is 512, following the released A5 code rather than
the paper's Table 5 width 1024. It adds 1,049,600 parameters; total training
parameters are 7,407,104. Only the 6,357,504 backbone parameters run at inference.

For each length-12 word, optimize ordinary same-position state CE plus
SmoothL1(beta=1) over the 11 adjacent latent transitions, with coefficient 1.
Average the latter over batch, transitions **and latent coordinates**. Stop
the target-role gradient only; source latents and conditioning embeddings
remain attached. No EMA encoder, KL training loss, auxiliary hard-label CE,
BOS, EOS masking, cross-word transition, or next-input-token LM loss is used.
Identity operation 0 remains a valid token. One complete backward, one global
clip at 1, and one AdamW step update all shared parameters once.

One-step predicted-state CE/KL and latent scale/error statistics are detached
diagnostics. They do not contribute additional training objectives or carry
predicted states to later positions.

## Code and checks

- `scripts/rt_a5_nextlat.py`: predictor, backbone-only wrapper, combined loss.
- `configs/rt_a5_nextlat/base.json`: pinned released predictor/objective recipe.
- `scripts/rt_a5_nextlat_train.py`: paired trainer and strict hybrid resume schema.
- `scripts/rt_a5_nextlat_validate.py`: bounded GPU checks, timing and overfit.
- `scripts/rt_a5_nextlat_report.py`: four-backbone comparison from saved evidence.
- `tests/test_rt_a5_nextlat.py` and `tests/test_rt_a5_nextlat_train.py`: 33 focused
  CPU checks passed together in the GPU-disabled project container.
- `tests/test_rt_a5_nextlat_report.py`: 14 additional checks cover report
  provenance, counts, comparison contracts and standalone plots.

The small GPU combined-loss oracle passed the inherited FP32 comparison
against naive RT. Setting the auxiliary coefficient to 0 exactly reproduced
baseline logits, CE, shared gradients and one Adam update for both backbones.
A predictor pre-hook that raises verified that ordinary evaluation bypasses
the predictor. The B1024/T12/D512 timing fixtures had finite FP32 parameter,
gradient and Adam state. Both D128 models memorized 32 fixed length-4 words in
the discarded overfit fixture. A fresh process resuming update 1 to 3 exactly
matched uninterrupted 3 for model, Adam, RNG, contract, counters and word order.

These are bounded integration checks. They do not reopen the prior BF16
qualification or claim exhaustive numerical equivalence at trained weights.

## Runtime and invocation

Use the project Docker launcher for model work. Verify the container and GPU
before CUDA execution. The pilots use full FP32 with TF32, autocast, compile
and CUDA graphs disabled, matching the saved A5 baselines. GPU: H100 80GB.

```bash
bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent &&
  nvidia-smi -L &&
  python -m scripts.rt_a5_nextlat_train \
    --data-dir .runtime/rt-a5/20260911T154748Z/data \
    --output-dir <fresh-output-directory> \
    --architecture rt \
    --updates 10000 \
    --wandb-group <new-group> \
    --wandb-run-name <new-name>
'
```

`--architecture seq` selects the ordinary Transformer control. Defaults are
D512, backbone/data-order seed 1234, predictor seed 1235, physical batch 1024,
constant AdamW LR 1e-4, betas (.9,.95), epsilon 1e-8, matrix decay .01 and zero
vector decay. The predictor has its own CPU RNG scope; constructing it does
not perturb canonical backbone initialization or shuffled word order.

Small development checks use 4096 words every 500 updates. Trained checkpoints
at 1k/5k/10k evaluate 102,400 words separately at length-12 and length-36. Initial
checkpoint 0 is also retained. The old baseline's 1k evaluation used 4,096 words,
so report its different denominator explicitly; 5k and10k are fully matched.
Training uses the existing 800k split. Independent confirmation stays untouched.

For a validation CLI, invoke `python scripts/rt_a5_nextlat_validate.py` by
filename so the unchanged historical validation helper's imports resolve.
Its modes are `checks`, `profile`, and `overfit`; each requires a fresh
`--output-dir` and `--wandb-group`. Their updates are discarded fixtures.
CPU tests run via the same launcher with `CDRM_DOCKER_GPUS=none`.

## Resume and evidence preservation

Hybrid checkpoints use schema `rt-a5-nextlat-training-v1` and store backbone,
predictor, named optimizer mapping, Adam state, RNG, word-order chain and the
complete source/data/model/objective/runtime contract. Resume into a **fresh**
output directory with `--resume <checkpoint>` and a larger absolute endpoint.
This syntax documents capability; extending the current pilot requires a new
user instruction.

Keep historical `scripts/rt_a5_*.py` execution sources, the OLMo tree and
`configs/rt_a5/*.json` unchanged. Adding JSON to that old directory also changes
the historical 43-file resume identity. New code belongs in separate files
and `configs/rt_a5_nextlat/`. The old source hash remains
`6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea`.
The frozen new training hash is
`1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961`.

One future configuration issue is deliberately outside this weight-1 pilot:
before a non-default auxiliary-weight sweep, resolve the nested base recipe's
weight metadata as well as `contract.objective.latent_weight`. Both are 1 in
the executed protocol. Do not change frozen executed files merely to prepare
an unrequested sweep.

W&B entity is `taylorbollman`, project `rt-a5-state-tracking`, group
`20260911T191702Z-nextlat`. Graphs use backbone metrics and separate one-step
diagnostic namespaces. No credentials should be copied into reports or archives.

Persistent cloud lineage:
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T191702Z-nextlat/`.
The lineage's `retain.py` records verified checkpoint and archive receipts.
Retain source snapshots, protocol, reports, plots, resume evidence and full
primary checkpoints. The original data is referenced by its manifest and
existing retained lineage rather than copied into each new run.

Read the final [pilot report](reports/rt-a5/nextlat-pilot/report.md) and lineage
evidence before deciding on any longer run. One seed at 10k does not establish
reproducibility or reproduce the paper's 400k-update budget.
