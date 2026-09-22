# OLMo O4: bounded Python continuation and retention pilot

Recorded 2026-09-21 before real-data GPU preflight or comparative training.
The user reviewed O3 and authorized continuing. This is the first bounded
learning comparison under the v3 plan, with FBT off. It is a recovery/direction
pilot of approximately 20–22M valid input tokens per arm, not the proposed
100M/250M efficacy stage or an FBT reproduction.

## Fixed backbone and four arms

Every arm starts independently from original OLMo-1B step60000 (~252B tokens),
revision `81b71efbce6f4dada57c94860301af4298bcd351`, checkpoint SHA256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Preserve the native 16-layer/D2048 model, full MHA, LayerNorm, RoPE, tied full
50,304-row readout, and absence of Q/K normalization. RT selects layer 0 only.

The arms are ordinary, ordinary+NextLat, RT, RT+NextLat. No arm starts from
another arm's adapted weights. NextLat uses the O3 pinned 1B horizon-one recipe:
factor1.6/hidden6528, latent SmoothL1 and detached-teacher/readout KL coefficients
1/1, no auxiliary token CE, isolated predictor seed20260921, no autonomous
predictor rollout. This adds 82,726,912 training-only parameters. Do not change
auxiliary weights alongside the RT ramp in this pilot.

## Data and masking

- CodeSearchNet Python: `code-search-net/code_search_net` revision
  `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`, official train/validation/test
  Parquet files. Use unchanged `func_code_string`, including its original
  docstring, as code continuation rather than inventing an instruction template.
- General-language retention: `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1` validation/test.
  Reconstruct articles from top-level headings; never train on WikiText.
- Verify pinned raw SHA256/size, native tokenizer bytes, and prepared artifacts.
  Deduplicate identical text/tokens with held-out precedence; check CodeSearchNet
  repository separation. Order documents deterministically using seed20260922.
  Prepare at least25M unique training CE targets, without cycling or reuse.
- Native tokenizer with no implicit special tokens; append one explicit EOS
  only at the actual document end. T512 windows use stride511, so one token
  overlaps as context without duplicating CE targets. Reset model state at every
  window. This omits one KL triple at each internal cut, recorded in the manifest.
- One document/window per padded row. No packed-document attention or cached
  cross-window training. CE, latent and KL cover all valid same-document targets;
  padding does not contribute to losses or exposure. Input-token counters include
  the explicit overlapping context tokens; supervised counters are separate.
- Fixed development prefix: first128 prepared windows for curves and first512
  for preflight/final comparison, capped by available rows. Selection precedes
  model scores. Official test splits are prepared/reserved but not evaluated.

Sources: [CodeSearchNet data/splits](https://github.com/github/CodeSearchNet#data),
[repository-specific code licenses](https://github.com/github/CodeSearchNet#licenses),
[pinned code dataset card](https://huggingface.co/datasets/code-search-net/code_search_net/blob/bd0cf261e357a3eb5c8fba490d23ec1a1cd59555/README.md),
[pinned WikiText card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md).
Dataset licenses/provenance are retained; code is treated as text and never
executed. Historical overlap with OLMo pretraining remains unknown. This is
not a clean-unseen programming benchmark or published WikiText perplexity.

## Preflight and frozen exposure schedule

On the actual H100, profile RT+NextLat at alpha1 with real full-length training
windows, B16 thenB32, BF16 mixed, default ordinary SDPA, mixed RT attention,
position chunk128, complete zero-LR AdamW steps (two warmups/three measurements).
Choose the largest passing point below55GiB allocated peak. An OOM is retained
and permits using the smaller passing candidate, not a silent CPU fallback.
Weights must remain unchanged. Measure ordinary, RT-alpha0 and RT-alpha1
development/retention baselines before training, including token accuracy.
If ordinary code capability is effectively at chance or evaluation is invalid,
resolve that before comparative runs.

Freeze the resulting physical batch and one common ordered sequence of updates:

1. First100 optimizer updates: RT alpha0, shared LR warmup.
2. Ramp: at least10M additional valid input tokens **and**200 updates. Before
   each update, alpha is the minimum of completed token/update fractions over
   that phase's frozen actual budgets. Both RT arms use the identical schedule.
3. Alpha1: at least the ramp's actual tokens **and**updates, including its
   one-batch boundary overshoot. Ordinary arms see exactly the same updates/data.

The prepared length vector fixes all phase boundaries before learning. Keep
valid input tokens, CE targets, latent pairs, KL triples, documents/windows,
updates and wall time separately. Equal data exposure is the primary comparison;
the pilot is not compute-matched. No batch-size/LR changes between arms.

Shared optimizer: fresh AdamW, LR1e-5, betas(.9,.95), epsilon1e-8, matrix weight
decay.1, gradient clipping1, first100-update linear LR warmup then constant LR.
FP32 parameters/AdamW moments with BF16 autocast; no compile, CUDA graphs or
distributed wrapper. A conservative LR and separate alpha-zero warmup address
the random NextLat head's initial perturbation without adding uncounted head-only
training or a changing KL coefficient.

## Evaluation, health and recovery

Primary: full-vocabulary held-out code mean token NLL, perplexity and next-token
accuracy. Secondary: same metrics on untouched WikiText development articles.
Use summed token NLL / actual target count; no mean-of-batch-means. Evaluation
uses only the backbone, no predictor. Restore all caller train/eval flags.
Record per-document rows at final512 evaluation for paired analysis; windows
from the same original document are not independent samples.

Evaluate the small128-window set initially, at update50, warmup/ramp/final
boundaries, and each1M-input-token threshold. Log every update's CE/latent/KL,
gradient norm before clipping, LR, alpha, token counters, wall time and VRAM
online to `taylorbollman/pretrained-fbt-rt-nextlat`. Final512 results have separate
metric names from the128-window curves.

Reject nonfinite losses/gradients before optimizer stepping. At saved boundaries,
check model weights and AdamW moments are finite. Save and halt an arm if **both**
code and retention NLL exceed its own initial128-window baseline by1.5nats at
two consecutive evaluations. This is a coarse catastrophic-health stop, not a
selection criterion favoring an arm. Normal retention/domain tradeoffs and
temporary conversion cost are reported rather than hidden. A failure stops the
queue for diagnosis; do not skip an arm or silently retune it.

Save full optimizer checkpoints at warmup/ramp/final boundaries and every30min
when needed, plus an explicit stop boundary. Store configuration, source/data
fingerprints, schedule, next-window cursor, counters, scheduler and RNG. Upload
each checkpoint to immutable `gs://fast-chunks` objects and verify generation,
size, server MD5 and SHA256 metadata. Keep the latest local full checkpoint;
remove an older local copy only after both it and its successor are retained.
Resume the same model/optimizer/data lineage. No checkpoint is serialized
asynchronously while its live weights change.

O3 supports the bounded gradient and platform basis, not a long-run BF16 equality
claim. This pilot supplies new evidence about actual updates. No broad numerical
campaign, architecture change, FBT, final-test tuning, autonomous latent rollout,
or multi-GPU validation is included. Stop after the four matched pilot endpoints
and report directional results/limitations before extending exposure.

## Preflight resolution and pre-training source amendment

Preflight `o4-preflight-01` passed. Selected B32: full-length real training
windows with NextLat/RT alpha1 peaked at45.99GiB allocated and2.249s/update.
The frozen schedule has2,634 updates and20,855,799 valid input tokens/arm:
100-update warmup (805,481 tokens),1,267-update ramp (10,002,783 tokens), then
1,267 alpha-one updates (10,047,535 tokens). Ramp ends after update1,367.
Schedule SHA256 `9c2b8343a1c6c8545051b15b033eb3c7f539386a64be13b5e6aa13ccf79a746c`.

Before any learning arm, review added one recovery-only startup check to
`scripts/olmo_o4_train.py`: retry cloud retention when resuming a nonfinal
checkpoint whose upload had failed. Preflight was already running and never
executed this training-resume branch. Numerical/data/profiling behavior did not
change, so it was not rerun. The authentic preflight report/source hashes remain
unchanged; the old script and `configuration.pre-retention-retry.json` are
retained alongside it. Final `configuration.json` records the exact old/new
hashes and this `provenance_amendment`; all four training arms enforce the same
final source inventory. Final configuration SHA256:
`6b7e9168f2aa1afc495d28cfdd84006e5fa3409f66fcc70c17c0fa83c8ff2e20`.
