# Portable campaign components and next integration

The entrypoint for new semantics is `cdrm.pretrained.campaign_recipe`; existing
training entrypoints retain their historical defaults. These are components and
a bounded probe, **not yet a production campaign trainer**.

## Model and optimizer

`CampaignRecipe(arm)` supports B, N, F, R, NF, NR, FR and NFR. B is the ordinary
backbone, N adds NextLat, F adds four-pass FBT, R enables native RT at 0/15.
`build_campaign_model(already_loaded_backbone, recipe)` wraps caller-loaded
FP32 master weights without resetting them. Campaign RT executes on the first
pass too. The wrapper's dormant fusion is frozen when F is absent; count
resident parameters separately from active trainable parameters. No predictor
is created when N is absent. `recipe.to_dict()` and `recipe.sha256` pin choices.

`build_campaign_adamw(model, recipe, fused=...)` makes explicit component/decay
groups. Tied input/readout weights have one optimizer owner and are excluded
from decay; biases, scalars and normalization vectors are also excluded.
GPU training chooses fused=True; CPU fixtures deliberately choose False.
The outer execution configuration must retain that choice and all attention,
precision, checkpointing, graph and compile flags; recipe identity alone is not
a complete execution fingerprint. Actual source artifact hashes must also be
retained after verifying the pretrained download.

`CampaignTokenSchedule(optimizer, update_tokens, warmup_tokens=...,
start_fraction=...)` takes the ordered valid-presented-token count of every
planned **logical optimizer update**. Call `validate_next_update(count)` before
any backward/Adam work. Advance once after that optimizer update. Include
`checkpoint_contract()` in the outer checkpoint configuration so changed plans
fail before any restored object is mutated. Predeclare the desired horizon;
changing it on resume needs an explicit prefix-preserving plan fork, not an
unchecked scheduler edit. Plateau follows warmup; cooldown is future work.

## Data and jitter

See [data-contract.md](data-contract.md) for `CampaignData` and the bounded local
raw-JSONL/gzip preflight. Both ingest and tokenized-data manifest hashes belong in
the run/checkpoint configuration. Supplied source order is fixed; no corpus
mixture, shuffle, implicit cycling or production storage policy is chosen.

`feedback_noise_for_rows(recipe, window_keys, logical_update=...,
sequence_length=..., width=..., device=..., dtype=...,
physical_batch_size=...)` generates separate deterministic unit noise for each
row and feedback pass. The model applies its configured amplitude. Optional
physical batch size appends zero-noise dummy rows; real-row values are invariant
to rank partition and padding length. Keys must identify unique row occurrences
when a future data policy repeats documents. CPU noise generation is a reference
implementation; overlap/device-generation performance is not measured here.

Pass the tuple as `feedback_noise` alongside `mode=recipe.mode()` to eager
`loss_sums`/backbone execution. Do not put these tensors into serialized DDP
`backbone_kwargs`: a separate per-rank input channel is still needed. Evaluation
disables jitter and rejects supplied noise. Graph training rejects a nonzero
jitter mode until buffer refill and replay ownership are qualified.

## Bounded checks

CPU-container regression command:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_campaign_*.py'
```

Actual-checkpoint GPU smoke (new output directory for each attempt):

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi && timeout 900s python scripts/olmo_campaign_probe.py --output-dir .runtime/olmo-campaign-readiness/new-probe --length 16 --updates 2'
```

Requires the already verified native OLMo checkpoint in
`.runtime/olmo1b-step60000/artifacts` and authorized W&B credentials. No CPU
fallback or hidden download occurs. Review [protocol.md](protocol.md) for scope.

## Next work in dependency order

1. Integrate deterministic logical updates with right-padded Flash batches.
   A valid-prefix padding fast path must retain correct loss, RT and feedback
   validity. Preserve actual counts rather than averaging local loss means.
2. Add per-rank jitter input buffers and graph-safe changing masks/counts,
   then accumulated updates. Validate empty rank slots and zero local KL targets
   without skipping globally active predictor parameters or collective calls.
3. On two GPUs, compare a bounded accumulated update to its eager reference,
   then stop/relaunch fresh processes at a completed-update checkpoint and
   compare continuation. Qualify the actual final world size separately.
4. Measure representative K4 T1024 memory and useful throughput before LR
   calibration. Use the chosen hardware and actual data layout; this B1/T16
   semantic probe cannot provide production capacity estimates.
5. Prepare a disk-backed pinned Dolma manifest, generation/evaluation contract,
   and cooldown/SFT schedules before authorizing the larger campaign.

Some implementation and tiny reference checks in steps 1–2 can still be done
on one GPU; their distributed acceptance should happen early on multiple GPUs.
No H200 allocation or campaign launch is implied. Preserve 20–30 minute recovery
boundaries and warn before longer non-resumable work.

Historical training configurations serialized with `asdict(FBTMode)` lack the
two new fields. Reconstructing them with the current dataclass changes their
configuration hash and is rejected by strict resume. Use the retained exact
legacy configuration or an audited migration. Pretrained tensor loading is
unchanged; this campaign intentionally starts with fresh optimizer state.
