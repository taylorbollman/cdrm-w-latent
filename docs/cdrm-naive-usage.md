The active workflow screens harder official MAD settings with **SEQ6 first**.
Run CDRM6 only on a selected setting where SEQ6 has not reliably aced the early
screen. The completed SEQ12 recall run is historical calibration; the proposed
CDRM12 research pair was superseded before a CDRM12 SYN run launched.

The completed seed-0/1 screens selected `copy-v16-t256-k96` for the first
CDRM6 comparison at 25 common epochs. Recall V128 remains deferred. Both copying
seeds retain learning headroom; neither satisfies the early-ace criterion. The
[frozen selection](reports/cdrm-naive/six-comparison-selection.json) records the
development evidence and original continuation policy. The later [runtime-only update](reports/cdrm-naive/six-continuation-policy-update.json)
adds common epoch 45 between 50 and the no-continuation endpoint 25. After all
four epoch-25 runs completed, the [immutable decision](reports/cdrm-naive/six-continuation-decision.json)
selected **45** at 14:53:44 UTC: 4,462.77 seconds including margin fits 5,085.70
seconds remaining; epoch 50 needs 5,561.22 seconds. All four serial continuations
completed epoch 45 (4,500 updates each) at 15:57:13 UTC. Six unique final
evaluations completed at 16:01:32 UTC, covering all eight roles in the
[frozen checkpoint-role plan](reports/cdrm-naive/final-evaluation-plan.json).
The [results](reports/cdrm-naive/results.md) report the mixed outcomes. The
[budget decision](reports/cdrm-naive/six-comparison-budget.json) freezes the
selected B128 CDRM launch after NUM and exact-recovery checks. Actual jobs are
`compare6-copy96-cdrm-s0-e25` and `compare6-copy96-cdrm-s1-e25`; the generic commands
below use fresh names to avoid reusing those output directories.

The [six-block manifest](../configs/cdrm/joint_d128_6_screening_first.json) resolves
D128/H16/MLP512 with early site 1 and late site 3. CDRM runs ordinary blocks 0–3,
uses post-block-1 and post-block-3 previews in one side scan sharing block 1's
canonical parameters, then bridges into ordinary block 4. Blocks 4–5 form the
suffix. The [profile catalog](../configs/cdrm/README.md) records exact details.

From the host checkout, enter the required GPU container:

```bash
bash scripts/docker_shell.sh
```

Inside it, verify the environment and select one retained setting:

```bash
test -f /.dockerenv
test "$PWD" = /workspace/cdrm-w-latent
nvidia-smi
export CDRM_SETTING=copy-v16-t256-k96
export CDRM_TASK=selective-copying
export CDRM_DATA_ROOT=.runtime/cdrm-naive/20260907T123830Z/data-screening-v1/$CDRM_SETTING
export CDRM_RUN_ROOT=.runtime/cdrm-naive/screening-$(date -u +%Y%m%dT%H%M%SZ)
```

| Setting | Task argument | Official one-factor change | Actual input length | Vocabulary |
|---|---|---|---:|---:|
| `recall-v128-t128` | `in-context-recall` | Vocabulary 16 → 128 | 127 | 128 |
| `copy-v16-t256-k96` | `selective-copying` | Tokens to copy 16 → 96 | 256 | 16 |

For the deferred recall setting, change both setting/task variables and update the data root. Each
root's `manifest.json` retains the resolved setting and official provenance.
Recall's configured length is 128; the native generator returns 127 positions.
No padding is added. Each setting has 12,800 fixed training examples and separate
development set of 1,280 and a planned final set of 1,280. At screening and
comparison launch, all harder-setting final arrays were absent. The selected
copying final split was generated only after the common endpoint and checkpoint
roles were frozen; its `manifest.added-final.json` preserves the original
train/dev manifest. Deferred settings still have no final arrays. For a new
screen, generate final data only after selection is frozen. New lengths or difficulty changes require a new
audited corpus. `max_sequence_length=512` is bias capacity, not input length.

Validate each newly requested topology/shape, then measure physical-batch cost:

```bash
python scripts/cdrm_validate.py --mode validate --preset d128_6 --topology seq \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 2 \
  --output-dir "$CDRM_RUN_ROOT/num6-seq-b2"
python scripts/cdrm_validate.py --mode benchmark --preset d128_6 --topology seq \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 128 \
  --warmup 5 --steps 20 --output-dir "$CDRM_RUN_ROOT/benchmark6-seq-b128"
```

Use `--topology cdrm` for its corresponding checks before a selected comparison.
These real loss/backward/clipping/Adam probes reuse a training batch; they supply
NUM/OPS evidence, not research learning. B128 is the declared physical batch.
Record a common smaller batch if a new shape does not fit; accumulation remains
unsupported. Output directories must be new. The D256/H4/custom-MQAR NUM profile
remains available separately with `--preset d256` and no MAD `--data-root`; the
current training CLI does not train CDRM on that custom MQAR stream.

Set an explicit future timezone-aware ISO8601 deadline; override the runner's
original session-specific default in a later session. Screen training seed 0:

```bash
: "${CDRM_DEADLINE_UTC:?Set the explicit UTC cutoff for this session}"
export CDRM_SCREEN_SEED=0
python scripts/cdrm_train.py --topology seq --preset d128_6 \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 128 \
  --stop-epochs 25 --seed "$CDRM_SCREEN_SEED" --shuffle-seed 45678 \
  --run-purpose screening --screen-early-ace \
  --ace-token-accuracy 0.999 --ace-exact-match 0.99 \
  --ace-consecutive-epochs 3 --ace-by-epoch 10 \
  --deadline-utc "$CDRM_DEADLINE_UTC" \
  --output-dir "$CDRM_RUN_ROOT/screen6-$CDRM_SETTING-seq-s$CDRM_SCREEN_SEED"
```

The early-ace rule requires development token accuracy ≥0.999 and direct
whole-example exact match ≥0.99 for three consecutive completed epochs, reached
by epoch 10. An early stop has `status=complete`, `termination_reason=early_ace`
and its actual epoch/update counters. Otherwise the screen reaches 25 epochs or
pauses at the wall limit. This session ran both seeds for every setting. A
future screen may repeat seed 1 conditionally when seed 0 aces early, but must
do so before classifying the setting reliably easy. **Skip CDRM on a setting SEQ6
reliably aces early.** One successful seed is not replication evidence.

Choose a harder useful setting from development behavior and cost. Freeze its
comparison endpoint before model final-set access. A compatible retained SEQ6
baseline may be reused or continued. For the selected setting only, run seeds 0 and 1 independently with their
matching retained SEQ6 baselines. The first selected endpoint is 25; set 50 or 45 only
after the common continuation decision. A fresh CDRM run uses the fresh paired
backbone seed, not trained SEQ weights:

```bash
export CDRM_COMPARE_EPOCHS=25
export CDRM_COMPARE_SEED=0
python scripts/cdrm_train.py --topology cdrm --preset d128_6 \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 128 \
  --stop-epochs "$CDRM_COMPARE_EPOCHS" --seed "$CDRM_COMPARE_SEED" --shuffle-seed 45678 \
  --run-purpose comparison --deadline-utc "$CDRM_DEADLINE_UTC" \
  --output-dir "$CDRM_RUN_ROOT/compare6-$CDRM_SETTING-cdrm-s$CDRM_COMPARE_SEED"
```

Pair depth, setting/corpus, initial backbone, seed, physical batch, ordered data
and completed epochs/updates. R3 and active same-depth are prepared deferred
attribution arms, not an automatic queue. They require applicable shape, memory
and recovery checks. Their topology names are `r3` and `same_depth`.

All arms use FP32, math attention, autocast/TF32 off, dropout 0 and no accumulation.
AdamW uses LR `5e-4`, betas `(0.9,0.98)`, epsilon `1e-8`, weight decay 0 and clip
norm 1. Cosine scheduling steps after each completed epoch over **200 epochs**,
to minimum LR `1e-6`, without warmup. Early stops do not compress the horizon.
B128 gives 100 updates per epoch. Shuffle seed 45678 reproduces the archived
permutations; no new examples are generated during training.

Native MAD labels are already aligned; no additional shift is applied. Recall's
training loss is dense next-token CE, including unpredictable next keys and first
presentations of random values. Its separate answer mask measures repeated-key
retrieval. Their CE difference is **not an ordinary train/dev generalization
gap**. Recall uses teacher-forced previous values, not free generation. Copying
uses masked output targets and blank input tokens in its output region. Metrics
are answer CE, accuracy over scored positions, and direct exact match over each
example's complete answer mask.

For separate repeated-batch OPS, add `--train-examples` equal to the physical batch
and `--ops-development-from-train`, with a fresh output path. The following 32
training examples become the monitor instead of research dev. A larger declared
subset tests distinct minibatches. Neither diagnostic establishes generalization.

Runs retain `resolved-config.json`, `learning-curve.jsonl`, `development.jsonl`,
`report.json` and `checkpoint-ledger.json`. The ledger maps initial, milestone,
latest and best-development roles to files/hashes. Names include
`initial-weights.pt`, `epoch-0025.pt` and `paused-uNNNNNNN.pt`; roles can share a
file. Do not assume `latest.pt` or `best.pt` filenames exist.

For this session, the controller prepared the immutable common-45
decision and completed all four continuations. **Do not rerun the following completed
commands into the same decision or output paths.** They document the two distinct
steps used from the host project root after all four epoch-25 reports completed:

```bash
python3 .runtime/cdrm-naive/20260907T123830Z/execution/six-continuation-queue.py --prepare
python3 .runtime/cdrm-naive/20260907T123830Z/execution/six-continuation-queue.py --execute
```

Preparation runs no model or data generation. It estimates each candidate from
all four original full-run wall durations, adds 60 seconds and a 15% margin, and
uses the original 16:18:30 UTC training cutoff. Execution rechecks source/data/
checkpoint hashes and remaining time, refuses occupied output directories, and
delegates each GPU job through the required container launcher. The selected
epoch-25 fallback launches nothing. The controller retains its decision and
execution receipt in the session root. It refuses silent restart of a partial
queue; retained partial work needs a separate explicit recovery decision.

For a later separately authorized session, to continue a compatible screened
SEQ6 baseline, select a later endpoint and a retained nonzero-update checkpoint. Use a new directory and omit early stopping:

```bash
: "${CDRM_RESUME_CHECKPOINT:?Choose the exact retained nonzero-update checkpoint}"
python scripts/cdrm_train.py --topology seq --preset d128_6 \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 128 \
  --stop-epochs "$CDRM_COMPARE_EPOCHS" --seed "$CDRM_COMPARE_SEED" --shuffle-seed 45678 \
  --run-purpose comparison --deadline-utc "$CDRM_DEADLINE_UTC" \
  --resume "$CDRM_RESUME_CHECKPOINT" \
  --output-dir "$CDRM_RUN_ROOT/continued6-$CDRM_SETTING-seq-s$CDRM_COMPARE_SEED"
```

Match the topology/seed of any other checkpoint. The runner checks model,
optimizer, scheduler, RNG, source/config and sampler identity. Stopping purpose
is outside that computational identity. Initial artifacts are weights-only;
update-zero exact resume is rejected. An immediate pause after restoring trained
state may have zero new gradients: it is bookkeeping evidence, not a failed
backward or a new update. Side memory is rebuilt from each input.

For example, this command would continue the retained CDRM seed-0 epoch-45
checkpoint to epoch 50 in a later session. It was **not executed** in this pilot.
Set a new deadline and a new output root using the variables above, and preserve
the archived computational source identity:

```bash
python scripts/cdrm_train.py --topology cdrm --preset d128_6 \
  --task selective-copying \
  --data-root .runtime/cdrm-naive/20260907T123830Z/data-screening-v1/copy-v16-t256-k96 \
  --batch 128 --stop-epochs 50 --seed 0 --shuffle-seed 45678 --monitor-every 100 \
  --run-purpose comparison --deadline-utc "$CDRM_DEADLINE_UTC" \
  --resume .runtime/cdrm-naive/20260907T123830Z/continue6-copy96-cdrm-s0-e45/epoch-0045.pt \
  --output-dir "$CDRM_RUN_ROOT/continued6-copy96-cdrm-s0-e50"
```

The corresponding seed-1 checkpoint replaces `s0` with `s1` and requires
`--seed 1`; SEQ uses the analogous `continue6-copy96-seq-*` paths and
`--topology seq`. A future comparison must reach a common endpoint in its
matched arms. After this pilot's final evaluation, use fresh confirmatory data
for claims arising from further experiment choices.

After all choices are fixed, explicitly evaluate the selected endpoint and
best-development checkpoint, matching original identity arguments:

```bash
: "${CDRM_EVAL_CHECKPOINT:?Choose the declared endpoint or best-development checkpoint}"
python scripts/cdrm_train.py --mode evaluate --topology cdrm --preset d128_6 \
  --task "$CDRM_TASK" --data-root "$CDRM_DATA_ROOT" --batch 128 \
  --seed "$CDRM_COMPARE_SEED" --shuffle-seed 45678 --run-purpose comparison \
  --checkpoint "$CDRM_EVAL_CHECKPOINT" --split final \
  --output-dir "$CDRM_RUN_ROOT/evaluation6-$CDRM_SETTING-cdrm-s$CDRM_COMPARE_SEED-endpoint"
```

From the host checkout, launch the report generator in an explicit CPU
container. It never evaluates a model. It excludes active/failed records from result curves and keeps unpaired
screens distinct from architecture comparisons:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'python scripts/cdrm_report.py --run-root .runtime/cdrm-naive/20260907T123830Z \
    --output-dir docs/reports/cdrm-naive --historical-depth 12 --screening-depth 6'
```

Resolved JSONs are complete `ModelConfig` snapshots for direct Python construction;
the CLI selects `--preset`/`--topology` and has no `--config` or gate override flag.
For prospective standalone construction inside the project container, load an
existing six-block profile and set the gates before constructing the model:

```python
import json
from pathlib import Path

from olmo.config import ModelConfig
from olmo.model import OLMo

raw = json.loads(
    Path("configs/cdrm/d128_6_copy_v16_t256_k96_cdrm.json").read_text()
)
raw.update(cdrm_epsilon=0.1, cdrm_rho=1.0, cdrm_lambda=0.01)
config = ModelConfig(**raw)
config.validate_cdrm()  # Checks supported topology, finite gates and rho in [0, 1].
model = OLMo(config)    # This profile constructs on CPU; no forward or training here.
```

Edit those three values for a new gate experiment. A changed computational
configuration starts a new experiment and cannot use exact resume from the old
configuration. This standalone constructor does not establish paired backbone
initialization; the research CLI above preserves the matched SEQ/CDRM
initialization protocol. The example is prospective and was not executed.

Gates are epsilon 0.1, rho 1 and lambda 0.01. The deep candidate is
`p_early + epsilon*Ad(Nd(p_late-p_early))`; the active same-depth candidate uses
`Nd(p_early)`. Both bridge via `p_late + lambda*Ab(Nb(hat_m-p_early))`, using
proposed `hat_m`, not interpolated `m`. Rho 0 leaves an active side branch;
lambda 0 is the matched SEQ bypass. Replacing the late preview with the early
preview would zero the difference input and is a separate removal ablation.

For weights-only loading, instantiate the exact saved
`checkpoint["identity"]["model_config"]`, construct `OLMo`, and call
`load_state_dict(checkpoint["model"], strict=True)`. The numerical validator's
one-update packet stores its configuration under `config`. Manual weight loading
does not restore optimizer/RNG/data state. A trained R3 checkpoint is not a paired
ordinary-preview initialization.

Request `output_cdrm_states=True` for graph-connected diagnostics. Compatibility
names `p3`, `p8` and `v8` mean the **configured** early preview, late preview and
bridge output. In the six-block profile they are post-block 1, post-block 3 and
the input of block 4; ordinary hidden-state index 4 is the bridge. The historical
twelve-block indices are 3, 8 and 9. Per-token `records`, `permanent_k`,
`permanent_v` and `read_outputs` expose actual temporal nodes; an unused terminal
write can have no gradient. Detach summaries and release diagnostic outputs.

Padding/custom masks, cached decoding, packing, GQA, RoPE, activation checkpointing
and non-FP32/autocast are outside this reference. Use actual timing and CUDA peak
memory; inherited OLMo FLOP estimates omit the side scan. Actual clearance,
executed screens and selected comparisons belong to session reports, not these
prospective commands.
