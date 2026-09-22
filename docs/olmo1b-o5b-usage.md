# OLMo O5b: running, recovery and reporting

This is the operational guide for the bounded, paired FBT learning pilot. The
preflight passed and selected physical16 accumulated twice (49.60GiB), retaining
effective batch32. **Learning results are pending.** Read the runtime
`report.json` and `queue.json` for status. The [frozen protocol](reports/olmo1b-o5b/protocol.md)
defines the experiment; the [handoff](fbt-rt-nextlat-handoff.md) records current
authorization. [O5a usage](olmo1b-fbt-usage.md) covers the model semantics and
bounded correctness evidence.

## Recipe and evidence locations

Both arms start independently from native OLMo-1B `step60000-tokens252B` and the
same fusion initialization. No O4-adapted weights are loaded. The queue runs
`ordinary`, then `fbt`. Both explicitly execute two shared-weight passes and
optimize **CE(pass 0) + CE(pass 1)** (`K=2`, `gamma=1`). Ordinary keeps feedback
`beta=0`; FBT ramps beta to one. The ordinary objective is therefore twice its
single-pass CE, providing the matching control. RT and NextLat are off in both
arms; no prefix sampling, hidden jitter, compilation or CUDA graphs are used.

The data are the unchanged O4 CodeSearchNet Python stream, with one independent
document window per row and maximum length 512. Effective batch is 32. Preflight
tests complete beta-one optimizer steps with zero LR and chooses physical batch
32 below 60 GiB allocated, otherwise physical batch 16 with two accumulated
microbatches. It must pass before comparative learning. BF16 mixed computation
uses FP32 master weights/AdamW moments, ordinary SDPA and TF32 off.

Native LR warms over 100 updates to `1e-5`, while beta remains zero. The FBT beta
ramp then takes at least 10M valid input tokens **and** 200 updates. Update 101
still uses beta zero; update 102 is the first potentially active fusion update.
Fusion LR starts there at `1e-6` and reaches `1e-4` at update 201. There is no
separate fusion warm start: beta zero provides no fusion gradients. The final
beta-one phase matches at least the ramp's actual token and update exposure.
Historical `alpha` fields in the schedule are interpreted as **FBT beta**.

The fixed prepared stream resolves to 2,634 updates, 20,855,799 valid input
tokens and 20,771,511 CE targets **per pass, per arm**. Data exposure is counted
once; two passes account for 41,711,598 stack-input-token positions. This is a
frozen endpoint, not permission to extend automatically.

Paths relative to the project root:

| Purpose | Path |
| --- | --- |
| Original checkpoint/tokenizer | `.runtime/olmo1b-step60000/artifacts/` |
| Existing prepared corpus | `.runtime/olmo1b-step60000/o4-data-01/prepared/` |
| Preflight/configuration | `.runtime/olmo1b-step60000/o5b-preflight-01/` |
| Queue and arm evidence | `.runtime/olmo1b-step60000/o5b-pilot-01/` |
| Reports and protocol | `docs/reports/olmo1b-o5b/` |

The queue directory contains `queue.json`, `<arm>.log`, and each arm's
`report.json`, `events.jsonl`, checkpoint receipts and latest local full
checkpoint. W&B links are recorded in reports under
[`taylorbollman/pretrained-fbt-rt-nextlat`](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat).
Do not print credentials or environment-file contents.

## Preflight and initial retention

From `/home/taylorbollman/cdrm-w-latent` on the host, enter the project container:

```bash
bash scripts/docker_shell.sh bash
```

Inside it, verify `pwd` is `/workspace/cdrm-w-latent` and `nvidia-smi` succeeds.
All GPU commands below run **inside that shell**, never directly on the host.
For a new pilot, replace the timestamp placeholder once with an actual UTC
stamp such as `YYYYMMDDTHHMMSSZ`; preserve that exact prefix for all arms,
restarts and retention phases. For this already-started pilot, reuse its recorded
prefix rather than inventing a new one.

```bash
pwd
nvidia-smi
O5B_PREFIX='gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/<UTC_TIMESTAMP>'

python scripts/olmo_o5b_preflight.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --output-dir .runtime/olmo1b-step60000/o5b-preflight-01
```

This command is the initial launch, **not** a restart command: preflight requires
a new output directory and the current preflight must not be launched twice.
After it reports `passed`, `configuration.json` freezes the source/data hashes,
batch selection, schedules and modes. Keep sources and protocol unchanged.

Retain initial evidence after preflight passes, before starting the queue:

```bash
env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_o5b_retain.py \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --preflight .runtime/olmo1b-step60000/o5b-preflight-01 \
  --runs .runtime/olmo1b-step60000/o5b-pilot-01 \
  --output-dir .runtime/olmo1b-step60000/o5b-initial-retention-01 \
  --report-dir docs/reports/olmo1b-o5b \
  --prefix "$O5B_PREFIX" --phase initial
```

The credential override uses the mounted ADC instead of a stale inherited
credentials-file setting. Retention verifies and references the existing O1
checkpoint and O4 prepared-data archive; it does not upload them again. Initial
and final evidence are create-only objects under `<prefix>/initial/` and
`<prefix>/final/`. Adapted full checkpoints use `<prefix>/<arm>/update-NNNNNN.pt`.
The local `upload-result.json` records the verified receipt object.

## Queue, stop and exact recovery

Launch, or resume an interrupted queue, with the same command and prefix:

```bash
env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_o5b_queue.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --configuration .runtime/olmo1b-step60000/o5b-preflight-01/configuration.json \
  --output-dir .runtime/olmo1b-step60000/o5b-pilot-01 \
  --storage-prefix "$O5B_PREFIX"
```

Run only one queue. The queue skips an already-completed arm only after checking
its lineage and retained endpoint. It resumes an interrupted arm from the
**latest recorded checkpoint only**; an older checkpoint is a different branch.
It will not silently restart an interrupted arm from native weights. Source,
configuration, runtime and data identities must match. Full model/fusion state,
AdamW, LR scheduler, RNG, completed counters and the exact next-window cursor
are restored; beta is derived from those counters. Authoritative evaluations
follow the restored lineage, with abandoned/superseded observations retained
separately for audit. A failed checkpoint upload is retried from its already
saved local bytes.

To request a pause of both the active arm and the remaining queue, this
filesystem-only command may run from the **host project directory**:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path('.runtime/olmo1b-step60000/o5b-pilot-01')
state = json.loads((root / 'queue.json').read_text())
(root / 'STOP').touch()
if state.get('active_arm'):
    (root / state['active_arm'] / 'STOP').touch()
PY
```

The root `STOP` is checked between arms; the arm's `STOP` is checked between
optimizer updates. The training child finishes its active work, saves a finite
completed boundary, verifies cloud retention and reports `paused`. SIGTERM or
SIGINT to the **training child** also requests a boundary stop. Killing the
container or queue parent is not the graceful-stop procedure.

Before resuming, confirm previous queue/child processes exited, then remove only
the requested `STOP` files and rerun the unchanged queue command. If the latest
local checkpoint is missing, restore its exact cloud object to the recorded
path using the arm report/receipt's URI, generation, size and SHA256, and verify
the bytes first. Do not substitute an earlier update. Previous local full
checkpoints are removed only after verified cloud retention of a successor;
the cloud copies and local receipts remain. A health-gate failure requires
diagnosis before resumption. No budget, batch or LR adjustment is automatic.

## Reporting and final retention

The following commands are CPU work. From the host project directory, open
another project shell with:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash
```

Set `O5B_PREFIX` to the same recorded prefix in this shell. The finisher may run
alongside the GPU queue; it waits for both arms to complete, builds the report,
then verifies and retains final evidence:

```bash
env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_o5b_finish.py \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --preflight .runtime/olmo1b-step60000/o5b-preflight-01 \
  --runs .runtime/olmo1b-step60000/o5b-pilot-01 \
  --retention-output .runtime/olmo1b-step60000/o5b-final-retention-01 \
  --report-dir docs/reports/olmo1b-o5b \
  --prefix "$O5B_PREFIX"
```

Run only one finisher. Status is `<runs>/finish-status.json`; successful final
retention also writes `docs/reports/olmo1b-o5b/final-storage-receipt.json`. It exits
if the queue stops rather than authorizing more training. Resume it separately
after a queue restart. For immutable-upload retries, reuse the original output
directory/prefix: it verifies and preserves already-built report/figure bytes.

For manual reporting **after both arms complete**, followed by final retention:

```bash
python scripts/olmo_o5b_report.py \
  --preflight .runtime/olmo1b-step60000/o5b-preflight-01 \
  --runs .runtime/olmo1b-step60000/o5b-pilot-01 \
  --output-dir docs/reports/olmo1b-o5b

env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_o5b_retain.py \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --preflight .runtime/olmo1b-step60000/o5b-preflight-01 \
  --runs .runtime/olmo1b-step60000/o5b-pilot-01 \
  --output-dir .runtime/olmo1b-step60000/o5b-final-retention-01 \
  --report-dir docs/reports/olmo1b-o5b \
  --prefix "$O5B_PREFIX" --phase final
```

Use either the finisher or this manual route. Once immutable final upload has
started, use the finisher's retry path instead of regenerating report timestamps
or plots under the existing cloud prefix. Retention can locate O4's retained
prepared-data manifest with its default local path; `--prior-data-manifest`
allows an exactly verified restored copy if needed.

The report requires matching completed endpoints, source/configuration/data
identities, exposure, evaluation selections and checkpoint receipts. Outputs
are `results.md`, `final-comparison.json`, `learning-curves.pdf/png` and
`online-comparison.pdf/png`; existing `protocol.md` is preserved.

- Curves score each finite pass on fixed **128-window** development prefixes.
  The final **512-window** comparison is separate; its NLL is per pass, never
  the summed training objective.
- Exact-online diagnostics score the first **32 windows truncated to at most
  64 tokens**, comparing pass 0, finite K2 and online on identical prefixes.
  They are teacher-forced next-token predictions, not free-running generation.
  Their short-context NLL is not interchangeable with the 512-window results.
- Paired confidence intervals use 1,000 original-document cluster resamples
  (seed 20260922), combining repeated windows and preserving token weighting.
  They describe sampled documents, not training-seed variability. Official test
  splits remain untouched; code NLL does not establish executable-code success.

Review these bounded recovery results before extending exposure or enabling RT,
NextLat, prefix sampling or noise. The queue and finisher do not authorize or
schedule those follow-up experiments.
