# OLMo O4 pilot: running, recovery and reporting

This guide describes the frozen O4 queue, which was launched on 2026-09-21.
It does **not** assert that training has completed. Read
[`queue.json`](../.runtime/olmo1b-step60000/o4-pilot-01/queue.json) and the per-arm
reports for current status. The scientific contract is the existing
[protocol](reports/olmo1b-o4/protocol.md); this guide does not change it.

## Frozen recipe and locations

All four arms independently start from original OLMo-1B step60000, approximately
252B pretraining tokens. The backbone remains 16 layers, width2048, native RoPE,
non-affine LayerNorm, full MHA and tied output embeddings. RT selects layer0
only. NextLat is the O3 training-only predictor, with fixed latent/KL weights1/1.
FBT is off. The arms execute in this order:
`ordinary`, `ordinary-nextlat`, `rt`, `rt-nextlat`.

The H100 preflight selected physical batch32, T512, BF16 mixed with FP32 master
parameters and AdamW moments. All arms use LR1e-5, 100-update LR warmup, identical
ordered CodeSearchNet Python windows and fixed evaluation prefixes. The exact
frozen budget is **2,634 optimizer updates and 20,855,799 valid input tokens**
per arm:

| Phase | Updates in phase | Valid input tokens | Boundary after update |
| --- | ---: | ---: | ---: |
| LR warmup, RT alpha0 | 100 | 805,481 | 100 |
| RT alpha transition | 1,267 | 10,002,783 | 1,367 |
| RT alpha1 | 1,267 | 10,047,535 | 2,634 |

Input exposure includes the single overlapping context token at internal
windows. CE targets are counted separately. Ordinary arms follow the same
schedule and data exposure while keeping recurrence off. This is a bounded
recovery/direction pilot, with one seed and unequal compute across architectures.

Paths below are relative to `/home/taylorbollman/cdrm-w-latent` on the host,
mounted at `/workspace/cdrm-w-latent` inside the container:

- Native starting artifacts: `.runtime/olmo1b-step60000/artifacts/`.
- Prepared data: `.runtime/olmo1b-step60000/o4-data-01/prepared/`.
- Preflight and frozen configuration: `.runtime/olmo1b-step60000/o4-preflight-01/`.
- Queue state: `.runtime/olmo1b-step60000/o4-pilot-01/queue.json`.
- Queue log: `.runtime/olmo1b-step60000/o4-pilot-01-queue.log`.
- Per-arm evidence: `.runtime/olmo1b-step60000/o4-pilot-01/<arm>/report.json`,
  `events.jsonl`, `update-NNNNNN.receipt.json` and the latest local
  `update-NNNNNN.pt`; console logs are `<arm>.log` in the queue directory.
- Cloud prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z/`.
  Checkpoints are `<prefix>/<arm>/update-NNNNNN.pt`; initial and final evidence
  archives live under `<prefix>/initial/` and `<prefix>/final/`.

Frozen configuration SHA256:
`6b7e9168f2aa1afc495d28cfdd84006e5fa3409f66fcc70c17c0fa83c8ff2e20`.
The configuration explicitly records a pre-learning, resume-only retention retry
amendment to the training script. The authentic preflight source hash, previous
script snapshot and previous configuration remain retained. No model, optimizer,
data, alpha schedule or measured preflight computation changed. Do not edit
frozen core sources between arms; the queue rejects mismatched source hashes.

## Stop and resume

The queue checks its root `STOP` file between arms. The active training process
checks a `STOP` file in its own arm directory between optimizer updates. To pause
both the active arm and subsequent queue entries, from the project directory:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path('.runtime/olmo1b-step60000/o4-pilot-01')
state = json.loads((root / 'queue.json').read_text())
(root / 'STOP').touch()
if state.get('active_arm'):
    (root / state['active_arm'] / 'STOP').touch()
PY
```

The active arm finishes the current update, checks finite model/optimizer state,
saves a complete checkpoint and verifies its cloud upload before reporting
`paused`. SIGTERM/SIGINT sent to the **training child** request the same boundary
stop; killing the container or queue parent is not this graceful-stop mechanism.

Before resuming, confirm that the previous queue/training processes have exited.
Remove only the requested `STOP` marker files after deciding to resume. Relaunch
the queue with exactly the original configuration, data and output directory:

```bash
bash scripts/docker_shell.sh bash -lc 'env -u GOOGLE_APPLICATION_CREDENTIALS \
  python scripts/olmo_o4_queue.py \
  --artifacts .runtime/olmo1b-step60000/artifacts \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --configuration .runtime/olmo1b-step60000/o4-preflight-01/configuration.json \
  --output-dir .runtime/olmo1b-step60000/o4-pilot-01 \
  --storage-prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z'
```

GPU work must run inside this container; never run the training script directly
on the host. The credential override uses the container's valid ADC rather than
the stale inherited credentials-file setting. It does not print a secret.

Completed arms are skipped. An interrupted arm resumes its latest recorded local
checkpoint; it does not restart from the native weights. The loader verifies
the checkpoint SHA256, configuration, source/data/runtime identity and parameter
ownership, and restores optimizer, LR scheduler, counters, RNG and next-window
cursor. Alpha is recomputed from the completed counters, rather than reset.
An upload that failed after a local save is retried on resume. A health-gate
failure requires diagnosis; the queue refuses to skip it automatically.

If the latest local `.pt` is missing, find its exact URI, SHA256, byte size and
GCS generation in the arm report or `.receipt.json`. Restore those bytes to the
recorded project path and verify the SHA256 before resuming. A W&B graph is not
a model backup. Each full checkpoint is roughly14–15GB; prior local copies are
removed only after both the previous and successor cloud objects are verified.
The immutable cloud copies remain available. The original native checkpoint has
its separate O1 receipt; an adapted arm checkpoint is never a new starting point
for another arm.

## Completed comparison report

[`scripts/olmo_o4_report.py`](../scripts/olmo_o4_report.py) is analysis-only and
requires all four arms to have completed the frozen endpoint. Run it in the CPU
container after the queue finishes:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_o4_report.py \
  --preflight .runtime/olmo1b-step60000/o4-preflight-01 \
  --runs .runtime/olmo1b-step60000/o4-pilot-01 \
  --output-dir docs/reports/olmo1b-o4'
```

For the current queue, `scripts/olmo_o4_finish.py` is already waiting in a CPU
container and will invoke this report and final evidence retention automatically.
Its state is `<runs>/finish-status.json`; its log is
`.runtime/olmo1b-step60000/o4-finish-01.log`. It does not merge the draft PR.
After a VM restart, relaunch it alongside the resumed GPU queue:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc 'env -u GOOGLE_APPLICATION_CREDENTIALS \
  python scripts/olmo_o4_finish.py \
  --data .runtime/olmo1b-step60000/o4-data-01/prepared \
  --preflight .runtime/olmo1b-step60000/o4-preflight-01 \
  --runs .runtime/olmo1b-step60000/o4-pilot-01 \
  --retention-output .runtime/olmo1b-step60000/o4-final-retention-01 \
  --report-dir docs/reports/olmo1b-o4 \
  --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z'
```

Run only one finisher. If final retention already started before an interruption,
inspect its receipts before choosing a new retention output/prefix; immutable
objects must not be overwritten with different bytes.

It checks complete configuration/source/data/schedule and exposure agreement,
final checkpoint receipts, evaluation identities and inference modes. It
rejects missing/running/failed arms and conflicting duplicate observations;
identical repeated small evaluations after a resume are coalesced. The declared
preflight amendment is checked explicitly, including the old source snapshot.

Outputs are `results.md`, `final-comparison.json`, and standalone
`learning-curves.pdf` / `learning-curves.png`. The existing `protocol.md` is
preserved. Curves use only the fixed128-window evaluations; the full512-window
final comparison has a separate table. Final NLL differences use 1,000 paired
original-document cluster bootstrap samples (seed20260922), combining windows
belonging to the same document before resampling and retaining token weighting.
The JSON includes each arm versus the original ordinary checkpoint, all six
between-arm contrasts, and the RT/NextLat interaction contrast.

These intervals describe evaluation-document sampling, not training-seed
uncertainty. The report retains the equal-exposure/unequal-compute qualification,
window resets and masks, unknown pretraining overlap, unused official tests,
W&B links and exact endpoint checkpoint cloud receipts. Code NLL is not an
executed programming-task score. Any larger budget or FBT experiment follows a
separate review of these pilot results.
