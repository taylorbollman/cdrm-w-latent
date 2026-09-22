# O5e ordinary additional-training control

Read the [current handoff](fbt-rt-nextlat-handoff.md) and
[protocol](reports/olmo1b-o5e/protocol.md) before launching or recovering work.
Only one 512-update control is authorized. It trains the shared O5b native
backbone on the exact O5c mixed plan with a single ordinary CE; FBT, RT and
NextLat are off. Fusion tensors are retained but unused and frozen.

Run GPU commands through the project container; host and container roots are
`/home/taylorbollman/cdrm-w-latent` and `/workspace/cdrm-w-latent` respectively.
Original preflight invocation:

```bash
bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_o5e_preflight.py --data .runtime/olmo1b-step60000/o5c-data-02/prepared --base-data .runtime/olmo1b-step60000/o4-data-01/prepared --output-dir .runtime/olmo1b-step60000/o5e-preflight-01'
```

The passing preflight freezes runtime, source hashes, configuration, initial
model hashes, optimizer ownership, data plan and starting evaluations. Do not
edit its sources or protocol and then resume a run under the old configuration.

Training uses `scripts/olmo_o5e_train.py` with the same `--data` and `--base-data`,
`--configuration .runtime/olmo1b-step60000/o5e-preflight-01/configuration.json`,
`--output-dir .runtime/olmo1b-step60000/o5e-pilot-01/ordinary` and the exact
`--storage-prefix` recorded in the handoff. Unset the stale
`GOOGLE_APPLICATION_CREDENTIALS` variable for GCS operations; mounted ADC supplies
credentials. Never print environment secrets.

The runner evaluates 128 windows per domain at update 0, 50 and every 64 updates;
the final evaluation uses 512 windows with original-document records. It saves
full model, AdamW, scheduler, counters, RNG and data cursor every 128 updates or
600 seconds. It verifies the GCS upload before removing older local checkpoints.
Allow disk space for the current and successor files, approximately 14 GB each.

SIGINT/SIGTERM or `STOP` in the output directory or its parent requests a safe
checkpoint-boundary stop. For an interrupted run, use the same arguments and add
`--resume` pointing to the latest recorded checkpoint. The runner checks exact
sources, runtime, data, configuration and checkpoint SHA, retries any unfinished
upload, restores optimizer/scheduler/RNG/cursor and resumes the same W&B run.
Do not resume a completed or health-gated run.

After completion, run the CPU reporter with GPU passthrough disabled:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_o5e_report.py --preflight .runtime/olmo1b-step60000/o5e-preflight-01 --run-dir .runtime/olmo1b-step60000/o5e-pilot-01/ordinary --output-dir docs/reports/olmo1b-o5e'
```

Retain evidence using `olmo_o5e_retain.py` and document the final checkpoint,
W&B URL, results and assessment in the handoff. This is an equal-data and
equal-exposure comparison with different trainable capacity, learning rate
and compute. No new online evaluation or automatic extension follows it.
