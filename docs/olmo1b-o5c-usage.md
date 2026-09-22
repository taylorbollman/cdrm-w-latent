# O5c fusion-only adaptation operations

This is the authorized paired code/general-text diagnostic following O5b.
Read [protocol](reports/olmo1b-o5c/protocol.md) and the current
[handoff](fbt-rt-nextlat-handoff.md) before starting or resuming anything.

Working branch: `feat/olmo1b-o5c-fusion-adaptation`, based on O5b merge
`4a53e48a2c56f71f4b9b4cc899e6c2c7ecd4e417`.

- Parent: O5b FBT update2634, SHA99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66.
- New data: `.runtime/olmo1b-step60000/o5c-data-02/prepared/`.
- Base/evaluation data: `.runtime/olmo1b-step60000/o4-data-01/prepared/`.
- Preflight: `.runtime/olmo1b-step60000/o5c-preflight-01/`.
- Queue: `.runtime/olmo1b-step60000/o5c-pilot-01/queue.json`.
- W&B project: `taylorbollman/pretrained-fbt-rt-nextlat`, group`olmo1b-o5c-fusion-only`.
- GCS prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/`.

Data manifest SHAf837f7f412dee17a304df5f78f4b65571f15163af440adecfc77b88cca8b1490.
Both arms512 updates /4,194,304 CEtargets. Code-only4,212,498 input positions /
18,194 segments; mixed4,208,250 positions /13,946 segments. Mixedcode and
general each2,097,152CE; inputpositions2,106,213 and2,102,037 respectively.
These are variable-row updates with physical chunks at most16, not B32fixed.
Targets are unique in each arm; context overlaps at segment boundaries.
Both arms cut code at identical4,096-target chunk boundaries; the mixed arm
uses the first half of exactly the same code segments. Supersededdata-01 was
never trained on and differs only in additional code context boundaries.
The unused code suffix begins at a new original document after the84,288
O5b windows. No cycling or warm-start optimizer moments.

Use the project Docker launcher for all execution. Explicitly disable GPU for
CPU tests. Actual GPU commands require the container and working nvidia-smi.
For GCS unset stale GOOGLE_APPLICATION_CREDENTIALS without printing secrets:

```bash
bash scripts/docker_shell.sh bash -lc 'env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/olmo_o5c_queue.py --data .runtime/olmo1b-step60000/o5c-data-02/prepared --base-data .runtime/olmo1b-step60000/o4-data-01/prepared --configuration .runtime/olmo1b-step60000/o5c-preflight-01/configuration.json --output-dir .runtime/olmo1b-step60000/o5c-pilot-01 --storage-prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/'
```

**Do not launch a duplicate queue.** Read queue/report status and check active
processes first. To request a clean stop, create`STOP`in the queue directory;
current arm saves a complete update-boundary checkpoint and the queue pauses.
Remove it only when explicitly resuming the authorized incomplete pair. The
same command resumes the latest recorded checkpoint or skips strictly verified
completed arms. A health-gate failure requires diagnosis first; no silentfresh
restart or older-checkpoint branch. Failed cloud uploads leave complete local
checkpoints and retry retention on resume. Restore missing local files from
the recorded GCS generation and verify SHA before resuming.

Every128 updates (or10minutes sooner), final and requestedstop: full model,
fusionoptimizer, scheduler, RNG, counters andnext-updatecursor are saved.
Only two fusion matrices have optimizer moments. The original O5b endpoint is
loaded with a scoped TorchVersionallowlist after fullSHAverification; newO5c
metadata is converted to built-in strings, so newcheckpoints use safe normal
weights-only loading. CPU exactfuture-update recovery is tested. Actual H100
nonzero-step/frozenstate checks do not claim fullGPUoptimizerreplay.

Final report: `python scripts/olmo_o5c_report.py --preflight
.runtime/olmo1b-step60000/o5c-preflight-01 --runs
.runtime/olmo1b-step60000/o5c-pilot-01 --output-dir docs/reports/olmo1b-o5c`.
The final retainer archives named data files, sourceinventory, reports/events,
plots andparentreceipts, reusing the retained O5b startingcheckpoint/O4base data.
Prepared generaldata are included in the newinitial archive. No .env or broad
workspace contents enter archives.
