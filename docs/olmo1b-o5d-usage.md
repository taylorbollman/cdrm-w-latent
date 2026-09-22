# O5d finite-pass versus exact sequential feedback diagnostic

**Completed and assessed2026-09-22. Do not relaunch the completed run.**
See [assessment](reports/olmo1b-o5d/assessment.md) and [results](reports/olmo1b-o5d/results.md).
The commands below document the original invocation and incomplete-run recovery,
not another queued evaluation. All16 cases passed in30.4minutes; exec19959 ended0.
W&B: [jhhx390b](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/jhhx390b).
Evidence: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5d-online-diagnostic/20260922T105024Z/`;
[storage receipt](reports/olmo1b-o5d/storage-receipt.json).

Read [protocol](reports/olmo1b-o5d/protocol.md) and the current
[handoff](fbt-rt-nextlat-handoff.md). Evaluation only: no training or optimizer.
Use the existing source and mixed checkpoints pinned in `olmo_o5d_common.py`
and inherited `olmo_o5c_common.py`. Do not substitute checkpoint filenames or
turn this into another learning run. Core model/evaluator sources stay frozen.

Host root `/home/taylorbollman/cdrm-w-latent`; container root
`/workspace/cdrm-w-latent`. All evaluation runs inside the GPU container.

```bash
bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_o5d_diagnose.py --data .runtime/olmo1b-step60000/o4-data-01/prepared --output-dir .runtime/olmo1b-step60000/o5d-diagnostic-01'
```

This first verifies container/GPU/runtime, endpoint report and checkpoint bytes,
data manifest and source inventories, then creates one online W&B run in
`taylorbollman/pretrained-fbt-rt-nextlat`, group `olmo1b-o5d-online-diagnostic`.
The output directory must be new. Both domains share the fixed first32 max64
and first512 max512 windows in all K2/K3/K4/online evaluations. Ordinary pass0
is retained separately. Reported NLL is not the multi-pass training objective.

The output `report.json` is atomically saved after every split and completed
case, with batch progress every eight batches. Model hashes are checked at
case boundaries; full checkpoints are unchanged and already retained in GCS.
SIGINT/SIGTERM or an output-directory `STOP` file requests a batch-boundary
pause. A completed report must not be resumed. To recover an incomplete run,
remove a deliberately created STOP file and add `--resume` to the same command.
The same W&B run resumes; sources/data/config/runtime must match. Completed
cases are reused and an unfinished case is recomputed from the original pinned
checkpoint. There is no optimizer/RNG training state to reconstruct.

After a passing report, generate paired comparisons, document bootstrap
intervals and plots with `olmo_o5d_report.py`. Keep short and full contexts
separate. Retain final evidence and parent checkpoint identities through
`olmo_o5d_retain.py`; this uploads no additional model copy. The final assessment
must distinguish teacher-forced sequential feedback from free-running generation
and from an equally additionally trained ordinary-model comparison.
