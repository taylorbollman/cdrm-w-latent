# Single-GPU readiness evidence

This helper combines the explicitly selected objective-preparation and graph-
recovery reports. It does not launch GPU work or infer success from run names.
Only finished reports are accepted; failures retain their original status.

After the four primary cases finish, select their exact directory names and
runtime commit (replace the placeholders):

```bash
python scripts/summarize_olmo_single_gpu_readiness.py \
  --runtime-commit COMMIT \
  --runs distributed/rt-b2-t512-01 distributed/combined-b2-t512-01 \
         recovery/rt-b2-t512-01 recovery/combined-b2-t512-01
```

Use `--run-commit recovery/NAME=COMMIT` when a selected run has a different
recorded runtime revision. Current source changes are recorded separately;
the run's snapshots, recorded hashes and frozen Git files must agree. Protocol
snapshots and installed dependency snapshots are checked as well. Both groups
reuse the original selected O1 checkpoint and its immutable GCS receipt.

The helper writes `summary.json` in this directory. A complete primary scope
requires passing B2/T512 RT-only and combined cases in both groups. Smaller or
partial diagnostic selections remain labeled incomplete. Distributed preparation
expects the configured one or two physical updates, with two microbatches per
update; the primary protocol uses two. Recovery requires six physical updates
across preparation/reference/restored branches, a four-update logical endpoint,
and a hash/size-matched disposal receipt for its successful temporary checkpoint.
It checks that the deleted file remains absent. This verifies the retained
receipt against the frozen diagnostic's checks, without pretending to reread
bytes already deleted.

Before retention, the milestone owner creates `results.md` and finalizes
`test-results.txt`, and writes the final idle GPU inventory to
`.runtime/olmo-single-gpu-readiness/final-gpu.log`. Each group's selected run must
have a log at `.runtime/GROUP/logs/NAME.log`; the legacy adjacent `NAME.log` is
accepted only when exactly one of those locations exists. Group protocol and
test-results files are required; group usage/results files are included when
present.

Run CPU-only retention preparation first, then verified upload:

```bash
python scripts/summarize_olmo_single_gpu_readiness.py --retain --dry-run
python scripts/summarize_olmo_single_gpu_readiness.py --retain
```

Retention rechecks every selected report before archiving. It includes raw
reports, frozen sources/protocols/dependencies, allowlisted tests, logs and
project notes. It never archives model/optimizer checkpoints, `.env`, W&B
directories or datasets. A failed recovery's diagnostic checkpoint stays local;
retaining its failure report does not automatically upload those weights.
Original O1 checkpoint provenance is reused by reference.

Uploads are create-only under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-single-gpu-readiness/TIMESTAMP/`.
Server metadata and a downloaded SHA256 are checked for every newly uploaded
object. The final receipt is written here as `storage-receipt.json`.

These reports establish only one-GPU objective/accumulation and recovery with
both branches rebuilding CUDA graphs. They do not establish genuine DDP/NCCL,
distributed graphs, ZeRO compatibility, throughput or task performance.
