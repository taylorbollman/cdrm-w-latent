# Campaign distributed runner and probes

The opt-in `CampaignDDPGraphTraining` wraps `CampaignObjective` in real DDP.
Initialize NCCL and set the rank's CUDA device first. Construct the adapter with
the world size and global per-term counts, then call `capture(warmup=11)`.
Preparation runs11 synchronized backward calls plus9 accumulation warmups;
capture records two more backward calls without advancing training clocks.

`backward(batches, feedback_noises=..., replay=True)` validates all local slots,
coordinates errors, reduces counts, clears gradients once and accumulates the
first M-1 slots with a local graph before one synchronized graph. Inspect raw
gradients before calling `step(result, optimizer, scheduler=..., counters=...)`,
or use `optimizer_step` for both. The step clips once, runs Adam once, advances
global token/target counters once and clears gradients in place. The optimizer
and scheduler remain outside capture. `discard_backward` clears an unstepped
diagnostic result without changing clocks.

Keep physical B/T, module/parameter participation, precision and graph storage
fixed. Tokens, right-padding, per-term masks, jitter and number of accumulated
slots may change. Every enabled objective needs a positive global denominator;
a local rank or any local slot may be entirely empty. All ranks must invoke
methods in the same order and supply the same number of physical slots.
Current path rejects packed multidocument rows. Dense fixed-capacity losses
perform some work on unused positions; measure the complete selected workload.

Use `with runner.checkpoint_boundary(): save_distributed_checkpoint(...)` only
after a complete or discarded update. This temporarily exposes `grad=None` to
the saver and restores graph-owned gradient storage. Resume into a fresh model
and process group, load state first, then construct/capture a new runner.
Never load a checkpoint into existing captured storage. Bucket views, changed
world size and in-flight collective recovery are not supported here.

## Launching bounded checks

All GPU commands run through the project Docker launcher. Example:

```bash
CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc '
  set -euo pipefail
  test -f /.dockerenv
  test "$PWD" = /workspace/cdrm-w-latent
  nvidia-smi
  env TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0 \
    timeout 900s torchrun --standalone --nproc-per-node=2 \
    scripts/olmo_campaign_ddp_probe.py --scale tiny --case graph \
    --output-dir .runtime/olmo-campaign-two-gpu/NEW-ATTEMPT
'
```

Run eager and graph separately. Tiny defaults to all8arms, B2/T8 FP32;
pretrained defaults to B/NFR, B2/T16 BF16 mixed with FP32 masters. Every attempt
needs a new output directory. The launcher log should be copied into the stage
after its writers stop, then retained with the source snapshot and report.

Default `--reference canonical` uses independently selected-position losses.
`--reference prepared` instead checks distributed execution against the same
dense masked arithmetic run locally without DDP or capture. It also records a
fixed-state canonical/prepared comparison separately. Read `operational_status`,
`independent_reference_status` and `qualification_failures` together: an
operational pass does not clear a failed independent reference. The observed
NFR BF16 discrepancy and localization are documented in qualification-plan.md.

`scripts/olmo_campaign_fp32_localize.py --output-dir NEW` is a one-GPU,
no-DDP/no-optimizer diagnostic at the same actual pretrained weights. It uses
full FP32, forced math SDPA and eager native RT to compare sparse/dense loss
layouts. This does not compare BF16 with FP32 or prove BF16 harmlessness.

`scripts/olmo_campaign_restart.py --phase write --scale tiny|pretrained --arm NFR`
requires new `--output-dir` and `--checkpoint-dir`. It saves after one update
and records the next update through the original live graph. Retain the
checkpoint, download it to a fresh location and verify hashes. Launch a new
`torchrun` with `--phase resume`, the restored checkpoint directory and:

```text
--reference-report WRITE-DIRECTORY/report.json
--reference-sha256 SHA256-OF-COMPLETED-WRITE-REPORT
--expected-manifest-sha256 SAVED-CHECKPOINT-MANIFEST-SHA256
```

The two phases require identical implementation hashes and configuration.
Source changes mean a new write/resume pair, not silently accepting old pins.
Model, optimizer, RNG, cursor and next-update comparisons are recorded per rank.

After correctness/restart gates pass, `scripts/olmo_campaign_capacity.py`
accepts one `--batch-size 8|16|32` and `--microbatches 1|2` candidate per launch.
It uses NFR K4/T1024, RT0/15 every pass, full-valid isolated document rows,
three eager warmup Adam updates before graph capture, one untimed replay
backward/discard and five timed graph updates. Actual Adam moments are resident
during capture; initial DDP construction with resident Adam still needs its own
T1024 check. Rates count global valid
input tokens once. CPU validation/refills, graph/NCCL work, clipping, Adam and
schedule are included; fixture generation, reporting and extra health scans
are outside timing. This is a directional resource check, not production
loader/packing or learning-quality acceptance.

## Durable recovery

Evidence and committed checkpoint uploads use `scripts/olmo_two_gpu_retain.py`.
Use the documented project GCS namespace and keep receipts outside the stage.
Large state upload checks server size/MD5 and SHA metadata; additionally restore
generation-pinned objects and hash all downloaded bytes before calling a cloud
copy verified recovery. The local SSD can disappear; boot-disk files persist
according to the user's VM setup. No checkpoint is overwritten automatically.
