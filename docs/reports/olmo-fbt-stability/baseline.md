# Ordinary-model control through update 128

The ordinary OLMo control completed updates 33–128 from its exact saved update 32
state. All 96 additional updates were finite, and none triggered gradient
clipping. Held-out CE nevertheless rose modestly, from 2.631794 to 2.687618 nats
per target (+0.055824). This is useful context for the feedback experiments:
healthy numerical execution does not guarantee improving development loss
under this continued-pretraining data and learning-rate recipe. It neither
explains nor excuses a much larger feedback-specific loss increase.

The model is the original 16-layer OLMo-1B backbone, width 2048, 16 heads and 8192 MLP
intermediate width, with tied vocabulary/readout and RoPE. No RT, FBT or NextLat
arithmetic is active. The wrapper's 8,388,608 fusion parameters are dormant;
1,176,764,416 backbone parameters are trained. Training uses BF16 mixed precision
with FP32 master weights/Adam, Flash SDPA, ordinary activation checkpointing and
prepared CUDA graphs. Context 1024, physical batch 32/rank on two H100s, eight
microbatches/rank and 524,288 valid input tokens per optimizer update. The restored
Adam history and original 128-update schedule are preserved: the 100-update
warmup reaches the 2e-4 plateau without a restart.

| Completed update | Development CE, nats/target |
|---:|---:|
|32, restored origin|2.631794|
|48|2.635389|
|64|2.642590|
|80|2.656053|
|96|2.671181|
|112|2.676973|
|128|2.687618|

All evaluations use the same 64-row dev-main prefix: 65,536 input tokens and 65,472
eligible within-chunk CE targets, common FP32 evaluation. This is one small
fixed development panel, not a broad quality evaluation. The resumed 32
measurement is repeated in this segment. The additional exposure is 50,331,648
input tokens; total exposure at 128 is 67,108,864.

The pre-clipping gradient norm over the 96 new updates has minimum 0.387519,
median 0.431697 and maximum 0.533986, all below the 1.0 clipping threshold.

| Measured timing scope | Input tokens/s |
|---|---:|
|Forward/loss/backward plus optimizer/cursor regions|73,266|
|Those regions plus batch materialization|70,935|
|Complete update callbacks, including evaluation at scheduled boundaries|68,293|
|Full executor segment including restore, graph setup and checkpoint waits|30,625|

These are aggregate two-GPU input-token rates; repeated-pass work is absent.
They are observed wall-region timings, not an optimized steady-state hardware
benchmark. Each update's denominator uses its slower rank. The full executor
segment took 1,643.49 seconds (27.39 minutes); launcher-side metadata resolution
and process setup are outside that executor timer. Graph preparation took
59.10 seconds and restoration 21.39 seconds on the slower ranks. Three synchronous
checkpoint regions cost approximately 70–71 seconds each before background
retention. The main loop accumulated 533.78 seconds of foreground worker waits,
including the terminal drain. Thus the full-segment rate is much lower than the
training-region rate.

Peak observed allocated memory was 35.133 GiB/GPU and peak reserved memory
42.430 GiB/GPU. Current allocated memory at update 128 was 22.080 GiB, with minimum
sampled free memory 35.188 GiB. These are sampled/cumulative allocator measures,
not a guarantee of unobserved minimum headroom.

The terminal report is `completed_plan`, W&B is synced, and checkpoint 128 is
verified in GCS. All 200 historical runtime source pins match both live files
and the run's source snapshots. Saved 128 boundary hashes match the final
boundary, and the local 128 manifest and state-file size match its published
receipt. Closure verified metadata and source files only; it did not reload or
rehash the large model/optimizer state.

W&B's final training-history checkpoint telemetry is stale: its last logged
fields say local 96/cloud 64/worker pending because the historical runner does
not emit a new history row after terminal checkpoint drainage. The terminal
report and verified receipts correctly say local 128/cloud 128/no pending worker.
This is a logging limitation, not a missing checkpoint.

W&B: https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/37uu86ip

Authorities: `.runtime/olmo-fbt-stability/native-b32-to128-01/report.json`,
SHA256 `9b8f44ce52163eca6679f78f4603fb26e023d6354273f2369749fd5b4798f7f3`;
metadata/analysis closure in `.runtime/olmo-fbt-stability/baseline-closure-01/`.
See [storage-receipt.md](storage-receipt.md) for retained evidence and checkpoint
references. Root-owned experiment conclusions remain in `results.md`.
