# Saved NFR state: packed execution bridge results

2026-09-29. The [planned bridge](packed-bridge-protocol.md) passed its operational
and graph checks in **617.73 seconds**. BF16 prepared eager execution and CUDA
graph replay produced **bitwise-identical losses and all 71 parameter-gradient
tensors**. The separate BF16 sparse/prepared compatibility qualification remains;
passing capture does not erase it.

The starting authority was the **BF16 update-4 endpoint** of the full NFR
diagnostic: OLMo-1B with K4 feedback, native RT at layers 0/15 on every pass,
NextLat latent and KL losses, beta 1 and jitter 0.02. The runner strictly restored
the original isolated-document checkpoint, including Adam, scheduler, counters,
RNG and cursor, and verified the complete saved boundary. It then released Adam
storage and explicitly switched only document policy to continuous-stream.
All weights, tying, parameter identities, trainability and modes stayed fixed.

The fixed development fixture contains two B1/T1024 records: **2,048 inputs,
2,046 CE targets, 2,040 latent pairs and 2,032 KL triples**, with six internal
document boundaries. The NF view required by the frozen fixture loader was used
only to construct data; the actual model remained NFR, and its keyed jitter
matched exactly. No optimizer updates or data-cursor advances occurred.

The table gives gradient relative L2 differences as percentages. Cross-precision
rows compare production BF16 Flash/native Triton against FP32 math/eager RT.

| Comparison | All parameters | Backbone | Fusion | Predictor | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 prepared vs sparse | 0.0000000217% | 0 | 0 | 0.0000000726% | Existing semantic budget passes |
| BF16 sparse vs FP32 sparse | 1.09781% | 1.10332% | 0.57314% | 1.04387% | Descriptive precision comparison |
| BF16 prepared vs sparse | 0.08792% | 0.09215% | 0.06188% | 0.00731% | Historical elementwise gradient budget fails |
| BF16 prepared vs FP32 prepared | 1.09552% | 1.10083% | 0.57456% | 1.04382% | Descriptive precision comparison |
| BF16 captured vs prepared eager | 0 | 0 | 0 | 0 | Bitwise gradient and metric equality |

For BF16 sparse/prepared, **the loss comparison passes**: CE and KL sums are
identical, latent-sum difference is `6.1035e-5`, and normalized objective
difference is `3.9084e-7`. The gradient qualification is specifically the unchanged
elementwise `atol=3e-5`, `rtol=3e-4` check: **50 of 71 tensors fail** (48 backbone,
one fusion and one predictor). The largest absolute difference is `0.00390196`
in layer-0 attention projection. Global gradient cosine is `0.999999614`.
This remains a recorded layout compatibility miss; that historical budget is
not a universal threshold for BF16 versus FP32. The latter comparison has global
cosine `0.99994008` in the prepared layout.

All five cases retained finite losses and gradients, complete parameter
participation, nonzero component gradients, exact target counts, unchanged
weights/buffers, inputs, masks, jitter, RNG and module modes. Capture and replay
also preserved the owned persistent tensor pointers. Actual accounting was
**10 measured physical backwards + 2 initialization backwards + 10 warmup
backwards + 1 capture backward = 23**, including two measured graph replays.
Setup gradients were discarded. Final checks confirmed restored production
flags, absent gradients and unchanged source, checkpoint and fixture files.
The independent CPU audit passed 18 checks, including exact hashes for all
**139 source files in both the saved snapshot and current checkout**.

The result connects this adapted NFR state to the intended prepared/captured
execution path at T1024 without introducing a new graph discrepancy. It supports
the next bounded optimizer-continuation check. It does not establish long-run
BF16 optimization equivalence, model quality, large-batch or DDP behavior, H200
readiness, or that the old sparse/prepared differences are harmless. Earlier
cold-state percentages used different states and fixtures; their numerical
reduction cannot be attributed to adaptation from this comparison alone.
The process peaked at about 15.28 GiB allocated and 24.75 GiB reserved; these are
diagnostic process peaks, not a production batch-size or throughput benchmark.

Evidence is in `.runtime/olmo-fusion-startup/packed-bridge-01/report.json`
(SHA256 `efee951a140bd13c7e27909e6de1f093439c7fa679bcbd19be4a2930d17537b6`)
and `packed-bridge-independent-audit.json` under the same runtime parent.
The checkpoint SHA256 is
`6030c92f1c09d2c561ca173eee816316ae51de4bc66b368a34996577b8c1dd0a`;
the fixture SHA256 is
`4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975`.
[W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/5dxrry8n).
