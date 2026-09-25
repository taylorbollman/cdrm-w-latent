# Native RT physical-batch milestone

**Measurements complete, 2026-09-25. Use physical B192 for RT and B128 for
K2 FBT + RT + NextLat at T512 on this H10080GB.** Keep Flash SDPA for ordinary
attention, native ordinary RoPE, rounded compiled ordinary SwiGLU and fused
AdamW. Native RT retains its Triton tiled attention/recompute implementation.
Both operating points pass exact own graph/eager checks, complete optimizer
updates and reverse-order fresh-process repeats. FA4 gives no useful reason to
switch: its resource gains are small, B256 RT capture still fails, and both
FA4 cross-configuration integration screens remain failed.

This closes the bounded one-GPU capacity investigation. Continue with the
[one-to-two-GPU readiness plan](../../native-rt-single-to-two-gpu-plan.md):
recoverable training, correct global-loss normalization and distributed adapter
preparation, then actual two-GPU correctness/recovery/scaling. Physical B512 and
further native RT leaf/backward optimizations are not prerequisites. There is
no quality-training result or new learning claim in this milestone.

## Recommended operating points and scope

Each repeated point consists of two independent processes with five synchronized
complete updates per run, after three real preparation updates and ten backward
warmups. The rate is the median of those two run medians; the range shows both
run medians. Profiles run afterwards and do not contribute timed samples.
Setup peaks are the maximum across those repeats. Free memory is the minimum
sampled at phase boundaries, not a continuously measured minimum.

| Configuration / physical B | Input tokens/s | Run-median range | Seconds/update | Setup allocated / reserved GiB | Current reserved GiB | Minimum sampled free GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| RT / 128 | 28,015.56 | 28,008.55–28,022.56 | 2.3393 | 45.746 / 48.377 | 48.377 | 29.684 |
| **RT / 192** | **29,746.02** | 29,743.65–29,748.39 | 3.3048 | 59.417 / 62.926 | 60.918 | 15.565 |
| Combined / 64 | 11,702.93 | 11,698.26–11,707.59 | 2.8000 | 38.925 / 41.713 | 41.213 | 36.613 |
| **Combined / 128** | **12,413.20** | 12,410.94–12,415.46 | 5.2795 | 58.095 / 65.971 | 65.113 | 12.520 |

RT B192 improves throughput by 6.18% over B128; combined B128 improves by 6.07%
over B64. The larger choices retain more than the roughly 8 GiB target headroom
at sampled boundaries. They are comfortable measured choices, not mathematical
maxima. RT B256 completes eager updates but fails graph capture; B512 was not
executed. Lower physical batch is required by the measured capacity of this
1.177B backbone/configuration, not evidence that small RT batches are inherently
faster. Accumulation does not enlarge the per-invocation RT matrices.

All measurements use the original OLMo-1B step60000 checkpoint (about 252B
pretraining tokens): 16 layers, width 2048, 16 heads, SwiGLU 8192 per branch and tied
50304-token vocabulary. Native RT is selected at indices 0 and 15. Combined is
K2 FBT: one ordinary bootstrap followed by one feedback pass with those RT
layers, plus NextLat. It does not use RT in both passes. Full valid CE uses
chunks 2048; the existing KL mask/grouping 128 is preserved. BF16 mixed execution
keeps FP32 model/gradient/Adam state, native nonaffine LayerNorm and FP32
residual/RoPE semantics. TF32 and autocast weight caching are off. No Q/K
normalization is added.

Ordinary layers use activation checkpointing. Graphs capture forward/loss/
backward; input validation, clipping, fused Adam and scheduler remain outside.
One physical batch is one optimizer update, without gradient accumulation.
Native RT retains reused RoPE/weight casts, K/V-only writes, Triton historical
tiles and recompute backward. Historical RT/FBT and native/author numerical
qualifications remain open. The [plot](capacity.pdf) separates source/preparation
cohorts and labels failed-integration arms; failed attempts remain in the
[complete summary](summary.json).

## B8 integration

| Case and candidate | Compatibility | Own operational gates | Updates | Run |
| --- | --- | --- | ---: | --- |
| RT, `optimized` (Dao RoPE) | CE relative difference 0.000325201 exceeds unchanged 0.00001 budget; outputs/gradients pass | 5/5 pass, including exact full Adam updates | 6 | [v6znjt7h](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/v6znjt7h) |
| RT, `compiled-native` | Pass; outputs/losses bitwise against control | 5/5 pass | 6 | [xhf63nea](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xhf63nea) |
| Combined, `compiled-native` | Pass; outputs/losses bitwise against control | 5/5 pass | 6 | [y9llge5g](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y9llge5g) |

The Dao CE discrepancy is about 0.0006894 nats per supervised target. The
compiled-native localization changes ordinary RoPE while retaining compiled
SwiGLU/fused Adam, supporting its use as the conservative capacity candidate.
This does not invalidate the separately recorded ordinary-only Dao evidence.

## Initial RT capacity ledger

Each successful row contains three real preparation updates, five synchronized
complete timed updates and three separate backward samples. Rates are medians
of the five full updates and include input handling, graph replay, clipping,
Adam and scheduler. The operating-point table above includes independent repeats;
this ledger preserves the initial controls, preparation intervention and failures.

`live` means the original eager comparison beside the allocated CUDA graph,
with transient cleanup disabled. `before` means eager CPU references before
capture, terminal graph release before eager comparison, and transient cleanup
before and after warmup. All exact loss/gradient gates are retained.

| RT arm / physical B | Validation | Input tokens/s | CE targets/s | Setup peak allocated / reserved GiB | Steady reserved GiB | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Control / 64 | live | 22,319.53 | 22,275.94 | 32.076 / 50.311 | 50.311 | [Passed](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xgm78a8u), 8 updates |
| Compiled-native / 64 | live | 23,402.82 | 23,357.12 | 32.077 / 52.225 | 52.225 | [Passed](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/z2qwh830), 8 updates |
| Compiled-native / 128 | live | 28,010.05 | 27,955.34 | 45.746 / 78.059 | 76.266 | [Passed](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/vvdkbb71), 8 updates; tight setup |
| Compiled-native / 192 | live | — | — | — | — | [OOM in validation_initial](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/w1ytzug4), 3 updates; capture succeeded |
| Compiled-native / 128 | before | 28,008.55 | 27,953.85 | 45.746 / 48.377 | 48.377 | [Passed](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/154naiyy), 8 updates |
| Compiled-native / 192 | before | 29,743.65 | 29,685.56 | 59.417 / 62.926 | 60.918 | [Passed](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4upxo3z9), 8 updates |
| Compiled-native / 256 | before | — | — | — | — | [OOM in capture_capture](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/s0zg48mj), 3 updates |

At matched B64, the first compiled-native measurement is 4.85% faster than
control. At B128, changing validation/cleanup keeps throughput essentially
unchanged (28,010.05 versus 28,008.55 input tokens/s), while setup reserved peak
falls from 78.059 to 48.377 GiB. Allocated peak remains 45.746 GiB. Device free
memory sampled at the setup boundary rises from 1.795 to 29.684 GiB. This tests
the validation-order and cleanup combination; it does not isolate their
individual contributions or reduce the model's underlying live tensor needs.

The corrected B192 point is 6.19% faster than corrected B128. It has 17.145 GiB
device free memory at the setup boundary, and the repeat supports choosing it as the comfortable RT point. Boundary samples are
not continuous minimum-free measurements.

The two OOMs have different meanings. Original B192 failed a 1.50 GiB ordinary
SwiGLU allocation during eager validation while 42.60 GiB occupied private graph
pools; its CUDA capture had already completed. The old report's raw stage still
says `capture`, so the error-marked memory phase is the authoritative phase
evidence. Corrected B256 failed a 2.00 GiB allocation during captured native RT
backward, with the traceback entering the batched `torch.autograd.grad` VJP in
`olmo_tiled.py`. It had completed three optimizer updates and pre-capture eager
references. That failure is a capacity limit of this measured capture path,
not evidence that B512 or another implementation has been tested.

The fixed FP32 parameter, gradient and Adam-moment floor is approximately
17.57 GiB for RT and 18.89 GiB for combined, before activations and workspace.
Full-sequence FP32 boundary states and native RT MLP replay add real costs that
scale with physical batch; historical-attention recompute does not remove them.
Graph replay reuses captured storage, so its reset allocated peak is not the
complete graph memory requirement: B192 reports 17.79 GiB allocated during
timing alongside 60.92 GiB reserved. See the [capacity audit](capacity-audit.md)
for phase minima, fixed-state arithmetic and qualified tensor-size estimates.

## Initial combined capacity

The combined model uses one ordinary bootstrap pass and a feedback pass with
RT at layers 0/15, plus NextLat. Rates count each original input token once;
combined pass-token work is twice that count. These successful reports all use
the corrected before-capture validation and cleanup, with eight updates each.

| Combined arm / physical B | Input tokens/s | Setup peak allocated / reserved GiB | Steady reserved GiB | Run |
| --- | ---: | ---: | ---: | --- |
| Control / 64 | 11,198.02 | 38.923 / 40.871 | 41.256 | [oh7n35js](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/oh7n35js) |
| Compiled-native / 64 | 11,707.59 | 38.925 / 41.713 | 41.213 | [aqw8yhev](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/aqw8yhev) |
| Compiled-native / 128 | 12,410.94 | 58.095 / 65.971 | 65.113 | [mzt0ukzu](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mzt0ukzu) |

The first matched B64 pair improves by 4.55%. Increasing candidate batch to128
improves throughput by a further 6.01%; its minimum boundary-sampled free memory
is 12.52 GiB. Repeats confirm both practical points (table above). The control
B64 timing peak exceeds its pre-timing setup peak, so total headroom must inspect
all phases rather than the setup table alone.

## Conditional ordinary FA4 integration

The additional B8 checks compare `fa4-native` with `compiled-native`: only
ordinary attention changes from Flash SDPA to FA4. Native RT attention remains
on its Triton implementation. Both candidates are finite and pass all five
own operational gates, including exact graph/eager full-Adam update parity.
**Both fail the cross-configuration integration screen.** Capacity rows that
pass their own operational gates therefore remain exploratory measurements;
they do not authorize FA4 adoption or establish equivalent training behavior.

| Case | Output comparison | Raw gradients | Loss comparison | Own gates / updates | Run |
| --- | --- | --- | --- | --- | --- |
| RT | Pass: relative L2 0.005058 | Pass: global relative L2 0.004551; all 65 tensor budgets pass | CE relative difference 0.000573327 exceeds 1e-5 | 5/5 / 6 | [fkaf6ccb](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/fkaf6ccb) |
| Combined | Fail: feedback-pass output relative L2 0.018277 exceeds 1/64; maximum-relative passes | Fail: global relative L2 0.046837; 67/71 tensor L2 failures, 15/71 maximum-relative failures | All 6 pass/objective terms exceed1e-5 | 5/5 / 6 | [9hyx08x3](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/9hyx08x3) |

The unchanged budgets are global-gradient/output relative L2 ≤ 1/64, individual
gradient-tensor relative L2 ≤ 1/32, maximum-relative ≤ 1/16, and loss relative
difference ≤ 1e-5. The combined worst tensor relative L2 is 0.076046 and worst
maximum-relative is 0.195136; 67 distinct tensors fail one or both tensor budgets.
The RT-only failure is loss-only. Own graph agreement cannot resolve either
cross-configuration discrepancy. Keep Flash SDPA as the working ordinary backend
for RT and combined; no broad new precision investigation is required here.

The qualified capacity comparison is complete, with a single FA4 measurement
per successful shape, compared below with the median of two SDPA run medians.

| Shape | Flash SDPA input tokens/s | FA4 input tokens/s | SDPA / FA4 setup reserved GiB | SDPA / FA4 current reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| RT B192 | 29,746.02 | 29,888.96 | 62.926 / 62.926 | 60.918 / 61.668 |
| Combined B128 | 12,413.20 | 12,503.52 | 65.971 / 64.113 | 65.113 / 64.113 |

The directional throughput differences are +0.48% for RT and +0.73% for combined.
Setup allocated peaks remain 59.417 GiB for RT and 58.095 GiB for combined.
Combined FA4 reduces peak reserved by 1.857 GiB and current reserved by 1 GiB;
RT shows no setup-reserved reduction and uses 0.75 GiB more current reservation.
These allocator results do not establish less live tensor storage. Minimum
boundary-sampled free memory in combined is 12.520 GiB SDPA and 13.877 GiB FA4.

[FA4 RT B256](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/rypl2s5d)
also fails during `capture_capture`, after three completed eager updates. Its
2 GiB allocation failure and completed warmups match the broad location of the
SDPA limit. FA4 therefore does not unlock that captured physical batch. Neither
B512 nor a larger combined boundary is claimed tested. Combined already has
comfortable B128 headroom; a marginal boundary search is not needed to select
it. Given the modest resource differences and retained integration failures,
keep Flash SDPA and close this conditional FA4 investigation.

## Profiles and next optimization priorities

The [profile audit](profile-audit.md) contains full accounting and limitations.
Each useful-batch profiled run adds one ninth optimizer update and passes final
changed-weight graph/eager parity. Separate unprofiled graph-only medians are
3.2766s for RT B192 and 5.2491s for combined B128, versus complete-update medians
3.3045s and 5.2786s in those runs.

| Full-step profile name category | RT B192 | Combined B128 |
| --- | ---: | ---: |
| Matrix multiplication | 41.94% | 38.40% |
| Copy/cast | 12.06% | 17.84% |
| Fill | 7.43% | 9.91% |
| FP32 addition | 9.04% | 9.66% |
| Ordinary Flash attention names | 2.67% | 2.33% |
| Named historical RT backward subset | 1.45% | 0.62% |
| Fused optimizer | 0.35% | 0.23% |

These percentages describe summed kernel-event time after excluding overlapping
GPU annotations. They are not hardware utilization, full-update wall fractions,
or model-layer attribution. Generic GEMMs, copies, normalization and pointwise
operations cannot be assigned exclusively to RT finish/writer. The named RT
subset is not total RT time. The inherited `ordinary_step/` CPU scopes cover
the complete mixed computation, and CPU dispatch durations do not locate GPU
work. Copies/fills/arithmetic support a later bounded contiguous-path fusion
probe; the large reconstructed backward workspace supports a separate lifetime/
chunking investigation. Neither should block two-GPU readiness.

## Parameters and arithmetic scope

Native RT reuses the backbone weights and adds no parameters. The RT wrapper
still registers an inactive fusion branch; registration is not the trainable
or deployable architecture count.

| Scope | Resident registered parameters | Trainable / optimizer-owned | Deployable inference |
| --- | ---: | ---: | ---: |
| RT | 1,185,153,024 | 1,176,764,416 | 1,176,764,416 |
| K2 FBT + RT + NextLat | 1,267,879,936 | 1,267,879,936 | 1,185,153,024 |

Combined adds 8,388,608 fusion parameters and 82,726,912 training-only NextLat
parameters to the 1,176,764,416 backbone. At the candidate operating points,
the resource ledgers estimate 0.955–1.016 PFLOP of matrix arithmetic per RT
B192 update and 1.363–1.449 PFLOP per combined B128 update. They count
multiply-add as two FLOPs, include modeled backward/checkpoint reconstruction,
and exclude elementwise operations, optimizer, launch and communication costs.
Ordinary attention and checkpoint early-stop ranges are accounting assumptions,
not measured hardware bounds; these figures must not be labeled hardware FLOPs
or hardware utilization. Full component ledgers are preserved in raw reports.

Each RT B192 update contains 98,304 input tokens and 98,112 CE targets. Combined
B128 contains 65,536 original input tokens, 131,072 pass-token work, 65,408 CE
positions, 65,408 latent pairs and 32,768 selected KL triples per objective pass.
These masks are fixed by the existing fixture; full valid CE does not imply
that every valid triple is selected for KL. The combined CE/latent work spans
both passes. Token-throughput comparisons count original input once.

## Verification, retention and reproduction

The final explicit selection contains **22 reports: 16 passed, 3 retained
numerical failures and 3 OOMs; 153 physical optimizer updates; 1,269 verified
frozen-source pairs**. Of 103 recorded gates, 100 pass. All 98 recorded operational
gates pass; the OOM reports have incomplete inventories and are not cleared.
Two of five cross-configuration integration gates pass. The failures are Dao
ordinary RoPE in RT, FA4 in RT, and FA4 in combined. All three completed their
own operational diagnosis, six physical updates each, and remain failed.
Four compressed profiles (graph-only/full-step for each selected mode) are
verified by exact bytes/hashes and annotation-safe derivation.

Primary measured runtime is `a2bc709`; earlier integration uses `56f2dd1` and
`ff35047`, and early capacity uses `508920c`/`3fdad1e` with identical frozen runtime
sources. Exact per-run commits, source/dependency hashes, fixture hashes, W&B
links, update counters and failed memory phases are in [summary.json](summary.json).
The [usage notes](usage.md) explain arm choices and preparation. The [test ledger](test-results.txt)
records overlapping CPU scopes without adding them into an inflated distinct
total. Final reporting-hardening and evidence checks pass 125 tests (45 harness,
80 evidence). Reporting-only commits `4e1bff5`/`f583703` landed after measurements;
their changes do not constitute new GPU-runtime evidence.

The canonical W&B group is
[olmo-rt-large-batch](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/groups/olmo-rt-large-batch).
These probes do not retain their short-lived updated model weights; the original
native checkpoint is retained by verified reference. Raw success/failure
reports, source/dependency snapshots, logs and traces are retained with the
report. The accompanying [storage receipt](storage-receipt.json) records the
create-only GCS objects, downloaded SHA256 verification and checkpoint reference.

To reproduce the completed summary and plots in the explicit CPU container:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc '
python - <<"PY"
import json
from pathlib import Path
from scripts.summarize_olmo_rt_large_batch import main
selection = json.loads(Path("docs/reports/olmo-rt-large-batch/summary.json").read_text())
args = ["--runtime-commit", selection["runtime_commit"], "--runs"]
args += [row["name"] for row in selection["runs"]]
for row in selection["runs"]:
    args += ["--run-commit", row["name"] + "=" + row["runtime_commit"]]
args += ["--plot"]
main(args)
PY
'
```
