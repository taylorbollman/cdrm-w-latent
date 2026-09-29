# Packed campaign readiness test ledger

Recorded 2026-09-29 from completed local reports and CPU logs. This ledger
separates model-boundary semantics, distributed operational correctness and
independent numerical qualification. Index cloud recovery and the bounded
precision-component diagnostic and T1024 actual-data write/continuation are
complete. The first T1024 fresh-process restart **failed bitwise continuation**;
isolated Flash tests identify missing deterministic-backward controls as a
plausible cause. The corrected deterministic write/continuation now passes;
its fresh-process resume remains pending, with no full restart pass yet. The
original failure remains retained.
See [protocol](protocol.md), [usage](usage.md) and [progress](progress.md).

## CPU checks

CPU tests ran in the project container with GPU passthrough disabled. These
checks do not substitute for CUDA/NCCL or cold-start memory qualification.

| Scope / evidence | Result |
| --- | --- |
| Initial integration, `cpu-integration-01.log` | 33 passed in 8.68 s |
| First broad command, `cpu-regression-01.log` | No tests ran: nonexistent `tests/test_fbt*.py` glob; command-selection failure retained |
| Corrected broad regression, `cpu-regression-02.log` | **971 passed in 75.32 s**, one dependency warning |
| Final packed-data focus after index-verification race hardening | **20 passed in 3.40 s**, focused agent-run output |
| Deterministic runner and Flash probe, `cpu-determinism-01.log` | **18 passed in 2.24 s** |

The scopes overlap; do not sum these counts as distinct tests. The final
20-test focus includes two new cases that replace `manifest.json` or
`documents.sqlite` after initial verification. Those ensure the bytes actually
opened cannot differ silently from the verified index. The broad run's warning
is Google API Core's announcement concerning a future grpcio minimum, not a
test failure.

The 18-test deterministic scope combines the six packed-runner tests and 12
Flash-repeatability probe tests. It overlaps the separately run six-test
runner focus and existing tests; it is not 18 additional disjoint regression
tests. New runner coverage checks deterministic setup before any CUDA device
initialization, rejection of already-initialized CUDA, returned control
metadata and the helper's presence in source pins.

New coverage includes independent stream masks/gradients for all eight arms;
policy agreement and unchanged isolated defaults; literal EOS at chunk edges
and inside documents; detached auxiliary targets; causal/no-cross-row feedback;
changing true boundaries without recapture; complete-document provenance and
counts; bounded token reads/FD cache; partition-invariant membership and jitter;
pure peeking, committed-cursor recovery and rejection of clock/partition drift;
mutated corpus/index rejection; atomic index publication; and CLI recovery pins.

## Complete real-corpus index check

`index-01/report.json` passes all **five checks** in 49.8735 s. It is CPU data
validation, not a model update. The selected existing train split contains
**12,283 unique documents, 6,947,277 tokens and 6,785 chunks** at T1024. The last
chunk has 461 valid tokens and 563 padding positions. There was no retokenizing,
resplitting, shuffling, cycling or production-mixture selection.

| Recorded check | Result |
| --- | --- |
| `immutable_index_published` | Pass: new immutable SQLite/manifest index |
| `every_token_true_document_id_and_chunk_count_matches_independent_iter_documents_oracle` | Pass: all 6,947,277 tokens, true document IDs and 6,785 chunk counts agree |
| `pure_peek_full_524288_valid_token_schedule` | Pass: 14 planned updates; reader remains at chunk/update zero |
| `world2_b12_first_two_updates_preserve_membership_masks_and_global_counts` | Pass: actual first two updates materialized and checked; no jitter allocation in this CPU check |
| `source_snapshots_match_execution_sources` | Pass: eight source files |

The finite schedule is thirteen updates of 524,288 valid input tokens followed
by one update of 131,533 tokens. The final update contains 129 real rows.
For each of the first two full updates, B12/rank on two ranks produces 22
physical slots per rank: rank 0 has 260 real rows and 4 dummy rows; rank 1 has
252 real rows and 12 dummy rows. Rank 1's last synchronized slot is wholly
empty. Global objective counts still use actual valid targets.

| Whole selected stream accounting | Count |
| --- | ---: |
| Valid input tokens | 6,947,277 |
| CE targets | 6,940,492 |
| Latent pairs | 6,928,229 |
| KL triples | 6,909,206 |
| Cross-document CE targets / excluded boundary latent pairs | 12,263 |
| Excluded boundary KL triples | 24,501 |
| Omitted cross-chunk CE / latent / KL targets | 6,784 / 6,765 / 13,505 |
| Document segments / completions | 19,048 / 12,283 |

Packed-index manifest SHA256:
`372a7e05f5164198bdeb1531fc45bd761f5d33222d424c6804b54f19fd75288d`.
This is the file-byte pin used by the runner, not its separate canonical
identity or document-order digest.

`index-restore-evidence-01/report.json` subsequently passes **three CPU checks**
in 3.2267 s: full-byte verification of exact-generation evidence/manifest cloud
downloads; safe extraction and SHA256/size verification of all 14 inventory
members; and equality of the restored index identity, first-update chunk keys,
counts and cursor. The restored manifest has the same SHA256 above, and its
first update contains the same 512 chunk keys. A hypothetical completed-update
cursor at chunk 512/update 1 restores exactly into a fresh reader. This is
index/data-cursor recovery only: no optimizer step or GPU model restart ran in
this CPU stage.

## Actual two-H100 execution checks

All three completed GPU probes use `continuous-stream-v1` and actual NCCL on
two H100 80 GB GPUs. Recorded runtime: Torch
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, driver 580.178.04. Tiny probes use
B2/rank, T8, FP32 math attention and eager native RT. The pretrained probe uses
B2/rank, T16, BF16 mixed precision with FP32 masters/gradients/Adam, forced
ordinary Flash SDPA, native Triton RT recomputation and ordinary activation
checkpointing. TF32 remains disabled.

| Stage / W&B | Operational result | Independent reference | Duration |
| --- | --- | --- | ---: |
| [tiny-eager-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6frhahk0) | **88/88 gates pass** | Canonical selected-position reference passes, all eight arms | 26.3122 s |
| [tiny-graph-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nclpkyza) | **88/88 gates pass** | Canonical selected-position reference passes, all eight arms | 25.1633 s |
| [pretrained-graph-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/kcqs2y3d) | **22/22 gates pass** | B passes; **NFR fails** its separately recorded initial canonical/prepared qualification | 141.9895 s |

Arms are B, N, F, R, NF, NR, FR and NFR, where B is ordinary OLMo, N is
NextLat, F is feedback (K4 FBT) and R is native temporal RT. Each arm has 11
operational gates: three independent-reference updates, preparation, three
raw-gradient comparisons, three complete Adam updates and final three-update
parity. The pretrained probe has two additional non-gating qualification rows;
its 24 total rows are **not** 24 passing gates. All three W&B records are synced.

Preparation performs 20 backward warmups per arm, without optimizer updates.
Graph probes additionally capture two backwards. Actual distributed training
performs three complete optimizer updates per arm: 24 updates per tiny stage
and six in the pretrained stage. Independently calculated reference updates
are separate diagnostic work, not extra training steps in that trajectory.

Fixtures exercise M1/M2/M3 accumulation, unequal target counts, moving packed
boundaries, masks/jitter changes, a wholly empty rank and empty final sync
slots. The graph checks require stable storage and unchanged RNG during
preparation/refills. Replicas, counters and scheduler agree exactly after the
three updates. Numerical budgets are those frozen in the protocol.

| Comparison | Maximum raw-gradient relative L2 | Parameter-update relative L2 after three updates | Adam-moment relative L2 |
| --- | ---: | ---: | ---: |
| Tiny eager, all arms | `4.51232672e-7` | `5.64338551e-6` | `3.56911802e-7` |
| Tiny graph, all arms | `4.51232672e-7` | `5.64338551e-6` | `3.56911802e-7` |
| Pretrained B graph versus local prepared reference | `6.11632014e-9` | `8.04026492e-7` | `7.76493325e-8` |
| Pretrained NFR graph versus local prepared reference | `9.21959115e-9` | `2.12203051e-8` | `7.32152908e-9` |

Parameter-update error is normalized by the actual reference update, not by
the much larger pretrained parameter norm. These operational comparisons
qualify distributed execution of the chosen arithmetic layout.

## Retained numerical qualification

The packed pretrained NFR initial canonical/prepared comparison fails the
unchanged raw-gradient budget with relative L2 **0.01695301636481997** (about
1.6953%). It is calculated with identical initial weights, physical rows and
noise, locally without DDP or CUDA graph replay. The loss-count and loss-value
checks pass: normalized objective absolute difference is `6.5109946501e-6`.
The ordinary B qualification passes with bitwise-equal raw gradients.

This does not resolve the earlier isolated-fixture **3.40224%** BF16 discrepancy.
The packed test uses different document boundaries/masks and is not a matched
before/after improvement. Read `operational_status` together with
`independent_reference_status` and `qualification_failures`; the top-level
operational pass is not full BF16 numerical clearance. No threshold changed,
and no claim of harmlessness for learning follows from the local/DDP agreement.

## Completed component and cross-precision diagnostic

[precision-components-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/6a48mv6r)
completed **12 backward probes plus one fixed-weight/source/RNG integrity
check**, all 13 operational rows passing, in 77.4641 s. W&B is synced. It uses
one initial pretrained NFR state, T16/B2, two virtual rank fixtures and the
original `isolated-v1` policy. There is no DDP, CUDA graph, optimizer update or
new checkpoint. Its status `passed_operational_diagnostic` does not mean
precision acceptance. Full interpretation and follow-up recommendations are
in [precision-assessment.md](precision-assessment.md).

| Objective | BF16 prepared versus sparse relative gradient L2 | BF16 sparse versus FP32 | BF16 prepared versus FP32 |
| --- | ---: | ---: | ---: |
| Combined | 3.40224% | 85.96% | 86.13% |
| CE only | 0% | 95.93% | 95.93% |
| Latent only | 2.17298% | 104.17% | 104.35% |
| KL only | 2.28388% | 71.95% | 72.55% |

The original 3.40224% layout discrepancy reproduces. CE alone agrees exactly
between the two BF16 loss layouts; either NextLat auxiliary contribution can
introduce the discrepancy. The combined gradient cosine between BF16 layouts
is 0.999422. Combined-backward rounding differs from separate component
backwards, so these component errors must not be added to apportion the total.

The **approximately 86% combined cross-precision difference is a larger,
additional qualification**: cosine against FP32 is about 0.5112 for BF16 sparse
and 0.5083 for BF16 prepared. It is not explained by the small mutual layout
difference, and CE alone already has a large cross-precision gap. However,
precision **and kernels change** in this comparison: BF16 uses ordinary Flash
SDPA and Triton native RT; full FP32 uses math SDPA and eager native RT. These
measurements do not isolate BF16 rounding or identify a specific faulty
operation. Old-budget flags are descriptive; no new cross-precision acceptance
threshold was introduced. Neither this diagnostic nor the operational packed
checks establish harmlessness for learning. A matched-kernel precision bridge
and fixed-input/cotangent localization remain follow-up work.

## Actual-data T1024 write and live-graph continuation

[pretrained-write-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8c13f3ne)
passes **13/13 gates on both ranks** in 835.5506 s; W&B is synced. The actual
NFR model has 1,267,879,936 resident/trainable parameters and uses B12 per rank,
T1024, K4, RT0/15 and both NextLat losses. Each update consumes 524,288 valid
input tokens through 22 accumulated physical slots per rank.

The gates cover preparation preserving model/Adam/RNG/clocks/cursor; raw
gradient replicas, model/Adam replicas, finite state, RNG/committed cursor and
actual-data counts for each of two updates; checkpoint publication preserving
the live-graph boundary; and unchanged source pins. Report-level independent
audit additionally confirms both updates' complete counts against the separate
index oracle, rank accounting summing to global counts, exact replica state
and metrics, correct cursor/token clocks, changed actual inputs/model state
between updates, and rate arithmetic.

| Quantity | First update | Original live-graph continuation |
| --- | ---: | ---: |
| Valid input tokens | 524,288 | 524,288 |
| CE / latent / KL targets | 523,776 / 523,768 / 523,248 | 523,776 / 523,770 / 523,252 |
| Cross-document CE targets | 8 | 6 |
| Committed next chunk / update | 512 / 1 | 1,024 / 2 |
| Cumulative input tokens | 524,288 | 1,048,576 |
| Global physical microbatch presentations | 44 | 44 |
| Loader and keyed-jitter time, slower rank | 7.8228 s | 7.9527 s |
| Backward/NCCL time, slower rank | 123.7015 s | 123.9474 s |
| Adam/scheduler/cursor-commit time, slower rank | 0.5971 s | 0.5764 s |
| Total timed segments, slower rank | 132.1213 s | 132.4765 s |
| Global valid input throughput | **3,968.23 tokens/s** | **3,957.59 tokens/s** |
| Gradient norm before clipping | 228.8580 | 196.2972 |
| Learning rate used | `2.0e-5` | `2.18e-5` |

Summing the slower rank's complete timed segments across both updates gives
264.5978 s and **3,962.91 valid input tokens/s**. Component-wise maxima in the
table need not sum exactly to the maximum rank's total. Timing includes CPU
token reads/jitter, validation/refills, graph backward/NCCL, clipping, Adam,
scheduler and cursor commit. It excludes diagnostic hashing/gates, extra
health scans, W&B and checkpoint I/O. It is a two-update directional rate, not
the whole 835.55-second launch amortized into steady training. Clip norm 1.0
and the token-based warmup schedule are the recorded recipe, unchanged here.

Both ranks report 33.35183 GiB peak allocated, 58.89844 GiB peak reserved and
14.39661 GiB sampled free after the updates. Immediately after capture and
before the first optimizer step, reserved memory is 48.89648 GiB and free
memory is 24.39856 GiB. **Adam was not resident before DDP construction in this
write phase**; these numbers do not qualify cold resume with resident moments.
Runner metadata shows 20 warmup backwards, two captures, 42 local graph replays
and two synchronized replays per rank. No recapture occurs between updates.

The checkpoint is at the first update's exact boundary. Its saved model/Adam,
scheduler/counter/RNG/cursor digests equal those recorded for that update; the
reference continuation equals the second live-graph update. Reported state
size is 15,214,757,825 bytes, state SHA256
`7b5e948eea2f1102676b26b4d9b883df398fad06c4b0523f3b0c79d12584b830`,
and manifest SHA256
`9238b186a33c80be85aa18aec11cc2ea15f097e137be34eece4e978ca2886a50`.
The checkpoint was subsequently published through `checkpoint-boundary-01`
and downloaded into a fresh directory. `checkpoint-restore-evidence-01`
records exact-generation full-download SHA256/size checks for both state and
manifest, totaling 15,214,815,890 bytes. The restored manifest has the pin above.
This verifies checkpoint bytes; the model-continuation result below is a
separate, failed acceptance gate.

The first two real updates have only eight and six internal document
boundaries, respectively. Boundary-rich tiny probes and the full-stream CPU
oracle provide complementary coverage; these two updates alone do not measure
representative production packing or short-document throughput. Neither finite
updates nor falling scalar objectives resolve the retained BF16 qualifications.

## Retained failed restart and deterministic Flash follow-up

[pretrained-resume-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/igpkw3gd)
finishes **failed**, with **8/9 gates passing**, in 663.6377 s. Configuration,
restored model/Adam/RNG/clocks/cursor, preparation invariance, actual next data
counts and within-run replicas pass. The final original-versus-restored next
update is not bitwise equal on either rank. W&B records the experiment as
`synced_failed_experiment`; its evidence and checkpoint remain retained.

Independent report comparison finds identical next input/noise fingerprints,
RNG draws, CE/latent/KL loss sums and normalized objective. All 65 backbone
and two fusion raw-gradient hashes differ; all four predictor hashes match.
The only scalar metric difference is preclip norm:
`196.2971954345703` versus `196.29531860351562`. Adam/updated-parameter hashes
then differ. This approximately `9.56e-6` relative norm change does **not** bound
the gradient-vector error; only digests, not raw tensors, were retained here.

The cold resume did load real Adam before DDP preparation and successfully
complete capture/update. After capture it reports 42.80221 GiB peak allocated,
58.34961 GiB peak reserved and 14.94543 GiB sampled free per rank; after the
update, peak reserved is 59.02734 GiB and sampled free is 14.26770 GiB. This
establishes observed memory fit for that failed-numerics attempt, not successful
recovery. New deterministic execution needs its own acceptance and memory data.

Four isolated BF16 ordinary Flash SDPA probes then hold Q/K/V and incoming
gradient fixed at B12, 16 heads and head dimension 128. Each performs three
eager backwards and three graph replays. All inputs, generator state, forwards
and source pins remain unchanged. These are single-operator probes, with no
model, RT, NextLat, DDP or optimizer.

| Stage / W&B | Deterministic algorithms | Backward repeatability | Duration |
| --- | --- | --- | ---: |
| [flash-t16-d0-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4rmeljzu) | Off | All eager/replay gradients bitwise equal | 2.7856 s |
| [flash-t16-d1-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/zul4zvyk) | On | All eager/replay gradients bitwise equal | 2.8525 s |
| [flash-t1024-d0-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/u9l20orh) | Off | **Not repeatable** in eager or repeated graph replay | 9.0160 s |
| [flash-t1024-d1-01](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nsnyngyf) | On | All eager/replay gradients bitwise equal | 7.7434 s |

All four have six finite/integrity rows passing. With deterministic mode off,
the status `passed_operational_diagnostic` means the measurement completed; it
does not require repeatability. In the T1024 off case, only query gradients
change: up to 451 elements differ, maximum query-relative L2 is `7.16686e-6`
and aggregate Q/K/V relative L2 is `3.91786e-6`. K/V gradients and forwards
remain exact. With deterministic mode on, every reported error is zero.
The short T16 success therefore could not establish T1024 repeatability.

The existing deterministic-control helper is now required **before CUDA
initialization** in the packed recovery runner, and its returned settings are
recorded in report/W&B/checkpoint metadata. Source `e5a593b` contains the runner
fix and microprobe source `2080c11` records its completed implementation.
The changed source/configuration requires a new matched write/resume pair;
the original failed comparison is not retested against incompatible pins.
See [restart-repeatability.md](restart-repeatability.md) for diagnosis and
scope. The isolated probes support this correction, but do not by themselves
prove a full-model restart fix or resolve the distinct BF16/FP32 discrepancy.
The completed deterministic write below does not by itself qualify recovery;
no full deterministic restart success is claimed in this ledger version.

## Corrected deterministic write and continuation

[pretrained-write-02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/k5ynlpin)
passes **13/13 gates on both ranks** in 965.5435 s; W&B is synced. It repeats
the same actual first two packed updates, B12/rank, T1024, K4/RT0,15/NextLat
and 524,288 valid inputs per logical update with explicit deterministic
controls. Report, execution configuration and saved checkpoint metadata all
record `deterministic_algorithms=true`, `cudnn_deterministic=true`,
`cudnn_benchmark=false` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Independent audit verifies all 77 source snapshots and repeats the first
attempt's count/cursor/state audit: both updates exactly match the independent
index oracle, rank counts sum to the global counts, raw gradients/model/Adam
and metrics match between replicas, saved boundaries equal the first update,
and the recorded reference continuation equals the second live-graph update.
The cursor advances to chunk 512/update 1 and then chunk 1,024/update 2, with
524,288 and 1,048,576 cumulative valid inputs respectively. Each update uses
22 physical microbatches per rank, with the same last-slot dummy rows and
523,776 CE targets. Latent/KL counts remain 523,768/523,248 for update one and
523,770/523,252 for update two.

| Quantity | First update | Original live-graph continuation |
| --- | ---: | ---: |
| Total timed segments, slower rank | 149.8208 s | 149.7915 s |
| Loader and keyed-jitter time, slower rank | 9.0142 s | 9.0451 s |
| Backward/NCCL time, slower rank | 140.1072 s | 140.0872 s |
| Adam/scheduler/cursor-commit time, slower rank | 0.6994 s | 0.6592 s |
| Global valid input throughput | **3,499.43 tokens/s** | **3,500.12 tokens/s** |
| Gradient norm before clipping | 229.0050 | 197.3801 |

Across the two updates, 1,048,576 valid inputs over 299.6123 timed seconds give
**3,499.78 tokens/s**. The timing scope retains the preceding table's diagnostic,
W&B and checkpoint exclusions; this is not steady-state throughput over many
updates. The observed rate is lower than the first nondeterministic pair, so
the old rate should not be presented as the corrected recovery path's rate.

Peak allocated/reserved memory remains 33.35183/58.89844 GiB on each rank,
with 12.94739 GiB sampled free after either update. After capture, reserved
memory is 48.89648 GiB and sampled free is 22.94934 GiB. As in write01, Adam
was not resident before DDP preparation; cold deterministic resume is still
required. Saving again preserves the original live-graph boundary exactly.

The new first-update checkpoint reports 15,214,758,081 state bytes, state SHA256
`c27186bb4847d4325821ef96916e85e69e3c9246a73eb4886d343941467e1199`
and manifest SHA256
`c1cdb79fc58b767160bf6466777b434665f4271f731e03698549b28d85e74ed1`.
Its separate `checkpoint-boundary-02` upload, cloud download verification and
fresh-process resume have not yet been accepted in this ledger version.

## Evidence audit and remaining stages

The independent review verified all declared source-snapshot bytes for the
three distributed GPU reports (70 files each), index report (eight files),
component diagnostic (72 files), index-cloud-restore report (two files),
original actual-data write and failed resume reports (76 each), four Flash probes
(six each), checkpoint-boundary report (76) and checkpoint-restore report
(two), plus the corrected deterministic write (77): **623 source/snapshot
pairs**. Report hashes at review time are:

| Stage | Completed report SHA256 |
| --- | --- |
| index-01 | `63a425a88b2a32798fd8502b579dbe818c0548e449eb59bf9e9c3d19c253cb7b` |
| tiny-eager-01 | `7e10e3bfd9f6cec43e4b05ccc15850902ec3a6239cf352c6c44412127aaf0e6d` |
| tiny-graph-01 | `b7cc280b0d46f10d839a19763eedc0ceaa75de4d4fd0de643489bf3cb271ff56` |
| pretrained-graph-01 | `200efad282fa37c872434a550d4d868d025c3a78bc6272de17e2a484198b7655` |
| precision-components-01 | `f6bf376ea511e9c853984f7290872cb24ae06bdb98b55130e9238b79261cf405` |
| index-restore-evidence-01 | `625218c61f111799c33d2745bc3fec529e797689eaeaa2fc8920c45e9976cf8c` |
| pretrained-write-01 | `960e65563ac4b66a51197546cb14e8eccd6d457bfdb32838c8136333d5c8ce38` |
| checkpoint-boundary-01 | `4debc7b7aee9cecb818f93acbad78c6c13b7da27b65d5f713cc5b5e2a1c79193` |
| checkpoint-restore-evidence-01 | `cca970fe759c30db934dda130a04a595a344873b459ec26a4acf8d878b2768db` |
| pretrained-resume-01 | `a8929837ca79b3db607999d1cc6eb20dae37270292f1cae1d241ee4b08389fcd` |
| flash-t16-d0-01 | `4f57516051f9f9b59a8b8bca2b1a7042c95431da038f2e0966aee8b76650b191` |
| flash-t16-d1-01 | `878c93f4379fdcb29c5bba75aa3ee65a3fcbb06fe44db300cc7c0d9db8757ea5` |
| flash-t1024-d0-01 | `40c16536338c371ef1f875344d8fe9db6d4fc0eee2662f20360dd1aa0825e2e1` |
| flash-t1024-d1-01 | `6931327b49cacf6f9db3692eb5b880f5fe1b59d743ac6fc8b67a1051bd676937` |
| pretrained-write-02 | `a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754` |

The listed stages preceding `pretrained-write-02` have local
`status: verified` retention receipts,
including the failed resume and separately published checkpoint boundary.
Cloud publication is not numerical or restart acceptance. The original model
checkpoint has separately verified downloaded bytes, while its full next-update
comparison failed as recorded above.

Pending in this ledger version: corrected-write/checkpoint retention and
full-download verification, the deterministic real-corpus T1024 cloud-restored
next-update comparison, and corresponding cold-setup memory/timing evidence.
These need their own completed reports. Production mixture/shuffling, long
quality training, H200, changed world size, sharding and interrupted in-flight
collective recovery are outside this milestone's current evidence.
