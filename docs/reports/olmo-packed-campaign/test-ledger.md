# Packed campaign readiness test ledger

Recorded 2026-09-29 from completed local reports and CPU logs. This ledger
separates model-boundary semantics, distributed operational correctness and
independent numerical qualification. Later T1024 actual-data recovery and
component-precision outcomes must be added only after their stages complete.
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

The scopes overlap; do not sum these counts as distinct tests. The final
20-test focus includes two new cases that replace `manifest.json` or
`documents.sqlite` after initial verification. Those ensure the bytes actually
opened cannot differ silently from the verified index. The broad run's warning
is Google API Core's announcement concerning a future grpcio minimum, not a
test failure.

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

## Evidence audit and remaining stages

The independent review verified all declared source-snapshot bytes for the
three GPU reports (70 files each) and the index report (eight files): **218
source/snapshot pairs**. Report hashes at review time are:

| Stage | Completed report SHA256 |
| --- | --- |
| index-01 | `63a425a88b2a32798fd8502b579dbe818c0548e449eb59bf9e9c3d19c253cb7b` |
| tiny-eager-01 | `7e10e3bfd9f6cec43e4b05ccc15850902ec3a6239cf352c6c44412127aaf0e6d` |
| tiny-graph-01 | `b7cc280b0d46f10d839a19763eedc0ceaa75de4d4fd0de643489bf3cb271ff56` |
| pretrained-graph-01 | `200efad282fa37c872434a550d4d868d025c3a78bc6272de17e2a484198b7655` |

Each of these four stages has a local `status: verified` retention receipt
with two cloud objects. That is recorded artifact-publication evidence; it is
not the still-pending actual-data checkpoint download/restart acceptance.

Pending at this ledger's initial publication: real-corpus T1024 write and
cloud-restored next-update comparison, cold T1024 DDP/capture with resident
Adam, short real-loader rate, and the per-loss BF16/FP32 component diagnostic.
These need their own completed reports. Production mixture/shuffling, long
quality training, H200, changed world size, sharding and interrupted in-flight
collective recovery are outside this milestone's current evidence.
