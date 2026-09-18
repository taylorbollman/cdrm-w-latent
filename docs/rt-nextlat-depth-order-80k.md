# A5 NextLat: four-layer Transformer and first-window RT, 80k each

**Training, saved-state audits and report audit are complete. Both fresh jobs
exited successfully at exactly 80,000 updates on 2026-09-14; no further
experiment has been launched.** The result and qualifications are in
[outcome.md](reports/rt-a5/nextlat-depth-order-80k/outcome.md), with the
[full comparison](reports/rt-a5/nextlat-depth-order-80k/report.md) and
[W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/tb8ey88n).

The first-window RT has zero observed errors on both full development sets
at each retained checkpoint from 30k through 80k. At 80k every state in all
102,400 length-36 words is correct. SEQ4 has E(14)=65.4463%, E(16)=10.6631%,
M(36)=40.0371% and E(36)=0. The two historical RT controls also have E(36)=0.
This is a one-seed development comparison. The RT window-order comparison
preserves initial learned tensors and parameter count; SEQ4 has more
parameters. There is no first-window RT control without NextLat yet.

The user authorized two fresh experiments, in this order, without another
approval pause: **our four-layer ALiBi Transformer + NextLat**, then **two-layer
RT + NextLat with window 2 in layer one and full recurrent attention in layer
two**. Both prospectively fixed 80,000-update runs are complete, in the
authorized order. Do not extend either budget without a new instruction.

The completed lineage is `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/`.
The pointer is `.runtime/rt-a5/current-depth-order-lineage.txt`.
`protocol.json` freezes the two resolved contracts, initialization metadata,
arguments, data identity, references and 55 source files. Its SHA256 is
`b036355978bac1c417a315b55f63fe7ab9661462cdfcf9d1d49b1454cf7d8f80`;
the source digest is
`cc9fbfa50ebd57387f4ad95803bb2db9d42b69a13875268c0400759858fe4176`.
Keep all 55 model/training/configuration files unchanged until both jobs and
their reporting are complete. New reporters and runtime audit helpers can be
added outside that manifest.

The sequential coordinator started on 2026-09-14 at 21:37:11 UTC, PID 69980.
It runs `seq4_alibi` into `train-seq4/`, followed by `rt_window2_first` into
`train-window-first/`. The first run is online at
[W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/4dfumcec).
Its launcher PID was 69982 and initial trainer PID was 70076;
`seq4_alibi-container.json` records the exact container. Confirm current
process identity before recovery or termination. Saved PIDs alone are not
proof that a job is still active.

SEQ4 completed exactly 80,000 updates with exit zero and W&B synced on
2026-09-14. Its final development whole-word accuracy is 95.1816%; on the
102,400 length-36 development words, prefix exactness E(13)/E(14)/E(16) is
87.1016% / 65.4463% / 10.6631%, and mean token accuracy is 40.0371%.
No entire length-36 word was correct. The four-arm report and final
saved-state audits are now complete.

The coordinator launched RT-first at 22:23:49 UTC, launcher PID 87631.
Its online run is [W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/m2klbr0d).
The read-only `rt_window2_first-audit-container.json` records container
`f6e52c739df5aecc96167849f379af77a2ce357eb0aea2a54fd828e83465d614`
and host init PID 87723. It trained from update zero through 80,000 and
exited successfully at 23:40:42 UTC with W&B synced.

`run_sequential.py` checks source hashes before each launch and requires the
first run to complete exactly 80k, sync W&B and retain the expected contract,
initialization and checkpoints before launching the second. It refuses an
existing output directory and will not silently resume a failed job. It writes
`sequence-status.json`, per-arm launcher records and atomic exit markers
`seq4_alibi-exit-code.txt` / `rt_window2_first-exit-code.txt`. Do not start a
duplicate trainer after an agent interruption. The host-only `status.py` reads
recent progress and the retention status without importing Torch or touching
the GPU.

Both models use width 512, eight heads, GELU FFN width 2048, LayerNorm with
learned full-width Q/K normalization, ALiBi, vocabulary 60, untied embedding
and head, no biases or dropout, and the original Mitchell initialization
rule. There are no RoPE, sinusoidal or learned positional embeddings.

| Variant | Backbone | NextLat predictor | Total parameters | Learned tensors |
| --- | ---: | ---: | ---: | ---: |
| Four-layer ordinary Transformer | 12,653,056 | 1,049,600 | 13,702,656 | 39 |
| Two-layer RT, short window first | 6,357,504 | 1,049,600 | 7,407,104 | 25 |

The four-layer model is a fresh canonical four-layer Mitchell draw with seed
1234. Equal seeds do not imply identical two-layer backbone tensors. Its
predictor retains the original class, configuration, seed 1235 and initial
tensors. This comparison holds width and training exposure fixed; it does not
match parameter count or compute. The reordered RT retains every original
RT2 learned tensor at its original layer index and changes only which block
uses the existing parameter-free two-record mask. Layer one reads temporary
self K/V and its preceding output's permanent K/V. Layer two retains full
prefix RT attention. Recurrent gradients remain attached throughout; the
short direct-read window does not truncate effective memory to two tokens.

The shared recipe is unchanged: batch 1024, length 12, 800,000 unique A5
training words, data-order seed 1234, constant AdamW learning rate 1e-4,
betas (.9, .95), epsilon 1e-8, matrix weight decay .01 / vector decay zero,
and one global gradient clip at 1. Full FP32 and math attention are used;
autocast, TF32, compilation and CUDA graphs are off. Each 80k endpoint sees
81,920,000 word presentations, or 102.4 nominal passes. The corpus is the
existing `.runtime/rt-a5/20260911T154748Z/data/`; its manifest is pinned in the
protocol.

The original joint update, optimizer, data-order implementation, backbone
evaluator and one-step diagnostic functions are reused as the same function
objects. The loss is state CE plus weight-one next-latent SmoothL1 with beta 1.
Only target-role latents are detached. Source latents and the next-operation
embedding remain attached. Accuracy evaluation uses the backbone; autonomous
NextLat MLP rollout remains disabled and unevaluated.

Checkpoints are saved at 0, 1k, 5k, 10k, 20k, 25k, 30k, 40k, 50k, 60k, 70k
and 80k. Routine development evaluation every 500 updates uses 4,096 words;
full checkpoint evaluation uses 102,400 words per role. The diagnostic sample
has 1,024 words. Final confirmation is untouched. E(t) requires all states
through t correct, A(t) checks state t alone, and M(t) averages token accuracy
through t. Full and boundary curves must use the same length-36 output rows.

The reference models are the accepted 80k original full-attention RT + NextLat
and the accepted 80k RT with window 2 in its second layer. Their original 100k
plans were superseded by user-directed stops after development review. The
full model's raw 1,607-update tail and the second-window model's raw
5,376-update tail are preserved and excluded. Neither has a retained 100k
outcome. The two revision records and hashes are pinned in this new protocol.
Both historical 80k endpoints are retrospective development selections;
the two new 80k endpoints are prospective. This remains one seed on reused
development data.

Validation before launch passed 12 model tests, five trainer integration tests
and the bounded GPU preflight. The first-window RT matched the existing
independent two-record oracle at lengths 12 and 36 using the inherited FP32
tolerances. Both actual D512/B1024 models completed ten discarded valid-A5
updates with finite model, gradient and Adam states, followed by length-36
output/causality checks. The successful GPU receipt is `preflight/report.json`,
with [preflight W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/p0oew0p5).
Use the script invocation `python scripts/rt_a5_depth_order_validate.py` if
reproducing it; a preceding module invocation failed on a legacy import path
before any model ran. The earlier CPU test attempt used an unsupported D64
default predictor fixture; the final D128 fixtures passed, without a change
to production model behavior.

All GPU work must use the project Docker launcher and verify the container
working directory plus `nvidia-smi` inside it. CPU report/audit work uses
`CDRM_DOCKER_GPUS=none`. The new implementations are
`scripts/rt_a5_depth_order.py`, `scripts/rt_a5_depth_order_train.py` and
`configs/rt_a5_depth_order/base.json`. Historical 52-source code remains
unchanged. The GPU validator is outside the training manifest.

The final report is complete at `docs/reports/rt-a5/nextlat-depth-order-80k/`.
Its reporter, `scripts/rt_a5_depth_order_report.py`, passed 18 focused tests,
an actual saved-artifact preflight and independent source review. Reporter
SHA256 is `17f223e8d1cf975024ed79bc69cde0faf36e3eeeb0cfd0ca09559170cca7eb63`.
The finalizer completed at 23:41:45 UTC (launch PID 83485). After both
successful 80k exits it passed both final CPU saved-state audits and
generated the report.
Its 16 dependencies are pinned in `finalizer-source-freeze.json`; it cannot
train, resume or run model inference. Check `finalizer-status.json` and
`report-exit-code.txt` after an interruption before starting recovery.

Both SEQ4 and RT-first initialization passed 37 saved-state checks each. The helper is
`validate_saved_state.py` in the lineage, SHA256
`45169e7ca5b3a2b40f6bdc5f69d5a3a0dca7ede8dcb4936007d3814c7d5e7638`.
The RT-first initial receipt/log are
`rt_window2_first-initial-state-validation.json` / `.log`; all 25 learned
tensors exactly match the original RT2+NextLat initialization, and Adam is
empty. Both initial audits used the GPU-disabled container.
Do not preempt the finalizer's final audits, which write separate append-only
receipts after both jobs complete.

The host-stdlib independent artifact audit passed: 2,112 metric rows,
88 checkpoint evaluations, 48 checkpoint hashes, 306 training-source
snapshots, 3,200 training bins, all 80,000 data-order hashes per arm,
68 original report artifacts, and both historical stop qualifications.
Its receipt is `evidence-audit.json`. Each final saved-state audit passed
40 checks, including finite FP32 model/Adam tensors and counters at 80k.

Root inspected the full curves, checkpoint curves and readable boundary
figures. The original zoomed figures have clipped tick labels; supplemental
`length-boundary-readable` and `length-boundary-10k-readable` PNG/PDF files
use fixed margins and exactly the same audited metric rows. The renderer
is `render_readable_boundaries.py`; `readable-boundaries.json` binds its
source, inputs, 288 checked rows and 24 plotted curves. No original report
or frozen source was changed. The outcome note links the readable figures.

All 24 checkpoints and both final reports/histories were retained and
verified at 23:44:16 UTC (`retention-complete.json`). Final archive and
readback receipts are `evidence-storage.json` and `evidence-readback.json`
in the lineage; inspect those receipts for final archive/readback status.
Keep archive command logs outside the archived lineage, and do not change
archived members afterward. Check `root-final-review.json` for final local
review and trainer-exit confirmation.

Retention destination:
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T212935Z-nextlat-depth-order80k/`.
