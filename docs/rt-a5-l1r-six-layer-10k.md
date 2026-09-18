# Six-layer L1R RT + NextLat: 10k depth diagnostic

**The initial10k run and report are complete.** Six-layer length36 whole-word
accuracy is82.0771%, versus85.7910% for the matched two-layer10k reference.
Six-layer length12 whole-word accuracy is99.7861%; length36 token accuracy
is97.0899%. The gap is3.7139 percentage points in length36 whole-word accuracy.
Added depth retained substantial state tracking in this bounded diagnostic.

Read the [10k outcome](reports/rt-a5/l1r-six-layer-nextlat10k/outcome.md),
[learning curves](reports/rt-a5/l1r-six-layer-nextlat10k/whole-word-vs-updates.pdf),
[full length curves](reports/rt-a5/l1r-six-layer-nextlat10k/length-full.pdf), and
[readable boundary supplement](reports/rt-a5/l1r-six-layer-nextlat10k/length-boundary-readable.pdf).
The frozen original boundary image has clipped ytick labels; the supplement
uses identical saved rows with corrected margins, preserving the original.
[Comparison W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/8ymcev0t).

Final saved-state checks passed: all61 model/Adam states are finite FP32,
all Adam counters are10000, and the four predictor initialization tensors
exactly match the reference. The10k checkpoint SHA256 is
`9f6e330741b5f363a46657df3f1bc34d1459e528e22dcd8d23b9c5753826ba6b`,
bytes240087811. The reporter verified all10000 matching minibatch-order hashes
and288 E/A/M metric rows. Root inspected the four plots and corrected boundary
supplement. `root-final-review.json` records closure checks; archive/readback
receipts are retained as companions to this stable initial-stage evidence.

**User extension:** while this run was around8.7k, the user authorized
continuing to20k unless they return and request an earlier stop. The exact10k
checkpoint automatically resumes in the separate lineage
`.runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k/`.
See [the continuation handoff](rt-a5-l1r-six-layer-20k.md). The initial10k
protocol/report remains intact; it is an intermediate comparison now.

The user confirmed **six total layers**: window2 at layer0, followed by five
full RT layers. Train one fresh model for exactly10000 updates and compare
its1k/5k/10k checkpoints with the successful two-layer L1R RT + NextLat at
the same update budgets. This paragraph describes the initial authorization;
the explicit continuation above supersedes its stopping point.

Current lineage: `.runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k/`.
Pointer: `.runtime/rt-a5/current-l1r-depth-lineage.txt`.
Training launched2026-09-15 at14:20:16 UTC, coordinatorPID11099,
launcherPID11100. Exact trainer container:
`9f72960e87683a4df5e6a993ee072f7d308622cf4e49f559092ccf734942fb30`.
An unrelated interactive container may also be running; do not stop it.

[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/fg4quqku)
is under `taylorbollman/rt-a5-state-tracking`. Check `launch-status.json`,
`training.log`, and `train-depth/report.json` before acting after interruption;
do not launch a duplicate. Host-only `status.py` reads progress. Training has
its own exact10k endpoint; `launch.py` never automatically retries or resumes.

`finalize.py` waits for successful training completion, then runs the saved-state
checker and report in CPU containers. Its launch/PID and nine frozen dependencies
are recorded in `finalizer-launch.json` and `finalizer-source-freeze.json`.
Inspect `finalizer-status.json` before any manual rerun. Its successful endpoint
still requires root figure/handoff review and final GCS archive/readback.

The backbone keeps D512/H8/GELU-FFN2048, ALiBi, learned LayerNorm and full-width
Q/K normalization, original Mitchell initialization, and untied V60 embedding
and head. All six blocks are tiled RT at rho1. The first directly reads
temporary self K/V and the previous recurrent output's permanent K/V; the
remaining five have full recurrent attention. States and gradients stay
attached. The first block's short read window does not truncate represented
history. There are18948608 backbone parameters and1049600 predictor parameters,
19998208 total across61 tensors. The original two-layer reference has7407104
parameters including the same predictor.

Initialization is a fresh canonical six-layer SEQ Mitchell draw, exhaustively
converted to RT, then a parameter-preserving replacement of only layer0 with
the existing window2 class. The actual-depth initialization rule changes
depth-dependent scaling and random draw order; **the six-layer backbone is
not claimed identical to the two-layer weights**. The independent predictor
seed and initial predictor tensors are unchanged. Backbone seed1234,
predictor seed1235, and data-order seed1234 match the reference.

NextLat is unchanged: same-position state CE plus weight-one, one-step latent
SmoothL1(beta1); target latent detached, source hidden state and next-input
embedding attached. The original `train_step`, optimizer, data-order,
backbone evaluator and one-step diagnostic functions are reused. No autonomous
MLP latent rollout is evaluated. AdamW uses constant1e-4, betas(.9,.95),
epsilon1e-8, matrix decay.01/vector0, and global clipping at1.

Batch1024, train length12,800000 unique words from the same frozen corpus
`.runtime/rt-a5/20260911T154748Z/data/`. Small development checks every500
updates use4096 words. Checkpoints0/1k/5k/10k are saved, with102400-word full
evaluations per role at trained checkpoints. Length12 development is separate
from length36 development. Full/boundary E/A/M curves use the same length36
rows. Confirmation remains unevaluated.

Runtime is allFP32, no autocast, TF32, compile or CUDA graphs. H10080GB,
GPU UUID `GPU-6217a7ce-a392-4c61-154e-1f91d9cb7d34`; this is a different
instance from the earlier reference, with the same NVIDIA26.06/PyTorch/CUDA
stack. Do not turn this diagnostic into a precision or compiler study.

Nine focused CPU structural tests passed, and independent review found the
update/evaluation/checkpoint loop unchanged. Bounded GPU preflight passed five
discarded actualB1024/T12 updates, finite FP32 model/gradients/Adam for all61
tensors, and a finite causal T36 forward check. Warm updates took about0.15s;
peak allocation was1076593152bytes.
[Preflight W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/rnqyzbrp).

Frozen protocol SHA256:
`bfa66d38f68594200c2232e8a614127f8a4220d463fbea238aebd9ab41656eca`.
Source58 SHA256:
`04d3854620854130eb9a67093fd0538b5762ac870b9464291fde8a6899fbc78b`.
The original55 sources remain unchanged. New training sources are
`scripts/rt_a5_l1r_depth.py`, `scripts/rt_a5_l1r_depth_train.py`, and
`configs/rt_a5_l1r_depth/base.json`. Reporting/runtime helpers are outside that
training manifest. Preserve closed prior lineages.

Reference: `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first/`;
use its1k/5k/10k results, not its80k result as the primary comparison.
Report destination: `docs/reports/rt-a5/l1r-six-layer-nextlat10k/`.
This is a one-seed diagnostic on reused development data. Failure at10k would
show loss of rapid learning under this recipe, not that deeper RT cannot
learn state tracking. Success only shows that this short depth test retained
the observed capability; it is not evidence about arbitrary lengths or depth.

Retain checkpoints and final artifacts under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k/`.
Host `retain.py checkpoints` uploads published checkpoints with checksum and
generation verification. After final saved-state checks, reporting, figure
review, and stable handoff, run `retain.py archive` and then
`verify_evidence.py <lineage>`, with logs outside the lineage. Do not mutate
archived members afterward. Archive/readback receipts are companions created
after the stable report and notes.
