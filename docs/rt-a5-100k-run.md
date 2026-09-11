# A5 RT: bounded continuation to 100,000 updates

**Completed 2026-09-11. Training stopped at exactly 100,000 total updates;
do not resume without the user's next decision.** The continuation took
84.83 minutes including evaluation and checkpointing. The endpoint and all
nine other new checkpoints are verified in GCS. The full report, plotted
curves and provenance are in [the results](reports/rt-a5/budget-100k/report.md)
and [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/wi73cfgf).

The extra budget improved training-length reliability but did not resolve
length generalization. On the same 102,400 frozen length-36 development words,
cumulative exactness E(t), requiring every state through t to be correct, is:

| Prefix length t | 10k updates | 100k updates |
| ---: | ---: | ---: |
| 12 | 99.1709% | 99.9199% |
| 13 | 79.8730% | 48.7432% |
| 14 | 21.5156% | 7.0781% |
| 15 | 2.8105% | 0.6270% |
| 16 | 0.2441% | 0.0420% |
| 36 | 0% observed | 0% observed |

At 100k, mean length-36 token accuracy is 35.9744%, while isolated final-state
accuracy is 1.6416% (1/60 guessing is 1.6667%). Correct early states account
for the much higher mean. Long-word cross-entropy rose from 4.6701 to
15.1718, showing much larger loss on out-of-distribution predictions despite
strong short-word performance. On the separate short development sample,
token accuracy is 99.9826% and whole-word exactness is 99.9111%.

Intermediate checkpoints fluctuate substantially: E(13) was 97.4492% at
90k before falling at 100k. This is a fixed-endpoint result, not evidence
that every additional update worsens extrapolation. The 50% exactness
horizon is 12 at 100k versus 13 at 10k; the 95% horizon remains 12. No
evaluated checkpoint establishes successful length-36 generalization.

Verification: the focused reporter suite passed 13 tests; the complete
history joins updates 1–100,000 exactly once; the source/data/model/optimizer
resume contracts and original parent hash match. CPU inspection found all
21 parameter tensors and all 21 Adam states finite FP32, with every Adam
step equal to 100,000. See `saved-state-check.json` and
`final-evidence-audit.json` in the lineage for the saved-state and independent
evidence checks. This was not a new gradient or mixed-precision study.

The curves measure prefixes within the length-36 forward pass, not separate
new samples or inference runs at every length. The earlier 10k shape checks
and their limited accuracy qualification remain documented in
[the length follow-up](rt-a5-length-followup-usage.md); they are not a new
100k logit-equivalence check. This is one development seed at 25% of the
400k reference update budget, using our RT recipe rather than an exact
paper architecture reproduction. The independent confirmation set is still
unevaluated. No additional ordinary-Transformer training was performed.

The launch and recovery record follows.

Started 2026-09-11 after the user reduced the requested 400k endpoint to
100k. This is **100,000 total updates**, including the original 10,000,
and must stop there for review. No ordinary-Transformer training is part of
this milestone. The independent confirmation set remains unevaluated.

Completed lineage: `.runtime/rt-a5/20260911T171239Z-rt100k/`.
Training output: `train-rt/`. Online progress:
[RT 100k](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/wk3271u4).

The two-layer D512/H8 RT has 6,357,504 parameters, both layers recurrent,
GELU FFN 2048, LayerNorm/full-width QK normalization and ALiBi. Execution
remains full FP32, autocast/TF32 off, no compile or CUDA graphs. No model,
optimizer or task code was changed. The model is inspired by the paper's
A5 experiment and retains our RT components; it is not the paper's GPT or
NextLat latent RNN architecture.

Resume source:
`.runtime/rt-a5/20260911T154748Z/train-rt/checkpoints/step-010000.pt`, SHA256
`4acd0a4e95208a1c4628e647ede4bce7bc0680b793eaed978c0bad41e2525044`.
The trainer strictly checks its full source/data/model/runtime/optimizer
contract and restores optimizer, RNG and data order. Shared execution-source
SHA remains `6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea`.

The original frozen million-word short corpus is split into 800k training
and 200k development words. Training length is 12, batch size 1,024, seed
and data-order seed 1234. AdamW keeps LR 1e-4, betas (0.9, 0.95), epsilon 1e-8,
matrix weight decay 0.01 and gradient clipping at 1. The 100k endpoint represents
102.4M word presentations or 128 nominal training passes: 25% of the
reference 400k recipe. Remaining work at launch was 90k updates, approximately
85–90 minutes at the measured speed.

The trainer evaluates fixed 4,096-word development subsets every 500 updates,
with full 102,400-word evaluations at retained checkpoints 20k, 25k, 30k,
40k, 50k, 60k, 70k, 80k, 90k and 100k. Each full evaluation covers short
development and length-36 OOD development, including E(t), A(t), mean token
accuracy and CE.
Keep the fixed 100k endpoint primary. Intermediate results are diagnostics,
especially because E(13) and E(14) worsened from 5k to 10k while E(12) improved.

Operational files in the lineage:

- `protocol.json`: frozen run settings and exact launcher command.
- `run-training.sh`, `launcher-pid.txt`, `training.log`: detached launcher
  and live records. The launcher always executes model work inside the verified
  GPU container. It writes `training-exit-code.txt` when finished.
- `train-rt/history.jsonl`: every update after 10k, including order-chain hash.
  `train-rt/report.json` is atomic and refreshed at retained checkpoints;
  use the history/log for current progress between them.
- `retain_checkpoints.py`, `retention-pid.txt`, `retention.log`: host-only
  watcher that uploads complete atomic checkpoints to GCS and verifies hashes.
- `checkpoint-storage.json`: verified uploaded checkpoint generations,
  hashes and sizes. `retention-complete.json` appears after training ends.
- `finalize.py`, `finalizer-pid.txt`: waits for a successful training exit,
  then invokes the tested CPU-only budget reporter. `report-exit-code.txt`
  and `final-report.log` record its result. It never resumes training.
- `status.py`: host-safe progress summary from log/history files.

The 13 focused budget-reporter tests passed before the finalizer was started.
The checkpoint watcher reloads the final atomic training report after seeing
the exit marker, so it cannot declare completion using an earlier checkpoint
list. Final retention must contain all ten new scheduled checkpoints.

GCS prefix:
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T171239Z-rt100k/`.
Checkpoint paths are `checkpoints/step-NNNNNN.pt`. Original data/checkpoints
remain retained in the historical `20260911T154748Z` lineage. Project/home
files persist; do not depend on SSD-only storage.

The briefly launched 400k continuation, lineage `20260911T171049Z-rt400k`,
was interrupted after the user revised the budget. Its report records
`KeyboardInterrupt` at update 10,606; its unsaved tail is excluded from this run.
The 100k run restarted from the original 10k checkpoint, so no overlapping
history or unrecorded partial optimizer step is included. Its `superseded.json`
records the reason. Do not combine that aborted run with the 100k trajectory.

The completed report assembles original updates 1–10,000 and continuation
updates 10,001–100,000 exactly once, and records the endpoint, full evaluations,
source/data identity, checkpoint hashes and learning curves. The evidence
archive's companion checksum receipt is `storage.json` in this lineage and
the GCS prefix. Do not resume beyond 100k without the user's subsequent
decision.
