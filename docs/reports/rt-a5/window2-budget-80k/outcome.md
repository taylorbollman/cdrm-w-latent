# Window-2 RT + NextLat: accepted 80k outcome

The windowed model improved from its 10k checkpoint and exceeded full attention at the matched 80k budget on the reported prefix accuracies. The original full-attention 10k checkpoint still had the strongest E(14) of these four endpoints.

| Model | Updates | L12 whole-word accuracy | E(13) | E(14) | E(16) | M(36) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer-2 window 2 | 10,000 | 93.0303% | 79.6963% | 26.0879% | 0.3457% | 37.1962% |
| Layer-2 window 2 | 80,000 | 99.9189% | 95.0840% | 36.3848% | 0.4375% | 38.1807% |
| Full attention | 10,000 | 98.3047% | 96.8652% | 62.2373% | 1.3350% | 39.1194% |
| Full attention | 80,000 | 99.5449% | 90.9102% | 29.6934% | 0.2236% | 37.8208% |

All four endpoints are RT + NextLat with original Mitchell initialization and ALiBi. In the windowed arm, layer 1 retains full-prefix RT attention; layer 2 reads temporary self K/V and the immediately preceding output's permanent K/V. Recurrent gradients remain attached. The architecture, objective, FP32 execution and optimizer are otherwise unchanged, and each continuation resumes its corresponding original 10k checkpoint.

E(t) requires every state through position t to be correct; M(t) averages token correctness through t. OOD curves use prefixes of the same 102,400 length-36 development words. L12 development is a separate short-word set. Observed E(36) was zero for every endpoint above. At window 80k, isolated A(36) was 1.7363%, while OOD CE was 7.405738; higher short-prefix accuracy does not mean the length-36 task was solved.

**Both 80k endpoints were selected by the user after reviewing development curves.** Their original prospective plans were 100k. These are one-seed development results with retrospective stopping, not confirmation of convergence or a general advantage. The window 80k versus full 10k comparison has unequal training budgets. Final confirmation and autonomous NextLat predictor rollout remain unevaluated.

The requested stop arrived after the window trainer had recorded update 85,376. The accepted model is the retained **80,000-update checkpoint**; 5,376 complete uncheckpointed updates, any interrupted in-flight step, and 20 later routine evaluation records are excluded from comparisons. Raw logs/history and the original 100k protocol are preserved. The `failed`/`KeyboardInterrupt` training status and `synced_failed_experiment` W&B status record this deliberate interruption.

Validation passed: eight focused reporter tests, saved-state audit 33/33, and independent artifact checks covering 1,056 metric rows, 44 checkpoint evaluations, 24 checkpoint files, and 1,600 accepted training bins across the two arms. Full-range and boundary plots share identical curve rows. All eight new retained window checkpoints are recorded in the GCS storage receipt; final evidence archival follows the completed closure notes.

[Detailed report and plots](report.md) · [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/o6x9t8fx) · [Window training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/rwj184da) · [Run handoff](../../../rt-nextlat-window2-100k-run.md)

The next authorized work is a separate fresh pair: four sequential Transformer layers with ALiBi + NextLat to 80k, followed by two RT layers with window 2 in the **first** layer and full attention in the second, also to 80k. Its lineage is `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/`. These new models are not part of this report.
