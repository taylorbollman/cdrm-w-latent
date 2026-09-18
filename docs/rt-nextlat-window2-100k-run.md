# RT + NextLat with layer-2 window 2: closed at accepted 80k

The user stopped this continuation at its retained **80,000-update checkpoint after reviewing development curves**. The original prospective plan was 100k; the historical filename and original protocol are preserved. No training remains active in this lineage, and neither this windowed experiment nor its full-attention RT + NextLat reference reached 100k.

Read the [outcome](reports/rt-a5/window2-budget-80k/outcome.md) and [report with plots](reports/rt-a5/window2-budget-80k/report.md). The [W&B comparison](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/o6x9t8fx) is synced. The original [training run](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/rwj184da) retains the explicit interruption status.

The lineage is `.runtime/rt-a5/20260914T201057Z-rt-nextlat-window2-100k/`, with training under `train-window2/`. Its unchanged `protocol.json` authorized an exact 10k→100k resume; `endpoint-revision.json` records the later user-directed stop and binds the original protocol, accepted checkpoint, raw training report/history and termination evidence. Revision SHA256:

```text
ed05fdfcb5353347f1ab028329955a7f962d04f1d474dec0820b5b87152a4acd
```

The trainer received SIGINT after completing update **85,376**. The accepted model is `train-window2/checkpoints/step-080000.pt`, SHA256:

```text
ec72136e3a6fa1f88fca8726231d9f997b45ee05f3c6ee7fc0718d9a8495debf
```

The **5,376 complete uncheckpointed updates after 80k**, any interrupted in-flight step, and 20 later routine evaluation records are excluded from accepted results. Their raw evidence remains intact. The raw report's `completed_updates` includes that tail, while its `order_chain` and `train_seconds` still describe the saved 80k checkpoint; this follows the unchanged trainer's exception handler. `failed`/`KeyboardInterrupt`, exit code 1, and W&B `synced_failed_experiment` record the deliberate stop. The training process and container were confirmed gone. Do not treat saved launch PIDs as current processes or restart this lineage.

Accepted training comprises the original 10k pilot plus 70k continuation updates, with each update 1–80,000 represented exactly once. The accepted order chain is:

```text
a9217af0352a93505e23d88c9262cc2c964bd2eee61fde39710f4ad29380f6e3
```

The model remains original Mitchell initialization, ALiBi, D512/H8/GELU-FFN2048, LayerNorm/full-width learned QK normalization, two tiled RT blocks at rho 1, vocabulary 60, untied head and no dropout. Layer 1 retains full-prefix RT attention. Layer 2 at t reads temporary self K/V from its input and permanent K/V from its immediately preceding full block output. Gradients remain attached through recurrent states; window length 2 restricts direct reads, not the full history encoded in those states.

The original NextLat predictor, weight-one SmoothL1(beta 1) plus state-CE objective, and target-only latent detachment are unchanged. Source latents and next-operation embeddings remain attached. Backbone/predictor parameter counts are 6,357,504/1,049,600, totaling 7,407,104 in 25 learned tensors. Accuracy evaluation uses the backbone without autonomous predictor rollout. Execution remains full FP32/math attention with autocast, TF32, compilation and CUDA graphs disabled.

The A5 corpus and order are unchanged: 800,000 unique training words of length 12, batch 1,024, backbone/data seed 1234 and predictor seed 1235. AdamW remains constant LR 1e-4, betas (.9,.95), epsilon 1e-8, matrix decay .01/vector 0 and global clip 1. Accepted exposure is 81,920,000 word presentations, or 102.4 nominal passes. Reported checkpoint development evaluations each use 102,400 short or length-36 words.

| RT + NextLat arm | Updates | L12 whole word | E(13) | E(14) | E(16) | M(36) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer-2 window 2 | 10k | 93.0303% | 79.6963% | 26.0879% | 0.3457% | 37.1962% |
| Layer-2 window 2 | 80k | 99.9189% | 95.0840% | 36.3848% | 0.4375% | 38.1807% |
| Full attention | 10k | 98.3047% | 96.8652% | 62.2373% | 1.3350% | 39.1194% |
| Full attention | 80k | 99.5449% | 90.9102% | 29.6934% | 0.2236% | 37.8208% |

The windowed model improves over its own 10k endpoint and exceeds full attention at the matched 80k budget on these prefix metrics. The original full-attention 10k endpoint still has stronger E(14). Both 80k endpoints were selected after development inspection, so these one-seed comparisons are qualified development findings, not convergence or population-level claims. Full 10k versus window 80k is an unequal-budget comparison. E(t) requires all states through t correct, A(t) checks only state t, and M(t) averages token correctness through t. E(36) is zero in every listed sample. Confirmation and autonomous latent rollout remain unevaluated.

The exact parent is the original window pilot's `step-010000.pt` under `.runtime/rt-a5/20260914T175404Z-nextlat-mitchell-window/train-window2/`, SHA256 `9385386a9a559b397f144ce7aab7132d016459a857b247c69f4bf83311ecb6dc`. Its model, Adam, RNG and word-order state were resumed. All 52 original training sources remain unchanged, with digest `9d12d61e994bdad57567cb1d30fd36f1ea3296c94f7aa401d574ecbb34d339c2`.

Completed validation:

- Eight focused stopped-reporter tests and the actual saved-artifact preflight passed.
- `stopped-state-validation.json` records 33/33 checks, including all 25 model/Adam tensors and their accepted 80k counters.
- `stopped-evidence-audit.json` independently verifies 1,056 metric rows, 44 checkpoint evaluations, 24 checkpoint files, 1,600 accepted training bins and 54 generated report artifacts.
- The independent source review passed; original-size full-range, matched-boundary and loss figures were inspected. Full and boundary views use the same curve rows.

The new reporter is `scripts/rt_a5_window_stopped_report.py`, SHA256 `955466c573afbb0e1e8ae700233797b9fb460c02ce519ecd5e3cdb3511d352fc`. Its dedicated CPU-only stopped helpers are `validate_stopped_state.py` and `audit_stopped_report.py` in this lineage. The original 100k reporter, finalizer, dependencies and failed-finalization records remain unchanged; their strict completion checks correctly refuse the intentionally interrupted job. Do not relax those guards or relabel the original job as completed at 100k.

Accepted window update-loop time is 4,450.980 seconds including the original pilot, of which 3,896.963 seconds belong to this continuation. The excluded tail consumed 299.335 seconds. Raw continuation update-loop time is 4,196.298 seconds, and raw job elapsed time is 4,309.218 seconds including evaluation/checkpoint/logging/interruption overhead.

All eight new retained checkpoints—20k, 25k, 30k, 40k, 50k, 60k, 70k and 80k—are recorded in `checkpoint-storage.json`. Retention destination:

```text
gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T201057Z-rt-nextlat-window2-100k/
```

Final evidence archival follows stable report/audit/closure notes; the separate archive and readback receipts are authoritative for that final step. Archive logs belong outside the archived lineage. The earlier full-attention 80k lineage is already closed; do not modify its archived files.

The next user-authorized work is a **fresh** sequential pair under `.runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/`: four sequential Transformer layers with ALiBi + NextLat to 80k, then two RT layers with window 2 in the **first** layer and full attention in the second, also to 80k. Neither run resumes this model. At this handoff the new pair's GPU preflight report has passed, but no completed training endpoint is available; execution state and the eventual frozen protocol belong to that new lineage. This closed window-2 report contains no outcomes from the new pair.
