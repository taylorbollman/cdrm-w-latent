# Development review before final test evaluation

Recorded at 2026-09-06T20:08:26Z.

All six research runs completed the frozen 2,000-update horizon. No learned final-test predictions have been generated at this review. The final checkpoints remain the primary endpoints; best-development weights are secondary retained artifacts.

| Task / primary condition | SEQ accuracy | R3 accuracy | SEQ CE | R3 CE |
|---|---:|---:|---:|---:|
| mqar / iid | 13.68% | 13.44% | 2.8111 | 2.8161 |
| noisy_recall / low | 7.62% | 8.98% | 3.2162 | 3.1584 |
| noisy_recall / moderate | 10.55% | 8.69% | 3.0665 | 3.0295 |
| state_tracking / iid | 21.00% | 20.41% | 1.7653 | 1.7691 |

## Decision

Proceed with the originally declared final-test evaluation to characterize this fixed one-seed pilot. Do not change the schedule, difficulty, or chosen final endpoint in response to these development results. No alternative learning-rate run or longer continuation is substituted for the initial outcome.

MQAR accuracy remains near the 12.5% uniform-observed-value shortcut; loss is still decreasing late in training. Noisy recall is modest, and both state models remain below the generous last-two-operations shortcut (42.48% on the main development fixture). The six separate fixed-batch calibrations reached 100% accuracy, establishing fitting ability but not generalization. Smooth losses and finite gradients do not specifically implicate an excessive learning rate.

These results justify a later, bounded development-only optimization study before drawing conclusions about achievable architectural capability. A paired longer MQAR trajectory is a useful first control for incomplete optimization; state tracking also needs a regime that clears its shortcut baselines. Preserve this pilot and freeze any follow-up with explicit optimizer/data lineage and its own budget. Once a useful regime is found, use training-seed replication and a fresh confirmatory evaluation set.

No inference about language modeling, CDRM, or latent objectives follows from this small symbolic pilot. A lack of benefit here does not cancel those separate hypotheses.
