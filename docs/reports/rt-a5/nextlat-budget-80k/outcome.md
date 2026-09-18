# Original full-attention RT + NextLat: accepted 80k outcome

The run is stopped at the user's chosen **80,000-update checkpoint**. It learned
the length-12 training distribution more accurately than at 10k, but its
retained 80k checkpoint generalized less well beyond that length. Intermediate
development results fluctuated; this is not a claim of monotonic decline.

| Metric | Original 10k | Accepted 80k |
| --- | ---: | ---: |
| Length-12 token accuracy | 99.7533% | 99.9404% |
| Length-12 whole-word accuracy | 98.3047% | 99.5449% |
| OOD cumulative exactness through position 13 | 96.8652% | 90.9102% |
| OOD cumulative exactness through position 14 | 62.2373% | 29.6934% |
| OOD cumulative exactness through position 16 | 1.3350% | 0.2236% |
| Length-36 mean token accuracy | 39.1194% | 37.8208% |
| Length-36 whole-word accuracy | 0 observed | 0 observed |

Each checkpoint result uses 102,400 words per development role. OOD prefix
exactness requires every state through that position to be correct, using
prefixes of the same length-36 words. It differs from accuracy of the token
at that position. The full and boundary figures use identical records.

The accepted model remains the original Mitchell + ALiBi RT + NextLat, with
full attention in both recurrent layers. It resumed its own 10k model,
optimizer, RNG and data-order state under the unchanged training contract.
The separate window-2 continuation resumes its own checkpoint and is
authorized to reach 100k. Matching-budget comparisons with this full-attention
lineage are available through 80k; **there is no full-attention RT + NextLat
100k checkpoint**.

The original prospective endpoint was 100k. The user chose 80k after reviewing
development curves, so this endpoint is retrospective development selection.
The trainer was interrupted after update 81,607: 1,607 completed unsaved
updates and later evaluations remain in raw evidence but are excluded from
accepted results. The raw `KeyboardInterrupt`/failed status reflects this
intentional stop. The original protocol and raw reports were preserved, with
the change recorded in `endpoint-revision.json`.

Saved-state inspection passed 31 checks, including all 25 finite model/Adam
states, Adam counters at 80k and the accepted minibatch-order history.
Reporter tests and independent artifact checks passed. The eight continuation
checkpoints are retained and verified under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260914T185736Z-rt-nextlat100k/`.
Final archive and readback receipts are `evidence-storage.json` and
`stopped-evidence-readback.json` in that lineage; they are generated after
this note and bind the complete evidence package.

This is a single seed on reused development data. Independent confirmation
and autonomous NextLat predictor rollout remain unevaluated.

[Report and figures](report.md) ·
[Operational handoff](../../../rt-nextlat-a5-100k-run.md) ·
[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/b77bpb54) ·
[Comparison W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/r0s16kq4)
