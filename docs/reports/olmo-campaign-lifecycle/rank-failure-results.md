# Abrupt rank-exit recovery result

2026-09-29. The bounded recovery acceptance passed: after rank 1 abruptly exited before update 2's backward, torchrun removed the failed process group, and fresh processes resumed checkpoint 1 with **bitwise-identical updates 2 and 3** and the same final state as the uninterrupted reference.

This used the [frozen protocol](rank-failure-protocol.md): a tiny two-layer, width-32 NFR model with the real OLMo vocabulary, K4, T16, physical B2 per rank and two H100 GPUs. The three logical updates consumed 64/128/192 valid tokens with M1/M2/M3 accumulation. The complete reference covered 384 input tokens, 360 CE/latent targets and 336 KL triples in 24 packed-row presentations. This is an operational check; it adds no pretrained-scale, task-quality or BF16 numerical clearance.

| Stage | External exit | Wall time | Result |
|---|---:|---:|---|
| `rank-reference-01` | 0 | 35.14 s | Complete uninterrupted reference |
| `rank-exit-01` | 1 | 54.36 s | Intended abrupt rank loss; last committed checkpoint remained update 1 |
| `rank-resume-01` | 0 | 26.52 s | Fresh processes reproduced reference updates 2/3 exactly |

The failure marker records rank 1, backward invocation 2, last committed update 1 and deliberate exit code **73**. The launcher then records torchrun sending SIGTERM to rank 0 and, after 30 seconds without termination, forcing SIGKILL. Torchrun returned nonzero before the external 180-second timeout. This was actual forced process teardown, not graceful shutdown or reuse of a damaged process group.

The failed stage's update 1 inputs, raw gradients, losses and complete boundary matched the reference. Its final published pointer still identifies checkpoint 1, with verified GCS retention and manifest SHA256 `0b6584db3719f0666afe7b1f0daa37d15a8ae9c73ebca9f41552a324ae24f345`. Only checkpoint directories 0 and 1 exist for that stage; no uncertain update 2 checkpoint was written or published.

Fresh resume loaded that **local checkpoint, already verified as retained in GCS**. It reproduced both ranks' inputs, raw gradients, metrics, model/Adam/scheduler/counters, committed cursors and rank-local RNG for updates 2 and 3. The final complete boundaries match the uninterrupted reference. Preparation/capture preserved both rank boundaries in every stage. This fault test did not remove local storage or exercise a fresh cloud download; whole-VM loss and mid-Adam rollback remain outside its scope.

The abrupt stage's saved report intentionally still says `running`, and its W&B record may appear unfinished: forced exit bypassed normal finalization. The external orchestration result, exit marker and launcher log establish its termination. Neither the historical report nor its tracking state was rewritten to disguise this expected failure. [Reference W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/tp61o5vm), [fault W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/7qbg3knr), [resume W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/eknjvowu).

Independent [stdlib audit](../../../.runtime/olmo-campaign-lifecycle/rank-failure-audit-01/report.json) passed **43 checks across 10 pinned evidence files**. It verified launcher/report hashes against the orchestration record, exact update/boundary equality, checkpoint pointer/manifest identity, absence of later checkpoints, actual teardown evidence and the three new source files against all snapshots. All stages share the same 91-source inventory; no broad historical source or model-weight rehash was repeated.

| Report | SHA256 |
|---|---|
| `rank-reference-01/report.json` | `53ceede7e03f5e858a89d7738ac9fd3088c1c5448eac837e350ff4b5f69ee701` |
| `rank-exit-01/report.json` | `70add06b60cd428c30bdd25d1d844c540de7cda7810884c48a4c200b6f68efaa` |
| `rank-resume-01/report.json` | `8bdd1988ee220b073b51784ab1fec2bb9e3f35639f63681b82b6f813f3ae0f23` |

The raw orchestration and reports remain under `.runtime/olmo-campaign-lifecycle/`; the audit is a separate evidence stage and does not modify them.
