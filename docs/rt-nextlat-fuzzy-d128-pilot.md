# Width-128 RT + NextLat Fuzzy Recall pilot

Status: **stopped at the user's request after 6,521 updates**, on 2026-09-16.
The endpoint checkpoint, development evaluation, figures and verified GCS
retention are complete. Training launched at 15:53 UTC; reporting and
retention finished at 18:19 UTC.

[Training W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/wrfaw42j).

The user narrowed the [earlier plan](rt-nextlat-fuzzy-a5-mixed-plan.md): start
with **width 128 and FFN 512** and train Fuzzy Recall alone. After reviewing
the plateau, the user authorized stopping this run, then training the same
architecture fresh on A5 to 10,000 updates. If A5 length-36 whole-word
accuracy becomes positive, proceed to the planned equal-example A5/Fuzzy
mixture without another approval pause. Longer Fuzzy lengths and parameter
matching remain outside this follow-up.

The first run uses sequence length 400, two tiled RT blocks, first block
restricted to temporary self plus the preceding permanent record, second
block full recurrent, original Mitchell initialization, ALiBi, and existing
normalizations. Use 16 attention heads (dimension 8) to match the paper's
small synthetic head count. This choice and two layers remain explicit;
the model is not parameter-matched to the paper's one-layer model.

NextLat training remains enabled, with predictor hidden width 128 and the
original residual predictor, SmoothL1 weight one and detached target role.
Inference uses only the backbone. There is no embedding bypass or learned
latent rollout. The common 76-row symbol interface from the plan is retained;
Fuzzy uses input IDs 60–75 and a local 16-class output distribution.

Run lineage:
`.runtime/rt-nextlat-fuzzy-a5/20260916T154000Z-d128-t400/`.
Its `protocol.json` freezes the prospective settings. Logical batch 128,
10,000 updates, constant LR 1e-4, AdamW (.9,.95), epsilon 1e-8, matrix decay
.01, normalization decay zero, clip one. Full FP32/eager, no TF32/compile/
CUDA graphs. Physical microbatch is selected by actual-shape profiling.
Checkpoints: 0, 1k, 5k, 10k; development every 500 updates.

Data: native MAD Fuzzy Recall, vocabulary 16, multi-query, maximum motifs 3/3,
no noise, 12,800 train and 1,280 development examples. Train/dev seeds
2026091601/2026091602, shuffle 2026091604. Seed 2026091603 is reserved for
later confirmation; confirmation is not generated/evaluated in this pilot.
Keep native dense training supervision and masked development answers.

Online W&B project: `taylorbollman/rt-nextlat-fuzzy-a5`. Retention prefix:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260916T154000Z-d128-t400/`.

Do not resume any earlier A5 embedding-series job as part of this milestone.
Historical code and data are preserved. New implementation files have a
separate task/model/trainer/config family; no broad numerical study is needed.

## Verified preflight

Seventeen focused CPU-container tests passed across the model, native data,
metrics, optimizer accumulation and resume helpers. The GPU T17/B2/D128/H16
naive-versus-tiled combined CE + NextLat comparison passed the inherited
FP32 tolerance (atol 2e-6, rtol 2e-5), including parameter gradients,
causality and isolation between calls.

Actual T400/B128 profiling passed through 35 discarded optimizer updates
(25 warmup plus 10 measured). Mean update time 1.357 seconds; peak allocation
3.11 GiB, reservation 4.30 GiB. Use physical batch 128. The measured estimate
is 3.77 hours for 10k training updates, plus evaluation/checkpoint overhead.
Parameters, gradients and Adam state remained finite FP32. Total parameters:
479,616 = 413,824 backbone + 65,792 predictor.

[GPU preflight W&B](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/nonihjj5).

Data preparation manifest SHA256:
`f7d4b1685d57488db5db22b7e022704cb6bc4d5fcb125c2b0db17baf989a4c5b`.
No duplicate inputs or train/dev overlap. Development has 34,897 native
answer tokens, of which eight have unavailable historical mappings and
remain scored. Query-ignoring answer-prefix shortcut accuracy is 40.6367%.

For GCS operations, the environment contains an obsolete external-machine
ADC path. A command-local `env -u GOOGLE_APPLICATION_CREDENTIALS` restores
the working default credentials; bucket access was verified. No credentials
or persistent environment configuration were changed.

## Stopped endpoint and interpretation

The authoritative endpoint is
`train/checkpoints/step-006521.pt` under the run lineage above; SHA256
`afeb822e2b0fba636b1e5d95b72d1a3a3606cdfb6a9737f2d982a38bc80d9c6a`
(5,812,083 bytes). It contains resumable weights, Adam state, RNG and data
cursor/order. The endpoint audit records finite FP32 parameters, gradients
and optimizer state. Checkpoints at 0, 1,000 and 5,000 updates are also retained.

| Development metric | Update 6,521 |
| --- | ---: |
| Native answer-token accuracy | 99.8854% (34,857 / 34,897) |
| Individual answer-motif exact match | 99.8111% (17,435 / 17,468) |
| All scored answers in a sequence exact | 97.5781% (1,249 / 1,280) |
| First value-token accuracy | 99.9542% (17,460 / 17,468) |
| Terminal-probe token accuracy | 99.5666% (2,527 / 2,538) |
| Terminal first-value-token accuracy | 99.6875% (1,276 / 1,280) |
| Answer-token accuracy with available history | 99.9054% (34,856 / 34,889) |

The answer-token curve is flat because it is close to 100%, rather than
because the model failed to learn the task. Answer-token accuracy increased
from 99.7794% at 5,000 updates to 99.8854% at the stopped endpoint; sequence
exact match increased from 95.1563% to 97.5781%, with a dip at 6,000 updates.
The endpoint was selected by the user's stop request, not by best-checkpoint
search. This supports ending the initial T400 pilot, while leaving longer
lengths and independent seeds untested.

The query-ignoring answer-prefix shortcut reaches only 40.6367% on these
same positions. Prefix-only causal lookup has 99.9771% coverage: eight answer
tokens in four examples have no earlier stored mapping. These remain in the
native score, and the model gets one of the eight correct. Coverage is a
diagnostic about information available to this lookup procedure, **not a
universal statistical accuracy ceiling**. First-value accuracy and strong
performance across all observed retrieval distances support real retrieval
beyond the answer-prefix shortcut.

All results use the same 1,280 development examples and one initialization.
Motif/sequence exactness is teacher-forced over scored recall answers, not
autonomous generation. Dense training CE includes non-answer positions,
whereas development CE scores answers; the two curves have different
supervision scopes. Reserved final confirmation remains unevaluated.

The [generated report](reports/rt-nextlat-fuzzy-a5/d128-t400-pilot/report.md),
[learning curves](reports/rt-nextlat-fuzzy-a5/d128-t400-pilot/learning-curves.png)
and [retrieval-distance plot](reports/rt-nextlat-fuzzy-a5/d128-t400-pilot/retrieval-distance.png)
were visually inspected after completion. Axes, legends, denominators and
the stopped endpoint agree with the underlying report; the empty 513+
distance bin is omitted. The plotted 257–512 bin contains only actual
distances possible in T400 examples.

Final archive:
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260916T154000Z-d128-t400/final-evidence.tar.gz`.
Its receipt, `retention/final-receipt.json`, reports status `verified`,
generation `1789582792710078`, 62,969,399 bytes, SHA256
`a1afe44ffd5d918ae0e3cb4348f2ae459fed9741870df778ad62cc09406d6030`.
Remote size and MD5 match the local archive, with SHA256 recorded in remote
metadata. The member manifest explicitly includes the endpoint checkpoint
with its matching SHA256. This post-run interpretation was written after
that archive; the archive preserves the launch-era handoff in `evidence-source/`.

The generated report's last sentence and the archived supervisor's scope
still contain the launch-time instruction to pause before A5/mixed runs.
The user's later authorization recorded above supersedes that instruction;
the archived evidence itself has not been rewritten.

## Execution and recovery record

Host supervisor PID 32603 ran `execute.py` inside the run lineage, starting
every tensor/GPU command through the project container launcher. It completed
training, report generation and final GCS retention. `execution-status.json`
records `training_report_retention_complete`; `train/report.json` records
`status: stopped`, requested endpoint 10,000 and actual endpoint 6,521.
Do not restart this stopped training job as part of the A5 follow-up.

The first training process launched at 15:53 UTC on 2026-09-16. Expected
duration approximately four hours. Training log: `train.log`; per-update
records: `train/history.jsonl`; checkpoints: `train/checkpoints/`.
Physical and logical batch both128. One fresh seed (1234; predictor1235;
Fuzzy rows1236). Data order seed2026091604.

`resume-proof.json` records an exact fresh-GPU-process continuation from
update1 to3 at T400/B4/microbatch2, matching the uninterrupted run's final
weights, Adam state, RNG, source/configuration contract and data cursor/order.
Those integration updates are discarded; the main pilot starts fresh.

For an authorized graceful stop, create the run-root `STOP` file. The trainer
checks it before the next update, saves its actual endpoint, evaluates and
then lets the supervisor finish reporting/retention. This run does not use
the old A5 10k-zero-exactness termination rule.

The dataset was uploaded and content-verified before launch. Its receipt is
`data/retention/native-data-receipt.json`, with archive under the declared
GCS prefix's `data/` subdirectory. Final report target:
`docs/reports/rt-nextlat-fuzzy-a5/d128-t400-pilot/`.

Figure review and endpoint interpretation are complete above. The following
A5 and conditional mixed runs must use new output lineages and preserve this
pilot's frozen source/data/checkpoint evidence. The same architecture means
D128/H16/FFN512, two RT layers with the first restricted to window two,
ALiBi, Mitchell, the common 76-row symbol interface and NextLat training;
it does not mean warm-starting from Fuzzy-trained weights.
