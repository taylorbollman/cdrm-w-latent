# Fuzzy length evaluation: baseline results and four-model comparison

**2026-09-18 update:** after stored-value addition completed, the user cancelled
the other two mechanisms. The replacement four-model observer is cancelled.
Only baseline and completed stored-value addition are now compared; the same
frozen T512/T1024 pools and native metrics remain applicable. Historical
four-arm scheduling notes below no longer describe the active queue.

**The baseline-first probe is complete and retained. Length 1024 provides
substantial accuracy headroom; length 512 remains mostly near ceiling.**
The user requested this immediate baseline check before starting the remaining
15k embedding runs. It used the original completed 15k baseline checkpoint,
made no training updates and evaluated 1,280 held-out examples at each length.
Read the [baseline report and figures](reports/rt-nextlat-fuzzy-a5/d128-b2560-baseline-length-probe/report.md)
or [W&B probe](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/ug72unjr).

| Fuzzy metric | T400, saved endpoint | T512, new development | T1024, new development |
| --- | ---: | ---: | ---: |
| Answer-token accuracy | 99.9112% | 99.4959% | 79.4008% |
| First-value-token accuracy | 99.9256% | 99.4326% | 74.4914% |
| Terminal-query token accuracy | 99.5272% | 97.3277% | 48.9054% |
| Terminal first-value accuracy | 99.6875% | 97.1875% | 41.8750% |
| Answer-motif exactness | 99.8397% | 99.0812% | 70.1713% |
| Whole-sequence exactness | 97.8906% | 82.2656% | 0% |
| Known-history answer accuracy | 99.9283% | 99.5179% | 79.4011% |

The T1024 drop is not merely the increased number of opportunities to lose
whole-sequence exactness. First-value and terminal accuracies also fall sharply,
although the required mapping is present in history for 99.9989% of scored
tokens. Its terminal first-value count is 536/1,280. Whole-sequence exactness is
now at a floor, so use answer, first-value, terminal and distance-conditioned
metrics to compare the embedding variants at this length.

| Most recent matching-key distance | T400 accuracy | T512 accuracy | T1024 accuracy | T1024 scored tokens |
| --- | ---: | ---: | ---: | ---: |
| 1–16 | 100% | 99.9690% | 97.1927% | 6,768 |
| 17–64 | 99.9875% | 99.9246% | 96.8734% | 21,909 |
| 65–128 | 99.9659% | 99.9489% | 96.1348% | 26,441 |
| 129–256 | 99.9359% | 99.7753% | 93.2902% | 43,578 |
| 257–512 | 99.6720% | 98.2363% | 76.5485% | 55,860 |
| 513+ | No scored tokens | No scored tokens | 38.2993% | 33,857 |

The decline even within the 257–512 bin suggests that the new distances above
512 are not the sole difference. Longer context, competing memories and query
position may contribute; this measurement does not separate them. These bins
also contain different examples at each length. At T1024, accuracy after one,
two and three-or-more prior occurrences is 80.3922%, 76.7567% and 76.8303%
respectively. Repetition has not made this pool uniformly easy, but these
conditional populations do not establish a causal effect of repetition.

Use T1024 as the informative length-generalization screen alongside the
unchanged T400 training comparison; retain T512 as an intermediate point.
This is still a single-seed development comparison, not evidence about
training at T1024 or a winning embedding route.

Baseline probe runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T080000Z-d128-baseline-length-probe/`.
The `retention/final-receipt.json` verifies
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T080000Z-d128-baseline-length-probe/final-evidence.tar.gz`,
generation `1789718966926260`, SHA256
`0a615e67ed0fc0a30ab2ca1cc5d4deb8fe83ae31044f976006b7e42cde4baa8b`.

**Four-model follow-up:** the user requested value → head → input training
order. The original observer, PID 361659, was cancelled before evaluation.
The replacement follows the [reordered handoff](rt-nextlat-mixed-embedding-reordered.md)
and waits for the corresponding complete, retained 15k checkpoints. Its
baseline point will use this same checkpoint and shared data. The early
input run was stopped at 452 updates and is scheduled to resume last.
The reordered training supervisor **382114** launched at **08:10 UTC on
2026-09-18**; the value-route preflight passed at 59.17 GiB peak allocated
memory and value training is running. Replacement length observer **382778**
is running and waiting for all three complete, retained endpoints. Consult
their live status files before taking any new GPU action.

Authorized on 2026-09-18 after the user asked whether the nearly saturated
Fuzzy evaluation should be harder. Evaluate the **15,000-update checkpoints**
of the baseline and all three embedding variants on new, shared development
examples at **lengths 512 and 1,024**. All models were trained at length 400.
This measures length generalization; it is not training on a harder dataset.

Eighteen focused data/evaluator CPU tests, the supervisor checks, restoration
of the actual baseline 15k checkpoint, and three baseline-wrapper CPU tests
passed. The baseline probe has now also completed GPU inference successfully.
Prepared data and source evidence are already verified at the GCS prefix below
as `prepared-evidence.tar.gz` (SHA256
`b940941ab8ba99a92de6c2bf50059eca9cb095c1daa543903df4cd2948187fed`).
The data manifest SHA256 is
`b27a1811dee0ede42777186be8aa55c3b060166e6164d8ee933a0ac00dcac5fe`.

The T512 pool has 54,353 scored answer tokens; T1024 has 188,415, including
33,857 at retrieval distances greater than 512. Counts and repeated evidence
differ across lengths; all four arms share exactly the same pool at a given
length. Full native unavailable-history counts are 12 and 2 respectively.

The baseline's full 15k result was 99.9112% Fuzzy answer accuracy and 97.8906%
whole-sequence exactness, alongside 92.8535% A5 length 36 whole-word accuracy.
Preserve those original length 400 metrics as the in-distribution reference.
New long examples are shared across the four models at each length; examples
at different lengths are independently generated.

## Fixed evaluation protocol

- Four arms: baseline, input addition at fixed lambda 0.01, permanent-value
  addition at fixed lambda 0.01, and the existing last-head embedding route.
- Use exactly each arm's retained 15k checkpoint, selected prospectively.
  No best-checkpoint selection, training update, optimizer or parameter change.
- Generate 1,280 native MAD development examples per length with independent
  seeds 2026091851 (T512) and 2026091852 (T1024). Generate no new training or
  confirmation split. Keep vocabulary 16, multi-query, no noise, key/value
  motif maxima 3 and the native held-out three-token-key convention.
- Preserve native input padding and answer masks, the existing offset 60 into
  the shared model vocabulary, and local 16-class readout. No extra positional
  embedding, attention mask or output shift is added.
- Full FP32 eager inference, existing ALiBi, model maximum length 1024, and
  evaluation microbatch 64. Reuse the original Fuzzy evaluator and metric
  definitions, including its teacher-conditioned NextLat diagnostic. No
  autonomous latent rollout or final confirmation is evaluated.
- Report answer-token, first-value, terminal-query, motif and whole-sequence
  accuracies, known-history/oracle coverage, distance-bin and repeat-count
  metrics with their actual denominators. Report padding/repetition metadata
  so a longer sequence is not assumed to be a strictly harder example.

Show all four models together at lengths 400/512/1024 in Markdown tables,
PNG/PDF plots and online W&B. Length 400 numbers come from each original
checkpoint-bound full evaluation. The longer examples do not share the
length 400 sampling seed, and their number of scored answers can differ.
All results are development evidence from one model seed. No vocabulary 32
training experiment is authorized by this addition.

## Four-model execution after the reordered training runs

The original training queue is closed; the replacement runs value, then head,
then resumes input from its exact 452-update state. A separate replacement
observer waits for all three variants to finish 15k, be summarized and be
retained before it uses the GPU. It verifies local endpoint hashes against
the retained archive members for all four arms, then runs the eight
model/length evaluations. A partial or failed training queue blocks this
four-model evaluation rather than substituting checkpoints silently.

Shared data and preparation runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T074000Z-d128-embedding-length-eval/data/`.
Replacement observer runtime:
`.runtime/rt-nextlat-fuzzy-a5/20260918T075200Z-d128-embedding-length-eval-reordered/`.
Its `execution-config.json`, `ready.json` and `status.json` are the authoritative
resolved inputs, readiness and live status; consult the reordered handoff for
launch details. The original runtime's cancelled status remains historical.

Implementation is separate: `scripts/rt_nextlat_fuzzy_length_prepare.py`,
`scripts/rt_nextlat_fuzzy_length_eval.py`, and
`scripts/rt_nextlat_fuzzy_length_reordered_queue.py`. The separate baseline
probe used `scripts/rt_nextlat_fuzzy_length_baseline.py` and reused the same
restoration, metric and plotting helpers. Bounded CPU tests cover data
identity/native semantics and checkpoint/report handling. Existing frozen
model, trainer and kernel files are unchanged.

Report destination:
`docs/reports/rt-nextlat-fuzzy-a5/d128-b2560-embedding-length-generalization/`.
W&B: [rt-nextlat-fuzzy-a5](https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5).
Prepared data/source evidence remains under
`gs://fast-chunks/cdrm-w-latent/rt-nextlat-fuzzy-a5/20260918T074000Z-d128-embedding-length-eval/`.
The eventual four-model report is retained under the replacement observer's
`20260918T075200Z-d128-embedding-length-eval-reordered/` prefix. Model checkpoints
themselves remain in the separately verified training archives.
Project-directory copies persist too.

Creating `STOP` in this evaluation runtime cancels it while waiting and has
no effect on the training queue. If inference has already started, it is a
bounded eight-evaluation job and the supervisor finishes report/retention.

For the earlier A5-only results, see the
[historical embedding comparison](rt-a5-embedding-history-summary.md).
