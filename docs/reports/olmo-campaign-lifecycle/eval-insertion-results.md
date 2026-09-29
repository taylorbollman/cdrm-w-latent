# Evaluation between live captured training updates

2026-09-29. **The two-H100 insertion check passes.** Running an eager held-out
evaluation after completed update 1 leaves the subsequent captured training
updates 2 and 3 bitwise identical to the shared-source, no-evaluation reference.
The independent audit compares complete nested update records; it does not rely
only on the runner's pass flag. Update 1 and the final boundaries also match.

| Stage | Result | Reported stage time | Host launcher time |
| --- | --- | ---: | ---: |
| [No-evaluation reference](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/onbggdvi) | Passed, three updates | 20.322 s | 30.769 s |
| [Evaluation after update 1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/csmpug86) | Passed, exact training continuation | 20.362 s | 30.113 s |

Both torchrun stages exited normally with status 0, well within the 180-second stage
bound, and both W&B runs synced. These short instrumented times include different
host/setup/I/O scopes; their difference is not a throughput comparison.

The model is the existing **tiny FP32 NFR acceptance fixture**: width 32, two
layers, four heads, native vocabulary 50,280, K4 feedback, RT at both layers and
NextLat training. Two ranks use T16/B2 and M1/M2/M3 accumulation, totaling 384
training input tokens, 360 CE targets, 360 latent pairs and 336 KL triples over
three updates. Each rank retains two actual captured backward graphs and executes
three local plus three synchronized replays. This is not pretrained execution.

Evaluation uses the unwrapped model in eager no-grad evaluation mode, with
feedback jitter explicitly zero. It measures **final-pass CE only**; the predictor
and auxiliary losses are unused. One dev-document prefix per rank supplies 16
Reddit tokens and 9 C4 tokens, right-padded to T16: 25 inputs and 23 CE targets.
The two documents are distinct from the training split. There are no intra-row
document boundaries or fabricated EOS tokens in this evaluation fixture.

The globally reduced CE is 258.113983 summed nats / 23 targets = **11.222347
nats/target**. The slower local evaluation took 0.12880 seconds, excluding the
outer before/after boundary hashes and global reduction. This scalar verifies
the evaluation path and denominator; it is neither a selected production
quality metric nor a meaningful model-quality result for this tiny fixture.

All eight preservation checks pass on both ranks: complete model/Adam/scheduler/
counter/cursor/RNG boundary, module modes, trainability, persistent gradient
values, graph-owned input/loss buffers, storage pointers, graph/DDP/stream owners,
and immutable evaluation input. Full next-update comparisons also cover input
digests, raw gradients and losses/metrics. Four checkpoint boundaries per stage
were retained with recorded generation-specific download verification; all eight
local manifest/state byte pairs were independently rehashed.

The new read-only audit passed **435 checks**, including **182 report/source
snapshot/live-file pairs** across 91 pinned sources per stage. It added no model
execution or GPU tests. Its report is
`.runtime/olmo-campaign-lifecycle/eval-audit-01/report.json`, SHA256
`44616592e78da01006accf5339f992266b56a3436f762867b3c09d2709165c26`.
The existing focused CPU suite passed 38 tests before launch; see the
[predeclared protocol](eval-insertion-protocol.md).

Authoritative report SHA256 pins:

- `eval-reference-01/report.json`:
  `c5d8ba4e555d1145ae08f92547b280f8d0675856ac9b045870bcb90bd156463e`.
- `eval-insert-01/report.json`:
  `5f2c6b7bf759805b9248d00aa3551fdee8fb5f7025271e28675d7d189cb14e39`.
- `eval-insertion-orchestration.json`:
  `ea00c4e290638ba53b76ab18a80b80e8a068a2cd9b753d42adebb94f8af8864e`.

Stage evidence is retained under the existing
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T075900Z/`
prefixes `loop-eval-reference-01` and `loop-eval-insert-01`; receipts are in
`.runtime/olmo-fusion-startup/retention/`. This audit checked local receipts and
bytes, without repeating cloud downloads.

This closes the bounded live-graph evaluation-insertion functionality question.
It does not qualify pretrained/BF16 evaluation insertion, H200 execution, other
topologies, long evaluation jobs, evaluation failure recovery, or document-boundary
evaluation behavior. The ordinary pretrained host-loop check remains a separate
acceptance, and NFR mixed-precision conclusions remain separate numerical work.
