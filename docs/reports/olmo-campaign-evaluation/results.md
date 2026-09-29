# Per-pass evaluation preserves live training

The new evaluation-enabled runner passes the bounded two-H100 acceptance tests.
Inserting common-FP32 held-out evaluation between native BF16 training updates
leaves the following update **exactly unchanged**: input/noise, raw gradients,
losses, model weights, Adam, schedule, RNG, counters and packed cursor all match
the retained no-evaluation reference. This establishes integration and recovery
behavior; it does not establish BF16/FP32 training equivalence or a quality win.

The implementation reports every trained pass's CE, latent and KL sums, target
counts and globally normalized means, plus the canonical weighted objective.
Its dev prefix and schedule are pinned in the declaration. Evaluation restores
native runtime flags, modes, zero-gradient storage and graph-owned inputs.
Failed integrity preserves the previous committed checkpoint as recovery
authority. A scheduled restored boundary evaluates once per new segment.
See [protocol](protocol.md), [operator notes](operator-notes.md) and
[next steps](next-steps.md).

## Acceptance evidence

The four final CPU scopes contain **197 distinct passing tests**, including
literal loss/mask oracles for all eight arms, actual tiny model identities,
uneven rank counts, dummy/disabled terms, restoration errors and report audits.
The [test ledger](test-ledger.md) retains exact scopes and logs.

| Two-GPU control | Result | Independent audit |
| --- | --- | ---: |
| Tiny live evaluation insertion after update 2 | Update 3 and final state exact | 1,787 checks |
| Tiny stop, verified GCS download and fresh-process resume | Restored evaluation and update 3 exact | 1,558 checks |
| Tiny evaluation-only resume at update 2 with segment stop 2 | No update, no capture, unchanged state | 1,362 checks |
| Native T1024 NFR insertion after update 2 | Updates 1–3 and final state exact | 1,742 checks |

The terminal segment test does not complete the three-update finite plan: it
deliberately stops at its already-restored segment limit. The first terminal
audit exposed an auditor-only assumption about an omitted empty `updates`
field. The failed audit is preserved; a regression and corrected audit pass.
No model, execution source or GPU result changed to fix that audit.

The native comparison verifies 319 source snapshots: 155 from the retained PR46
reference and 164 from the new runner. Only the declared evaluator additions,
evaluation/configuration identity and output-location metadata differ. Training
sources, recipe, data, startup and runtime match. This offline comparison does
not permit checkpoint resume across the two different execution identities.

## Native configuration and held-out observations

The backbone is the selected original OLMo-1B step-60,000 checkpoint: 16 layers,
width 2,048, 16 heads, SwiGLU intermediate size 8,192, RoPE, tied readout, no Q/K
normalization, and the complete 50,304-way output vocabulary. Startup imports
the accepted fusion-only update-128 weights and creates fresh all-active Adam.
It does not import the full-NFR update-20 state used by the separate optimizer
diagnostic. Total trainable parameters are **1,267,879,936**.

NFR enables NextLat, FBT K4 and native RT at layers 0/15. Training uses T1024,
B1 per rank, three real rows per update, two accumulation slots per rank and a
dummy final slot on rank 1. It retains BF16 mixed execution, FP32 masters/Adam,
fused AdamW, Flash SDPA, native Triton RT, activation checkpointing, CUDA graphs
and training jitter 0.02. Three updates consume 9,216 input tokens, 9,207 CE
targets, 9,207 latent pairs and 9,198 KL triples. These tiny update budgets are
operational acceptance, not the intended production batch or throughput setup.

At update 2, common FP32/no-jitter evaluation reads 3,072 dev input tokens from
two C4 documents. Its 3,069 CE targets include one real document boundary;
same-document eligibility leaves 3,068 latent pairs and 3,064 KL triples.
The entire dev index has only 104 documents, and this prefix is not a
representative quality benchmark.

| Pass, one-based | CE per target | Latent mean | KL mean |
| ---: | ---: | ---: | ---: |
| 1 | 4.058672 | 0.834084 | 5.900382 |
| 2 | 8.711058 | 0.718907 | 1.352321 |
| 3 | 8.111227 | 0.716669 | 1.459364 |
| 4 | 7.999385 | 0.720243 | 1.536846 |

The campaign-weighted means are CE 6.166281, latent 0.747476 and KL 2.562228.
Later passes still have higher CE. This short functionality run does not
demonstrate a refinement benefit; the per-pass evaluator now makes that question
measurable during a properly specified pilot.

## Timing, memory and retention

Native wall time was **1,113.07 seconds (18.55 minutes)**. Graph preparation
took 367.40 seconds per rank. Captured forward/loss/backward regions took
6.958–6.978 seconds per update. The evaluation/restoration scope took about
13 seconds per rank; the whole evaluation boundary, including full acceptance
hashing and global accounting, took 53.51 seconds. The final checkpoint's
verified cloud retention took 291.98 seconds. These regions are reported
separately and are not an optimized end-to-end throughput benchmark.

Post-evaluation recorded CUDA peaks were about 24.41 GiB allocated and
37.16 GiB reserved per rank. This bounded B1 allocation does not establish
capacity at large physical batch sizes or on another topology.

Native [W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/i0jnqdyy)
is synced. Report SHA256:
`fa8134680a6b30aa00905ec034117e8b5984fdc2ed51315e31708fb612967a4b`.
The independent native audit SHA256 is
`cce02a410f892b5e53ca000ca76fb3153579f0c80c590de63a6903d820986011`.
All checkpoint and evidence authorities are recorded in
[storage-receipt.md](storage-receipt.md). Both GPUs are idle; no further run is
queued. The companion [optimizer-history result](../olmo-optimizer-history/results.md)
adds evidence about numerical sensitivity without changing the training recipe.
