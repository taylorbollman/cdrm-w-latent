# First 32-update adaptation pilot

Completed 2026-09-30. All three arms executed 32 finite updates, both scheduled
FP32 development evaluations, and verified cloud checkpoint32. Training is
stopped; no extension beyond32 is queued. Model/runtime sources are unchanged.

**The execution works, but useful multi-pass refinement is not established.**
NF and NFR improve between the two development observations, while their later
passes remain much worse than their own first pass. Adding RT improves later-pass
CE relative to NF by about0.37nats/target, but slightly worsens first-pass CE.
This calls for a bounded saved-state diagnosis, not a verdict that RT is useful
or useless. See [next steps](next-steps.md).

## Matched observations

B is ordinary OLMo continuation; NF is K4 feedback plus NextLat; NFR adds native
RT at layers0/15 on every pass. All use T1024, two H10080GB GPUs, BF16 mixed
training, and the same ordered524,288 valid inputs/update:16,777,216 new inputs
per arm. B starts original OLMo step60000/tokens252B weights. NF/NFR share the
fusion128 import and exactly matching71 initial parameter tensors, fresh paired
predictor initialization, and fresh Adam. Prior fusion exposure is separately
charged:1,073,565 inputs/1,048,576 CE targets. This is not identical ancestry to B.

The fixed65,536-input development panel is evaluated in common FP32 without
jitter. Lower CE is better; values are nats per target.

| Arm / update | Pass1 | Pass2 | Pass3 | Pass4 |
| --- | ---: | ---: | ---: | ---: |
| B16 | 2.63112 | — | — | — |
| B32 | 2.63179 | — | — | — |
| NF16 | 3.25361 | 7.77496 | 7.79839 | 7.81583 |
| NF32 | 2.96039 | 7.39349 | 7.42586 | 7.43514 |
| NFR16 | 3.23371 | 7.40805 | 7.42014 | 7.43545 |
| NFR32 | 3.04301 | 7.02627 | 7.05529 | 7.06382 |

At32, NFR minus NF is+0.08262 on pass1 and−0.36723/−0.37057/−0.37133 on
passes2–4. Within NFR, later passes still trail pass1 by3.98326–4.02080; NF's
gaps are4.43310–4.47475. Both models improve in absolute CE from16 to32, so the
small gap reductions are not artifacts of a deteriorating first pass over time.
B remains essentially flat (+0.000676 between observations). No update-zero
panel was measured; do not infer pristine-start damage from these two points.

## Gradients and losses

| Arm | Clipped updates | First / final raw norm | Median norm updates1–8 /25–32 |
| --- | ---: | ---: | ---: |
| B | 0/32 | 0.4532 /0.3853 | 0.3990 /0.3948 |
| NF | 32/32 | 424.7315 /6.6018 | 69.3502 /12.9802 |
| NFR | 32/32 | 211.1214 /10.5631 | 45.8993 /9.5321 |

The clipping limit is1.0. Norms fall substantially but fluctuate: NF has an
update17 spike to119.8, while NFR's second-half maximum is20.7. Every accumulated
update remains finite. Clipping coefficients are not Adam effective-LR
multipliers, and scalar loss magnitudes do not identify gradient dominance.

Across the first versus last eight-update windows, NF weighted training CE is
5.4915→5.2448, latent0.7261→0.1273, KL4.0971→1.0417. NFR is
6.0386→5.1464, latent0.6717→0.1144, KL3.1639→1.0333. These windows contain
different training examples. Falling auxiliary losses coexist with poor
later-pass language prediction; this does not by itself demonstrate collapse,
a backward bug, or useful latent representations. The deficit exists without RT.

The [assessment guide](assessment-guide.md) gives exact weighting and detach
semantics. CE is half pass1 plus one-sixth each later pass; latent and KL are
four-pass means, with all three objective weights1.0. B auxiliary means are
null, not measured zeros. All arms remain in warmup: update32 uses7.58e-5 versus
the2e-4 peak reached after100 updates. These are fresh optimizers, not continuous
pretraining optimizer histories. No universal BF16 clearance is claimed.

## Practical cost on two H10080GB GPUs

| Arm | Physical batch/GPU × slots | Compute + materialization inputs/s | Full callback inputs/s | Reserved highwater/GPU | Minimum sampled free/GPU | Executor elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 32 ×8 | 71,006 | 68,547 | 42.50GiB | 35.11GiB | 12.62min |
| NF | 12 ×22 | 7,783 | 7,674 | 58.70GiB | 18.76GiB | 51.87min |
| NFR | 12 ×22 | 3,566 | 3,247 | 59.08GiB | 12.77GiB | 113.81min |

Rates count real input tokens once, not pass-tokens. NF/NFR include16 dummy rows
globally per update, excluded from valid inputs and loss denominators. Selected
compute+materialization excludes health/logging/evaluation/checkpoint gaps. Full
callbacks include development insertion; executor elapsed additionally includes
setup, local saves and final retention, but is not entire container-launch wall.
Memory figures come from capture/after-update samples and allocator highwaters;
free memory is not continuously sampled. Do not subtract reservation from card
capacity and call the remainder available memory.

Graph preparation costs32.58/70.75/446.08s for B/NF/NFR. Both development
insertions together cost6.19/23.60/451.76s. RT's small-batch FP32 diagnostic
path is particularly expensive; its cost is kept separate from training.

B has1,176,764,416 active/trainable parameters; the registered model also holds
8,388,608 dormant fusion parameters. NF/NFR have1,267,879,936 trainable
parameters: the same backbone,8,388,608 fusion and82,726,912 predictor parameters.
Their deployed model without the training predictor is1,185,153,024 parameters.
RT adds computation but no parameter tensors. Equal inputs are not equal compute.

## Checkpointing, recovery and validation

Immutable SSD saves are followed by asynchronous upload/readback verification.
Published boundaries are B0/32; NF0/8/17/26/32; and
NFR0/2/7/12/16/21/26/31/32. All16 checkpoints are cloud-verified. The600-second
save trigger is not a maximum rollback interval. Final retention is drained.

Selected local synchronous save regions total95.93/325.35/630.77s for B/NF/NFR.
Background workers overlap full update callbacks for84.70/1183.29/2715.44s
respectively. These are wall-clock interval overlaps, including any evaluation
inside callbacks, not proof of simultaneous GPU execution/cloud transfer or
pure-training speedups. Aggregate blocking-poll waits are
319.87/411.24/453.06s. Final worker durations are319.94/411.55/404.03s; the runtime
does not separately expose exact terminal-only wait. NFR retained both31 and32;
this pilot did not coalesce a cadence save adjacent to its terminal milestone.

A chat-server restart killed the original host scheduler during NF's terminal
retention. NF's training container survived and closed normally. Its original
launcher exit code is unknown, not fabricated as0. Recovery queue02 validated
final authorities, preserved queue01, and launched only untouched NFR. No
training update was repeated; B/NFR launcher exits are0.

The independent JSON summary validates all200 frozen runtime source pins,
declaration/resolved bytes, complete32-update histories, all128 declared ordered
memberships, actual counters, mask-derived denominators and allocations, LR, parameter ownership, paired
NF/NFR origin parameters, development membership and checkpoint authorities.
Lean evaluation preservation covers tensor ownership/version, zero gradients,
cursor/graph metadata, runtime/modes and RNG; it is not a new full-byte model/
Adam equivalence experiment. No model code changed and no new numerical sweep
was run. See [validation](validation.md), [storage](storage-receipt.md), and
[progress](progress.md).

This is one seed and an early-warmup pilot on a development prefix covering
seven of nine source strata, omitting books/Wikipedia. There is no final-test,
generation-quality, steady-state or scientific efficacy claim. Keep the existing
128-update declaration as a ceiling; no automatic continuation is authorized
by this report. The next recommendation is small saved-state localization.

## Evidence

Joint overlaid charts: [p7vk0qz0](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p7vk0qz0).

- B: [nxm2prv9](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/nxm2prv9).
- NF: [uf1ojrgl](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/uf1ojrgl).
- NFR: [5byv5pkq](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/5byv5pkq).
- Joint summary `.runtime/olmo-adaptation-pilot/summary-cohort-01/report.json`,
  SHA256 `ffb6e4eae228e4ae8e57394bc117a90ce2918eab3d3adb7820f64084d3e1178c`.
- B/NF/NFR report SHA256 respectively
  `203b8fd63448cd4da7a90f424d374d37417758a73c85ca9825d26129bb5035bb`,
  `8e6fe4d8aa933650943320c4f277fd92f4d59e530e08ef456d23634e79cbd5a0`,
  `01bceb2a1191adb513bea974d8dcb0a5b52e8b5c3b384dbd0e69d21cfea665f7`.
