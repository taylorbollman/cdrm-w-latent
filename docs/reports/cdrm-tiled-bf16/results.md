# Five-block tiled CDRM BF16: implementation and preliminary validation

The five-block, one-fabric CDRM now runs with a tiled side scan in FP32 or
explicit BF16 mixed precision. The local derivative, temporal-credit and
ownership tests pass. Paired 100-update training is stable, and BF16 midpoint
recovery reproduces the uninterrupted state and non-timing metrics bit for bit.

**Recommendation: retain this as an opt-in implementation for bounded
experiments, with the numerical qualifications below.** This is a useful PR
review point. It is not a claim that every frozen numerical screen passed or
that a long synthetic-task comparison has been completed.

## Architecture and implementation

CDRM is **Cross-Depth Read-Conditioned Recurrent Memory**. Five ordinary blocks
have zero-based indices 0–4. The single side fabric connects the outputs after
blocks 1 and 3 and shares block 1's canonical weights; its bridge feeds block 4.
Each ordinary block executes once. See the
[equations and approved plan](../../cdrm-tiled-bf16-plan.md) and
[usage guide](../../cdrm-tiled-bf16-usage.md).

The current read uses earlier permanent records plus the current temporary
key/value pair. The permanent write depends on that read and the deep-conditioned
candidate and becomes visible only to subsequent tokens. Candidate construction,
query/temporary-KV preprocessing and the bridge remain in ordinary autograd.
The custom scan returns proposed memory and explicit gradients for its tensor
inputs and unique canonical owner parameters. Reverse-time processing combines
direct proposed-memory credit with credit from future readers before
differentiating the owner's Post operation. Batched dense parameter VJPs avoid
per-token accumulation into parameter `.grad`; outer autograd sums the shared
contributions once.

The BF16 policy uses BF16 dense operations and projected storage, with FP32
master parameters, residuals, differences, normalization arithmetic, attention
accumulators, temporal adjoints, final gradients and Adam state. Adapter outputs
are promoted before gate multiplication and residual addition. Ordinary output
logits are BF16 and native CE is computed in FP32. The existing naive CDRM
reference remains strict FP32. Only the two adapters add parameter owners.

Forward saves permanent projected records in linear memory. **Backward still
reconstructs a quadratic attention matrix.** This milestone does not establish
linear training memory or a fused high-throughput implementation.

## Fixed configuration and evidence

Lineage: `20260907T191403Z`, on an H100 80 GB in the project container using
PyTorch `2.13.0a0+8145d630e8.nv26.06`. Model D128/H16, full MHA, MLP512/GELU,
learned pre-normalization and Q/K normalization, ALiBi, no dropout; early/late
sites 1/3; epsilon 0.1, rho 1, lambda 0.01.

The task is native MAD selective copying, vocabulary 16, **sequence length 256**,
96 copied symbols and physical batch 64. Native aligned training labels and
their mask are authoritative; no additional label shift is applied. Training
uses 12,800 examples (seed 925701), development 256 (925702), and fresh
confirmation 128 (925703). Initialization confirmation uses examples 0–63 and
the trained check examples 64–127. No exact input duplicates cross these sets.

All arms share the saved five-block initialization. The trained same-state
comparison uses the FP32 update-50 checkpoint, including its Adam moments. AdamW
uses LR 0.0005, betas (0.9, 0.98), epsilon 1e-8, zero weight decay and clipping
at 1. The inherited 200-epoch cosine schedule steps only at completed epochs;
this 100-update half-epoch run stays at its initial learning rate.

The [contract](validation-contract.md) was frozen before fresh confirmation
(SHA256 `26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20`).
No criterion or model arithmetic was changed to erase a failed screen. Source,
configuration, checkpoint, data and report identities are retained per run.

## Numerical results and disposition

Relative errors below compare concatenated unique parameter gradients.
The isolated side packet uses identical independent preview leaves and an
unnormalized incoming gradient on proposed memory, before lambda scaling.

| Comparison | Initialization B64 | Trained B64 |
|---|---:|---:|
| Tiled FP32 vs naive FP32, actual CE gradient relative L2 | 8.29e-8 | 1.80e-7 |
| Tiled BF16 vs tiled FP32, actual CE gradient relative L2 | 0.6522% | **2.3065%** |
| Tiled BF16 vs tiled FP32, isolated side gradient relative L2 | 0.4642% | 0.4020% |
| BF16 absolute same-state CE difference | 0.0009234 nats | 0.00002074 nats |
| BF16 individual-tensor and maximum-gradient screens | Pass | Pass |
| BF16 unscaled side-state and isolated side-gradient screens | Pass | Pass |
| BF16 Adam guard | Cosine 0.99547; pass | Update relative L2 1.1781%; pass |

Two classes of frozen failure remain visible in the machine reports:

1. **FP32 raw side-probe coordinate failures.** Actual CE gradients, logits and
   states pass the strict elementwise FP32 comparison at both B64 checkpoints.
   The unnormalized isolated probe flags 8 parameter tensors at initialization
   and 9 after training, despite global relative errors of 5.15e-7 and 4.81e-7.
   Independent FP64 follow-up at B2 and physical B64 finds coordinate failures
   in both FP32 implementations. At B64 their global errors versus FP64 are
   3.90e-7 (naive) and 4.16e-7 (tiled); every tensor's maximum error is below
   7.50e-6 times its reference RMS. Both preview-input gradients and proposed
   memory pass every FP64 coordinate check. This supports a rounding-scale
   disposition, with the strict failures retained. The independent oracle
   reconstructs ALiBi in FP64; its bias differs from saved FP32 ALiBi by up to
   9.96e-6 at distant positions, which is disclosed rather than treated as an
   exact fixed-bias comparison.
2. **One trained BF16 aggregate task-gradient failure.** Error 2.3065% exceeds
   the floor-inclusive allowance of about 2.1481%, a 7.37% excess over the
   allowed error norm. Every per-tensor, maximum, unscaled-memory, loss and
   optimizer screen passes. The allowance is an engineering screen, not an
   authors' reported tolerance or a correctness theorem.

The trained follow-up reuses the exact saved tokens and checkpoint. Its active
FP32/BF16 controls reproduce the original logits, loss and retained gradients
bit for bit. Bypassing CDRM with lambda zero still gives **2.2982%** global
gradient error and fails the same aggregate screen. An FP32 final output head
reduces active-model error to **1.2028%**, passing; Adam update error becomes
**0.7535%**. Keeping final normalization FP32 in addition makes no further
difference because it is already FP32.

The native BF16 head gradient exactly matches its same-operand FP64 contraction
rounded once to BF16. Changed upstream loss cotangents, arising from the rounded
forward/logits, dominate that head comparison. Its error-energy share alone is
not evidence of a head-specific defect: it has 52.68% of error energy and 52.07%
of reference-gradient energy. Together these controls support ordinary
mixed-precision sensitivity rather than a CDRM temporal-backward defect.
**FP32-head execution remains a diagnostic, not the shipped precision policy.**
It is a concrete optional refinement for a subsequent explicitly validated
policy if larger experiments warrant it.

Tiny fixed-operand dense diagnostics also match all four BF16 dense weight
contractions exactly after reference rounding. Writer replay operands match;
some Post intermediates differ with local versus batched BF16 execution, so
bitwise replay equivalence is not claimed. Hook-free compiled/eager comparisons
remain small, and fixed-forward backward scaling at 1/32 and 32 is exact.

## Operation, recovery and cost

Both precision arms completed 100 updates on identical saved batches. All
intended gradients and master/optimizer states remained finite FP32; all owner
and adapter parameters had active gradient signals. Both arms clipped 8 times.
The mean BF16-minus-FP32 training CE gap was 0.0001535 nats over all updates and
0.0002042 over the last 50; the largest absolute step gap was 0.002849 nats.

| Final development metric | Tiled FP32 | Tiled BF16 |
|---|---:|---:|
| Native CE | 2.584587 | 2.584482 |
| Answer-token accuracy | 11.4868% | 11.3892% |
| Whole-sequence exact match | 0/256 | 0/256 |

The final CE gap is -0.000105 nats and the frozen 0.02-nat operational guards
pass. Accuracy is low: these short runs establish operational stability, not
learning equivalence or a CDRM advantage over SEQ. Development data was used;
no final research test set was selected or consumed.

BF16 resume from update 50 to 100 matches model, optimizer, RNG, data position
and non-timing metrics bit for bit when the same retained compiler cache and
runtime are reused. This is not a cross-version or cross-hardware guarantee.

The complete-update benchmark uses one prepared GPU batch, 5 warmup updates and
20 measured updates, without diagnostic monitors. Both arms record zero new
compiled graphs during measurement. Timing includes zeroing gradients, forward,
FP32 CE, backward, clipping and Adam; it excludes data loading, compiler startup
and W&B publication. Reserved memory includes the allocator pool from warmup.

| Warmed B64/T256/D128 measurement | Tiled FP32 | Tiled BF16 |
|---|---:|---:|
| Mean complete-update time | 0.5454 s | 0.5968 s |
| Input tokens/second | 30,043 | 27,454 |
| Peak allocated memory | 2.884 GiB | 2.647 GiB |
| Peak reserved memory | 3.066 GiB | 2.846 GiB |

BF16 is **9.43% slower** and uses **8.22% less peak allocated memory** in this
small profile. This establishes a warmed precision comparison only; it does
not measure a tiling speedup versus naive or CDRM cost versus SEQ. Large-width
or long-context performance cannot be inferred from it.

The separate BF16 repeated-batch smoke check uses two fixed examples for 30
updates. Native CE decreases from **3.29517 to 2.35301 nats**, with active owner
and adapter gradients and finite state. This is a bounded optimization check,
not a claim of memorization or task mastery.

## Regression coverage and limits

The existing CDRM-reference/HF compatibility suite passed 64 tests. The CPU
foundation/checkpoint/new-CDRM suite passed 68 with 35 expected CUDA skips.
The entire new tiled test file passed all 23 cases in the verified H100
container, including its 15 CUDA cases. Five operational-harness tests also
passed. These suite counts overlap; they are not summed as unique tests. The
legacy CUDA cases skipped in the CPU integration suite were not rerun as a full
GPU regression suite; new CUDA cases have their separate successful run.

Coverage includes tiny independent FP64 compositions at T1/5/16/17, causal
perturbations, independent-example memory, read-conditioned writes, earlier
independent deep-preview temporal credit, lambda-zero SEQ equivalence, shared
gradient summation, `autograd.grad` without parameter side effects, frozen
input/owner combinations, and first-order gradient scaling.

The first B2 validator attempt failed before model execution due to a source
snapshot path bug; its corrected rerun is retained. The first head-localization
attempt completed its numerical work but failed in compiler-report bookkeeping;
the clean identical-fixture replay is also retained. These harness fixes did
not change model arithmetic, thresholds or confirmation examples.

Scope remains one fabric, rho one, zero dropout, unpadded independent examples,
first derivatives and the tested runtime/shape. General rho, multiple fabrics,
packing, cached generation, activation checkpointing, accumulation, distributed
execution, long-context scaling and a renewed SEQ-versus-CDRM research pilot
are outside this milestone. No BF16 speed advantage is assumed.

## Artifacts

The [W&B milestone summary](https://wandb.ai/taylorbollman/cdrm-tiled-bf16-validation/runs/b5pnva2m)
combines the graphs and retained flags; its
[project](https://wandb.ai/taylorbollman/cdrm-tiled-bf16-validation) also contains
the individual numerical, training, recovery and performance runs. Local
artifacts are under `.runtime/cdrm-tiled-bf16/20260907T191403Z/`.
The compact [machine summary](results.json), [standalone plot](summary.svg),
[PR description](pr-description.md) and [storage verification](storage.json)
accompany this report. Checkpoints, data, raw numerical tensors, independent
oracles, source snapshots and the compiler cache are retained under
`gs://fast-chunks/cdrm-w-latent/cdrm-tiled-bf16/20260907T191403Z/`.

The previous tiled-R3 milestone's 137 manifest-listed artifacts were separately
rehashed without changes. Its results remain distinct from this new model.
