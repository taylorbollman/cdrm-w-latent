# Recurrence/precision test ledger

2026-09-29. Matrix source freeze: `6ed920a`; conditional fusion source:
`a5e9540`. Exact per-file hashes in each final report are authoritative. The
[protocol](protocol.md), [fusion protocol](fusion-protocol.md) and imported
PR40 helpers remain frozen.

## CPU preparation and independent audit

`cpu-diagnostics-01.log` under `.runtime/olmo-recurrence-precision/` records
**66 passed in 7.14 seconds** in the explicitly CPU-only project container.
The single warning is the existing CPU RMSNorm BF16-input/FP32-weight fused
dispatch warning, not a GPU failure. This suite includes older diagnostic
checks and must not be added to their previous totals as unique tests.

`cpu-fusion-01.log` records **27 passed in 4.41 seconds**: 15 existing
recurrence checks plus 12 new fusion checks. This overlaps the 66-test suite;
the totals must not be summed as distinct tests.

The prior baseline audit reconstructed all four arm fixtures on CPU and
verified 308 old source/snapshot pairs, four unchanged PR40 reports and local
retained evidence for five prior receipts. Details are in
[baseline-and-controls.md](baseline-and-controls.md).

After the matrix finalized and W&B reported `synced`, a separate stdlib-only
CPU audit checked:

- All **79 current source/snapshot pairs**, including unchanged prior helper
  pins and the exact saved bridge report.
- All eight complete cases and **68 per-case health checks**, separately from
  the additional shared-state, previous-anchor and final integrity checks.
- Exact shared backbone, predictor and fusion parameter/buffer hashes across
  arms, including the fusion output-scale buffer. N/NR have dormant frozen
  fusion; NF/NFR have active fusion.
- Each arm's complete input/mask/document-ID fingerprints against the CPU
  reconstruction, positive counts `(CE=25, latent=25, KL=21)`, 29 valid inputs,
  four documents and two physical records. NF/NFR use identical keyed noise;
  N/NR use none.
- First-pass fingerprints for N↔NF and NR↔NFR at both precisions. All four
  paired comparisons are exact. Incoming gradients are deliberately excluded
  from this cross-arm equality requirement.
- Exact reproduction of both saved NFR CE endpoints: metrics, forward
  fingerprints/precision geometry, gradient-group summaries, and BF16-versus-
  FP32 per-parameter statistics and aggregate geometry. Old full gradient
  vectors were not retained, so this does not claim a comparison against
  unavailable old vector bytes.

Audit evidence is frozen at
`.runtime/olmo-recurrence-precision/matrix-audit-01/`, containing the reusable
`audit.py`, its source snapshot, copies of both input reports and an atomic
`report.json`. It reports `passed` and makes no model-execution or numerical-
acceptance claim. The original reports remain unchanged.

## GPU execution and measurement scope

`matrix-01` ran in one process on CUDA device 0 inside the required project
container, with an external 900-second limit. The second GPU was unused; two
virtual input records are not DDP ranks. The stage completed in **119.94 s**,
with **8 aggregate cases / 16 physical model backwards** and zero optimizer
updates. W&B [lt54objk](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/lt54objk)
is synced. Final status is `passed_operational_diagnostic`.

Every arm uses CE only while computing the existing NextLat branches with
zero auxiliary cotangents. Predictor gradients are exactly zero. N/NR use
one pass weighted 1; NF/NFR use four CE weights `(1/2,1/6,1/6,1/6)` with the
same one-pass global denominator. No losses, counts or predictor were removed
to simplify the comparison.

The common backbone has **1,176,764,416 parameters / 65 tensors** and the
predictor has **82,726,912 / 4 tensors**. Fusion has **8,388,608 / 2 tensors**,
resident in every arm but trainable only in NF/NFR. RT adds no parameters.
Within-arm gradient vectors contain 69 active tensors without FBT or 71 with
FBT; zero predictor tensors remain included. Reported shared-backbone geometry
always uses the same 65 tensor names.

FP32 uses math SDPA/eager native RT; BF16 uses Flash SDPA/Triton native RT with
mixed RT attention. Both retain FP32 master parameters and raw gradients.
Deterministic controls were configured before CUDA; TF32 is off, autocast
caching is off, BF16 matmul reduced-precision reduction is enabled, and
math-SDPA reduced-precision reduction is disabled. Runtime and controls match
the pinned PR40 bridge. All arm state/input/RNG integrity checks pass.

Forward geometry selects actual valid positions and records hidden states
plus **total** incoming gradients, including contributions through later FBT
passes. First-pass matching confirms shared forward controls; it does not
make those incoming gradients common across architectures. Full raw-gradient
comparisons are calculated within each freshly evaluated precision pair.

## Completed conditional fusion probe

`fusion-01` ran NF only with three aggregate cases / six physical backwards:
FP32 reference, production BF16 reference and BF16 with autocast disabled only
inside `FBTGateProduct.forward`. It completed operationally in **38.67 s**;
W&B [wih59gy7](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/wih59gy7)
is synced. It used the same single-device/deterministic/no-update controls.
Together the two GPU stages contain **11 aggregate cases / 22 physical
model-backward calls**, including the repeated references.

Independent CPU audit verified **82 source/snapshot pairs** for this stage,
all 31 per-case health checks, the exact matrix NF state/fixture/noise contract,
both prior endpoint summaries and precision geometries, and final integrity.
The candidate's first-pass fingerprints equal production BF16. Exactly six
fusion calls receive and return FP32 tensors of shape `[2,15,2048]`, with
outer autocast enabled, inner autocast disabled and outer state restored.
The instance's original forward callable is restored. All other execution
flags match production BF16; no old helper or core module was modified.

Backbone-gradient error decreases from 60.8698% to 56.1640%, but both records'
final hidden-state errors increase. Fusion's own gradient error also increases.
The audit preserves these measurements and records that forward/backbone
agreement did not improve jointly. This is an operational pass with a mixed
numerical outcome, not a newly accepted correction. The frozen criterion stops
further promotions in this milestone; the candidate is not adopted.

Evidence is in `.runtime/olmo-recurrence-precision/fusion-audit-01/`, with
reusable script/source snapshot, copies of both input reports and an atomic
report. Matrix and fusion audits cover **161 new source/snapshot pairs** in
total; this is separate from the 308 old pairs rechecked by the baseline audit.

## Evidence pins and retention scope

| Evidence | SHA-256 |
| --- | --- |
| `matrix-01/report.json` | `bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` |
| `fusion-01/report.json` | `4f98763c43d5e94fc6c5f783712151a2f648f74ea4147b1cf2f6424bba6446a6` |
| `matrix-audit-01/report.json` | `800555b5b265a5d402be5a81f5997f91720bce2a974b050b8c904cd46710a500` |
| `matrix-audit-01/audit.py` | `e286c66871fba186140710fef55193e1495ef296395032187c43c152badece07` |
| `fusion-audit-01/report.json` | `72c45f2b42d637e42f069ad3b9cff2b45e60020196f66b91e7885ecf3514398b` |
| `fusion-audit-01/audit.py` | `80afc96ac7c09c8507e93002b22853784faee0934e9775d2203f4073cc3fa71a` |
| `baseline-audit-01/report.json` | `917c5b59376f87ca409da3de7044969c27322d3dcda7ddc5592733d9234beb51` |
| Prior bridge reference | `39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b` |
| Native checkpoint weights | `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c` |

All three baseline-audit, matrix and fusion retention receipts report `verified` under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T053000Z/`.
Independent CPU readback downloaded all six exact-generation objects and
verified their full bytes, 168 evidence members and 162 archived source pairs
(161 from GPU stages, one baseline-audit source). The historical prefix does
not imply distributed execution. See [storage receipt](storage-receipt.md)
for exact object pins and final closeout status. No trained checkpoint was
created, and this is not a checkpoint restore.

Numerical errors are descriptive. No new BF16 tolerance, precision correction,
Q/K-normalization change, training trajectory, graph/restart acceptance or
packed T1024 qualification follows from these operational passes.

The bounded NF-only FP32-fusion follow-up is complete. No further GPU
precision changes are queued in this milestone.
