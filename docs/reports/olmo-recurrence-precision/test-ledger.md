# Recurrence/precision test ledger

2026-09-29. Matrix source freeze: `6ed920a`. Exact per-file hashes in the
final report are authoritative. The [protocol](protocol.md) and imported
PR40 helpers remain frozen.

## CPU preparation and independent audit

`cpu-diagnostics-01.log` under `.runtime/olmo-recurrence-precision/` records
**66 passed in 7.14 seconds** in the explicitly CPU-only project container.
The single warning is the existing CPU RMSNorm BF16-input/FP32-weight fused
dispatch warning, not a GPU failure. This suite includes older diagnostic
checks and must not be added to their previous totals as unique tests.

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

## Evidence pins and retention scope

| Evidence | SHA-256 |
| --- | --- |
| `matrix-01/report.json` | `bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` |
| `matrix-audit-01/report.json` | `800555b5b265a5d402be5a81f5997f91720bce2a974b050b8c904cd46710a500` |
| `matrix-audit-01/audit.py` | `e286c66871fba186140710fef55193e1495ef296395032187c43c152badece07` |
| `baseline-audit-01/report.json` | `917c5b59376f87ca409da3de7044969c27322d3dcda7ddc5592733d9234beb51` |
| Prior bridge reference | `39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b` |
| Native checkpoint weights | `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c` |

The matrix and baseline-audit retention receipts report `verified` under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T053000Z/`.
The historical prefix does not imply distributed execution. Independent cloud
readback and retention of this new matrix audit are tracked separately;
this ledger does not claim a new checkpoint restore. No trained checkpoint
was created.

Numerical errors are descriptive. No new BF16 tolerance, precision correction,
Q/K-normalization change, training trajectory, graph/restart acceptance or
packed T1024 qualification follows from these operational passes.

The selected [NF-only FP32-fusion follow-up](fusion-protocol.md) is pending.
Its three aggregate cases / six physical backwards are additional planned
work and are not included in the completed matrix's counts above.
