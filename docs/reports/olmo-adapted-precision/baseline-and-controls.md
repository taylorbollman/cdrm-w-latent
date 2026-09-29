# Adapted-state baseline and import controls

2026-09-29. A bounded, independent CPU audit verified the saved O5c authority
and unchanged cold-NF reference. It did not load tensors, generate new fixtures,
run a model, download cloud objects or repeat the 4.8 GB checkpoint hash read.
Tensor compatibility is a separate importer gate before the paired GPU probe.

## Checkpoint authority

Use `.runtime/olmo1b-step60000/o5c-pilot-01/mixed/update-000512.pt`, with
4,807,843,871 bytes and SHA-256
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.
Its completed report remains
`020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.
The report's endpoint record exactly matches the local checkpoint receipt and
the current file size. Root verified the full local checkpoint hash on
2026-09-29 during the preceding milestone; this audit relies on that verification
and the immutable report/receipt, rather than claiming another full-byte read.

The previously retained object is
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/mixed/update-000512.pt`,
generation `1790059437165208`. No new cloud verification is claimed here.

O5c completed 512 fusion-only updates and 4,194,304 CE targets from an **already
adapted O5b backbone**, whose source checkpoint SHA was
`99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`.
Its two fusion matrices trained; the recorded 65 native parameter tensors and
fixed fusion scale remained unchanged during O5c. Both backbone and fusion differ
from the cold NF baseline. Better precision agreement, if observed, would not
isolate fusion adaptation as its cause.

Historical training used K2, beta 1, no RT, no NextLat and no jitter. The new
diagnostic imports those weights into the current K4/jitter NF recipe. It is
neither an exact O5c replay nor a training continuation.

## Source and tensor mapping

All **37 historical source hashes** match Git `9a1fd3a`. Of those, 31 still match
the current working sources. Exactly six have the already-recorded changes:

| Source | Relevant change since O5c |
| --- | --- |
| `fbt_training.py` | Explicit campaign CE/auxiliary pass weights and document-policy validation; historical legacy default retained |
| `lm_training.py` | Builtin-only metadata canonicalization and optional fused AdamW; no optimizer state is imported here |
| `nextlat.py` | Independent CE chunk size and explicit document policy; optional defaults preserve historical serialized configuration |
| `olmo.py` | Optional ordinary attention/RoPE/SwiGLU implementations; no learned tensor renaming identified |
| `olmo_fbt.py` | Explicit first-pass RT policy, external jitter, document policy and causal Flash dispatch options |
| `olmo_tiled.py` | Ordinary activation checkpointing and backend dispatch, native RT optimizations, author backend option and right-padding support |

Exact old/current hash pairs are retained in the baseline audit. Do not weaken
`olmo_o5d_common.endpoint_metadata()` or pretend its historical-source validator
accepts today's runtime. The new importer must use a separate, explicit mapping
and reject changes beyond these pinned allowances.

Required tensor controls:

- Use `weights_only=True`, verify the checkpoint's schema/configuration/counters
  against its immutable report, and reject unexplained missing or extra state.
- Map native `backbone.backbone.*` and fusion `backbone.fusion.*` state strictly,
  with exact shapes/dtypes, finite tensors and complete before/after state hashes.
  Preserve native tied embedding/readout ownership.
- Preserve saved `fusion.output_scale` **0.03707655891776085** and every buffer.
  Do not replace it with a scale measured from the adapted embedding matrix.
- The historical checkpoint has no predictor. Initialize only that absent module
  with the current NF seed and verify its state against the cold fixture.
- Historical optimizer ownership is fusion-only. For the numerical diagnostic,
  backbone, fusion and predictor parameters must have the current NF trainability;
  do not inherit historical freezing or the old evaluation importer's `.eval()`.
  Do not restore optimizer, scheduler, data cursor or RNG state.

The native model configuration matches the current NF model exactly: 16 layers,
width 2048, 16 heads, MLP width 8192, vocabulary 50,304, RoPE base 10,000, native
tied readout and unchanged normalization. Runtime flags and loss policy still
require explicit current-fixture matching; equal tensor geometry does not imply
equal historical execution.

## Cold NF and unchanged fixture

The cold matrix report remains
`bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412`.
The boundary report remains
`60432b04554d1eeadc069b9b63f562049be38f23b65317ca4ff60775df559ff1`.
Every shared NF field matches exactly between them: contract, model/configuration,
runtime flags, initial-state hashes, fixture/noise pins, recipe and checkpoint.
The matrix's 79 source pins are a subset of the boundary's 89; all 89 current
sources and saved snapshots were rehashed successfully.

Keep two physical B2/T16 records with valid lengths `(16,5)` and `(6,2)`, four
documents and 29 valid tokens. Global counts remain CE 25, latent 25 and KL 21.
NF has no selected RT layers, K4, beta 1, jitter 0.02 and `isolated-v1` document
policy. CE pass weights remain `(1/2,1/6,1/6,1/6)`. NextLat branches execute with
zero auxiliary cotangents; the predictor remains present. Predictor/fusion/jitter
seeds remain `20260921` / `20260922` / `20260928`. The NF recipe SHA is
`ee5e737cc93f293d1f4bb82b52538ded4a36c56b63e7a36f2d6e4d5dfe7d3cef`.

The cold source is OLMo-1B step60000, revision
`81b71efbce6f4dada57c94860301af4298bcd351`, native weight SHA
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
Its NF backbone FP32/BF16 gradient difference was 60.8698%. Compare the new
precisions **within the same adapted state**; do not require its loss, hidden
states or gradients to equal the cold model. Existing reports retain geometry
and summary fingerprints, not complete historical gradient vectors.

Audit evidence is under
`.runtime/olmo-adapted-precision/baseline-audit-01/`, with an atomic report and
self-source snapshot. This verifies authority and source continuity only. Final
import validation and paired numerical results remain pending; no BF16
qualification or architecture choice follows from this baseline audit.
