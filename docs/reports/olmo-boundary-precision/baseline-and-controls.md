# Fixed-boundary baseline and capture controls

2026-09-29. The two PR41 references were independently rehashed before this
milestone. Their final report bytes and **161 source/snapshot pairs** remain
unchanged: 79 for the matrix and 82 for the fusion probe. Both report
`passed_operational_diagnostic`; this is continuity evidence, not numerical
clearance. No model or GPU execution was needed for this check.

| Prior evidence under `.runtime/olmo-recurrence-precision/` | SHA-256 |
| --- | --- |
| `matrix-01/report.json` | `bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412` |
| `fusion-01/report.json` | `4f98763c43d5e94fc6c5f783712151a2f648f74ea4147b1cf2f6424bba6446a6` |

The fusion report pins the matrix; the matrix pins the earlier bridge report
`39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b`.
Full source inventories and prior cloud verification remain in those reports
and the [PR41 storage receipt](../olmo-recurrence-precision/storage-receipt.md).
This baseline check did not repeat cloud downloads or rehash unrelated stages.

The source remains `allenai/OLMo-1B`, `step60000-tokens252B`, revision
`81b71efbce6f4dada57c94860301af4298bcd351`; native weights are 4,707,065,440
bytes with SHA-256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
This note verifies the reports' checkpoint identity, not a fresh full-weight
read. Existing model loading still validates its pinned source.

## Fixed NF baseline

Use the unchanged initial NF model: four ordinary-backbone FBT passes, no
temporal RT, beta 1, jitter 0.02 and native RoPE/normalization. Fusion and
predictor seeds remain `20260922` and `20260921`; jitter seed is `20260928`.
NextLat branches stay present, with zero latent/KL cotangents. CE weights are
`(1/2,1/6,1/6,1/6)`, with global denominator 25. The original two B2/T16
records have valid lengths `(16,5)` and `(6,2)`, 29 valid inputs, 25 latent
pairs and 21 KL triples. They are isolated rows, not packed data.

NF's full FP32 and production BF16 CE objectives were
`7.743894934654236` and `7.739312648773193`. Shared-backbone gradient error was
60.8698%. The previous FP32-fusion candidate reduced it to 56.1640% but worsened
both records' final hidden-state errors; that candidate is **not** the new
production anchor. Reproduce the original FP32 and production BF16 endpoints.
Prior gradient summaries and within-pair geometry are retained; complete old
gradient vectors are not. Do not label summary matching as bytewise comparison
against unavailable old gradient vectors.

## Minimal reusable boundary export

Retain one small JSON/base64 tensor fixture using the existing tensor encoder.
A `.pt` export is unnecessary and would be excluded by the current small-
evidence retention suffix filter. Approximately **6–7 MiB of raw FP32 tensor
data / 8–10 MiB base64**, plus small metadata, covers the four selected sites
without deduplication, depending on whether both origins' cotangents are saved;
actual dtypes and serialized size must be recorded.
No model weights or full parameter-gradient vectors need duplication.

For record 0, transitions entering passes 1 and 3, retain:

| Module | Both-origin inputs | Both-origin captured output | Common incoming gradient |
| --- | --- | --- | --- |
| Fusion | Actual `previous_hidden` **after jitter** and `token_input`, each `[2,15,2048]` | Actual fusion return, `[2,15,2048]` | Detached production-BF16-origin output cotangent, full shape |
| Following backbone stack | Actual input after eligibility selection and concatenation, `[2,16,2048]` | Actual `last_hidden_state`, `[2,16,2048]` | Detached production-BF16-origin output cotangent, full shape |

“Record 0” is one physical B2 batch, not a single example. Preserve both rows
and padding. In the current code, jitter is applied **before fusion**, then
eligibility blending chooses fusion versus original embeddings, then the first
token is prepended before the stack. Capture actual arguments rather than
reconstructing this sequence from report summaries.

Each payload needs dtype, shape, raw-byte digest and layout metadata (stride,
storage offset and any aliasing qualification). Save the exact validity and
position tensors, relevant document/eligibility masks, RT mode with no selected
layers, right-padded-causal option, no-logits/no-cache flags and training mode.
Retain tensor hashes for the original row/noise fixture, source/report pins,
parameter/buffer hashes and the exact fusion `output_scale` value/dtype.
Seeded module reconstruction must reproduce parameter and buffer hashes.

“BF16-origin” describes how a tensor was produced, not a command to cast it.
These module boundaries can be FP32 under mixed precision. Preserve their
actual values and dtype. For each site, A is FP32 at FP32-origin inputs, B is
FP32 at BF16-origin inputs, and C is production BF16 at those same BF16-origin
inputs, all using the **same captured incoming gradient**. A and C must reproduce
the captured local FP32 and BF16 outputs exactly. Export both-origin outputs so this
identity and future replays can be checked without another full-model capture.

Full tensor/mask/cotangent shapes belong to the VJP; selecting valid entries
is a reporting operation. Report unused parameters explicitly: for example,
an ordinary stack receiving detached `inputs_embeds` and returning no logits
does not exercise the tied embedding/readout parameter locally. Its parameter
VJP therefore is not the complete model gradient.

A↔B measures response to inherited input changes with arithmetic fixed;
B↔C measures the module-level precision/backend difference at a common state.
The stack comparison still includes internal forward rounding. Neither is by
itself a backward-kernel defect test, and the relative errors are not additive
contributions to the full-model error. Final runtime controls and execution
bounds belong to the separately frozen protocol.
