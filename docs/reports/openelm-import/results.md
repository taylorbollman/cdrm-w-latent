# Native OpenELM-1.1B checkpoint import: first PR

Validated 2026-09-21 on the H100 80GB in the project Docker container. This
milestone establishes faithful ordinary-model loading before RT/FBT/NextLat
modifications. No research training run was started.

The final diagnostic is [W&B run z4jzbq02](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/z4jzbq02).
The complete local evidence is
[report.json](../../../.runtime/openelm-import/validation-final/report.json).
[Usage](../../openelm-import-usage.md), [handoff](../../fbt-rt-nextlat-handoff.md)
and [research plan](../../fbt-rt-nextlat-research-plan-v2.md) describe continuation.

## Imported model and authority

- Apple's individual **OpenELM-1.1B 300k** model-only checkpoint, iteration
  index 299999, not the HF final/averaged artifact.
- **1,080,153,600 parameters; 226 parameter tensors; 28 layers; residual width 2048.**
  Native variable attention/FFN widths, GQA 4:1, head dimension 64, Q/K RMSNorm,
  RoPE and SwiGLU are preserved.
- One tied embedding/readout parameter with **32,128 rows**, native padding
  ID 32,000. No vocabulary cropping or parameter reinitialization after import.
- Official authorized Llama-2 SentencePiece tokenizer; native ftfy NFC cleanup,
  BOS/EOS insertion and tokenizer/padding metadata are recorded and verified.
- CoreNet mathematical reference pinned to
  `f9f83e616a34d02c422733a06a3fe5bde63ae575`; its small licensed source snapshot
  is executed with explicit framework scaffolding. The full-size reference
  derives its own geometry through the original CoreNet config constructor.

Checkpoint size: 4,320,718,260 bytes. SHA256:
`0e79fef600d022da33111f86dae4e7f4a4fd1d2e9045364c5ccd4c2283c4d9c6`.
This is our first-download checksum from Apple's published URL, not a checksum
published by Apple. Import verifies all native keys, shapes and finite tensors.

## Actual-checkpoint results

Two fixed text/code fixtures contain 44 and 64 input tokens, physical batch 1.
Both have output/hidden-state/loss comparisons; the 44-token fixture also
compares **every parameter gradient**. FP32 parameters are retained in all
cases. BF16 uses autocast, matching the intended mixed-precision setup.

| Runtime, same backend on both models | Maximum logit difference | Global gradient relative L2 | Maximum gradient difference | Result |
| --- | ---: | ---: | ---: | --- |
| FP32, SDPA math | 0 | 0 | 0 | Pass; bitwise equal on tested quantities |
| BF16 mixed, SDPA math | 0 | 0 | 0 | Pass; bitwise equal on tested quantities |
| BF16 mixed, default fused SDPA | 0 | 2.07e-9 | 1.49e-8 | Pass |

Shifted CE per target token, adapter and reference equal within each row:

| Runtime | Text fixture | Code fixture |
| --- | ---: | ---: |
| FP32 math | 3.174387 | 1.462590 |
| BF16 mixed math | 3.174062 | 1.462096 |
| BF16 mixed default | 3.176566 | 1.464194 |

These are fixed fidelity fixtures, not held-out capability or learning scores.
The default runtime selected `aten::_scaled_dot_product_cudnn_attention` and
an H100 fused flash-style kernel. This verifies **cuDNN SDPA dispatch**, not
FA4/CuTE integration. The ordinary BF16 fused-versus-math output difference
was 0.3546% relative L2 (maximum absolute logit difference 0.25); its CE difference
on the text fixture was 0.002505. These are descriptive kernel-rounding results.
Both native and adapter agree when using the same backend; no historical RT
precision budget or training-quality conclusion is applied here.

Cached decoding was checked on 13 tokens with a 6-token prefill and 4/3-token
adapter chunks. The independent native cache used its supported one-token
decode calls. Adapter-cache versus native-cache maximum logit difference was
1.335e-5, relative L2 1.891e-7. Adapter cached versus full-sequence maximum
difference was 6.866e-5; native cached versus full was 6.294e-5. All passed.
Changing future tokens left earlier logits exactly unchanged. Caches preserved
the native per-layer KV-head counts and normalized unrotated keys.

Peak allocated VRAM was 17.51 GiB, reserved 18.51 GiB, for the **paired reference
and adapter validation workload**, including gradients. It is not a single-model
training-memory or maximum-batch measurement. The final diagnostic took about
23 seconds including local loading/checks after container startup.

## Issues found and resolved

The first full GPU attempt exposed a device-placement issue in the reference
scaffolding: an upstream legacy tensor constructor created CPU parameters even
inside a CUDA device context. Explicit module materialization fixed the harness.
No pretrained tensor values or original mathematical methods were changed.

The first mixed-precision gradient comparison then caught a real porting issue.
The adapter had called `x.float()` separately for RMSNorm's two derivative
branches. Forward values matched, but the branches rounded their gradients to
BF16 separately before addition. Native CoreNet casts once, combines those
contributions in FP32, then casts the sum. Sharing the cast restored native
behavior; the same cast-sharing discipline is used for RoPE.

Before that correction, actual-checkpoint global gradient relative L2 was
0.4575% on the fixture. After correction it is zero with the math backend.
A cancellation-sensitive unit regression prevents this forward-equal,
backward-different implementation change. This was a faithful-port correction,
not an added precision policy or a change to OpenELM's architecture.

Earlier attempts remain recorded locally and in W&B; the final passed run
above is the retained acceptance record. Raw failed diagnostics are not being
represented as additional research experiments.

## Reproducibility and limits

The scoped tests cover strict/corrupt/resumed artifact handling, native
tokenization, nonuniform GQA geometry, source integrity, causal outputs and
gradients, padding, cached chunks and gradients, tied readout ownership, and
serialized AdamW resume with an identical next update. Retention tests cover
explicit artifact selection, checksums and rejection of conflicting GCS objects.
Final test counts are recorded in [test-results.txt](test-results.txt).

The native checkpoint, tokenizer, source overlay and validation evidence are
retained under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T182701Z/`.
The verified object generations/checksums are recorded in
[storage-receipt.json](storage-receipt.json). Retention completed with matching
remote sizes, server MD5 checksums and SHA256 metadata.

This establishes ordinary import and bounded numerical fidelity. Long-context
training, RT/FBT/NextLat gradients, multi-GPU behavior and quality improvements
are subsequent milestones. **The next planned PR is the native-block RT
reference and its alpha0/intermediate/1 contract**, using the preserved ordinary
model as the control. No later learning run is launched by this PR.
