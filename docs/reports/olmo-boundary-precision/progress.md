# Fixed-boundary precision progress

2026-09-29. Diagnostic complete on `feat/olmo-boundary-precision`, from
`802a186`/PR41. Runtime helper/tests/protocol frozen at `1023d7e` before GPU
execution. No production model change, precision-policy change or training.
Read [results](results.md) and [next steps](next-steps.md).

## Completed execution

Two NF aggregate anchors/four physical backwards and twelve local VJPs completed
in 147.869 seconds. Both full anchors reproduce saved summaries/fingerprints and
geometry exactly, including the 60.8698% backbone gradient discrepancy. All eight
local A/C endpoints, 104 health checks and ten integrity checks pass. GPU report:
`60432b04554d1eeadc069b9b63f562049be38f23b65317ca4ff60775df559ff1`.
W&B [d43pjmvw](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/d43pjmvw)
is synced. Zero optimizer updates. GPU0 alone ran; both H100s were idle with
0 MiB allocated and 0% utilization at postflight. No next GPU run is queued.

Fusion common-input parameter-VJP differences are 0.387/0.390%; ordinary stack
7.678/8.188%. Changing only inherited stack inputs under FP32 yields
11.428/29.136%. These local norms are not additive explanations of the full-model
error. Whole-stack comparisons include forward rounding. BF16 is not cleared.

## Verification and persistence

Final focused CPU suite: 31 passed in 3.85 seconds, no warnings (10 new boundary,
12 fusion, 9 attention-local tests). Earlier 10-test run overlaps. Log SHA:
`fe2835620d2720f970c59d1b99819fcc05151ca5534f5f34ecd7c0857382d8b7`.
Helper SHA `33d8e6dc0a36fe681935bf425c0582616a28f099d1aad795a03d0ae17a75717d`;
tests SHA `d3471cdc53444abbdbdb3a06a8f918f9f91077bab4de8910efb4db3166719329`.
Independent boundary audit verified 89 sources and 44 encoded tensors. Independent
cloud readback verified both exact-generation objects and all 92 inventory
members. See [test ledger](test-ledger.md) and [storage receipt](storage-receipt.md).

Runtime root: `.runtime/olmo-boundary-precision/`. GCS namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T061000Z/`.
The completed `boundary-01` evidence is retained. Final closeout/PR records will
be appended below. No checkpoint was created or local artifact deleted.

## Proposed continuation

Use the saved adapted O5c checkpoint for a matched current-runtime FP32/BF16 NF
comparison, no new training. Root verified its complete local file hash and
report; actual tensor-schema import remains untested. Read next-steps for the
required new explicit import, preserved saved buffers, old-source guard and
adapted-backbone/K2-to-K4 limitations. No new GPU test has been launched.
