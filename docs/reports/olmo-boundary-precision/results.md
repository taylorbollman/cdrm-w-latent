# Fixed-boundary precision results

2026-09-29. This diagnostic finds **both inherited-input sensitivity and local
precision differences inside the ordinary backbone stack**. At identical inputs
and incoming gradients, fusion's parameter-VJP difference is about 0.39%; the
following stack's is 7.68–8.19%. Changing only the stack's input from the FP32
trajectory to the BF16 trajectory, while keeping stack arithmetic FP32, produces
11.43–29.14% parameter-VJP differences. These measurements localize sensitivity;
they neither establish a faulty backward kernel nor clear the original full-model
BF16 discrepancy.

The frozen [protocol](protocol.md) completed in **147.87 seconds**, with two
aggregate anchors/four physical model backwards and twelve local vector-Jacobian
products (VJPs). No training update occurred. The
[W&B run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/d43pjmvw)
is synced. Runtime is diagnostic wall time, not a throughput measurement.

## Anchors and comparison meaning

The model is the original **NF** fixture: NextLat branches present with zero
auxiliary cotangents, four feedback passes, and **no temporal RT**. It uses the
pretrained OLMo-1B step60000 source, initial full-strength fusion, beta 1 and jitter
0.02. Both B2/T16 records retain their original masks, weights and denominators.

The FP32 and production BF16 anchors exactly reproduce the saved NF forward
fingerprints, loss/gradient summaries and within-pair gradient geometry. CE
objectives remain 7.743894934654236 and 7.739312648773193; the shared-backbone
gradient difference remains **60.8698%**. Complete historical gradient vectors
were not retained, so this is exact summary reproduction, not comparison against
old vector bytes.

For record 0, transitions entering passes 1 and 3, we inspect fusion and the
following full ordinary backbone stack. Each site has three replays:

- **A:** FP32 arithmetic at the captured FP32-origin inputs.
- **B:** FP32 arithmetic at the captured BF16-origin inputs.
- **C:** production BF16 arithmetic at those same BF16-origin inputs.

All three use the same full incoming gradient captured from the BF16 anchor.
Thus B versus A measures response to inherited input changes, while C versus B
measures precision/backend differences introduced inside the replayed module.
Actual boundary inputs and incoming gradients are FP32 in all cases;
“BF16-origin” describes their computation history.

## Local measurements

All table entries are relative L2 percentages. Output comparisons use complete
outputs; the stack's padded outputs are already zero. Fusion's valid-token-only
output results are similar and remain in the retained report.

| Site | Inherited input difference | B/A output | B/A parameter VJP | C/B output | C/B parameter VJP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pass 1 fusion | 0.835% previous hidden; token input exact | 0.833% | 1.045% | 0.444% | 0.387% |
| Pass 1 stack | 0.687% | 2.609% | 11.428% | 3.550% | 7.678% |
| Pass 3 fusion | 10.596% previous hidden; token input exact | 10.534% | 6.152% | 0.455% | 0.390% |
| Pass 3 stack | 7.552% | 11.373% | 29.136% | 3.113% | 8.188% |

Stack **input VJPs** also show both effects. Inherited-input changes produce
26.84% and 35.31% differences at passes 1 and 3; common-input BF16 versus FP32
produces 13.25% and 12.33%. Fusion's common-input VJP differences are smaller:
0.46–0.48% for previous hidden states and 0.61–0.62% for token inputs.

This supports looking beyond fusion arithmetic alone. In particular, the later
stack responds substantially to the state differences it inherits, while its own
mixed-precision computation adds measurable local differences even at fixed input.
It does **not** show that these error norms sum to the full-model gradient error,
or identify which internal stack operation is responsible. An entire-stack replay
includes internal forward rounding as well as backward computation.

## Controls and limits

All eight A/C endpoints exactly match their captured outputs. The independent
CPU audit verified 89 source/snapshot pairs, decoded all 44 retained tensor
payloads, checked masks/layouts and common cotangents, and checked all 104 local
health conditions. Observers counted the original forwards separately from
checkpoint recomputation, and all ten final integrity checks passed. See the
[test ledger](test-ledger.md) for evidence pins.

Fusion uses two parameter tensors locally. The stack uses 64; its tied
embedding/readout parameter is correctly unused for detached `inputs_embeds`
with no logits. These local parameter VJPs therefore cover different parameter
spaces from the full-model gradient. No full parameter-gradient vectors were
exported. Replay leaves preserve shape, dtype and strides but have independent
storage with reset offsets; aliasing is not reproduced.

This samples two transitions in one record of the initial isolated T16 NF
fixture. It does not clear other passes, RT/NFR, trained states, packed T1024,
combined auxiliary losses, CUDA graphs or distributed training. No precision
policy, architecture, Q/K normalization or numerical acceptance threshold was
changed. The full-model BF16 qualification remains open.
