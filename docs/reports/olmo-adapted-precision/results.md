# Adapted-state NF precision results

2026-09-29. **Aggregate FP32/BF16 gradients agree much more closely at the saved
adapted checkpoint, but intermediate forward agreement is still uneven.** The
backbone relative L2 difference drops from 60.87% in cold NF to **0.909%**;
fusion drops from 65.21% to **1.405%**. Record 0's first feedback-pass hidden
state nevertheless differs by **12.437%**, versus 4.091% in cold NF. This is
encouraging evidence about operating-state dependence, not numerical clearance.

The frozen [protocol](protocol.md) completed in **63.21 seconds**, with two
aggregate precision cases/four physical model backwards and no updates.
[W&B](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/8ymbic44) is
synced. This wall time includes loading/validation and is not training throughput.

## Gradient comparison

Each row compares production BF16 against FP32 **within the same model state**.
Gradients are raw and unclipped. Predictor gradients are exactly zero, as intended
for the zero-auxiliary-cotangent diagnostic.

| State / group | Relative L2 | FP32 norm | BF16 norm | Absolute difference norm | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold backbone | 60.8698% | 546.6146 | 392.7756 | 332.7231 | 0.797300 |
| Adapted backbone | **0.9085%** | 50.6929 | 50.6782 | **0.4606** | **0.999959** |
| Cold fusion | 65.2122% | 86.9398 | 66.3264 | 56.6953 | 0.758129 |
| Adapted fusion | **1.4051%** | 1.3191 | 1.3180 | **0.01853** | **0.999902** |

The improvement is not caused by a larger reference denominator: FP32 gradient
norms became smaller, while absolute discrepancies fell much further. Whole-model
relative L2 is 0.9089%. Adapted CE objectives are 5.2147265673 FP32 and
5.2138382196 BF16, a difference of −0.0008883476 nats per selected target.
Latent/KL losses execute and are reported, but have zero backward weight here.

## Forward qualification

Valid-token hidden-state relative L2 differences across the four passes are:

| Physical record | Pass 0 | Pass 1 | Pass 2 | Pass 3 |
| --- | ---: | ---: | ---: | ---: |
| 0, cold | 0.853% | 4.091% | 10.854% | 12.663% |
| 0, adapted | 0.770% | **12.437%** | 2.582% | 1.928% |
| 1, adapted | 1.887% | 1.507% | 3.144% | 2.220% |

The large record-0/pass-1 discrepancy has absolute difference norm 25.7915
against reference norm 207.3833, with maximum element difference 15.7165. It is
not hidden by the improved aggregate gradients. Adapted pass-output incoming
gradient differences are 0.385–0.593% across all eight sites; these include later
feedback and are not held fixed between precisions. Smaller aggregate gradients
do not establish that the intermediate state difference is harmless.

## Import controls and interpretation

The explicit weights-only import preserved all 68 historical state tensors:
65 native parameters, two fusion matrices and the saved fusion-scale buffer.
Only the absent predictor remains freshly seeded. Parameter identities,
trainability, tied readout, runtime settings, training modes, masks and keyed
noise remain intact. `adapted_import` identifies the actual O5c weights;
`source_checkpoint` inside construction metadata identifies the original model
construction artifacts, **not** the final loaded weights.

O5c inherited an already-adapted O5b backbone and then trained fusion only.
It trained at K2 without NextLat or jitter; this test uses current K4/jitter NF.
The result therefore cannot isolate fusion learning, prove initialization caused
the cold discrepancy, or justify adopting this checkpoint as the campaign start.
It motivates a controlled feedback-startup test while retaining the forward-state
qualification. The [test ledger](test-ledger.md) records operational evidence.

No precision policy, error threshold or architecture changed. This short
isolated T16 CE-only fixture does not clear RT/NFR, auxiliary gradients, packed
T1024, optimizer updates/restarts, CUDA graphs, DDP or training quality.
