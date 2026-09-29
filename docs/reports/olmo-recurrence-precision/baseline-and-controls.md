# Baseline and controls for recurrence/precision separation

2026-09-29. The saved PR40 references and original CPU fixture were independently
rechecked before the new eight-case experiment. **This is a baseline audit, not
a new GPU precision result.** The execution protocol is recorded separately.

## What the four arms mean here

All four arms retain the existing NextLat predictor and compute its loss
branches. The CE diagnostic multiplies the normalized latent and KL terms by
zero before backward; it does not remove the predictor, disable NextLat,
change stop-gradient rules or alter its configured loss weights. Predictor
gradients should consequently be zero. These are not the predictor-free
`B`, `R`, `F`, `FR` implementations, nor a test of learning with NextLat losses.

| Arm | Backbone computation | CE pass weights | Feedback noise |
| --- | --- | --- | --- |
| `N` | Ordinary, one pass | `(1)` | None |
| `NR` | Native RT at layers 0/15, one pass | `(1)` | None |
| `NF` | Ordinary layers, four FBT passes | `(1/2, 1/6, 1/6, 1/6)` | Original keyed jitter |
| `NFR` | Native RT at layers 0/15 on all four FBT passes | `(1/2, 1/6, 1/6, 1/6)` | Same keyed jitter as NF |

The campaign policy keeps the CE weights summing to one. Auxiliary pass weights
before the diagnostic's zero cotangents are `(1)` or `(1/4,1/4,1/4,1/4)`.
All losses use their global one-pass target counts; K4 does not multiply the
denominator by four. FBT uses `configured-rt-v1` from the first pass, beta 1,
and jitter scale 0.02. Non-FBT arms do not consume feedback noise.

All arms start from the same pretrained backbone, fusion seed `20260922` and
predictor seed `20260921`. Fusion remains resident but frozen and unused in
N/NR; it participates in NF/NFR. RT introduces no new learned parameters.
Compare BF16 versus FP32 **within each arm**, recording both the shared
65-tensor backbone and the full active parameter vector. The predictor's four
tensors remain present with zero cotangents; the two fusion tensors are active
only with FBT. Different pass weights and active fusion mean that cross-arm
raw gradient norms are not direct causal effect sizes.

## Checkpoint and exact data controls

The source is `allenai/OLMo-1B`, native branch `step60000-tokens252B`, immutable
revision `81b71efbce6f4dada57c94860301af4298bcd351`. The conversion branch uses
the approximate label 251B; the exact historical training-token counter was
not independently established. These are the same 16-layer, width-2048 weights
with native RoPE, tied embedding/readout and no Q/K normalization.

`native/model.safetensors` is **4,707,065,440 bytes**, SHA-256
`ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c`.
The new audit checks the existing manifest identity and file size and rehashes
the small configuration/tokenizer artifacts. It does not reread the 4.7 GB
checkpoint; prior model loaders already verified those full bytes. New model
construction must retain its existing full-source validation.

The original isolated fixture is update 0, T16, two physical B2 records with
valid row lengths `(16,5)` and `(6,2)`: **29 valid inputs, 25 CE targets, 25
latent pairs and 21 KL triples**. It uses one document per row and right
padding, not the packed campaign stream. Each valid row ends with EOS 50279;
this fixture also fills masked suffix positions with EOS. Eligibility comes
from validity/document masks, not token-value inference. The recipe's T1024
field remains campaign metadata; actual diagnostic tensors have T16.

CPU reconstruction used the pinned tokenizer and original text seed, verified
every input/mask/document-ID digest for all four arms, and reproduced all six
FP32 `[2,15,2048]` feedback-noise tensors exactly for NF/NFR. Their keys retain
logical update 0 and jitter seed `20260928`; N/NR have `None` noise. The NFR
recipe SHA remains
`29409f66064a2ab2034cf549fe45ecc26f83841d0f9755d87954616e0cdd5da8`.

The two precision endpoints retain PR40's explicit settings: FP32 math SDPA /
eager RT versus BF16 mixed Flash SDPA / Triton RT, FP32 master weights and raw
gradients, autocast cache off, deterministic controls before CUDA and TF32 off.
This endpoint comparison changes precision and permitted kernel implementation;
it is not a pure same-kernel dtype experiment. No optimizer, DDP, CUDA graphs,
architecture change or training result is part of this comparison.

## Saved reference scope and pins

The authoritative NFR reference is the CE pair in PR40 `bridge-01`. FP32 and
production BF16 CE objectives were **7.280283451080322** and
**7.304819941520691**. Their full raw-gradient relative L2 difference was about
95.93% (shared backbone about 95.85%). The new NFR pair must reproduce saved
metrics and forward fingerprints; gradient summaries/comparisons supply
additional continuity checks.

PR40 compared full gradient vectors in memory, then retained their summaries,
per-parameter comparison statistics and aggregate geometry. **It did not save
complete reusable gradient vectors.** Matching the saved summaries is therefore
not a fresh bitwise comparison with all old gradient coordinates. New within-arm
FP32/BF16 comparisons must use freshly collected full gradient references.
The crossed-backend CE equality and eight-site local VJPs retain their original
scope; they do not clear the broader BF16 qualification.

| Unchanged prior report under `.runtime/olmo-precision-localization/` | SHA-256 |
| --- | --- |
| `bridge-01/report.json` | `39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b` |
| `auxiliary-01/report.json` | `c4946da63c6275a4fcd926292b0296337f57d10553233c7184e94e38b617d5fc` |
| `backend-cross-01/report.json` | `97ced83fd037c907bd6a8ad34c377b96a0c1bc04dc424c21950cbf2194da9a65` |
| `attention-local-01/report.json` | `aa105ea3d1678f787840dc84d85997ed9e8ba60c33ae007f007aa3723717f51e` |

Frozen helper pins include:

| Source | SHA-256 |
| --- | --- |
| `cdrm/pretrained/campaign_recipe.py` | `6a0b726de552959972d4bbc94a497c378aba4f9682fc5aeedbfb9daa84192a5d` |
| `scripts/olmo_campaign_ddp_probe.py` | `d42aefcf3f9d4e30f6d4253e61f66610adff2c0bec83652963f4c013ac8c28a4` |
| `scripts/olmo_campaign_precision_components.py` | `fe6dd3918551f8cfbf3c0ebbdf80ccda45d328d2796f10382074c3774cbd0b60` |
| `scripts/olmo_campaign_precision_bridge.py` | `9fac9de8a5323630c89b41e94f672e5770039c193961cfb80b6672023d822ab7` |

The full inventories remain in the old reports and the new audit:
`.runtime/olmo-recurrence-precision/baseline-audit-01/{audit.py,report.json}`.
The CPU-only audit checked all **308 source/snapshot pairs**, the four unchanged
reports and local retained bytes for all five PR40 receipts (including
closeout): **10 objects and 352 inventory members**. It checked these against
the already verified receipts; it performed no new cloud download. All four
arm fixtures and loss-count controls passed. Old sources, reports and retained
archives were not modified. See the prior
[storage receipt](../olmo-precision-localization/storage-receipt.md) for cloud
generation pins and the original independent readback evidence.
