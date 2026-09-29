# Numerical-localization test ledger

2026-09-29. Initial implementation source freeze: `c8575d7`; exact per-file
source inventories in each report are authoritative. The initial
[protocol](protocol.md), bridge and auxiliary source bytes remain frozen.
Crossed-backend work has its own [adaptive protocol](adaptive-protocol.md).

## CPU checks

All listed commands ran in the explicitly CPU-only project container. These
suites overlap; their totals must not be summed as distinct tests.

| Evidence under `.runtime/olmo-precision-localization/` | Result | Scope |
| --- | --- | --- |
| `cpu-bridge-01.log` | 13 passed, 4.80 s | Bridge and existing component checks |
| `cpu-diagnostics-01.log` | 33 passed, 5.14 s | Combined bridge and auxiliary scope |
| `cpu-aux-final-01.log` | 20 passed, 3.07 s | Final auxiliary reporting/source-pin scope |

The last two suites each record the same CPU warning: a BF16-input/FP32-weight
RMSNorm fixture cannot use the fused CPU implementation. This is a dispatch
warning in the test fixture, not a CUDA numerical failure or a GPU fallback.

## GPU cases and health gates

All three probes ran as single processes on CUDA device 0 inside the project
container, with a 900-second external timeout. The second H100 was not used;
two virtual input records do not imply DDP or two-GPU execution. No optimizer
update, gradient clipping, scheduler advance or graph capture occurred.

| Stage | Numerical cases | Physical backward calls | Operational result | Duration | W&B |
| --- | --- | ---: | --- | ---: | --- |
| `bridge-01` | CE/combined × FP32 math/eager, BF16 math/eager, BF16 Flash/Triton: 6 aggregate cases | 12 model backwards | All six cases and final input/weight/RNG/source integrity checks pass | 51.88 s | [p6oooxmt](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p6oooxmt) |
| `auxiliary-01` | Latent/KL × FP32/BF16 × sparse/prepared: 8 aggregate cases | 16 loss-only backwards | All eight cases and final fixture/weight/RNG/source integrity check pass | 34.72 s | [mdo63etu](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mdo63etu) |
| `backend-cross-01` | CE-only math/eager and Flash/Triton references, plus new Flash/eager condition: 3 aggregate cases | 6 model backwards | All three cases, reference reproduction and final integrity checks pass | 39.70 s | [mrgqe7di](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/mrgqe7di) |

All three final statuses are `passed_operational_diagnostic`. Cross-precision and
layout comparison flags are descriptive. An operational pass does not clear
the original numerical qualifications or declare gradients equivalent.

Deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic
mode and cuDNN benchmarking off are recorded. TF32 is off. BF16 matmul reduced
precision reduction remains enabled, while math-SDPA reduced precision
reduction is disabled; these recorded runtime controls are not inferred from
the shorthand precision labels. BF16 RT attention remains `mixed` in both
BF16 bridge paths. Other flags match the existing production BF16 path except
the intended SDPA and RT tile forward/backward selection.

## Independently checked provenance

A separate read-only inspection verified **228 source/snapshot pairs**:
76 in the bridge, 74 in the auxiliary report and 78 in the crossed-backend
report. It also verified the complete
exported fixture file SHA256/size and decoded all **22 base64 tensor payloads**
against their reported dtype, shape and raw-byte SHA256. The shared checkpoint,
recipe and bridge source inventories, readout/predictor hashes and fixture
provenance match across the bridge export and auxiliary report.

The anchor is 3,517,265 bytes, with two records containing four pass-output
states each. Hidden states and token embeddings have actual dtype FP32 and
shape `[2,16,2048]`. The batch tensors have shape `[2,16]`. The auxiliary probe
retains these values for both loss layouts; BF16 refers to its autocast
arithmetic, not a forced conversion of the anchor payload. The readout is
fixed; active predictor gradients and incoming hidden/embedding cotangents are
measured. No readout-weight optimization or backbone recomputation is claimed.

The source is OLMo-1B step 60000, revision
`81b71efbce6f4dada57c94860301af4298bcd351`. The original NFR K4 recipe, masks,
keyed jitter, global counts and pass weights are shared. Actual diagnostic
shape is T16/B2/two records, with 29 valid inputs, 25 CE targets, 25 latent
pairs and 21 KL triples. The recipe's campaign `sequence_length` remains 1024;
the explicit diagnostic fixture overrides its input length to 16. These probes
therefore provide no T1024 numerical acceptance.

| Evidence | SHA256 |
| --- | --- |
| `bridge-01/report.json` | `39bf047c9908c852364ae5bc4e6f126bf2a3dc52bcc03cec561ebcb84727bb0b` |
| `auxiliary-01/report.json` | `c4946da63c6275a4fcd926292b0296337f57d10553233c7184e94e38b617d5fc` |
| `backend-cross-01/report.json` | `97ced83fd037c907bd6a8ad34c377b96a0c1bc04dc424c21950cbf2194da9a65` |
| `bridge-01/auxiliary-fixture.json` | `aeab58a88c7eba15448a1b7630c9af747e492b3760b2364da5cd53380e063b27` |
| Frozen `protocol.md` | `31647b30dac411c99d4ace0ca52f82d0c761f71684a20109546ab060473687b9` |
| Recipe | `29409f66064a2ab2034cf549fe45ecc26f83841d0f9755d87954616e0cdd5da8` |
| Native checkpoint weights | `ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c` |
| Reconstructed readout | `1d7ba25fa485040fdb7f1f96821d38ec2a1e0fac112cebb8de40bbb8227de87f` |

Predictor tensor hashes remain in the fixture/report rather than being
duplicated in this table. Their equality was checked; no large new weight
dump was created.

The independent inspection also compared the bridge with
`.runtime/olmo-packed-campaign/precision-components-01/report.json`: fixture,
source checkpoint and recipe pins match; FP32/current-BF16 CE and combined
loss sums, counts and scalar objectives reproduce exactly. Current-BF16
per-parameter error rows and gradient geometry versus FP32 also reproduce
exactly. This confirms continuity of the initial diagnostic; it does not
convert its prior failures into passes.

The crossed-backend audit independently verified the finalized bridge-report
SHA, fixture/recipe/checkpoint pins, runtime controls and both reproduced
reference rows. New Flash/eager and reference Flash/Triton metrics and forward
fingerprints match exactly. Their in-process gradient comparison records zero
maximum absolute error and zero relative L2 for **all 71 parameter tensors**;
all **eight pass/record hidden-state and incoming-cotangent comparisons** also
have zero difference. Each Flash path's comparison with math/eager is identical,
including full gradient relative L2 `0.7984590233137122`. The recomputed anchors
were necessary because the initial bridge did not retain full gradient vectors.
These observations qualify the tested CE/T16 pairing, not combined objectives
or longer contexts, and do not certify either BF16 path against FP32.

## Retention and remaining scope

Both `retention/bridge-01.json` and `retention/auxiliary-01.json` report verified
retention under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T043105Z/`,
using suffixes `precision-bridge-01` and `precision-auxiliary-01`. Their source
snapshots and the bridge's small anchor export are retained. Final storage
object auditing is tracked separately by root; this ledger does not claim a
new checkpoint restoration exercise.

Root is handling crossed-backend cloud retention separately. Its completed
source/fixture checks are recorded above; this ledger does not substitute for
the storage-object audit. Across the three probes there were nine model
gradient cases (18 physical model backwards, including the repeated reference
cases) and eight auxiliary cases (16 loss-only backwards).

No fixed-QKV ordinary-attention result, common-cotangent backbone propagation,
selective precision correction, changed-state/real-packed-fixture check or long
training trajectory has yet been established by these three reports. A bounded
local ordinary-attention check is under consideration; no fix is claimed.
