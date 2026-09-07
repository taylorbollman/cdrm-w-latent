# Existing FP32 recurrent-backward evidence audit

This is a read-only audit of the retained Stage B NUM records. No GPU work, new numerical measurement, source change, or clearance decision was performed. The original failures remain failures. The [machine-readable audit](existing-evidence.json) preserves exact configurations, provenance, hashes, compiler counters, all compared names, and all 222 failing FP32 tensor rows.

## What the original checks did

Every fixture uses seed 937, batch 2, 12 blocks, four full-MHA heads, pre-norm ALiBi, GELU/MLP 4D, affine Q/K normalization, Mitchell initialization, no biases/dropout/RoPE/embedding norm, an untied head, and recurrent block 3 at rho 1. The tiled implementation uses four backward MLP chunks and compiled helpers; the naïve implementation uses eager reference helpers. All 100 parameter gradients plus the embedding-output gradient are checked. The old input hook is **not** at block 3 input.

These fixtures backpropagate a random output cotangent, not masked cross-entropy. Model construction and randomness occur in this order: CUDA naïve R3 initialization, CUDA tiled R3 initialization, weight conversion, CUDA token draw, then CUDA normal cotangent draw. The cotangent is rounded through BF16 back to FP32. FP32 comparison modes disable autocast and run before separate BF16 modes, so rounding the cotangent does not mean BF16 model arithmetic.

The unit-direction variant divides that whole `[B,T,V]` FP32 cotangent by its FP32 L2 norm before backward. The normalized values need not remain BF16-representable. Three recorded unit norms equal 1.0; D256/T128 records 0.9999999403953552. Raw D32 cotangent norms were not saved; raw D256/T16 records 90.52316284179688.

The legacy rule is `abs(error) <= 2e-6 + 2e-5*abs(reference)` at every coordinate. Reducing cotangent magnitude reduces gradient magnitudes while leaving the absolute tolerance fixed. Therefore a unit-direction pass is a separate result, not a retroactive pass of the raw check.

## Raw FP32 results

Each row covers 101 gradient tensors. All compared gradients are finite and all four logit checks pass. The largest error, largest relative L2, and largest error/RMS can belong to different tensors.

| Fixture | Failed gradients (parameters + input) | Largest absolute gradient error | Worst tensor relative L2 | Worst max-error/reference-RMS |
|---|---:|---:|---:|---:|
| [D32/T128/V256](../stage-b/backend-num-d32-t128.json) | 61 (60 + 1) | 0.000213623 | 8.79638e-07 | 4.4947e-06 |
| [D32/T256/V256](../stage-b/backend-num-d32-t256.json) | 60 (59 + 1) | 0.000221252 | 7.91278e-07 | 6.9653e-06 |
| [D32/T512/V256](../stage-b/backend-num-d32-t512.json) | 62 (61 + 1) | 0.000457764 | 8.1075e-07 | 4.76437e-06 |
| [D256/T16/V256](../stage-b/backend-num-d256-t16.json) | 39 (38 + 1) | 7.24792e-05 | 1.04751e-06 | 1.0471e-05 |

The largest raw absolute error is 4.57763671875e-4 in D32/T512 `transformer.blocks.3.ff_out.weight`. In D256/T16 it is 7.2479248046875e-5 in the embedding parameter/input comparison. These maxima are not guaranteed to identify the coordinates responsible for the legacy failures.

The retained relative metrics use FP64 reductions: relative L2 divides by `max(norm(reference),1e-12)`; max-error/RMS divides by `max(sqrt(mean(reference**2)),1e-12)`. These floors are inactive in the audited records. Per-tensor relative L2 stays below 1.05e-6, but that alone neither establishes correctness nor proves cancellation as the cause.

## Unit-direction results

All four separately labeled unit-direction fixtures pass the unchanged legacy bounds for logits and gradients. D32 raw/unit token hashes match for each length, consistent with the recorded generation order. D256/T128 uses vocabulary 1024 and has no matching raw record among these eight fixtures.

| Fixture | Largest absolute gradient error | Worst tensor relative L2 | Worst max-error/reference-RMS |
|---|---:|---:|---:|
| [D32/T128/V256](../stage-b/backend-num-d32-t128-unit-fp32.json) | 6.55651e-07 | 9.03512e-07 | 6.24246e-06 |
| [D32/T256/V256](../stage-b/backend-num-d32-t256-unit-fp32.json) | 8.34465e-07 | 8.32919e-07 | 4.92685e-06 |
| [D32/T512/V256](../stage-b/backend-num-d32-t512-unit-fp32.json) | 7.7486e-07 | 7.64301e-07 | 7.13834e-06 |
| [D256/T128/V1024](../stage-b/backend-num-d256-t128-unit-fp32.json) | 4.47035e-07 | 1.00413e-06 | 9.9954e-06 |

## Provenance and reproduction limits

Recorded environment: H100 80 GB, Python 3.12.3, PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3, cuDNN 92300. Deterministic algorithms and TF32-off are recorded. Recompile limit 64, accumulated limit 256, and failure on recompile-limit fallback are recorded. Compiled graphs were observed; graph-break dictionaries are empty. Compiler totals include subsequent BF16 modes when those were run, so they are not isolated FP32 graph counts.

The earliest D32/T128 report used automatic SDPA dispatch, as documented in the [retained backend summary](../stage-b/backend-summary.md); its JSON does not explicitly identify dispatch. All subsequent audited reports explicitly record math SDPA. A strict-math rerun of the earliest fixture must be labeled a changed execution profile.

All recorded runs identify project HEAD `a605d3c167034e2a1da8f3acc4677f3d07b9c06f` (dirty; the harness was not tracked at that HEAD) and clean recurrent-transformer fork HEAD `0a82392fa394084e05f24cef09fe53dda85de71a`. Current model/config/conversion/common hashes match the retained records. Exact source hashes and present-day verification are in the JSON.

| Harness record | Recorded SHA-256 | Matches current harness |
|---|---|---|
| backend-num-d32-t128.json | `6c863ce7de64e65620879a6b8152d1fd36999c1c85622584af93a626cd4fadc4` | No |
| backend-num-d32-t256.json | `b53dab6d77a9d4b0bfefa7946d1be14543ba2917a46c3352bc59fe99edb455d1` | No |
| backend-num-d32-t512.json | `8f20c07e852686a65fdd6d4ba57c381f7fe313599a655e52bd4f4cb06a905b99` | No |
| backend-num-d256-t16.json | `41e0ba1b5f01a645718e44766e78f4894fa92855bc10de7647b7c51f730d0441` | Yes |

All unit-direction records also match current harness SHA-256 `41e0ba1b5f01a645718e44766e78f4894fa92855bc10de7647b7c51f730d0441`.

The current harness requests highest FP32 matmul precision, one CPU thread, and a default `CUBLAS_WORKSPACE_CONFIG=:4096:8` using `setdefault`. The effective historical CUBLAS environment value was not recorded. Old reports retain arguments, configurations, token/cotangent hashes, software/source identities and aggregate metrics, but no initialized checkpoint/model-state hash, gradient arrays, failing coordinate/count, or raw cotangent array. The byte-identical early D32 harness was not recovered in this audit. Reproduction can verify generated token/cotangent hashes; it should not claim stronger provenance than these records provide.

The current matching harness can regenerate the D256/T16 raw fixture using its recorded arguments; run a new output path and the required project container. `--precision fp32` omits the separate BF16 modes and preserves the earlier FP32 computation sequence, but changes the original command and total compiler counters. The root investigation will separately reproduce the earliest D32/T128 fixture under its automatic-SDPA profile and compare explicit math SDPA.

Raw fixtures initialize recurrent models on CUDA. The actual Stage B pilot initialized SEQ on CPU and converted to R3, so these are not its initialization or trained checkpoints. Actual masked-CE, recurrent-block-input gradients, earlier persistent-write influence, independent cotangent scaling, and paired optimizer updates remain new evidence requested for the investigation.

## Interpretation and new-check policy

The original summaries cannot identify near-zero offending entries. A tensor can have a large reference norm while a few coordinates nearly cancel; conversely, a small aggregate error does not demonstrate that cancellation caused a particular failure. Save the coordinate with the largest tolerance violation, its reference/candidate values, the count of failing coordinates, tensor norms, and structural-zero status. Do not substitute the maximum-absolute-error coordinate for that information.

The new harness will retain all legacy flags and separately apply fixed per-tensor relative-L2 and max-error/reference-RMS limits of 2e-5, reusing the original relative-error coefficient as an explicit engineering budget. This is not a formal floating-point theorem or a waiver. Scaling checks should compare `G(alpha*v)/alpha` with `G(v)` at fixed forward state for alpha 1/32,1,32 and a CE-sized direction, with documented denominator floors. Actual CE and optimizer-update comparisons are required before drawing a clearance conclusion. Parameter-update deltas and optimizer moments must be reported separately from post-step weight differences.

BF16 and optional accumulation failures are separate unsupported paths. They are not used here as proof that Stage B FP32/no-accumulation results are invalid. No source/core patch or final correctness decision follows from this audit alone.

## Complete raw FP32 failing-tensor lists

The tables below are copied from the recorded FP32 comparison rows, without applying the new criteria. The JSON also lists every compared gradient name, including those that passed.

### D32/T128/V256

Source: [backend-num-d32-t128.json](../stage-b/backend-num-d32-t128.json); record SHA-256 `764508da8d66481acc06f99157e62449cc0298448a5dc6840e22ba03c8c3406c`.

| Failing tensor | Max absolute error | Relative L2 | Reference RMS | Max error / RMS |
|---|---:|---:|---:|---:|
| `input` | 5.34057617e-05 | 4.26405811e-07 | 21.3636207 | 2.49984599e-06 |
| `transformer.wte.weight` | 6.48498535e-05 | 4.2528443e-07 | 22.2043895 | 2.9205871e-06 |
| `transformer.blocks.0.k_norm.weight` | 3.43322754e-05 | 4.90729124e-07 | 28.4209287 | 1.20799274e-06 |
| `transformer.blocks.0.q_norm.weight` | 3.05175781e-05 | 4.29175398e-07 | 28.4209278 | 1.07377135e-06 |
| `transformer.blocks.0.attn_out.weight` | 0.000152587891 | 4.52864372e-07 | 69.2664785 | 2.20291105e-06 |
| `transformer.blocks.0.ff_out.weight` | 0.000213623047 | 4.20727667e-07 | 62.815569 | 3.40079777e-06 |
| `transformer.blocks.0.att_proj.weight` | 9.15527344e-05 | 5.04969075e-07 | 34.4664022 | 2.65628927e-06 |
| `transformer.blocks.0.ff_proj.weight` | 6.10351562e-05 | 4.40963132e-07 | 20.7961565 | 2.93492484e-06 |
| `transformer.blocks.0.attn_norm.weight` | 5.7220459e-05 | 4.27340381e-07 | 68.324364 | 8.37482497e-07 |
| `transformer.blocks.1.attn_out.weight` | 0.000122070312 | 3.68414985e-07 | 59.8226934 | 2.04053521e-06 |
| `transformer.blocks.1.ff_out.weight` | 0.000122070312 | 3.70302943e-07 | 48.4096145 | 2.52161299e-06 |
| `transformer.blocks.1.att_proj.weight` | 5.34057617e-05 | 3.70112649e-07 | 20.5362055 | 2.60056619e-06 |
| `transformer.blocks.1.ff_proj.weight` | 3.05175781e-05 | 4.2299855e-07 | 11.8882536 | 2.56703626e-06 |
| `transformer.blocks.2.attn_out.weight` | 0.000106811523 | 4.14893369e-07 | 46.7614744 | 2.28417783e-06 |
| `transformer.blocks.2.ff_out.weight` | 0.000106811523 | 3.79064071e-07 | 44.5730063 | 2.39632756e-06 |
| `transformer.blocks.2.att_proj.weight` | 3.05175781e-05 | 3.9186412e-07 | 12.2460866 | 2.49202698e-06 |
| `transformer.blocks.2.ff_proj.weight` | 2.47955322e-05 | 4.03592656e-07 | 8.77510323 | 2.82566844e-06 |
| `transformer.blocks.3.attn_out.weight` | 0.000122070312 | 4.7692956e-07 | 45.4814017 | 2.68396109e-06 |
| `transformer.blocks.3.ff_out.weight` | 0.000183105469 | 4.6439524e-07 | 45.8924541 | 3.98988183e-06 |
| `transformer.blocks.3.ff_proj.weight` | 2.67028809e-05 | 4.58620752e-07 | 7.76868164 | 3.43724741e-06 |
| `transformer.blocks.3.attn_norm.weight` | 2.67028809e-05 | 4.94102045e-07 | 14.9868436 | 1.78175482e-06 |
| `transformer.blocks.3.ff_norm.weight` | 1.36494637e-05 | 3.3873972e-07 | 16.2808777 | 8.38373946e-07 |
| `transformer.blocks.3.q_proj.weight` | 1.26361847e-05 | 5.61818182e-07 | 4.74687987 | 2.661998e-06 |
| `transformer.blocks.3.kv_proj.weight` | 5.34057617e-05 | 3.88429196e-07 | 13.3974278 | 3.9862698e-06 |
| `transformer.blocks.4.attn_out.weight` | 7.62939453e-05 | 4.12468109e-07 | 36.1326884 | 2.11149374e-06 |
| `transformer.blocks.4.ff_out.weight` | 9.15527344e-05 | 3.98285188e-07 | 32.74298 | 2.79610269e-06 |
| `transformer.blocks.4.att_proj.weight` | 2.02655792e-05 | 3.62900155e-07 | 8.54817049 | 2.37075047e-06 |
| `transformer.blocks.4.ff_proj.weight` | 1.52587891e-05 | 4.25974348e-07 | 5.25036775 | 2.90623244e-06 |
| `transformer.blocks.4.ff_norm.weight` | 6.67572021e-06 | 3.04813219e-07 | 10.8198765 | 6.16986729e-07 |
| `transformer.blocks.5.attn_out.weight` | 8.01086426e-05 | 3.84094084e-07 | 46.2448238 | 1.73227263e-06 |
| `transformer.blocks.5.ff_out.weight` | 9.91821289e-05 | 3.9880095e-07 | 30.9987411 | 3.19955344e-06 |
| `transformer.blocks.5.att_proj.weight` | 2.45571136e-05 | 4.75514131e-07 | 6.60394154 | 3.71855406e-06 |
| `transformer.blocks.5.ff_proj.weight` | 1.28746033e-05 | 4.35353785e-07 | 4.53370612 | 2.8397525e-06 |
| `transformer.blocks.6.attn_out.weight` | 7.62939453e-05 | 3.77321371e-07 | 38.917261 | 1.96041405e-06 |
| `transformer.blocks.6.ff_out.weight` | 0.000114440918 | 4.11091163e-07 | 30.0341111 | 3.81036474e-06 |
| `transformer.blocks.6.att_proj.weight` | 1.52587891e-05 | 3.69574889e-07 | 5.57388384 | 2.7375506e-06 |
| `transformer.blocks.6.ff_proj.weight` | 1.23977661e-05 | 4.52264485e-07 | 4.01118831 | 3.09079634e-06 |
| `transformer.blocks.7.attn_out.weight` | 9.15527344e-05 | 3.90607422e-07 | 32.4552778 | 2.82088894e-06 |
| `transformer.blocks.7.ff_out.weight` | 8.01086426e-05 | 4.26383217e-07 | 27.0863055 | 2.95753301e-06 |
| `transformer.blocks.7.att_proj.weight` | 1.71661377e-05 | 3.72905577e-07 | 5.53009751 | 3.1041293e-06 |
| `transformer.blocks.7.ff_proj.weight` | 1.23977661e-05 | 4.45509804e-07 | 3.56470998 | 3.47791719e-06 |
| `transformer.blocks.7.attn_norm.weight` | 1.71661377e-05 | 5.75102541e-07 | 7.3484545 | 2.33602014e-06 |
| `transformer.blocks.8.attn_out.weight` | 6.86645508e-05 | 3.68929641e-07 | 33.625805 | 2.04201954e-06 |
| `transformer.blocks.8.ff_out.weight` | 7.62939453e-05 | 3.85712731e-07 | 31.0474742 | 2.45733179e-06 |
| `transformer.blocks.8.att_proj.weight` | 1.43051147e-05 | 4.11820982e-07 | 4.72508921 | 3.02748035e-06 |
| `transformer.blocks.8.ff_proj.weight` | 1.23977661e-05 | 4.73559182e-07 | 3.26081232 | 3.80204835e-06 |
| `transformer.blocks.9.attn_out.weight` | 0.000106811523 | 3.60521693e-07 | 34.0772146 | 3.13439712e-06 |
| `transformer.blocks.9.ff_out.weight` | 6.86645508e-05 | 3.85762872e-07 | 27.9602291 | 2.45579356e-06 |
| `transformer.blocks.9.att_proj.weight` | 1.71661377e-05 | 3.5958892e-07 | 4.74069321 | 3.62101848e-06 |
| `transformer.blocks.9.ff_proj.weight` | 9.53674316e-06 | 4.22947598e-07 | 3.32086909 | 2.87176125e-06 |
| `transformer.blocks.9.attn_norm.weight` | 8.34465027e-06 | 5.0894384e-07 | 4.83860362 | 1.72459886e-06 |
| `transformer.blocks.9.ff_norm.weight` | 5.00679016e-06 | 3.51298688e-07 | 6.75089231 | 7.41648649e-07 |
| `transformer.blocks.10.attn_out.weight` | 0.000183105469 | 3.34472785e-07 | 40.7380932 | 4.49469904e-06 |
| `transformer.blocks.10.ff_out.weight` | 7.62939453e-05 | 3.88392865e-07 | 27.7501696 | 2.74931456e-06 |
| `transformer.blocks.10.att_proj.weight` | 7.62939453e-06 | 3.91377099e-07 | 3.65851849 | 2.08537815e-06 |
| `transformer.blocks.10.ff_proj.weight` | 9.53674316e-06 | 4.25312307e-07 | 3.05539737 | 3.12127753e-06 |
| `transformer.blocks.11.attn_out.weight` | 6.10351562e-05 | 3.05226792e-07 | 38.9857302 | 1.56557684e-06 |
| `transformer.blocks.11.ff_out.weight` | 6.86645508e-05 | 3.69398331e-07 | 28.515558 | 2.40796799e-06 |
| `transformer.blocks.11.att_proj.weight` | 1.14440918e-05 | 2.99747199e-07 | 4.65094212 | 2.46059647e-06 |
| `transformer.blocks.11.ff_proj.weight` | 7.15255737e-06 | 4.16175332e-07 | 2.96633655 | 2.4112427e-06 |
| `transformer.ff_out.weight` | 2.0980835e-05 | 2.69240743e-07 | 15.8216273 | 1.32608578e-06 |

### D32/T256/V256

Source: [backend-num-d32-t256.json](../stage-b/backend-num-d32-t256.json); record SHA-256 `384e76b27d7f8e3f00b520a2f480a25d8f7b97f1b4e7f9637e1e317e30a8851d`.

| Failing tensor | Max absolute error | Relative L2 | Reference RMS | Max error / RMS |
|---|---:|---:|---:|---:|
| `input` | 6.48498535e-05 | 4.78616996e-07 | 19.7391438 | 3.28534278e-06 |
| `transformer.wte.weight` | 9.72747803e-05 | 4.82618853e-07 | 28.5064467 | 3.4123783e-06 |
| `transformer.blocks.0.attn_out.weight` | 0.000221252441 | 5.46136142e-07 | 86.7642833 | 2.55004056e-06 |
| `transformer.blocks.0.ff_out.weight` | 0.000213623047 | 3.8933722e-07 | 81.9777321 | 2.60586676e-06 |
| `transformer.blocks.0.att_proj.weight` | 0.000137329102 | 5.85319883e-07 | 45.0031816 | 3.05154206e-06 |
| `transformer.blocks.0.ff_proj.weight` | 5.34057617e-05 | 4.19927283e-07 | 27.1575869 | 1.96651351e-06 |
| `transformer.blocks.1.attn_out.weight` | 0.000213623047 | 5.13449737e-07 | 66.2616253 | 3.2239331e-06 |
| `transformer.blocks.1.ff_out.weight` | 0.000122070312 | 3.86497697e-07 | 58.1257254 | 2.1001082e-06 |
| `transformer.blocks.1.att_proj.weight` | 7.62939453e-05 | 4.40729708e-07 | 24.0489154 | 3.17244849e-06 |
| `transformer.blocks.1.ff_proj.weight` | 4.19616699e-05 | 3.74493148e-07 | 16.141185 | 2.59966476e-06 |
| `transformer.blocks.2.attn_out.weight` | 0.000137329102 | 5.152787e-07 | 52.2527722 | 2.62816872e-06 |
| `transformer.blocks.2.ff_out.weight` | 9.91821289e-05 | 3.86481312e-07 | 52.9070933 | 1.8746471e-06 |
| `transformer.blocks.2.att_proj.weight` | 4.57763672e-05 | 4.7389306e-07 | 14.3132011 | 3.19819215e-06 |
| `transformer.blocks.2.ff_proj.weight` | 2.76565552e-05 | 3.88003829e-07 | 11.0058262 | 2.51290133e-06 |
| `transformer.blocks.2.attn_norm.weight` | 2.67028809e-05 | 3.92708425e-07 | 25.3192527 | 1.05464728e-06 |
| `transformer.blocks.2.ff_norm.weight` | 1.62124634e-05 | 3.75644356e-07 | 19.8857944 | 8.15278638e-07 |
| `transformer.blocks.3.k_norm.weight` | 1.23977661e-05 | 6.45991364e-07 | 5.81683216 | 2.13136047e-06 |
| `transformer.blocks.3.q_norm.weight` | 1.14440918e-05 | 6.50702493e-07 | 5.81683178 | 1.96740979e-06 |
| `transformer.blocks.3.attn_out.weight` | 0.00016784668 | 6.34726286e-07 | 50.6926556 | 3.31106504e-06 |
| `transformer.blocks.3.ff_out.weight` | 0.000213623047 | 4.91318884e-07 | 54.6190641 | 3.9111444e-06 |
| `transformer.blocks.3.ff_proj.weight` | 3.05175781e-05 | 4.7308247e-07 | 9.61987574 | 3.17234639e-06 |
| `transformer.blocks.3.q_proj.weight` | 3.0040741e-05 | 6.95658441e-07 | 6.51701718 | 4.60958444e-06 |
| `transformer.blocks.3.kv_proj.weight` | 5.34057617e-05 | 4.23660821e-07 | 15.1053986 | 3.53554137e-06 |
| `transformer.blocks.4.attn_out.weight` | 0.000137329102 | 5.24248212e-07 | 46.1501083 | 2.97570486e-06 |
| `transformer.blocks.4.ff_out.weight` | 0.000106811523 | 3.78417126e-07 | 43.2707748 | 2.4684449e-06 |
| `transformer.blocks.4.att_proj.weight` | 6.86645508e-05 | 4.89037868e-07 | 9.85809343 | 6.96529722e-06 |
| `transformer.blocks.4.ff_proj.weight` | 3.19480896e-05 | 4.10578222e-07 | 7.14128419 | 4.47371772e-06 |
| `transformer.blocks.4.attn_norm.weight` | 3.43322754e-05 | 5.78973186e-07 | 15.3245218 | 2.24034888e-06 |
| `transformer.blocks.5.attn_out.weight` | 0.000213623047 | 5.22333778e-07 | 57.4798816 | 3.71648376e-06 |
| `transformer.blocks.5.ff_out.weight` | 9.15527344e-05 | 3.9123682e-07 | 38.975214 | 2.34899889e-06 |
| `transformer.blocks.5.att_proj.weight` | 3.43322754e-05 | 5.05147044e-07 | 8.29902338 | 4.13690549e-06 |
| `transformer.blocks.5.ff_proj.weight` | 1.52587891e-05 | 3.96859238e-07 | 6.15571009 | 2.47880242e-06 |
| `transformer.blocks.6.attn_out.weight` | 0.000175476074 | 4.92616131e-07 | 51.3997153 | 3.41395031e-06 |
| `transformer.blocks.6.ff_out.weight` | 9.15527344e-05 | 3.95078174e-07 | 37.5688389 | 2.43693276e-06 |
| `transformer.blocks.6.att_proj.weight` | 2.14576721e-05 | 4.21796792e-07 | 7.49852127 | 2.86158715e-06 |
| `transformer.blocks.6.ff_proj.weight` | 1.04904175e-05 | 3.96413755e-07 | 5.34090136 | 1.96416612e-06 |
| `transformer.blocks.7.attn_out.weight` | 9.91821289e-05 | 4.90865863e-07 | 40.105706 | 2.4730179e-06 |
| `transformer.blocks.7.ff_out.weight` | 9.15527344e-05 | 3.69553689e-07 | 35.7932531 | 2.557821e-06 |
| `transformer.blocks.7.att_proj.weight` | 2.03251839e-05 | 3.71029791e-07 | 7.26800708 | 2.7965278e-06 |
| `transformer.blocks.7.ff_proj.weight` | 1.09672546e-05 | 3.93435049e-07 | 4.77978351 | 2.29450866e-06 |
| `transformer.blocks.7.attn_norm.weight` | 7.62939453e-06 | 3.69418358e-07 | 9.00749243 | 8.47005378e-07 |
| `transformer.blocks.8.attn_out.weight` | 0.000152587891 | 4.69312246e-07 | 40.934594 | 3.7276024e-06 |
| `transformer.blocks.8.ff_out.weight` | 9.91821289e-05 | 3.69121899e-07 | 40.1724475 | 2.46890929e-06 |
| `transformer.blocks.8.att_proj.weight` | 3.05175781e-05 | 3.86266272e-07 | 6.66898811 | 4.57604327e-06 |
| `transformer.blocks.8.ff_proj.weight` | 1.00135803e-05 | 3.88031378e-07 | 4.60122933 | 2.17628369e-06 |
| `transformer.blocks.9.attn_out.weight` | 0.000160217285 | 4.03019663e-07 | 46.4380116 | 3.45013233e-06 |
| `transformer.blocks.9.ff_out.weight` | 8.96453857e-05 | 3.38704886e-07 | 39.0931042 | 2.29312529e-06 |
| `transformer.blocks.9.att_proj.weight` | 1.90734863e-05 | 3.32695148e-07 | 7.43124049 | 2.56666251e-06 |
| `transformer.blocks.9.ff_proj.weight` | 9.53674316e-06 | 3.99760219e-07 | 4.3193111 | 2.20793153e-06 |
| `transformer.blocks.10.q_norm.weight` | 4.76837158e-06 | 7.91277753e-07 | 2.57010553 | 1.85532132e-06 |
| `transformer.blocks.10.attn_out.weight` | 0.000122070312 | 3.84330503e-07 | 50.2748532 | 2.42805905e-06 |
| `transformer.blocks.10.ff_out.weight` | 0.000106811523 | 3.38518991e-07 | 37.5770269 | 2.84246872e-06 |
| `transformer.blocks.10.att_proj.weight` | 2.00271606e-05 | 4.69506149e-07 | 5.15619685 | 3.88409544e-06 |
| `transformer.blocks.10.ff_proj.weight` | 9.29832458e-06 | 3.47663156e-07 | 4.39563529 | 2.11535398e-06 |
| `transformer.blocks.10.attn_norm.weight` | 8.58306885e-06 | 4.66347802e-07 | 7.29338333 | 1.17682953e-06 |
| `transformer.blocks.11.attn_out.weight` | 9.15527344e-05 | 3.74732922e-07 | 49.4316571 | 1.85210733e-06 |
| `transformer.blocks.11.ff_out.weight` | 8.39233398e-05 | 3.42340948e-07 | 37.2589712 | 2.2524331e-06 |
| `transformer.blocks.11.att_proj.weight` | 1.90734863e-05 | 3.83802456e-07 | 5.97137364 | 3.19415389e-06 |
| `transformer.blocks.11.ff_proj.weight` | 9.53674316e-06 | 3.68716742e-07 | 3.97986235 | 2.3962495e-06 |
| `transformer.ff_out.weight` | 3.05175781e-05 | 2.70669228e-07 | 22.7509383 | 1.34137668e-06 |

### D32/T512/V256

Source: [backend-num-d32-t512.json](../stage-b/backend-num-d32-t512.json); record SHA-256 `0d441f104f33041fb3c805e18c122502682ac6d4715575ba961f155ab3c60f91`.

| Failing tensor | Max absolute error | Relative L2 | Reference RMS | Max error / RMS |
|---|---:|---:|---:|---:|
| `input` | 9.15527344e-05 | 4.7699123e-07 | 19.2161213 | 4.76437118e-06 |
| `transformer.wte.weight` | 0.000122070312 | 4.7836144e-07 | 39.0057469 | 3.12954685e-06 |
| `transformer.blocks.0.k_norm.weight` | 6.10351562e-05 | 4.32218391e-07 | 50.5613312 | 1.2071509e-06 |
| `transformer.blocks.0.q_norm.weight` | 4.95910645e-05 | 4.48549017e-07 | 50.5613291 | 9.80810144e-07 |
| `transformer.blocks.0.attn_out.weight` | 0.000198364258 | 3.96550139e-07 | 123.883681 | 1.60121379e-06 |
| `transformer.blocks.0.ff_out.weight` | 0.000274658203 | 3.14669253e-07 | 148.533461 | 1.84913353e-06 |
| `transformer.blocks.0.att_proj.weight` | 0.000152587891 | 4.81429104e-07 | 60.2493327 | 2.53260715e-06 |
| `transformer.blocks.0.ff_proj.weight` | 7.05718994e-05 | 4.07324571e-07 | 40.6910706 | 1.73433381e-06 |
| `transformer.blocks.1.attn_out.weight` | 0.000274658203 | 3.49252734e-07 | 112.67862 | 2.4375361e-06 |
| `transformer.blocks.1.ff_out.weight` | 0.000183105469 | 3.18693657e-07 | 101.345309 | 1.80674834e-06 |
| `transformer.blocks.1.att_proj.weight` | 9.15527344e-05 | 3.13167078e-07 | 40.5834823 | 2.25591125e-06 |
| `transformer.blocks.1.ff_proj.weight` | 5.7220459e-05 | 3.63032533e-07 | 23.7032657 | 2.41403272e-06 |
| `transformer.blocks.2.attn_out.weight` | 0.000183105469 | 3.03397885e-07 | 96.2495036 | 1.90240429e-06 |
| `transformer.blocks.2.ff_out.weight` | 0.00016784668 | 2.98721398e-07 | 95.1527889 | 1.76397015e-06 |
| `transformer.blocks.2.att_proj.weight` | 5.34057617e-05 | 3.34072313e-07 | 24.0082075 | 2.22447935e-06 |
| `transformer.blocks.2.ff_proj.weight` | 4.38690186e-05 | 3.58962638e-07 | 17.2828861 | 2.53829241e-06 |
| `transformer.blocks.3.k_norm.weight` | 1.52587891e-05 | 7.3574475e-07 | 7.79486483 | 1.95754377e-06 |
| `transformer.blocks.3.q_norm.weight` | 9.53674316e-06 | 4.69491516e-07 | 7.79486557 | 1.22346474e-06 |
| `transformer.blocks.3.attn_out.weight` | 0.000244140625 | 4.27449843e-07 | 93.6468587 | 2.60703486e-06 |
| `transformer.blocks.3.ff_out.weight` | 0.000457763672 | 4.66537592e-07 | 100.340725 | 4.56209254e-06 |
| `transformer.blocks.3.ff_proj.weight` | 7.62939453e-05 | 5.01073009e-07 | 16.3923559 | 4.65423921e-06 |
| `transformer.blocks.3.q_proj.weight` | 1.90734863e-05 | 4.83110851e-07 | 9.37156373 | 2.0352512e-06 |
| `transformer.blocks.3.kv_proj.weight` | 0.000106811523 | 3.27074389e-07 | 29.7749319 | 3.58729698e-06 |
| `transformer.blocks.4.attn_out.weight` | 9.15527344e-05 | 2.75733505e-07 | 94.0695244 | 9.73245426e-07 |
| `transformer.blocks.4.ff_out.weight` | 0.000183105469 | 3.14366129e-07 | 73.1048648 | 2.50469609e-06 |
| `transformer.blocks.4.att_proj.weight` | 4.00543213e-05 | 2.65053807e-07 | 19.0896524 | 2.09822161e-06 |
| `transformer.blocks.4.ff_proj.weight` | 2.47955322e-05 | 3.42995337e-07 | 10.8314481 | 2.28921673e-06 |
| `transformer.blocks.5.attn_out.weight` | 0.000137329102 | 3.01075416e-07 | 102.483665 | 1.34000967e-06 |
| `transformer.blocks.5.ff_out.weight` | 0.000141143799 | 3.23634017e-07 | 65.1353002 | 2.1669325e-06 |
| `transformer.blocks.5.att_proj.weight` | 3.81469727e-05 | 2.89236725e-07 | 17.2384537 | 2.21289991e-06 |
| `transformer.blocks.5.ff_proj.weight` | 2.28881836e-05 | 3.42958741e-07 | 10.0242045 | 2.28329176e-06 |
| `transformer.blocks.6.k_norm.weight` | 1.14440918e-05 | 5.85823782e-07 | 5.22413178 | 2.19062081e-06 |
| `transformer.blocks.6.q_norm.weight` | 1.33514404e-05 | 6.52308722e-07 | 5.22413196 | 2.55572419e-06 |
| `transformer.blocks.6.attn_out.weight` | 0.000137329102 | 2.52679562e-07 | 97.7500389 | 1.40490074e-06 |
| `transformer.blocks.6.ff_out.weight` | 0.000144004822 | 3.41313081e-07 | 61.9294994 | 2.32530253e-06 |
| `transformer.blocks.6.att_proj.weight` | 3.05175781e-05 | 2.62971231e-07 | 13.5800732 | 2.24723222e-06 |
| `transformer.blocks.6.ff_proj.weight` | 1.90734863e-05 | 3.5593784e-07 | 8.55351846 | 2.22989948e-06 |
| `transformer.blocks.7.attn_out.weight` | 0.000114440918 | 2.72952444e-07 | 80.537063 | 1.42097208e-06 |
| `transformer.blocks.7.ff_out.weight` | 0.000122070312 | 3.35586505e-07 | 58.5544431 | 2.08473185e-06 |
| `transformer.blocks.7.att_proj.weight` | 2.67028809e-05 | 2.4855855e-07 | 14.2452019 | 1.87451755e-06 |
| `transformer.blocks.7.ff_proj.weight` | 1.38282776e-05 | 3.58951605e-07 | 7.58279038 | 1.8236397e-06 |
| `transformer.blocks.8.attn_out.weight` | 0.000106811523 | 2.51038832e-07 | 77.8140497 | 1.37265087e-06 |
| `transformer.blocks.8.ff_out.weight` | 0.000129699707 | 2.98448932e-07 | 68.652202 | 1.88922865e-06 |
| `transformer.blocks.8.att_proj.weight` | 2.28881836e-05 | 2.65061818e-07 | 11.9390861 | 1.91708003e-06 |
| `transformer.blocks.8.ff_proj.weight` | 1.38282776e-05 | 3.37042671e-07 | 7.40646905 | 1.86705399e-06 |
| `transformer.blocks.8.attn_norm.weight` | 1.14440918e-05 | 3.41447963e-07 | 15.1854688 | 7.53621236e-07 |
| `transformer.blocks.8.ff_norm.weight` | 1.52587891e-05 | 4.72208188e-07 | 12.5370436 | 1.21709628e-06 |
| `transformer.blocks.9.attn_out.weight` | 0.000122070312 | 2.36114585e-07 | 75.3884552 | 1.61921759e-06 |
| `transformer.blocks.9.ff_out.weight` | 0.000152587891 | 3.16247078e-07 | 62.2217049 | 2.45232577e-06 |
| `transformer.blocks.9.att_proj.weight` | 2.38418579e-05 | 2.31484024e-07 | 11.9431879 | 1.99627253e-06 |
| `transformer.blocks.9.ff_proj.weight` | 1.23977661e-05 | 3.55267284e-07 | 6.82283749 | 1.81709826e-06 |
| `transformer.blocks.10.k_norm.weight` | 5.7220459e-06 | 5.87741839e-07 | 3.83258724 | 1.49299821e-06 |
| `transformer.blocks.10.attn_out.weight` | 9.15527344e-05 | 2.48347366e-07 | 76.7775517 | 1.19244144e-06 |
| `transformer.blocks.10.ff_out.weight` | 0.000106811523 | 3.20597578e-07 | 56.7531 | 1.88203857e-06 |
| `transformer.blocks.10.att_proj.weight` | 2.28881836e-05 | 3.03724599e-07 | 8.58982887 | 2.66456805e-06 |
| `transformer.blocks.10.ff_proj.weight` | 1.45435333e-05 | 3.72353186e-07 | 5.98126686 | 2.43151387e-06 |
| `transformer.blocks.10.ff_norm.weight` | 1.43051147e-05 | 5.37512374e-07 | 11.9851604 | 1.19356891e-06 |
| `transformer.blocks.11.attn_out.weight` | 0.000106811523 | 2.48842226e-07 | 78.8075802 | 1.35534581e-06 |
| `transformer.blocks.11.ff_out.weight` | 0.000122070312 | 3.29966843e-07 | 55.391906 | 2.20375721e-06 |
| `transformer.blocks.11.att_proj.weight` | 1.90734863e-05 | 2.87242465e-07 | 8.80944947 | 2.16511672e-06 |
| `transformer.blocks.11.ff_proj.weight` | 1.52587891e-05 | 3.63588022e-07 | 5.97568268 | 2.55348048e-06 |
| `transformer.ff_out.weight` | 4.57763672e-05 | 2.78046927e-07 | 32.4790444 | 1.40941238e-06 |

### D256/T16/V256

Source: [backend-num-d256-t16.json](../stage-b/backend-num-d256-t16.json); record SHA-256 `b233ee77c5c32fa029ec8bf69e445ce55377e5d028fd15eed19e03ec100bed8a`.

| Failing tensor | Max absolute error | Relative L2 | Reference RMS | Max error / RMS |
|---|---:|---:|---:|---:|
| `input` | 7.2479248e-05 | 5.9848138e-07 | 19.5395889 | 3.70935378e-06 |
| `transformer.wte.weight` | 7.2479248e-05 | 6.04485134e-07 | 6.92187999 | 1.04710351e-05 |
| `transformer.ln_f.weight` | 7.62939453e-06 | 2.84261019e-07 | 5.39918149 | 1.41306503e-06 |
| `transformer.blocks.0.k_norm.weight` | 5.31971455e-06 | 6.80140422e-07 | 2.50477584 | 2.12382859e-06 |
| `transformer.blocks.0.q_norm.weight` | 5.30481339e-06 | 7.056906e-07 | 2.50477575 | 2.11787957e-06 |
| `transformer.blocks.0.attn_out.weight` | 2.28881836e-05 | 4.63442686e-07 | 7.71746372 | 2.96576498e-06 |
| `transformer.blocks.0.ff_out.weight` | 2.86102295e-05 | 4.24750271e-07 | 5.51852345 | 5.18439937e-06 |
| `transformer.blocks.0.att_proj.weight` | 1.90734863e-05 | 6.02085632e-07 | 3.81470207 | 4.99999371e-06 |
| `transformer.blocks.0.ff_proj.weight` | 7.62939453e-06 | 4.57469181e-07 | 1.9610669 | 3.89043053e-06 |
| `transformer.blocks.0.attn_norm.weight` | 1.33514404e-05 | 6.27575943e-07 | 6.31894772 | 2.11292149e-06 |
| `transformer.blocks.1.attn_out.weight` | 1.71661377e-05 | 3.79296511e-07 | 5.92214338 | 2.89863595e-06 |
| `transformer.blocks.1.ff_out.weight` | 1.8119812e-05 | 3.44680824e-07 | 4.41614266 | 4.10308575e-06 |
| `transformer.blocks.1.att_proj.weight` | 1.04904175e-05 | 4.83147821e-07 | 1.81261264 | 5.78745688e-06 |
| `transformer.blocks.1.ff_proj.weight` | 3.81469727e-06 | 3.72423178e-07 | 1.14393137 | 3.33472565e-06 |
| `transformer.blocks.2.attn_out.weight` | 1.52587891e-05 | 3.23153376e-07 | 5.31381111 | 2.87153396e-06 |
| `transformer.blocks.2.ff_out.weight` | 1.52587891e-05 | 3.0953611e-07 | 3.80576478 | 4.00938837e-06 |
| `transformer.blocks.2.att_proj.weight` | 8.58306885e-06 | 4.16587151e-07 | 1.3937035 | 6.15846114e-06 |
| `transformer.blocks.3.attn_out.weight` | 1.33514404e-05 | 3.29895094e-07 | 5.35929815 | 2.49126659e-06 |
| `transformer.blocks.3.ff_out.weight` | 1.43051147e-05 | 3.65962424e-07 | 3.7090397 | 3.85682438e-06 |
| `transformer.blocks.4.attn_out.weight` | 1.14440918e-05 | 3.01263396e-07 | 5.03309403 | 2.27376872e-06 |
| `transformer.blocks.4.ff_out.weight` | 1.52587891e-05 | 3.19901626e-07 | 3.66212235 | 4.16665191e-06 |
| `transformer.blocks.4.att_proj.weight` | 4.76837158e-06 | 4.14554535e-07 | 0.877811037 | 5.43211623e-06 |
| `transformer.blocks.5.attn_out.weight` | 1.33514404e-05 | 2.94453049e-07 | 4.51293415 | 2.95848332e-06 |
| `transformer.blocks.5.ff_out.weight` | 1.52587891e-05 | 3.22184838e-07 | 3.38658186 | 4.50566078e-06 |
| `transformer.blocks.5.att_proj.weight` | 4.14252281e-06 | 4.12104104e-07 | 0.75835969 | 5.46247759e-06 |
| `transformer.blocks.6.attn_out.weight` | 9.53674316e-06 | 2.79823792e-07 | 4.31315721 | 2.21108174e-06 |
| `transformer.blocks.6.ff_out.weight` | 1.33514404e-05 | 3.22925211e-07 | 3.30294519 | 4.04228338e-06 |
| `transformer.blocks.6.att_proj.weight` | 4.05311584e-06 | 4.05935961e-07 | 0.665464701 | 6.09065491e-06 |
| `transformer.blocks.7.attn_out.weight` | 1.14440918e-05 | 2.67377902e-07 | 4.48204432 | 2.55331964e-06 |
| `transformer.blocks.7.ff_out.weight` | 1.33514404e-05 | 3.16773979e-07 | 3.2617572 | 4.09332749e-06 |
| `transformer.blocks.8.attn_out.weight` | 9.53674316e-06 | 2.60042076e-07 | 4.22817037 | 2.25552481e-06 |
| `transformer.blocks.8.ff_out.weight` | 9.53674316e-06 | 3.12080253e-07 | 3.15634216 | 3.02145417e-06 |
| `transformer.blocks.9.attn_out.weight` | 8.58306885e-06 | 2.39732365e-07 | 4.31702765 | 1.98818945e-06 |
| `transformer.blocks.9.ff_out.weight` | 9.53674316e-06 | 3.16870203e-07 | 3.14827837 | 3.02919311e-06 |
| `transformer.blocks.10.attn_out.weight` | 5.7220459e-06 | 2.32083378e-07 | 4.24379089 | 1.34833361e-06 |
| `transformer.blocks.10.ff_out.weight` | 1.04904175e-05 | 3.2019415e-07 | 3.03639214 | 3.45489548e-06 |
| `transformer.blocks.11.attn_out.weight` | 6.67572021e-06 | 2.28303717e-07 | 3.98578205 | 1.67488341e-06 |
| `transformer.blocks.11.ff_out.weight` | 1.04904175e-05 | 3.20040079e-07 | 3.05322007 | 3.4358537e-06 |
| `transformer.ff_out.weight` | 9.53674316e-06 | 2.49874612e-07 | 5.75029272 | 1.6584796e-06 |
