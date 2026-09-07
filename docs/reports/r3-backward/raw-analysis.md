# Independent analysis of the reproduced raw FP32 failure

**Classification: the evidence supports ordinary FP32 evaluation-order rounding error for this raw fixture; no backward-scaling defect was observed.** Original elementwise failures remain recorded failures. This analysis does not independently clear the actual Stage B task-loss/optimizer regime.

The analysis reads two root-run GPU reports and their retained tensor packets. All additional tensor inspection ran in a CPU-only container (`CDRM_DOCKER_GPUS=none`); no GPU work or model/source edits were performed. Exact metadata, source/artifact hashes, all per-tensor base summaries, representative coordinates, scaling and FP64 results are in [raw-analysis.json](raw-analysis.json).

## Reproduction and scope

The fixture is D32, 12 blocks, H4/full MHA, MLP128/GELU, pre-norm ALiBi, affine Q/K normalization, vocabulary 256, T128, batch 2, seed 937, recurrent block 3/rho 1, and four tiled backward MLP chunks. Parameters/computation are FP32, autocast and TF32 are disabled, deterministic algorithms are enabled, and tiled helpers are compiled. Neither whole-model compilation nor CUDA graphs is used. Both new profiles record 12 compiled graphs, no graph breaks/unsupported counters, and failure on recompile-limit fallback.

The automatic-SDPA profile matches the old report's full model configuration, every recorded model/support source hash, and exact token/cotangent hashes. It reproduces all 101 retained gradient summaries—maximum absolute error, RMS, relative L2, error/RMS, and pass/fail—within 1e-12 relative serialization/reduction agreement. This is a comparison of recorded summaries, not a relaxed gradient threshold or a claim of bitwise identity to unavailable old arrays.

The old 61 failing tensors become 62 solely because the new harness also measures `input/block3_input`: it has two failing coordinates. The old `input` maps to `input/embedding_output`; the original failing set otherwise matches exactly. Strict math SDPA changes the tiled numerical evaluation and yields 59 failing tensors. Saved naïve logits and all 102 naïve gradient tensors are bitwise equal between profiles; saved tiled logits and all 102 tiled gradient tensors differ. Saved initial weights and fixture tensors are identical between profiles.

The two runs used harness SHA-256 `6798ddbc38c5241059204870e9733961ea450ed001b0c3c844df56ece2e93f2d`. That initial harness did not yet fail closed on historical config/source identity; this audit verifies those identities externally. Later harness versions add explicit assertions. The old fixture had no saved initialized state or raw gradient arrays, so stronger historical bitwise claims are unsupported.

Root commands represented by the retained arguments (run through the project GPU container launcher):

```bash
python scripts/r3_backward_validate.py --case raw --legacy-report docs/reports/stage-b/backend-num-d32-t128.json --sdpa auto --output-dir .runtime/r3-backward/20260906T210249Z/raw-d32-auto
python scripts/r3_backward_validate.py --case raw --legacy-report docs/reports/stage-b/backend-num-d32-t128.json --sdpa math --fp64-reference --output-dir .runtime/r3-backward/20260906T210249Z/raw-d32-math
```

## Raw and scaled between-backend comparisons

Every comparison covers 100 parameter gradients plus embedding-output and block 3-input gradients. All intended gradients are present, finite, and shape-matched. The legacy coordinate bound remains `2e-6 + 2e-5*abs(reference)`. The independent per-tensor relative-L2 and maximum-error/reference-RMS bounds remain 2e-5. Reductions use CPU FP64 with fixed 1e-12 denominator floors; these floors are inactive in every recorded comparison.

| Profile | Cotangent scale | Failing tensors, legacy | Failing coordinates | Max absolute error | Worst relative L2 | Worst max error / RMS |
|---|---|---:|---:|---:|---:|---:|
| auto | 1 (raw) | 62 | 906 | 0.000213623 | 8.796384e-07 | 4.494699e-06 |
| auto | 1_over_32 | 0 | 0 | 6.67572e-06 | 8.796384e-07 | 4.494699e-06 |
| auto | 32 | 81 | 2071 | 0.006835938 | 8.796384e-07 | 4.494699e-06 |
| auto | ce_norm | 0 | 0 | 1.490116e-07 | 8.134189e-07 | 4.548223e-06 |
| math | 1 (raw) | 59 | 1020 | 0.0002288818 | 1.024271e-06 | 4.869257e-06 |
| math | 1_over_32 | 2 | 2 | 7.152557e-06 | 1.024271e-06 | 4.869257e-06 |
| math | 32 | 79 | 2199 | 0.007324219 | 1.024271e-06 | 4.869257e-06 |
| math | ce_norm | 0 | 0 | 1.937151e-07 | 9.629026e-07 | 5.190181e-06 |

All rows pass both independent normwise bounds. Legacy failures change with scale because the absolute allowance does not scale with the cotangent; the relative error is unchanged for exact power-of-two scaling.

The raw cotangent norm is 255.78836708153727. A separately generated valid MQAR batch at the same weights gives a mean-masked-CE cotangent norm 0.2502059734955985. Their ratio is 0.0009781757331280068. The `ce_norm` experiment scales the **original random direction on its original forward graph** by this ratio. It is not the actual CE gradient direction and must not substitute for the separate task-loss comparison. FP32 rounding gives a scaled norm 0.25020598643918923.

## Independent homogeneity

For alpha 1/32 and 32, `G(alpha*v)/alpha` equals `G(v)` exactly for every parameter and both input gradients in both backends and both SDPA profiles. The maximum error is zero in all four backend/profile combinations at both scales.

The non-power-of-two CE-norm scale introduces rounding before and during backward. After dividing by alpha back to raw scale, both backends retain legacy elementwise failures, while all independent normwise checks pass:

| Profile | Backend | Legacy failing tensors | Max normalized absolute error | Worst relative L2 | Worst max error / RMS |
|---|---|---:|---:|---:|---:|
| auto | naive | 62 | 0.0002797936 | 1.002032e-06 | 7.651684e-06 |
| auto | tiled | 63 | 0.0002297572 | 8.522181e-07 | 7.185155e-06 |
| math | naive | 62 | 0.0002797936 | 1.002032e-06 | 7.651684e-06 |
| math | tiled | 58 | 0.0002558104 | 7.240285e-07 | 7.54402e-06 |

The same effect in naïve autograd is evidence against attributing these normalized elementwise failures specifically to the custom backward. Exact power-of-two homogeneity alone would not prove correctness of a linear operator; the FP64 comparison below is independent corroboration.

## Which coordinates fail

The declared near-zero subset is `abs(reference) <= 0.001*reference_RMS`. Only 90/906 automatic-profile failures and 94/1020 math-profile failures meet that definition. No exact-zero reference coordinate fails. Calling every failure “near zero” would be incorrect.

| Profile | Exact zero | Nonzero ≤0.001 RMS | (0.001,0.01] RMS | (0.01,0.1] RMS | >0.1 RMS |
|---|---:|---:|---:|---:|---:|
| auto | 0 | 90 | 475 | 341 | 0 |
| math | 0 | 94 | 495 | 429 | 2 |

These are counts over parameter and input-gradient tensors, not statistically independent observations. The maximum failing reference magnitude is 0.07645 RMS under automatic dispatch and 0.10782 RMS under math SDPA.

For example, automatic dispatch's largest tolerance violation is block 0 `attn_norm.weight[9]`: reference 0.08255434036, actual 0.08250379562, error 5.05447e-5, allowed 3.65109e-6 (13.84× violation). Its reference tensor RMS is 68.32436. This coordinate is small relative to tensor scale but slightly above the predefined near-zero cutoff. By contrast, the embedding's largest absolute error occurs at `[132,25]` with reference 42.26548 and error 6.48499e-5; that coordinate **passes** its 8.47310e-4 allowance. Maximum absolute error and worst tolerance violation are different diagnostics.

## Naïve FP64 comparison

The math-profile reference is naïve autograd evaluated with the same FP32-representable initial weights and cotangent promoted to FP64. The configuration uses default LayerNorm and no RoPE; the model branches that force FP32 RMS/Tanh normalization or rotary computation are inactive. Tiled is evaluated only in FP32, never labeled FP64.

| Compared with naïve FP64 | Legacy failing gradient tensors | Failing coordinates | Largest absolute error | Worst relative L2 | Worst max error / RMS |
|---|---:|---:|---:|---:|---:|
| naive FP32 | 72 | 1535 | 0.0001811721 | 1.201607e-06 | 4.952121e-06 |
| tiled FP32 | 68 | 1511 | 0.0002445219 | 1.465349e-06 | 6.819697e-06 |

Both FP32 backends pass every independent normwise bound against FP64; both also pass the legacy logit check. Naïve FP32 itself fails more gradient-tensor legacy checks than tiled FP32. Thus naïve FP32 is not an exact coordinate-level oracle at the original absolute tolerance. The two FP32 implementations sometimes land on opposite sides of the FP64 value.

Representative saved coordinates (the JSON records full precision, reference RMS and signed errors):

| Tensor coordinate | Naïve FP64 | Naïve FP32 | Tiled automatic FP32 | Tiled math FP32 |
|---|---:|---:|---:|---:|
| `transformer.blocks.0.attn_norm.weight[9]` | 0.082512869603 | 0.0825543403625 | 0.0825037956238 | 0.0824794769287 |
| `transformer.wte.weight[174,2]` | 0.0201763310026 | 0.0201482772827 | 0.0201263427734 | 0.0201396942139 |
| `transformer.wte.weight[253,3]` | 0.0139059460927 | 0.0139293670654 | 0.0139102935791 | 0.0138874053955 |
| `transformer.blocks.0.attn_out.weight[2,23]` | -0.00841602047973 | -0.00842752214521 | -0.0084026074037 | -0.00842456892133 |
| `transformer.blocks.0.ff_out.weight[9,47]` | 252.86736721 | 252.867401123 | 252.867324829 | 252.867172241 |
| `input/block3_input[1,67,5]` | -7.08816233895 | -7.0881652832 | -7.08816146851 | -7.08815908432 |

At block 0 `attn_norm.weight[9]`, errors against FP64 are +4.14708e-5 for naïve FP32, -9.07398e-6 for automatic tiled FP32, and -3.33927e-5 for math tiled FP32. The observed disagreement is compatible with rounding in both execution orders, not a demonstrably exact naïve coordinate plus an erroneous tiled coordinate. Other coordinates favor naïve; no claim that tiled is uniformly more accurate is made.

## Worst tensors at raw scale

The following rows rank each profile by maximum absolute gradient error. Complete per-tensor base summaries are retained in the JSON.

| Profile | Tensor | Reference L2 | Error L2 | Max absolute error | Relative L2 | Max error / RMS |
|---|---|---:|---:|---:|---:|---:|
| auto | `transformer.blocks.0.ff_out.weight` | 4020.1964 | 0.0016914079 | 0.00021362305 | 4.2072767e-07 | 3.4007978e-06 |
| auto | `transformer.blocks.3.ff_out.weight` | 2937.1171 | 0.0013639832 | 0.00018310547 | 4.6439524e-07 | 3.9898818e-06 |
| auto | `transformer.blocks.10.attn_out.weight` | 1303.619 | 0.00043602507 | 0.00018310547 | 3.3447278e-07 | 4.494699e-06 |
| auto | `transformer.blocks.0.attn_out.weight` | 2216.5273 | 0.0010037862 | 0.00015258789 | 4.5286437e-07 | 2.202911e-06 |
| auto | `transformer.blocks.1.attn_out.weight` | 1914.3262 | 0.00070526645 | 0.00012207031 | 3.6841499e-07 | 2.0405352e-06 |
| auto | `transformer.blocks.1.ff_out.weight` | 3098.2153 | 0.0011472783 | 0.00012207031 | 3.7030294e-07 | 2.521613e-06 |
| math | `transformer.blocks.0.ff_out.weight` | 4020.1964 | 0.0019132069 | 0.00022888184 | 4.7589887e-07 | 3.6437119e-06 |
| math | `transformer.blocks.10.attn_out.weight` | 1303.619 | 0.00046733782 | 0.00019836426 | 3.5849265e-07 | 4.8692573e-06 |
| math | `transformer.blocks.2.ff_out.weight` | 2852.6724 | 0.0013244415 | 0.00018310547 | 4.6428098e-07 | 4.1079901e-06 |
| math | `transformer.blocks.3.ff_out.weight` | 2937.1171 | 0.0014861294 | 0.00015258789 | 5.0598235e-07 | 3.3249015e-06 |
| math | `transformer.blocks.1.attn_out.weight` | 1914.3262 | 0.00085225171 | 0.0001373291 | 4.451967e-07 | 2.2956021e-06 |
| math | `transformer.blocks.1.ff_out.weight` | 3098.2153 | 0.0013511448 | 0.0001373291 | 4.3610422e-07 | 2.8368146e-06 |

For all parameter tensors combined (excluding input-gradient tensors), relative L2 disagreement is 4.06821e-7 under automatic dispatch and 4.45098e-7 under math SDPA. Per-tensor checks remain primary; the global metric does not hide a failed tensor.

## Decision boundary

The exact retained-summary reproduction, power-of-two homogeneity, stable tensor-relative errors across scales, untouched structural zeros, and comparable naïve/tiled errors against FP64 jointly support ordinary FP32 roundoff as the explanation for this raw fixture. They do not prove that each arithmetic path or every trained configuration is correct.

No legacy threshold is loosened or reclassified as a pass. BF16, accumulation, long-sequence extensions, and distributed/CDRM paths remain separate questions. Clearing the intended Stage B FP32/no-accumulation regime additionally requires the root investigation's actual masked-CE, persistent-write, retained-checkpoint and optimizer-update evidence.
