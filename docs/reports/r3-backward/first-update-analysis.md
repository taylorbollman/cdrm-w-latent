# First Adam step: retained numerical evidence

This CPU-only analysis uses stored FP32 gradients and optimizer results. It runs no model, changes no acceptance thresholds, and incorporates explicitly supplied retained naive FP64 gradient references. Those are higher-precision references, not exact real-arithmetic gradients.

At the first Adam step with zero initial moments and zero weight decay, ideal real arithmetic gives `delta = -lr * g / (abs(g) + eps)`, where `g` is the clipped gradient. Both beta bias corrections cancel. Its derivative is `-lr * eps / (abs(g) + eps)^2`; gradients near epsilon can therefore produce appreciable update differences from tiny absolute gradient differences. All ideal formulas below run in FP64 on the retained FP32 gradient values.

Recorded deltas were computed as the **FP64 difference of the after and before FP32 weights**. This subtraction is exact for these nearby stored values; the update was already rounded when stored in the FP32 parameter. An unrounded CUDA Adam increment was not recorded. The FP64 formula using recorded moments separates moment-storage effects; rounding the gradient-based ideal result into the initial FP32 parameter estimates final storage effects.

The unchanged practical update screen requires each tensor's relative update L2 error ≤ 0.001 **and** maximum absolute update error ≤ 0.01 × LR. The separately reported moment/gradient scale diagnostic requires relative L2 and maximum error/reference RMS both ≤ 2e-5. Global summaries below are descriptive; they do not replace tensor-level screens.

| Case | Shape | Recorded flagged update tensors | Ideal-gradient flagged tensors | Coordinates above 1% LR / all | Recorded global relative update L2 | Max recorded update error/LR | Moment scale flags |
|---|---|---:|---:|---:|---:|---:|---:|
| tiny-ce | [2, 32] | 1 | 1 | 1 / 214560 | 0.000176632 | 0.0752509 | 2 |
| init-b2 | [2, 128] | 10 | 10 | 14 / 9974016 | 2.81757e-05 | 0.0268221 | 27 |
| init-b64 | [64, 128] | 14 | 14 | 17 / 9974016 | 3.28395e-05 | 0.0268221 | 26 |

## tiny-ce

The largest recorded update discrepancy is `transformer.blocks.0.att_proj.weight` at [28, 3]. Its tensor relative update L2 error is 0.00135773518; the coordinate differs by 7.52508640289e-07 (0.0752509 × LR, 18.985% of this coordinate's naive update).

| Quantity | Naive | Tiled |
|---|---:|---:|
| raw_gradient | 1.0657382176759711e-07 | 7.6789362424278806e-08 |
| clipped_gradient | 6.5649325975414285e-09 | 4.7302139982718927e-09 |
| clipped_gradient_over_epsilon | 0.65649325975414285 | 0.47302139982718927 |
| before_weight | 0.09203704446554184 | 0.09203704446554184 |
| after_weight | 0.092033080756664276 | 0.092033833265304565 |
| ideal_from_gradients | -3.9631508060079876e-06 | -3.2112323682648663e-06 |
| from_recorded_moments | -3.9631507543737595e-06 | -3.2112323473554366e-06 |
| rounded_ideal | -3.9637088775634766e-06 | -3.2112002372741699e-06 |
| recorded | -3.9637088775634766e-06 | -3.2112002372741699e-06 |
| recorded_minus_ideal | -5.5807155548896168e-10 | 3.2130990696383957e-11 |
| recorded_minus_rounded_ideal | 0 | 0 |
| weight_fp32_spacing_up | 7.4505805969238281e-09 | 7.4505805969238281e-09 |

The ideal FP64 Adam mapping of the two stored clipped gradients differs by 7.51918437743e-07. The recorded difference is 7.52508640289e-07; its signed residual after subtracting the ideal-gradient difference is 5.90202546185e-10. Thus the ideal formula tests whether the local update difference persists without FP32 optimizer arithmetic or parameter storage rounding.

At this same coordinate the retained naive FP64 reference has raw gradient 8.3185512636979951e-08, clipped gradient 5.1242164546634892e-09, and ideal first update -3.3880872242366373e-06. The tiled raw gradient and tiled ideal update are closer here. Clipping for this reference uses its FP64 global parameter-gradient norm; a common-coefficient comparison is also retained.

| Error relative to FP64 reference at this coordinate | Naive | Tiled |
|---|---:|---:|
| raw_gradient_error | 2.33883091306e-08 | -6.3961502127e-09 |
| clipped_gradient_error | 1.44071614288e-09 | -3.94002456392e-10 |
| ideal_update_error | -5.75063581771e-07 | 1.76854855972e-07 |
| recorded_update_error | -5.75621653327e-07 | 1.76886986962e-07 |
| ideal_update_error_with_common_clip_coefficient | -5.75063743494e-07 | 1.76854694249e-07 |

All previously flagged update tensors' worst coordinates receive the same reference comparison in the JSON. Coordinate-specific proximity does not identify one implementation as uniformly more accurate.

The following uses the same fixed diagnostic limits against the higher-precision reference; these additional comparisons do not replace the original naive-versus-tiled screen.

| FP32 arm ideal update versus FP64-gradient ideal update | Global relative L2 | Max error/LR | Coordinates above 1% LR | Flagged tensors |
|---|---:|---:|---:|---:|
| naive | 0.000134803 | 0.0575064 | 1 | 1 |
| tiled | 4.25196e-05 | 0.0176855 | 1 | 1 |

Flagged moments retain the original diagnostic result:

| Tensor/moment | Relative L2 | Max error/reference RMS | Worst-coordinate relative error | Coordinate magnitude/reference RMS |
|---|---:|---:|---:|---:|
| `transformer.wte.weight/exp_avg_sq` | 4.97704e-07 | 2.80181e-05 | 1.15114e-06 | 24.3395 |
| `transformer.blocks.3.kv_proj.weight/exp_avg_sq` | 6.48413e-07 | 2.04634e-05 | 8.30613e-07 | 24.6365 |

For the second moment, ideal `v = (1-beta2)*g^2`: a coordinate gradient perturbation `dg` changes it by `(1-beta2)*(2*g*dg + dg^2)`. The maximum-error/reference-RMS flag can arise at a large moment coordinate even when its own relative error and the tensor relative L2 error are small. Exact gradients, moments, ideal differences and arithmetic residuals for every flagged coordinate are retained in the JSON.

The second-moment scale flag count is 2 in the recorded FP32 moments and 2 when ideal FP64 moments are computed from those same clipped gradients. Squaring and coordinate concentration explain the amplification without removing the diagnostic failures.

Source: [`report.json`](../../../.runtime/r3-backward/20260906T210249Z/tiny-ce/report.json) and [`ce-and-update-tensors.pt`](../../../.runtime/r3-backward/20260906T210249Z/tiny-ce/ce-and-update-tensors.pt); SHA-256 values are in the JSON.

## init-b2

The largest recorded update discrepancy is `transformer.blocks.0.attn_out.weight` at [217, 238]. Its tensor relative update L2 error is 0.000108824266; the coordinate differs by 2.68220901489e-07 (0.0268221 × LR, 16.3636% of this coordinate's naive update).

| Quantity | Naive | Tiled |
|---|---:|---:|
| raw_gradient | -5.029141902923584e-08 | -6.0535967350006104e-08 |
| clipped_gradient | -1.9598425105016304e-09 | -2.3590696063280348e-09 |
| clipped_gradient_over_epsilon | -0.19598425105016304 | -0.23590696063280348 |
| before_weight | 0.057523541152477264 | 0.057523541152477264 |
| after_weight | 0.057525180280208588 | 0.057525448501110077 |
| ideal_from_gradients | 1.6386858846851397e-06 | 1.9087760498736532e-06 |
| from_recorded_moments | 1.6386859308852838e-06 | 1.9087760422728953e-06 |
| rounded_ideal | 1.6391277313232422e-06 | 1.9073486328125e-06 |
| recorded | 1.6391277313232422e-06 | 1.9073486328125e-06 |
| recorded_minus_ideal | 4.4184663810252543e-10 | -1.4274170611531656e-09 |
| recorded_minus_rounded_ideal | 0 | 0 |
| weight_fp32_spacing_up | 3.7252902984619141e-09 | 3.7252902984619141e-09 |

The ideal FP64 Adam mapping of the two stored clipped gradients differs by 2.70090165189e-07. The recorded difference is 2.68220901489e-07; its signed residual after subtracting the ideal-gradient difference is -1.86926369926e-09. Thus the ideal formula tests whether the local update difference persists without FP32 optimizer arithmetic or parameter storage rounding.

Flagged moments retain the original diagnostic result:

| Tensor/moment | Relative L2 | Max error/reference RMS | Worst-coordinate relative error | Coordinate magnitude/reference RMS |
|---|---:|---:|---:|---:|
| `transformer.wte.weight/exp_avg_sq` | 7.18793e-07 | 5.59376e-05 | 7.47876e-07 | 74.7954 |
| `transformer.blocks.0.att_proj.weight/exp_avg_sq` | 7.51567e-07 | 2.08448e-05 | 1.22105e-06 | 17.0712 |
| `transformer.blocks.1.ff_proj.weight/exp_avg_sq` | 5.80817e-07 | 3.16949e-05 | 1.73262e-06 | 18.2931 |
| `transformer.blocks.2.ff_out.weight/exp_avg_sq` | 4.53092e-07 | 2.21252e-05 | 9.53526e-07 | 23.2036 |
| `transformer.blocks.2.att_proj.weight/exp_avg_sq` | 5.11767e-07 | 2.1324e-05 | 8.38886e-07 | 25.4195 |
| `transformer.blocks.3.ff_out.weight/exp_avg_sq` | 7.16706e-07 | 2.87906e-05 | 9.09032e-07 | 31.6717 |
| `transformer.blocks.3.ff_proj.weight/exp_avg_sq` | 7.47143e-07 | 3.59834e-05 | 1.6563e-06 | 21.7252 |
| `transformer.blocks.3.q_proj.weight/exp_avg_sq` | 8.60202e-07 | 2.1975e-05 | 2.46413e-06 | 8.91799 |
| `transformer.blocks.3.kv_proj.weight/exp_avg_sq` | 4.30763e-07 | 2.36203e-05 | 9.25446e-07 | 25.5232 |
| `transformer.blocks.4.ff_out.weight/exp_avg_sq` | 4.72183e-07 | 2.25038e-05 | 7.16965e-07 | 31.3876 |
| `transformer.blocks.4.att_proj.weight/exp_avg_sq` | 4.41445e-07 | 3.63917e-05 | 5.38261e-07 | 67.6097 |
| `transformer.blocks.4.ff_proj.weight/exp_avg_sq` | 4.61608e-07 | 2.98464e-05 | 5.83442e-07 | 51.1559 |
| `transformer.blocks.5.att_proj.weight/exp_avg_sq` | 4.73186e-07 | 2.12797e-05 | 1.21572e-06 | 17.5038 |
| `transformer.blocks.5.ff_proj.weight/exp_avg_sq` | 5.28781e-07 | 3.33253e-05 | 1.12728e-06 | 29.5626 |
| `transformer.blocks.6.ff_out.weight/exp_avg_sq` | 4.78368e-07 | 2.5626e-05 | 1.20602e-06 | 21.2484 |
| `transformer.blocks.6.ff_proj.weight/exp_avg_sq` | 5.09829e-07 | 2.31274e-05 | 1.28318e-06 | 18.0236 |
| `transformer.blocks.7.att_proj.weight/exp_avg_sq` | 4.16735e-07 | 4.17228e-05 | 5.50288e-07 | 75.8199 |
| `transformer.blocks.7.ff_proj.weight/exp_avg_sq` | 5.01939e-07 | 2.98282e-05 | 7.65042e-07 | 38.9889 |
| `transformer.blocks.8.ff_out.weight/exp_avg_sq` | 4.78488e-07 | 2.14836e-05 | 7.59999e-07 | 28.2679 |
| `transformer.blocks.8.att_proj.weight/exp_avg_sq` | 4.00098e-07 | 2.28202e-05 | 9.66655e-07 | 23.6074 |
| `transformer.blocks.8.ff_proj.weight/exp_avg_sq` | 5.2438e-07 | 3.18635e-05 | 1.47599e-06 | 21.5878 |
| `transformer.blocks.9.att_proj.weight/exp_avg_sq` | 3.83872e-07 | 2.00573e-05 | 7.12893e-07 | 28.1351 |
| `transformer.blocks.9.ff_proj.weight/exp_avg_sq` | 5.11556e-07 | 3.00696e-05 | 6.58998e-07 | 45.6292 |
| `transformer.blocks.10.ff_out.weight/exp_avg_sq` | 4.92771e-07 | 2.18855e-05 | 1.07851e-06 | 20.2923 |
| `transformer.blocks.10.ff_proj.weight/exp_avg_sq` | 4.8316e-07 | 2.50068e-05 | 5.8138e-07 | 43.0128 |
| `transformer.blocks.11.ff_out.weight/exp_avg_sq` | 4.52484e-07 | 2.03746e-05 | 6.95942e-07 | 29.2763 |
| `transformer.ff_out.weight/exp_avg_sq` | 3.58584e-07 | 2.31922e-05 | 1.12579e-06 | 20.6009 |

For the second moment, ideal `v = (1-beta2)*g^2`: a coordinate gradient perturbation `dg` changes it by `(1-beta2)*(2*g*dg + dg^2)`. The maximum-error/reference-RMS flag can arise at a large moment coordinate even when its own relative error and the tensor relative L2 error are small. Exact gradients, moments, ideal differences and arithmetic residuals for every flagged coordinate are retained in the JSON.

The second-moment scale flag count is 27 in the recorded FP32 moments and 27 when ideal FP64 moments are computed from those same clipped gradients. Squaring and coordinate concentration explain the amplification without removing the diagnostic failures.

Source: [`report.json`](../../../.runtime/r3-backward/20260906T210249Z/init-b2/report.json) and [`ce-and-update-tensors.pt`](../../../.runtime/r3-backward/20260906T210249Z/init-b2/ce-and-update-tensors.pt); SHA-256 values are in the JSON.

## init-b64

The largest recorded update discrepancy is `transformer.wte.weight` at [716, 233]. Its tensor relative update L2 error is 7.03380055e-05; the coordinate differs by 2.68220901489e-07 (0.0268221 × LR, 12.2034% of this coordinate's naive update).

| Quantity | Naive | Tiled |
|---|---:|---:|
| raw_gradient | 1.4639226719737053e-08 | 1.2456439435482025e-08 |
| clipped_gradient | 2.8109987670887904e-09 | 2.3918635960740176e-09 |
| clipped_gradient_over_epsilon | 0.28109987670887904 | 0.23918635960740176 |
| before_weight | 0.078234151005744934 | 0.078234151005744934 |
| after_weight | 0.078231953084468842 | 0.078232221305370331 |
| ideal_from_gradients | -2.1942073511943443e-06 | -1.9301887706638463e-06 |
| from_recorded_moments | -2.1942074788360307e-06 | -1.9301888079456152e-06 |
| rounded_ideal | -2.1979212760925293e-06 | -1.9297003746032715e-06 |
| recorded | -2.1979212760925293e-06 | -1.9297003746032715e-06 |
| recorded_minus_ideal | -3.7139248981850204e-09 | 4.8839606057479821e-10 |
| recorded_minus_rounded_ideal | 0 | 0 |
| weight_fp32_spacing_up | 7.4505805969238281e-09 | 7.4505805969238281e-09 |

The ideal FP64 Adam mapping of the two stored clipped gradients differs by 2.6401858053e-07. The recorded difference is 2.68220901489e-07; its signed residual after subtracting the ideal-gradient difference is 4.20232095876e-09. Thus the ideal formula tests whether the local update difference persists without FP32 optimizer arithmetic or parameter storage rounding.

At this same coordinate the retained naive FP64 reference has raw gradient 1.5349349077192504e-08, clipped gradient 2.9473551860644297e-09, and ideal first update -2.2764148690666521e-06. The naive raw gradient and naive ideal update are closer here. Clipping for this reference uses its FP64 global parameter-gradient norm; a common-coefficient comparison is also retained.

| Error relative to FP64 reference at this coordinate | Naive | Tiled |
|---|---:|---:|
| raw_gradient_error | -7.10122357455e-10 | -2.89290964171e-09 |
| clipped_gradient_error | -1.36356418976e-10 | -5.5549158999e-10 |
| ideal_update_error | 8.22075178723e-08 | 3.46226098403e-07 |
| recorded_update_error | 7.84935929741e-08 | 3.46714494463e-07 |
| ideal_update_error_with_common_clip_coefficient | 8.22075466263e-08 | 3.46225990715e-07 |

All previously flagged update tensors' worst coordinates receive the same reference comparison in the JSON. Coordinate-specific proximity does not identify one implementation as uniformly more accurate.

The following uses the same fixed diagnostic limits against the higher-precision reference; these additional comparisons do not replace the original naive-versus-tiled screen.

| FP32 arm ideal update versus FP64-gradient ideal update | Global relative L2 | Max error/LR | Coordinates above 1% LR | Flagged tensors |
|---|---:|---:|---:|---:|
| naive | 6.88095e-05 | 0.0801961 | 64 | 28 |
| tiled | 6.67227e-05 | 0.0711001 | 67 | 27 |

Flagged moments retain the original diagnostic result:

| Tensor/moment | Relative L2 | Max error/reference RMS | Worst-coordinate relative error | Coordinate magnitude/reference RMS |
|---|---:|---:|---:|---:|
| `transformer.wte.weight/exp_avg_sq` | 1.17513e-06 | 3.61951e-05 | 2.83019e-06 | 12.789 |
| `transformer.blocks.0.att_proj.weight/exp_avg_sq` | 1.08111e-06 | 2.80538e-05 | 2.65021e-06 | 10.5855 |
| `transformer.blocks.1.ff_proj.weight/exp_avg_sq` | 6.71043e-07 | 2.5382e-05 | 1.0795e-06 | 23.5128 |
| `transformer.blocks.3.ff_out.weight/exp_avg_sq` | 6.95188e-07 | 2.91188e-05 | 9.5907e-07 | 30.3616 |
| `transformer.blocks.3.ff_proj.weight/exp_avg_sq` | 7.87697e-07 | 2.83739e-05 | 1.45579e-06 | 19.4904 |
| `transformer.blocks.3.q_proj.weight/exp_avg_sq` | 1.00077e-06 | 2.06726e-05 | 2.23023e-06 | 9.26929 |
| `transformer.blocks.3.kv_proj.weight/exp_avg_sq` | 4.80797e-07 | 2.10022e-05 | 1.07943e-06 | 19.4568 |
| `transformer.blocks.4.att_proj.weight/exp_avg_sq` | 6.20637e-07 | 4.32095e-05 | 1.00333e-06 | 43.0659 |
| `transformer.blocks.5.ff_out.weight/exp_avg_sq` | 5.04428e-07 | 2.17484e-05 | 7.10435e-07 | 30.6128 |
| `transformer.blocks.5.att_proj.weight/exp_avg_sq` | 5.46195e-07 | 3.1863e-05 | 8.9952e-07 | 35.4222 |
| `transformer.blocks.5.ff_proj.weight/exp_avg_sq` | 5.76922e-07 | 2.27573e-05 | 7.25722e-07 | 31.3581 |
| `transformer.blocks.6.ff_out.weight/exp_avg_sq` | 4.88508e-07 | 2.11704e-05 | 4.94126e-07 | 42.8441 |
| `transformer.blocks.6.att_proj.weight/exp_avg_sq` | 4.51665e-07 | 2.98442e-05 | 6.75502e-07 | 44.1807 |
| `transformer.blocks.6.ff_proj.weight/exp_avg_sq` | 5.81853e-07 | 2.18721e-05 | 1.0667e-06 | 20.5045 |
| `transformer.blocks.7.ff_out.weight/exp_avg_sq` | 5.01626e-07 | 2.24815e-05 | 6.17758e-07 | 36.3921 |
| `transformer.blocks.7.att_proj.weight/exp_avg_sq` | 4.82469e-07 | 2.81241e-05 | 9.18789e-07 | 30.6099 |
| `transformer.blocks.7.ff_proj.weight/exp_avg_sq` | 5.35057e-07 | 2.75979e-05 | 5.55489e-07 | 49.6821 |
| `transformer.blocks.8.att_proj.weight/exp_avg_sq` | 4.87674e-07 | 2.83524e-05 | 9.48246e-07 | 29.8999 |
| `transformer.blocks.8.ff_proj.weight/exp_avg_sq` | 5.38797e-07 | 2.03921e-05 | 7.89624e-07 | 25.825 |
| `transformer.blocks.9.att_proj.weight/exp_avg_sq` | 4.63777e-07 | 2.45974e-05 | 8.86305e-07 | 27.7527 |
| `transformer.blocks.10.ff_out.weight/exp_avg_sq` | 4.65472e-07 | 2.11993e-05 | 7.18569e-07 | 29.502 |
| `transformer.blocks.10.ff_proj.weight/exp_avg_sq` | 5.19033e-07 | 2.36458e-05 | 1.16344e-06 | 20.324 |
| `transformer.blocks.11.ff_out.weight/exp_avg_sq` | 4.54623e-07 | 2.41349e-05 | 9.38941e-07 | 25.7044 |
| `transformer.blocks.11.att_proj.weight/exp_avg_sq` | 4.45567e-07 | 2.65153e-05 | 8.72267e-07 | 30.3981 |
| `transformer.blocks.11.ff_proj.weight/exp_avg_sq` | 4.84515e-07 | 2.44731e-05 | 1.18955e-06 | 20.5733 |
| `transformer.ff_out.weight/exp_avg_sq` | 3.89588e-07 | 2.11407e-05 | 5.25496e-07 | 40.2301 |

For the second moment, ideal `v = (1-beta2)*g^2`: a coordinate gradient perturbation `dg` changes it by `(1-beta2)*(2*g*dg + dg^2)`. The maximum-error/reference-RMS flag can arise at a large moment coordinate even when its own relative error and the tensor relative L2 error are small. Exact gradients, moments, ideal differences and arithmetic residuals for every flagged coordinate are retained in the JSON.

The second-moment scale flag count is 26 in the recorded FP32 moments and 24 when ideal FP64 moments are computed from those same clipped gradients. Squaring and coordinate concentration explain the amplification without removing the diagnostic failures.

Source: [`report.json`](../../../.runtime/r3-backward/20260906T210249Z/init-b64/report.json) and [`ce-and-update-tensors.pt`](../../../.runtime/r3-backward/20260906T210249Z/init-b64/ce-and-update-tensors.pt); SHA-256 values are in the JSON.

The flagged first-step updates are real differences in the stored parameter changes under the predeclared screen. A small post-step weight error or a passed raw-gradient comparison does not erase them. Their first-step epsilon sensitivity is an explanation of how the discrepancies propagate, not a waiver or evidence that either backward implementation is wrong. The retained higher-precision references give additional coordinate and tensor evidence for assessing floating-point order sensitivity; this one-step analysis does not establish a material effect on a complete training trajectory.
