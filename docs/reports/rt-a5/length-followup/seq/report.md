# A5 SEQ: length-generalization boundary

Two blocks, width 512, 6,357,504 parameters; full FP32, ALiBi, seed 1234. The primary endpoint remains 10,000 updates.

The curves below reuse the same first 102,400 frozen OOD development words at 5,000 and 10,000 updates. They are prefixes of historical length-36 model outputs, not newly inferred independent datasets at every length.

**E(t)** requires every state through t to be correct. **A(t)** requires only the state at t. **M(t)** averages token correctness through t. At the end of a word, E is whole-word exactness, A is final-state accuracy and M is mean token accuracy. Only A has a flat 1/60 state-guessing reference.

| Updates | Last E ≥95% | Last E ≥50% | Last E ≥10% | Last E ≥1% | First zero-success length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 3 | 4 | 4 | 5 | 8 |
| 10,000 | 3 | 5 | 5 | 6 | 10 |

Thresholds are inclusive and are measured on this finite sample. Observed zeros come from actual zero integer success counts, not rounded percentages; they do not establish zero population success. Intervals below are pointwise Wilson 95% intervals across sampled words at a fixed checkpoint; they do not measure variation across training seeds. No mean-token interval assumes independent positions.

![Full length curves](length-full.png)

![Boundary detail](length-zoom.png)

The new forward-only truncation check uses **1,024 words**, distinct from the **102,400-word historical curves** above. There were **1 changed predicted class(es)** across the prefix comparisons (T12: 0, T13: 1). The saved-output assessment verified that each word/position retained the same correct-or-wrong indicator: changed predictions remained wrong, and every E/A/M count was identical. Weights were unchanged, with no gradients or optimizer updates. [Original prefix-check snapshot](inputs/prefix-check.json).

**Qualification:** the original elementwise FP32 logit screen failed and remains recorded as failed. The separate saved-output assessment supports using the exactly matching accuracy metrics for this bounded length check; it does not retroactively make that strict screen pass.

This post-run assessment accepts only the observed E/A/M length metrics: every per-word, per-position correct/incorrect indicator and every integer accuracy count agrees. Wrong-class predictions may differ; all such changes and their margins are reported. Original strict logit-screen and prediction-agreement failures remain failures. Probability/loss changes and margin certificates are descriptive; no numerical tolerance was changed. No claim about all inputs, backward gradients or mixed precision.

[Separate assessment snapshot](inputs/prefix-assessment.json)

## 5,000 updates: all lengths

| t | All-correct words / 102,400 | E(t), % [95% interval] | State-correct words / 102,400 | A(t), % [95% interval] | M(t), % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 102,400 | 100.0000 [99.9962, 100.0000] | 102,400 | 100.0000 [99.9962, 100.0000] | 100.0000 |
| 2 | 100,708 | 98.3477 [98.2677, 98.4239] | 100,708 | 98.3477 [98.2677, 98.4239] | 99.1738 |
| 3 | 99,272 | 96.9453 [96.8381, 97.0490] | 99,355 | 97.0264 [96.9206, 97.1287] | 98.4580 |
| 4 | 71,894 | 70.2090 [69.9281, 70.4883] | 72,619 | 70.9170 [70.6381, 71.1944] | 91.5728 |
| 5 | 7,032 | 6.8672 [6.7139, 7.0237] | 8,059 | 7.8701 [7.7068, 8.0366] | 74.8322 |
| 6 | 467 | 0.4561 [0.4166, 0.4992] | 2,265 | 2.2119 [2.1236, 2.3038] | 62.7288 |
| 7 | 26 | 0.0254 [0.0173, 0.0372] | 1,769 | 1.7275 [1.6495, 1.8092] | 54.0144 |
| 8 | 0 | 0.0000 [0.0000, 0.0038] | 1,716 | 1.6758 [1.5990, 1.7562] | 47.4720 |
| 9 | 0 | 0.0000 [0.0000, 0.0038] | 1,611 | 1.5732 [1.4988, 1.6513] | 42.3722 |
| 10 | 0 | 0.0000 [0.0000, 0.0038] | 1,717 | 1.6768 [1.5999, 1.7572] | 38.3026 |
| 11 | 0 | 0.0000 [0.0000, 0.0038] | 1,663 | 1.6240 [1.5484, 1.7033] | 34.9682 |
| 12 | 0 | 0.0000 [0.0000, 0.0038] | 1,679 | 1.6396 [1.5637, 1.7193] | 32.1908 |
| 13 | 0 | 0.0000 [0.0000, 0.0038] | 1,722 | 1.6816 [1.6047, 1.7622] | 29.8440 |
| 14 | 0 | 0.0000 [0.0000, 0.0038] | 1,738 | 1.6973 [1.6199, 1.7782] | 27.8335 |
| 15 | 0 | 0.0000 [0.0000, 0.0038] | 1,689 | 1.6494 [1.5732, 1.7293] | 26.0879 |
| 16 | 0 | 0.0000 [0.0000, 0.0038] | 1,753 | 1.7119 [1.6343, 1.7932] | 24.5644 |
| 17 | 0 | 0.0000 [0.0000, 0.0038] | 1,759 | 1.7178 [1.6400, 1.7992] | 23.2205 |
| 18 | 0 | 0.0000 [0.0000, 0.0038] | 1,654 | 1.6152 [1.5398, 1.6943] | 22.0202 |
| 19 | 0 | 0.0000 [0.0000, 0.0038] | 1,650 | 1.6113 [1.5360, 1.6903] | 20.9460 |
| 20 | 0 | 0.0000 [0.0000, 0.0038] | 1,754 | 1.7129 [1.6352, 1.7942] | 19.9844 |
| 21 | 0 | 0.0000 [0.0000, 0.0038] | 1,697 | 1.6572 [1.5808, 1.7373] | 19.1117 |
| 22 | 0 | 0.0000 [0.0000, 0.0038] | 1,789 | 1.7471 [1.6686, 1.8291] | 18.3224 |
| 23 | 0 | 0.0000 [0.0000, 0.0038] | 1,674 | 1.6348 [1.5589, 1.7143] | 17.5968 |
| 24 | 0 | 0.0000 [0.0000, 0.0038] | 1,773 | 1.7314 [1.6533, 1.8132] | 16.9358 |
| 25 | 0 | 0.0000 [0.0000, 0.0038] | 1,736 | 1.6953 [1.6180, 1.7762] | 16.3261 |
| 26 | 0 | 0.0000 [0.0000, 0.0038] | 1,703 | 1.6631 [1.5866, 1.7432] | 15.7622 |
| 27 | 0 | 0.0000 [0.0000, 0.0038] | 1,780 | 1.7383 [1.6600, 1.8202] | 15.2428 |
| 28 | 0 | 0.0000 [0.0000, 0.0038] | 1,631 | 1.5928 [1.5179, 1.6713] | 14.7553 |
| 29 | 0 | 0.0000 [0.0000, 0.0038] | 1,719 | 1.6787 [1.6018, 1.7592] | 14.3044 |
| 30 | 0 | 0.0000 [0.0000, 0.0038] | 1,687 | 1.6475 [1.5713, 1.7273] | 13.8825 |
| 31 | 0 | 0.0000 [0.0000, 0.0038] | 1,788 | 1.7461 [1.6677, 1.8281] | 13.4910 |
| 32 | 0 | 0.0000 [0.0000, 0.0038] | 1,773 | 1.7314 [1.6533, 1.8132] | 13.1235 |
| 33 | 0 | 0.0000 [0.0000, 0.0038] | 1,643 | 1.6045 [1.5293, 1.6833] | 12.7744 |
| 34 | 0 | 0.0000 [0.0000, 0.0038] | 1,739 | 1.6982 [1.6209, 1.7792] | 12.4486 |
| 35 | 0 | 0.0000 [0.0000, 0.0038] | 1,755 | 1.7139 [1.6362, 1.7952] | 12.1419 |
| 36 | 0 | 0.0000 [0.0000, 0.0038] | 1,726 | 1.6855 [1.6085, 1.7662] | 11.8515 |

## 10,000 updates: all lengths

| t | All-correct words / 102,400 | E(t), % [95% interval] | State-correct words / 102,400 | A(t), % [95% interval] | M(t), % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 102,400 | 100.0000 [99.9962, 100.0000] | 102,400 | 100.0000 [99.9962, 100.0000] | 100.0000 |
| 2 | 100,663 | 98.3037 [98.2228, 98.3810] | 100,663 | 98.3037 [98.2228, 98.3810] | 99.1519 |
| 3 | 100,280 | 97.9297 [97.8407, 98.0151] | 100,875 | 98.5107 [98.4347, 98.5831] | 98.9382 |
| 4 | 93,832 | 91.6328 [91.4617, 91.8008] | 94,226 | 92.0176 [91.8500, 92.1820] | 97.2080 |
| 5 | 63,378 | 61.8926 [61.5947, 62.1896] | 67,033 | 65.4619 [65.1701, 65.7526] | 90.8588 |
| 6 | 8,812 | 8.6055 [8.4352, 8.7788] | 10,818 | 10.5645 [10.3777, 10.7542] | 77.4764 |
| 7 | 891 | 0.8701 [0.8150, 0.9289] | 3,203 | 3.1279 [3.0231, 3.2363] | 66.8552 |
| 8 | 75 | 0.0732 [0.0584, 0.0918] | 1,988 | 1.9414 [1.8587, 2.0277] | 58.7410 |
| 9 | 12 | 0.0117 [0.0067, 0.0205] | 1,789 | 1.7471 [1.6686, 1.8291] | 52.4083 |
| 10 | 0 | 0.0000 [0.0000, 0.0038] | 1,777 | 1.7354 [1.6572, 1.8172] | 47.3410 |
| 11 | 0 | 0.0000 [0.0000, 0.0038] | 1,645 | 1.6064 [1.5312, 1.6853] | 43.1833 |
| 12 | 0 | 0.0000 [0.0000, 0.0038] | 1,735 | 1.6943 [1.6171, 1.7752] | 39.7259 |
| 13 | 0 | 0.0000 [0.0000, 0.0038] | 1,716 | 1.6758 [1.5990, 1.7562] | 36.7990 |
| 14 | 0 | 0.0000 [0.0000, 0.0038] | 1,667 | 1.6279 [1.5522, 1.7073] | 34.2868 |
| 15 | 0 | 0.0000 [0.0000, 0.0038] | 1,695 | 1.6553 [1.5789, 1.7353] | 32.1113 |
| 16 | 0 | 0.0000 [0.0000, 0.0038] | 1,820 | 1.7773 [1.6982, 1.8601] | 30.2155 |
| 17 | 0 | 0.0000 [0.0000, 0.0038] | 1,683 | 1.6436 [1.5675, 1.7233] | 28.5348 |
| 18 | 0 | 0.0000 [0.0000, 0.0038] | 1,689 | 1.6494 [1.5732, 1.7293] | 27.0411 |
| 19 | 0 | 0.0000 [0.0000, 0.0038] | 1,702 | 1.6621 [1.5856, 1.7422] | 25.7054 |
| 20 | 0 | 0.0000 [0.0000, 0.0038] | 1,665 | 1.6260 [1.5503, 1.7053] | 24.5014 |
| 21 | 0 | 0.0000 [0.0000, 0.0038] | 1,662 | 1.6230 [1.5474, 1.7023] | 23.4120 |
| 22 | 0 | 0.0000 [0.0000, 0.0038] | 1,700 | 1.6602 [1.5837, 1.7402] | 22.4233 |
| 23 | 0 | 0.0000 [0.0000, 0.0038] | 1,746 | 1.7051 [1.6276, 1.7862] | 21.5225 |
| 24 | 0 | 0.0000 [0.0000, 0.0038] | 1,757 | 1.7158 [1.6381, 1.7972] | 20.6972 |
| 25 | 0 | 0.0000 [0.0000, 0.0038] | 1,670 | 1.6309 [1.5551, 1.7103] | 19.9345 |
| 26 | 0 | 0.0000 [0.0000, 0.0038] | 1,684 | 1.6445 [1.5684, 1.7243] | 19.2311 |
| 27 | 0 | 0.0000 [0.0000, 0.0038] | 1,744 | 1.7031 [1.6257, 1.7842] | 18.5819 |
| 28 | 0 | 0.0000 [0.0000, 0.0038] | 1,791 | 1.7490 [1.6705, 1.8311] | 17.9807 |
| 29 | 0 | 0.0000 [0.0000, 0.0038] | 1,686 | 1.6465 [1.5703, 1.7263] | 17.4175 |
| 30 | 0 | 0.0000 [0.0000, 0.0038] | 1,696 | 1.6562 [1.5799, 1.7363] | 16.8921 |
| 31 | 0 | 0.0000 [0.0000, 0.0038] | 1,657 | 1.6182 [1.5427, 1.6973] | 16.3994 |
| 32 | 0 | 0.0000 [0.0000, 0.0038] | 1,718 | 1.6777 [1.6009, 1.7582] | 15.9393 |
| 33 | 0 | 0.0000 [0.0000, 0.0038] | 1,704 | 1.6641 [1.5875, 1.7442] | 15.5067 |
| 34 | 0 | 0.0000 [0.0000, 0.0038] | 1,676 | 1.6367 [1.5608, 1.7163] | 15.0988 |
| 35 | 0 | 0.0000 [0.0000, 0.0038] | 1,769 | 1.7275 [1.6495, 1.8092] | 14.7168 |
| 36 | 0 | 0.0000 [0.0000, 0.0038] | 1,706 | 1.6660 [1.5894, 1.7462] | 14.3542 |

[CSV with exact counts and denominators](length-curves.csv) · [Plot data and provenance](plot-data.json) · [Machine-readable report](report.json)

The 10,000-update endpoint is only 2.5% of the reference 400,000-update budget. This single-seed development result is not a convergence or architecture-impossibility claim. The 5,000-update curve is a diagnostic comparison, not a newly selected endpoint. Final confirmation remains unevaluated. No training or gradient analysis was performed for this report.
