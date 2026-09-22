# O5d: does fusion adaptation transfer to sequential feedback?

Evaluation only, with unchanged checkpoints, fixed beta 1, RT/NextLat off and BF16 mixed precision. Every execution within a selection scores identical teacher-forced targets; exact online means sequential feedback, not free-running generation or a high-precision numerical oracle. Short prefixes and full contexts are different selections and their NLL levels must not be compared as matched samples. The ordinary reference is the shared frozen pass, not an equally additionally trained control. Development data, one training seed, and no reserved-test evaluation; this is a transfer diagnostic, not an FBT efficacy claim.

All NLL values score an individual final pass or the exact online state; the summed training objective is not reported.

## Short-prefix diagnostic

First 32 development windows per domain, maximum 64 input tokens per window; batch 8.

| Endpoint / execution | Code NLL | WikiText NLL | Code accuracy | WikiText accuracy |
| --- | ---: | ---: | ---: | ---: |
| Shared frozen ordinary | 2.165378 | 4.243235 | 56.278% | 31.548% |
| O5b source / K2 | 2.200133 | 5.083958 | 55.528% | 22.321% |
| O5b source / K3 | 2.200488 | 5.013408 | 55.428% | 23.611% |
| O5b source / K4 | 2.201095 | 5.033363 | 55.378% | 23.313% |
| O5b source / online | 2.200238 | 5.032388 | 55.428% | 23.413% |
| O5c mixed / K2 | 2.186332 | 3.859321 | 55.678% | 34.673% |
| O5c mixed / K3 | 2.184171 | 3.901344 | 55.728% | 34.226% |
| O5c mixed / K4 | 2.183877 | 3.897338 | 55.728% | 34.325% |
| O5c mixed / online | 2.184656 | 3.899966 | 55.728% | 34.325% |

Paired NLL differences with original-document bootstrap 95% intervals:

| Contrast | Code | WikiText |
| --- | ---: | ---: |
| Mixed minus source, K2 | -0.013802 [-0.024621, -0.003671] | -1.224636 [-1.471108, -0.993437] |
| Mixed minus source, K3 | -0.016317 [-0.027723, -0.006341] | -1.112064 [-1.271904, -0.950022] |
| Mixed minus source, K4 | -0.017218 [-0.029232, -0.007038] | -1.136025 [-1.312998, -0.962863] |
| Mixed minus source, online | -0.015582 [-0.026820, -0.005597] | -1.132422 [-1.313131, -0.961765] |
| Source online minus K2 | +0.000105 [-0.003540, +0.003999] | -0.051569 [-0.116640, +0.032177] |
| Mixed online minus K2 | -0.001675 [-0.003699, +0.000565] | +0.040645 [+0.023119, +0.064726] |
| Mixed online minus ordinary | +0.019278 [+0.007951, +0.029437] | -0.343269 [-0.502780, -0.243549] |
| (Online − K2) mixed minus source | -0.001781 [-0.004964, +0.001489] | +0.092214 [+0.009325, +0.169796] |

Selection: code: 25 original documents, 1,999 CE targets; WikiText: 6 original documents, 2,016 CE targets. Intervals quantify evaluation-document variability only, not training-seed uncertainty. Short selections contain few original documents.

## Full-context extension

First 512 development windows per domain, maximum 512 input tokens per window; batch 8.

| Endpoint / execution | Code NLL | WikiText NLL | Code accuracy | WikiText accuracy |
| --- | ---: | ---: | ---: | ---: |
| Shared frozen ordinary | 1.698216 | 3.183361 | 64.219% | 40.738% |
| O5b source / K2 | 1.737004 | 4.969719 | 63.645% | 24.357% |
| O5b source / K3 | 1.741812 | 4.825642 | 63.602% | 25.044% |
| O5b source / K4 | 1.742301 | 5.090731 | 63.602% | 23.191% |
| O5b source / online | 1.742469 | 5.101028 | 63.604% | 23.207% |
| O5c mixed / K2 | 1.720900 | 3.061443 | 63.847% | 41.637% |
| O5c mixed / K3 | 1.723029 | 3.141061 | 63.789% | 40.540% |
| O5c mixed / K4 | 1.723191 | 3.138426 | 63.815% | 40.605% |
| O5c mixed / online | 1.723307 | 3.140935 | 63.804% | 40.597% |

Paired NLL differences with original-document bootstrap 95% intervals:

| Contrast | Code | WikiText |
| --- | ---: | ---: |
| Mixed minus source, K2 | -0.016103 [-0.018235, -0.014188] | -1.908275 [-2.050092, -1.746142] |
| Mixed minus source, K3 | -0.018782 [-0.021304, -0.016450] | -1.684581 [-1.784253, -1.575496] |
| Mixed minus source, K4 | -0.019110 [-0.021696, -0.016799] | -1.952304 [-2.095913, -1.789831] |
| Mixed minus source, online | -0.019161 [-0.021762, -0.016791] | -1.960093 [-2.103950, -1.786748] |
| Source online minus K2 | +0.005465 [+0.004544, +0.006404] | +0.131309 [+0.027684, +0.258650] |
| Mixed online minus K2 | +0.002407 [+0.001929, +0.002910] | +0.079491 [+0.072627, +0.086351] |
| Mixed online minus ordinary | +0.025091 [+0.021790, +0.028122] | -0.042427 [-0.056161, -0.029047] |
| (Online − K2) mixed minus source | -0.003058 [-0.003791, -0.002356] | -0.051818 [-0.174050, +0.048743] |

Selection: code: 422 original documents, 127,850 CE targets; WikiText: 59 original documents, 246,910 CE targets. Intervals quantify evaluation-document variability only, not training-seed uncertainty. Short selections contain few original documents.

## Execution and provenance

All model and buffer hashes were unchanged at both endpoints. Their native backbone and fixed fusion scale are identical; only the fusion matrices differ.

| Endpoint | Checkpoint SHA256 |
| --- | --- |
| source | `99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66` |
| mixed | `7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a` |

Case durations are recorded in report.json. They are eager evaluation wall times, not an optimized throughput comparison.

[Finite/online figure](finite-online.pdf) · [Adaptation transfer figure](adaptation-transfer.pdf) · [Full report](report.json)

[W&B evaluation run](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/jhhx390b)
