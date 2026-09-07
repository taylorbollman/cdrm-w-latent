# Posthoc modal-value shortcut baseline

This supplemental analysis was requested **after inspecting the pilot's noisy-recall model metrics**. It is not a prespecified primary baseline. No model, generator, frozen fixture, training setting or original baseline manifest was changed; no tie-breaking rule was tuned.

The pinned MAD generator samples its terminal query uniformly among distinct observed keys. Deduplicating repeated key/value records and returning the value associated with the most distinct keys is therefore the optimal constant answer conditional on the visible mapping while ignoring the query. Ties use the smallest value token ID. Repeated appearances of one key do not receive extra votes. [Pinned MAD source](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py).

Expected accuracy is the largest value multiplicity divided by the number of distinct observed keys, averaged over fixtures. Empirical accuracy scores that fixed prediction against actual held-out answers. These differ through finite sampling of the terminal query. The predictor reads only complete context records; it receives neither labels nor metadata and never reads the terminal query. Deduplication, fixed ties, query-token mutation and controlled-delay invariance were verified.

| Task / condition | Dev empirical | Test empirical | Test conditional expected | Frozen uniform observed-value expected |
| --- | ---: | ---: | ---: | ---: |
| noisy_recall / low | 13.379% | 11.255% | 12.184% | 5.699% |
| noisy_recall / low_delay256 | 13.379% | 11.255% | 12.184% | 5.699% |
| noisy_recall / low_delay512 | 13.379% | 11.255% | 12.184% | 5.699% |
| noisy_recall / moderate | 15.430% | 15.210% | 14.346% | 7.421% |
| noisy_recall / moderate_delay256 | 15.430% | 15.210% | 14.346% | 7.421% |
| noisy_recall / moderate_delay512 | 15.430% | 15.210% | 14.346% | 7.421% |
| mqar / associations16 | 6.250% | 6.250% | 6.250% | 6.250% |
| mqar / delay256 | 12.500% | 12.500% | 12.500% | 12.500% |
| mqar / delay512 | 12.500% | 12.500% | 12.500% | 12.500% |
| mqar / iid | 12.500% | 12.500% | 12.500% | 12.500% |

MQAR stores distinct values and queries every association once, so this query-ignoring constant prediction has exactly 1/K answer accuracy and zero all-answers-correct sequence accuracy when K>1. The modal baseline is more informative for MAD because different keys may share a value. A learned score near this shortcut cannot by itself establish key-conditioned retrieval; this posthoc comparison does not prove which algorithm a model learned.

Longer-delay conditions preserve the base mappings and labels, so this baseline produces exactly the same predictions and accuracy there. These repeated controls are not independent samples. No confidence interval or training-seed inference is claimed by this supplement.

[Complete supplemental metrics, source references and hashes](modal-value-baseline.json) · [Standalone CPU analysis script](modal_value_baseline.py)

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && OMP_NUM_THREADS=1 python docs/reports/stage-b/modal_value_baseline.py --plan configs/stage_b/pilot.json --fixtures-dir .runtime/stage-b/20260906T190223Z/fixtures --output-json docs/reports/stage-b/modal-value-baseline.json --output-markdown docs/reports/stage-b/modal-value-baseline.md'
```

The command refuses existing outputs; use new output paths for an independent rerun.
