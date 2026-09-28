# CPU verification

Runtime commit79e850a (identical cherry-pick of isolated b66b0f4): **122 tests
passed**,67 existing JIT deprecation warnings,3.54seconds. CPU container only,
`CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed`:

```bash
python -m pytest -q \
  tests/test_olmo_combined_long_context.py \
  tests/test_olmo_ordinary_long_context.py \
  tests/test_olmo_ordinary_two_gpu_selection.py \
  tests/test_olmo_two_gpu_graph.py \
  tests/test_olmo_two_gpu_single_reference.py \
  tests/test_two_gpu_zero1_graph.py
```

This covers narrow T2048 combined selection, rejected combined FA4/Dao routes,
required protocol, observed forward/backward tile accounting, and existing
ordinary/two-GPU behavior. No production model math changed.

Evidence helper5f6c6ac (identical cherry-pick of isolated63f72a6): **15 tests
passed** in0.11seconds:

```bash
python -m pytest -q tests/test_report_combined_long_context.py
```

These test failure exclusion, historical/native backend provenance, combined
objective/parameter checks, dispatch accounting, timing arithmetic and pooling.
The two scopes are disjoint:137tests total. They do not replace actual GPU
dispatch, finite updates or eager/graph checks. Logs are retained with closeout
evidence under`logs/cpu-tests.log` and`logs/report-cpu-tests.log`.
