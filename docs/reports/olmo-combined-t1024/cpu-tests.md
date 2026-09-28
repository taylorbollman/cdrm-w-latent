# CPU verification

Runtime `dcbde27`, identical cherry-pick of isolated `bdde587`: **137 tests
passed**, 67 existing JIT deprecation warnings, 3.64 seconds. The final test
command explicitly selected the isolated worktree through `CDRM_ROOT`, with
`CDRM_DOCKER_GPUS=none` and installed Flash namespace:

```bash
python -m pytest -q \
  tests/test_olmo_combined_long_context.py \
  tests/test_olmo_ordinary_long_context.py \
  tests/test_olmo_ordinary_two_gpu_selection.py \
  tests/test_olmo_two_gpu_graph.py \
  tests/test_olmo_two_gpu_single_reference.py \
  tests/test_two_gpu_zero1_graph.py
```

Checks cover the narrow T1024 selector, independent required protocol and W&B
group, preserved T2048/T512 behavior, rejected combined FA4/Dao routes and actual
T1024 dispatch counting (2,046 forward tiles: 2,044 Triton, two eager; 2,046
recomputed backward calls). No production math changed. GPU checks remain
necessary and are recorded separately in each benchmark report.

Log: `.runtime/olmo-combined-t1024/logs/cpu-tests.log`. The evidence-helper extension (`46c03f0`, identical to isolated `d1ac7bb`)
passes 23 tests in `tests/test_report_combined_long_context.py` (0.19 seconds).
Its retained log is explicitly a transcript summary of the completed tool
output, not a rerun or raw terminal capture. The runtime and evidence suites
are disjoint: 160 scoped tests in total. The default T2048 audit and both saved
context references were also re-audited without GPU work.
