# Execution notes

All GPU work runs through the project container on one H10080GB. This log records
diagnostic attempts and implementation fixes; final selected evidence is listed
in `summary.json` when the queue is complete.

- `f483646`: initial frozen backend, harness and protocol; 122 CPU tests pass.
  `verify-author-fp32-b1-t32` failed during checkpoint loading with zero optimizer
  updates. The isolated stack registers `layers`, while native checkpoint keys
  use `transformer.blocks`. This is a harness mapping failure, not a numerical
  result. Its failed report/source snapshot remains retained.
- `6eea309`: explicit native-to-isolated block-name mapping. The unit fixture
  now uses an actual tiny `OLMoForCausalLM.state_dict()` instead of an isolated
  stack state dictionary. All 27 harness tests pass after the correction.
  Native-width FP32 retry: `verify-author-fp32-b1-t32-r2`.
- The retry passes all eight FP32/operational checks. Author tiled versus native
  scan aggregate parameter-gradient L2 is5.703865e-7; same-candidate outputs,
  gradients and three-update Adam state are exact under graph replay.
- Author BF16 B1/T32 passes its reference/compatibility/operational checks.
  Aggregate parameter-gradient difference from FP32 is0.0049730 for author and
  0.0050534 for native. Those cross-precision rows remain diagnostics.
- Author and native BF16 B8/T512 each pass all five checks, including exact
  graph/update replay. Cross-backend gradient L2 is0.00413293. Four successful
  verification reports execute24 actual optimizer updates; the failed load
  attempt executes zero. Capacity follows only after these operational gates.
