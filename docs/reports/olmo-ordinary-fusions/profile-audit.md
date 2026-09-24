# Full-step profile audit

2026-09-24, frozen runtime `b70b3ec`. This is a derived analysis correction;
runtime code, raw reports/traces and synchronized throughput samples are unchanged.
The full-step profiles each execute the separately counted ninth optimizer update.

## Why the raw sum was too large

Kineto records both CUDA kernel/memory events and `gpu_user_annotation` intervals.
The harness selected CUDA events by device type, so its raw `device_kernels`
inventory also contains intervals named `ordinary_step/*` and
`Optimizer.step#AdamW.step`. These annotation intervals overlap the real kernels.
Adding them to kernel times double counts execution and cannot measure step time.

The derived report verifies compressed trace bytes against the immutable report,
then excludes exact `ordinary_step/` markers and every name identified as
`gpu_user_annotation` in that trace. Exclusions, raw totals and raw counts remain
explicit in the summary. Categories, shares and rankings use the remaining CUDA
kernel/memory events. The separate forward/loss/backward operator profile is
unaffected: the control's summed device time remains 790.729979 ms.

The raw `cpu_phase_scopes` dictionary also mixes CPU and CUDA key averages. Shared
names can overwrite CPU values with zero. The derived report instead reads CPU
`user_annotation` complete (`ph="X"`) events from the verified trace. Those
inclusive CPU scopes overlap and must not be added together; CPU dispatch duration
is not GPU execution duration.

## Actual trace results

| Measurement | Native RoPE / scalar AdamW | Dao RoPE / fused AdamW |
|---|---:|---:|
| Raw CUDA inventory sum, including annotations | 1,702.687856 ms | 1,498.304000 ms |
| Excluded GPU annotation intervals | 5 | 5 |
| CUDA kernel/memory sum after exclusion | 829.600276 ms | 738.375572 ms |
| Recovered CPU optimizer scope | 5.677950 ms | 0.546848 ms |
| Optimizer-associated CUDA event sum, linked below | 31.195528 ms | 11.252450 ms |
| Optimizer-associated CUDA event count | 520 | 10 |
| GPU optimizer annotation interval, descriptive | 32.133630 ms | 11.267203 ms |

The combined candidate has nine kernels whose names contain `FusedAdamMathFunctor`,
totaling 11.250434 ms. Its `rotary_kernel` has 96 calls totaling 25.847029 ms.
Native scalar Adam and native RoPE use shared pointwise kernels, so the absence
of unique optimizer/rotary names is not zero cost. In particular, these figures
do not provide a standalone native RoPE cost or an isolated RoPE speedup.

Profile sources:

- [Control raw report](../../../.runtime/olmo-ordinary-fusions/capacity-control-b64-t512-01/report.json)
  and [compressed trace](../../../.runtime/olmo-ordinary-fusions/capacity-control-b64-t512-01/full-step-trace.json.gz).
- [Combined raw report](../../../.runtime/olmo-ordinary-fusions/capacity-dao-rope-fused-adam-b64-t512-01/report.json)
  and [compressed trace](../../../.runtime/olmo-ordinary-fusions/capacity-dao-rope-fused-adam-b64-t512-01/full-step-trace.json.gz).

These exact selected reports/traces are included by the evidence retainer.

## Reproduce optimizer association

This additional trace analysis links GPU kernel/memset/memcpy events by their
`args["External id"]` to CPU operations contained in the `ordinary_step/optimizer`
scope on the same process/thread. It does not classify generic kernels by their
names or allocate CPU scope duration to GPU work. The result is a sum of linked
device event durations, excluding GPU annotations, launch gaps and CPU overhead.
Only these inspected traces are covered; absent external IDs or asynchronous work
launched outside the CPU scope could make this method incomplete in another trace.
CUDA graph kernels often have different linkage, so this method is not being used
to attribute captured model computation to layers.

Run this read-only Python snippet from the project root (CPU only):

```python
import gzip
import hashlib
import json
import math
from pathlib import Path

root = Path(".runtime/olmo-ordinary-fusions")
for name in ("capacity-control-b64-t512-01",
             "capacity-dao-rope-fused-adam-b64-t512-01"):
    report = json.loads((root / name / "report.json").read_text())
    profile = report["full_step_profile"]
    data = (root / name / profile["trace_file"]).read_bytes()
    assert len(data) == profile["trace_bytes"]
    assert hashlib.sha256(data).hexdigest() == profile["trace_sha256"]
    events = json.loads(gzip.decompress(data))["traceEvents"]
    scope = next(event for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == "ordinary_step/optimizer")
    external_ids = {
        event.get("args", {}).get("External id")
        for event in events
        if event.get("cat") == "cpu_op"
        and event.get("pid") == scope["pid"]
        and event.get("tid") == scope["tid"]
        and event.get("ts", -1) >= scope["ts"]
        and event.get("ts", -1) + event.get("dur", 0)
            <= scope["ts"] + scope["dur"]
    }
    external_ids.discard(None)
    linked = [event for event in events
        if event.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")
        and event.get("args", {}).get("External id") in external_ids]
    print(name, len(linked), math.fsum(event["dur"] for event in linked) / 1000)
```

These are single untimed profiles, not confidence intervals or throughput
measurements. Use the separately synchronized complete-step timing samples for
performance conclusions. A regression fixture verifies annotation exclusion,
unchanged source evidence, corrected category shares and recovered CPU durations.
