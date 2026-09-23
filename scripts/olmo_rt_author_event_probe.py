#!/usr/bin/env python3
"""Functional preflight for timing segments inside one training CUDA graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.olmo_validation import require_container_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = require_container_gpu()
    torch.manual_seed(20260923)
    x = torch.randn(16, 32, device="cuda", requires_grad=True)
    w = torch.randn(32, 32, device="cuda", requires_grad=True)
    x.grad, w.grad = torch.zeros_like(x), torch.zeros_like(w)
    start, middle, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]

    def body(record):
        x.grad.zero_()
        w.grad.zero_()
        if record:
            start.record()
        loss = (x @ w).square().mean()
        if record:
            middle.record()
        loss.backward()
        if record:
            end.record()
        return loss

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            body(False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_loss = body(True)
    checks = []
    for index in range(2):
        with torch.no_grad():
            x.add_(0.125)
            w.mul_(0.99)
        reference_loss = body(False).detach().clone()
        reference_grads = x.grad.clone(), w.grad.clone()
        graph.replay()
        torch.cuda.synchronize()
        segments = {"forward_ms": start.elapsed_time(middle),
                    "backward_ms": middle.elapsed_time(end),
                    "forward_backward_ms": start.elapsed_time(end)}
        exact = torch.equal(captured_loss, reference_loss) and all(
            torch.equal(a, b) for a, b in zip((x.grad, w.grad), reference_grads))
        checks.append({"replay": index, "exact_loss_and_gradients": exact, **segments,
                       "passed": exact and all(value > 0 for value in segments.values())
                       and abs(segments["forward_ms"] + segments["backward_ms"]
                               - segments["forward_backward_ms"]) < 1e-4})
    report = {"schema": "olmo-author-event-preflight-v1", "runtime": runtime,
              "status": "passed" if all(check["passed"] for check in checks) else "failed",
              "scope": "External CUDA events in one training graph; functional instrumentation test, not a throughput benchmark.",
              "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))
    if report["status"] != "passed":
        raise AssertionError("External training-graph event preflight failed")


if __name__ == "__main__":
    main()
