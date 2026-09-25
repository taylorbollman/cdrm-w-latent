#!/usr/bin/env python3
"""Real two-H100 NCCL sanity and bounded collective timing (no training)."""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from cdrm.pretrained.artifacts import write_json, sha256_file
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_validation import require_container_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    configure_determinism(True)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    runtime = require_container_gpu()
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('This protocol requires exactly two GPU ranks')
    dist.init_process_group('nccl', device_id=torch.device('cuda', local),
                            timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(schema='olmo-two-gpu-nccl-v1', runtime=runtime,
        nccl=list(torch.cuda.nccl.version()), world_size=2, rows=[], passed=False,
        source_sha256=sha256_file(Path(__file__)))
    tracker = None
    try:
        if rank == 0:
            tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat',
                group='olmo-two-gpu', name=args.output_dir.name, output_dir=args.output_dir)
            tracker.start(dict(stage='nccl', sizes_bytes=[4, 4<<20, 25<<20, 100<<20, 256<<20]))
            report['wandb'] = tracker.record
            print(tracker.record['run_url'], flush=True)
        dist.barrier()
        for size in (4, 4<<20, 25<<20, 100<<20, 256<<20):
            tensor = torch.full((size//4,), rank+1., device=f'cuda:{local}')
            dist.all_reduce(tensor)
            correct = torch.tensor(int(bool(torch.all(tensor == 3))), device=tensor.device)
            dist.all_reduce(correct, op=dist.ReduceOp.MIN)
            if not int(correct):
                raise AssertionError(f'Incorrect sum for {size} bytes')
            elapsed = []
            for iteration in range(25):
                tensor.fill_(rank+1.)
                torch.cuda.synchronize(local)
                dist.barrier()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                dist.all_reduce(tensor)
                end.record()
                end.synchronize()
                timing = torch.tensor(begin.elapsed_time(end), dtype=torch.float64, device=tensor.device)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                if iteration >= 5:
                    elapsed.append(float(timing))
            median = statistics.median(elapsed)
            row = dict(bytes=size, correct=True, median_rank_max_ms=median,
                       payload_GB_per_second=size/1e9/(median/1e3), measured=20, warmup=5)
            report['rows'].append(row)
            if rank == 0:
                print(row, flush=True)
                tracker.log({f'collective/{size}/{k}': v for k,v in row.items()})
                write_json(args.output_dir/'progress.json', report)
            del tensor
        report['passed'] = True
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if rank == 0:
            if tracker is not None:
                tracker.finish(succeeded=report['passed'])
            write_json(args.output_dir/'report.json', report)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
