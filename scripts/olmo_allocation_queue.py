"""Host supervisor for a bounded allocation cell, always executing GPU work in Docker.

One cell is either one two-rank job or two concurrent single-rank jobs. The
performance runner, not this supervisor, owns model execution and measurements.
Each cell has a new directory and a bounded lifetime; no training auto-restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('B', 'NFR'), required=True)
    parser.add_argument('--layout', choices=('pair', 'singles'), required=True)
    parser.add_argument('--origins', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup-updates', type=int, default=2)
    parser.add_argument('--measured-updates', type=int, default=4)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    if not 60 <= args.timeout_seconds <= 7200:
        parser.error('timeout-seconds must be between60 and7200; zero disables GNU timeout')
    if not 1 <= args.warmup_updates <= 4 or not 1 <= args.measured_updates <= 12:
        parser.error('Require1–4warmup and1–12measured updates in a bounded cell')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    relative = output.relative_to(root)
    if output.exists():
        raise ValueError('Use a new output directory; never overwrite a benchmark cell')
    origin = json.loads(args.origins.read_text())[args.arm]
    data = origin['configuration']['execution_identity']['payload']['data']
    output.mkdir(parents=True)
    cr = Path('/workspace/cdrm-w-latent')
    container_output = cr / relative
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'schema': 'olmo-allocation-supervisor-v1', 'status': 'running',
              'arm': args.arm, 'layout': args.layout, 'started_unix': time.time(),
              'origin_report_sha256': origin['report_sha256'],
              'origins_sha256': sha(args.origins), 'producer_sha256': sha(Path(__file__)),
              'jobs': []}
    write_json(output / 'supervisor.json', report)
    gate_id = output.name
    gate_file = output / 'release.json'
    jobs = [('pair', '0,1', 2)] if args.layout == 'pair' else [('single0', '0', 1), ('single1', '1', 1)]
    children = []
    started = time.monotonic()
    try:
        for name, devices, ranks in jobs:
            child_output = container_output / name
            invocation = ['env', '-u', 'GOOGLE_APPLICATION_CREDENTIALS',
                f'CUDA_VISIBLE_DEVICES={devices}', 'OMP_NUM_THREADS=1',
                'NCCL_ASYNC_ERROR_HANDLING=0', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=0',
                'timeout', '--signal=TERM', '--kill-after=30s', f'{args.timeout_seconds}s',
                'torchrun', '--standalone', f'--nproc_per_node={ranks}', '-m',
                'scripts.olmo_allocation_benchmark', '--arm', args.arm, '--scale', 'native',
                '--checkpoint', origin['checkpoint'], '--checkpoint-sha256', origin['checkpoint_sha256'],
                '--corpus', data['corpus'], '--index', data['index'],
                '--index-sha256', data['index_manifest_sha256'], '--output-dir', str(child_output),
                '--batch-size', '32' if args.arm == 'B' else '12',
                '--warmup-updates', str(args.warmup_updates),
                '--measured-updates', str(args.measured_updates)]
            if args.layout == 'singles':
                invocation += ['--start-gate', str(container_output / 'release.json'), '--gate-id', gate_id]
            preflight = ('set -euo pipefail\n'
                'test -f /.dockerenv\n'
                'test "$PWD" = /workspace/cdrm-w-latent\n'
                f'hostname > {shlex.quote(str(container_output / (name + ".container-id")))}\n'
                f'nvidia-smi -i {devices} --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader\n'
                f'test -z "$(nvidia-smi -i {devices} --query-compute-apps=pid --format=csv,noheader)"\n')
            command = ['bash', 'scripts/docker_shell.sh', 'bash', '-lc', preflight + shlex.join(invocation)]
            record = {'name': name, 'devices': devices, 'ranks': ranks, 'command': command,
                      'started_unix': time.time(), 'status': 'running'}
            log = (output / (name + '.log')).open('x')
            child = subprocess.Popen(command, cwd=root,
                env={**os.environ, 'CDRM_ROOT': str(root), 'CDRM_DOCKER_GPUS': 'all',
                     'CDRM_FLASH_ATTENTION_SOURCE': 'installed'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['host_pid'] = child.pid
            children.append((child, log, record))
            report['jobs'].append(record)
            write_json(output / 'supervisor.json', report)
        released = args.layout == 'pair'
        while any(child.poll() is None for child, _, _ in children):
            if time.monotonic() - started > args.timeout_seconds + 120:
                raise TimeoutError('Bounded allocation cell exceeded its supervisor timeout')
            failed = [(child.returncode, record['name']) for child, _, record in children
                      if child.poll() is not None and child.returncode != 0]
            if failed:
                raise RuntimeError(f'Allocation child failed: {failed}')
            if not released and all((output / name / 'ready.json').is_file() for name, _, _ in jobs):
                write_json(gate_file, {'schema': 'olmo-allocation-benchmark-v1',
                                      'gate_id': gate_id, 'released_unix': time.time()})
                released = True
                report['gate_released_unix'] = time.time()
                write_json(output / 'supervisor.json', report)
            time.sleep(2)
        for child, _, record in children:
            record.update(exit_code=child.returncode, finished_unix=time.time())
            if child.returncode != 0:
                raise RuntimeError(f"Child {record['name']} exited {child.returncode}")
            result_path = output / record['name'] / 'report.json'
            result = json.loads(result_path.read_text())
            if result.get('status') != 'completed':
                raise RuntimeError(f"Child {record['name']} has no completed report")
            record.update(status='completed', report_sha256=sha(result_path))
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        # Stop only containers whose IDs were recorded by this new cell. The
        # independent inner timeout remains a fallback if Docker is unavailable.
        docker = ['docker']
        if subprocess.run(docker + ['info'], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            docker = ['sudo', 'docker']
        for child, _, record in children:
            identity = output / (record['name'] + '.container-id')
            if child.poll() is None and identity.is_file():
                container_id = identity.read_text().strip()
                if len(container_id) in (12, 64) and all(c in '0123456789abcdef' for c in container_id):
                    stopped = subprocess.run(docker + ['stop', '--time', '30', container_id],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                    record['owned_container_stop_exit'] = stopped.returncode
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        record['host_wait_timeout'] = True
            record['exit_code'] = child.poll()
        raise
    finally:
        report['finished_unix'] = time.time()
        write_json(output / 'supervisor.json', report)
        for _, log, _ in children:
            log.close()


if __name__ == '__main__':
    main()
