#!/usr/bin/env python3
"""CPU attribution of saved native attention error into explicit storage effects.

This preserves the independent FP64 interval failures. Matching the local FP32
operation after its intended storage cast is a separate descriptive observation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from cdrm_precision_adam_reference import digest, save_json
from experiment_tracking import OnlineTracker


def decomposition(reference, local_fp32, native):
    r, p, n = (x.detach().double() for x in (reference, local_fp32, native))
    if not all(bool(torch.isfinite(x).all()) for x in (r, p, n)):
        raise ValueError("Attribution requires finite saved values")
    rounded = local_fp32.to(native.dtype).double()
    parts = {"native_extra": n-rounded, "storage_cast": rounded-p,
             "fp32_arithmetic": p-r}
    total = n-r
    ref_energy = float(r.square().sum())
    total_energy = float(total.square().sum())
    rows = {name: {"energy": float(value.square().sum()),
                   "relative_l2": float(value.norm()/r.norm().clamp_min(1e-30)),
                   "maximum_absolute": float(value.abs().max())}
            for name, value in parts.items()}
    names = list(parts)
    interactions = {f"{a}__{b}": float(2*(parts[a]*parts[b]).sum())
                    for i, a in enumerate(names) for b in names[i+1:]}
    reconstruction = sum(parts.values())-total
    energy_residual = total_energy-sum(v["energy"] for v in rows.values())-sum(interactions.values())
    if abs(energy_residual) > 1e-12*max(total_energy, ref_energy, 1e-30):
        raise AssertionError("Error energy decomposition does not close")
    return {"native_dtype": str(native.dtype), "elements": n.numel(),
            "native_equals_cast_local_fp32": torch.equal(native, local_fp32.to(native.dtype)),
            "native_cast_local_fp32_mismatches": int((n != rounded).sum()),
            "reference_energy": ref_energy, "total_error_energy": total_energy,
            "total_relative_l2": float(total.norm()/r.norm().clamp_min(1e-30)),
            "components": rows, "twice_cross_inner_products": interactions,
            "energy_identity_residual": energy_residual,
            "vector_identity_maximum_residual": float(reconstruction.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    tracker = None
    inputs = {str(p): digest(p) for d in (args.probe_dir, args.reference_dir)
              for p in (d/'report.json', d/('tensors.pt' if d == args.probe_dir else 'references.pt'))}
    source = {str(Path(__file__)): digest(__file__),
              'scripts/cdrm_precision_adam_reference.py': digest('scripts/cdrm_precision_adam_reference.py'),
              'scripts/experiment_tracking.py': digest('scripts/experiment_tracking.py')}
    report = {'schema': 'cdrm-attention-storage-attribution-v1', 'status': 'running',
              'numerical_clearance': False, 'device': 'cpu', 'source_sha256': source,
              'input_sha256': inputs, 'observations': {},
              'scope': 'Same-operand saved local math SDPA and independent analytic FP64 reference. Original interval failures remain unchanged; no revised acceptance threshold.',
              'identity': 'native-reference64=(native-cast(local32))+(cast(local32)-local32)+(local32-reference64)'}
    try:
        pr = json.loads((args.probe_dir/'report.json').read_text())
        rr = json.loads((args.reference_dir/'report.json').read_text())
        if any(x['status'] != 'diagnostics_complete' for x in (pr, rr)):
            raise ValueError('Require completed parent diagnostics')
        for d, r, filename in ((args.probe_dir, pr, 'tensors.pt'), (args.reference_dir, rr, 'references.pt')):
            if r['tensor_artifact']['sha256'] != inputs[str(d/filename)]:
                raise ValueError('Retained tensor hash differs from its report')
        if (rr['probe']['report_sha256'] != inputs[str(args.probe_dir/'report.json')]
                or rr['probe']['tensors_sha256'] != inputs[str(args.probe_dir/'tensors.pt')]):
            raise ValueError('Local reference was computed from a different probe')
        native = torch.load(args.probe_dir/'tensors.pt', map_location='cpu', weights_only=False)
        reference = torch.load(args.reference_dir/'references.pt', map_location='cpu', weights_only=False)
        (args.output_dir/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        tracker = OnlineTracker(project='cdrm-numerical-resolution', entity='taylorbollman',
                                group='20260907T232931Z', name='attention-storage-attribution',
                                output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({'scope': report['scope'], 'input_sha256': inputs, 'source_sha256': source})
        for name, values in reference.items():
            arm, block_name = name.split('/')
            block = block_name.removeprefix('block')
            capture = native['captures'][arm]['blocks'][block]['sdpa']
            rows = {key: decomposition(value, values['local_fp32'][key], capture[key])
                    for key, value in values['analytic'].items()}
            report['observations'][name] = {'rows': rows,
                'inherited_native_local_interval_pass': rr['observations'][name]['native_local_interval_pass'],
                'inherited_reference_floor_supported': rr['observations'][name]['reference_floor_supported']}
            tracker.log({'block': int(block), **{f'{arm}/{key}/native_extra_relative_l2': row['components']['native_extra']['relative_l2']
                                                for key, row in rows.items()},
                         **{f'{arm}/{key}/cast_local_fp32_exact': row['native_equals_cast_local_fp32'] for key, row in rows.items()}})
            print(json.dumps({'case': name, 'cast_local_fp32_mismatches': {k: v['native_cast_local_fp32_mismatches'] for k, v in rows.items()},
                              'extra_relative_l2': {k: v['components']['native_extra']['relative_l2'] for k,v in rows.items()}}), flush=True)
        if any(digest(p) != h for p,h in {**inputs, **source}.items()):
            raise RuntimeError('Attribution inputs or sources changed')
        report['status'] = 'diagnostics_complete'
        tracker.summary({'all_native_equal_cast_local_fp32': all(row['native_equals_cast_local_fp32']
                        for obs in report['observations'].values() for row in obs['rows'].values()),
                         'numerical_clearance': False})
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            if tracker:
                tracker.finish(succeeded=report['status'] == 'diagnostics_complete')
        except Exception as error:
            report.update(status='failed', tracking_error_type=type(error).__name__)
            raise
        finally:
            save_json(args.output_dir/'report.json', report)


if __name__ == '__main__':
    main()
