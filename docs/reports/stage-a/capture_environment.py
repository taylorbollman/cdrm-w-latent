"""Capture bounded installation/hardware provenance inside the project container."""
import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import site
import subprocess

root = Path('/workspace/cdrm-w-latent')
assert Path.cwd() == root and Path('/.dockerenv').exists()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader'], text=True).strip()
import torch
import olmo
assert torch.cuda.is_available(), 'GPU requested; CPU fallback is forbidden'
assert Path(olmo.__file__).resolve().parent == root / 'recurrent-transformer/olmo'
packages = ('ai2-olmo', 'ai2-olmo-core', 'numpy', 'pytest', 'transformers', 'huggingface-hub', 'datasets', 'flash-attn-4', 'triton')
result = {
    'evidence_category': 'OPS', 'status': 'verified',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'container': True, 'working_directory': str(Path.cwd()),
    'user_site_enabled': site.ENABLE_USER_SITE,
    'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu_query': gpu, 'cuda_device_count': torch.cuda.device_count(),
    'olmo_import': olmo.__file__,
    'editable_distribution': json.loads(metadata.distribution('ai2-olmo').read_text('direct_url.json')),
    'packages': {name: metadata.version(name) for name in packages},
    'parent_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'fork_revision': subprocess.check_output(['git', '-C', 'recurrent-transformer', 'rev-parse', 'HEAD'], text=True).strip(),
    'fork_dirty': bool(subprocess.check_output(['git', '-C', 'recurrent-transformer', 'status', '--porcelain'], text=True).strip()),
    'requirements_sha256': hashlib.sha256((root / 'docker/requirements-docker.txt').read_bytes()).hexdigest(),
}
pip_check = subprocess.run(['python', '-m', 'pip', 'check'], text=True, capture_output=True)
result['pip_check'] = {'exit_code': pip_check.returncode, 'stdout': pip_check.stdout.strip(), 'stderr': pip_check.stderr.strip()}
if pip_check.returncode:
    result['status'] = 'failed'
(root / 'docs/reports/stage-a/environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(pip_check.returncode)
