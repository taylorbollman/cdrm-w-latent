# CDRM numerical diagnostics and candidate replay

See the [results and qualifications](reports/cdrm-numerical-resolution/results.md)
and [prospective local reference contract](reports/cdrm-numerical-resolution/reference-contract.md).
The lineage is `.runtime/cdrm-numerical-resolution/20260907T232931Z/`.

The selected diagnostic policy is `bf16_attention_fp32`: ordinary attention
from QKV projection through output projection runs in FP32, while the MLPs,
final head and tiled CDRM fabric retain BF16 computation. Existing FP32
normalization, residual, recurrent-state, loss and optimizer behavior is
preserved. This is an explicit wrapper in `cdrm_precision_probe.py`; it has not
been added as a training default or a general model configuration option.

## Entry points

| Script | Purpose |
| --- | --- |
| `cdrm_precision_probe.py` | Exact old-case replay, observed dtype/adjoint capture, bypass and selective precision controls. |
| `cdrm_precision_attention_reference.py` | Independent analytic FP64 attention forward/VJP on captured native operands, with local FP32 and bias controls. |
| `cdrm_precision_attention_attribution.py` | CPU vector decomposition into internal FP32 arithmetic, storage cast and additional native error. |
| `cdrm_precision_block_attribution.py` | Block-0 replay and fixed-incoming-cotangent attribution, requiring bitwise full-model parity. |
| `cdrm_precision_adam_reference.py` | Independent CPU FP64 clipping/Adam and explicit FP32 weight-write analysis. |
| `cdrm_precision_prepare.py` | Generate the one predeclared numerical corpus and distinct initialization after an anchored source/policy/criteria freeze. |
| `cdrm_precision_confirm.py` | Four-arm actual-CE, unscaled-side, Adam and cotangent-scaling comparisons on a frozen assigned role. |
| `cdrm_precision_report.py` | Provenance-checked plots, compact metrics and retained failure/exclusion records. |

These tools use saved numerical fixtures. They do not carry hypothetical
optimizer updates into a training trajectory or evaluate task accuracy.
The inherited pilot artifacts and failed numerical screens remain immutable.

## Container execution

Run CUDA commands only through the project launcher, with explicit container
and GPU checks. The following replays an existing confirmation role, so its
results would be repeat evidence on the original case, not new fresh evidence.
Use a new output directory on every invocation.

```bash
CDRM_TORCHINDUCTOR_CACHE_DIR=/workspace/cdrm-w-latent/.runtime/cdrm-numerical-resolution/20260907T232931Z/inductor-cache/replay \
bash scripts/docker_shell.sh bash -lc '
  test -f /.dockerenv &&
  test "$PWD" = /workspace/cdrm-w-latent &&
  nvidia-smi -L &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/cdrm_precision_confirm.py \
    --decision .runtime/cdrm-numerical-resolution/20260907T232931Z/decisions/candidate-freeze-v2.json \
    --decision-sha256 f5bd65e4621afa48ad8429569d287cc48a549ea71b4e1dc0e73d624b00dd288b \
    --fixtures .runtime/cdrm-numerical-resolution/20260907T232931Z/prepare/fixtures.json \
    --fixtures-sha256 c9899879b47398cec5784008a9db0bf4524a1083f70236ecdfa6dc1f3964f65a \
    --role fp32_trained_u1000 \
    --output-dir .runtime/cdrm-numerical-resolution/20260907T232931Z/confirmation/replay-fp32-trained \
    --wandb-project cdrm-numerical-resolution \
    --wandb-entity taylorbollman \
    --wandb-group 20260907T232931Z \
    --wandb-run-name replay-fp32-trained
'
```

Source checks intentionally reject a changed candidate, harness, contract or
checkpoint. Use the matching archived source snapshot when reproducing this
lineage after later implementation changes. Do not disable those checks or
rewrite the frozen decision. CPU references and tests use the same launcher
with `CDRM_DOCKER_GPUS=none` explicitly.

## Reading a confirmation result

`status=diagnostics_complete` describes execution. The original mixed policy,
the candidate and the naive-versus-tiled FP32 comparison retain separate flags.
`candidate_prospective_checks_pass` combines the candidate's applicable original
screens with the additional scaling diagnostics. The overall raw
`machine_screens_pass` also includes the original policy and strict FP32
reference comparisons, and can therefore remain false even when the candidate
passes. The reviewed report states the resulting scope and qualifications.

The numerical profile is five blocks, sites 1/3, D128/H16, physical B64/T256,
V16 and 96 native answer targets per sequence, fixed lambda 0.01, rho 1,
epsilon 0.1, math SDPA, zero dropout and the saved optimizer state. No speed,
long-sequence, task-learning or general production claim follows from these
bounded checks.
