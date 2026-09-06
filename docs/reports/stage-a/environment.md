# Stage A environment and pristine upstream numerical baseline

Date: 2026-09-06. This report records environment and bounded NUM checks only.
It does not establish research training, held-out language-model performance, or
an authorized research budget. Proposed stages and budgets remain in the
[unaltered input protocol](../../inputs/recurrent_transformer_initial_run_protocol.md).

## Relocation and GPU shell

The maintained `recurrent-transformer` submodule moved from `vendors/` to the
project root. Its starting revision remains
`a21b42d2bc292edb86ed1b62cee4bcab809a9d21`; the parent starting revision is
`302e3389390d975f2e0ceb35b6e6c318e63c2b5a`. The working fork contains concurrent
Stage A implementation edits, recorded separately from this upstream pin.

Updated `.dockerignore`, Dockerfile, build-path validation, and documentation.
Rebuilt using `CDRM_BASE_IMAGE=cdrm-w-latent:dev bash scripts/docker_build.sh`
to reuse the populated environment. Final image identity is in [image.txt](image.txt).
No PyTorch/CUDA replacement was needed. The editable distribution and imported
module both resolve to `/workspace/cdrm-w-latent/recurrent-transformer`.

GPU bootstrap previously rebuilt the image on every shell launch, including when
a working image already existed. It now reuses the existing image and already
mounted SSD. A contemporaneous older launch was observed building from NGC; it
was left untouched and finished before the relocated image rebuild completed.
The exact cause of the user's earlier difficulty was not reproduced, but the
unnecessary build dependency was removed and the full home-wrapper launch now
passes. No RAID creation, formatting, or deletion was performed.

Verified both the direct launcher and:

```bash
bash /home/taylorbollman/start.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L'
```

The startup smoke output is saved in [bootstrap_output.txt](bootstrap_output.txt).
Shell syntax checks pass for the changed bootstrap/build scripts and launcher.
Rebuild explicitly after dependency or editable-path changes; opening a shell
no longer implies rebuilding the image.

## Hardware and installation

[environment.json](environment.json) is the machine-readable capture; its driver
is [capture_environment.py](capture_environment.py). It records import location,
package versions, revisions, and `pip check` output without credentials or a full
environment-variable dump.

| Item | Observed |
|---|---|
| Container working directory | `/workspace/cdrm-w-latent` |
| GPU | One NVIDIA H100 80GB HBM3 |
| Reported GPU memory | 81,559 MiB |
| Host kernel driver | 580.173.02 |
| Compute capability | 9.0 |
| Python | 3.12.3 |
| PyTorch module version | `2.13.0a0+8145d630e8.nv26.06` |
| CUDA | 13.3, NGC forward compatibility mode |
| Editable package | `ai2-olmo==0.6.0` |
| Dependency consistency | `pip check` passes |

The initial capture detected `huggingface-hub==1.23.0` in the persisted container
user site, shadowing the image's explicit `1.22.0` pin. Running Python with its
user site disabled resolves the pinned image version; Torch and OLMo resolve from
the image in either case. This is distinct from `pip check`, which verifies
installed dependency constraints rather than exact project pins. The launcher now
sets `PYTHONNOUSERSITE=1`; persisted user-site files are unchanged. The final
capture and repeat `pip check` pass with the image's pinned `huggingface-hub==1.22.0`.
The earlier capture is retained in
[environment_before_user_site_isolation.json](environment_before_user_site_isolation.json).

## Unmodified upstream baseline

The source was obtained with `git archive` at the upstream pin into the ignored
`.runtime/upstream-a21b42d/` directory. No source files in that snapshot were
edited, and its import location and model-source hash are recorded. The small
standalone driver imports only that source, not the fork being edited.

Reproduction:

```bash
mkdir -p .runtime/upstream-a21b42d
git -C recurrent-transformer archive a21b42d2bc292edb86ed1b62cee4bcab809a9d21 \
  | tar -x -C .runtime/upstream-a21b42d
bash scripts/docker_shell.sh python docs/reports/stage-a/upstream_baseline.py
bash scripts/docker_shell.sh python docs/reports/stage-a/upstream_baseline.py --float64
```

Explicit fixture: batch 2, sequence length 8, width 32, four heads, MLP width 128,
two layers, vocabulary 64, seed 20260906, ALiBi, full MHA, Q/K norms, dropout zero,
no RoPE, no CUDA graphs, no top-level model compile, backward MLP chunks one.
The upstream internal compiled helpers were retained unchanged. Both post-norm
(the upstream debug script's active setting) and pre-norm were checked. The
upstream converter was also retained, so these are default-norm backend checks,
not an exhaustive conversion test.

The declared elementwise tolerance is **atol=1e-5, rtol=1e-4**, unchanged after
observing failures. TF32 was disabled for matrix multiplication and cuDNN;
float32 matrix-multiplication precision was set to `highest`. All 21 named
parameter gradients were compared, without missing gradients, for each case.

| Run | Surrogate backward | Random output-cotangent backward |
|---|---|---|
| First FP32, post-norm | Pass | Pass |
| First FP32, pre-norm | Pass | Fail: block 0 K/V projection |
| Diagnostic FP32, post-norm | Pass | Fail: block 0 MLP output and K/V projections |
| Diagnostic FP32, pre-norm | Pass | Pass |
| Higher-precision model, both norm modes | Pass | Pass |

Logits pass in every case. The first FP32 run's maximum logit discrepancy is
1.37e-6; its failing pre-norm random-cotangent case has maximum parameter-gradient
absolute discrepancy 4.58e-5. The diagnostic FP32 rerun records four violating
parameter elements, with absolute errors 1.55e-5 to 4.13e-5 and allowed-error ratios
1.05 to 1.73. Affected tensor relative squared errors are about 5e-13. The
specific FP32 failures varied across repeated compiled executions.

Higher-precision model parameters/activations reduce the discrepancy and pass the
same declared tolerances (maximum random-cotangent gradient discrepancy 1.01e-5
post-norm, 4.41e-6 pre-norm). This is **not an entirely FP64 custom backward**:
upstream `recompute_alphas` explicitly computes its softmax in FP32. Random
cotangents are sampled in each run's dtype, so the FP64 run is an additional
backend fixture, not an identical-cotangent FP32/FP64 convergence measurement. These
observations are consistent with floating-point reduction sensitivity, but they
do not erase the failed FP32 tolerance check or validate the fork's tiled path.
That path requires its own current-source gate.

Artifacts:

- [First run](upstream_baseline_first.json), with [original output](upstream_baseline_first_output.txt).
- [Diagnostic FP32 rerun](upstream_baseline.json), including offending elements.
- [Higher-precision comparison](upstream_baseline_fp64.json).
- [Diagnostic execution output](upstream_baseline_diagnostic_output.txt).
- [Reproduction driver](upstream_baseline.py).

The mean-logit surrogate intentionally reproduces upstream's objective. The
random cotangent probes a broader vector–Jacobian product. Neither substitutes
for shifted next-token cross-entropy, actual optimizer updates, a deterministic
resume check, or a whole-model performance benchmark. No wall-time or throughput
claim is derived from these checks.
