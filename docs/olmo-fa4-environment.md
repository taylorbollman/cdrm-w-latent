# Selecting the installed FlashAttention-4 environment

The launcher supports `CDRM_FLASH_ATTENTION_SOURCE=installed` for explicit
FA4 experiments. It removes only `vendors/flash-attention` from the container
`PYTHONPATH`; project, Apex, vLLM and the editable recurrent-model installation
remain available. The default, `vendor`, preserves the historical path order.
Unknown selectors fail before Docker starts. Set the variable on the launcher
command, rather than inside its child shell; the launcher records the resolved
choice in the container environment.

CPU-only import check, from the project root:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -c "import torch; import flash_attn.cute.interface as fa; print(fa.__file__); print(torch.cuda.is_initialized())"'
```

For an authorized GPU experiment, use the same selector with the usual GPU
container launcher, then verify container location and `nvidia-smi` before the
experiment. The selector changes import resolution only. It does not route
native RT attention through FA4, change ordinary PyTorch SDPA dispatch, or
establish GPU/kernel correctness. Separate GPU smoke and numerical checks are
required before using the package in a model backend.

## Audited environment

CPU-container inspection on 2026-09-22 found the following installed versions.
No packages were installed, removed or rebuilt for this change.

| Distribution | Installed version |
| --- | --- |
| `flash-attn-4` | `4.0.0b20` |
| `nvidia-cutlass-dsl` | `4.6.0.dev0` |
| `nvidia-cutlass-dsl-libs-base` | `4.6.0.dev0` |
| `nvidia-cutlass-dsl-libs-cu13` | `4.6.0.dev0` |
| `torch` metadata | `2.13.0a0+8145d630e8.nv26.6.54250401` |
| `torch.__version__` | `2.13.0a0+8145d630e8.nv26.06` |
| `cuda-python` / `cuda-bindings` | `13.3.1` |
| `triton` | `3.7.0+gitb7fa781f.nv26.6` |
| `apache-tvm-ffi` | `0.1.14.post0` |
| `torch-c-dlpack-ext` | `0.1.5` |
| `quack-kernels` | `0.5.3` |

The existing [Docker requirements](../docker/requirements-docker.txt) pin
`flash-attn-4[cu13]==4.0.0b20`. That installed wheel's own metadata requires
`nvidia-cutlass-dsl==4.6.0.dev0`, including the CUDA-13 extra. Both imports and
that dependency pairing are consistent in this image.

With `vendor`, Python resolves `flash_attn` to
`/workspace/cdrm-w-latent/vendors/flash-attention/flash_attn/__init__.py`.
Importing its `cute.interface` raises
`AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'`.
The vendored source references this attribute, while the installed FA4 wheel
imports successfully against the installed DSL. Selecting the matching wheel
therefore addresses the observed shadowing problem without a dependency
reinstall. It leaves the vendored source unchanged for historical execution.

## Source identity

These are hashes of actual inspected files, not a parent repository revision:

| Source | SHA-256 |
| --- | --- |
| Installed `/usr/local/lib/python3.12/dist-packages/flash_attn/cute/interface.py` | `5706f8dd21d744836d0a16b442fa3ae9b6d0867008f1763d2a285cd6df72b9ca` |
| Installed `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py` | `6728688542bbb84711a04fb160753ecd26c8cf9634e6c139f7aef459bc5abf12` |
| Vendored `vendors/flash-attention/flash_attn/cute/interface.py` | `4d8aee676c4a1ae6373d71d458837dc065449d41d124b2c77a3fde0a2d2b360a` |

The installed `flash_attn/cute` Python source tree contains 50 `.py` files.
Its source-list digest is
`362faf9744295545fa0adfd7d47a7d688d89c4ad2f73c755e44ddfa08ae836de`:
sort file paths, form `SHA256(file) + " " + relative_posix_path + "\n"` for each,
then SHA-256 the concatenated UTF-8 records. This covers that Python tree,
not binary dependencies or generated kernels. Future GPU reports should record
their actual imported source and runtime versions again.

Validation: eight mocked-Docker launcher tests pass; actual CPU containers
reproduce the default import failure and successfully import the installed
interface. CUDA remains uninitialized in both checks. Invalid selection also
fails on the real launcher before container startup. These checks establish
environment selection, not attention numerical/performance clearance.
