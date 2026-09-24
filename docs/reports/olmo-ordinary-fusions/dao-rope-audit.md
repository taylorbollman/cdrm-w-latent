# Dao RoPE compatibility audit

2026-09-24. Source inspection only; GPU numerical/performance evidence belongs
to the milestone's measured reports. The selected adapter uses the installed
Dao rotary helper without upgrading FlashAttention or changing RT arithmetic.

## Exact installed source

The CPU-only project container reports `flash_attn`
`2.7.4.post1+git5231d95fe1.54157505`, FA4 `4.0.0b20`, and Triton
`3.7.0+gitb7fa781f.nv26.6`. The installed rotary files exactly match upstream
commit `5231d95fe13733fb534c01895f7ea88c6a6c7793` when downloaded and hashed:

| Path under `/usr/local/lib/python3.12/dist-packages/flash_attn/` | SHA256 |
| --- | --- |
| `layers/rotary.py` | `32ab1467e1b16b439c60bb4bf0eb231c83a4ab24da7101e4306a6a0036b67ba4` |
| `ops/triton/rotary.py` | `587eed10b04b676df39eec676140333fb616ab0317266460510f01eb8563767e` |

The [installed-version kernel](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/flash_attn/ops/triton/rotary.py)
requires input and tables to have matching dtypes and supports head dimensions
up to256. It loads arithmetic operands into FP32. Split-half mode matches native
OLMo; inputs can have explicit strides. Its launch does not disable FP32 FMA.
The [autograd wrapper](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/flash_attn/layers/rotary.py)
uses conjugate rotation for backward and retains a legacy gradient clone for
out-of-place split-half execution. Its higher-level table module casts tables
to activation dtype, so that module is not used by our adapter.

## Newer upstream differs

Main at audit time was `eed1971f5132630dc296fe37601e834d4b57a248`.
The newer [kernel](https://github.com/Dao-AILab/flash-attention/blob/eed1971f5132630dc296fe37601e834d4b57a248/flash_attn/ops/triton/rotary.py)
removes the matching-dtype assertion, groups two heads per tile and uses the
PyTorch Triton wrapper. Its SHA256 is
`d2037e623854c702fb075784fe4292d4a89d6b9ac1ae33cab3d6788f8f94f240`.
The newer [autograd wrapper](https://github.com/Dao-AILab/flash-attention/blob/eed1971f5132630dc296fe37601e834d4b57a248/flash_attn/layers/rotary.py)
removes that old backward clone; SHA256 is
`a79cb42dfc33e126227bdba27ae7a024d7d5cc5c8fd7f7a5f26c68d501024e0a`.
These are possible later opportunities, not the current measured candidate.

## Selected minimal adapter

`ordinary_rope_backend="dao"` requires `reuse_rope=True`; `"native"` remains
default. Q/K receive one FP32 cast, out-of-place Dao rotation with existing
native FP32 phases, and a cast back to their projection dtype. FP32 residuals,
normalization, tables and pre-cast backward accumulation are preserved. FMA may
still change rounding, so a bounded comparison is required. The wrapper remains
independent of SDPA/FA4 and is not called by native tiled or author RT.

Preparation copies the first duplicate rotary half into contiguous tables
shaped `[B*T,D/2]`. Batch offsets `b*T` map each token to its exact native phase,
including different and nonconsecutive positional coordinates across rows.
Preparation checks that the two native table halves agree. It runs outside
capture; each ordinary layer and checkpoint replay reuse the same allocations.
The extra tables and int64 offsets cost `4*B*T*D + 8*B` bytes. Existing native
tables remain available for RT and the independent reference.

The adapter never modifies projection views or persistent keys. Prefix and
exported caches are initially rejected whenever ordinary Dao blocks execute.
This avoids changing the native cache convention, which stores unrotated keys.
Parameters, buffers and checkpoint keys remain unchanged. Static layout guards
pin the backend and compact-table ownership, storage and tensor versions.

Minimum GPU follow-up is a raw-cotangent local check with FP32/BF16 projections,
actual pretrained-model loss/all-parameter gradients, checkpoint reconstruction,
and exact same-candidate changed-input/weight graph and optimizer checks. CPU
mocked kernels verify the adapter contract, not Triton FMA or GPU performance.
