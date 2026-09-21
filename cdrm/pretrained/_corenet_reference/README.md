# Pinned native CoreNet oracle

These eight upstream files are byte-identical copies from Apple CoreNet commit
`f9f83e616a34d02c422733a06a3fe5bde63ae575`. `manifest.json` records their original
paths, download URLs, byte counts, and SHA256 hashes. The original Apple license
and copyright headers are retained. This is validation-only source, not the
production model implementation.

`../reference.py` verifies the snapshot and executes the seven Python files in
dependency order. It removes only imports whose module starts with `corenet`.
It supplies no-op registry decorators, a minimal `nn.Module` base constructor,
and factories selecting the original RMSNorm and Swish classes. Every upstream
model/attention/MLP/normalization/RoPE/embedding/linear method is unchanged.
After construction, the wrapper explicitly moves the model to the requested
device: upstream `LinearLayer` uses `torch.Tensor(...)`, which does not honor
the surrounding `torch.device(...)` construction context. This placement step
does not change its forward arithmetic.

The default model derives 1.1B geometry through upstream `GPTConfig.from_name`.
Explicit configurations are supported for bounded small fixtures; those tests
exercise the same upstream layer implementations. The oracle does not import
our model implementation, apply recurrence, rewrite masking, or trim vocabulary.

Native cache support covers unpadded causal prefill and single-token decoding.
The wrapper evaluates a cached multi-token chunk one token at a time, preserving
that supported behavior. Extended padding/position APIs must be compared to
unpadded examples separately. FP32 parameters with optional BF16 autocast match
the native pretraining execution policy; casting the entire oracle to BF16 is
outside the supported validation contract.
