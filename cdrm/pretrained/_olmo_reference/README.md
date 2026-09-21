# Pristine original OLMo reference snapshot

Source: `allenai/OLMo`, revision
`b3741bc21f1dd504838b7dbd9878ee077ded63bd` (tag `v0.2.4`). All upstream Python
files and `LICENSE` are stored byte-for-byte. `checkpoint_config.json` is the
native `allenai/OLMo-1B` step60000 checkpoint configuration at immutable revision
`81b71efbce6f4dada57c94860301af4298bcd351`. `manifest.json` records exact URLs,
byte sizes and SHA256 checksums; the loader verifies them before execution.

`../olmo_reference.py` executes these files in an isolated module namespace,
without importing the project's modified `recurrent-transformer/olmo` package.
The auditable AST adaptations are:

1. Remove top-level package-relative imports. Their required symbols come from
   other files in this same snapshot.
2. Execute only the unchanged `StrEnum` class from `util.py`; unrelated logging,
   networking and CLI dependencies are not needed by the model.

No model class, forward method, attention, LayerNorm, rotary embedding, activation,
initialization or arithmetic expression is rewritten. Original beam search,
checkpoint-loading and other framework methods are outside the supported oracle
API; their relative imports are not redirected into the research fork.

The builder resolves architecture defaults through the original `ModelConfig`.
For tiny fixtures, only dimensions, vocabulary IDs and context size change.
Native parameters remain FP32; BF16 execution uses autocast. The native model
constructor changes global SDPA flags; the wrapper restores caller flags after
construction so validation controls backend selection explicitly.

The ordinary forward wrapper captures the final LayerNorm output with a hook.
For `inputs_embeds`, a temporary hook replaces the embedding lookup output; all
remaining native forward code executes unchanged. Unpadded, contiguous positions
and native cached chunks are the source-parity scope. Padding/explicit-position
adapter extensions have independent semantic tests.

`../olmo_recurrent_oracle.py` is separate research code, not pristine upstream.
It reconstructs RT history explicitly and uses manual FP32 attention over these
native block primitives. This deliberately differs from the production cache
scan so agreement checks more than duplicated scheduling code.

The snapshot is licensed under the accompanying Apache-2.0 `LICENSE`.
