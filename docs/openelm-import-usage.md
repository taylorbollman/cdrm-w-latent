# Native OpenELM-1.1B import and validation

This is the first pretrained-model PR. The adapter implements ordinary OpenELM;
RT, FBT and NextLat will be added in subsequent stages. See the
[handoff](fbt-rt-nextlat-handoff.md) for current status and the
[research plan](fbt-rt-nextlat-research-plan-v2.md) for those stages.

## Environment

Build the project image after pulling the dependency change (`ftfy==6.3.1`
preserves native tokenizer cleanup):

```bash
bash scripts/docker_build.sh
bash scripts/docker_shell.sh
```

Inside the container, verify the project location and GPU before GPU work:

```bash
test -f /.dockerenv
test "$PWD" = /workspace/cdrm-w-latent
nvidia-smi
```

All following Python commands run inside that container. There is no automatic
CPU fallback for the validation command. CPU unit tests are intentional and
can run with `CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh`.

## Prepare the checkpoint, sources and tokenizer

```bash
python scripts/openelm_prepare.py \
  --artifact-root .runtime/openelm-import/artifacts
```

The command prepares:

- Apple's individual native 300k model-only checkpoint, iteration index 299999.
- Pinned CoreNet and HF code/config snapshots and their hashes.
- The official Llama-2 tokenizer model using existing authorized HF access.
- Portable relative-path manifests with source URLs, revisions, sizes and SHA256.

Use `--only checkpoint`, `--only sources` or `--only tokenizer` to prepare a
component independently. Interrupted checkpoint downloads resume from verified
partial metadata. Complete existing files are rehashed; corrupted or unrecorded
files are rejected. An authorized local tokenizer file can be supplied with
`--tokenizer-model`; it must match the pinned official file. There is no mirror
or substitute tokenizer fallback.

The source model file is 4,320,718,260 bytes. Its SHA256 is
`0e79fef600d022da33111f86dae4e7f4a4fd1d2e9045364c5ccd4c2283c4d9c6`.
This digest was measured on our first download from Apple's published URL;
it is not represented as an Apple-published checksum.

Inspect all native tensor keys/shapes without GPU execution:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/openelm_prepare.py \
  --artifact-root .runtime/openelm-import/artifacts --only inspect
```

The expected model has 1,080,153,600 parameters and 226 state tensors. Preserve
all 32,128 vocabulary rows and native padding ID 32,000. Cropping to the HF
release's 32,000 rows changes the softmax denominator and fails this contract.
Tokenization applies native `ftfy` NFC cleanup and adds BOS/EOS by default.

## Load the ordinary adapter

```python
from pathlib import Path
import torch
from cdrm.pretrained.artifacts import (
    CHECKPOINT_FILENAME, OpenELMTokenizer, load_native_state_dict,
    validate_prepared_manifest,
)
from cdrm.pretrained.openelm import OpenELMConfig, OpenELMModel

root = Path(".runtime/openelm-import/artifacts")
manifest = validate_prepared_manifest(root)
config = OpenELMConfig.native_1_1b()
# Meta construction avoids allocating random full-size weights before loading.
model = OpenELMModel(config, device="meta")
state = load_native_state_dict(
    root / "checkpoint" / CHECKPOINT_FILENAME,
    expected_shapes={name: tuple(tensor.shape) for name, tensor in model.state_dict().items()},
)
model.load_state_dict(state, strict=True, assign=True)
model = model.to("cuda").eval()
del state
tokenizer = OpenELMTokenizer(root / "tokenizer/tokenizer.model")
ids = torch.tensor([tokenizer.encode("A running total starts at zero.")], device="cuda")
with torch.no_grad():
    output = model(ids, use_cache=True)
    next_id = output.logits[:, -1].argmax(-1, keepdim=True)
    continuation = model(next_id, past_key_values=output.past_key_values, use_cache=True)
```

Create the optimizer **after** `assign=True` loading and device materialization.
There is one owned parameter for lookup and readout:
`model.readout_weight is model.token_embeddings.weight`. Do not create a second
head parameter, globally freeze the shared matrix, or use a loader that resets
weights after import.

`output.last_hidden_state` is the final normalized state. `return_logits=False`
skips the vocabulary projection. `inputs_embeds` allows future latent inputs;
exactly one of `input_ids` or `inputs_embeds` must be supplied.

The cache retains native KV-head counts and normalized **unrotated** keys.
It remains attached to autograd and is not mutated in place. Cached multi-token
chunks use explicit offset causal masks. An optional binary `attention_mask`
covers the complete cached-plus-current sequence, and cached validity cannot
be changed. Padding query outputs are excluded by the caller's loss mask.
Passing a padding ID alone preserves the native unmasked behavior; provide a
mask when actual padded examples should exclude those keys.

## Bounded checks

Run the scoped unit suite:

```bash
python -m pytest -q tests/test_openelm_model.py \
  tests/test_openelm_reference.py tests/test_openelm_artifacts.py
```

Run the actual-checkpoint GPU diagnostic into a **new** output directory:

```bash
python scripts/openelm_validate.py \
  --artifacts .runtime/openelm-import/artifacts \
  --output-dir .runtime/openelm-import/my-new-validation
```

The diagnostic requires online W&B logging under `taylorbollman`, project
`pretrained-fbt-rt-nextlat`. Credentials come from the container environment.
It writes the real run URL, source hashes, token fixtures, parameter inventory,
all per-tensor gradient comparisons and results to `report.json`.

The independent oracle executes the pinned CoreNet mathematical methods from
the [licensed source snapshot](../cdrm/pretrained/_corenet_reference/README.md).
Small framework constructors and registries are supplied locally; this is not
a claim that the complete CoreNet training framework has been installed.
The actual 1.1B reference derives geometry through upstream's original config
constructor, rather than copying the adapter's resolved lists.

The check compares native versus adapter FP32 and BF16-autocast outputs,
hidden states, shifted CE and all parameter gradients on bounded fixtures.
It also checks cache/chunk behavior and future-token isolation. A separate
ordinary BF16 forward records actual SDPA kernel dispatch. Finite fused-versus-
math differences are reported observationally; they are not an RT precision
budget. Peak VRAM includes both reference and adapter plus paired gradients,
so it is not a single-model training-memory estimate.

## Retention and resumption

Artifacts and reports under `.runtime/openelm-import/` reside on the persistent
project disk. Retain the selected lineage in GCS before considering this PR
complete; the retention utility stores the checkpoint separately from the small
evidence archive, avoiding a second compressed copy of the weights:

```bash
python scripts/openelm_retain.py \
  --artifacts .runtime/openelm-import/artifacts \
  --validation .runtime/openelm-import/my-new-validation \
  --output-dir .runtime/openelm-import/my-retention \
  --prefix gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T182701Z/
```

Choose a fresh UTC timestamp for a new lineage. The receipt records remote
generations, sizes, MD5 and SHA256. Existing objects
are reused only if they match; the utility does not overwrite a different run.
Restore the checkpoint and evidence as described in the archive's instructions,
then rerun offline manifest verification. W&B is a convenient view; local/GCS
records remain the source of reproducible validation evidence.

On the validation VM, an inherited `GOOGLE_APPLICATION_CREDENTIALS` value
pointed to a nonexistent old path. We verified that the mounted standard ADC
credentials already had access to `fast-chunks`, then ran retention with
`env -u GOOGLE_APPLICATION_CREDENTIALS python scripts/openelm_retain.py ...`.
This is an explicit environment correction for that VM; the utility does not
silently ignore an explicitly configured credential file.

This milestone establishes faithful ordinary import. It does not establish
quality gains, long-context performance, recurrent gradients, multi-GPU behavior,
or an optimal physical batch size for later experiments.
