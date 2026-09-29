# Operator recovery assets verified

The new [recovery-bundle helper](recovery-bundle-protocol.md) restored the actual
guarded-stop update-1 checkpoint and its T16 packed index into a fresh persistent
directory, using immutable GCS generation records. It verified **six downloads
(23,851,490 bytes), 91 archive members, 88 model-run source snapshots against the
current checkout, and all 86 required corpus files**. The restored checkpoint
metadata, state bytes, index and committed two-rank cursor match their original
authorities.

This is an operator asset check. The helper imports no torch, loads no model
tensors, performs no GPU work and does not execute the emitted resume command.
Its final status is deliberately **`assets_verified_launch_pending`**. Exact
target runtime/topology and full model/Adam/RNG restore still belong to the
unchanged guarded runner. The earlier [actual two-GPU lifecycle result](loop-results.md)
supplies the separate restart evidence.

Artifacts are under `.runtime/olmo-campaign-lifecycle/recovered-stop-02/`:

- `recovery-manifest.json` records input receipt pins, all six downloaded object
  generations/hashes, original runtime/configuration, source fingerprint and
  unresolved launch prerequisites.
- `checkpoint/` contains the exact saved state and its committed manifest.
- `index/` contains the exact SQLite metadata database and index manifest.
- `stage/evidence/` contains the original report and its verified source overlay.
  The helper compares it with the selected checkout and never applies it there.
- `authorities/` preserves both independently pinned input receipts.
- `resume-command.txt` contains an unexecuted bounded two-rank command with the
  checkpoint/index pins and a new immutable checkpoint-retention prefix.

The actual command now distinguishes the **host** checkout
`/home/taylorbollman/cdrm-w-latent` from artifact paths inside the container at
`/workspace/cdrm-w-latent`. Its outer command changes to the host checkout and
uses the project's Docker launcher; the inner command checks the container and
working directory before GPU use. The prepared corpus remains at the existing
local-SSD path and must be mounted there. This helper verifies those existing
corpus bytes; it does not claim that downloading only a checkpoint and index
recreates the corpus, container image or original VM.
The corpus is already retained completely, with an earlier actual86-file cloud
restore. [Corpus recovery instructions](corpus-recovery.md) give its canonical
receipt/generations and the separate operator command to use after SSD loss.

The required saved runtime remains two H100 80GB GPUs, Torch
`2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3 and the recorded driver/runtime settings.
No H200 or other-runtime compatibility exception was introduced. Source changes
or missing/corrupt data prevent a completion marker rather than producing a
silently modified launch. Interrupted download bytes remain visibly partial.
The final manifest is published atomically after verification.

An initial restore in `recovered-stop-01` already authenticated the same assets,
but its generated outer `cd` used the container path. That operator-command
limitation is retained with the original result. A narrow explicit host-path
argument and regression check corrected the new helper, followed by the fresh
second restore above. No original checkpoint, model runner or prior evidence
was modified. The generated command has not been run in either stage.

The final focused CPU scope passed **14 tests in 0.15 seconds**, covering complete
fake-storage recovery, exact generation requests, interrupted/missing/corrupt
downloads, source/data/receipt failures, unsafe archive paths and links, and
host/container command mapping. These tests use real archive bytes and never
launch a model. The final log is `cpu-recovery-bundle-03.log` under the lifecycle
runtime directory.

Evidence pins:

| Artifact | SHA256 |
| --- | --- |
| Final recovery manifest | `09b5dc49f94bab3af06343d1c0fc5486845ff5e122eb89661f5fe4a4f140c247` |
| Conditional resume command | `24f7845b2e8ddf7b16204b6079a7ed1e6c1510f5691c6775707c52d3ccab833e` |
| Restored checkpoint manifest | `b6a170822f303a00ec707349496ee372e521ebff6b81c54231bcc867cc98667c` |
| Restored checkpoint state | `f9399e952089098ba87f066edc27e1edb14e4246705dae260bf738f469862b79` |
| Restored index manifest | `dc7dae59199e75b5d801565f948eebc94d97a67a65e2fc1e13d3d1bdafc452f1` |
| Restored SQLite index | `68f37f9c0ba304581c822ec0227ced4027b49df3a61ce88eb0cc8e7bc84df0ef` |

Cloud activity was read-only for both restores. No new optimizer update, W&B
training run or numerical-clearance result is implied by these verified assets.
