# Operator recovery bundle: bounded CPU/cloud acceptance

This stage fills the gap between retained artifacts and a concrete operator
resume command. Earlier GPU suites establish exact model replay once authorities
are assembled. This helper assembles and verifies those authorities in a new
directory; it does not repeat GPU execution or relax any restore guard.

Scope is the completed guarded tiny two-rank checkpoint lineage plus its exact
T16 packed index. Inputs are an independently SHA-pinned stage-retention receipt,
the separately pinned explicit index-retention receipt, completed update 1 or 2,
an existing prepared corpus root, a matching source checkout, its explicit host
checkout path for the Docker launcher, a fresh persistent
output directory under that checkout, and a new immutable storage prefix for a
possible later continuation. It supports neither compact fusion checkpoints nor
arbitrary pretrained checkpoint schemas.

Download exactly six recorded GCS objects by explicit generation: stage evidence
archive and retention manifest, checkpoint state and manifest, index SQLite and
index manifest. Verify every object's SHA256 and byte count before promoting its
partial download. Authenticate the archive's complete external/internal member
inventory, reject unsafe paths/links/duplicates, and preserve its source overlay
in the new directory. Compare every saved source SHA with the requested live
checkout; never apply the overlay to that checkout. Existing prepared corpus
files must all match the index's SHA/size inventory. Missing/corrupt sources or
data are blockers, not requests to fabricate, fetch unpinned data or weaken the
configuration.

Crosscheck checkpoint metadata with its stage report, including the complete
configuration/source fingerprint, selected update, rank cursor topology and
index identity. Model/Adam tensor bytes are authenticated but not deserialized.
The original runtime, determinism and full configuration are preserved in the
recovery manifest. The helper does not import torch or initialize CUDA. It does
not contact W&B because it performs no model experiment.

Only after all asset checks pass, emit `recovery-manifest.json` and a conditional
`resume-command.txt`. The command uses the matching guarded runner, exact index
and checkpoint manifest hashes, two ranks, the restored persistent directories,
the existing corpus mount, the explicit new checkpoint retention prefix and
bounded external timeout. It is text, not executed by the recovery helper.
Target GPU/runtime compatibility remains pending: the unchanged runner must
match the original configuration and topology exactly. An H200 or another
runtime is not implicitly cleared. Confirm the corpus is mounted at the declared
container path; an asset bundle alone is not a complete VM or Docker image.
When preparing inside a CPU container, `--checkout-root` is its mounted
`/workspace/cdrm-w-latent`, while `--host-checkout-root` is the real host location,
for example `/home/taylorbollman/cdrm-w-latent`. The emitted outer `cd` uses the
host path and inner artifact paths use the container mount. The CPU helper
records this declared mapping but cannot independently inspect the host mount.

CPU tests use generation-addressed fake storage and actual archive bytes, testing
complete recovery, interrupted/corrupt/missing-generation downloads, source/data
and receipt pin failures, unsafe archive paths/links, and no launch/publication
on failure. A single real CPU/cloud acceptance restores guarded stop update 1
into a fresh local directory and rehashes the actual corpus and source overlay.
Reads only: no GCS uploads, source modifications, optimizer steps or GPU launch.

Partial directories retain explicit failure evidence and are never labeled
complete. Use a new directory for retry; the bounded tiny state is about 20 MB,
so resumable bulk transfer machinery is unnecessary here. Final status is
`assets_verified_launch_pending`, not training-ready or numerically cleared.
