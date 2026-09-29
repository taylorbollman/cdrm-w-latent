# Prepared corpus recovery after local-SSD loss

The current corpus is already retained completely. No new upload or repeated
full restore was necessary for this audit. Canonical cloud prefix:

`gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/readiness-20260928/tokenized`

Its **86 committed files occupy21,016,338 bytes**, including config,28 complete
document shards and the final manifest. They contain7,054,230 tokens across
12,512 unique documents. This remains the seven-source readiness coverage
fixture, not the eventual production mixture. The packed SQLite index is a
separately retained derivative; it does not replace the corpus payloads.

The canonical receipt is
`.runtime/olmo-document-shards/retention/tokenized.json`, SHA256
`22a63e729d80fe953ec86470de8e69b34e807df1ccea840ffac267d9a9ec399c`.
It lists all86 object generations, sizes and SHA256 values, with successful
server metadata and generation-pinned streamed download verification. Key pins:

| Object relative to readiness namespace | Generation | SHA256 |
| --- | --- | --- |
| `tokenized/manifest.json` | `1790633355018643` | `f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76` |
| `tokenized/config.json` | `1790633332534723` | `21816eac3f9ede9620dd9eb85c979a1294416d8d37b66d9793912fb07cbb62fa` |
| `evidence/receipts/tokenized.json` | `1790633526333948` | `22a63e729d80fe953ec86470de8e69b34e807df1ccea840ffac267d9a9ec399c` |

The published [document-shard receipt](../olmo-document-shards/storage-receipt.md)
also retains raw extracts independently. The receipt above is sufficient to
recover the prepared bytes without retokenization or fetching upstream Dolma.

## Existing actual recovery evidence

The earlier two-GPU setup downloaded/verified the full86-file corpus and ran
`verify_document_shards`. Its report is
`.runtime/olmo-campaign-two-gpu/data-restore/report.json`, SHA256
`d1b3ce2a42de5519de01199a9733fb97b9a8f519641195c4dd1ca195dea97cfb`.
This is separate from the still earlier seven-file partial-shard preparation
restart rehearsal. The full-restore report, script and launcher log are retained
in the following immutable archive:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T235700Z/campaign-data-restore/evidence.tar.gz`

Generation`1790639942418202`, SHA256
`f3e2116c60359f9481d52c33e63dd2e5437a0232f05e96b962ff43332757d5cb`.
The relevant members are `evidence/restore.py` and `evidence/report.json`.
The local stage receipt is
`.runtime/olmo-campaign-two-gpu/retention/data-restore.json`, SHA256
`4061a806b52aa54340e5f19072ab6e06d6a47e2f848ee0bb5056f4179d4ce896`.

On2026-09-29, the independent recovery audit rehashed every currently required
SSD file against the canonical receipt: **86/86 sizes and SHA256 values match**,
total21,016,338 bytes. It also verified the original restore-script/report pins.
This latest audit made no cloud requests, changed no corpus bytes, and adds no
new claim about cloud availability beyond the retained verified restore evidence.

## Operator command with the persistent home checkout

After SSD loss, bootstrap/mount the project container and local SSD normally.
The following uses the existing tested recovery implementation. It makes a new
operator copy with only its report destination changed, preserving the original
frozen evidence. The receipt, original script and project sources must still
exist under the persistent `/home/taylorbollman/cdrm-w-latent` checkout.
Choose a fresh operator suffix if this directory already exists.

```bash
cd /home/taylorbollman/cdrm-w-latent
python3 - <<'PY'
from pathlib import Path
import hashlib
receipt = Path('.runtime/olmo-document-shards/retention/tokenized.json')
source = Path('.runtime/olmo-campaign-two-gpu/data-restore/restore.py')
assert hashlib.sha256(receipt.read_bytes()).hexdigest() == '22a63e729d80fe953ec86470de8e69b34e807df1ccea840ffac267d9a9ec399c'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '748b9d38472dff9357e5f73a79a43e24ed31dce26bacddf2ccaa20758f9ab65e'
target = Path('.runtime/olmo-campaign-lifecycle/operator-corpus-restore-01')
target.mkdir(parents=True, exist_ok=False)
text = source.read_text()
old = '.runtime/olmo-campaign-two-gpu/data-restore/report.json'
assert text.count(old) == 1
(target/'restore.py').write_text(text.replace(old, str(target/'report.json')))
PY
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'env -u GOOGLE_APPLICATION_CREDENTIALS python .runtime/olmo-campaign-lifecycle/operator-corpus-restore-01/restore.py'
```

This restores missing files to the original container-visible path
`/mnt/localssd/cdrm-data/olmo-dolma-v1_5-readiness-20260928/tokenized`, verifies
each exact generation/size/SHA before promotion, verifies existing files instead
of overwriting them, and validates all document-shard manifests/offsets/hashes.
Conflicting existing files fail explicitly. Cloud reads use the container's
mounted authenticated ADC; no credentials are printed. The command is provided
for future recovery and was not rerun during this latest audit.

For a **fresh checkout without persistent `.runtime` files**, first recover the
canonical receipt from its exact generation above and recover the original
`restore.py` from the pinned full-restore archive. Verify both SHA256 values,
then place them at the two paths used by the command. Extract only the named
regular archive member after verifying the archive hash; never apply an
unverified source overlay. Use a matching project checkout/container; the saved
verifier `cdrm/pretrained/document_shards.py` has SHA256
`da5dc7fa170d6cba6684b635405c43ebf1a40201599199f3a878cacd3d8542d6`.
The original archive and data receipts provide the historical authority; they
do not bootstrap Docker, credentials or an arbitrary new runtime automatically.

Only after the corpus is restored should the
[checkpoint/index recovery bundle](recovery-bundle-results.md) validate it and
emit a conditional training-resume command. That helper intentionally **requires
the86 corpus files to exist already**; it does not download them itself. Full
model/Adam/RNG restoration and matching GPU/runtime checks remain separate.
